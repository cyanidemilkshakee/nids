"""Comprehensive RandomForest training with optional synthetic augmentation.
Adds metrics persistence for API dashboard and feature name saving.
"""

import os
import sys
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import numpy as np
# Make project root importable when running as a script
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from utils.logging_utils import setup_logging, log_stage, log_kv
MAX_ROWS = int(os.environ.get('MAX_ROWS', '250000'))

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
os.makedirs(MODEL_DIR, exist_ok=True)

PRIMARY_FILE = os.environ.get('PRIMARY_DATA', 'unsw_selected_features.csv')  # or cicids_selected_features.csv
SYNTH_FILE = os.environ.get('SYNTH_DATA', 'synthetic_gan_samples.csv')  # optional
OUTPUT_MODEL = os.environ.get('OUTPUT_MODEL', 'saved_model.joblib')

log = setup_logging("train.random_forest")
log.info("Starting RandomForest training")
log_kv(log, PRIMARY_DATA=PRIMARY_FILE, SYNTH_DATA=SYNTH_FILE, OUTPUT_MODEL=OUTPUT_MODEL, MAX_ROWS=MAX_ROWS)

primary_path = os.path.join(DATA_DIR, PRIMARY_FILE)
with log_stage(log, "load_primary_dataset"):
	log_kv(log, path=primary_path)
	df = pd.read_csv(primary_path)
	log_kv(log, rows=len(df), cols=len(df.columns))

# Optional synthetic augmentation (expects same feature columns + label)
synth_path = os.path.join(DATA_DIR, SYNTH_FILE)
if os.path.exists(synth_path):
	try:
		synth_df = pd.read_csv(synth_path)
		# Align columns (add missing, drop extras)
		base_cols = df.columns
		for col in base_cols:
			if col not in synth_df.columns:
				synth_df[col] = 0
		synth_df = synth_df[base_cols]
		df = pd.concat([df, synth_df], ignore_index=True)
		log.info(f"Augmented with synthetic samples: {len(synth_df)} rows -> total {len(df)}")
	except Exception as e:
		print(f"[WARN] Failed to use synthetic data: {e}")
else:
	log.info("No synthetic augmentation file found; proceeding with primary dataset only.")

# ---------------------------------------------------------------------------
# Optional in-script lightweight GAN augmentation (tabular) if TRAIN_GAN=1
# This trains a very small GAN to generate additional minority class samples.
# Controlled to remain fast (< ~10 min typical) with low epochs.
# ---------------------------------------------------------------------------
if os.environ.get('TRAIN_GAN', '0') == '1':
	try:
		import tensorflow as tf
		feature_cols = [c for c in df.columns if c != 'label']
		num_features = len(feature_cols)
		class_counts = df['label'].value_counts()
		max_count = class_counts.max()
		# Select minority classes (those below 60% of max)
		minority_classes = class_counts[class_counts < 0.6 * max_count].index.tolist()
		if minority_classes:
			log.info(f"[GAN] Minority classes targeted: {minority_classes}")
			noise_dim = 32
			def build_generator():
				g = tf.keras.Sequential([
					tf.keras.layers.Input(shape=(noise_dim + 1,)),  # +1 for class embedding scalar
					tf.keras.layers.Dense(64, activation='relu'),
					tf.keras.layers.Dense(128, activation='relu'),
					tf.keras.layers.Dense(num_features, activation='tanh')
				])
				return g
			def build_discriminator():
				d = tf.keras.Sequential([
					tf.keras.layers.Input(shape=(num_features + 1,)),  # +1 class label
					tf.keras.layers.Dense(128, activation='relu'),
					tf.keras.layers.Dropout(0.3),
					tf.keras.layers.Dense(64, activation='relu'),
					tf.keras.layers.Dense(1, activation='sigmoid')
				])
				d.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy')
				return d
			generator = build_generator()
			discriminator = build_discriminator()
			discriminator.trainable = False
			gan_input = tf.keras.layers.Input(shape=(noise_dim + 1,))
			fake_features = generator(gan_input)
			# append class condition (last element of input) back to features for discriminator
			class_cond = tf.keras.layers.Lambda(lambda z: z[:, -1:])(gan_input)
			disc_in = tf.keras.layers.Concatenate(axis=1)([fake_features, class_cond])
			validity = discriminator(disc_in)
			gan = tf.keras.Model(gan_input, validity)
			gan.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy')
			EPOCHS = int(os.environ.get('GAN_EPOCHS', '15'))
			BATCH = 128
			for cls in minority_classes:
				subset = df[df['label'] == cls][feature_cols].values
				if subset.shape[0] < 10:
					continue
				target = max_count - subset.shape[0]
				target = min(target, int(0.5 * max_count))  # cap to avoid explosion
				if target <= 0:
					continue
				log.info(f"[GAN] Training class {cls} to generate ~{target} samples")
				# Simple scaling to [-1,1] assuming original roughly [0,1]
				real_scaled = (subset * 2.0) - 1.0
				real_labels = np.ones((real_scaled.shape[0], 1))
				for epoch in range(EPOCHS):
					# Train discriminator
					idx = np.random.randint(0, real_scaled.shape[0], size=min(BATCH, real_scaled.shape[0]))
					real_batch = real_scaled[idx]
					class_batch = np.full((real_batch.shape[0], 1), cls, dtype=np.float32)
					real_disc_in = np.concatenate([real_batch, class_batch], axis=1)
					noise = np.random.randn(real_batch.shape[0], noise_dim)
					noise_input = np.concatenate([noise, class_batch], axis=1)
					gen_batch = generator.predict(noise_input, verbose=0)
					fake_disc_in = np.concatenate([gen_batch, class_batch], axis=1)
					d_loss_real = discriminator.train_on_batch(real_disc_in, np.ones((real_batch.shape[0], 1)))
					d_loss_fake = discriminator.train_on_batch(fake_disc_in, np.zeros((real_batch.shape[0], 1)))
					# Train generator
					noise = np.random.randn(BATCH, noise_dim)
					class_cond_batch = np.full((BATCH, 1), cls, dtype=np.float32)
					g_loss = gan.train_on_batch(np.concatenate([noise, class_cond_batch], axis=1), np.ones((BATCH, 1)))
					if (epoch+1) % 5 == 0:
						log.info(f"[GAN][Class {cls}] Epoch {epoch+1}/{EPOCHS} d_loss_real={d_loss_real:.3f} d_loss_fake={d_loss_fake:.3f} g_loss={g_loss:.3f}")
				# Generate target samples
				gen_needed = target
				noise = np.random.randn(gen_needed, noise_dim)
				class_cond_batch = np.full((gen_needed, 1), cls, dtype=np.float32)
				gen_samples = generator.predict(np.concatenate([noise, class_cond_batch], axis=1), verbose=0)
				# Inverse scale back to [0,1]
				gen_samples = (gen_samples + 1.0) / 2.0
				gen_df = pd.DataFrame(gen_samples, columns=feature_cols)
				gen_df['label'] = cls
				df = pd.concat([df, gen_df], ignore_index=True)
			log.info(f"[GAN] Augmented dataset size after GAN: {len(df)}")
		else:
			print("[GAN] No minority classes identified; skipping GAN phase.")
	except Exception as e:
		print(f"[GAN][WARN] Skipping GAN augmentation due to error: {e}")

# Final cap to MAX_ROWS (post augmentation)
if len(df) > MAX_ROWS:
	before = len(df)
	with log_stage(log, "cap_max_rows"):
		df = df.sample(MAX_ROWS, random_state=42)
		log_kv(log, before=before, after=len(df), MAX_ROWS=MAX_ROWS)

X = df.drop('label', axis=1)
y = df['label']

classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
cw_dict = {cls: w for cls, w in zip(classes, class_weights)}
log.info(f"Class weights: {cw_dict}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(
	n_estimators=400,  # more trees for stability (still reasonable time with n_jobs=-1)
	max_depth=None,
	n_jobs=-1,
	random_state=42,
	class_weight=cw_dict,
	oob_score=False
)
with log_stage(log, "fit_model"):
	clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

log.info("Detailed Classification Report:\n" + classification_report(y_test, y_pred, zero_division=0))
log.info(f"Confusion Matrix:\n{cm}")

# Save model
model_path = os.path.join(MODEL_DIR, OUTPUT_MODEL)
joblib.dump(clf, model_path)
log.info(f"Model saved -> {model_path}")

# Persist feature names for evaluation script alignment
feature_names_path = os.path.join(MODEL_DIR, 'feature_names.json')
with open(feature_names_path, 'w', encoding='utf-8') as f:
	json.dump(list(X.columns), f, indent=2)
log.info(f"Feature names saved -> {feature_names_path}")

# Persist metrics for dashboard consumption
metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
if os.path.exists(metrics_path):
	try:
		with open(metrics_path, 'r', encoding='utf-8') as f:
			metrics_data = json.load(f)
	except Exception:
		metrics_data = {}
else:
	metrics_data = {}

key = os.path.splitext(OUTPUT_MODEL)[0]
metrics_data[key] = {
	'model': OUTPUT_MODEL,
	'metrics': {
		'accuracy': round(accuracy_score(y_test, y_pred), 4),
		'precision_weighted': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
		'recall_weighted': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
		'f1_weighted': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)
	},
	'confusion_matrix': cm.tolist(),
	'class_weights': {str(int(k)): float(v) for k, v in cw_dict.items()},
	'samples': len(df)
}
with open(metrics_path, 'w', encoding='utf-8') as f:
	json.dump(metrics_data, f, indent=2)
log.info(f"Metrics updated -> {metrics_path}")
