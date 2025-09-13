"""GAN + CNN Training Pipeline
Generates minority class synthetic samples using a lightweight conditional GAN then trains the 1D CNN.
Environment Variables (optional):
  PRIMARY_DATA=unsw_selected_features.csv
  MAX_ROWS=250000
  GAN_EPOCHS=10
  GAN_NOISE_DIM=32
  GAN_BATCH=128
  GAN_MINORITY_RATIO=0.6   # classes below 60% of max count targeted
  OUTPUT_MODEL=saved_model_gan_cnn.keras
Writes metrics to models/metrics.json with key derived from OUTPUT_MODEL.
"""
import os, json, numpy as np, pandas as pd, math
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

PRIMARY_FILE = os.environ.get('PRIMARY_DATA', 'unsw_selected_features.csv')
MAX_ROWS = int(os.environ.get('MAX_ROWS', '250000'))
GAN_EPOCHS = int(os.environ.get('GAN_EPOCHS', '10'))
NOISE_DIM = int(os.environ.get('GAN_NOISE_DIM', '32'))
BATCH = int(os.environ.get('GAN_BATCH', '128'))
MINORITY_RATIO = float(os.environ.get('GAN_MINORITY_RATIO', '0.6'))
OUTPUT_MODEL = os.environ.get('OUTPUT_MODEL', 'saved_model_gan_cnn.keras')
OUTPUT_MODEL = OUTPUT_MODEL.strip()
if not (OUTPUT_MODEL.endswith('.keras') or OUTPUT_MODEL.endswith('.h5')):
    # default to keras format
    OUTPUT_MODEL += '.keras'

print(f"[PIPE] Loading dataset {PRIMARY_FILE}")
path = os.path.join(DATA_DIR, PRIMARY_FILE)
df = pd.read_csv(path)

if len(df) > MAX_ROWS:
    before = len(df)
    # Stratified proportional downsample while keeping at least 3 per class
    frac = MAX_ROWS / before
    parts = []
    for cls_val, group in df.groupby('label'):
        take = int(round(len(group) * frac))
        take = max(min(len(group), take), min(3, len(group)))  # ensure >=3 if possible
        parts.append(group.sample(take, random_state=42))
    df = pd.concat(parts, ignore_index=True)
    # If rounding drift causes slight mismatch adjust by random add/drop
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42)
    elif len(df) < MAX_ROWS:
        short = MAX_ROWS - len(df)
        df = pd.concat([df, df.sample(short, replace=True, random_state=42)], ignore_index=True)
    print(f"[PIPE] Initial stratified cap {before}->{len(df)} (target {MAX_ROWS})")

feature_cols = [c for c in df.columns if c != 'label']
num_features = len(feature_cols)
class_counts = df['label'].value_counts()
max_count = class_counts.max()
minority_classes = class_counts[class_counts < MINORITY_RATIO * max_count].index.tolist()
print(f"[PIPE] Minority classes: {minority_classes}")

# --- Build conditional GAN (lightweight) ---
from tensorflow.keras import layers, Model, Sequential

def build_generator():
    return Sequential([
        layers.Input(shape=(NOISE_DIM + 1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_features, activation='tanh')
    ])

def build_discriminator():
    model = Sequential([
        layers.Input(shape=(num_features + 1,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy')
    return model

gen = build_generator()
# Build discriminator first as trainable, then create composite GAN with it frozen
disc = build_discriminator()

# Helper to ensure float32 contiguous arrays
def to_f32(a):
    return np.asarray(a, dtype=np.float32)

z_in = layers.Input(shape=(NOISE_DIM + 1,))
fake_feats = gen(z_in)
cls_part = layers.Lambda(lambda z: z[:, -1:])(z_in)
comb = layers.Concatenate(axis=1)([fake_feats, cls_part])
valid = disc(comb)
gan = Model(z_in, valid)
# Freeze discriminator inside GAN only (keep original disc for standalone training)
disc.trainable = False
gan.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy')

# --- Training GAN per minority class ---
for cls in minority_classes:
    subset = df[df['label'] == cls][feature_cols].values
    if subset.shape[0] < 10:
        continue
    target = max_count - subset.shape[0]
    target = min(target, int(0.5 * max_count))
    if target <= 0:
        continue
    print(f"[GAN] Class {cls} aiming to add ~{target} samples")
    real_scaled = (subset * 2.0) - 1.0
    for epoch in range(GAN_EPOCHS):
        idx = np.random.randint(0, real_scaled.shape[0], size=min(BATCH, real_scaled.shape[0]))
        real_batch = to_f32(real_scaled[idx])
        class_batch = to_f32(np.full((real_batch.shape[0], 1), cls))
        real_disc_in = to_f32(np.concatenate([real_batch, class_batch], axis=1))
        noise = to_f32(np.random.randn(real_batch.shape[0], NOISE_DIM))
        noise_input = to_f32(np.concatenate([noise, class_batch], axis=1))
        gen_batch = to_f32(gen.predict(noise_input, verbose=0))
        fake_disc_in = to_f32(np.concatenate([gen_batch, class_batch], axis=1))
        # Train discriminator (ensure it's trainable for this step)
        disc.trainable = True
        d_loss_real = disc.train_on_batch(real_disc_in, to_f32(np.ones((real_batch.shape[0], 1))))
        d_loss_fake = disc.train_on_batch(fake_disc_in, to_f32(np.zeros((real_batch.shape[0], 1))))
        # Train generator via frozen discriminator in composite model
        disc.trainable = False
        noise = to_f32(np.random.randn(BATCH, NOISE_DIM))
        class_cond_batch = to_f32(np.full((BATCH, 1), cls))
        g_loss = gan.train_on_batch(to_f32(np.concatenate([noise, class_cond_batch], axis=1)), to_f32(np.ones((BATCH, 1))))
        if (epoch + 1) % max(1, GAN_EPOCHS // 5) == 0:
            print(f"[GAN][{cls}] {epoch+1}/{GAN_EPOCHS} d_real={d_loss_real:.3f} d_fake={d_loss_fake:.3f} g={g_loss:.3f}")
    gen_needed = target
    noise = to_f32(np.random.randn(gen_needed, NOISE_DIM))
    class_cond_batch = to_f32(np.full((gen_needed, 1), cls))
    gen_samples = to_f32(gen.predict(to_f32(np.concatenate([noise, class_cond_batch], axis=1)), verbose=0))
    gen_samples = to_f32((gen_samples + 1.0) / 2.0)
    gen_df = pd.DataFrame(gen_samples, columns=feature_cols)
    gen_df['label'] = cls
    df = pd.concat([df, gen_df], ignore_index=True)
    print(f"[GAN] Added {len(gen_df)} samples for class {cls}; total rows now {len(df)}")

if len(df) > MAX_ROWS:
    before = len(df)
    # Repeat stratified logic post GAN augmentation
    frac = MAX_ROWS / before
    parts = []
    for cls_val, group in df.groupby('label'):
        take = int(round(len(group) * frac))
        take = max(min(len(group), take), min(3, len(group)))
        parts.append(group.sample(take, random_state=42))
    df = pd.concat(parts, ignore_index=True)
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42)
    elif len(df) < MAX_ROWS:
        short = MAX_ROWS - len(df)
        df = pd.concat([df, df.sample(short, replace=True, random_state=42)], ignore_index=True)
    print(f"[PIPE] Post-GAN stratified cap {before}->{len(df)} (target {MAX_ROWS})")

# --- CNN Training ---
X = df.drop('label', axis=1).values.astype('float32')
y = df['label'].values.astype('int64')
# Ensure minimum 2 samples per class for stratify; duplicate if needed
value_counts = pd.Series(y).value_counts()
rare_classes = value_counts[value_counts < 2].index.tolist()
if rare_classes:
    add_rows = []
    for rc in rare_classes:
        rows = df[df['label'] == rc]
        if not rows.empty:
            add_rows.append(rows.sample(2 - len(rows), replace=True, random_state=42))
    if add_rows:
        df = pd.concat([df] + add_rows, ignore_index=True)
        X = df.drop('label', axis=1).values.astype('float32')
        y = df['label'].values.astype('int64')
classes = np.unique(y)
class_weights_vals = compute_class_weight('balanced', classes=classes, y=y)
class_weight = {int(c): float(w) for c, w in zip(classes, class_weights_vals)}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_rs = X_train.reshape(-1, X_train.shape[1], 1)
X_test_rs = X_test.reshape(-1, X_test.shape[1], 1)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_rs.shape[1], 1)),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

history = model.fit(
    X_train_rs, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.15,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2
)

probs = model.predict(X_test_rs, verbose=0)
y_pred = np.argmax(probs, axis=1)
metrics = {
    'accuracy': round(accuracy_score(y_test, y_pred), 4),
    'precision_weighted': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
    'recall_weighted': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
    'f1_weighted': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)
}
print(f"[PIPE] CNN metrics: {metrics}")

model_path = os.path.join(MODEL_DIR, OUTPUT_MODEL)
print(f"[PIPE] Saving model to {model_path}")
model.save(model_path)
print(f"[PIPE] Saved model -> {model_path}")

# Update metrics
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
    'metrics': metrics,
    'samples': int(len(df)),
    'type': 'cnn_gan'
}
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, indent=2)
print(f"[PIPE] Metrics updated -> {metrics_path}")

