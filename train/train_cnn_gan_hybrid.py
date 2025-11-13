"""CNN-GAN Hybrid training script for AI NIDS.
Designed for ~1 hour training with real-time progress monitoring.
Combines CNN for feature extraction with GAN for data augmentation.

Environment Variables:
  PRIMARY_DATA=unsw_selected_features.csv
  OUTPUT_MODEL=saved_model_cnn_gan_hybrid.h5
  MAX_ROWS=180000
  GAN_EPOCHS=50
  CNN_EPOCHS=50
  BATCH_SIZE=256
  LATENT_DIM=100
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[CNN-GAN] GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"[CNN-GAN] GPU setup error: {e}")
else:
    print("[CNN-GAN] No GPU found, using CPU")

# Project paths
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logging_utils import setup_logging, log_stage, log_kv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Configuration
PRIMARY_FILE = os.environ.get('PRIMARY_DATA', 'unsw_selected_features.csv')
OUTPUT_MODEL = os.environ.get('OUTPUT_MODEL', 'saved_model_cnn_gan_hybrid.h5')
MAX_ROWS = int(os.environ.get('MAX_ROWS', '180000'))
TEST_SIZE = float(os.environ.get('TEST_SIZE', '0.2'))
GAN_EPOCHS = int(os.environ.get('GAN_EPOCHS', '50'))
CNN_EPOCHS = int(os.environ.get('CNN_EPOCHS', '50'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '256'))
LATENT_DIM = int(os.environ.get('LATENT_DIM', '100'))
RANDOM_STATE = 42

# Set random seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

log = setup_logging("train.cnn_gan_hybrid")
log.info("=" * 80)
log.info("Starting CNN-GAN Hybrid Training for AI-NIDS")
log.info("=" * 80)
log_kv(log, PRIMARY_DATA=PRIMARY_FILE, OUTPUT_MODEL=OUTPUT_MODEL)
log_kv(log, MAX_ROWS=MAX_ROWS, GAN_EPOCHS=GAN_EPOCHS, CNN_EPOCHS=CNN_EPOCHS, BATCH_SIZE=BATCH_SIZE)

start_time = time.time()

# Load data
primary_path = os.path.join(DATA_DIR, PRIMARY_FILE)
if not os.path.exists(primary_path):
    raise FileNotFoundError(f"Primary dataset not found: {primary_path}")

with log_stage(log, "load_dataset"):
    log_kv(log, path=primary_path)
    df = pd.read_csv(primary_path)
    log_kv(log, rows=len(df), cols=len(df.columns))

# Check class distribution
class_dist = df['label'].value_counts().sort_index()
log.info("Original class distribution:")
log.info(f"\n{class_dist}")

# Limit rows if needed
if len(df) > MAX_ROWS:
    before = len(df)
    with log_stage(log, "sample_data"):
        df = df.sample(MAX_ROWS, random_state=RANDOM_STATE)
        log_kv(log, before=before, after=len(df), MAX_ROWS=MAX_ROWS)

log_kv(log, final_shape=df.shape)
X = df.drop('label', axis=1).values
y = df['label'].values
num_features = X.shape[1]
num_classes = len(np.unique(y))

log.info(f"Feature dimension: {num_features}")
log.info(f"Number of classes: {num_classes}")

# Normalize data
log.info("Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
log_kv(log, train_shape=X_train.shape, test_shape=X_test.shape)

# ============================================================================
# PHASE 1: GAN Training for Data Augmentation
# ============================================================================
log.info("\n" + "="*80)
log.info("PHASE 1: GAN TRAINING FOR DATA AUGMENTATION")
log.info("="*80 + "\n")

def build_generator(latent_dim, output_dim):
    model = models.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        
        layers.Dense(1024),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        
        layers.Dense(output_dim, activation='tanh')
    ], name='generator')
    return model

def build_discriminator(input_dim):
    model = models.Sequential([
        layers.Dense(1024, input_dim=input_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Dense(1, activation='sigmoid')
    ], name='discriminator')
    return model

log.info("Building GAN components...")
generator = build_generator(LATENT_DIM, num_features)
discriminator = build_discriminator(num_features)

discriminator.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

discriminator.trainable = False
gan_input = layers.Input(shape=(LATENT_DIM,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = models.Model(gan_input, gan_output, name='gan')

gan.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)

log.info("Starting GAN training...")
gan_start = time.time()

for epoch in range(GAN_EPOCHS):
    epoch_start = time.time()
    
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    
    num_batches = len(X_train) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        real_samples = X_train_shuffled[start_idx:end_idx]
        
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, LATENT_DIM))
        fake_samples = generator.predict(noise, verbose=0)
        
        discriminator.trainable = True
        real_labels = np.ones((BATCH_SIZE, 1)) * 0.9
        fake_labels = np.zeros((BATCH_SIZE, 1))
        
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        discriminator.trainable = False
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, LATENT_DIM))
        g_loss = gan.train_on_batch(noise, real_labels)
    
    epoch_time = time.time() - epoch_start
    progress = ((epoch + 1) / GAN_EPOCHS) * 100
    
    print(f"\n{'='*80}")
    print(f"✓ GAN EPOCH {epoch + 1}/{GAN_EPOCHS} completed in {epoch_time:.2f}s")
    print(f"  Progress: {progress:.1f}% [{('█' * int(progress/2)).ljust(50)}]")
    print(f"  D_loss: {d_loss[0]:.4f} | D_acc: {d_loss[1]:.4f} | G_loss: {g_loss:.4f}")
    print(f"{'='*80}")

gan_time = time.time() - gan_start
log.info(f"GAN training completed in {gan_time/60:.2f} minutes")

# Generate synthetic samples
log.info("Generating synthetic samples with trained GAN...")
num_synthetic = int(len(X_train) * 0.3)  # 30% augmentation
noise = np.random.normal(0, 1, size=(num_synthetic, LATENT_DIM))
synthetic_samples = generator.predict(noise, verbose=1)

# Assign labels based on minority class distribution
minority_classes = [cls for cls in np.unique(y_train) if np.sum(y_train == cls) < len(y_train) / num_classes]
if minority_classes:
    synthetic_labels = np.random.choice(minority_classes, size=num_synthetic)
else:
    synthetic_labels = np.random.choice(y_train, size=num_synthetic)

log.info(f"Generated {num_synthetic} synthetic samples")

# Augment training data
X_train_augmented = np.vstack([X_train, synthetic_samples])
y_train_augmented = np.hstack([y_train, synthetic_labels])
log.info(f"Augmented training set: {X_train_augmented.shape}")

# ============================================================================
# PHASE 2: CNN Training with Augmented Data
# ============================================================================
log.info("\n" + "="*80)
log.info("PHASE 2: CNN TRAINING WITH AUGMENTED DATA")
log.info("="*80 + "\n")

# One-hot encode labels
y_train_cat = to_categorical(y_train_augmented, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Reshape for CNN
X_train_cnn = X_train_augmented.reshape(X_train_augmented.shape[0], X_train_augmented.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

log.info(f"CNN input shapes: train={X_train_cnn.shape}, test={X_test_cnn.shape}")

# Compute class weights
classes = np.unique(y_train_augmented)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_augmented)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
log.info(f"Class weights: {dict(zip(classes, np.round(class_weights, 3)))}")

# Build CNN
log.info("Building CNN architecture...")
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

cnn_model = build_cnn(input_shape=(X_train_cnn.shape[1], 1), num_classes=num_classes)

cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

log.info("CNN architecture:")
cnn_model.summary(print_fn=lambda x: log.info(x))

# Callbacks
class HybridProgressCallback(callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"CNN EPOCH {epoch + 1}/{self.total_epochs}")
        print(f"{'='*80}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        progress = ((epoch + 1) / self.total_epochs) * 100
        
        print(f"\n{'='*80}")
        print(f"✓ CNN Epoch {epoch + 1}/{self.total_epochs} completed in {epoch_time:.2f}s")
        print(f"  Progress: {progress:.1f}% [{('█' * int(progress/2)).ljust(50)}]")
        print(f"  Loss: {logs.get('loss', 0):.4f} | Accuracy: {logs.get('accuracy', 0):.4f}")
        print(f"  Val Loss: {logs.get('val_loss', 0):.4f} | Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
        print(f"{'='*80}\n")

progress_callback = HybridProgressCallback(total_epochs=CNN_EPOCHS)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Train CNN
log.info("Starting CNN training...")
cnn_start = time.time()

history = cnn_model.fit(
    X_train_cnn,
    y_train_cat,
    validation_split=0.2,
    epochs=CNN_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[progress_callback, early_stop, reduce_lr],
    verbose=0
)

cnn_time = time.time() - cnn_start
log.info(f"CNN training completed in {cnn_time/60:.2f} minutes")

# Evaluate
log.info("Evaluating hybrid model on test set...")
test_loss, test_accuracy, test_precision, test_recall = cnn_model.evaluate(
    X_test_cnn,
    y_test_cat,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred_probs = cnn_model.predict(X_test_cnn, batch_size=BATCH_SIZE, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test

f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_true, y_pred)

total_training_time = gan_time + cnn_time

metrics = {
    'accuracy': round(float(test_accuracy), 4),
    'precision_weighted': round(float(test_precision), 4),
    'recall_weighted': round(float(test_recall), 4),
    'f1_weighted': round(float(f1), 4),
    'test_loss': round(float(test_loss), 4),
    'gan_training_time_seconds': round(gan_time, 2),
    'cnn_training_time_seconds': round(cnn_time, 2),
    'total_training_time_seconds': round(total_training_time, 2),
    'total_training_time_minutes': round(total_training_time / 60, 2),
    'synthetic_samples_generated': num_synthetic,
    'gan_epochs': GAN_EPOCHS,
    'cnn_epochs': len(history.history['loss'])
}

log.info("="*80)
log.info("FINAL HYBRID MODEL METRICS:")
log.info("="*80)
for metric, value in metrics.items():
    log.info(f"  {metric}: {value}")
log.info("="*80)

print("\nConfusion Matrix:")
print(cm)

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

# Save models
log.info("Saving hybrid models...")
model_path = os.path.join(MODEL_DIR, OUTPUT_MODEL)
cnn_model.save(model_path)
log.info(f"CNN model saved -> {model_path}")

generator_path = os.path.join(MODEL_DIR, 'cnn_gan_hybrid_generator.h5')
generator.save(generator_path)
log.info(f"Generator saved -> {generator_path}")

# Save training history
history_path = os.path.join(MODEL_DIR, 'cnn_gan_hybrid_history.json')
history_dict = {
    'cnn_loss': [float(x) for x in history.history['loss']],
    'cnn_accuracy': [float(x) for x in history.history['accuracy']],
    'cnn_val_loss': [float(x) for x in history.history['val_loss']],
    'cnn_val_accuracy': [float(x) for x in history.history['val_accuracy']],
}
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)
log.info(f"Training history saved -> {history_path}")

# Update metrics file
metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
    except Exception:
        metrics_data = {}
else:
    metrics_data = {}

key = os.path.splitext(os.path.basename(OUTPUT_MODEL))[0]
metrics_data[key] = {
    'model': OUTPUT_MODEL,
    'metrics': metrics,
    'confusion_matrix': cm.tolist(),
    'samples': int(len(df)),
    'algorithm': 'CNN-GAN-Hybrid',
    'training_date': datetime.now().isoformat(),
    'feature_count': num_features,
    'class_distribution': class_dist.to_dict(),
    'model_params': {
        'gan_epochs': GAN_EPOCHS,
        'cnn_epochs': CNN_EPOCHS,
        'batch_size': BATCH_SIZE,
        'latent_dim': LATENT_DIM
    }
}

with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, indent=2)

total_time = time.time() - start_time
log_kv(log, total_time_s=round(total_time, 2), total_time_min=round(total_time/60, 2))
log.info(f"Metrics updated -> {metrics_path}")
log.info("="*80)
log.info("CNN-GAN Hybrid training pipeline completed successfully!")
log.info("="*80)

if __name__ == "__main__":
    pass
