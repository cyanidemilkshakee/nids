"""RNN (Recurrent Neural Network) training script for AI NIDS.
Designed for ~1 hour training with real-time progress monitoring.
Uses vanilla RNN architecture for sequential network traffic analysis.

Environment Variables:
  PRIMARY_DATA=unsw_selected_features.csv
  OUTPUT_MODEL=saved_model_rnn.h5
  MAX_ROWS=200000
  TEST_SIZE=0.2
  EPOCHS=45
  BATCH_SIZE=256
  RNN_UNITS=128
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
from sklearn.utils.class_weight import compute_class_weight

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[RNN] GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"[RNN] GPU setup error: {e}")
else:
    print("[RNN] No GPU found, using CPU")

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
OUTPUT_MODEL = os.environ.get('OUTPUT_MODEL', 'saved_model_rnn.h5')
MAX_ROWS = int(os.environ.get('MAX_ROWS', '200000'))
TEST_SIZE = float(os.environ.get('TEST_SIZE', '0.2'))
EPOCHS = int(os.environ.get('EPOCHS', '45'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '256'))
RNN_UNITS = int(os.environ.get('RNN_UNITS', '128'))
RANDOM_STATE = 42

# Set random seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

log = setup_logging("train.rnn")
log.info("=" * 80)
log.info("Starting RNN Training for AI-NIDS")
log.info("=" * 80)
log_kv(log, PRIMARY_DATA=PRIMARY_FILE, OUTPUT_MODEL=OUTPUT_MODEL)
log_kv(log, MAX_ROWS=MAX_ROWS, TEST_SIZE=TEST_SIZE, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, RNN_UNITS=RNN_UNITS)

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

# Ensure numeric
X = X.astype(np.float32)
num_classes = len(np.unique(y))
num_features = X.shape[1]
log.info(f"Number of classes: {num_classes}")
log.info(f"Number of features: {num_features}")

# One-hot encode labels
y_categorical = to_categorical(y, num_classes=num_classes)
log.info(f"Label shape after one-hot encoding: {y_categorical.shape}")

# Split data
log.info("Splitting train/test with stratify...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
y_train_labels = y[:len(X_train)]

log_kv(log, train_shape=X_train.shape, test_shape=X_test.shape)

# Reshape for RNN (samples, timesteps, features)
# For network traffic, we treat each feature as a timestep with 1 feature dimension
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
log.info(f"Reshaped for RNN: train={X_train_reshaped.shape}, test={X_test_reshaped.shape}")

# Compute class weights
log.info("Computing class weights...")
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
log.info(f"Class weights: {dict(zip(classes, np.round(class_weights, 3)))}")

# Build RNN model
log.info("Building RNN architecture...")
def build_rnn_model(input_shape, num_classes, rnn_units=128):
    model = models.Sequential([
        # First SimpleRNN layer with return sequences
        layers.SimpleRNN(rnn_units, return_sequences=True, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second SimpleRNN layer with return sequences
        layers.SimpleRNN(rnn_units // 2, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third SimpleRNN layer without return sequences
        layers.SimpleRNN(rnn_units // 4),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_rnn_model(
    input_shape=(X_train_reshaped.shape[1], 1),
    num_classes=num_classes,
    rnn_units=RNN_UNITS
)

# Compile model
log.info("Compiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Model summary
log.info("Model architecture:")
model.summary(print_fn=lambda x: log.info(x))

# Callbacks for real-time monitoring
class RealTimeProgressCallback(callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{self.total_epochs}")
        print(f"{'='*80}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        progress = ((epoch + 1) / self.total_epochs) * 100
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (self.total_epochs - epoch - 1)
        
        print(f"\n{'='*80}")
        print(f"✓ Epoch {epoch + 1}/{self.total_epochs} completed in {epoch_time:.2f}s")
        print(f"  Progress: {progress:.1f}% [{('█' * int(progress/2)).ljust(50)}]")
        print(f"  Loss: {logs.get('loss', 0):.4f} | Accuracy: {logs.get('accuracy', 0):.4f}")
        print(f"  Val Loss: {logs.get('val_loss', 0):.4f} | Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
        print(f"  Precision: {logs.get('precision', 0):.4f} | Recall: {logs.get('recall', 0):.4f}")
        print(f"  Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
        print(f"{'='*80}\n")
        
    def on_batch_end(self, batch, logs=None):
        # Show progress every 30 batches
        if batch % 30 == 0 and batch > 0:
            print(f"  Batch {batch}: loss={logs.get('loss', 0):.4f}, accuracy={logs.get('accuracy', 0):.4f}")

progress_callback = RealTimeProgressCallback(total_epochs=EPOCHS)

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

checkpoint_path = os.path.join(MODEL_DIR, 'rnn_checkpoint.h5')
checkpoint = callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train model
log.info(f"Starting RNN training for {EPOCHS} epochs...")
log.info("Real-time progress will be displayed below:")
print("\n" + "="*80)
print("TRAINING START")
print("="*80 + "\n")

training_start = time.time()

history = model.fit(
    X_train_reshaped,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[progress_callback, early_stop, reduce_lr, checkpoint],
    verbose=0  # Suppress default output, use custom callback
)

training_end = time.time()
training_time = training_end - training_start

print("\n" + "="*80)
print("TRAINING COMPLETED")
print("="*80 + "\n")

log_kv(log, training_time_s=round(training_time, 2), training_time_min=round(training_time/60, 2))

# Evaluate on test set
log.info("Evaluating on test set...")
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
    X_test_reshaped,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Generate predictions
log.info("Generating predictions...")
y_pred_probs = model.predict(X_test_reshaped, batch_size=BATCH_SIZE, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate metrics
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_true, y_pred)

metrics = {
    'accuracy': round(float(test_accuracy), 4),
    'precision_weighted': round(float(test_precision), 4),
    'recall_weighted': round(float(test_recall), 4),
    'f1_weighted': round(float(f1), 4),
    'test_loss': round(float(test_loss), 4),
    'training_time_seconds': round(training_time, 2),
    'training_time_minutes': round(training_time / 60, 2),
    'epochs_trained': len(history.history['loss'])
}

log.info("="*80)
log.info("FINAL TEST METRICS:")
log.info("="*80)
for metric, value in metrics.items():
    log.info(f"  {metric}: {value}")
log.info("="*80)

print("\nConfusion Matrix:")
print(cm)

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

# Save model
log.info(f"Saving RNN model to {os.path.join(MODEL_DIR, OUTPUT_MODEL)}")
model_path = os.path.join(MODEL_DIR, OUTPUT_MODEL)
model.save(model_path)
log.info(f"Model saved -> {model_path}")

# Save model architecture as JSON
model_json = model.to_json()
json_path = os.path.join(MODEL_DIR, 'rnn_architecture.json')
with open(json_path, 'w') as f:
    f.write(model_json)
log.info(f"Model architecture saved -> {json_path}")

# Save training history
history_path = os.path.join(MODEL_DIR, 'rnn_training_history.json')
history_dict = {
    'loss': [float(x) for x in history.history['loss']],
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'precision': [float(x) for x in history.history['precision']],
    'recall': [float(x) for x in history.history['recall']],
}
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)
log.info(f"Training history saved -> {history_path}")

# Update metrics file
log.info("Updating metrics file...")
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
    'algorithm': 'RNN',
    'training_date': datetime.now().isoformat(),
    'feature_count': num_features,
    'class_distribution': class_dist.to_dict(),
    'model_params': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'rnn_units': RNN_UNITS,
        'optimizer': 'Adam',
        'learning_rate': 0.001
    }
}

with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, indent=2)

total_time = time.time() - start_time
log_kv(log, total_time_s=round(total_time, 2), total_time_min=round(total_time/60, 2))
log.info(f"Metrics updated -> {metrics_path}")
log.info("="*80)
log.info("RNN training pipeline completed successfully!")
log.info("="*80)

if __name__ == "__main__":
    pass
