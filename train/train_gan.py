"""GAN (Generative Adversarial Network) training script for AI NIDS.
Designed for ~1 hour training with real-time progress monitoring.
Generates synthetic attack samples to augment minority classes.

Environment Variables:
  PRIMARY_DATA=unsw_selected_features.csv
  OUTPUT_MODEL=saved_model_gan
  MAX_ROWS=150000
  EPOCHS=100
  BATCH_SIZE=128
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
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GAN] GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"[GAN] GPU setup error: {e}")
else:
    print("[GAN] No GPU found, using CPU")

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
OUTPUT_MODEL = os.environ.get('OUTPUT_MODEL', 'saved_model_gan')
MAX_ROWS = int(os.environ.get('MAX_ROWS', '150000'))
EPOCHS = int(os.environ.get('EPOCHS', '100'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '128'))
LATENT_DIM = int(os.environ.get('LATENT_DIM', '100'))
RANDOM_STATE = 42

# Set random seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

log = setup_logging("train.gan")
log.info("=" * 80)
log.info("Starting GAN Training for AI-NIDS")
log.info("=" * 80)
log_kv(log, PRIMARY_DATA=PRIMARY_FILE, OUTPUT_MODEL=OUTPUT_MODEL)
log_kv(log, MAX_ROWS=MAX_ROWS, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, LATENT_DIM=LATENT_DIM)

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

# Separate features and labels
X = df.drop('label', axis=1).values
y = df['label'].values
num_features = X.shape[1]

log.info(f"Feature dimension: {num_features}")
log.info(f"Number of classes: {len(np.unique(y))}")

# Normalize data
log.info("Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
log_kv(log, train_shape=X_train.shape, test_shape=X_test.shape)

# Build Generator
def build_generator(latent_dim, output_dim):
    model = models.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(1024),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(output_dim, activation='tanh')
    ], name='generator')
    return model

# Build Discriminator
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

# Build GAN
log.info("Building GAN architecture...")
generator = build_generator(LATENT_DIM, num_features)
discriminator = build_discriminator(num_features)

# Compile discriminator
discriminator.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Build and compile GAN
discriminator.trainable = False
gan_input = layers.Input(shape=(LATENT_DIM,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = models.Model(gan_input, gan_output, name='gan')

gan.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)

# Model summaries
log.info("\nGenerator architecture:")
generator.summary(print_fn=lambda x: log.info(x))
log.info("\nDiscriminator architecture:")
discriminator.summary(print_fn=lambda x: log.info(x))

# Training tracking
d_losses = []
g_losses = []
d_accuracies = []

log.info("="*80)
log.info("STARTING GAN TRAINING")
log.info("="*80)

training_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    
    num_batches = len(X_train) // BATCH_SIZE
    d_loss_epoch = []
    g_loss_epoch = []
    d_acc_epoch = []
    
    for batch_idx in range(num_batches):
        # Get real samples
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        real_samples = X_train_shuffled[start_idx:end_idx]
        
        # Generate fake samples
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, LATENT_DIM))
        fake_samples = generator.predict(noise, verbose=0)
        
        # Train discriminator
        discriminator.trainable = True
        
        # Labels for real and fake samples (with label smoothing)
        real_labels = np.ones((BATCH_SIZE, 1)) * 0.9
        fake_labels = np.zeros((BATCH_SIZE, 1))
        
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, LATENT_DIM))
        g_loss = gan.train_on_batch(noise, real_labels)
        
        d_loss_epoch.append(d_loss[0])
        g_loss_epoch.append(g_loss)
        d_acc_epoch.append(d_loss[1])
        
        # Show batch progress every 50 batches
        if batch_idx % 50 == 0 and batch_idx > 0:
            print(f"  Batch {batch_idx}/{num_batches}: D_loss={d_loss[0]:.4f}, G_loss={g_loss:.4f}, D_acc={d_loss[1]:.4f}")
    
    # Calculate epoch metrics
    avg_d_loss = np.mean(d_loss_epoch)
    avg_g_loss = np.mean(g_loss_epoch)
    avg_d_acc = np.mean(d_acc_epoch)
    
    d_losses.append(avg_d_loss)
    g_losses.append(avg_g_loss)
    d_accuracies.append(avg_d_acc)
    
    epoch_time = time.time() - epoch_start
    progress = ((epoch + 1) / EPOCHS) * 100
    elapsed_time = time.time() - training_start
    eta = (elapsed_time / (epoch + 1)) * (EPOCHS - epoch - 1)
    
    # Print epoch summary
    print(f"\n{'='*80}")
    print(f"✓ EPOCH {epoch + 1}/{EPOCHS} completed in {epoch_time:.2f}s")
    print(f"  Progress: {progress:.1f}% [{('█' * int(progress/2)).ljust(50)}]")
    print(f"  Discriminator Loss: {avg_d_loss:.4f} | Accuracy: {avg_d_acc:.4f}")
    print(f"  Generator Loss: {avg_g_loss:.4f}")
    print(f"  Elapsed: {elapsed_time/60:.1f}min | ETA: {eta/60:.1f}min")
    print(f"{'='*80}\n")
    
    # Save checkpoint every 20 epochs
    if (epoch + 1) % 20 == 0:
        checkpoint_path = os.path.join(MODEL_DIR, f'{OUTPUT_MODEL}_checkpoint_epoch{epoch+1}.h5')
        generator.save(checkpoint_path)
        log.info(f"Checkpoint saved: {checkpoint_path}")

training_end = time.time()
training_time = training_end - training_start

print("\n" + "="*80)
print("GAN TRAINING COMPLETED")
print("="*80 + "\n")

log_kv(log, training_time_s=round(training_time, 2), training_time_min=round(training_time/60, 2))

# Generate synthetic samples
log.info("Generating synthetic samples...")
num_synthetic = 10000
noise = np.random.normal(0, 1, size=(num_synthetic, LATENT_DIM))
synthetic_samples = generator.predict(noise, verbose=1)

# Inverse transform to original scale
synthetic_samples_original = scaler.inverse_transform(synthetic_samples)

# Save synthetic samples
synthetic_df = pd.DataFrame(synthetic_samples_original, columns=df.drop('label', axis=1).columns)
synthetic_df['label'] = np.random.choice(y_train, size=num_synthetic)  # Assign random labels from training set
synthetic_path = os.path.join(DATA_DIR, 'gan_synthetic_samples.csv')
synthetic_df.to_csv(synthetic_path, index=False)
log.info(f"Synthetic samples saved -> {synthetic_path}")

# Save models
log.info("Saving GAN models...")
generator_path = os.path.join(MODEL_DIR, f'{OUTPUT_MODEL}_generator.h5')
discriminator_path = os.path.join(MODEL_DIR, f'{OUTPUT_MODEL}_discriminator.h5')
gan_path = os.path.join(MODEL_DIR, f'{OUTPUT_MODEL}_full.h5')

generator.save(generator_path)
discriminator.save(discriminator_path)
gan.save(gan_path)

log.info(f"Generator saved -> {generator_path}")
log.info(f"Discriminator saved -> {discriminator_path}")
log.info(f"GAN saved -> {gan_path}")

# Save training history plot
log.info("Plotting training history...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(d_losses, label='Discriminator Loss', color='blue')
plt.plot(g_losses, label='Generator Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Losses')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(d_accuracies, label='Discriminator Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Discriminator Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(1, EPOCHS+1), [i for i in range(1, EPOCHS+1)], 'g-', alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('Epoch Number')
plt.title('Training Progress')
plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, 'gan_training_history.png')
plt.savefig(plot_path, dpi=150)
plt.close()
log.info(f"Training plot saved -> {plot_path}")

# Save training history
history_path = os.path.join(MODEL_DIR, 'gan_training_history.json')
history_dict = {
    'discriminator_loss': [float(x) for x in d_losses],
    'generator_loss': [float(x) for x in g_losses],
    'discriminator_accuracy': [float(x) for x in d_accuracies],
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'latent_dim': LATENT_DIM
}
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)
log.info(f"Training history saved -> {history_path}")

# Update metrics file
metrics = {
    'final_d_loss': round(float(d_losses[-1]), 4),
    'final_g_loss': round(float(g_losses[-1]), 4),
    'final_d_accuracy': round(float(d_accuracies[-1]), 4),
    'training_time_seconds': round(training_time, 2),
    'training_time_minutes': round(training_time / 60, 2),
    'epochs_trained': EPOCHS,
    'synthetic_samples_generated': num_synthetic
}

log.info("="*80)
log.info("FINAL GAN METRICS:")
log.info("="*80)
for metric, value in metrics.items():
    log.info(f"  {metric}: {value}")
log.info("="*80)

metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
    except Exception:
        metrics_data = {}
else:
    metrics_data = {}

key = OUTPUT_MODEL
metrics_data[key] = {
    'model': OUTPUT_MODEL,
    'metrics': metrics,
    'samples': int(len(df)),
    'algorithm': 'GAN',
    'training_date': datetime.now().isoformat(),
    'feature_count': num_features,
    'class_distribution': class_dist.to_dict(),
    'model_params': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'latent_dim': LATENT_DIM,
        'learning_rate': 0.0002
    }
}

with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, indent=2)

total_time = time.time() - start_time
log_kv(log, total_time_s=round(total_time, 2), total_time_min=round(total_time/60, 2))
log.info(f"Metrics updated -> {metrics_path}")
log.info("="*80)
log.info("GAN training pipeline completed successfully!")
log.info("="*80)

if __name__ == "__main__":
    pass
