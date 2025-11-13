"""Extra Trees (Extremely Randomized Trees) training script for AI NIDS.
Similar interface to existing train.py script.
Enhanced with progress monitoring and resource optimization.
Environment Variables:
  PRIMARY_DATA=unsw_selected_features.csv
  SYNTH_DATA=synthetic_gan_samples.csv
  OUTPUT_MODEL=saved_model_extratrees.joblib
  MAX_ROWS=250000
  TEST_SIZE=0.2
"""
import os, sys, json, joblib, pandas as pd, numpy as np
import time
from datetime import datetime
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
# Make project root importable when running as a script
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logging_utils import setup_logging, log_stage, log_kv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

PRIMARY_FILE = os.environ.get('PRIMARY_DATA', 'unsw_selected_features.csv')
SYNTH_FILE = os.environ.get('SYNTH_DATA', 'synthetic_gan_samples.csv')
OUTPUT_MODEL = os.environ.get('OUTPUT_MODEL', 'saved_model_extratrees.joblib')
MAX_ROWS = int(os.environ.get('MAX_ROWS', '250000'))
TEST_SIZE = float(os.environ.get('TEST_SIZE', '0.2'))
RANDOM_STATE = 42

log = setup_logging("train.extratrees")
log.info("Starting Extra Trees training")
log_kv(log, PRIMARY_DATA=PRIMARY_FILE, SYNTH_DATA=SYNTH_FILE, OUTPUT_MODEL=OUTPUT_MODEL)
log_kv(log, MAX_ROWS=MAX_ROWS, TEST_SIZE=TEST_SIZE)

start_time = time.time()

primary_path = os.path.join(DATA_DIR, PRIMARY_FILE)
if not os.path.exists(primary_path):
    raise FileNotFoundError(f"Primary dataset not found: {primary_path}")

with log_stage(log, "load_primary_dataset"):
    log_kv(log, path=primary_path)
    df = pd.read_csv(primary_path)
    log_kv(log, rows=len(df), cols=len(df.columns))

# Check class distribution
class_dist = df['label'].value_counts().sort_index()
log.info("Original class distribution:")
log.info(f"\n{class_dist}")

# Optional synthetic augmentation
synth_path = os.path.join(DATA_DIR, SYNTH_FILE)
if os.path.exists(synth_path):
    try:
        with log_stage(log, "augment_synthetic"):
            log_kv(log, path=synth_path)
            synth_df = pd.read_csv(synth_path)
            missing_cols = [c for c in df.columns if c not in synth_df.columns]
            for mc in missing_cols:
                synth_df[mc] = 0
            synth_df = synth_df[df.columns]
            df = pd.concat([df, synth_df], ignore_index=True)
            log_kv(log, added_rows=len(synth_df), total=len(df))
    except Exception as e:
        log.warning(f"Failed synthetic augmentation: {e}")
else:
    log.info(f"No synthetic augmentation file found at {synth_path}")

if len(df) > MAX_ROWS:
    before = len(df)
    with log_stage(log, "cap_max_rows"):
        df = df.sample(MAX_ROWS, random_state=RANDOM_STATE)
        log_kv(log, before=before, after=len(df), MAX_ROWS=MAX_ROWS)

log_kv(log, final_shape=df.shape)
X = df.drop('label', axis=1)
y = df['label']

# Ensure all features are numeric
log.info("Converting categorical features to numeric if needed...")
for col in X.columns:
    if X[col].dtype == 'object':
        log.info(f"Converting column '{col}' from object to numeric")
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(0)

log_kv(log, feature_matrix=X.shape)
log.info(f"DTypes:\n{X.dtypes.value_counts()}")
log.info("Computing class weights for imbalance...")
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}

log.info(f"Class weights: {dict(zip(classes, np.round(class_weights, 3)))}")

log.info("Splitting train/test with stratify...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

log_kv(log, train_shape=X_train.shape, test_shape=X_test.shape)

log.info("Initializing ExtraTreesClassifier with tuned parameters...")
model = ExtraTreesClassifier(
    n_estimators=1000,       # Increased for better performance
    max_depth=None,
    min_samples_split=5,     # Better balance between bias and variance
    min_samples_leaf=2,
    max_features='sqrt',     # Feature subsampling
    class_weight=class_weight_dict,
    n_jobs=-1,              # Use all available CPU cores
    random_state=RANDOM_STATE,
    bootstrap=False,
    verbose=2,              # Show progress (2 = detailed progress)
    warm_start=False,       # Don't reuse previous fit
    oob_score=False,        # OOB not applicable when bootstrap=False
    max_samples=None        # Use all samples
)

log.info(f"Starting training with n_estimators={model.n_estimators}")
training_start = time.time()

# Fit the model
with log_stage(log, "fit_model"):
    model.fit(X_train, y_train)

training_end = time.time()
training_time = training_end - training_start

log_kv(log, training_time_s=round(training_time, 2))

log.info("Generating predictions on test set...")
probs = model.predict_proba(X_test)
y_pred = probs.argmax(axis=1)

log.info("Computing evaluation metrics...")
metrics = {
    'accuracy': round(accuracy_score(y_test, y_pred), 4),
    'precision_weighted': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
    'recall_weighted': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
    'f1_weighted': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
    'training_time_seconds': round(training_time, 2)
}

log.info("Final metrics:")
for metric, value in metrics.items():
    log.info(f"  {metric}: {value}")

cm = confusion_matrix(y_test, y_pred)
log.info(f"Confusion Matrix:\n{cm}")

# Print detailed classification report
log.info("Detailed Classification Report:")
log.info("\n" + classification_report(y_test, y_pred, zero_division=0))

# Get feature importances
feature_importance = model.feature_importances_
top_features = sorted(zip(X.columns, feature_importance), key=lambda x: x[1], reverse=True)[:10]
print(f"[XT] Top 10 most important features:")
for i, (feature, importance) in enumerate(top_features, 1):
    print(f"  {i:2d}. {feature}: {importance:.4f}")

log.info(f"Saving model to {os.path.join(MODEL_DIR, OUTPUT_MODEL)}")
model_path = os.path.join(MODEL_DIR, OUTPUT_MODEL)
joblib.dump(model, model_path)
log.info(f"Model saved -> {model_path}")

feature_names_path = os.path.join(MODEL_DIR, 'feature_names.json')
if not os.path.exists(feature_names_path):
    log.info(f"Saving feature names to {feature_names_path}")
    with open(feature_names_path, 'w', encoding='utf-8') as f:
        json.dump(list(X.columns), f, indent=2)

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
    'algorithm': 'ExtraTrees',
    'training_date': datetime.now().isoformat(),
    'feature_count': len(X.columns),
    'class_distribution': class_dist.to_dict(),
    'top_features': [{'name': name, 'importance': float(imp)} for name, imp in top_features]
}

with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, indent=2)

total_time = time.time() - start_time
log_kv(log, total_time_s=round(total_time, 2))
log.info(f"Metrics updated -> {metrics_path}")
log.info("Extra Trees training pipeline completed successfully")

if __name__ == "__main__":
    # This allows the script to be run directly without unnecessary launcher scripts
    pass
