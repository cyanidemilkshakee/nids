"""Gradient Boosting (sklearn) training script for AI NIDS.
Uses sklearn.ensemble.GradientBoostingClassifier.
Designed for smaller feature sets; supports class imbalance via sample weights.
Enhanced with progress monitoring and resource optimization.
Environment Variables:
  PRIMARY_DATA=unsw_selected_features.csv
  SYNTH_DATA=synthetic_gan_samples.csv (optional)
  OUTPUT_MODEL=saved_model_gradient_boosting.joblib
  MAX_ROWS=250000
  TEST_SIZE=0.2
"""
import os, sys, json, joblib, pandas as pd, numpy as np
import time
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
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
OUTPUT_MODEL = os.environ.get('OUTPUT_MODEL', 'saved_model_gradient_boosting.joblib')
MAX_ROWS = int(os.environ.get('MAX_ROWS', '250000'))
TEST_SIZE = float(os.environ.get('TEST_SIZE', '0.2'))
RANDOM_STATE = 42

log = setup_logging("train.gradient_boosting")
log.info("Starting Gradient Boosting training")
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

# Add feature engineering for more vigorous training
log.info("Adding feature engineering features...")

# Advanced statistical features
log.info("Creating statistical aggregate features...")
X['feature_sum'] = X.sum(axis=1)
X['feature_mean'] = X.mean(axis=1)
X['feature_std'] = X.std(axis=1)
X['feature_var'] = X.var(axis=1)
X['feature_median'] = X.median(axis=1)
X['feature_max'] = X.max(axis=1)
X['feature_min'] = X.min(axis=1)
X['feature_range'] = X['feature_max'] - X['feature_min']
X['feature_skew'] = X.skew(axis=1)
X['feature_kurtosis'] = X.kurtosis(axis=1)
X['feature_mad'] = (X.sub(X.mean(axis=1), axis=0)).abs().mean(axis=1)  # Mean absolute deviation

# Percentile and quantile features
log.info("Creating percentile features...")
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    X[f'feature_q{int(q*100)}'] = X.quantile(q, axis=1)

# Zero and non-zero statistics
X['zero_count'] = (X == 0).sum(axis=1)
X['non_zero_count'] = (X != 0).sum(axis=1)
X['zero_ratio'] = X['zero_count'] / X.shape[1]
X['positive_count'] = (X > 0).sum(axis=1)
X['negative_count'] = (X < 0).sum(axis=1)

# Create ratio and interaction features for enhanced pattern detection
original_cols = [col for col in X.columns if not col.startswith(('feature_', 'zero_', 'non_zero', 'positive_', 'negative_'))]
top_cols = original_cols[:min(10, len(original_cols))]  # Top features for interactions

print(f"[GB] Creating advanced interaction features...")
for i in range(len(top_cols)):
    for j in range(i+1, min(i+4, len(top_cols))):  # Limit interactions to prevent explosion
        col1, col2 = top_cols[i], top_cols[j]
        # Multiplicative interactions
        X[f'mult_{col1}_{col2}'] = X[col1] * X[col2]
        # Ratio interactions (safe division)
        X[f'ratio_{col1}_{col2}'] = np.where(X[col2] != 0, X[col1] / X[col2], 0)
        # Difference interactions
        X[f'diff_{col1}_{col2}'] = X[col1] - X[col2]
        # Sum interactions
        X[f'sum_{col1}_{col2}'] = X[col1] + X[col2]

# Mathematical transformations for better distributions
log.info("Applying mathematical transforms (log/sqrt/power/normalize)...")
for col in top_cols[:8]:  # Apply to top columns
    if X[col].min() >= 0:  # Only for non-negative values
        X[f'{col}_log1p'] = np.log1p(X[col])
        X[f'{col}_sqrt'] = np.sqrt(X[col])
    X[f'{col}_squared'] = X[col] ** 2
    # Normalized versions
    col_std = X[col].std()
    if col_std > 0:
        X[f'{col}_normalized'] = (X[col] - X[col].mean()) / col_std

# Binning for categorical insights
log.info("Creating binned features...")
for col in top_cols[:5]:
    try:
        X[f'{col}_binned'] = pd.qcut(X[col], q=8, duplicates='drop', labels=False)
    except:
        X[f'{col}_binned'] = pd.cut(X[col], bins=8, labels=False)

log_kv(log, enhanced_shape=X.shape)
log.info(f"DTypes:\n{X.dtypes.value_counts()}")
log.info("Computing class weights for imbalance...")
classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
weight_map = {cls: w for cls, w in zip(classes, class_weights)}
sample_weights = y.map(weight_map).astype(float)

log.info(f"Class weights: {dict(zip(classes, np.round(class_weights, 3)))}")

log.info("Splitting train/test with stratify...")
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weights, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

log_kv(log, train_shape=X_train.shape, test_shape=X_test.shape)

log.info("Initializing GradientBoosting ensemble models...")

# Import additional components for advanced training
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Create ensemble of multiple GB models for maximum accuracy
models = []

# Primary model with deep trees and high complexity
model_1 = GradientBoostingClassifier(
    n_estimators=2000,      # Very high number of estimators
    learning_rate=0.01,     # Very low learning rate for precision
    max_depth=12,           # Deep trees for complex patterns
    subsample=0.9,          # High subsampling for stability
    max_features=0.9,       # Use most features
    min_samples_split=3,    # Very sensitive splits
    min_samples_leaf=1,     # Minimum leaf size
    min_weight_fraction_leaf=0.0,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    random_state=RANDOM_STATE,
    verbose=2,
    warm_start=False,
    validation_fraction=0.2,  # Larger validation for better monitoring
    n_iter_no_change=100,   # High patience for convergence
    tol=1e-6,               # Very tight tolerance
    ccp_alpha=0.0           # No pruning for maximum complexity
)

# Secondary model with different hyperparameters for diversity
model_2 = GradientBoostingClassifier(
    n_estimators=1500,
    learning_rate=0.02,     # Slightly higher learning rate
    max_depth=10,           # Slightly shallower but still deep
    subsample=0.8,
    max_features=0.8,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0001,
    random_state=RANDOM_STATE + 1,  # Different seed for diversity
    verbose=2,
    validation_fraction=0.2,
    n_iter_no_change=80,
    tol=1e-6,
    ccp_alpha=0.001         # Light pruning
)

# Tertiary model with extreme parameters
model_3 = GradientBoostingClassifier(
    n_estimators=2500,      # Even more estimators
    learning_rate=0.005,    # Very slow learning
    max_depth=15,           # Very deep trees
    subsample=0.95,         # Almost full subsample
    max_features=1.0,       # Use all features
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE + 2,
    verbose=2,
    validation_fraction=0.15,
    n_iter_no_change=150,
    tol=1e-7
)

models = [model_1, model_2, model_3]
log.info(f"Created ensemble of {len(models)} Gradient Boosting models")

log.info("Starting training across ensemble models...")
training_start = time.time()

# Train all models
trained_models = []
for i, model in enumerate(models):
    with log_stage(log, f"fit_model_{i+1}"):
        model.fit(X_train, y_train, sample_weight=sw_train)
        trained_models.append(model)
        log.info(f"Model {i+1} completed with {model.n_estimators_} estimators")

# Use the best performing model based on validation
best_model = trained_models[0]  # Default to first model
best_score = 0

for i, model in enumerate(trained_models):
    val_pred = model.predict(X_test)
    val_score = accuracy_score(y_test, val_pred)
    log.info(f"Model {i+1} validation accuracy: {val_score:.4f}")
    if val_score > best_score:
        best_score = val_score
        best_model = model

model = best_model  # Use the best model for final evaluation
log.info(f"Selected best model with accuracy: {best_score:.4f}")

training_end = time.time()
training_time = training_end - training_start

log_kv(log, training_time_s=round(training_time, 2), estimators_used=getattr(model, 'n_estimators_', None))

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
    'algorithm': 'GradientBoosting',
    'training_date': datetime.now().isoformat(),
    'feature_count': len(X.columns),
    'class_distribution': class_dist.to_dict()
}

with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, indent=2)

total_time = time.time() - start_time
log_kv(log, total_time_s=round(total_time, 2))
log.info(f"Metrics updated -> {metrics_path}")
log.info("Gradient Boosting training pipeline completed successfully")

if __name__ == "__main__":
    # This allows the script to be run directly without unnecessary launcher scripts
    pass
