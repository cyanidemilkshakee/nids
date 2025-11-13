"""LightGBM training script for AI NIDS.
Fast gradient boosting framework optimized for efficiency and accuracy.
Enhanced with progress monitoring and resource optimization.
Environment Variables:
  PRIMARY_DATA=unsw_selected_features.csv (or cicids_selected_features.csv)
  SYNTH_DATA=synthetic_gan_samples.csv (optional augmentation)
  OUTPUT_MODEL=saved_model_lightgbm.joblib
  MAX_ROWS=250000
  TEST_SIZE=0.2
Dependencies:
  pip install lightgbm
"""
import os, sys, json, joblib, pandas as pd, numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
# Make project root importable when running as a script
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logging_utils import setup_logging, log_stage, log_kv

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise SystemExit("lightgbm not installed. Run: pip install lightgbm")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

PRIMARY_FILE = os.environ.get('PRIMARY_DATA', 'unsw_selected_features.csv')
SYNTH_FILE = os.environ.get('SYNTH_DATA', 'synthetic_gan_samples.csv')
OUTPUT_MODEL = os.environ.get('OUTPUT_MODEL', 'saved_model_lightgbm.joblib')
MAX_ROWS = int(os.environ.get('MAX_ROWS', '250000'))
TEST_SIZE = float(os.environ.get('TEST_SIZE', '0.2'))
RANDOM_STATE = 42

log = setup_logging("train.lightgbm")
log.info("Starting LightGBM training")
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

# Ensure all features are float32 for efficiency
X = X.astype('float32')

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
X['feature_iqr'] = X.quantile(0.75, axis=1) - X.quantile(0.25, axis=1)
X['feature_mad'] = (X.sub(X.mean(axis=1), axis=0)).abs().mean(axis=1)  # Mean absolute deviation
X['feature_sem'] = X.sem(axis=1)  # Standard error of mean

# Advanced percentile features for robust statistics
log.info("Creating advanced percentile features...")
percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
for p in percentiles:
    X[f'feature_p{int(p*100)}'] = X.quantile(p, axis=1)

# Count-based features
log.info("Creating count-based features...")
X['zero_count'] = (X == 0).sum(axis=1)
X['non_zero_count'] = (X != 0).sum(axis=1)
X['positive_count'] = (X > 0).sum(axis=1)
X['negative_count'] = (X < 0).sum(axis=1)
X['zero_ratio'] = X['zero_count'] / X.shape[1]
X['positive_ratio'] = X['positive_count'] / X.shape[1]
X['negative_ratio'] = X['negative_count'] / X.shape[1]

# Outlier detection features
log.info("Creating outlier/outlier-ratio features...")
Q1 = X.quantile(0.25, axis=1)
Q3 = X.quantile(0.75, axis=1)
IQR = Q3 - Q1
X['outlier_count_iqr'] = ((X.T < (Q1 - 1.5 * IQR)) | (X.T > (Q3 + 1.5 * IQR))).sum()
X['outlier_ratio_iqr'] = X['outlier_count_iqr'] / X.shape[1]

# Advanced interaction features
original_features = [col for col in X.columns if not col.startswith(('feature_', 'zero_', 'non_zero', 'positive_', 'negative_', 'outlier_'))]
log.info(f"Creating interaction features from {len(original_features)} original features...")

# Select top variance features for interactions
feature_variances = X[original_features].var().sort_values(ascending=False)
top_var_features = feature_variances.head(15).index.tolist()

# Create comprehensive interaction features
for i in range(len(top_var_features)):
    for j in range(i+1, min(i+5, len(top_var_features))):  # Limit to prevent explosion
        feat1, feat2 = top_var_features[i], top_var_features[j]
        # Multiplicative interactions
        X[f'mult_{feat1}_{feat2}'] = X[feat1] * X[feat2]
        # Ratio interactions (safe division)
        X[f'ratio_{feat1}_{feat2}'] = np.where(X[feat2] != 0, X[feat1] / X[feat2], 0)
        # Difference interactions
        X[f'diff_{feat1}_{feat2}'] = X[feat1] - X[feat2]
        # Harmonic mean interactions
        X[f'hmean_{feat1}_{feat2}'] = np.where((X[feat1] > 0) & (X[feat2] > 0), 
                                              2 * X[feat1] * X[feat2] / (X[feat1] + X[feat2]), 0)

# Mathematical transformations for better feature distributions
log.info("Applying mathematical transforms (log/sqrt/power/normalize/rank)...")
for feature in top_var_features[:10]:
    if X[feature].min() >= 0:  # Only for non-negative features
        X[f'{feature}_log1p'] = np.log1p(X[feature])
        X[f'{feature}_sqrt'] = np.sqrt(X[feature])
        X[f'{feature}_cbrt'] = np.cbrt(X[feature])  # Cube root
    
    # Power transformations
    X[f'{feature}_squared'] = X[feature] ** 2
    X[f'{feature}_cubed'] = X[feature] ** 3
    
    # Normalized versions
    feature_std = X[feature].std()
    if feature_std > 0:
        X[f'{feature}_normalized'] = (X[feature] - X[feature].mean()) / feature_std
        X[f'{feature}_minmax'] = (X[feature] - X[feature].min()) / (X[feature].max() - X[feature].min() + 1e-8)
    
    # Rank-based features
    X[f'{feature}_rank'] = X[feature].rank()
    X[f'{feature}_rank_norm'] = X[f'{feature}_rank'] / len(X)

# Binning features for categorical insights
log.info("Creating binned features (quantile/equal-width)...")
for feature in top_var_features[:8]:
    try:
        # Quantile-based binning
        X[f'{feature}_qbinned'] = pd.qcut(X[feature], q=10, duplicates='drop', labels=False)
        # Equal-width binning
        X[f'{feature}_ebinned'] = pd.cut(X[feature], bins=10, labels=False)
    except:
        X[f'{feature}_qbinned'] = 0
        X[f'{feature}_ebinned'] = 0

# Distance and similarity features
log.info("Creating distance and similarity features...")
original_feature_matrix = X[original_features]
feature_mean_vector = original_feature_matrix.mean()
feature_median_vector = original_feature_matrix.median()

# Multiple distance metrics
X['euclidean_dist_from_mean'] = np.sqrt(((original_feature_matrix - feature_mean_vector) ** 2).sum(axis=1))
X['manhattan_dist_from_mean'] = (original_feature_matrix - feature_mean_vector).abs().sum(axis=1)
X['euclidean_dist_from_median'] = np.sqrt(((original_feature_matrix - feature_median_vector) ** 2).sum(axis=1))
X['manhattan_dist_from_median'] = (original_feature_matrix - feature_median_vector).abs().sum(axis=1)
X['cosine_similarity_to_mean'] = (original_feature_matrix * feature_mean_vector).sum(axis=1) / (
    np.sqrt((original_feature_matrix ** 2).sum(axis=1)) * np.sqrt((feature_mean_vector ** 2).sum()) + 1e-8)

# Fourier transform features for frequency domain analysis
log.info("Creating frequency-domain (FFT) features for top variance columns...")
for feature in top_var_features[:5]:
    fft = np.fft.fft(X[feature].values)
    X[f'{feature}_fft_real'] = np.real(fft)
    X[f'{feature}_fft_imag'] = np.imag(fft)
    X[f'{feature}_fft_magnitude'] = np.abs(fft)

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

# Create validation set for early stopping
X_train_fit, X_val, y_train_fit, y_val, sw_train_fit, sw_val = train_test_split(
    X_train, y_train, sw_train, test_size=0.1, stratify=y_train, random_state=RANDOM_STATE
)

log_kv(log, train_fit_shape=X_train_fit.shape, val_shape=X_val.shape)

log.info("Initializing LGBMClassifier with tuned parameters...")

# Advanced LightGBM for maximum accuracy
model = LGBMClassifier(
    objective='multiclass',
    num_class=len(classes),
    boosting_type='gbdt',
    num_leaves=255,             # Higher leaf count for more complex trees
    max_depth=20,               # Very deep trees for complex patterns
    learning_rate=0.01,         # Very low learning rate for precision
    n_estimators=10000,         # Very high with early stopping
    subsample_for_bin=300000,   # More samples for bin construction
    min_split_gain=0.005,       # Higher minimum gain for quality splits
    min_child_weight=1e-1,      # Stricter minimum weight
    min_child_samples=5,        # Smaller minimum samples for detailed patterns
    subsample=0.9,              # High subsampling for stability
    subsample_freq=1,           # Frequency of subsampling
    colsample_bytree=0.9,       # Use most features
    reg_alpha=0.5,              # Strong L1 regularization
    reg_lambda=3.0,             # Strong L2 regularization
    max_bin=1023,               # Maximum bins for finest splits
    min_data_in_bin=3,          # Minimum data in bin
    feature_fraction_bynode=0.9, # High feature fraction by node
    bagging_fraction=0.9,       # High bagging fraction
    bagging_freq=1,             # Bagging frequency
    lambda_l1=0.5,              # L1 lambda
    lambda_l2=3.0,              # L2 lambda
    min_gain_to_split=0.005,    # Minimum gain to split
    drop_rate=0.05,             # Lower dropout rate for stability
    max_drop=80,                # Max number of dropped trees
    skip_drop=0.3,              # Probability of skipping dropout
    extra_trees=False,          # Use regular trees
    path_smooth=0.05,           # Path smoothing
    num_iterations=10000,       # Maximum iterations
    early_stopping_rounds=300,  # High patience for convergence
    feature_fraction=0.9,       # High feature fraction
    bagging_seed=RANDOM_STATE,
    feature_fraction_seed=RANDOM_STATE,
    random_state=RANDOM_STATE,
    n_jobs=-1,                  # Use all CPU cores
    class_weight='balanced',    # Handle class imbalance
    importance_type='gain',     # Feature importance calculation
    verbose=1,                  # Show progress
    linear_tree=False,          # Use tree-based learning
    device_type='cpu',          # Force CPU usage
    deterministic=True,         # Deterministic training
    force_col_wise=True,        # Column-wise memory access
    histogram_pool_size=256,    # Histogram pool size
    max_conflict_rate=0.1,      # Maximum conflict rate
    # Advanced accuracy parameters
    first_metric_only=False,    # Use all metrics
    boost_from_average=True,    # Boost from average
    is_unbalance=True,          # Handle unbalanced data
    metric_freq=50,             # Metric frequency
    feature_pre_filter=False,   # No pre-filtering
    verbosity=1                 # Verbosity level
)

log.info(f"Starting training with n_estimators={model.n_estimators}")
training_start = time.time()

# Enhanced training with multiple validation strategies
# Create additional validation split for robust monitoring
X_train_main, X_train_val2, y_train_main, y_train_val2, sw_train_main, sw_train_val2 = train_test_split(
    X_train_fit, y_train_fit, sw_train_fit, test_size=0.15, stratify=y_train_fit, random_state=RANDOM_STATE + 1
)

# Advanced callbacks for maximum accuracy
callbacks = [
    lgb.early_stopping(stopping_rounds=300, verbose=True),  # High patience
    lgb.log_evaluation(period=200),  # Less frequent logging for long training
    # Custom learning rate decay
    lgb.reset_parameter(learning_rate=lambda iter: 0.01 * (0.995 ** (iter // 100))),
    # Reset parameters for fine-tuning
    lgb.reset_parameter(feature_fraction=lambda iter: max(0.7, 0.9 * (0.9999 ** iter)))
]

# Fit with comprehensive validation and advanced callbacks
with log_stage(log, "fit_model"):
    model.fit(
        X_train_main, y_train_main,
        sample_weight=sw_train_main,
        eval_set=[
            (X_train_main, y_train_main), 
            (X_train_val2, y_train_val2),
            (X_val, y_val),
            (X_test, y_test)
        ],
        eval_names=['train_main', 'train_val2', 'valid', 'test'],
        eval_sample_weight=[sw_train_main, sw_train_val2, sw_val, sw_test],
        eval_metric=['multi_logloss', 'multi_error'],  # Multiple metrics
        callbacks=callbacks
    )

training_end = time.time()
training_time = training_end - training_start

log_kv(log, training_time_s=round(training_time, 2))
log.info(f"Best iteration: {model.best_iteration_}")
log.info(f"Best score: {model.best_score_}")

log.info("Generating predictions on test set...")
probs = model.predict_proba(X_test)
y_pred = probs.argmax(axis=1)

log.info("Computing evaluation metrics...")
metrics = {
    'accuracy': round(accuracy_score(y_test, y_pred), 4),
    'precision_weighted': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
    'recall_weighted': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
    'f1_weighted': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
    'training_time_seconds': round(training_time, 2),
    'best_iteration': int(model.best_iteration_),
    'best_score': round(float(model.best_score_['valid']['multi_logloss']), 4)
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
print(f"[LGB] Top 10 most important features:")
for i, (feature, importance) in enumerate(top_features, 1):
    print(f"  {i:2d}. {feature}: {importance:.4f}")

log.info(f"Saving model to {os.path.join(MODEL_DIR, OUTPUT_MODEL)}")
model_path = os.path.join(MODEL_DIR, OUTPUT_MODEL)
joblib.dump(model, model_path)
log.info(f"Model saved -> {model_path}")

# Feature names persistence for consistency
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
    'algorithm': 'LightGBM',
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
log.info("LightGBM training pipeline completed successfully")

if __name__ == "__main__":
    # This allows the script to be run directly without unnecessary launcher scripts
    pass
