from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import csv
import json
from threading import Lock
from datetime import datetime
import tensorflow as tf

# Only import if needed for explainability
import shap
try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

app = Flask(__name__)
CORS(app)

# Utilities
def _resolve_path(path_str: str) -> str:
    """Return absolute path; if relative, resolve from this file's directory."""
    if os.path.isabs(path_str):
        return path_str
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_path = os.path.normpath(os.path.join(base_dir, path_str))
    # Ensure the directory exists for file creation
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
    return resolved_path

# Use absolute paths to prevent working directory issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = _resolve_path('../data/feedback.csv')  # Path to store feedback data
METRICS_PATH = _resolve_path('../models/metrics.json')
MODEL_CACHE = {}
EXPLAINER_CACHE = {}
cache_lock = Lock()
PRESET_DATA_CACHE = {}
PRESET_CACHE_LOCK = Lock()

# Initialize feedback file if it doesn't exist
try:
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w', newline='') as f:
            pass  # Create empty file
except Exception as e:
    print(f"[WARN] Could not initialize feedback file: {e}")

# Extend dataset definitions dynamically by scanning models directory
MODELS_DIR = _resolve_path('../models')

# Model registry structure: { key: { 'path': str, 'type': 'sklearn'|'keras', 'features': list } }
MODEL_REGISTRY = {}


# Helper functions for multi-model support (ensure defined before routes using them)
if 'MODEL_REGISTRY' not in globals():
    MODEL_REGISTRY = {}
if 'GLOBAL_FEATURE_ORDER' not in globals():
    GLOBAL_FEATURE_ORDER = []

# Define helper functions for multi-model support
def _scan_models():
    """Scan the models directory for available model files."""
    try:
        if not os.path.exists(MODELS_DIR):
            print(f"[WARN] Models directory not found: {MODELS_DIR}")
            return
            
        for fname in os.listdir(MODELS_DIR):
            lower = fname.lower()
            fpath = os.path.join(MODELS_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            if lower.endswith(('.joblib', '.pkl')):
                key = os.path.splitext(fname)[0]
                MODEL_REGISTRY[key] = {'path': fpath, 'type': 'sklearn'}
            elif lower.endswith(('.keras', '.h5')):
                key = os.path.splitext(fname)[0]
                MODEL_REGISTRY[key] = {'path': fpath, 'type': 'keras'}
        print(f"[INFO] Scanned models directory, found {len(MODEL_REGISTRY)} models")
    except Exception as e:
        print(f"[WARN] Model scan failed: {e}")

def _load_model(model_key):
    """Load a specific model by key from the registry."""
    with cache_lock:
        if model_key in MODEL_CACHE:
            return MODEL_CACHE[model_key]
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_key}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    meta = MODEL_REGISTRY[model_key]
    mpath = meta['path']
    mtype = meta['type']
    if mtype == 'sklearn':
        loaded_obj = joblib.load(mpath)
        # Handle both old format (direct model) and new format (model package)
        if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
            # New format: model package with preprocessing
            model_package = loaded_obj
            model = model_package['model']
        else:
            # Old format: direct model
            model_package = {
                'model': loaded_obj,
                'preprocessing': None,
                'scaler': None,
                'feature_names': None
            }
            model = loaded_obj
    else:
        model = tf.keras.models.load_model(mpath)
        model_package = {
            'model': model,
            'preprocessing': None,
            'scaler': None,
            'feature_names': None
        }
    with cache_lock:
        MODEL_CACHE[model_key] = (model_package, mtype)
    return model_package, mtype

def _prepare_features(rows):
    """Prepare feature matrix from input rows."""
    if not GLOBAL_FEATURE_ORDER:
        raise ValueError('Feature names not available; retrain or provide feature_names.json')
    arr = []
    for r in rows:
        if len(r) != len(GLOBAL_FEATURE_ORDER):
            raise ValueError(f'Expected {len(GLOBAL_FEATURE_ORDER)} features, got {len(r)}')
        try:
            arr.append([float(v) for v in r])
        except Exception as e:
            raise ValueError(f'Non-numeric value encountered: {e}')
    return pd.DataFrame(arr, columns=GLOBAL_FEATURE_ORDER)

# Load canonical feature order if present
FEATURE_NAMES_PATH = _resolve_path('../models/feature_names.json')
GLOBAL_FEATURE_ORDER = []
try:
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, 'r', encoding='utf-8') as f:
            GLOBAL_FEATURE_ORDER = json.load(f)
except Exception as e:
    print(f"[WARN] Could not load feature names: {e}")


# ---------------------------------------------------------------------------
# Dataset & model metadata
# ---------------------------------------------------------------------------

def _load_label_map(json_filename: str):
    """Utility to load a label mapping JSON from the models folder if it exists.
    Returns dict[int,str] or empty dict.
    """
    try:
        path = _resolve_path(f"../models/{json_filename}")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # allow either {"0": "Normal"} or {0: "Normal"}
            cleaned = {}
            for k, v in data.items():
                try:
                    cleaned[int(k)] = v
                except Exception:
                    continue
            return cleaned
    except Exception as e:
        print(f"[WARN] Failed to load label map {json_filename}: {e}")
    return {}

# Base UNSW mapping (fallback) – attempt to override with file if present
_unsw_fallback_map = {
    0: "Normal", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
    6: "Fuzzers", 7: "Generic", 8: "Reconnaissance", 9: "Worms"
}
_unsw_loaded_map = _load_label_map('unsw_label_mapping.json') or _unsw_fallback_map

# Attempt to load CICIDS label map if present
_cicids_loaded_map = {}
try:
    cicids_raw_path = _resolve_path('../models/cicids_label_mapping.json')
    if os.path.exists(cicids_raw_path):
        with open(cicids_raw_path, 'r', encoding='utf-8') as f:
            cic_json = json.load(f)
        # Accept structures with index_to_label / label_to_index
        if 'index_to_label' in cic_json:
            for k, v in cic_json['index_to_label'].items():
                try:
                    _cicids_loaded_map[int(k)] = v.replace('\ufffd', '-') if isinstance(v, str) else v
                except Exception:
                    continue
        elif 'classes' in cic_json and isinstance(cic_json['classes'], list):
            for idx, name in enumerate(cic_json['classes']):
                if isinstance(name, str):
                    _cicids_loaded_map[idx] = name.replace('\ufffd', '-')
except Exception as e:
    print(f"[WARN] Failed enhanced parse cicids label map: {e}")

DATASETS = {
    "unsw": {
        "model_path": "../models/saved_model_rf.joblib",  # Use the actual available model
        "feature_path": "../data/processed/unsw_selected_features.csv",
        "label_map": _unsw_loaded_map
    }
}

# Conditionally add CICIDS dataset if model & feature file exist
_cicids_model_path = _resolve_path('../models/saved_model_cicids.joblib')
_cicids_feat_path = _resolve_path('../data/processed/cicids_selected_features.csv')
if os.path.exists(_cicids_model_path) and os.path.exists(_cicids_feat_path):
    DATASETS['cicids'] = {
        "model_path": "../models/saved_model_cicids.joblib",
        "feature_path": "../data/processed/cicids_selected_features.csv",
        "label_map": _cicids_loaded_map or {0: "Normal"}  # fallback minimal map
    }

# Initial scan of models directory for multi-model selector usage
_scan_models()

# Auto-load UNSW dataset by default on startup
DEFAULT_DATASET = 'unsw'
DEFAULT_MODEL = None
DEFAULT_FEATURES = []

def _initialize_default_dataset():
    """Initialize default dataset and features on startup"""
    global DEFAULT_MODEL, DEFAULT_FEATURES, GLOBAL_FEATURE_ORDER
    
    try:
        model_tuple, features, label_map, err = load_model_and_features(DEFAULT_DATASET)
        if not err:
            DEFAULT_MODEL = model_tuple
            DEFAULT_FEATURES = features
            GLOBAL_FEATURE_ORDER = features
            print(f"[INFO] Default dataset '{DEFAULT_DATASET}' loaded successfully with {len(features)} features")
        else:
            print(f"[WARN] Failed to load default dataset: {err}")
    except Exception as e:
        print(f"[WARN] Error initializing default dataset: {e}")

def load_model_and_features(dataset_key):
    if dataset_key not in DATASETS:
        return None, None, None, f"Dataset '{dataset_key}' not supported."
    
    entry = DATASETS[dataset_key]
    
    try:
        model_path = _resolve_path(entry['model_path'])
        feature_path = _resolve_path(entry['feature_path'])
        label_map = entry['label_map']
        
        if not os.path.exists(model_path):
            return None, None, None, f"Model file not found: {model_path}"
        if not os.path.exists(feature_path):
            return None, None, None, f"Feature file not found: {feature_path}"
        
        # Only support joblib models now
        cache_key = dataset_key
        with cache_lock:
            if cache_key in MODEL_CACHE:
                model_package, model_type = MODEL_CACHE[cache_key]
            else:
                if model_path.endswith('.joblib') or model_path.endswith('.pkl'):
                    try:
                        loaded_obj = joblib.load(model_path)
                        # Handle both old format (direct model) and new format (model package)
                        if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
                            # New format: model package with preprocessing
                            model_package = loaded_obj
                        else:
                            # Old format: direct model
                            model_package = {
                                'model': loaded_obj,
                                'preprocessing': None,
                                'scaler': None,
                                'feature_names': None
                            }
                        model_type = 'sklearn'
                    except Exception as e:
                        return None, None, None, f"Failed to load model from {model_path}: {str(e)}"
                else:
                    return None, None, None, f"Unsupported model format. Only .joblib files are supported."
                MODEL_CACHE[cache_key] = (model_package, model_type)
        
        # Load feature order from the model training (not from dataset)
        feature_names_path = _resolve_path("../models/feature_names.json")
        feature_order = []
        
        if os.path.exists(feature_names_path):
            try:
                with open(feature_names_path, 'r') as f:
                    feature_order = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load feature names from {feature_names_path}: {e}")
        
        # Fallback: try to get from dataset (old behavior)
        if not feature_order:
            try:
                feature_df = pd.read_csv(feature_path, nrows=1)
                feature_order = [col for col in feature_df.columns if col != 'label']
            except Exception as e:
                return None, None, None, f"Failed to read features from {feature_path}: {str(e)}"
        
        return (model_package, model_type), feature_order, label_map, None
        
    except Exception as e:
        return None, None, None, f"Unexpected error loading model and features: {str(e)}"

@app.route('/')
def home():
    # Get quick performance summary
    try:
        metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        
        # Find best performing model
        best_accuracy = 0
        best_model = None
        total_models = 0
        
        for model_key, model_data in metrics.items():
            if 'metrics' in model_data and isinstance(model_data['metrics'], dict):
                accuracy = model_data['metrics'].get('accuracy')
                if accuracy and accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_key
                    total_models += 1
        
        # Convert to percentage
        best_accuracy_pct = round(best_accuracy * 100, 2) if best_accuracy > 0 else None
        
        performance_summary = {
            'best_model': best_model,
            'best_accuracy': f"{best_accuracy_pct}%" if best_accuracy > 0 else None,
            'total_trained_models': total_models
        }
    except Exception:
        performance_summary = {
            'best_model': None,
            'best_accuracy': None,
            'total_trained_models': 0
        }
    
    return jsonify({
        "status": "AI NIDS API is running",
        "default_dataset": DEFAULT_DATASET,
        "supported_datasets": list(DATASETS.keys()),
        "models_available": len(MODEL_REGISTRY),
        "performance": performance_summary,
        "endpoints": {
            "model_performance": "/model_performance",
            "models": "/models",
            "metrics": "/metrics",
            "predict": "/predict",
            "dashboard": "/dashboard/"
        },
        "dashboard_url": "http://localhost:5000/dashboard/"
    })

@app.route('/initialize', methods=['POST'])
def initialize_models():
    """Initialize and load all available models"""
    try:
        _scan_models()
        _initialize_default_dataset()
        
        # Load metrics
        metrics = {}
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
            except Exception:
                pass
        
        # Build model info
        models_info = []
        for key, meta in MODEL_REGISTRY.items():
            try:
                model_package, mtype = _load_model(key)
                
                # Get model info
                if isinstance(model_package, dict) and 'model' in model_package:
                    model = model_package['model']
                    algorithm = type(model).__name__
                    preprocessing = model_package.get('preprocessing', 'none')
                else:
                    model = model_package
                    algorithm = type(model).__name__
                    preprocessing = 'none'
                
                # Get metrics
                model_metrics = metrics.get(key, {}).get('metrics', {})
                
                models_info.append({
                    'model_key': key,
                    'algorithm': algorithm,
                    'preprocessing': preprocessing,
                    'type': mtype,
                    'metrics': model_metrics,
                    'path': os.path.basename(meta['path'])
                })
                
            except Exception as e:
                print(f"[WARN] Failed to load model {key}: {e}")
                continue
        
        return jsonify({
            'status': 'success',
            'message': f'Initialized {len(models_info)} models with {DEFAULT_DATASET} dataset',
            'default_dataset': DEFAULT_DATASET,
            'feature_count': len(GLOBAL_FEATURE_ORDER),
            'features': GLOBAL_FEATURE_ORDER,
            'models': models_info
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Initialization failed: {str(e)}'
        }), 500

@app.route('/datasets', methods=['GET'])
def list_datasets():
    # Returns dataset keys and human label maps
    return jsonify({
        "datasets": [
            {"key": k, "labels": v.get("label_map", {}), "model_path": v.get("model_path")}
            for k, v in DATASETS.items()
        ]
    })

@app.route('/features/<dataset_key>', methods=['GET'])
def get_features(dataset_key):
    entry = DATASETS.get(dataset_key)
    if not entry:
        return jsonify({"error": f"Dataset '{dataset_key}' not supported."}), 404
    
    # Get features from model training (not dataset file)
    feature_names_path = _resolve_path("../models/feature_names.json")
    if os.path.exists(feature_names_path):
        try:
            with open(feature_names_path, 'r') as f:
                features = json.load(f)
            return jsonify({"dataset": dataset_key, "features": features})
        except Exception as e:
            return jsonify({"error": f"Failed to read model features: {str(e)}"}), 500
    else:
        # Fallback to dataset file
        feature_path = _resolve_path(entry["feature_path"])
        if not os.path.exists(feature_path):
            return jsonify({"error": f"Feature file not found: {feature_path}"}), 404
        try:
            df = pd.read_csv(feature_path, nrows=1)
            features = [c for c in df.columns if c != 'label']
            return jsonify({"dataset": dataset_key, "features": features})
        except Exception as e:
            return jsonify({"error": f"Failed to read features: {str(e)}"}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Tries to load metrics from models/metrics.json; if missing, returns empty
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.normpath(os.path.join(base_dir, '..', 'models', 'metrics.json'))
    try:
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({})
    except Exception as e:
        return jsonify({"error": f"Failed to read metrics: {str(e)}"}), 500

@app.route('/model_meta', methods=['GET'])
def model_meta():
    """Return metadata per dataset: labels, model path, feature count, metrics if present."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.normpath(os.path.join(base_dir, '..', 'models', 'metrics.json'))
    try:
        metrics_data = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
        out = []
        for key, entry in DATASETS.items():
            feat_path = _resolve_path(entry['feature_path'])
            model_path = _resolve_path(entry['model_path'])
            feat_count = None
            model_exists = os.path.exists(model_path)
            
            if os.path.exists(feat_path):
                try:
                    sample_df = pd.read_csv(feat_path, nrows=1)
                    feat_count = len([c for c in sample_df.columns if c != 'label'])
                except Exception:
                    feat_count = None
                    
            # Try multiple keys for metrics lookup
            model_rel = entry['model_path']
            model_name = os.path.splitext(os.path.basename(model_rel))[0]
            metric_entry = metrics_data.get(model_name, metrics_data.get(key, {}))
            
            # Calculate model size if exists
            model_size = None
            if model_exists:
                try:
                    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                except:
                    pass
                    
            out.append({
                'dataset': key,
                'model_path': model_rel,
                'model_exists': model_exists,
                'model_size_mb': round(model_size, 2) if model_size else None,
                'feature_count': feat_count,
                'labels': entry.get('label_map', {}),
                'metrics': metric_entry.get('metrics', {}),
                'samples': metric_entry.get('samples'),
                'online_accuracy': metric_entry.get('metrics', {}).get('online_accuracy') if metric_entry.get('metrics') else None,
                'model_type': 'sklearn',
                'confusion_matrix': metric_entry.get('confusion_matrix', []),
                'class_weights': metric_entry.get('class_weights', {})
            })
        return jsonify({'datasets': out})
    except Exception as e:
        return jsonify({'error': f'Model metadata retrieval failed: {e}'}), 500


@app.route('/model_performance', methods=['GET'])
def model_performance():
    """Return a comprehensive overview of all model performance metrics, sorted by accuracy."""
    try:
        metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        
        performance_data = []
        
        for model_key, model_data in metrics.items():
            if 'metrics' in model_data and isinstance(model_data['metrics'], dict):
                raw_metrics = model_data['metrics']
                
                # Skip online-only metrics entries
                if 'online_accuracy' in raw_metrics and len(raw_metrics) == 1:
                    continue
                
                # Extract key metrics
                accuracy = raw_metrics.get('accuracy')
                f1_score = raw_metrics.get('f1_weighted') or raw_metrics.get('f1')
                precision = raw_metrics.get('precision_weighted')
                recall = raw_metrics.get('recall_weighted')
                
                if accuracy is not None:
                    # Values are already in decimal format (0.7980 for 79.80%)
                    accuracy_pct = round(accuracy * 100, 2)
                    f1_pct = round(f1_score * 100, 2) if f1_score else None
                    precision_pct = round(precision * 100, 2) if precision else None
                    recall_pct = round(recall * 100, 2) if recall else None
                    
                    # Determine model type and dataset
                    model_type = model_data.get('type', 'sklearn')
                    algorithm = model_data.get('algorithm', 'Unknown')
                    samples = model_data.get('samples', 0)
                    
                    # Guess dataset from model name
                    dataset = 'unknown'
                    if 'unsw' in model_key.lower():
                        dataset = 'UNSW-NB15'
                    elif 'cicids' in model_key.lower():
                        dataset = 'CICIDS2017'
                    
                    # Guess algorithm from key if not available
                    if algorithm == 'Unknown':
                        key_lower = model_key.lower()
                        if 'randomforest' in key_lower:
                            algorithm = 'Random Forest'
                        elif 'gradientboosting' in key_lower:
                            algorithm = 'Gradient Boosting'
                        elif 'extratrees' in key_lower:
                            algorithm = 'Extra Trees'
                        elif 'svm' in key_lower:
                            algorithm = 'Support Vector Machine'
                        elif 'logistic' in key_lower:
                            algorithm = 'Logistic Regression'
                        elif 'mlp' in key_lower or 'neural' in key_lower:
                            algorithm = 'Neural Network'
                        elif 'decisiontree' in key_lower:
                            algorithm = 'Decision Tree'
                        elif 'cnn' in key_lower or 'gan' in key_lower:
                            algorithm = 'CNN/GAN'
                        elif 'adversarial' in key_lower:
                            algorithm = 'Adversarial CNN'
                    
                    performance_data.append({
                        'model_key': model_key,
                        'algorithm': algorithm,
                        'dataset': dataset,
                        'model_type': model_type,
                        'accuracy': round(accuracy_pct, 2),
                        'f1_score': round(f1_pct, 2) if f1_pct else None,
                        'precision': round(precision_pct, 2) if precision_pct else None,
                        'recall': round(recall_pct, 2) if recall_pct else None,
                        'samples_trained': samples,
                        'performance_rank': accuracy_pct  # for sorting
                    })
        
        # Sort by accuracy (descending)
        performance_data.sort(key=lambda x: x['performance_rank'], reverse=True)
        
        # Add rankings
        for idx, model in enumerate(performance_data):
            model['rank'] = idx + 1
            del model['performance_rank']  # Remove temporary sorting key
        
        # Calculate summary statistics
        if performance_data:
            accuracies = [m['accuracy'] for m in performance_data if m['accuracy']]
            f1_scores = [m['f1_score'] for m in performance_data if m['f1_score']]
            
            summary = {
                'total_models': len(performance_data),
                'best_accuracy': max(accuracies) if accuracies else None,
                'average_accuracy': round(sum(accuracies) / len(accuracies), 2) if accuracies else None,
                'best_f1_score': max(f1_scores) if f1_scores else None,
                'average_f1_score': round(sum(f1_scores) / len(f1_scores), 2) if f1_scores else None,
                'best_model': performance_data[0]['model_key'] if performance_data else None
            }
        else:
            summary = {
                'total_models': 0,
                'best_accuracy': None,
                'average_accuracy': None,
                'best_f1_score': None,
                'average_f1_score': None,
                'best_model': None
            }
        
        return jsonify({
            'summary': summary,
            'models': performance_data,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve model performance: {str(e)}'}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """Return list of all model artifacts in models directory for UI model selector.
    Response schema:
      {
        "models": [
            {"model_key": str, "type": "sklearn"|"keras", "size_mb": float|None,
             "path": str, "dataset_guess": str|None, "metrics": {...}}
        ]
      }
    """
    try:
        # Re-scan on every request to pick up new models without restart (cheap: directory listing only)
        _scan_models()
        metrics = {}
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
            except Exception:
                metrics = {}
        items = []
        for key, meta in MODEL_REGISTRY.items():
            path = meta['path']
            mtype = meta['type']
            size_mb = None
            try:
                if os.path.exists(path):
                    size_mb = round(os.path.getsize(path) / (1024*1024), 2)
            except Exception:
                pass
            # Heuristic dataset guess
            lower = key.lower()
            if 'cicids' in lower:
                dataset_guess = 'cicids'
            elif 'unsw' in lower:
                dataset_guess = 'unsw'
            else:
                dataset_guess = None
            # metrics entry can be by key (model base name) or dataset
            metric_entry = metrics.get(key, metrics.get(dataset_guess or '', {}))
            
            # Special handling for saved_model_rf.joblib -> map to 'saved_model'
            if key == 'saved_model_rf' and not metric_entry.get('metrics'):
                metric_entry = metrics.get('saved_model', {})
            
            raw_metrics = metric_entry.get('metrics', {}) if isinstance(metric_entry, dict) else {}
            # Normalize metric keys & provide fallbacks
            norm_metrics = {
                'accuracy': raw_metrics.get('accuracy') or raw_metrics.get('acc'),
                'precision_weighted': raw_metrics.get('precision_weighted') or raw_metrics.get('precision'),
                'recall_weighted': raw_metrics.get('recall_weighted') or raw_metrics.get('recall'),
                'f1_weighted': raw_metrics.get('f1_weighted') or raw_metrics.get('f1') or raw_metrics.get('f1_score'),
                'online_accuracy': raw_metrics.get('online_accuracy')
            }
            # Round metrics for better display (values are already in correct format)
            for metric_key in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
                if norm_metrics[metric_key] is not None:
                    norm_metrics[metric_key] = round(norm_metrics[metric_key], 4)
            samples_used = metric_entry.get('samples') if isinstance(metric_entry, dict) else None
            # Attempt secondary lookup for samples via alternate key structure
            if samples_used is None and isinstance(metric_entry, dict):
                samples_used = metric_entry.get('running_counts', {}).get('n')
            # Algorithm guess from key if not stored
            algorithm_guess = metric_entry.get('algorithm') if isinstance(metric_entry, dict) else None
            if not algorithm_guess:
                if 'rf' in lower or 'randomforest' in lower:
                    algorithm_guess = 'Random Forest'
                elif 'gan' in lower and 'cnn' in lower:
                    algorithm_guess = 'CNN-GAN'
                elif 'extratrees' in lower:
                    algorithm_guess = 'Extra Trees'
                elif 'gradientboosting' in lower or 'gb' in lower:
                    algorithm_guess = 'Gradient Boosting'
                elif 'svm' in lower:
                    algorithm_guess = 'SVM'
                elif 'logistic' in lower:
                    algorithm_guess = 'Logistic Regression'
                elif 'mlp' in lower or 'neural' in lower:
                    algorithm_guess = 'Neural Network'
                elif 'decisiontree' in lower or 'tree' in lower:
                    algorithm_guess = 'Decision Tree'
                elif 'cnn' in lower or 'adversarial' in lower:
                    algorithm_guess = 'CNN'
            balanced = bool('balanced' in lower)
            items.append({
                'model_key': key,
                'type': mtype,
                'size_mb': size_mb,
                'path': os.path.basename(path),
                'dataset_guess': dataset_guess,
                'metrics': norm_metrics,
                'samples': samples_used,
                'algorithm_guess': algorithm_guess,
                'balanced': balanced
            })
        # Sort deterministic: dataset first alphabetical, then key
        items.sort(key=lambda x: (x['dataset_guess'] or 'zzz', x['model_key']))
        return jsonify({'models': items, 'feature_count': len(GLOBAL_FEATURE_ORDER)})
    except Exception as e:
        return jsonify({'error': f'Model listing failed: {e}'}), 500

# Serve the dashboard statically so it can be launched at http://localhost:5000/dashboard/
@app.route('/dashboard')
def dashboard_redirect():
    return redirect('/dashboard/', code=302)

@app.route('/dashboard/')
def dashboard_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dash_dir = os.path.normpath(os.path.join(base_dir, '..', 'dashboard'))
    return send_from_directory(dash_dir, 'index.html')

@app.route('/dashboard/<path:filename>')
def dashboard_static(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dash_dir = os.path.normpath(os.path.join(base_dir, '..', 'dashboard'))
    return send_from_directory(dash_dir, filename)

def _update_metrics_placeholder(dataset: str, correct: bool):
    """Very lightweight running accuracy tracker; extend with full metrics offline."""
    try:
        metrics_abs = _resolve_path(METRICS_PATH)
        os.makedirs(os.path.dirname(metrics_abs), exist_ok=True)
        data = {}
        
        if os.path.exists(metrics_abs):
            try:
                with open(metrics_abs, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"[WARN] Failed to read metrics file, creating new: {e}")
                data = {}
                
        entry = data.get(dataset, {
            "model": DATASETS.get(dataset, {}).get('model_path', ''), 
            "metrics": {}, 
            "running_counts": {"n": 0, "correct": 0}
        })
        
        rc = entry.setdefault("running_counts", {"n": 0, "correct": 0})
        rc['n'] += 1
        if correct:
            rc['correct'] += 1
            
        # derive simple accuracy
        if rc['n'] > 0:
            entry.setdefault('metrics', {})['online_accuracy'] = round(rc['correct'] / rc['n'], 4)
            entry['last_update'] = datetime.utcnow().isoformat()
            
        data[dataset] = entry
        
        with open(metrics_abs, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"[WARN] Failed to write metrics: {e}")


def _apply_preprocessing(df, model_package):
    """Apply preprocessing to features if needed."""
    if not isinstance(model_package, dict):
        return df
    
    preprocessing = model_package.get('preprocessing')
    scaler = model_package.get('scaler')
    
    if preprocessing and scaler:
        try:
            df_processed = pd.DataFrame(
                scaler.transform(df),
                columns=df.columns,
                index=df.index
            )
            return df_processed
        except Exception as e:
            print(f"[WARN] Preprocessing failed: {e}")
            return df
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_json = request.json
        if not req_json:
            return jsonify({'error': 'Request body must be valid JSON.'}), 400
            
        # Use default dataset if not specified
        dataset = req_json.get('dataset', DEFAULT_DATASET)
        data = req_json.get('features')

        if not data or not isinstance(data, list):
            return jsonify({'error': 'Request must include "features" (list).'}), 400

        # Use default model and features if available
        if dataset == DEFAULT_DATASET and DEFAULT_MODEL and DEFAULT_FEATURES:
            model_package, model_type = DEFAULT_MODEL
            feature_order = DEFAULT_FEATURES
            label_map = DATASETS[dataset].get('label_map', {})
        else:
            # Fallback to loading model
            if dataset not in DATASETS:
                return jsonify({'error': f'Dataset "{dataset}" not supported. Available: {list(DATASETS.keys())}'}), 400

            m_tuple, feature_order, label_map, err = load_model_and_features(dataset)
            if err:
                return jsonify({'error': err}), 400
            model_package, model_type = m_tuple
        
        # Extract actual model from package
        if isinstance(model_package, dict) and 'model' in model_package:
            model = model_package['model']
        else:
            model = model_package

        # Validate feature count
        if len(data) != len(feature_order):
            return jsonify({'error': f'Expected {len(feature_order)} features for dataset "{dataset}", received {len(data)}.'}), 400

        # Validate feature values are numeric
        try:
            numeric_data = [float(x) for x in data]
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'All feature values must be numeric. Error: {str(e)}'}), 400

        print(f"[DEBUG] Dataset: {dataset}, Features length: {len(data)}, Model type: {model_type}")
        
        # Convert to DataFrame for sklearn model
        df = pd.DataFrame([numeric_data], columns=feature_order)
        
        # Apply preprocessing if needed
        df_processed = _apply_preprocessing(df, model_package)
        
        prediction = model.predict(df_processed)[0]
        
        # Get probabilities and confidence
        probabilities = None
        confidence = None
        
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(df_processed)[0]
                probabilities = {str(i): float(p) for i, p in enumerate(probs)}
                confidence = float(np.max(probs))
                print(f"[DEBUG] Sklearn probs snippet: {probs[:5]} ...")
            except Exception as e:
                print(f"[WARN] Failed to get probabilities: {e}")
        elif hasattr(model, 'decision_function'):
            try:
                # For SVM and similar models
                decision = model.decision_function(df_processed)[0]
                if len(decision.shape) > 0 and decision.shape[0] > 1:
                    # Multi-class
                    probabilities = {str(i): float(d) for i, d in enumerate(decision)}
                    confidence = float(np.max(decision))
                else:
                    # Binary classification
                    confidence = float(abs(decision))
            except Exception as e:
                print(f"[WARN] Failed to get decision function: {e}")
        else:
            confidence = 0.5  # Default confidence for models without probability
            
        print(f"[DEBUG] Prediction raw label: {prediction}")
        threat_name = label_map.get(int(prediction), 'Unknown')
        is_attack = bool(prediction != 0)  # Assuming 0 is normal traffic - convert to Python bool
        
        # Update metrics
        _update_metrics_placeholder(dataset, correct=True)
        
        return jsonify({
            'prediction': int(prediction),
            'threat': threat_name,
            'confidence': round(confidence, 4) if confidence else None,
            'probabilities': probabilities,
            'is_attack': is_attack,
            'feature_order': feature_order,
            'model_type': model_type,
            'dataset': dataset
        })
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/preset/sample', methods=['POST'])
def preset_sample():
    """Return a real feature vector sampled from the processed dataset.
    Body JSON: {"dataset": "unsw", "label": <optional int>, "random": <optional bool>}
    If label provided, sample a row with that label. Falls back gracefully if none found.
    """
    payload = request.json or {}
    dataset = payload.get('dataset', DEFAULT_DATASET)  # Default to UNSW
    # Accept common typo alias
    if dataset == 'cicds':
        dataset = 'cicids'
    label_request = payload.get('label')  # numeric index
    label_name = payload.get('label_name')  # optional human-readable string for cicids
    random_flag = payload.get('random')
    
    # Use default features if available for UNSW
    if dataset == DEFAULT_DATASET and DEFAULT_FEATURES:
        feature_order = DEFAULT_FEATURES
        label_map = DATASETS[dataset].get('label_map', {})
    else:
        # Load model to get canonical feature order
        m_tuple, feature_order, label_map, err = load_model_and_features(dataset)
        if err:
            return jsonify({'error': err}), 400
    
    entry = DATASETS.get(dataset)
    if not entry:
        return jsonify({'error': f'Dataset {dataset} not supported'}), 400
    feature_path = _resolve_path(entry['feature_path'])
    if not os.path.exists(feature_path):
        return jsonify({'error': f'Feature file not found for dataset {dataset}: {feature_path}'}), 400
    # Load / cache a lightweight DataFrame (only needed columns)
    try:
        with PRESET_CACHE_LOCK:
            cached = PRESET_DATA_CACHE.get(dataset)
        if cached is None:
            # Read full CSV first to avoid usecols mismatch and then align
            df = pd.read_csv(feature_path)
            # Normalize column names (strip whitespace)
            df.columns = [c.strip() for c in df.columns]
            # Ensure label column exists; if not, create with 0
            if 'label' not in df.columns:
                df['label'] = 0
            # Add any missing feature columns with zeros (float)
            missing = [c for c in feature_order if c not in df.columns]
            if missing:
                for m in missing:
                    df[m] = 0.0
            # Reorder columns to feature_order then label
            ordered_cols = [c for c in feature_order] + ['label']
            df = df[ordered_cols]
            if missing:
                print(f"[INFO] Added {len(missing)} missing feature columns for dataset {dataset}: {missing[:8]}{'...' if len(missing)>8 else ''}")
            # Optional sample cap to reduce memory (env PRESET_SAMPLE_LIMIT)
            try:
                cap = int(os.environ.get('PRESET_SAMPLE_LIMIT', '50000'))
            except ValueError:
                cap = 50000
            if len(df) > cap:
                df = df.sample(cap, random_state=42).reset_index(drop=True)
            with PRESET_CACHE_LOCK:
                PRESET_DATA_CACHE[dataset] = df
            cached = df
        df = cached
        # Filter by label if requested
        subset = df
        # If label_name provided (e.g., "DoS Hulk") map to label index using dataset label_map
        if label_name and dataset in DATASETS:
            lm = DATASETS[dataset].get('label_map', {})
            # Build reverse map
            rev = {v.lower(): k for k, v in lm.items()}
            idx = rev.get(label_name.lower())
            if idx is not None:
                label_request = idx
        if label_request is not None:
            try:
                subset = df[df['label'] == int(label_request)]
            except Exception:
                subset = df
            if subset.empty:
                subset = df  # fallback
        # Random sample row
        row = subset.sample(1).iloc[0]
        features_values = [float(row[f]) for f in feature_order]
        row_label = int(row['label']) if 'label' in row else None
        return jsonify({
            'dataset': dataset,
            'label': row_label,
            'threat': label_map.get(row_label, 'Unknown') if row_label is not None else None,
            'features': features_values,
            'feature_order': feature_order,
            'from_label_request': label_request is not None or bool(label_name),
            'matched_label_request': (label_request is not None and row_label == int(label_request)),
            'requested_label_name': label_name
        })
    except Exception as e:
        return jsonify({'error': f'Failed to sample preset: {e}'}), 500

@app.route('/status', methods=['GET'])
def system_status():
    """Return system status including loaded models, cache info, and basic stats"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check model files
        models_available = 0
        total_models = len(DATASETS)
        
        for key, entry in DATASETS.items():
            model_path = _resolve_path(entry['model_path'])
            if os.path.exists(model_path):
                models_available += 1
        
        # Cache info
        with cache_lock:
            models_cached = len(MODEL_CACHE)
            explainers_cached = len(EXPLAINER_CACHE)
        
        # Check feedback file
        feedback_path = _resolve_path(FEEDBACK_FILE)
        feedback_rows = 0
        if os.path.exists(feedback_path):
            try:
                import pandas as pd
                fb_df = pd.read_csv(feedback_path, header=None)
                feedback_rows = len(fb_df)
            except:
                pass
        
        return jsonify({
            'status': 'online',
            'models': {
                'available': models_available,
                'total': total_models,
                'cached': models_cached
            },
            'explainers_cached': explainers_cached,
            'feedback_samples': feedback_rows,
            'supported_datasets': list(DATASETS.keys()),
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': f'Status check failed: {e}'}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content for favicon

@app.route('/feedback', methods=['POST'])
def receive_feedback():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Request body must be valid JSON.'}), 400
            
        features = data.get('features')
        predicted = data.get('predicted_label')
        feedback_label = data.get('feedback_label')  # E.g., 'correct' or 'false_positive'

        if not (features and predicted is not None and feedback_label):
            return jsonify({'error': 'Missing data: features, predicted_label, and feedback_label are required.'}), 400

        # Validate feedback_label
        valid_feedback = ['correct', 'false_positive', 'false_negative']
        if feedback_label not in valid_feedback:
            return jsonify({'error': f'feedback_label must be one of: {valid_feedback}'}), 400

        # Validate features is a list
        if not isinstance(features, list):
            return jsonify({'error': 'features must be a list of numeric values.'}), 400

        # Ensure feedback file directory exists
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            timestamp = datetime.utcnow().isoformat()
            writer.writerow(features + [predicted, feedback_label, timestamp])
            
        # Adjust running accuracy metric if feedback indicates error
        # We currently don't know dataset here (could be added by client). For now treat false_* as incorrect.
        correct = feedback_label == 'correct'
        _update_metrics_placeholder('online', correct)
        
        return jsonify({'message': 'Feedback recorded successfully'}), 200
        
    except Exception as e:
        print(f"[ERROR] Feedback recording failed: {e}")
        return jsonify({'error': f'Failed to record feedback: {str(e)}'}), 500


@app.route('/explain/shap', methods=['POST'])
def explain_shap():
    req_json = request.json or {}
    dataset = req_json.get('dataset')
    data = req_json.get('features')  # single instance
    if not dataset or data is None:
        return jsonify({'error': 'dataset and features required'}), 400
    m_tuple, feature_order, label_map, err = load_model_and_features(dataset)
    if err:
        return jsonify({'error': err}), 400
    model, model_type = m_tuple
    
    if model_type != 'sklearn':
        return jsonify({'error': 'SHAP explanation only supported for sklearn models'}), 400
        
    df = pd.DataFrame([data], columns=feature_order)
    # Build / reuse SHAP explainer
    cache_key = f"shap::{dataset}"
    with cache_lock:
        explainer = EXPLAINER_CACHE.get(cache_key)
    try:
        if explainer is None:
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                if hasattr(model, 'predict_proba'):
                    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(pd.DataFrame([data], columns=feature_order), 1))
                else:
                    explainer = shap.KernelExplainer(model.predict, shap.sample(pd.DataFrame([data], columns=feature_order), 1))
            with cache_lock:
                EXPLAINER_CACHE[cache_key] = explainer
                
        shap_vals = explainer.shap_values(df)
        if isinstance(shap_vals, list):
            # Multi-class - aggregate absolute mean across classes
            agg = np.mean(np.abs(np.array(shap_vals)), axis=0)[0]
        else:
            agg = np.abs(shap_vals[0])
            
        contrib = sorted([{ 'feature': f, 'importance': float(v)} for f, v in zip(feature_order, agg)], key=lambda x: -x['importance'])[:20]
        return jsonify({'shap_values': contrib})
    except Exception as e:
        return jsonify({'error': f'SHAP explanation failed: {e}'}), 500


@app.route('/explain/lime', methods=['POST'])
def explain_lime():
    if LimeTabularExplainer is None:
        return jsonify({'error': 'LIME not installed'}), 500
    req_json = request.json or {}
    dataset = req_json.get('dataset')
    data = req_json.get('features')
    if not dataset or data is None:
        return jsonify({'error': 'dataset and features required'}), 400
    m_tuple, feature_order, label_map, err = load_model_and_features(dataset)
    if err:
        return jsonify({'error': err}), 400
    model, model_type = m_tuple
    
    if model_type != 'sklearn':
        return jsonify({'error': 'LIME explanation only supported for sklearn models'}), 400
        
    # For efficiency use a small background sample from feature file
    feature_path = _resolve_path(DATASETS[dataset]['feature_path'])
    full_df = pd.read_csv(feature_path).drop(columns=['label'])
    sample_bg = full_df.sample(min(500, len(full_df)), random_state=42)
    explainer = LimeTabularExplainer(sample_bg.values, feature_names=feature_order, class_names=list(label_map.values()), discretize_continuous=True, random_state=42)
    instance = np.array(data)
    
    if hasattr(model, 'predict_proba'):
        predict_fn = model.predict_proba
    else:
        predict_fn = lambda x: np.eye(len(label_map))[model.predict(x)]
        
    exp = explainer.explain_instance(instance, predict_fn, num_features=15)
    return jsonify({'lime_explanation': [{'feature': f, 'weight': float(w)} for f, w in exp.as_list()]})


@app.route('/incremental/update', methods=['POST'])
def incremental_update():
    """Trigger incremental model update from feedback file.
    Optionally include JSON: {"dataset": "unsw"}
    """
    payload = request.json or {}
    dataset = payload.get('dataset', 'unsw')
    if dataset not in DATASETS:
        return jsonify({'error': f'Dataset {dataset} not supported.'}), 400
    model_path = _resolve_path(DATASETS[dataset]['model_path'])
    feature_path = _resolve_path(DATASETS[dataset]['feature_path'])
    feedback_path = _resolve_path(FEEDBACK_FILE)
    if not os.path.exists(feedback_path):
        return jsonify({'error': 'No feedback file present.'}), 400
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    try:
        fb = pd.read_csv(feedback_path, header=None)
        if fb.empty:
            return jsonify({'error': 'Feedback file empty.'}), 400
        # expect last two cols predicted_label, feedback_label; only reinforce correct
        correct_rows = fb[fb.iloc[:, -1] == 'correct']
        if correct_rows.empty:
            return jsonify({'error': 'No correct feedback rows.'}), 400
        X_new = correct_rows.iloc[:, :-2].values.astype(np.float32)
        y_new = correct_rows.iloc[:, -2].values.astype(int)
        if os.path.exists(model_path):
            inc_model = joblib.load(model_path)
        else:
            # initialize new model with all classes from base dataset
            base_df = pd.read_csv(feature_path)
            classes = np.unique(base_df['label']) if 'label' in base_df.columns else np.unique(y_new)
            inc_model = SGDClassifier(loss='log_loss', learning_rate='optimal', random_state=42)
            inc_model.partial_fit(X_new, y_new, classes=classes)
        
        # Check if model supports incremental learning
        if hasattr(inc_model, 'partial_fit'):
            inc_model.partial_fit(X_new, y_new)
        else:
            return jsonify({'error': 'Model does not support incremental learning'}), 400
            
        joblib.dump(inc_model, model_path)
        with cache_lock:
            MODEL_CACHE.pop(dataset, None)  # force reload next predict
        _update_metrics_placeholder(dataset, correct=True)
        return jsonify({'message': 'Incremental model updated', 'samples_used': int(len(X_new))})
    except Exception as e:
        return jsonify({'error': f'Incremental update failed: {e}'}), 500

# Replace existing /predict implementation with multi-model aware version
@app.route('/predict_multi', methods=['POST'])
def predict_multi():
    payload = request.json or {}
    model_key = payload.get('model')  # e.g., 'saved_model_unsw', 'saved_model_adversarial_unsw'
    rows = payload.get('instances')  # list of feature lists
    if not model_key or not rows:
        return jsonify({'error': 'model and instances are required'}), 400
    try:
        model_package, mtype = _load_model(model_key)
        
        # Extract actual model from package
        if isinstance(model_package, dict) and 'model' in model_package:
            model = model_package['model']
        else:
            model = model_package
            
        df = _prepare_features(rows)
        
        # Apply preprocessing if needed
        df_processed = _apply_preprocessing(df, model_package)
        
        preds = None
        probs_out = None
        if mtype == 'sklearn':
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(df_processed)
                preds = probs.argmax(axis=1)
                probs_out = [{str(i): float(p) for i,p in enumerate(row)} for row in probs]
            else:
                preds = model.predict(df_processed)
        else:  # keras
            x = df_processed.values.astype('float32')
            x = x.reshape((x.shape[0], x.shape[1], 1))
            probs = model.predict(x, verbose=0)
            preds = probs.argmax(axis=1)
            probs_out = [{str(i): float(p) for i,p in enumerate(row)} for row in probs]
        # Map label names if available from metrics class_weights keys length vs label_map
        # For now just numeric classes
        return jsonify({
            'model': model_key,
            'count': len(rows),
            'predictions': [int(p) for p in preds],
            'probabilities': probs_out,
            'feature_order': GLOBAL_FEATURE_ORDER
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

if __name__ == '__main__':
    print("\nAI NIDS API starting...")
    print("Dashboard available at: http://localhost:5000/dashboard\n")
    
    # Initialize default dataset on startup
    print("Initializing default dataset and models...")
    _initialize_default_dataset()
    print(f"Loaded {len(MODEL_REGISTRY)} models from registry")
    
    app.run(debug=True)
