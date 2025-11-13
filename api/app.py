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

# Only import if needed for explainability
import shap
try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

app = Flask(__name__)
CORS(app)
# Prefer JSON responses for API endpoints on errors
def _wants_json_response() -> bool:
    try:
        api_prefixes = (
            '/explain', '/predict', '/models', '/metrics', '/preset',
            '/initialize', '/status', '/datasets', '/model_', '/predict_batch'
        )
        return request.accept_mimetypes.accept_json or request.path.startswith(api_prefixes)
    except Exception:
        return False

@app.errorhandler(404)
def handle_404(e):
    if _wants_json_response():
        return jsonify({'error': 'Not Found', 'message': 'Endpoint not found', 'path': request.path}), 404
    return e

@app.errorhandler(405)
def handle_405(e):
    if _wants_json_response():
        return jsonify({'error': 'Method Not Allowed', 'message': 'Use the correct HTTP method for this endpoint', 'path': request.path}), 405
    return e

@app.errorhandler(500)
def handle_500(e):
    if _wants_json_response():
        return jsonify({'error': 'Internal Server Error', 'message': 'The server encountered an error processing the request'}), 500
    return e


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

# Model registry structure: { key: { 'path': str, 'type': 'sklearn', 'features': list } }
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
    # Only sklearn models are supported
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
    # Try to capture feature names from the model for alignment
    try:
        feature_names = None
        if hasattr(model, 'feature_names_in_'):
            # sklearn-compatible estimators
            feature_names = [str(x) for x in model.feature_names_in_.tolist()]
        else:
            # XGBoost sklearn wrapper may expose booster feature names
            algoname = type(model).__name__
            if 'XGB' in algoname and hasattr(model, 'get_booster'):
                booster = model.get_booster()
                try:
                    fn = booster.feature_names
                    if fn:
                        feature_names = [str(x) for x in fn]
                except Exception:
                    pass
        if feature_names:
            model_package['feature_names'] = feature_names
    except Exception:
        pass
    with cache_lock:
        MODEL_CACHE[model_key] = (model_package, 'sklearn')
    return model_package, 'sklearn'

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
        
        # Prefer per-model feature names if packaged with the model
        feature_order = []
        try:
            if isinstance(model_package, dict):
                packaged = model_package.get('feature_names')
                if packaged and isinstance(packaged, (list, tuple)):
                    feature_order = list(packaged)
        except Exception:
            pass

        # Fallback: read from dataset file to avoid cross-model contamination
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
    
    # Prefer features tied to the configured model for this dataset
    try:
        model_path = _resolve_path(entry['model_path'])
        if os.path.exists(model_path):
            loaded_obj = joblib.load(model_path)
            if isinstance(loaded_obj, dict):
                model_features = loaded_obj.get('feature_names')
                if model_features:
                    return jsonify({"dataset": dataset_key, "features": model_features})
    except Exception:
        pass

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
                        # CNN/GAN models removed
                    
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
                        {"model_key": str, "type": "sklearn", "size_mb": float|None,
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
                # CNN/GAN models removed
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
                # CNN models removed
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
            
        # Support both dataset and model_key specification
        dataset = req_json.get('dataset', DEFAULT_DATASET)
        model_key = req_json.get('model_key')  # Optional specific model override
        data = req_json.get('features')

        if not data or not isinstance(data, list):
            return jsonify({'error': 'Request must include "features" (list).'}), 400

        # Choose model based on priority: model_key > dataset preference > default
        if model_key and model_key in MODEL_REGISTRY:
            # Use specific model requested
            model_package, model_type = _load_model(model_key)
            feature_order = GLOBAL_FEATURE_ORDER
            # Guess dataset from model key for label mapping
            if 'unsw' in model_key.lower():
                label_map = DATASETS.get('unsw', {}).get('label_map', {})
                dataset_used = 'unsw'
            elif 'cicids' in model_key.lower():
                label_map = DATASETS.get('cicids', {}).get('label_map', {})
                dataset_used = 'cicids'
            else:
                label_map = DATASETS.get(DEFAULT_DATASET, {}).get('label_map', {})
                dataset_used = DEFAULT_DATASET
            print(f"[DEBUG] Using specific model: {model_key} with {dataset_used} labels")
        elif dataset == DEFAULT_DATASET and DEFAULT_MODEL and DEFAULT_FEATURES:
            # Use default pre-loaded model
            model_package, model_type = DEFAULT_MODEL
            feature_order = DEFAULT_FEATURES
            label_map = DATASETS[dataset].get('label_map', {})
            dataset_used = dataset
            model_key = 'default'
        else:
            # Fallback to loading model by dataset
            if dataset not in DATASETS:
                return jsonify({'error': f'Dataset "{dataset}" not supported. Available: {list(DATASETS.keys())}'}), 400

            m_tuple, feature_order, label_map, err = load_model_and_features(dataset)
            if err:
                return jsonify({'error': err}), 400
            model_package, model_type = m_tuple
            dataset_used = dataset
            model_key = f'dataset_{dataset}'
        
        # Extract actual model from package
        if isinstance(model_package, dict) and 'model' in model_package:
            model = model_package['model']
        else:
            model = model_package

        # Validate feature values are numeric
        try:
            numeric_data = [float(x) for x in data]
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'All feature values must be numeric. Error: {str(e)}'}), 400

        # Determine model-specific feature order if available
        model_features = None
        if isinstance(model_package, dict):
            model_features = model_package.get('feature_names')
        # Fallback to dataset/global feature order (what client likely used)
        base_input_order = feature_order if feature_order else GLOBAL_FEATURE_ORDER

        df = None
        # Case 1: model exposes its own features
        if model_features:
            if len(numeric_data) == len(model_features):
                # Direct match, use as-is
                df = pd.DataFrame([numeric_data], columns=model_features)
            else:
                # Try to map from dataset CSV feature list (base 30) to model features (engineered)
                dataset_entry = DATASETS.get(dataset_used)
                base_csv_features = None
                if dataset_entry:
                    try:
                        feat_df = pd.read_csv(_resolve_path(dataset_entry['feature_path']), nrows=1)
                        base_csv_features = [c for c in feat_df.columns if c != 'label']
                    except Exception:
                        base_csv_features = None
                # Determine source names for incoming vector
                source_names = None
                if base_csv_features and len(base_csv_features) == len(numeric_data):
                    source_names = base_csv_features
                elif len(base_input_order) == len(numeric_data):
                    source_names = base_input_order
                # If we can infer names, build mapping
                if source_names:
                    values_map = {name: val for name, val in zip(source_names, numeric_data)}
                    aligned_row = [values_map.get(name, 0.0) for name in model_features]
                    df = pd.DataFrame([aligned_row], columns=model_features)
                    print(f"[DEBUG] Mapped {len(source_names)} -> {len(model_features)} (filled_missing={sum(1 for n in model_features if n not in values_map)})")
                else:
                    return jsonify({'error': f'Feature count mismatch for model: expected {len(model_features)}, received {len(numeric_data)}.'}), 400
        else:
            # No per-model list; fall back to expected input order
            if len(numeric_data) != len(base_input_order):
                return jsonify({'error': f'Expected {len(base_input_order)} features, received {len(numeric_data)}.'}), 400
            df = pd.DataFrame([numeric_data], columns=base_input_order)

        print(f"[DEBUG] Model: {model_key}, Dataset: {dataset_used}, InputFeatures: {len(numeric_data)}, ModelFeatures: {df.shape[1]}, Type: {model_type}")
        
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
                print(f"[DEBUG] Probabilities for top classes: {sorted(probabilities.items(), key=lambda x: -x[1])[:3]}")
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
            
        print(f"[DEBUG] Prediction: {prediction}, Threat: {label_map.get(int(prediction), 'Unknown')}")
        threat_name = label_map.get(int(prediction), 'Unknown')
        is_attack = bool(prediction != 0)  # Assuming 0 is normal traffic
        
        # Extract algorithm name for response
        algorithm_name = type(model).__name__
        if 'XGB' in algorithm_name:
            algorithm_name = 'XGBoost'
        elif 'LGBM' in algorithm_name or 'LightGBM' in algorithm_name:
            algorithm_name = 'LightGBM'
        elif 'RandomForest' in algorithm_name:
            algorithm_name = 'Random Forest'
        elif 'GradientBoosting' in algorithm_name:
            algorithm_name = 'Gradient Boosting'
        elif 'ExtraTrees' in algorithm_name:
            algorithm_name = 'Extra Trees'
        
        # Update metrics
        _update_metrics_placeholder(dataset_used, correct=True)
        
        return jsonify({
            'prediction': int(prediction),
            'threat': threat_name,
            'confidence': round(confidence, 4) if confidence else None,
            'probabilities': probabilities,
            'is_attack': is_attack,
            'feature_order': feature_order,
            'model_type': model_type,
            'model_key': model_key,
            'algorithm': algorithm_name,
            'dataset': dataset_used
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


@app.route('/explain/shap', methods=['POST', 'GET'])
@app.route('/explain/shap/', methods=['POST', 'GET'])
def explain_shap():
    if request.method == 'GET':
        return jsonify({
            'usage': 'POST JSON to /explain/shap',
            'body': {'dataset': 'unsw', 'features': 'list[float]'},
            'optional': {'detailed': False, 'auto_pad': 'query param true/false'}
        }), 200
    req_json = request.json or {}
    dataset = req_json.get('dataset')
    data = req_json.get('features')  # single instance (list of numeric)
    detailed = req_json.get('detailed', False)  # Request detailed explanation
    
    if not dataset or data is None:
        return jsonify({'error': 'dataset and features required'}), 400

    # Load model & features
    m_tuple, feature_order, label_map, err = load_model_and_features(dataset)
    if err:
        return jsonify({'error': err}), 400
    model_package, model_type = m_tuple

    if model_type != 'sklearn':
        return jsonify({'error': 'SHAP explanation only supported for sklearn models'}), 400

    # Extract actual estimator
    estimator = model_package.get('model') if isinstance(model_package, dict) else model_package

    # Validate feature vector length; optionally allow auto padding via flag
    expected_len = len(feature_order)
    if not isinstance(data, (list, tuple)):
        return jsonify({
            'error': 'features must be a list of numeric values',
            'expected_count': expected_len,
            'received_count': 0,
            'hint': f'Use /preset/sample endpoint to get valid feature vectors for dataset "{dataset}"'
        }), 400
    if len(data) != expected_len:
        # allow optional auto padding with zeros when client passes ?auto_pad=true
        auto_pad = request.args.get('auto_pad', 'false').lower() in ('1', 'true', 'yes')
        if auto_pad and len(data) < expected_len:
            data = list(data) + [0.0] * (expected_len - len(data))
        else:
            return jsonify({
                'error': f'Invalid features length: expected {expected_len}, got {len(data)}',
                'expected_count': expected_len,
                'received_count': len(data),
                'hint': f'Use /preset/sample endpoint to get valid feature vectors for dataset "{dataset}", or add ?auto_pad=true to pad with zeros',
                'sample_request': f'curl -X POST http://localhost:5000/preset/sample -H "Content-Type: application/json" -d \'{{"dataset":"{dataset}"}}\''
            }), 400

    # Build input frame in expected feature order
    try:
        df = pd.DataFrame([data], columns=feature_order)
    except Exception as e:
        return jsonify({'error': f'Invalid features vector: {e}'}), 400

    # Apply same preprocessing used at inference
    df_processed = _apply_preprocessing(df, model_package)
    
    # Get prediction for context
    prediction = estimator.predict(df_processed)[0]
    proba = None
    if hasattr(estimator, 'predict_proba'):
        proba = estimator.predict_proba(df_processed)[0]

    # Create explainer (prefer TreeExplainer for tree models, otherwise KernelExplainer)
    # Include model mtime in cache key to auto-invalidate when model file changes
    try:
        model_path_abs = _resolve_path(DATASETS[dataset]['model_path'])
        model_mtime = int(os.path.getmtime(model_path_abs)) if os.path.exists(model_path_abs) else 0
    except Exception:
        model_mtime = 0
    cache_key = f"shap::{dataset}::{type(estimator).__name__}::{model_mtime}"
    with cache_lock:
        explainer = EXPLAINER_CACHE.get(cache_key)
    try:
        if explainer is None:
            try:
                # Works for RandomForest/ExtraTrees/GradientBoosting/XGBoost (sklearn API)
                # Use check_additivity=False to avoid additivity warnings on some ensemble models
                explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
            except Exception:
                # KernelExplainer needs a background dataset and a predict function
                # Build a small background sample in RAW feature space then preprocess inside predict_fn
                try:
                    feature_path = _resolve_path(DATASETS[dataset]['feature_path'])
                    bg_df = pd.read_csv(feature_path).drop(columns=['label'], errors='ignore')
                    # Align background to feature_order and cap size
                    missing = [c for c in feature_order if c not in bg_df.columns]
                    for m in missing:
                        bg_df[m] = 0.0
                    bg_df = bg_df[feature_order]
                    bg_sample = bg_df.sample(min(200, len(bg_df)), random_state=42)
                    # Try using SHAP kmeans background for better KernelExplainer performance
                    try:
                        if hasattr(shap, 'kmeans') and len(bg_sample) > 10:
                            k = min(50, len(bg_sample))
                            bg_kmeans = shap.kmeans(bg_sample.values, k)
                            background = bg_kmeans
                        else:
                            background = bg_sample
                    except Exception:
                        background = bg_sample
                except Exception:
                    # Fallback to the single instance as minimal background
                    background = df.copy()

                def predict_fn(x):
                    X = pd.DataFrame(x, columns=feature_order)
                    Xp = _apply_preprocessing(X, model_package)
                    if hasattr(estimator, 'predict_proba'):
                        return estimator.predict_proba(Xp)
                    # Fallback to decision_function or predict probabilities-like
                    if hasattr(estimator, 'decision_function'):
                        vals = estimator.decision_function(Xp)
                        # Ensure 2D array
                        return np.atleast_2d(vals)
                    preds = estimator.predict(Xp)
                    return np.atleast_2d(preds)

                explainer = shap.KernelExplainer(predict_fn, background)

            with cache_lock:
                EXPLAINER_CACHE[cache_key] = explainer

        # Compute shap values for the processed instance
        shap_vals = explainer.shap_values(df_processed)

        # Extract signed shap values for the predicted class, and absolute importances
        try:
            if isinstance(shap_vals, list):
                # List of arrays per class
                idx = int(prediction) if int(prediction) < len(shap_vals) else 0
                arr = np.asarray(shap_vals[idx])
                signed = arr[0] if arr.ndim == 2 else arr.squeeze()
            elif isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim == 3:
                    signed = shap_vals[0, :, int(prediction)]
                elif shap_vals.ndim == 2:
                    signed = shap_vals[0, :]
                else:
                    signed = shap_vals.flatten()[:len(feature_order)]
            else:
                signed = np.asarray(shap_vals).flatten()[:len(feature_order)]

            # Defensive alignment
            if len(signed) != len(feature_order):
                if isinstance(shap_vals, list) and len(shap_vals) > 0:
                    arr = np.asarray(shap_vals[0])
                    signed = (arr[0] if arr.ndim == 2 else arr.squeeze())[:len(feature_order)]
                else:
                    signed = signed[:len(feature_order)]
        except Exception as e_extract:
            print(f"[WARN] Failed to extract signed SHAP values: {e_extract}")
            signed = np.zeros(len(feature_order), dtype=float)

        agg = np.abs(signed)

        # Create detailed feature contributions with actual values and interpretations
        contrib = []
        for f, importance, value, s in zip(feature_order, agg, data, signed):
            interpretation = _get_feature_interpretation(f, importance, value)
            contrib.append({
                'feature': f,
                'importance': float(importance),
                'feature_value': float(value),
                'interpretation': interpretation,
                'shap_value': float(s)
            })
        
        contrib.sort(key=lambda x: -x['importance'])
        # Identify top positive/negative contributors based on signed SHAP
        pos = [c for c in contrib if c['shap_value'] > 0]
        neg = [c for c in contrib if c['shap_value'] < 0]
        pos.sort(key=lambda x: -x['shap_value'])
        neg.sort(key=lambda x: x['shap_value'])

        # Base value for predicted class (where available)
        try:
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                base_value = float(ev[int(prediction)]) if int(prediction) < len(ev) else float(ev[0])
            else:
                base_value = float(ev)
        except Exception:
            base_value = None
        
        # Build response with context
        response = {
            'prediction': {
                'class': int(prediction),
                'label': label_map.get(int(prediction), f'Class {prediction}'),
                'confidence': float(max(proba)) if proba is not None else None,
                'is_attack': int(prediction) != 0
            },
            'shap_values': contrib[:20],
            'top_attack_indicators': pos[:5],
            'top_normal_indicators': neg[:5],
            'base_value': base_value,
            'summary': _generate_shap_summary(contrib[:10], label_map.get(int(prediction), 'Unknown'))
        }
        
        return jsonify(response)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] LIME explanation failed: {error_details}")
        return jsonify({
            'error': f'LIME explanation failed: {str(e)}',
            'hint': 'Try using a real sample from /preset/sample instead of zero-padded features',
            'details': str(e)
        }), 500
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] SHAP explanation failed: {error_details}")
        return jsonify({
            'error': f'SHAP explanation failed: {str(e)}',
            'hint': 'Try using a real sample from /preset/sample instead of zero-padded features',
            'details': str(e)
        }), 500


def _get_feature_interpretation(feature_name, importance, value):
    """Generate human-readable interpretation of feature contribution"""
    # Feature descriptions
    descriptions = {
        'sbytes': 'Source bytes transferred',
        'dbytes': 'Destination bytes transferred',
        'sload': 'Source bits per second',
        'dload': 'Destination bits per second',
        'dur': 'Connection duration',
        'rate': 'Packet transmission rate',
        'sttl': 'Source time-to-live value',
        'dttl': 'Destination time-to-live value',
        'ct_srv_dst': 'Connections to same service',
        'ct_dst_src_ltm': 'Connections from destination',
        'service_dns': 'DNS service activity',
        'service_-': 'Unidentified service',
        'proto_udp': 'UDP protocol usage',
        'synack': 'SYN-ACK packet count',
        'tcprtt': 'TCP round trip time',
        'dpkts': 'Destination packet count',
        'dloss': 'Destination packet loss',
    }
    
    desc = descriptions.get(feature_name, feature_name.replace('_', ' ').title())
    
    # Determine strength
    if importance > 0.1:
        strength = "STRONG"
    elif importance > 0.05:
        strength = "MODERATE"
    elif importance > 0.01:
        strength = "WEAK"
    else:
        strength = "MINIMAL"
    
    # Contextualize value
    if value == 0:
        context = "absent"
    elif value < 0.1:
        context = "very low"
    elif value < 0.3:
        context = "low"
    elif value < 0.7:
        context = "moderate"
    elif value < 0.9:
        context = "high"
    else:
        context = "very high"
    
    return f"{desc} ({context}) - {strength} indicator"


def _generate_shap_summary(top_features, threat_name):
    """Generate textual summary of SHAP analysis"""
    if not top_features:
        return "No significant features identified"
    
    summary = f"Classification: {threat_name}\n\n"
    summary += "Top contributing features:\n"
    
    for i, feat in enumerate(top_features[:5], 1):
        summary += f"{i}. {feat['interpretation']} (importance: {feat['importance']:.4f})\n"
    
    return summary


@app.route('/explain/lime', methods=['POST', 'GET'])
@app.route('/explain/lime/', methods=['POST', 'GET'])
def explain_lime():
    if request.method == 'GET':
        return jsonify({
            'usage': 'POST JSON to /explain/lime',
            'body': {'dataset': 'unsw', 'features': 'list[float]'},
            'notes': 'Ensure Content-Type: application/json'
        }), 200
    if LimeTabularExplainer is None:
        return jsonify({'error': 'LIME not installed'}), 500

    req_json = request.json or {}
    dataset = req_json.get('dataset')
    data = req_json.get('features')
    if not dataset or data is None:
        return jsonify({'error': 'dataset and features required'}), 400

    # Load model & features
    m_tuple, feature_order, label_map, err = load_model_and_features(dataset)
    if err:
        return jsonify({'error': err}), 400
    model_package, model_type = m_tuple

    if model_type != 'sklearn':
        return jsonify({'error': 'LIME explanation only supported for sklearn models'}), 400

    # Extract estimator
    estimator = model_package.get('model') if isinstance(model_package, dict) else model_package

    # Validate feature vector length first (with optional auto-pad)
    expected_len = len(feature_order)
    if not isinstance(data, (list, tuple)):
        return jsonify({
            'error': 'features must be a list of numeric values',
            'expected_count': expected_len,
            'received_count': 0,
            'hint': f'Use /preset/sample endpoint to get valid feature vectors for dataset "{dataset}"'
        }), 400
    if len(data) != expected_len:
        auto_pad = request.args.get('auto_pad', 'false').lower() in ('1', 'true', 'yes')
        if auto_pad and len(data) < expected_len:
            data = list(data) + [0.0] * (expected_len - len(data))
        else:
            return jsonify({
                'error': f'Invalid features length: expected {expected_len}, got {len(data)}',
                'expected_count': expected_len,
                'received_count': len(data),
                'hint': f'Use /preset/sample endpoint to get valid feature vectors for dataset "{dataset}", or add ?auto_pad=true to pad with zeros',
                'sample_request': f'curl -X POST http://localhost:5000/preset/sample -H "Content-Type: application/json" -d \'{{"dataset":"{dataset}"}}\''
            }), 400

    # Background sample in RAW space; preprocessing applied inside predict_fn
    try:
        feature_path = _resolve_path(DATASETS[dataset]['feature_path'])
        full_df = pd.read_csv(feature_path).drop(columns=['label'], errors='ignore')
        # Align columns to feature_order, add any missing as zeros, cap size
        missing = [c for c in feature_order if c not in full_df.columns]
        for m in missing:
            full_df[m] = 0.0
        full_df = full_df[feature_order]
        sample_bg = full_df.sample(min(500, len(full_df)), random_state=42)
    except Exception as e:
        return jsonify({'error': f'Failed to build LIME background: {e}'}), 500

    # Build predict function that applies preprocessing to inputs
    def predict_fn(x):
        X = pd.DataFrame(x, columns=feature_order)
        Xp = _apply_preprocessing(X, model_package)
        if hasattr(estimator, 'predict_proba'):
            return estimator.predict_proba(Xp)
        if hasattr(estimator, 'decision_function'):
            vals = estimator.decision_function(Xp)
            return np.atleast_2d(vals)
        preds = estimator.predict(Xp)
        # Map to a two-class probability-like output if needed
        preds = np.asarray(preds)
        # Make a simple one-hot style matrix if shape not 2D
        if preds.ndim == 1:
            classes = sorted(set(int(v) for v in preds.tolist()))
            k = max(max(classes) + 1, 2)
            out = np.zeros((len(preds), k), dtype=float)
            for i, v in enumerate(preds):
                idx = int(v) if 0 <= int(v) < k else 0
                out[i, idx] = 1.0
            return out
        return preds

    # Get prediction for context
    instance = np.array(data, dtype=np.float64)
    df_instance = pd.DataFrame([instance], columns=feature_order)
    df_processed = _apply_preprocessing(df_instance, model_package)
    prediction = estimator.predict(df_processed)[0]
    proba = None
    if hasattr(estimator, 'predict_proba'):
        proba = estimator.predict_proba(df_processed)[0]

    # Cache explainer keyed by dataset, model class, and model mtime
    try:
        model_path_abs = _resolve_path(DATASETS[dataset]['model_path'])
        model_mtime = int(os.path.getmtime(model_path_abs)) if os.path.exists(model_path_abs) else 0
    except Exception:
        model_mtime = 0
    lime_cache_key = f"lime::{dataset}::{type(estimator).__name__}::{model_mtime}"
    with cache_lock:
        explainer = EXPLAINER_CACHE.get(lime_cache_key)
    if explainer is None:
        # Optional kmeans reduction for background
        try:
            if hasattr(shap, 'kmeans') and len(sample_bg) > 50:
                k = min(100, len(sample_bg))
                bg = shap.kmeans(sample_bg.values, k)
            else:
                bg = sample_bg.values
        except Exception:
            bg = sample_bg.values

        explainer = LimeTabularExplainer(
            bg,
            feature_names=feature_order,
            class_names=list(label_map.values()) if label_map else None,
            discretize_continuous=False,
            random_state=42
        )
        with cache_lock:
            EXPLAINER_CACHE[lime_cache_key] = explainer

    try:
        exp = explainer.explain_instance(instance, predict_fn, num_features=15)
        
        # Extract explanation as list
        explanation_list = exp.as_list(label=int(prediction))
        
        # Enhanced explanations with interpretations
        detailed_explanations = []
        for feature_condition, weight in explanation_list:
            # Extract feature name from condition
            parts = feature_condition.split()
            feature_name = parts[0] if parts else feature_condition
            
            # Get feature value from instance
            try:
                feat_idx = feature_order.index(feature_name)
                feat_value = float(instance[feat_idx])
            except (ValueError, IndexError):
                feat_value = None
            
            interpretation = _get_feature_interpretation(feature_name, abs(weight), feat_value if feat_value is not None else 0)
            
            detailed_explanations.append({
                'feature': feature_name,
                'condition': feature_condition,
                'weight': float(weight),
                'feature_value': feat_value,
                'interpretation': interpretation,
                'direction': 'attack' if weight > 0 else 'normal'
            })
        
        response = {
            'prediction': {
                'class': int(prediction),
                'label': label_map.get(int(prediction), f'Class {prediction}'),
                'confidence': float(max(proba)) if proba is not None else None,
                'is_attack': int(prediction) != 0
            },
            'lime_explanation': detailed_explanations,
            'intercept': float(exp.intercept[int(prediction)]) if hasattr(exp, 'intercept') else None,
            'raw_local_prediction': float(exp.local_pred[int(prediction)]) if hasattr(exp, 'local_pred') else None,
            'summary': _generate_lime_summary(detailed_explanations[:5], label_map.get(int(prediction), 'Unknown'))
        }
        return jsonify(response)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] LIME explanation failed: {error_details}")
        return jsonify({
            'error': f'LIME explanation failed: {str(e)}',
            'hint': 'Try using a real sample from /preset/sample instead of zero-padded features',
            'details': str(e)
        }), 500


def _generate_lime_summary(top_explanations, threat_name):
    """Generate textual summary of LIME analysis"""
    if not top_explanations:
        return "No significant explanations identified"
    
    summary = f"Classification: {threat_name}\n\n"
    summary += "Key explanatory factors:\n"
    
    for i, exp in enumerate(top_explanations, 1):
        direction = "increases" if exp['weight'] > 0 else "decreases"
        summary += f"{i}. {exp['interpretation']}\n   Weight: {exp['weight']:.4f} ({direction} attack probability)\n"
    
    return summary


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
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(df_processed)
            preds = probs.argmax(axis=1)
            probs_out = [{str(i): float(p) for i,p in enumerate(row)} for row in probs]
        else:
            preds = model.predict(df_processed)
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

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Run predictions across all available models for comparison.
    Body: {"features": [list of feature values]}
    Returns: {"results": [{"model_key": str, "prediction": int, "confidence": float, ...}]}
    """
    try:
        req_json = request.json
        if not req_json:
            return jsonify({'error': 'Request body must be valid JSON.'}), 400
            
        features = req_json.get('features')
        if not features or not isinstance(features, list):
            return jsonify({'error': 'Request must include "features" (list).'}), 400
        
        # Validate feature count
        if len(features) != len(GLOBAL_FEATURE_ORDER):
            return jsonify({'error': f'Expected {len(GLOBAL_FEATURE_ORDER)} features, received {len(features)}.'}), 400
        
        # Validate feature values are numeric
        try:
            numeric_data = [float(x) for x in features]
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'All feature values must be numeric. Error: {str(e)}'}), 400
        
        results = []
        
        # Run prediction on each available model
        for model_key in MODEL_REGISTRY.keys():
            try:
                model_package, model_type = _load_model(model_key)
                
                # Extract actual model from package
                if isinstance(model_package, dict) and 'model' in model_package:
                    model = model_package['model']
                else:
                    model = model_package
                
                # Determine per-model feature list if available and align
                model_features = None
                if isinstance(model_package, dict):
                    model_features = model_package.get('feature_names')
                if model_features:
                    if len(numeric_data) == len(model_features):
                        df = pd.DataFrame([numeric_data], columns=model_features)
                    else:
                        # Try to map from global/dataset features
                        source_names = GLOBAL_FEATURE_ORDER if len(GLOBAL_FEATURE_ORDER) == len(numeric_data) else None
                        if not source_names:
                            # Attempt dataset CSV for guessed dataset
                            dataset_guess = 'unsw' if 'unsw' in model_key.lower() else ('cicids' if 'cicids' in model_key.lower() else DEFAULT_DATASET)
                            entry = DATASETS.get(dataset_guess)
                            base_csv_features = None
                            if entry:
                                try:
                                    feat_df = pd.read_csv(_resolve_path(entry['feature_path']), nrows=1)
                                    base_csv_features = [c for c in feat_df.columns if c != 'label']
                                    if len(base_csv_features) == len(numeric_data):
                                        source_names = base_csv_features
                                except Exception:
                                    pass
                        if source_names:
                            values_map = {name: val for name, val in zip(source_names, numeric_data)}
                            aligned_row = [values_map.get(name, 0.0) for name in model_features]
                            df = pd.DataFrame([aligned_row], columns=model_features)
                        else:
                            df = pd.DataFrame([numeric_data], columns=GLOBAL_FEATURE_ORDER)
                else:
                    # Fallback: use global order
                    df = pd.DataFrame([numeric_data], columns=GLOBAL_FEATURE_ORDER)
                
                # Apply preprocessing if needed
                df_processed = _apply_preprocessing(df, model_package)
                
                # Make prediction
                prediction = model.predict(df_processed)[0]
                
                # Get confidence if available
                confidence = None
                if hasattr(model, 'predict_proba'):
                    try:
                        probs = model.predict_proba(df_processed)[0]
                        confidence = float(np.max(probs))
                    except Exception:
                        pass
                elif hasattr(model, 'decision_function'):
                    try:
                        decision = model.decision_function(df_processed)[0]
                        if hasattr(decision, '__len__') and len(decision) > 1:
                            confidence = float(np.max(decision))
                        else:
                            confidence = float(abs(decision))
                    except Exception:
                        pass
                
                # Guess dataset and get label mapping
                if 'unsw' in model_key.lower():
                    label_map = DATASETS.get('unsw', {}).get('label_map', {})
                    dataset_guess = 'unsw'
                elif 'cicids' in model_key.lower():
                    label_map = DATASETS.get('cicids', {}).get('label_map', {})
                    dataset_guess = 'cicids'
                else:
                    label_map = DATASETS.get(DEFAULT_DATASET, {}).get('label_map', {})
                    dataset_guess = DEFAULT_DATASET
                
                threat_name = label_map.get(int(prediction), 'Unknown')
                is_attack = bool(prediction != 0)
                
                # Extract algorithm name
                algorithm_name = type(model).__name__
                if 'XGB' in algorithm_name:
                    algorithm_name = 'XGBoost'
                elif 'LGBM' in algorithm_name or 'LightGBM' in algorithm_name:
                    algorithm_name = 'LightGBM'
                elif 'RandomForest' in algorithm_name:
                    algorithm_name = 'Random Forest'
                elif 'GradientBoosting' in algorithm_name:
                    algorithm_name = 'Gradient Boosting'
                elif 'ExtraTrees' in algorithm_name:
                    algorithm_name = 'Extra Trees'
                
                results.append({
                    'model_key': model_key,
                    'algorithm': algorithm_name,
                    'prediction': int(prediction),
                    'threat': threat_name,
                    'confidence': round(confidence, 4) if confidence else None,
                    'is_attack': is_attack,
                    'dataset': dataset_guess,
                    'model_type': model_type
                })
                
            except Exception as e:
                print(f"[WARN] Batch prediction failed for model {model_key}: {e}")
                results.append({
                    'model_key': model_key,
                    'algorithm': 'Unknown',
                    'error': str(e),
                    'prediction': None,
                    'threat': None,
                    'confidence': None,
                    'is_attack': False,
                    'dataset': None,
                    'model_type': None
                })
        
        return jsonify({
            'results': results,
            'total_models': len(results),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'feature_count': len(GLOBAL_FEATURE_ORDER)
        })
        
    except Exception as e:
        print(f"[ERROR] Batch prediction failed: {e}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("\nAI NIDS API starting...")
    print("Dashboard available at: http://localhost:5000/dashboard\n")
    
    # Initialize default dataset on startup
    print("Initializing default dataset and models...")
    _initialize_default_dataset()
    print(f"Loaded {len(MODEL_REGISTRY)} models from registry")
    
    app.run(debug=True)
