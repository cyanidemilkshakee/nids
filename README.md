# AI Network Intrusion Detection System (AI_NIDS)

Modern, data-driven intrusion detection pipeline with a REST API, interactive dashboard, explainability, feedback capture, and optional incremental updates. The primary supported dataset (as currently wired into the API) is **UNSW-NB15**; the codebase also contains utilities that can be extended for CICIDS or additional datasets.

---
## 1. Key Capabilities
| Capability | Status | Notes |
|------------|--------|-------|
| Preprocessing (UNSW, CICIDS) | Complete | `preprocess/preprocess.py` handles scaling + label encoding |
| Feature Selection | Complete | `feature_selection/select_features.py` (RandomForest importance ranking) |
| Core Model (RandomForest) | Complete | Trained via `train/train.py` -> `models/saved_model.joblib` |
| Adversarial / DL Model | Present (file) | `models/saved_model_adversarial.keras` not yet served by API |
| Ensemble & CNN scripts | Present | `train/ensemble.py`, `train/cnn_model.py` (not exposed in API) |
| Incremental Update | Basic | `/incremental/update` uses feedback + SGD partial_fit fallback |
| Explainability | SHAP + LIME | `/explain/shap`, `/explain/lime` endpoints |
| Real Data Presets | NEW | `/preset/sample` returns real row (by label) to populate form |
| Feedback Loop | CSV-based | `/feedback` appends to `data/feedback.csv` |
| Metrics Tracking | Lightweight | `models/metrics.json` updated (online accuracy, confusion matrix) |
| Dashboard | Implemented | Static HTML/JS served from Flask under `/dashboard/` |
| Dockerfile | Present (baseline) | Adjust as needed for production |

---
## 2. Repository Structure
```
api/                Flask API (prediction, explainability, presets, feedback)
dashboard/          Static UI (index.html, styles.css, script.js)
data/
  raw/              Original datasets (UNSW_NB15*, CICIDS subdirs)
  processed/        Scaled + selected feature CSVs
  feedback.csv      Appended user feedback (features + predicted + label correctness)
explainability/     SHAP & LIME artifacts (e.g. generated HTML)
feature_selection/  Feature ranking & selection script
models/             Trained artifacts (RandomForest, adversarial Keras, metrics, feature list)
preprocess/         Unified preprocessing pipeline
train/              Training scripts (RF, CNN, ensemble, incremental)
test_api.py         Smoke test for the running API
Dockerfile          Container build recipe (adjustable)
README.md           This documentation
```

---
## 3. Data Pipeline Overview
1. Preprocess UNSW / CICIDS (merges train+test, scales numeric, encodes labels) -> `data/processed/*_processed.csv`.
2. Feature selection script reduces dimensionality and writes `*_selected_features.csv` (and feature names JSON).
3. Training script (`train/train.py`):
   - Loads processed (selected) dataset.
   - Optional synthetic augmentation (if file exists or GAN env flags used).
   - Trains RandomForest (n=400 trees) with class balancing.
   - Saves model + metrics + canonical feature order.
4. API loads model + `feature_names.json` ensuring consistent feature index order for predictions & presets.

---
## 4. Models Included
### 4.1 RandomForest (Primary)
Trained via `train/train.py` with:
* n_estimators: 400
* class_weight: balanced (computed from data distribution)
* Stored at: `models/saved_model.joblib`
* Feature order persisted in `models/feature_names.json`

### 4.2 Adversarial / Deep Model
File: `models/saved_model_adversarial.keras` (Keras format) – currently **not** wired into the API endpoints (only sklearn/joblib model path supported). To integrate, extend `load_model_and_features` with a TF/Keras branch.

### 4.3 CNN & Ensemble Scripts
* `train/cnn_model.py` – prototype CNN training (for experimentation)
* `train/ensemble.py` – possible voting/stacking logic
Not currently deployed by Flask; adapt `DATASETS` mapping to add additional model entries.

### 4.4 Incremental Update Path
Endpoint `/incremental/update` attempts to reuse the existing stored model, performing a `partial_fit` if supported (SGD fallback logic). RandomForest (sklearn) is not inherently incremental; switch to `SGDClassifier` or similar for fully online learning in a production scenario.

### 4.5 Newly Added Boosted / Ensemble Tree Scripts
The repository now includes additional training utilities for stronger tabular performance and model diversity:

| Script | Algorithm | Key Parameters | Output Model (default) |
|--------|-----------|----------------|------------------------|
| `train/train_xgboost.py` | XGBoost (multi:softprob) | n_estimators=800, max_depth=7, lr=0.05, subsample=0.8 | `models/saved_model_xgboost.joblib` |
| `train/train_gradient_boosting.py` | sklearn GradientBoosting | n_estimators=600, lr=0.05, max_depth=3 | `models/saved_model_gradient_boosting.joblib` |
| `train/train_extratrees.py` | ExtraTreesClassifier | n_estimators=800, max_features=sqrt, class_weight balanced | `models/saved_model_extratrees.joblib` |

All scripts:
- Accept the same environment variables (`PRIMARY_DATA`, `SYNTH_DATA`, `OUTPUT_MODEL`, `MAX_ROWS`, `TEST_SIZE`).
- Persist metrics to `models/metrics.json` under a key matching the output filename stem.
- Re-use `feature_names.json` if already present (to maintain consistent ordering across models).

Example (Windows CMD) to train XGBoost on UNSW selected features:
```cmd
set PRIMARY_DATA=unsw_selected_features.csv
set OUTPUT_MODEL=saved_model_xgboost.joblib
python train\train_xgboost.py
```
Then list models via API:
```cmd
curl http://localhost:5000/models
```
Add a model selection feature in the dashboard by referencing the new entries in `models/metrics.json`.

---
## 5. Label Mapping (UNSW-NB15)
Current API `DATASETS` mapping (in `api/app.py`) uses:
```
0: Normal
1: Analysis
2: Backdoor
3: DoS
4: Exploits
6: Fuzzers
7: Generic
8: Reconnaissance
9: Worms
```
Note: Index 5 is intentionally absent (some UNSW label configurations skip a class depending on the subset available). The sampling & prediction logic tolerates this gap. If you need to re-align labels, regenerate preprocessing and update the `label_map` accordingly.

---
## 6. API Endpoints
Base URL (default): `http://localhost:5000`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Service status + supported datasets |
| GET | `/datasets` | Compact dataset list + label maps |
| GET | `/model_meta` | Rich metadata (features count, metrics, confusion matrix) |
| GET | `/features/<dataset>` | Ordered feature names required for prediction |
| POST | `/predict` | Perform inference on a single feature vector |
| POST | `/feedback` | Submit feedback (correct / false_positive / false_negative, etc.) |
| POST | `/preset/sample` | NEW: Return a real sample (optionally filtered by label) |
| POST | `/explain/shap` | Top feature contributions (global approximation for instance) |
| POST | `/explain/lime` | Local surrogate explanation (if LIME installed) |
| POST | `/incremental/update` | Attempt incremental training using feedback file |
| GET | `/status` | Health snapshot (cache counts, feedback rows) |
| GET | `/dashboard/` | Serves the HTML UI |

### 6.1 Prediction Request / Response
Request:
```json
{
  "dataset": "unsw",
  "features": [0.0, 0.12, 0.03, ...]
}
```
Response:
```json
{
  "prediction": 3,
  "threat": "DoS",
  "confidence": 0.9471,
  "probabilities": {"0": 0.02, "3": 0.9471, "7": 0.01, "8": 0.01},
  "is_attack": true,
  "feature_order": ["dur", "sbytes", ...],
  "model_type": "sklearn"
}
```

### 6.2 Preset Sampling
Fetch real feature vectors to eliminate synthetic preset drift:
```http
POST /preset/sample
{
  "dataset": "unsw",
  "label": 3
}
```
Returns aligned `features` array in the canonical order for direct prediction.

### 6.3 Feedback
```json
{
  "features": [...],
  "predicted_label": 3,
  "feedback_label": "correct"
}
```
Appends a line to `data/feedback.csv`. Incremental update can consume only rows marked `correct`.

---
## 7. Real-Data Presets (Why They Matter)
Older heuristic presets produced unrealistic distributions → poor or confusing predictions. The new `/preset/sample` endpoint pulls genuine, scaled examples from the processed dataset and aligns them with the trained model’s feature order. The dashboard now:
* Lists each threat class as a clickable chip.
* Requests a real sample for that label.
* Populates the form precisely without manual value heuristics.
* Provides a random sample when the “Random” preset is used (`random: true`).

---
## 8. Explainability
| Technique | Implementation | Notes |
|-----------|----------------|-------|
| SHAP | `shap.TreeExplainer` fallback to KernelExplainer | Aggregates absolute mean importance across classes |
| LIME | `lime.lime_tabular.LimeTabularExplainer` | Requires LIME dependency; samples up to 500 background rows |

Both endpoints return trimmed lists of feature contributions suitable for UI display. For full plots, extend the code to save HTML artifacts per request (currently not implemented for API responses).

---
## 9. Incremental Learning Caveats
The stored RandomForest model is not intrinsically incremental. The incremental path is opportunistic:
* Loads existing model; if it exposes `partial_fit` (e.g., switched to SGDClassifier in a future revision) it applies updates.
* Otherwise returns an error.
To fully leverage online learning, retrain using an incremental-friendly algorithm and persist that model instead.

---
## 10. Running the System (Windows Examples)
### 10.1 Install Dependencies
```cmd
pip install -r requirements.txt
```
Ensure any optional libraries (e.g. `lime`, `shap`) are present if you want explainability endpoints.

### 10.2 Start API (serves dashboard)
```cmd
python api\app.py
```
Visit: http://localhost:5000/dashboard/

### 10.3 Test API
```cmd
python test_api.py
```

### 10.4 Sample cURL Prediction
```cmd
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"dataset\":\"unsw\",\"features\":[0,0,0]}"
```
(Use the exact feature length returned by `/features/unsw`).

---
## 11. Adding Another Dataset (High-Level Steps)
1. Preprocess & scale -> produce `data/processed/<name>_selected_features.csv` and feature list JSON.
2. Train a model -> save into `models/` as `<name>_model.joblib` and update `feature_names.json` if shared or create a dataset-specific feature list loader path.
3. Extend `DATASETS` in `api/app.py` with:
```python
DATASETS["<name>"] = {
  "model_path": "../models/<file>.joblib",
  "feature_path": "../data/processed/<file>.csv",
  "label_map": { ... }
}
```
4. Restart API. Frontend auto-populates dataset select + labels.

---
## 12. Docker (Baseline)
`Dockerfile` is present. Typical build/run:
```cmd
docker build -t ai_nids .
docker run -p 5000:5000 ai_nids
```
Adjust for production: multi-stage build, non-root user, pinned dependency versions, healthcheck, and optional volume mounts for models/logs.

---
## 13. Metrics & Monitoring
* `models/metrics.json` persists evaluation metrics (accuracy, precision, recall, F1, confusion matrix) and an `online_accuracy` updated after each prediction (placeholder logic—treats every prediction as correct unless feedback indicates otherwise in future extensions).
* `/status` returns lightweight operational stats (cache sizes, feedback count).

---
## 14. Security Considerations (To Address in Production)
| Area | Recommended Action |
|------|--------------------|
| Input Validation | Enforce numeric ranges & schema for feature vectors |
| Auth | API keys / OAuth / mTLS for protected deployments |
| Rate Limiting | Mitigate brute-force / abuse scenarios |
| Data Integrity | Sign or hash model & feature list assets |
| Feedback Sanitization | Guard against CSV injection / payload tampering |

---
## 15. Known Gaps / Future Enhancements
* True multi-dataset support in runtime (currently only `unsw` is configured).
* Keras / DL model serving path.
* Richer frontend analytics (time trends, confusion matrix visualization).
* Database-backed feedback store (replace CSV) + retraining trigger pipeline.
* Unit & integration test coverage expansion.
* OpenAPI / Swagger documentation generation.

---
## 16. Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| 400: feature length mismatch | Frontend/sample mismatch | Reload features with the Refresh button; ensure preset uses new sample endpoint |
| SHAP fails for non-tree model | Explainer fallback limitations | Install correct SHAP version; consider KernelExplainer sample size tuning |
| LIME endpoint error | lime not installed | `pip install lime` |
| Incremental update error | Model lacks `partial_fit` | Replace model with incremental-friendly estimator |
| Wrong threat label | Label map misalignment | Regenerate preprocessing + update `DATASETS` mapping |

---
## 17. Licensing & Dataset Notes
UNSW-NB15 & CICIDS2017 are subject to their respective licenses / usage terms. Ensure you have the right to process and redistribute derived artifacts before deploying commercially. This project code itself can be licensed as you choose (add a LICENSE file if distributing publicly).

---
## 18. At a Glance (Quick Commands)
```cmd
:: Install
pip install -r requirements.txt

:: Start API + Dashboard
python api\app.py

:: Get feature list
curl http://localhost:5000/features/unsw

:: Sample preset (label 3 – DoS)
curl -X POST http://localhost:5000/preset/sample -H "Content-Type: application/json" -d "{\"dataset\":\"unsw\",\"label\":3}"

:: Predict (fill real length features array!)
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"dataset\":\"unsw\",\"features\":[0,0,...]}"
```

---
## 19. Contributing
1. Fork / branch
2. Run `test_api.py` before and after changes
3. Add or update metrics if retraining models
4. Open PR with clear summary of changes

---
## 20. Summary
This project delivers a pragmatic intrusion detection workflow centered on a RandomForest model with explainability, real-data presets, and a lightweight feedback loop. Extend it by adding multi-dataset support, production hardening (auth, DB, logging), and richer model serving layers as needed.

Happy analyzing and stay secure.