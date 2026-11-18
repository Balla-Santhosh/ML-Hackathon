# PathSnap – Map-Matching & Route Reconstruction

Noisy GPS tracks are matched to a road network in two stages:

1. **Candidate classifier** – for every GPS fix, the nearest road segments (arcs) within 80 m (max 8) become
   candidates; a classifier scores each candidate with P(candidate is the true segment) from 22 geometric,
   heading, context and topology features.
2. **HMM + Viterbi** – candidates are hidden states, classifier probabilities are emissions, and transitions
   compare network driving distance with straight-line GPS displacement. Viterbi decoding returns one
   connected road segment per GPS fix.

The system is deployed on two laptops: a **FastAPI** backend (model, HMM, road network, server-side scoring)
and a **Streamlit** evaluator UI.

## Results (student-testing partition, 10 trips, 19,160 fixes)

| Stage | Top-1 accuracy | Route continuity | Route mismatch |
|---|---|---|---|
| Tuned LightGBM, classifier only | 0.9127 | 0.9810 | 0.0971 |
| Tuned LightGBM + HMM/Viterbi | **0.9289** | **0.9984** | **0.0548** |

9 models were compared (`results/model_comparison.csv`); boosting and Random Forest reached ~0.951 CV macro-F1.
Full tables: `report/results_tables.md`.

## Model comparison (5-fold GroupKFold CV on 80 training trips; test = 10 student-test trips)

| Model | Family | CV macro-F1 | Test top-1 | Test macro-F1 | Test log loss | Route continuity |
|---|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.9440 | 0.8862 | 0.9340 | 0.1029 | 0.9726 |
| Gaussian NB | Probabilistic | 0.9347 | 0.8677 | 0.9231 | 0.6807 | 0.9531 |
| KNN | Instance-based | 0.9424 | 0.8843 | 0.9328 | 0.1407 | 0.9670 |
| Decision Tree | Tree | 0.9460 | 0.8888 | 0.9355 | 0.1032 | 0.9724 |
| Random Forest | Bagging | 0.9508 | 0.9059 | 0.9456 | 0.0742 | 0.9794 |
| HistGradientBoosting | Boosting | 0.9495 | 0.9045 | 0.9447 | 0.0821 | 0.9771 |
| XGBoost | Boosting | 0.9506 | 0.9141 | 0.9504 | 0.0731 | 0.9812 |
| LightGBM | Boosting | 0.9509 | 0.9106 | 0.9483 | 0.0746 | 0.9810 |
| MLP (PyTorch) | Neural network | 0.9484 | 0.9011 | 0.9427 | 0.0801 | 0.9780 |

LightGBM (highest CV macro-F1) was tuned with Optuna (30 TPE trials): CV macro-F1 0.9509 → 0.9517,
test top-1 0.9106 → 0.9127. Decoder parameters (β = 5, α = 2) were tuned on 12 held-out training trips.

## Feature selection (training trips only)

* Spearman |ρ| ≥ 0.9 pairs: dist / dist_minus_min / dist_ratio, dist_minus_min / ctx_minus_min, ctx_mean / ctx_max / ctx_wide_mean.
* Permutation importance (log-loss increase): head_cos_wide 0.195, ctx_minus_min 0.101, dist 0.049, head_cos 0.021, t_along 0.015.
* Top-k ablation (LightGBM, macro-F1): 3 → 0.922, 5 → 0.949, 8 → 0.950, 12 → 0.952, 16 → 0.953, 22 → 0.953. All 22 features kept.

## Repository layout

| Folder / file | Contents |
|---|---|
| `pathsnap/` | Library: `io.py` (raw loading/cleaning), `geo.py` (projection, geometry), `network.py` (cropped graph, Dijkstra), `labels.py` (route → per-fix labels), `candidates.py`, `features.py`, `models.py`, `metrics.py`, `hmm.py` (HMM/Viterbi), `evaluation.py`, `pipeline.py` (inference), `viz.py` (Folium) |
| `scripts/01_eda.py` | Dataset inventory, missing-value checks, connectivity, plots, Folium EDA maps |
| `scripts/02_build_dataset.py` | Candidate generation, feature engineering, label alignment → `data_cache/` |
| `scripts/03_feature_selection.py` | Feature correlation, mutual information, permutation importance, top-k ablation |
| `scripts/03_train_models.py` | 9 models, 5-fold GroupKFold CV, student-test evaluation, comparison plots |
| `scripts/04_bayes_opt.py` | Optuna TPE Bayesian optimisation of the chosen model; exports `artifacts/model.joblib` |
| `scripts/05_hmm_decoder.py` | Decoder tuning, classifier-only vs HMM comparison, Folium route maps |
| `scripts/06_report_tables.py` | Builds Markdown result tables from `results/` |
| `api/main.py` | FastAPI backend (Laptop 1) |
| `ui/app.py` | Streamlit evaluator UI (Laptop 2) |
| `artifacts/` | Final model, model metadata, HMM parameters |
| `results/` | CSV/JSON experiment results, logs, `figures/`, `maps/` (interactive Folium HTML), `eda/` |
| `results/feature_selection/` | Feature importance, correlation and ablation CSVs |
| `report/results_tables.md` | Auto-generated result tables for the report |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Raw data is not committed. By default the code expects it at `../Student folder/Student folder/{train,student_test}`;
override with the `PATHSNAP_DATA` environment variable.

## Reproducing the experiments

```bash
python scripts/02_build_dataset.py     # candidates + features + labels (~1 min, parallel)
python scripts/01_eda.py               # EDA tables, figures, Folium maps
python scripts/03_feature_selection.py # feature analysis + ablation (~5 min)
python scripts/03_train_models.py      # model comparison (~30-40 min)
python scripts/04_bayes_opt.py         # Bayesian optimisation of LightGBM (~30 min), exports artifacts/
python scripts/05_hmm_decoder.py       # HMM tuning + classifier-vs-HMM evaluation + route maps (~10 min)
python scripts/06_report_tables.py     # result tables -> report/results_tables.md
```

Random seed: 42 everywhere (`pathsnap/config.py`).

## Deployment (two laptops)

**Laptop 1 – API** (put the 10 evaluation trips in a folder, one sub-folder per trip, same layout as `student_test`):

```bash
set PATHSNAP_EVAL_DIR=D:\path\to\evaluator_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

(PowerShell: `$env:PATHSNAP_EVAL_DIR="D:\path\to\evaluator_test"`.) Without the variable the API serves the
student-testing trips. Every request and its accuracy are logged in this terminal. Allow port 8000 through the
Windows firewall and note the laptop's IP (`ipconfig`).

**Laptop 2 – UI** (only needs `streamlit`, `folium`, `requests`):

```bash
streamlit run ui/app.py -- --api http://<laptop1-ip>:8000
```

The UI lists the evaluation trips, sends the selected `trip_id` to `POST /predict`, and displays the predicted
route, GPS fixes, road network and the server-computed metrics.

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | status, model name, HMM parameters |
| GET | `/trips` | – | evaluation trip ids and fix counts |
| POST | `/predict` | `{"trip_id": "00000007"}` | predicted arc per fix, route arcs and geometry, metrics, timings |

Reference routes are loaded only by `server_side_evaluation()` after prediction and are never returned to the
UI or used by the predictor.

## Data-leakage safeguards

* No trip is split across partitions; CV uses `GroupKFold` by `trip_id`.
* Preprocessing has no learned transforms fitted outside training folds (scalers live inside model pipelines).
* Decoder parameters are tuned on 12 held-out *training* trips; the student-test set is used only for comparison.
* `pipeline.Predictor` loads trips with `with_route=False`.
