# PathSnap – Map-Matching & Route Reconstruction

Noisy GPS tracks are matched to a road network. **Phase 1 (this commit)** covers data understanding and the
machine-learning pipeline: raw-data processing, exploratory analysis, label construction, candidate generation,
feature engineering, feature selection, and the training / comparison of five classifiers from five algorithm
families (Logistic Regression, KNN, Decision Tree, Random Forest, LightGBM).

*Phase 2 will add: further models, Bayesian hyperparameter optimisation, route reconstruction with HMM/Viterbi,
the FastAPI backend and the evaluator UI.*

## ML flow

```
raw trip files (.track .nodes .arcs .route)
   │  pathsnap/io.py        read + clean (finite coords, sorted time, no duplicate timestamps)
   ▼
local metric projection     pathsnap/geo.py      (equirectangular around the trip, metres)
   ▼
cropped road graph          pathsnap/network.py  (track bbox + 400 m, directed arcs, global arc ids kept)
   ▼
per-fix reference labels    pathsnap/labels.py   (monotone DP alignment of fixes to the .route arc sequence)
   ▼
candidate segments          pathsnap/candidates.py (KD-tree on 10 m arc samples, ≤ 8 arcs within 80 m)
   ▼
22 features / candidate     pathsnap/features.py (distance, heading, context, topology, gaps)
   ▼
feature selection           scripts/03_feature_selection.py (correlation, mutual info, permutation importance, ablation)
   ▼
binary classifier           pathsnap/models.py   P(candidate = true road segment)  [5 models]
   ▼
top-1 candidate per fix  →  metrics (pathsnap/metrics.py): top-1, macro-F1, log loss, continuity
```

## Repository layout

| Path | Contents |
|---|---|
| `pathsnap/` | Library modules listed above, plus `dataset.py` (loading cached tables) and `evaluation.py` (trip-level scoring) |
| `scripts/01_eda.py` | Dataset inventory, missing-value checks, time gaps, network connectivity, plots, Folium EDA maps |
| `scripts/02_build_dataset.py` | Candidate generation, feature engineering, label alignment → `data_cache/` |
| `scripts/03_feature_selection.py` | Feature correlation, mutual information, permutation importance, top-k ablation |
| `scripts/03_train_models.py` | 5 models, 5-fold GroupKFold CV, student-test evaluation, comparison plots |
| `results/eda/` | `inventory.csv`, `data_quality.csv`, `partition_summary.csv` |
| `results/candidate_summary.csv` | Per-trip candidate miss rate and label-alignment quality |
| `results/feature_selection/` | `feature_importance.csv`, `spearman_correlation.csv`, `ablation_topk.csv` |
| `results/model_comparison.csv`, `model_cv_folds.csv` | Model comparison (incl. hyperparameters) and per-fold CV results |
| `results/figures/` | EDA plots, feature-selection plots (`fs_*`), CV macro-F1, student-test comparison, log loss, confusion matrices, LightGBM training curve |
| `results/maps/` | Interactive Folium EDA maps (GPS, road network, reference route) |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Raw data is not committed. By default the code expects it at `../Student folder/Student folder/{train,student_test}`;
override with the `PATHSNAP_DATA` environment variable.

## Reproducing Phase 1

```bash
python scripts/02_build_dataset.py     # candidates + features + labels (~1 min, parallel)
python scripts/01_eda.py               # EDA tables, figures, Folium maps
python scripts/03_feature_selection.py # feature analysis + ablation (~5 min)
python scripts/03_train_models.py      # 5 models (~12 min)
```

Random seed: 42 everywhere (`pathsnap/config.py`).

## Feature selection (training trips only)

* Spearman |ρ| ≥ 0.9 pairs: dist / dist_minus_min / dist_ratio, dist_minus_min / ctx_minus_min, ctx_mean / ctx_max / ctx_wide_mean.
* Permutation importance (log-loss increase, fold-0 validation trips): head_cos_wide 0.195, ctx_minus_min 0.101,
  dist 0.049, head_cos 0.021, t_along 0.015 … dist_rank, ctx_mean, arc_len ≈ 0.
* Top-k ablation (LightGBM, CV folds 1-4, macro-F1): 3 → 0.922, 5 → 0.949, 8 → 0.950, 12 → 0.952, 16 → 0.953, 22 → 0.953.
* Decision: keep all 22 features (best score; the weak ones are cheap and redundant rather than harmful).

## Phase 1 results

Protocol: 5-fold `GroupKFold` grouped by trip on the 80 training trips; evaluation on all 19,160 fixes of the
10 student-testing trips. Top-1 = share of GPS fixes whose highest-scoring candidate is the reference segment.

| Model | Family | Hyperparameters | CV macro-F1 | Test top-1 | Test macro-F1 | Test log loss | Route continuity |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear | C=1, L2, standardised | 0.9440 ± 0.0120 | 0.8862 | 0.9340 | 0.1029 | 0.9726 |
| KNN | Instance-based | k=25, distance-weighted | 0.9424 ± 0.0130 | 0.8843 | 0.9328 | 0.1407 | 0.9670 |
| Decision Tree | Tree | depth 12, min leaf 20 | 0.9460 ± 0.0126 | 0.8888 | 0.9355 | 0.1032 | 0.9724 |
| Random Forest | Bagging | 150 trees, depth 18 | 0.9508 ± 0.0118 | 0.9059 | 0.9456 | 0.0742 | 0.9794 |
| **LightGBM** | Boosting | 400 trees, 63 leaves, lr 0.05 | **0.9509 ± 0.0128** | **0.9106** | **0.9483** | 0.0746 | **0.9810** |

LightGBM is selected (highest CV macro-F1, best student-test top-1 / macro-F1, fast to train) and will be tuned in Phase 2.
Fit check: Logistic Regression underfits (train 0.904 ≈ CV 0.903); Decision Tree and Random Forest overfit
somewhat (train 0.931 / 0.955 vs CV 0.907 / 0.915); LightGBM's validation loss levels off after ~150 rounds.
Candidates miss the true segment for only 0.36 % of student-test fixes, so the ceiling is ~99.6 %.

## Data-leakage safeguards

* No trip is split across partitions; cross-validation uses `GroupKFold` by `trip_id`.
* Scalers live inside model pipelines, so they are fitted on training folds only.
* Fixes more than 30 m from their aligned reference arc (unreliable labels, e.g. train trips 33 and 54) are excluded from training only.
* The evaluator-testing partition is not used anywhere in development.
