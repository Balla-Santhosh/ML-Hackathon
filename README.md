# PathSnap – Map-Matching & Route Reconstruction

Noisy GPS tracks are matched to a road network. **Phase 1 (this commit)** covers data understanding and the
machine-learning pipeline: raw-data processing, exploratory analysis, label construction, candidate generation,
feature engineering, and a first baseline comparison of two classifiers (Logistic Regression, Decision Tree).

*Phase 2 will add: more model families and the full comparison, Bayesian hyperparameter optimisation,
route reconstruction with HMM/Viterbi, the FastAPI backend and the evaluator UI.*

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
binary classifier           pathsnap/models.py   P(candidate = true road segment)  [Phase 1: 2 baselines]
   ▼
top-1 candidate per fix  →  metrics (pathsnap/metrics.py): top-1, macro-F1, log loss, continuity
```

## Repository layout

| Path | Contents |
|---|---|
| `pathsnap/` | Library modules listed above, plus `dataset.py` (loading cached tables) and `evaluation.py` (trip-level scoring) |
| `scripts/01_eda.py` | Dataset inventory, missing-value checks, time gaps, network connectivity, plots, Folium EDA maps |
| `scripts/02_build_dataset.py` | Candidate generation, feature engineering, label alignment → `data_cache/` |
| `scripts/03_train_models.py` | Baseline models, 5-fold GroupKFold CV, student-test evaluation, comparison plots |
| `results/eda/` | `inventory.csv`, `data_quality.csv`, `partition_summary.csv` |
| `results/candidate_summary.csv` | Per-trip candidate miss rate and label-alignment quality |
| `results/model_comparison.csv`, `model_cv_folds.csv` | Model comparison and per-fold CV results |
| `results/figures/` | EDA plots, CV macro-F1, student-test comparison, log loss, confusion matrices |
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
python scripts/03_train_models.py      # baseline models (~5 min)
```

Random seed: 42 everywhere (`pathsnap/config.py`).

## Phase 1 results (baselines)

Protocol: 5-fold `GroupKFold` grouped by trip on the 80 training trips; evaluation on all 19,160 fixes of the
10 student-testing trips. Top-1 = share of GPS fixes whose highest-scoring candidate is the reference segment.

| Model | Family | CV macro-F1 | Student-test top-1 | Student-test macro-F1 | Student-test log loss | Route continuity |
|---|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.9440 ± 0.0120 | 0.8862 | 0.9340 | 0.1029 | 0.9726 |
| Decision Tree | Tree | 0.9460 ± 0.0126 | 0.8888 | 0.9355 | 0.1032 | 0.9724 |

Candidate generation misses the true segment for only 0.36 % of student-test fixes, so the ceiling is ~99.6 %.
The ~11 % gap between the baselines and that ceiling is expected to come from junctions and two-way roads; Phase 2
targets it with stronger nonlinear models and with HMM decoding that enforces road connectivity.
The Decision Tree's training top-1 (0.931) exceeds its CV top-1 (0.907), showing mild overfitting; Logistic
Regression has no such gap (0.904 vs 0.903), which suggests it underfits.

## Data-leakage safeguards

* No trip is split across partitions; cross-validation uses `GroupKFold` by `trip_id`.
* Scalers live inside model pipelines, so they are fitted on training folds only.
* Fixes more than 30 m from their aligned reference arc (unreliable labels, e.g. train trips 33 and 54) are excluded from training only.
* The evaluator-testing partition is not used anywhere in development.
