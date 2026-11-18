# PathSnap – Run Guide (single laptop or two laptops)

This guide explains how to set up and run every part of PathSnap: the experiment scripts, the API and the UI.
Everything can run on **one laptop** (API and UI both on `localhost`) or on **two laptops** (as in the final evaluation).

---

## 1. Prerequisites

* Windows 10/11, macOS or Linux, Python **3.10+** (developed on Python 3.13).
* Git.
* About 1 GB free disk space (raw data ~100 MB, cached features ~115 MB, Python packages).
* Internet access for map tiles in the Folium maps / UI (the GPS points and routes still draw offline).

## 2. Folder layout

The raw data is **not** in the repository. Place the clone next to the data folders like this:

```
Hackathon/                         <- any parent folder
├── ML-Hackathon/                  <- this repository (git clone)
├── Student folder/
│   └── Student folder/
│       ├── train/<trip_id>/        80 trips: .track .nodes .arcs .route
│       └── student_test/<trip_id>/ 10 trips
└── evaluator_test/
    └── evaluator_test/<trip_id>/   10 reserved evaluation trips
```

With this layout no configuration is needed. Otherwise use the environment variables in Section 8.

## 3. One-time setup

```bash
git clone https://github.com/Balla-Santhosh/ML-Hackathon.git
cd ML-Hackathon
python -m venv .venv
```

Activate the virtual environment:

| Shell | Command |
|---|---|
| PowerShell | `.venv\Scripts\Activate.ps1` |
| Command Prompt | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

Then install the dependencies:

```bash
pip install -r requirements.txt
```

Run every command below from the `ML-Hackathon` folder with the environment activated.

## 4. Running only the demo (API + UI)

The trained model (`artifacts/model.joblib`) and decoder settings (`artifacts/hmm_params.json`) are committed,
so **no training is needed** to run the system.

### 4a. Everything on one laptop

**Terminal 1 – API**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete`. The first log line shows the loaded model and the folder of trips being served.
Check it in a browser:

* http://localhost:8000/health → `"status": "ok"` and `"n_eval_trips": 10`
* http://localhost:8000/docs → interactive API documentation (you can call `/predict` from here)

**Terminal 2 – UI** (activate the environment again in the new terminal)

```bash
streamlit run ui/app.py -- --api http://localhost:8000
```

Streamlit opens http://localhost:8501. In the left sidebar, choose a trip and press **Reconstruct route**.
The page shows top-1 accuracy (HMM and classifier-only), route continuity, route precision/recall, processing
time and an interactive map (GPS points, road network, predicted route, optional classifier-only segments).
Terminal 1 logs every request, e.g.

```
INFO | Inference request for trip 00000015
INFO | trip 00000015: 6854 fixes | top1 HMM=0.9002 clf=0.8321 | continuity=0.9987 | breaks=0 | 2.10s
INFO | POST /predict from 127.0.0.1 -> 200 (2650 ms)
```

Stop either program with `Ctrl+C`.

### 4b. Two laptops (final evaluation setup)

| | Laptop 1 – API | Laptop 2 – UI |
|---|---|---|
| Needs | Full repo, `evaluator_test` data, `pip install -r requirements.txt` | Repo (only `ui/app.py` is used), `pip install streamlit folium requests` |
| Command | `uvicorn api.main:app --host 0.0.0.0 --port 8000` | `streamlit run ui/app.py -- --api http://<laptop1-ip>:8000` |

1. Connect both laptops to the same network (a phone hotspot works).
2. On Laptop 1, find its IP with `ipconfig` (Windows; look for "IPv4 Address") or `ip addr` / `ifconfig`.
3. Allow inbound TCP port 8000 on Laptop 1. When Windows asks on first start, allow Python on **private** networks,
   or run this in an administrator PowerShell:
   `New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`
4. From Laptop 2, check `http://<laptop1-ip>:8000/health` in a browser, then start the UI.
5. During evaluation, keep Laptop 1's terminal visible: evaluators want to see the request logs.

## 5. Reproducing all experiments (optional)

Run in this order. Every script prints progress and writes to `results/`. All use random seed 42, so rerunning gives the same numbers.

| # | Command | Time* | Produces |
|---|---|---|---|
| 1 | `python scripts/02_build_dataset.py` | ~20 s | `data_cache/` candidates, features, labels (**required by 3–6**) |
| 2 | `python scripts/01_eda.py` | ~40 s | `results/eda/`, `figures/eda_*.png`, `maps/eda_*.html` |
| 3 | `python scripts/03_feature_selection.py` | ~5 min | `results/feature_selection/`, `figures/fs_*.png` |
| 4 | `python scripts/03_train_models.py` | ~35 min | `results/model_comparison.csv`, `model_cv_folds.csv`, comparison figures |
| 5 | `python scripts/04_bayes_opt.py` | ~30 min | `results/bayes_opt_*`, **overwrites `artifacts/model.joblib`** |
| 6 | `python scripts/05_hmm_decoder.py` | ~10 min | `results/hmm_*`, `maps/route_student_test_*.html`, **overwrites `artifacts/hmm_params.json`** |
| 7 | `python scripts/06_report_tables.py` | seconds | `report/results_tables.md` |

\*Measured on the development laptop; times vary with CPU.

Step 4 trains only some models if you pass their names, e.g. `python scripts/03_train_models.py LogisticRegression LightGBM`
(this rewrites `model_comparison.csv` with only those models).
Step 5 takes an optional model name and trial count: `python scripts/04_bayes_opt.py LightGBM 30`.

Steps 1–7 never read the `evaluator_test` folder; it is used only by the API.

## 6. Where to look at results without running anything

| What | File |
|---|---|
| Dataset summary | `results/eda/partition_summary.csv`, `inventory.csv`, `data_quality.csv` |
| EDA plots | `results/figures/eda_*.png` |
| Interactive maps | `results/maps/*.html` (open in a browser) |
| Feature selection | `results/feature_selection/*.csv`, `results/figures/fs_*.png` |
| Model comparison | `results/model_comparison.csv`, `results/figures/cv_macro_f1.png`, `student_test_top1_f1.png`, `confusion_matrices.png`, `training_curves.png` |
| Bayesian optimisation | `results/bayes_opt_summary.json`, `bayes_opt_trials.csv`, `figures/bayes_opt_*.png` |
| HMM vs classifier | `results/hmm_student_test.csv`, `hmm_summary.json`, `figures/hmm_vs_classifier.png` |
| All tables in one file | `report/results_tables.md` |

## 7. Calling the API without the UI

```bash
curl http://localhost:8000/trips
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\": \"00000015\"}"
```

In PowerShell use `curl.exe` instead of `curl`, or open http://localhost:8000/docs and use **Try it out**.
The response contains the predicted arc per GPS point, the route arcs and geometry, metrics and timings. It never
contains the reference route.

## 8. Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `PATHSNAP_DATA` | `../Student folder/Student folder` | Folder containing `train/` and `student_test/` |
| `PATHSNAP_EVAL_DIR` | `../evaluator_test/evaluator_test` if present, else `student_test` | Trips served by the API |
| `PATHSNAP_ARTIFACTS` | `artifacts/` | Model and decoder settings loaded by the API |
| `PATHSNAP_API_URL` | `http://localhost:8000` | API address used by the UI (same as `--api`) |

Set a variable in the **same terminal** before running a command:

| Shell | Example |
|---|---|
| PowerShell | `$env:PATHSNAP_EVAL_DIR="D:\data\evaluator_test\evaluator_test"` |
| Command Prompt | `set PATHSNAP_EVAL_DIR=D:\data\evaluator_test\evaluator_test` |
| macOS / Linux | `export PATHSNAP_EVAL_DIR=/data/evaluator_test/evaluator_test` |

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError ... Student folder` | Use the folder layout in Section 2 or set `PATHSNAP_DATA` |
| `FileNotFoundError ... features_train.parquet` | Run `python scripts/02_build_dataset.py` first |
| `ModuleNotFoundError` (lightgbm, folium, pyarrow, …) | Activate the virtual environment and run `pip install -r requirements.txt` |
| `/health` shows `n_eval_trips` other than 10 | The API is not seeing `evaluator_test`; check the layout or set `PATHSNAP_EVAL_DIR` |
| UI says "Cannot reach API" | Start the API first; check the URL in the UI sidebar; on two laptops check the IP and firewall (Section 4b) |
| `[Errno 10048]` / "address already in use" | Another program uses the port: stop it, or start the API with `--port 8001` and point the UI at it |
| `uvicorn` or `streamlit` "not recognized" | Use `python -m uvicorn ...` / `python -m streamlit ...` |
| PowerShell refuses to activate the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt |
| Map background is blank | No internet for map tiles; routes and points still draw |
| Experiment step is slow | Close other heavy programs; the timings in Section 5 are for a typical laptop |
