"""Central configuration: paths and pipeline constants.

Paths can be overridden with environment variables so the same code runs on
both laptops and on the evaluator-testing folder at demo time.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw data (not committed to git). Default = the folder handed out to students.
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT.parent / "Student folder" / "Student folder"))
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"

# Folder with the 10 reserved evaluation trips served by the API.
# At demo time point this to the evaluator folder; defaults to student_test for development.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", STUDENT_TEST_DIR))

CACHE_DIR = ROOT / "data_cache"          # derived feature tables (git-ignored)
ARTIFACT_DIR = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))  # final model + configs
RESULTS_DIR = ROOT / "results"           # csv / json experiment outputs
FIG_DIR = RESULTS_DIR / "figures"
MAP_DIR = RESULTS_DIR / "maps"

for _d in (CACHE_DIR, ARTIFACT_DIR, RESULTS_DIR, FIG_DIR, MAP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---- candidate generation ----
CAND_RADIUS_M = 80.0     # search radius around each GPS fix
CAND_MAX_K = 8           # keep at most K nearest arcs per fix
SAMPLE_SPACING_M = 10.0  # arc densification step for the KD-tree index
BBOX_MARGIN_M = 400.0    # network is cropped to track bbox + margin

# ---- feature context windows ----
HEADING_WIN = 2          # +-points used for heading / speed estimate
CTX_WIN = 3              # +-points used for local context distance
CTX_WIN_WIDE = 8
