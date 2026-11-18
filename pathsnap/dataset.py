"""Loading the cached feature tables for experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CACHE_DIR, CAND_MAX_K
from .features import FEATURES

LABEL_MAX_ALIGN_M = 30.0   # fixes further than this from the aligned reference arc have unreliable labels
TRAIN_STRIDE = 2           # 1 Hz fixes are highly autocorrelated; keep every 2nd fix for training


def load_partition(name: str, reliable_only: bool = False, stride: int = 1) -> pd.DataFrame:
    f = pd.read_parquet(CACHE_DIR / f"features_{name}.parquet")
    o = pd.read_parquet(CACHE_DIR / f"obs_{name}.parquet", columns=["trip_id", "obs", "align_dist", "gt_arc", "gt_in_cands"])
    f = f.merge(o, on=["trip_id", "obs"], how="left")
    if reliable_only:
        f = f[f.align_dist <= LABEL_MAX_ALIGN_M]
    if stride > 1:
        f = f[f.obs % stride == 0]
    return f.reset_index(drop=True)


def load_train():
    return load_partition("train", reliable_only=True, stride=TRAIN_STRIDE)


def load_student_test():
    return load_partition("student_test")


def to_matrix(df_trip: pd.DataFrame, values: np.ndarray, n_obs: int, k: int = CAND_MAX_K, fill=np.nan):
    """Scatter per-row values of one trip back into an (n_obs, K) slot matrix."""
    M = np.full((n_obs, k), fill, dtype=float)
    M[df_trip["obs"].values, df_trip["slot"].values] = values
    return M


__all__ = ["FEATURES", "load_train", "load_student_test", "load_partition", "to_matrix", "LABEL_MAX_ALIGN_M"]
