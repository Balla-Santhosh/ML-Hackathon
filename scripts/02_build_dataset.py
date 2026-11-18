"""Build the candidate/feature tables for the training and student-testing partitions.

Output (data_cache/):
  features_train.parquet, features_student_test.parquet  -- one row per (fix, candidate)
  obs_train.parquet, obs_student_test.parquet            -- one row per GPS fix (label info)
and results/candidate_summary.csv (candidate miss rate, label alignment quality).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import CACHE_DIR, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR  # noqa: E402
from pathsnap.features import build_features, prepare_trip  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.labels import align_to_route  # noqa: E402


def process(trip_dir):
    t0 = time.time()
    trip = load_trip(trip_dir, with_route=True)
    proj, xy, net, cand, dist, tpar = prepare_trip(trip)
    df = build_features(trip, xy, net, cand, dist, tpar)
    gt, adist, _ = align_to_route(xy, net.node_xy, trip.arcs, trip.route)
    df["y"] = (df["arc_id"].values == gt[df["obs"].values]).astype(np.int8)
    hit = np.zeros(len(xy), bool)
    hit[df.loc[df.y == 1, "obs"].values] = True
    obs = pd.DataFrame({
        "trip_id": trip.trip_id, "obs": np.arange(len(xy)), "t": trip.track[:, 2],
        "lon": trip.track[:, 0], "lat": trip.track[:, 1],
        "gt_arc": gt, "align_dist": adist, "gt_in_cands": hit,
        "n_cands": (cand >= 0).sum(1), "nearest_dist": dist[:, 0],
    })
    summ = {
        "trip_id": trip.trip_id, "n_obs": len(xy), "n_rows": len(df),
        "arcs_total": net.n_arcs_total, "arcs_cropped": len(net.arc_ids),
        "route_len": len(trip.route), "miss_rate": 1 - hit.mean(),
        "align_dist_median": float(np.median(adist)), "align_dist_p95": float(np.percentile(adist, 95)),
        "secs": time.time() - t0, **trip.cleaning,
    }
    return df, obs, summ


def main():
    summaries = []
    for name, part in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
        dirs = list_trips(part)
        print(f"{name}: {len(dirs)} trips")
        out = Parallel(n_jobs=-1, verbose=5)(delayed(process)(d) for d in dirs)
        feats = pd.concat([o[0] for o in out], ignore_index=True)
        obs = pd.concat([o[1] for o in out], ignore_index=True)
        for o in out:
            o[2]["partition"] = name
            summaries.append(o[2])
        feats.to_parquet(CACHE_DIR / f"features_{name}.parquet", index=False)
        obs.to_parquet(CACHE_DIR / f"obs_{name}.parquet", index=False)
        print(f"  rows={len(feats):,} obs={len(obs):,} pos_rate={feats.y.mean():.3f} "
              f"miss_rate={1 - obs.gt_in_cands.mean():.4f}")
    pd.DataFrame(summaries).to_csv(RESULTS_DIR / "candidate_summary.csv", index=False)


if __name__ == "__main__":
    main()
