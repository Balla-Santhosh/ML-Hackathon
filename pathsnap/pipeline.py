"""End-to-end inference pipeline used by the API (no reference labels involved)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from .config import ARTIFACT_DIR
from .dataset import to_matrix
from .features import FEATURES, build_features, prepare_trip
from .hmm import HMMParams, viterbi
from .io import load_trip
from .metrics import collapse


class Predictor:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        self.model = joblib.load(artifact_dir / "model.joblib")
        self.meta = json.load(open(artifact_dir / "model_meta.json"))
        hp = artifact_dir / "hmm_params.json"
        self.hmm_params = HMMParams(**json.load(open(hp))) if hp.exists() else HMMParams()
        assert self.meta["features"] == FEATURES, "feature list mismatch between artifact and code"

    def predict_trip(self, trip_dir: Path) -> dict:
        timings = {}
        t0 = time.time()
        trip = load_trip(trip_dir, with_route=False)          # reference route is NOT loaded here
        proj, xy, net, cand, dist, tpar = prepare_trip(trip)
        df = build_features(trip, xy, net, cand, dist, tpar)
        timings["features_s"] = time.time() - t0

        t1 = time.time()
        prob = self.model.predict_proba(df[FEATURES].values)[:, 1]
        timings["classifier_s"] = time.time() - t1
        n = len(xy)
        P = to_matrix(df, prob, n, cand.shape[1])
        P[cand < 0] = np.nan
        clf_slots = np.nanargmax(np.where(np.isnan(P), -1, P), axis=1)

        t2 = time.time()
        hmm_slots, breaks = viterbi(net, xy, trip.track[:, 2], cand, tpar, P, self.hmm_params)
        timings["hmm_s"] = time.time() - t2
        timings["total_s"] = time.time() - t0

        rows = np.arange(n)
        hmm_local = cand[rows, hmm_slots]
        clf_local = cand[rows, clf_slots]
        return {
            "trip": trip, "proj": proj, "xy": xy, "net": net, "cand": cand, "tpar": tpar,
            "clf_local": clf_local, "hmm_local": hmm_local,
            "clf_offset": np.nan_to_num(tpar[rows, clf_slots]) * net.alen[clf_local],
            "hmm_offset": np.nan_to_num(tpar[rows, hmm_slots]) * net.alen[hmm_local],
            "clf_arcs": net.arc_ids[clf_local], "hmm_arcs": net.arc_ids[hmm_local],
            "hmm_conf": P[rows, hmm_slots], "breaks": breaks, "timings": timings,
            "route_arcs": collapse(net.arc_ids[hmm_local].tolist()),
        }
