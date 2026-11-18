"""Trip-level evaluation: classifier-only vs classifier + HMM on whole trips."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import to_matrix
from .features import prepare_trip
from .io import load_route, load_trip
from .labels import align_to_route
from .metrics import continuity, route_metrics


class TripContext:
    """Everything needed to decode / score one trip (network, candidates, labels)."""

    def __init__(self, trip_dir: Path):
        trip = load_trip(trip_dir, with_route=False)
        self.trip = trip
        self.proj, self.xy, self.net, self.cand, self.dist, self.tpar = prepare_trip(trip)
        self.t = trip.track[:, 2]
        self.n = len(self.xy)
        self.offsets_all = np.where(self.cand >= 0, self.tpar * self.net.alen[np.where(self.cand >= 0, self.cand, 0)], 0.0)
        # server-side reference (never used by the predictor)
        self.route = load_route(trip_dir)
        self.gt, self.align_dist, _ = align_to_route(self.xy, self.net.node_xy, trip.arcs, self.route)
        nxy = self.net.node_xy
        self.arc_len_global = np.linalg.norm(nxy[trip.arcs[:, 1]] - nxy[trip.arcs[:, 0]], axis=1)

    def score(self, slots: np.ndarray, label: str) -> dict:
        rows = np.arange(self.n)
        local = self.cand[rows, slots]
        glob = self.net.arc_ids[local]
        offs = self.offsets_all[rows, slots]
        out = {
            f"{label}_top1": float(np.mean(glob == self.gt)),
            f"{label}_continuity": continuity(self.net, local, offs, self.xy),
        }
        out.update({f"{label}_{k}": v for k, v in route_metrics(glob, self.route, self.arc_len_global).items()})
        return out


def decode_trip(ctx: TripContext, df_trip: pd.DataFrame, prob: np.ndarray, params=None, do_hmm=True):
    P = to_matrix(df_trip, prob, ctx.n, ctx.cand.shape[1])
    P[ctx.cand < 0] = np.nan
    clf_slots = np.nanargmax(np.where(np.isnan(P), -1, P), axis=1)
    res = {"trip_id": ctx.trip.trip_id, "n_obs": ctx.n}
    res.update(ctx.score(clf_slots, "clf"))
    if do_hmm:
        from .hmm import HMMParams, viterbi  # sequential decoder (only needed for route reconstruction)
        params = params or HMMParams()
        t0 = time.time()
        hmm_slots, breaks = viterbi(ctx.net, ctx.xy, ctx.t, ctx.cand, ctx.tpar, P, params)
        res["hmm_secs"] = time.time() - t0
        res["hmm_breaks"] = breaks
        res.update(ctx.score(hmm_slots, "hmm"))
        return res, clf_slots, hmm_slots
    return res, clf_slots, None
