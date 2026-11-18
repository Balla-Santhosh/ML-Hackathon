"""Evaluation metrics (candidate-level, observation-level, route-level)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from .network import RoadNetwork


def argmax_decision(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """1 for the highest-scoring candidate of each (trip, obs), else 0."""
    key = df["trip_id"].astype(str).values + ":" + df["obs"].astype(str).values
    s = pd.Series(score, index=np.arange(len(df)))
    best = s.groupby(key).idxmax().values
    dec = np.zeros(len(df), np.int8)
    dec[best] = 1
    return dec


def candidate_metrics(df: pd.DataFrame, prob: np.ndarray) -> dict:
    """Candidate-level and observation-level metrics for a scored candidate table.

    Macro-F1 is computed on the binary candidate labels using the route-matching
    decision (top-1 candidate per fix = positive), so it is threshold-free and
    comparable across models whose probabilities are not calibrated.
    """
    y = df["y"].values
    dec = argmax_decision(df, prob)
    n_obs = df.groupby(["trip_id", "obs"]).ngroups
    top1 = (dec & y).sum() / n_obs
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return {
        "top1_acc": float(top1),
        "macro_f1": float(f1_score(y, dec, average="macro")),
        "macro_f1_thr05": float(f1_score(y, (prob >= 0.5).astype(int), average="macro")),
        "precision_pos": float(precision_score(y, dec, zero_division=0)),
        "recall_pos": float(recall_score(y, dec, zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def continuity(net: RoadNetwork, local_arcs: np.ndarray, offsets: np.ndarray, xy: np.ndarray) -> float:
    """Share of consecutive fixes whose matched segments form a drivable connection.

    Pair (i, i+1) is continuous if the network driving distance from the matched
    position at i to the one at i+1 is at most 2*straight-line distance + 50 m.
    """
    ok = 0
    n = len(local_arcs)
    for i in range(n - 1):
        gc = float(np.hypot(*(xy[i + 1] - xy[i])))
        cut = 2 * gc + 50.0
        d = net.route_distance(int(local_arcs[i]), float(offsets[i]), int(local_arcs[i + 1]), float(offsets[i + 1]), cut)
        ok += np.isfinite(d)
    return ok / max(n - 1, 1)


def collapse(seq):
    seq = list(seq)
    return [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]


def route_metrics(pred_arcs_global, ref_route, arc_len_global) -> dict:
    """Sequence-level comparison of the matched route with the reference route."""
    pred = collapse(pred_arcs_global)
    ps, rs = set(pred), set(int(a) for a in ref_route)
    inter = ps & rs
    L = lambda s: float(sum(arc_len_global[a] for a in s))
    lp, lr, li = L(ps), L(rs), L(inter)
    return {
        "route_precision": len(inter) / max(len(ps), 1),
        "route_recall": len(inter) / max(len(rs), 1),
        "route_precision_len": li / lp if lp else 0.0,
        "route_recall_len": li / lr if lr else 0.0,
        "route_mismatch_fraction": (lp - li + lr - li) / lr if lr else np.nan,
        "n_pred_segments": len(pred),
    }
