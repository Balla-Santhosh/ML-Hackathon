"""Turning (GPS fix, candidate arc) pairs into a feature table.

Every row = one candidate road segment for one GPS observation. The classifier
learns P(candidate is the true segment | features) and the HMM later uses those
probabilities as emission scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import CandidateIndex
from .config import CAND_MAX_K, CAND_RADIUS_M, CTX_WIN, CTX_WIN_WIDE, HEADING_WIN
from .geo import LocalProjection, point_segment
from .io import Trip
from .network import RoadNetwork

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands", "gap12",
    "head_cos", "head_cos_wide", "speed", "turn_rate",
    "arc_len", "t_along", "at_endpoint",
    "ctx_mean", "ctx_max", "ctx_wide_mean", "ctx_minus_min", "ctx_rank",
    "out_deg_to", "in_deg_from", "twin_in_cands", "dt_prev",
]


def _heading(xy, t, w):
    n = len(xy)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    d = xy[i1] - xy[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    return np.arctan2(d[:, 0], d[:, 1]), np.hypot(d[:, 0], d[:, 1]) / dt


def _ctx_dist(xy, A, B, cand, offsets):
    """Mean / max distance of neighbouring fixes (i+o) to each candidate arc of fix i."""
    n, k = cand.shape
    c = np.where(cand >= 0, cand, 0)
    acc = np.zeros((n, k))
    mx = np.zeros((n, k))
    for o in offsets:
        j = np.clip(np.arange(n) + o, 0, n - 1)
        d, _ = point_segment(xy[j][:, None, :], A[c], B[c])
        acc += d
        mx = np.maximum(mx, d)
    return acc / len(offsets), mx


def prepare_trip(trip: Trip):
    """Projection, cropped network and candidate set for a trip (no labels used)."""
    proj = LocalProjection.for_track(trip.track)
    xy = proj.to_xy(trip.track[:, 0], trip.track[:, 1])
    net = RoadNetwork(trip.nodes, trip.arcs, proj, track_xy=xy)
    index = CandidateIndex(net)
    cand, dist, tpar = index.query(xy, CAND_RADIUS_M, CAND_MAX_K)
    return proj, xy, net, cand, dist, tpar


def build_features(trip: Trip, xy, net: RoadNetwork, cand, dist, tpar) -> pd.DataFrame:
    t = trip.track[:, 2]
    n, k = cand.shape
    valid = cand >= 0
    c = np.where(valid, cand, 0)

    head, speed = _heading(xy, t, HEADING_WIN)
    head_w, _ = _heading(xy, t, 3 * HEADING_WIN)
    # turn rate: heading change between backward and forward displacement
    nidx = np.arange(n)
    back = xy[nidx] - xy[np.clip(nidx - HEADING_WIN, 0, n - 1)]
    fwd = xy[np.clip(nidx + HEADING_WIN, 0, n - 1)] - xy[nidx]
    turn = np.abs(np.angle(np.exp(1j * (np.arctan2(fwd[:, 0], fwd[:, 1]) - np.arctan2(back[:, 0], back[:, 1])))))

    abear = net.abear[c]
    head_cos = np.cos(head[:, None] - abear)
    head_cos_w = np.cos(head_w[:, None] - abear)

    dmin = np.where(valid, dist, np.inf).min(1, keepdims=True)
    ds = np.sort(np.where(valid, dist, np.inf), axis=1)
    gap12 = np.where(np.isfinite(ds[:, 1]), ds[:, 1] - ds[:, 0], CAND_RADIUS_M)
    n_c = valid.sum(1)

    offs = [o for o in range(-CTX_WIN, CTX_WIN + 1)]
    offs_w = [o for o in range(-CTX_WIN_WIDE, CTX_WIN_WIDE + 1, 2)]
    ctx_mean, ctx_max = _ctx_dist(xy, net.A, net.B, cand, offs)
    ctx_wide, _ = _ctx_dist(xy, net.A, net.B, cand, offs_w)
    ctx_mean_v = np.where(valid, ctx_mean, np.inf)
    ctx_rank = np.argsort(np.argsort(ctx_mean_v, axis=1), axis=1)
    ctx_min = ctx_mean_v.min(1, keepdims=True)

    twin = net.twin[c]
    twin_in = np.zeros((n, k), bool)
    for j in range(k):
        twin_in[:, j] = ((cand == twin[:, j][:, None]) & valid).any(1) & (twin[:, j] >= 0)

    dt_prev = np.r_[1.0, np.diff(t)]

    feats = {
        "dist": dist,
        "dist_rank": np.broadcast_to(np.arange(k), (n, k)),
        "dist_minus_min": dist - dmin,
        "dist_ratio": (dist + 1.0) / (dmin + 1.0),
        "n_cands": np.broadcast_to(n_c[:, None], (n, k)),
        "gap12": np.broadcast_to(gap12[:, None], (n, k)),
        "head_cos": head_cos,
        "head_cos_wide": head_cos_w,
        "speed": np.broadcast_to(speed[:, None], (n, k)),
        "turn_rate": np.broadcast_to(turn[:, None], (n, k)),
        "arc_len": net.alen[c],
        "t_along": tpar,
        "at_endpoint": ((tpar <= 1e-3) | (tpar >= 1 - 1e-3)).astype(float),
        "ctx_mean": ctx_mean,
        "ctx_max": ctx_max,
        "ctx_wide_mean": ctx_wide,
        "ctx_minus_min": ctx_mean - ctx_min,
        "ctx_rank": ctx_rank,
        "out_deg_to": net.out_deg[net.ato[c]],
        "in_deg_from": net.in_deg[net.afrom[c]],
        "twin_in_cands": twin_in.astype(float),
        "dt_prev": np.broadcast_to(dt_prev[:, None], (n, k)),
    }
    rows_obs, rows_k = np.nonzero(valid)
    df = pd.DataFrame({name: np.asarray(v, dtype=float)[rows_obs, rows_k] for name, v in feats.items()})
    df.insert(0, "slot", rows_k)
    df.insert(0, "arc_local", cand[rows_obs, rows_k])
    df.insert(0, "arc_id", net.arc_ids[cand[rows_obs, rows_k]])
    df.insert(0, "obs", rows_obs)
    df.insert(0, "trip_id", trip.trip_id)
    return df
