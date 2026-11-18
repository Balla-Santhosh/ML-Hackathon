"""Deriving observation-level reference labels from the .route arc sequence.

The .route file is an ordered list of arc ids but it does NOT say which GPS fix
belongs to which arc. We align the GPS fixes to the route with a monotone
dynamic programme: fix i is assigned route position j(i) with j non-decreasing
in time, minimising the total point-to-segment distance. This respects the
order of travel, so a fix near a junction (or near a place visited twice) is
attached to the correct pass of the route.

This module is ONLY used for training labels and server-side scoring.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment


def align_to_route(xy: np.ndarray, node_xy: np.ndarray, arcs: np.ndarray, route: np.ndarray):
    """Return (gt_arc_global (n,), align_dist (n,), route_pos (n,))."""
    A = node_xy[arcs[route, 0]]
    B = node_xy[arcs[route, 1]]
    n, m = len(xy), len(route)
    bp = np.zeros((n, m), dtype=np.int16 if m < 32000 else np.int32)
    idx = np.arange(m)
    acc = None
    for i in range(n):
        d, _ = point_segment(xy[i][None, :], A, B)
        if acc is None:
            acc = d
            continue
        cm = np.minimum.accumulate(acc)
        arg = np.maximum.accumulate(np.where(acc <= cm, idx, 0))
        bp[i] = arg
        acc = d + cm
    pos = np.empty(n, np.int64)
    pos[-1] = int(np.argmin(acc))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = bp[i, pos[i]]
    dist, _ = point_segment(xy, A[pos], B[pos])
    return route[pos], dist, pos
