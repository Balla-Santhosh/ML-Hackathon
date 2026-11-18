"""Candidate road-segment generation for every GPS fix."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .config import CAND_MAX_K, CAND_RADIUS_M, SAMPLE_SPACING_M
from .geo import point_segment
from .network import RoadNetwork


class CandidateIndex:
    """KD-tree over densified arc samples of a cropped network."""

    def __init__(self, net: RoadNetwork, spacing: float = SAMPLE_SPACING_M):
        self.net = net
        self.spacing = spacing
        nseg = np.maximum(1, np.ceil(net.alen / spacing)).astype(np.int64)
        cnt = nseg + 1
        owner = np.repeat(np.arange(len(nseg)), cnt)
        start = np.repeat(np.cumsum(cnt) - cnt, cnt)
        t = (np.arange(owner.size) - start) / nseg[owner]
        S = net.A[owner] + t[:, None] * (net.B - net.A)[owner]
        self.owner = owner
        self.tree = cKDTree(S)

    def query(self, xy: np.ndarray, radius: float = CAND_RADIUS_M, k: int = CAND_MAX_K):
        """Return (cand_local, dist, tpar), each (n,k); padding = -1 / inf / nan."""
        net = self.net
        n, L = len(xy), len(net.alen)
        lists = self.tree.query_ball_point(xy, r=radius + self.spacing / 2 + 1e-6, workers=-1)
        lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=n)
        pi = np.repeat(np.arange(n), lens)
        si = np.fromiter((s for x in lists for s in x), dtype=np.int64, count=int(lens.sum()))
        key = np.unique(pi * L + self.owner[si])
        pi, al = key // L, key % L
        d, t = point_segment(xy[pi], net.A[al], net.B[al])
        m = d <= radius
        pi, al, d, t = pi[m], al[m], d[m], t[m]

        # fallback: fixes with no arc inside the radius -> take nearest arcs anyway
        have = np.zeros(n, bool)
        have[pi] = True
        miss = np.flatnonzero(~have)
        if len(miss):
            kk = min(64, len(self.owner))
            _, idx = self.tree.query(xy[miss], k=kk, workers=-1)
            idx = np.atleast_2d(idx)
            fp = np.repeat(miss, idx.shape[1])
            key2 = np.unique(fp * L + self.owner[idx.ravel()])
            fp, fa = key2 // L, key2 % L
            fd, ft = point_segment(xy[fp], net.A[fa], net.B[fa])
            pi, al = np.r_[pi, fp], np.r_[al, fa]
            d, t = np.r_[d, fd], np.r_[t, ft]

        order = np.lexsort((d, pi))
        pi, al, d, t = pi[order], al[order], d[order], t[order]
        first = np.searchsorted(pi, pi, side="left")
        rank = np.arange(len(pi)) - first
        m = rank < k
        pi, al, d, t, rank = pi[m], al[m], d[m], t[m], rank[m]

        cand = np.full((n, k), -1, np.int64)
        dist = np.full((n, k), np.inf)
        tpar = np.full((n, k), np.nan)
        cand[pi, rank] = al
        dist[pi, rank] = d
        tpar[pi, rank] = t
        return cand, dist, tpar
