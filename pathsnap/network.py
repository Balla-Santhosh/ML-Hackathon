"""Road network around a trip: cropping, arc geometry and bounded Dijkstra."""
from __future__ import annotations

import heapq

import numpy as np

from .config import BBOX_MARGIN_M
from .geo import LocalProjection


class RoadNetwork:
    """Directed road graph cropped to the trip's bounding box.

    Arcs keep their *global* arc_id (row index in the .arcs file) so that the
    candidate -> road-segment correspondence is preserved end to end.
    """

    def __init__(self, nodes_ll, arcs, proj: LocalProjection, track_xy=None, margin=BBOX_MARGIN_M):
        self.proj = proj
        self.node_xy = proj.to_xy(nodes_ll[:, 0], nodes_ll[:, 1])
        self.n_arcs_total = len(arcs)
        A = self.node_xy[arcs[:, 0]]
        B = self.node_xy[arcs[:, 1]]
        keep = np.ones(len(arcs), bool)
        if track_xy is not None:
            lo = track_xy.min(0) - margin
            hi = track_xy.max(0) + margin
            amin, amax = np.minimum(A, B), np.maximum(A, B)
            keep = (amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) & (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1])
        self.arc_ids = np.flatnonzero(keep)                 # global ids of kept arcs
        self.afrom = arcs[keep, 0]
        self.ato = arcs[keep, 1]
        self.A = A[keep]
        self.B = B[keep]
        self.alen = np.linalg.norm(self.B - self.A, axis=1)
        self.abear = np.arctan2(self.B[:, 0] - self.A[:, 0], self.B[:, 1] - self.A[:, 1])
        self.g2l = {int(g): i for i, g in enumerate(self.arc_ids)}

        # CSR adjacency on kept arcs
        order = np.argsort(self.afrom, kind="stable")
        self._adj_to = self.ato[order]
        self._adj_len = self.alen[order]
        self._adj_from = self.afrom[order]
        nn = len(self.node_xy)
        self._ptr = np.searchsorted(self._adj_from, np.arange(nn + 1))
        self.out_deg = np.bincount(self.afrom, minlength=nn)
        self.in_deg = np.bincount(self.ato, minlength=nn)
        # reverse-twin lookup (u,v) -> arc with (v,u)
        pair = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(self.afrom, self.ato))}
        self.twin = np.array([pair.get((int(v), int(u)), -1) for u, v in zip(self.afrom, self.ato)], dtype=np.int64)
        self._cache: dict[int, tuple[float, dict]] = {}

    # ------------------------------------------------------------------
    def dijkstra(self, src: int, cutoff: float) -> dict:
        """Shortest network distance from node `src` to every node within `cutoff` metres.

        Results are cached per source and reused when the cached cutoff is large enough.
        """
        c = self._cache.get(src)
        if c is not None and c[0] >= cutoff:
            return c[1]
        dist = {src: 0.0}
        done = {}
        heap = [(0.0, src)]
        ptr, to, ln = self._ptr, self._adj_to, self._adj_len
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done[u] = d
            for k in range(ptr[u], ptr[u + 1]):
                v = int(to[k])
                nd = d + ln[k]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        if len(self._cache) > 50000:
            self._cache.clear()
        self._cache[src] = (cutoff, done)
        return done

    def route_distance(self, la: int, sa: float, lb: int, sb: float, cutoff: float, back_tol: float = 15.0) -> float:
        """Driving distance from offset sa on local arc la to offset sb on local arc lb.

        Returns np.inf if not reachable within cutoff.
        """
        if la == lb and sb >= sa - back_tol:
            return max(sb - sa, 0.0)
        rest = self.alen[la] - sa
        u, v = int(self.ato[la]), int(self.afrom[lb])
        if u == v:
            return rest + sb
        budget = cutoff - rest - sb
        if budget < 0:
            return np.inf
        d = self.dijkstra(u, budget).get(v)
        if d is not None and d > budget:
            return np.inf
        return np.inf if d is None else rest + d + sb

    def arc_polyline_ll(self, global_ids):
        """List of [[lat,lon],[lat,lon]] per global arc id (for maps)."""
        out = []
        for g in global_ids:
            l = self.g2l.get(int(g))
            if l is None:
                continue
            ll = self.proj.to_lonlat(np.vstack([self.A[l], self.B[l]]))
            out.append([[ll[0, 1], ll[0, 0]], [ll[1, 1], ll[1, 0]]])
        return out
