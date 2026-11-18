"""HMM map matching decoded with the Viterbi algorithm.

Hidden state at time i : one of the K candidate road segments (arc + offset along it)
                         generated for GPS fix i.
Emission  log P(z_i | s) : alpha * log p_clf(s)  -- the tuned classifier's probability
                         that candidate s is the true segment for fix i.
Transition log P(s'|s)   : -|d_route(s, s') - d_gc(z_i, z_i+1)| / beta_i
                         (Newson & Krumm 2009): the driving distance through the network
                         between the two projected positions should match the
                         straight-line distance between the fixes. Pairs not connected
                         within the search cutoff get -inf (invalid transition).
Gaps                     : beta_i = beta * sqrt(max(dt,1)) and the cutoff grows with the
                         straight-line distance, so long time gaps are allowed more slack.
Breaks                   : if no candidate at i+1 is reachable from any state at i, the
                         chain is broken and restarted at i+1 from the best previous score
                         (emission only), instead of failing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .network import RoadNetwork


@dataclass
class HMMParams:
    beta: float = 5.0
    alpha: float = 1.0
    cutoff_factor: float = 3.0
    cutoff_min: float = 150.0
    prob_floor: float = 1e-4

    def to_dict(self):
        return asdict(self)


def transition_matrix(net: RoadNetwork, ca, sa, cb, sb, gc, dt, p: HMMParams):
    cutoff = max(p.cutoff_factor * gc, gc + p.cutoff_min)
    beta = p.beta * np.sqrt(max(dt, 1.0))
    for la in set(int(x) for x in ca):
        net.dijkstra(int(net.ato[la]), cutoff)          # warm cache once per source node
    T = np.full((len(ca), len(cb)), -np.inf)
    for i, (la, a_off) in enumerate(zip(ca, sa)):
        for j, (lb, b_off) in enumerate(zip(cb, sb)):
            d = net.route_distance(int(la), float(a_off), int(lb), float(b_off), cutoff)
            if np.isfinite(d):
                T[i, j] = -abs(d - gc) / beta
    return T


def viterbi(net: RoadNetwork, xy, t, cand, tpar, prob, p: HMMParams = HMMParams()):
    """Return (chosen slot per fix (n,), number of chain breaks)."""
    n, k = cand.shape
    off = np.where(cand >= 0, tpar * net.alen[np.where(cand >= 0, cand, 0)], 0.0)
    logE = p.alpha * np.log(np.clip(np.nan_to_num(prob, nan=0.0), p.prob_floor, 1.0))
    logE[cand < 0] = -np.inf

    valid0 = np.flatnonzero(cand[0] >= 0)
    score = logE[0, valid0].copy()
    slots = [valid0]
    back = []
    breaks = 0
    for i in range(1, n):
        vb = np.flatnonzero(cand[i] >= 0)
        va = slots[-1]
        gc = float(np.hypot(*(xy[i] - xy[i - 1])))
        dt = float(t[i] - t[i - 1])
        T = transition_matrix(net, cand[i - 1, va], off[i - 1, va], cand[i, vb], off[i, vb], gc, dt, p)
        tot = score[:, None] + T
        bp = np.argmax(tot, axis=0)
        new = tot[bp, np.arange(len(vb))]
        if not np.isfinite(new).any():               # disconnected: break the chain
            breaks += 1
            bp = np.full(len(vb), int(np.argmax(score)))
            new = np.full(len(vb), score.max())
        score = new + logE[i, vb]
        score -= score.max()                          # keep numbers bounded
        slots.append(vb)
        back.append(bp)
    path = np.empty(n, np.int64)
    j = int(np.argmax(score))
    path[-1] = slots[-1][j]
    for i in range(n - 1, 0, -1):
        j = int(back[i - 1][j])
        path[i - 1] = slots[i - 1][j]
    return path, breaks
