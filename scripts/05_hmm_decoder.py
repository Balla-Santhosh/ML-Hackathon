"""HMM / Viterbi route reconstruction: decoder tuning and classifier-only vs HMM comparison.

1. Hold out 12 training trips, fit the tuned classifier on the remaining 68 and
   predict every fix of the held-out trips (honest, out-of-sample probabilities).
2. Grid-search decoder parameters (beta, alpha) on those held-out training trips.
3. Evaluate the final classifier (+ chosen decoder) on the 10 student-testing trips:
   classifier-only vs classifier+HMM, pointwise top-1, continuity and route-level metrics.
4. Folium maps for every student-testing trip.
Outputs: results/hmm_grid.csv, results/hmm_student_test.csv, results/hmm_summary.json,
         results/figures/hmm_*.png, results/maps/route_<trip>.html, artifacts/hmm_params.json
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import ARTIFACT_DIR, CACHE_DIR, FIG_DIR, MAP_DIR, RESULTS_DIR, SEED, STUDENT_TEST_DIR, TRAIN_DIR  # noqa: E402
from pathsnap.dataset import FEATURES, load_partition, load_student_test, load_train  # noqa: E402
from pathsnap.evaluation import TripContext, decode_trip  # noqa: E402
from pathsnap.hmm import HMMParams  # noqa: E402
from pathsnap.viz import route_map  # noqa: E402

BETAS = [1.0, 2.0, 3.0, 5.0, 8.0, 15.0]
ALPHAS = [0.5, 1.0, 2.0]


def main():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    summ = pd.read_csv(RESULTS_DIR / "candidate_summary.csv", dtype={"trip_id": str})
    good = summ[(summ.partition == "train") & (summ.align_dist_median <= 30)].trip_id.tolist()
    rng = np.random.default_rng(SEED)
    hold = sorted(rng.choice(good, 12, replace=False).tolist())
    print("decoder-tuning trips (held out of classifier fit):", hold)

    tr = load_train()
    m = clone(model).fit(tr.loc[~tr.trip_id.isin(hold), FEATURES].values, tr.loc[~tr.trip_id.isin(hold), "y"].values)
    full = load_partition("train")
    full = full[full.trip_id.isin(hold)].reset_index(drop=True)
    p_hold = m.predict_proba(full[FEATURES].values)[:, 1]

    ctxs = {t: TripContext(TRAIN_DIR / t) for t in hold}
    grid = []
    for beta in BETAS:
        for alpha in ALPHAS:
            prm = HMMParams(beta=beta, alpha=alpha)
            rows = []
            for t, ctx in ctxs.items():
                msk = (full.trip_id == t).values
                r, _, _ = decode_trip(ctx, full[msk], p_hold[msk], prm)
                rows.append(r)
            df = pd.DataFrame(rows)
            w = df.n_obs / df.n_obs.sum()
            grid.append({"beta": beta, "alpha": alpha, "clf_top1": (df.clf_top1 * w).sum(), "hmm_top1": (df.hmm_top1 * w).sum(),
                         "hmm_continuity": df.hmm_continuity.mean(), "hmm_route_mismatch": df.hmm_route_mismatch_fraction.mean(),
                         "hmm_breaks": int(df.hmm_breaks.sum())})
            print(grid[-1])
    grid = pd.DataFrame(grid)
    grid.to_csv(RESULTS_DIR / "hmm_grid.csv", index=False)
    b = grid.sort_values(["hmm_top1", "hmm_continuity"], ascending=False).iloc[0]
    params = HMMParams(beta=float(b.beta), alpha=float(b.alpha))
    json.dump(params.to_dict(), open(ARTIFACT_DIR / "hmm_params.json", "w"), indent=2)
    print("chosen decoder params:", params)

    piv = grid.pivot(index="alpha", columns="beta", values="hmm_top1")
    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_xlabel("beta (transition scale, m)"); ax.set_ylabel("alpha (emission weight)")
    ax.set_title("HMM top-1 on 12 held-out training trips"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(FIG_DIR / "hmm_grid.png", dpi=130); plt.close(fig)

    # ---- student-test evaluation with the final model ----
    st = load_student_test()
    p_st = model.predict_proba(st[FEATURES].values)[:, 1]
    rows = []
    for d in sorted(p for p in STUDENT_TEST_DIR.iterdir() if p.is_dir()):
        ctx = TripContext(d)
        msk = (st.trip_id == d.name).values
        r, clf_slots, hmm_slots = decode_trip(ctx, st[msk], p_st[msk], params)
        rows.append(r)
        rr = np.arange(ctx.n)
        fm = route_map(ctx.trip.track, ctx.net, ctx.xy, hmm_local=ctx.cand[rr, hmm_slots], clf_local=ctx.cand[rr, clf_slots],
                       ref_global=ctx.route, title=f"student_test {d.name}: HMM top-1 {r['hmm_top1']:.3f} | classifier-only {r['clf_top1']:.3f}")
        fm.save(MAP_DIR / f"route_student_test_{d.name}.html")
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "hmm_student_test.csv", index=False)
    w = res.n_obs / res.n_obs.sum()
    agg = {f"{c}_weighted": float((res[c] * w).sum()) for c in ["clf_top1", "hmm_top1"]}
    agg.update({f"{c}_mean": float(res[c].mean()) for c in res.columns if c.startswith(("clf_", "hmm_"))})
    agg["params"] = params.to_dict()
    agg["tuning_trips"] = hold
    json.dump(agg, open(RESULTS_DIR / "hmm_summary.json", "w"), indent=2)
    print(json.dumps(agg, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    x = np.arange(len(res)); w_ = 0.4
    for a, (c, title) in zip(ax, [("top1", "Pointwise top-1 accuracy"), ("continuity", "Route continuity"),
                                  ("route_mismatch_fraction", "Route mismatch fraction (lower better)")]):
        a.bar(x - w_ / 2, res[f"clf_{c}"], w_, label="classifier only")
        a.bar(x + w_ / 2, res[f"hmm_{c}"], w_, label="classifier + HMM")
        a.set_xticks(x, res.trip_id.str[-3:], rotation=0); a.set_title(title); a.set_xlabel("student-test trip")
    ax[0].legend(); ax[0].set_ylim(res[["clf_top1", "hmm_top1"]].min().min() - 0.05, 1)
    fig.tight_layout(); fig.savefig(FIG_DIR / "hmm_vs_classifier.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
