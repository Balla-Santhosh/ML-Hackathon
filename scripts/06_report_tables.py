"""Generate Markdown tables for the report directly from the saved experiment results."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import ROOT, RESULTS_DIR  # noqa: E402

OUT = ROOT / "report" / "results_tables.md"
OUT.parent.mkdir(exist_ok=True)


def f(x, d=4):
    return f"{x:.{d}f}"


def main():
    lines = ["# Auto-generated result tables (from results/*.csv, *.json)\n"]

    eda = pd.read_csv(RESULTS_DIR / "eda" / "partition_summary.csv")
    lines += ["## Dataset partitions\n", eda.to_markdown(index=False, floatfmt=".2f"), ""]
    cs = pd.read_csv(RESULTS_DIR / "candidate_summary.csv", dtype={"trip_id": str})
    g = cs.groupby("partition").apply(lambda s: pd.Series({
        "fixes": s.n_obs.sum(), "candidate rows": s.n_rows.sum(),
        "candidate miss rate (fix-weighted)": (s.miss_rate * s.n_obs).sum() / s.n_obs.sum(),
        "median align dist (m)": s.align_dist_median.median()}), include_groups=False)
    lines += ["## Candidate generation\n", g.to_markdown(floatfmt=".4f"), ""]

    c = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    t = pd.DataFrame({
        "Model": c.model, "Family": c.family, "Key configuration": c.config,
        "CV Macro-F1": [f"{a:.4f} ± {b:.4f}" for a, b in zip(c.cv_macro_f1, c.cv_macro_f1_std)],
        "ST Top-1": c.st_top1.map(f), "ST Route continuity": c.st_route_continuity.map(f),
        "ST Macro-F1": c.st_macro_f1.map(f), "ST Log loss": c.st_log_loss.map(f),
        "Inference (ms/1000 fixes)": c.inference_ms_per_1000_fixes.map(lambda x: f"{x:.1f}"),
        "Fit time (s)": c.fit_secs_full.map(lambda x: f"{x:.1f}"),
    })
    lines += ["## 3.1 Model comparison (CV = 5-fold GroupKFold on 80 training trips; ST = 10 student-test trips)\n",
              t.to_markdown(index=False), ""]
    t2 = pd.DataFrame({"Model": c.model, "Train top-1 (resubst.)": c.train_top1.map(f), "CV top-1": c.cv_top1.map(f),
                       "ST top-1": c.st_top1.map(f), "ST precision": c.st_precision.map(f), "ST recall": c.st_recall.map(f),
                       "ST macro-F1 @0.5": c.st_macro_f1_thr05.map(f), "CV log loss": c.cv_log_loss.map(f)})
    lines += ["## Over/under-fitting evidence\n", t2.to_markdown(index=False), ""]

    bo = RESULTS_DIR / "bayes_opt_summary.json"
    if bo.exists():
        b = json.load(open(bo))
        rows = []
        for k in ["cv_macro_f1", "cv_top1", "cv_log_loss", "st_top1", "st_macro_f1", "st_log_loss", "inference_ms_per_1000_fixes", "fit_secs"]:
            rows.append({"Metric": k, "Untuned": f(b["untuned"][k]), "Tuned": f(b["tuned"][k])})
        lines += [f"## 4. Bayesian optimisation – {b['model']} ({b['n_trials']} trials, {b['n_complete']} complete, "
                  f"{b['n_pruned']} pruned, {b['search_secs'] / 60:.1f} min)\n",
                  "Best parameters: `" + json.dumps(b["best_params"]) + "`\n", pd.DataFrame(rows).to_markdown(index=False), ""]

    hs = RESULTS_DIR / "hmm_student_test.csv"
    if hs.exists():
        h = pd.read_csv(hs, dtype={"trip_id": str})
        cols = ["trip_id", "n_obs", "clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity",
                "clf_route_mismatch_fraction", "hmm_route_mismatch_fraction", "hmm_breaks", "hmm_secs"]
        tot = h[cols].copy()
        w = h.n_obs / h.n_obs.sum()
        avg = {"trip_id": "fix-weighted mean", "n_obs": h.n_obs.sum()}
        for col in cols[2:]:
            avg[col] = (h[col] * w).sum() if "top1" in col else h[col].mean()
        tot = pd.concat([tot, pd.DataFrame([avg])])
        lines += ["## 5. Classifier-only vs classifier + HMM (student-test)\n", tot.to_markdown(index=False, floatfmt=".4f"), ""]
        grid = pd.read_csv(RESULTS_DIR / "hmm_grid.csv")
        lines += ["## Decoder grid (12 held-out training trips)\n", grid.to_markdown(index=False, floatfmt=".4f"), ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
