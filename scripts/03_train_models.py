"""Train and compare candidate classifiers.

Protocol
  * training data   : 80 training trips, fixes with reliable labels, every 2nd fix
  * cross-validation: 5-fold GroupKFold grouped by trip_id (no trip in two folds)
  * development test: 10 student-testing trips, all fixes
  * metrics         : observation-level Top-1 accuracy, macro-F1 (top-1 decision),
                      log loss, precision/recall, route continuity (argmax route),
                      training and inference time
Outputs: results/model_cv_folds.csv, results/model_comparison.csv,
         results/figures/*.png, data_cache/st_prob_<model>.npy
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import CACHE_DIR, FIG_DIR, RESULTS_DIR, SEED, STUDENT_TEST_DIR  # noqa: E402
from pathsnap.dataset import FEATURES, load_student_test, load_train  # noqa: E402
from pathsnap.evaluation import TripContext, decode_trip  # noqa: E402
from pathsnap.metrics import argmax_decision, candidate_metrics  # noqa: E402
from pathsnap.models import MODEL_ZOO, TorchMLP  # noqa: E402


def fit_model(name, X, y, X_val=None, y_val=None):
    family, cfg, factory, max_rows = MODEL_ZOO[name]
    m = factory()
    if max_rows is not None and len(X) > max_rows:
        idx = np.random.default_rng(SEED).choice(len(X), max_rows, replace=False)
        X, y = X[idx], y[idx]
    t0 = time.time()
    if isinstance(m, TorchMLP):
        m.fit(X, y, X_val, y_val)
    elif name in ("XGBoost",) and X_val is not None:
        m.fit(X, y, eval_set=[(X, y), (X_val, y_val)], verbose=False)
    elif name == "LightGBM" and X_val is not None:
        m.fit(X, y, eval_set=[(X, y), (X_val, y_val)], eval_names=["train", "val"])
    else:
        m.fit(X, y)
    return m, time.time() - t0


def history_of(name, m):
    if isinstance(m, TorchMLP):
        return m.history
    if name == "XGBoost" and hasattr(m, "evals_result"):
        r = m.evals_result()
        return {"train_loss": r["validation_0"]["logloss"], "val_loss": r["validation_1"]["logloss"]}
    if name == "LightGBM" and getattr(m, "evals_result_", None):
        r = m.evals_result_
        return {"train_loss": r["train"]["binary_logloss"], "val_loss": r["val"]["binary_logloss"]}
    return None


def main(only=None):
    tr = load_train()
    st = load_student_test()
    X, y, g = tr[FEATURES].values, tr["y"].values, tr["trip_id"].values
    Xs, ys = st[FEATURES].values, st["y"].values
    print(f"train rows={len(tr):,} obs={tr.groupby(['trip_id','obs']).ngroups:,} trips={tr.trip_id.nunique()} | "
          f"student_test rows={len(st):,} obs={st.groupby(['trip_id','obs']).ngroups:,}")
    folds = list(GroupKFold(n_splits=5).split(X, y, g))

    print("preparing student-test trip contexts ...")
    ctxs = {d.name: TripContext(d) for d in sorted(STUDENT_TEST_DIR.iterdir()) if d.is_dir()}

    fold_rows, comp_rows, histories = [], [], {}
    names = only or list(MODEL_ZOO)
    for name in names:
        family, cfg, _, _ = MODEL_ZOO[name]
        print(f"\n=== {name} ===")
        for k, (a, b) in enumerate(folds):
            m, fit_s = fit_model(name, X[a], y[a], X[b], y[b])
            p_val = m.predict_proba(X[b])[:, 1]
            met = candidate_metrics(tr.iloc[b], p_val)
            fold_rows.append({"model": name, "fold": k, "fit_secs": fit_s, **met})
            if k == 0:
                h = history_of(name, m)
                if h:
                    histories[name] = h
            print(f"  fold {k}: top1={met['top1_acc']:.4f} macroF1={met['macro_f1']:.4f} logloss={met['log_loss']:.4f} ({fit_s:.1f}s)")

        # refit on all training trips, evaluate on student-test
        m, fit_s = fit_model(name, X, y)
        p_tr_full = m.predict_proba(X)[:, 1]
        train_met = candidate_metrics(tr, p_tr_full)
        t0 = time.time()
        p_st = m.predict_proba(Xs)[:, 1]
        inf_s = time.time() - t0
        np.save(CACHE_DIR / f"st_prob_{name}.npy", p_st)
        st_met = candidate_metrics(st, p_st)
        # classifier-only route continuity on whole student-test trips
        conts, rprec, rrec = [], [], []
        for tid, ctx in ctxs.items():
            msk = (st.trip_id == tid).values
            r, _, _ = decode_trip(ctx, st[msk], p_st[msk], do_hmm=False)
            conts.append(r["clf_continuity"]); rprec.append(r["clf_route_precision_len"]); rrec.append(r["clf_route_recall_len"])
        cv = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        dec = argmax_decision(st, p_st)
        cm = confusion_matrix(ys, dec)
        row = {
            "model": name, "family": family, "config": cfg,
            "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
            "cv_top1": cv.top1_acc.mean(), "cv_top1_std": cv.top1_acc.std(), "cv_log_loss": cv.log_loss.mean(),
            "train_top1": train_met["top1_acc"], "train_macro_f1": train_met["macro_f1"],
            "st_top1": st_met["top1_acc"], "st_macro_f1": st_met["macro_f1"], "st_macro_f1_thr05": st_met["macro_f1_thr05"],
            "st_precision": st_met["precision_pos"], "st_recall": st_met["recall_pos"], "st_log_loss": st_met["log_loss"],
            "st_route_continuity": float(np.mean(conts)), "st_route_precision_len": float(np.mean(rprec)),
            "st_route_recall_len": float(np.mean(rrec)),
            "fit_secs_full": fit_s, "inference_ms_per_1000_fixes": 1000 * inf_s / st.groupby(["trip_id", "obs"]).ngroups * 1000,
            "cm_tn": int(cm[0, 0]), "cm_fp": int(cm[0, 1]), "cm_fn": int(cm[1, 0]), "cm_tp": int(cm[1, 1]),
        }
        comp_rows.append(row)
        print(f"  CV macroF1={row['cv_macro_f1']:.4f}±{row['cv_macro_f1_std']:.4f} | train top1={row['train_top1']:.4f} | "
              f"ST top1={row['st_top1']:.4f} F1={row['st_macro_f1']:.4f} LL={row['st_log_loss']:.4f} "
              f"cont={row['st_route_continuity']:.4f}")
        pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "model_cv_folds.csv", index=False)
        pd.DataFrame(comp_rows).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        json.dump(histories, open(RESULTS_DIR / "training_histories.json", "w"))

    make_plots(pd.DataFrame(comp_rows), pd.DataFrame(fold_rows), histories)


def make_plots(comp, folds, histories):
    comp = comp.sort_values("cv_macro_f1")
    # CV macro-F1 with fold spread
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(comp.model, comp.cv_macro_f1, xerr=comp.cv_macro_f1_std, color="tab:blue")
    for _, r in folds.iterrows():
        ax.plot(r.macro_f1, list(comp.model).index(r.model), "k.", ms=3)
    ax.set_xlim(max(0, comp.cv_macro_f1.min() - 0.05), 1); ax.set_xlabel("5-fold GroupKFold macro-F1 (train trips)")
    ax.set_title("Cross-validation macro-F1 per model (bars = mean ± std, dots = folds)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.38; xs = np.arange(len(comp))
    ax.bar(xs - w / 2, comp.st_top1, w, label="Top-1 accuracy")
    ax.bar(xs + w / 2, comp.st_macro_f1, w, label="Macro-F1")
    ax.set_xticks(xs, comp.model, rotation=30, ha="right"); ax.set_ylim(max(0, comp[["st_top1", "st_macro_f1"]].min().min() - 0.05), 1)
    ax.set_title("Student-testing partition (10 trips): Top-1 accuracy and macro-F1"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "student_test_top1_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(comp.model, comp.st_log_loss, color="tab:red"); ax[0].set_yscale("log")
    ax[0].set_title("Student-test log loss (lower is better, log scale)"); ax[0].tick_params(axis="x", rotation=35)
    ax[1].scatter(comp.inference_ms_per_1000_fixes, comp.st_top1)
    for _, r in comp.iterrows():
        ax[1].annotate(r.model, (r.inference_ms_per_1000_fixes, r.st_top1), fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("inference ms / 1000 fixes"); ax[1].set_ylabel("student-test top-1")
    ax[1].set_title("Accuracy vs inference cost")
    fig.tight_layout(); fig.savefig(FIG_DIR / "log_loss_and_speed.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(comp.model, comp.cv_top1, "o-", label="CV top-1 (train trips)")
    ax.plot(comp.model, comp.st_top1, "s-", label="student-test top-1")
    ax.plot(comp.model, comp.train_top1, "^--", label="training-set top-1 (resubstitution)")
    ax.plot(comp.model, comp.st_route_continuity, "d-", label="student-test route continuity")
    ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8); ax.set_title("Generalisation check: train vs CV vs student-test")
    fig.tight_layout(); fig.savefig(FIG_DIR / "generalisation_gap.png", dpi=130); plt.close(fig)

    n = len(comp)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6.4))
    for a, (_, r) in zip(axs.flat, comp.iterrows()):
        cm = np.array([[r.cm_tn, r.cm_fp], [r.cm_fn, r.cm_tp]])
        a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8)
        a.set_title(r.model, fontsize=9); a.set_xticks([0, 1], ["pred 0", "pred 1"]); a.set_yticks([0, 1], ["true 0", "true 1"])
    for a in axs.flat[n:]:
        a.axis("off")
    fig.suptitle("Student-test candidate confusion matrices (top-1 decision, row-normalised colour)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=130); plt.close(fig)

    if histories:
        fig, axs = plt.subplots(1, len(histories), figsize=(5 * len(histories), 3.5))
        for a, (name, h) in zip(np.atleast_1d(axs), histories.items()):
            a.plot(h["train_loss"], label="train"); a.plot(h["val_loss"], label="validation (fold 0)")
            a.set_title(f"{name}: log loss per {'epoch' if name == 'MLP' else 'boosting round'}"); a.legend()
        fig.tight_layout(); fig.savefig(FIG_DIR / "training_curves.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
