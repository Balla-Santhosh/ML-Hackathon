"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected classifier.

Objective : mean 5-fold GroupKFold macro-F1 (top-1 decision) on the 80 training trips
            -- the same primary metric used for model selection.
Pruning   : MedianPruner on per-fold intermediate values.
Outputs   : results/bayes_opt_trials.csv, results/bayes_opt_summary.json,
            results/figures/bayes_opt_*.png, artifacts/model.joblib, artifacts/model_meta.json
Usage     : python scripts/04_bayes_opt.py [ModelName] [n_trials]
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import ARTIFACT_DIR, CACHE_DIR, FIG_DIR, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.dataset import FEATURES, LABEL_MAX_ALIGN_M, TRAIN_STRIDE, load_student_test, load_train  # noqa: E402
from pathsnap.metrics import candidate_metrics  # noqa: E402

SPACES = {
    "LightGBM": lambda tr: LGBMClassifier(
        n_estimators=tr.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=tr.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=tr.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=tr.suggest_int("min_child_samples", 5, 200, log=True),
        subsample=tr.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=tr.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=tr.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        random_state=SEED, verbose=-1, n_jobs=-1),
    "XGBoost": lambda tr: XGBClassifier(
        n_estimators=tr.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=tr.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=tr.suggest_int("max_depth", 3, 10),
        min_child_weight=tr.suggest_float("min_child_weight", 0.5, 50, log=True),
        subsample=tr.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=tr.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        gamma=tr.suggest_float("gamma", 1e-4, 5.0, log=True),
        tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED),
    "HistGradientBoosting": lambda tr: HistGradientBoostingClassifier(
        max_iter=tr.suggest_int("max_iter", 100, 800, step=50),
        learning_rate=tr.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_leaf_nodes=tr.suggest_int("max_leaf_nodes", 15, 255, log=True),
        min_samples_leaf=tr.suggest_int("min_samples_leaf", 5, 200, log=True),
        l2_regularization=tr.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        random_state=SEED),
}


def pick_model():
    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    comp = comp[comp.model.isin(SPACES)].sort_values("cv_macro_f1", ascending=False)
    return comp.iloc[0].model


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else pick_model()
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(f"Bayesian optimisation of {name}, {n_trials} trials")
    tr = load_train()
    st = load_student_test()
    X, y, g = tr[FEATURES].values, tr["y"].values, tr["trip_id"].values
    folds = list(GroupKFold(n_splits=5).split(X, y, g))

    def objective(trial):
        model_f = SPACES[name]
        scores, lls, top1s = [], [], []
        for k, (a, b) in enumerate(folds):
            m = model_f(trial)
            m.fit(X[a], y[a])
            p = m.predict_proba(X[b])[:, 1]
            met = candidate_metrics(tr.iloc[b], p)
            scores.append(met["macro_f1"]); lls.append(met["log_loss"]); top1s.append(met["top1_acc"])
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1", float(np.mean(top1s)))
        trial.set_user_attr("cv_macro_f1_std", float(np.std(scores)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
                                study_name=f"pathsnap_{name}")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    search_secs = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(RESULTS_DIR / "bayes_opt_trials.csv", index=False)
    best = study.best_trial
    print("best CV macro-F1", best.value, best.params)

    # refit tuned model on all training trips
    fixed = optuna.trial.FixedTrial(best.params)
    model = SPACES[name](fixed)
    t1 = time.time(); model.fit(X, y); fit_s = time.time() - t1
    t1 = time.time(); p_st = model.predict_proba(st[FEATURES].values)[:, 1]; inf_s = time.time() - t1
    np.save(CACHE_DIR / "st_prob_tuned.npy", p_st)
    st_met = candidate_metrics(st, p_st)
    n_obs = st.groupby(["trip_id", "obs"]).ngroups

    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv").set_index("model")
    u = comp.loc[name]
    summary = {
        "model": name, "n_trials": n_trials, "n_complete": int((trials.state == "COMPLETE").sum()),
        "n_pruned": int((trials.state == "PRUNED").sum()), "search_secs": search_secs,
        "objective": "mean 5-fold GroupKFold macro-F1 (top-1 decision) on training trips",
        "best_params": best.params,
        "untuned": {"cv_macro_f1": u.cv_macro_f1, "cv_top1": u.cv_top1, "cv_log_loss": u.cv_log_loss,
                    "st_top1": u.st_top1, "st_macro_f1": u.st_macro_f1, "st_log_loss": u.st_log_loss,
                    "inference_ms_per_1000_fixes": u.inference_ms_per_1000_fixes, "fit_secs": u.fit_secs_full},
        "tuned": {"cv_macro_f1": best.value, "cv_top1": best.user_attrs["cv_top1"], "cv_log_loss": best.user_attrs["cv_log_loss"],
                  "st_top1": st_met["top1_acc"], "st_macro_f1": st_met["macro_f1"], "st_log_loss": st_met["log_loss"],
                  "inference_ms_per_1000_fixes": 1e6 * inf_s / n_obs, "fit_secs": fit_s},
    }
    json.dump(summary, open(RESULTS_DIR / "bayes_opt_summary.json", "w"), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))

    joblib.dump(model, ARTIFACT_DIR / "model.joblib", compress=3)
    json.dump({"model": name, "params": best.params, "features": FEATURES, "label_max_align_m": LABEL_MAX_ALIGN_M,
               "train_stride": TRAIN_STRIDE, "seed": SEED, "student_test_metrics": st_met},
              open(ARTIFACT_DIR / "model_meta.json", "w"), indent=2, default=float)

    # plots
    done = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(done.number, done.value, "o", label="trial CV macro-F1")
    ax[0].plot(done.number, done.value.cummax(), "r-", label="best so far")
    ax[0].axhline(u.cv_macro_f1, color="gray", ls="--", label="untuned default")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV macro-F1"); ax[0].legend(); ax[0].set_title(f"Optuna TPE history ({name})")
    labels = ["CV macro-F1", "ST top-1", "ST macro-F1"]
    uv = [u.cv_macro_f1, u.st_top1, u.st_macro_f1]
    tv = [summary["tuned"]["cv_macro_f1"], st_met["top1_acc"], st_met["macro_f1"]]
    xs = np.arange(3)
    ax[1].bar(xs - 0.2, uv, 0.4, label="untuned"); ax[1].bar(xs + 0.2, tv, 0.4, label="tuned")
    ax[1].set_xticks(xs, labels); ax[1].set_ylim(min(uv + tv) - 0.03, 1); ax[1].legend(); ax[1].set_title("Untuned vs tuned")
    fig.tight_layout(); fig.savefig(FIG_DIR / "bayes_opt_history.png", dpi=130); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1]); ax.set_title("Hyperparameter importance (fANOVA)")
        fig.tight_layout(); fig.savefig(FIG_DIR / "bayes_opt_importance.png", dpi=130); plt.close(fig)
    except Exception as e:  # importance needs >1 complete trial
        print("importance plot skipped:", e)


if __name__ == "__main__":
    main()
