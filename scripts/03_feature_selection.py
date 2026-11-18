"""Feature analysis and selection for the candidate classifier.

1. Spearman correlation between the 22 engineered features (redundancy check).
2. Mutual information of each feature with the label (model-free relevance).
3. Permutation importance (drop in log loss) of a LightGBM model trained on
   GroupKFold fold-0 training trips and measured on fold-0 validation trips.
4. Ablation: CV top-1 / macro-F1 using the top-k features of that ranking,
   evaluated on folds 1-4 only (fold 0 was used to build the ranking).
Outputs: results/feature_selection/*.csv, results/figures/fs_*.png
"""
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.config import FIG_DIR, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.dataset import FEATURES, load_train  # noqa: E402
from pathsnap.metrics import candidate_metrics  # noqa: E402

OUT = RESULTS_DIR / "feature_selection"
OUT.mkdir(exist_ok=True)


def lgbm():
    return LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=-1)


def main():
    tr = load_train()
    X, y, g = tr[FEATURES], tr["y"].values, tr["trip_id"].values
    rng = np.random.default_rng(SEED)
    samp = rng.choice(len(tr), 150_000, replace=False)

    # 1. correlation
    corr = X.iloc[samp].corr(method="spearman")
    corr.to_csv(OUT / "spearman_correlation.csv")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=7)
    fig.colorbar(im, shrink=0.8); ax.set_title("Spearman correlation of features (train, 150k candidate rows)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fs_correlation.png", dpi=130); plt.close(fig)
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:] if abs(corr.loc[a, b]) >= 0.9]
    print("highly correlated pairs (|rho|>=0.9):", [(a, b, round(r, 3)) for a, b, r in pairs])

    # 2. mutual information
    mi = mutual_info_classif(X.iloc[samp].values, y[samp], random_state=SEED)
    mi = pd.Series(mi, index=FEATURES).sort_values(ascending=False)

    # 3. permutation importance on fold-0 validation trips
    folds = list(GroupKFold(n_splits=5).split(X, y, g))
    a, b = folds[0]
    m = lgbm().fit(X.values[a], y[a])
    vb = rng.choice(b, min(80_000, len(b)), replace=False)
    t0 = time.time()
    pi = permutation_importance(m, X.values[vb], y[vb], scoring="neg_log_loss", n_repeats=3, random_state=SEED, n_jobs=1)
    print(f"permutation importance {time.time() - t0:.0f}s")
    perm = pd.Series(pi.importances_mean, index=FEATURES)
    gain = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
    imp = pd.DataFrame({"mutual_info": mi, "perm_importance_logloss": perm, "perm_std": pi.importances_std,
                        "lgbm_gain_share": gain / gain.sum()}).sort_values("perm_importance_logloss", ascending=False)
    imp.to_csv(OUT / "feature_importance.csv")
    print(imp.round(4).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].barh(imp.index[::-1], imp.perm_importance_logloss[::-1], xerr=imp.perm_std[::-1], color="tab:blue")
    ax[0].set_title("Permutation importance (log-loss increase, fold-0 validation trips)")
    mi_s = mi.sort_values()
    ax[1].barh(mi_s.index, mi_s.values, color="tab:green"); ax[1].set_title("Mutual information with label")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fs_importance.png", dpi=130); plt.close(fig)

    # 4. ablation on folds 1-4
    ranking = list(imp.index)
    rows = []
    for k in [3, 5, 8, 12, 16, len(FEATURES)]:
        feats = ranking[:k]
        mets = []
        for a, b in folds[1:]:
            mm = lgbm().fit(tr[feats].values[a], y[a])
            mets.append(candidate_metrics(tr.iloc[b], mm.predict_proba(tr[feats].values[b])[:, 1]))
        d = pd.DataFrame(mets)
        rows.append({"k": k, "features": ",".join(feats), "cv_top1": d.top1_acc.mean(), "cv_macro_f1": d.macro_f1.mean(),
                     "cv_macro_f1_std": d.macro_f1.std(), "cv_log_loss": d.log_loss.mean()})
        print(rows[-1]["k"], round(rows[-1]["cv_top1"], 4), round(rows[-1]["cv_macro_f1"], 4), round(rows[-1]["cv_log_loss"], 4))
    ab = pd.DataFrame(rows)
    ab.to_csv(OUT / "ablation_topk.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ab.k, ab.cv_macro_f1, yerr=ab.cv_macro_f1_std, marker="o", capsize=3, label="CV macro-F1")
    ax.plot(ab.k, ab.cv_top1, "s--", label="CV top-1")
    ax.set_xlabel("number of features (top-k by permutation importance)"); ax.legend()
    ax.set_title("Feature-count ablation (LightGBM, GroupKFold folds 1-4)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fs_ablation.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
