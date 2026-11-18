"""Candidate classifiers compared in the experiments (one per algorithm family)."""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import SEED

# name -> (family, key configuration, factory, max training rows or None)
MODEL_ZOO = {
    "LogisticRegression": ("Linear / probabilistic", "C=1.0, L2, standardised",
                           lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)), None),
    "KNN": ("Instance-based", "k=25, distance-weighted, standardised, 60k-row reference set",
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)), 60_000),
    "DecisionTree": ("Tree-based", "max_depth=12, min_samples_leaf=20",
                     lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED), None),
    "RandomForest": ("Ensemble (bagging)", "150 trees, max_depth=18, min_samples_leaf=5",
                     lambda: RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=5, n_jobs=-1, random_state=SEED), None),
    "LightGBM": ("Boosting", "400 trees, num_leaves=63, lr=0.05",
                 lambda: LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=-1), None),
}
