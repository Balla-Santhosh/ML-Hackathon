"""Candidate classifiers compared in the experiments.

Phase 1: two baseline models from different families (linear vs tree-based).
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import SEED

# name -> (family, key configuration, factory, max training rows or None)
MODEL_ZOO = {
    "LogisticRegression": ("Linear / probabilistic", "C=1.0, L2, standardised",
                           lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)), None),
    "DecisionTree": ("Tree-based", "max_depth=12, min_samples_leaf=20",
                     lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED), None),
}
