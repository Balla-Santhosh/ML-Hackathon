"""Candidate classifiers compared in the experiments."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import SEED


class TorchMLP:
    """Small feed-forward network (PyTorch) with an sklearn-like interface.

    Records per-epoch training / validation BCE loss in `history`.
    """

    def __init__(self, hidden=(64, 32), lr=1e-3, epochs=20, batch=2048, weight_decay=1e-5, dropout=0.1, seed=SEED):
        self.hidden, self.lr, self.epochs, self.batch = hidden, lr, epochs, batch
        self.weight_decay, self.dropout, self.seed = weight_decay, dropout, seed
        self.history = {"train_loss": [], "val_loss": []}

    def _net(self, d):
        import torch.nn as nn
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y, X_val=None, y_val=None):
        import torch
        torch.manual_seed(self.seed)
        self.scaler = StandardScaler().fit(X)
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        if X_val is not None:
            Xv = torch.tensor(self.scaler.transform(X_val), dtype=torch.float32)
            yv = torch.tensor(np.asarray(y_val), dtype=torch.float32)
        self.model = self._net(X.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = torch.nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(self.seed)
        self.history = {"train_loss": [], "val_loss": []}
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(Xt), generator=g)
            tot = 0.0
            for s in range(0, len(Xt), self.batch):
                idx = perm[s:s + self.batch]
                opt.zero_grad()
                loss = lossf(self.model(Xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            self.history["train_loss"].append(tot / len(Xt))
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    self.history["val_loss"].append(lossf(self.model(Xv).squeeze(1), yv).item())
        return self

    def predict_proba(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            z = self.model(torch.tensor(self.scaler.transform(X), dtype=torch.float32)).squeeze(1)
            p = torch.sigmoid(z).numpy()
        return np.c_[1 - p, p]


# name -> (family, key configuration, factory, max training rows or None)
MODEL_ZOO = {
    "LogisticRegression": ("Linear / probabilistic", "C=1.0, L2, standardised",
                           lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)), None),
    "GaussianNB": ("Probabilistic (generative)", "var_smoothing=1e-9",
                   lambda: GaussianNB(), None),
    "KNN": ("Instance-based", "k=25, distance-weighted, standardised, 60k-row reference set",
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)), 60_000),
    "DecisionTree": ("Tree-based", "max_depth=12, min_samples_leaf=20",
                     lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED), None),
    "RandomForest": ("Ensemble (bagging)", "150 trees, max_depth=18, min_samples_leaf=5",
                     lambda: RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=5, n_jobs=-1, random_state=SEED), None),
    "HistGradientBoosting": ("Boosting", "300 iters, lr=0.1, max_leaf_nodes=31",
                             lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED), None),
    "XGBoost": ("Boosting", "400 trees, depth=6, lr=0.1, subsample=0.8, colsample=0.8",
                lambda: XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                      tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED), None),
    "LightGBM": ("Boosting", "400 trees, num_leaves=63, lr=0.05",
                 lambda: LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=-1), None),
    "MLP": ("Neural network", "PyTorch MLP 64-32, ReLU, dropout 0.1, Adam lr=1e-3, 20 epochs",
            lambda: TorchMLP(), None),
}
