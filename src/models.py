"""Baselines, gradient boosting (+ quantile bands), LSTM and ensemble."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE


# ------------------------------------------------------------------- baselines
class NaiveDayBefore:
    """Forecast = load 24h ago (yesterday, same hour). Needs the `lag_24` feature."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["lag_24"].values


class SeasonalNaiveWeek:
    """Forecast = load 168h ago (same hour, same weekday last week)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["lag_168"].values


# ----------------------------------------------------------- gradient boosting
def _make_gbm(**kw):
    try:
        from lightgbm import LGBMRegressor

        params = dict(
            n_estimators=600, learning_rate=0.03, num_leaves=48, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1,
        )
        params.update(kw)
        return LGBMRegressor(**params)
    except ImportError:  # fallback keeps the project runnable without lightgbm
        from sklearn.ensemble import HistGradientBoostingRegressor

        loss = "quantile" if kw.get("objective") == "quantile" else "squared_error"
        extra = {"quantile": kw["alpha"]} if loss == "quantile" else {}
        return HistGradientBoostingRegressor(loss=loss, max_iter=400, learning_rate=0.05,
                                             random_state=RANDOM_STATE, **extra)


class GBMForecaster:
    def __init__(self, **kw):
        self.kw = kw
        self.model = None

    def fit(self, X, y):
        self.model = _make_gbm(**self.kw)
        self.model.fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X):
        return self.model.predict(X[self.feature_names_])


class QuantileBand:
    """Lower / upper prediction interval via quantile regression (default 10%-90%)."""

    def __init__(self, low: float = 0.1, high: float = 0.9):
        self.low, self.high = low, high

    def fit(self, X, y):
        self.lo = GBMForecaster(objective="quantile", alpha=self.low).fit(X, y)
        self.hi = GBMForecaster(objective="quantile", alpha=self.high).fit(X, y)
        return self

    def predict(self, X):
        lo, hi = self.lo.predict(X), self.hi.predict(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)


# ------------------------------------------------------------------------ LSTM
class LSTMForecaster:
    """LSTM over a sliding window of the last `seq_len` feature rows -> load at the last row.

    Every feature row is already leakage-safe for its own target hour, so a window of
    consecutive rows is too. Requires PyTorch (optional dependency).
    """

    def __init__(self, seq_len: int = 24, hidden: int = 64, layers: int = 2, epochs: int = 25,
                 lr: float = 2e-3, batch_size: int = 256, patience: int = 4, seed: int = RANDOM_STATE):
        self.seq_len, self.hidden, self.layers = seq_len, hidden, layers
        self.epochs, self.lr, self.batch_size, self.patience, self.seed = epochs, lr, batch_size, patience, seed

    # -- helpers
    def _windows(self, arr: np.ndarray) -> np.ndarray:
        from numpy.lib.stride_tricks import sliding_window_view

        w = sliding_window_view(arr, self.seq_len, axis=0)  # (N-seq+1, F, seq)
        return np.ascontiguousarray(w.transpose(0, 2, 1))   # (N-seq+1, seq, F)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import torch
        from sklearn.preprocessing import StandardScaler
        from torch import nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.cols_ = list(X.columns)
        self.xs_, self.ys_ = StandardScaler(), StandardScaler()
        Xs = self.xs_.fit_transform(X.values)
        ys = self.ys_.fit_transform(y.values.reshape(-1, 1)).ravel()

        Xw = self._windows(Xs).astype(np.float32)
        yw = ys[self.seq_len - 1:].astype(np.float32)
        n_val = max(int(0.1 * len(Xw)), 1)
        Xtr, ytr, Xva, yva = Xw[:-n_val], yw[:-n_val], Xw[-n_val:], yw[-n_val:]

        class Net(nn.Module):
            def __init__(s, f, h, l):
                super().__init__()
                s.lstm = nn.LSTM(f, h, l, batch_first=True, dropout=0.1 if l > 1 else 0.0)
                s.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(s, x):
                o, _ = s.lstm(x)
                return s.head(o[:, -1]).squeeze(-1)

        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        net = Net(Xw.shape[2], self.hidden, self.layers).to(self.device_)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
        Xva_t, yva_t = torch.tensor(Xva).to(self.device_), torch.tensor(yva).to(self.device_)

        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xtr_t))
            for i in range(0, len(perm), self.batch_size):
                b = perm[i:i + self.batch_size]
                xb, yb = Xtr_t[b].to(self.device_), ytr_t[b].to(self.device_)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(Xva_t), yva_t).item()
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_ = net.eval()
        return self

    def predict(self, X: pd.DataFrame, context: pd.DataFrame | None = None):
        """`context` = the last seq_len-1 rows before X (e.g. tail of training data)."""
        import torch

        full = X if context is None else pd.concat([context, X])
        Xs = self.xs_.transform(full[self.cols_].values)
        Xw = self._windows(Xs).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xw), 1024):
                xb = torch.tensor(Xw[i:i + 1024]).to(self.device_)
                out.append(self.net_(xb).cpu().numpy())
        pred = self.ys_.inverse_transform(np.concatenate(out).reshape(-1, 1)).ravel()
        return pred[-len(X):]


# --------------------------------------------------------------------- ensemble
def ensemble_mean(preds: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.mean([preds[n] for n in names], axis=0)
