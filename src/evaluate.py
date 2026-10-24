"""Metrics and walk-forward validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def peak_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Error on the daily peak: how well do we predict the highest load of each day?"""
    t = pd.Series(np.asarray(y_true), index=y_true.index)
    p = pd.Series(np.asarray(y_pred), index=y_true.index)
    day = t.index.normalize()
    t_peak, p_peak = t.groupby(day).max(), p.groupby(day).max()
    t_hr = t.groupby(day).idxmax().dt.hour
    p_hr = p.groupby(day).idxmax().dt.hour
    return {
        "peak_MAPE_%": mape(t_peak, p_peak),
        "peak_hour_within_1h_%": float((np.abs(t_hr - p_hr) <= 1).mean() * 100),
    }


def score(y_true: pd.Series, y_pred, name: str = "") -> dict:
    y_pred = pd.Series(np.asarray(y_pred), index=y_true.index)
    out = {
        "model": name,
        "MAPE_%": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }
    out.update(peak_metrics(y_true, y_pred))
    return out


def walk_forward_cv(model_factory, X, y, n_splits: int = 5, test_days: int = 30) -> pd.DataFrame:
    """Expanding-window walk-forward validation (no shuffling, no leakage).

    model_factory: callable returning a fresh object with fit(X, y) / predict(X).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_days * 24)
    rows = []
    for i, (tr, te) in enumerate(tscv.split(X), 1):
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        r = score(y.iloc[te], pred)
        r["fold"] = i
        r["train_end"] = X.index[tr[-1]]
        rows.append(r)
    return pd.DataFrame(rows)
