"""Feature engineering for day-ahead hourly load forecasting (leakage-safe)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LAGS


def _holiday_flags(index: pd.DatetimeIndex) -> pd.Series:
    """1 on Indian public holidays (incl. Diwali, Holi, ...) if `holidays` is installed."""
    years = sorted(set(index.year))
    dates: set = set()
    try:
        import holidays

        dates = set(holidays.India(years=years).keys())
    except Exception:
        # Fallback: only fixed national holidays
        for y in years:
            dates |= {pd.Timestamp(y, 1, 26).date(), pd.Timestamp(y, 8, 15).date(), pd.Timestamp(y, 10, 2).date()}
    return pd.Series(np.isin(index.date, list(dates)).astype(int), index=index)


def make_features(df: pd.DataFrame, lags=LAGS, dropna: bool = True) -> pd.DataFrame:
    """Build model-ready features. Expects columns: load (+ optional temp, humidity).

    All load-derived features use shift >= 24h, so the model could be run the day
    before. Weather columns are the *actual* weather here; in production you would
    plug in the weather *forecast* for the target day.
    """
    d = df.copy()
    idx = d.index

    # Calendar
    d["hour"] = idx.hour
    d["dayofweek"] = idx.dayofweek
    d["month"] = idx.month
    d["dayofyear"] = idx.dayofyear
    d["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    d["is_holiday"] = _holiday_flags(idx).values

    # Cyclical encodings
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["dow_sin"] = np.sin(2 * np.pi * d["dayofweek"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["dayofweek"] / 7)

    # Load lags and rolling stats (all >= 24h old)
    for lag in lags:
        d[f"lag_{lag}"] = d["load"].shift(lag)
    past = d["load"].shift(24)
    d["roll_mean_24"] = past.rolling(24).mean()
    d["roll_std_24"] = past.rolling(24).std()
    d["roll_mean_168"] = past.rolling(168).mean()
    d["roll_max_24"] = past.rolling(24).max()

    # Weather
    if "temp" in d.columns:
        d["cdd"] = np.clip(d["temp"] - 24, 0, None)  # cooling degree (AC load)
        d["hdd"] = np.clip(14 - d["temp"], 0, None)  # heating degree
        d["temp_lag_24"] = d["temp"].shift(24)
        d["temp_roll_24"] = d["temp"].rolling(24).mean()
    if dropna:
        d = d.dropna()
    return d


def split_xy(feat: pd.DataFrame):
    """Return X (features) and y (target)."""
    y = feat["load"]
    X = feat.drop(columns=["load"])
    return X, y


def time_split(X: pd.DataFrame, y: pd.Series, test_days: int = 90):
    """Chronological split: last `test_days` days are the held-out test set."""
    cut = len(X) - test_days * 24
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
