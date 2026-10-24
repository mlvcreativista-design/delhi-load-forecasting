"""Data loading: real Delhi load CSV + Open-Meteo weather, with a synthetic fallback.

Real data
---------
Download hourly (or 15-min) Delhi demand from Delhi SLDC (delhisldc.org) or
Grid-India (grid-india.in) and save as CSV. Then run:

    python train.py --data data/delhi_load.csv --datetime-col timestamp --load-col demand_mw

Synthetic data
--------------
If no CSV is given, a realistic synthetic Delhi-like series is generated so the whole
pipeline can be run and tested end-to-end. Replace with real data before putting
numbers on your resume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, LAT, LON, RANDOM_STATE, TIMEZONE


# ----------------------------------------------------------------------------- load
def load_load_csv(path, datetime_col: str = "timestamp", load_col: str = "load") -> pd.DataFrame:
    """Read a load CSV and return an hourly, gap-filled DataFrame with a single 'load' column."""
    raw = pd.read_csv(path)
    raw[datetime_col] = pd.to_datetime(raw[datetime_col], dayfirst=True)
    s = (
        raw.set_index(datetime_col)[load_col]
        .astype(float)
        .sort_index()
    )
    s = s[~s.index.duplicated(keep="first")]
    s = s.resample("h").mean()  # 15-min -> hourly
    s = s.interpolate(limit=6)  # fill short gaps only
    df = s.to_frame("load").dropna()
    df.index.name = "timestamp"
    return df


# -------------------------------------------------------------------------- weather
def fetch_weather(start: str, end: str, lat: float = LAT, lon: float = LON) -> pd.DataFrame:
    """Hourly temperature + humidity from the free Open-Meteo archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": TIMEZONE,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    w = pd.DataFrame(
        {
            "temp": h["temperature_2m"],
            "humidity": h["relative_humidity_2m"],
        },
        index=pd.to_datetime(h["time"]),
    )
    w.index.name = "timestamp"
    return w


# ---------------------------------------------------------------------- synthetic
def generate_synthetic(start="2021-01-01", end="2024-12-31", seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Synthetic Delhi-like hourly load (MW) + weather. Heavy AC load in summer, evening peak."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end + " 23:00" if len(end) == 10 else end, freq="h")
    n = len(idx)
    doy = idx.dayofyear.values
    hour = idx.hour.values

    # Temperature: seasonal (peak ~ June) + daily cycle (peak ~ 15:00) + noise
    seasonal = 25 + 12 * np.sin(2 * np.pi * (doy - 105) / 365.25)
    daily = 5 * np.sin(2 * np.pi * (hour - 9) / 24)
    temp = seasonal + daily + rng.normal(0, 1.5, n)

    humidity = np.clip(55 + 25 * np.sin(2 * np.pi * (doy - 200) / 365.25) - 1.2 * (temp - 25)
                       + rng.normal(0, 6, n), 10, 100)

    # Load components
    base = 3800 + 0.15 * np.arange(n) / 24  # slow growth
    hour_profile = (
        600 * np.exp(-0.5 * ((hour - 21) / 2.5) ** 2)    # evening peak
        + 350 * np.exp(-0.5 * ((hour - 12) / 3.0) ** 2)  # midday
        - 500 * np.exp(-0.5 * ((hour - 4) / 2.5) ** 2)   # night trough
    )
    cooling = 110 * np.clip(temp - 24, 0, None) * (1 + 0.004 * humidity)
    heating = 35 * np.clip(14 - temp, 0, None)
    weekend = np.where(idx.dayofweek >= 5, -250, 0)
    load = base + hour_profile + cooling + heating + weekend + rng.normal(0, 90, n)

    df = pd.DataFrame({"load": load, "temp": temp, "humidity": humidity}, index=idx)
    df.index.name = "timestamp"
    return df


# ------------------------------------------------------------------------ top-level
def get_dataset(
    csv_path: str | None = None,
    datetime_col: str = "timestamp",
    load_col: str = "load",
    use_weather_api: bool = True,
    cache: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Return (df, source) where df has columns load, temp, humidity and an hourly index."""
    if csv_path is None:
        return generate_synthetic(), "synthetic"

    df = load_load_csv(csv_path, datetime_col, load_col)
    source = "real_load"

    if use_weather_api:
        cache_file = DATA_DIR / "weather_cache.csv"
        try:
            if cache and cache_file.exists():
                w = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            else:
                w = fetch_weather(df.index.min().strftime("%Y-%m-%d"), df.index.max().strftime("%Y-%m-%d"))
                if cache:
                    DATA_DIR.mkdir(exist_ok=True)
                    w.to_csv(cache_file)
            df = df.join(w, how="left")
            df[["temp", "humidity"]] = df[["temp", "humidity"]].interpolate(limit=6)
            df = df.dropna()
            source = "real_load+open_meteo"
        except Exception as e:  # network / API problems should not kill the run
            print(f"[warn] weather fetch failed ({e}); continuing without weather features")
    return df, source
