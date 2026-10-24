# Delhi Electricity Demand Forecasting + BESS Peak Shaving

End-to-end day-ahead hourly load forecasting for Delhi, with weather and calendar features, a model ladder
(baselines -> LightGBM -> LSTM -> ensemble), walk-forward validation, SHAP explainability, probabilistic
10-90% bands, and a forecast-driven battery (BESS) peak-shaving optimiser. Includes a Streamlit dashboard.

## Quick start

```bash
pip install -r requirements.txt

# 1) Test the whole pipeline on synthetic data
python train.py

# 2) Real data (Delhi SLDC / Grid-India CSV)
python train.py --data data/delhi_load.csv --datetime-col timestamp --load-col demand_mw

# 3) Dashboard
streamlit run app.py
```

Useful flags: `--skip-lstm` (no PyTorch), `--skip-cv`, `--skip-shap`, `--no-weather-api`, `--test-days 90`,
`--bess-power 300 --bess-energy 1200`.

## Project structure

```
data/         raw CSVs (+ weather cache)
src/
  data.py       load CSV, Open-Meteo weather, synthetic generator
  features.py   calendar, holiday, lag (>=24h), rolling, cyclical, degree-day features
  models.py     naive baselines, LightGBM, quantile band, LSTM, ensemble
  evaluate.py   MAPE / RMSE / MAE, daily-peak metrics, walk-forward CV
  bess.py       LP peak-shaving optimiser (scipy HiGHS)
train.py      full pipeline -> artifacts/
app.py        Streamlit dashboard
artifacts/    metrics.csv, predictions.csv, plots, models.joblib
```

## Method notes

- **No leakage:** all load lags/rolling features are >= 24h old (day-ahead setting); split is chronological;
  CV is expanding-window walk-forward.
- **Weather:** actual weather is used for training/testing. In production, plug in the weather *forecast*.
- **Baselines first:** a model only counts if it beats seasonal-naive (same hour last week).
- **BESS:** schedule is optimised on the *forecast* and applied to the *actual* load, so reported peak
  reduction includes forecast error.

## Results (fill in after running on real data)

| Model | MAPE % | RMSE | Peak MAPE % |
|-------|--------|------|-------------|
| Seasonal naive | | | |
| LightGBM | | | |
| LSTM | | | |
| Ensemble | | | |

> The numbers from the default run use **synthetic data**. Do not put them on your resume. Re-run with real
> Delhi data and use those.

## Resume line (after real-data run)

"Built Delhi hourly load forecasting pipeline using weather and calendar features (LightGBM, LSTM, ensemble)
with walk-forward validation; reduced MAPE from X% to Y% over seasonal baseline; added BESS peak-shaving
optimisation achieving Z% peak reduction."
