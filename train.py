"""End-to-end pipeline: data -> features -> models -> evaluation -> SHAP -> BESS demo.

Examples
--------
Synthetic data (runs anywhere, good for testing):
    python train.py

Real Delhi data:
    python train.py --data data/delhi_load.csv --datetime-col timestamp --load-col demand_mw

Faster / without PyTorch:
    python train.py --skip-lstm --skip-cv
"""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.bess import daily_peak_shaving
from src.config import ARTIFACT_DIR, BESS_ENERGY_MWH, BESS_POWER_MW
from src.data import get_dataset
from src.evaluate import score, walk_forward_cv
from src.features import make_features, split_xy, time_split
from src.models import (GBMForecaster, LSTMForecaster, NaiveDayBefore, QuantileBand,
                        SeasonalNaiveWeek, ensemble_mean)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="CSV with timestamp + load columns (omit for synthetic data)")
    p.add_argument("--datetime-col", default="timestamp")
    p.add_argument("--load-col", default="load")
    p.add_argument("--no-weather-api", action="store_true", help="do not call Open-Meteo")
    p.add_argument("--test-days", type=int, default=90)
    p.add_argument("--skip-lstm", action="store_true")
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--lstm-epochs", type=int, default=25)
    p.add_argument("--bess-power", type=float, default=BESS_POWER_MW)
    p.add_argument("--bess-energy", type=float, default=BESS_ENERGY_MWH)
    return p.parse_args()


def main():
    a = parse_args()
    out = ARTIFACT_DIR
    out.mkdir(exist_ok=True)

    # 1. Data + features -----------------------------------------------------
    df, source = get_dataset(a.data, a.datetime_col, a.load_col, use_weather_api=not a.no_weather_api)
    feat = make_features(df)
    X, y = split_xy(feat)
    Xtr, Xte, ytr, yte = time_split(X, y, a.test_days)
    print(f"[data] source={source} rows={len(X)} features={X.shape[1]} "
          f"train={len(Xtr)} test={len(Xte)} ({Xte.index.min().date()} -> {Xte.index.max().date()})")

    preds: dict[str, np.ndarray] = {}
    metrics = []

    # 2. Baselines -----------------------------------------------------------
    for name, m in [("naive_day_before", NaiveDayBefore()), ("seasonal_naive_week", SeasonalNaiveWeek())]:
        preds[name] = m.fit(Xtr, ytr).predict(Xte)
        metrics.append(score(yte, preds[name], name))

    # 3. LightGBM + quantile band -------------------------------------------
    gbm = GBMForecaster().fit(Xtr, ytr)
    preds["lightgbm"] = gbm.predict(Xte)
    metrics.append(score(yte, preds["lightgbm"], "lightgbm"))

    band = QuantileBand(0.1, 0.9).fit(Xtr, ytr)
    q_lo, q_hi = band.predict(Xte)
    coverage = float(((yte.values >= q_lo) & (yte.values <= q_hi)).mean() * 100)
    print(f"[quantile] 10-90% band coverage on test: {coverage:.1f}% (ideal ~80%)")

    # 4. LSTM (optional) -----------------------------------------------------
    lstm = None
    if not a.skip_lstm:
        try:
            lstm = LSTMForecaster(epochs=a.lstm_epochs).fit(Xtr, ytr)
            preds["lstm"] = lstm.predict(Xte, context=Xtr.tail(lstm.seq_len - 1))
            metrics.append(score(yte, preds["lstm"], "lstm"))
        except ImportError:
            print("[warn] PyTorch not installed -> skipping LSTM (pip install torch)")

    # 5. Ensemble ------------------------------------------------------------
    ml_names = [n for n in ("lightgbm", "lstm") if n in preds]
    if len(ml_names) > 1:
        preds["ensemble"] = ensemble_mean(preds, ml_names)
        metrics.append(score(yte, preds["ensemble"], "ensemble"))

    table = pd.DataFrame(metrics).set_index("model").round(3)
    print("\n=== Test-set metrics ===")
    print(table.to_string())
    table.to_csv(out / "metrics.csv")

    best = table["MAPE_%"].idxmin()
    base = table.loc["seasonal_naive_week", "MAPE_%"]
    print(f"\nBest model: {best}  MAPE {table.loc[best, 'MAPE_%']:.2f}% "
          f"vs seasonal-naive {base:.2f}%  "
          f"({(1 - table.loc[best, 'MAPE_%'] / base) * 100:.0f}% lower error)")

    # 6. Walk-forward CV (LightGBM) -----------------------------------------
    if not a.skip_cv:
        cv = walk_forward_cv(lambda: GBMForecaster(n_estimators=300), X, y, n_splits=5, test_days=30)
        cv.to_csv(out / "cv_metrics.csv", index=False)
        print("\n=== Walk-forward CV (LightGBM) ===")
        print(cv[["fold", "MAPE_%", "RMSE", "MAE"]].round(3).to_string(index=False))
        print(f"mean MAPE {cv['MAPE_%'].mean():.2f}% +/- {cv['MAPE_%'].std():.2f}")

    # 7. Save predictions + models ------------------------------------------
    pred_df = pd.DataFrame({"actual": yte.values, **preds, "q10": q_lo, "q90": q_hi}, index=yte.index)
    pred_df.index.name = "timestamp"
    pred_df.to_csv(out / "predictions.csv")
    joblib.dump({"gbm": gbm, "band": band, "features": list(X.columns)}, out / "models.joblib")

    # 8. SHAP explainability -------------------------------------------------
    if not a.skip_shap:
        try:
            import shap

            sample = Xte.sample(min(1000, len(Xte)), random_state=0)
            sv = shap.TreeExplainer(gbm.model).shap_values(sample)
            plt.figure()
            shap.summary_plot(sv, sample, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(out / "shap_summary.png", dpi=150)
            plt.close()
            print("[shap] saved shap_summary.png")
        except Exception as e:
            print(f"[warn] SHAP skipped ({e})")

    # 9. BESS peak shaving demo ---------------------------------------------
    best_pred = pd.Series(preds[best if best in preds else "lightgbm"], index=yte.index)
    bess = daily_peak_shaving(best_pred, yte, power_mw=a.bess_power, energy_mwh=a.bess_energy)
    bess.to_csv(out / "bess_summary.csv", index=False)
    print(f"\n=== BESS peak shaving ({a.bess_power:.0f} MW / {a.bess_energy:.0f} MWh, forecast-driven) ===")
    print(f"avg daily peak reduction: {bess['reduction_mw'].mean():.1f} MW "
          f"({bess['reduction_pct'].mean():.2f}%) over {len(bess)} days")

    # 10. Plots --------------------------------------------------------------
    last = pred_df.iloc[-24 * 7:]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(last.index, last["actual"], label="actual", color="black", lw=1.6)
    ax.plot(last.index, last["lightgbm"], label="LightGBM", color="tab:blue", lw=1.2)
    if "lstm" in last:
        ax.plot(last.index, last["lstm"], label="LSTM", color="tab:orange", lw=1.0, alpha=0.8)
    ax.fill_between(last.index, last["q10"], last["q90"], color="tab:blue", alpha=0.15, label="10-90% band")
    ax.set_ylabel("Load (MW)")
    ax.set_title("Last 7 days of test set: actual vs forecast")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(out / "forecast_last_week.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    table["MAPE_%"].sort_values(ascending=False).plot.barh(ax=ax, color="tab:green")
    ax.set_xlabel("MAPE (%) - lower is better")
    ax.set_title("Model comparison (test set)")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)

    with open(out / "run_info.json", "w") as f:
        json.dump({"source": source, "best_model": best, "test_days": a.test_days,
                   "band_coverage_pct": coverage, "bess_power_mw": a.bess_power,
                   "bess_energy_mwh": a.bess_energy}, f, indent=2)
    print(f"\nArtifacts saved in {out}")


if __name__ == "__main__":
    main()
