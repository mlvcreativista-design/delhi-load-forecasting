"""Streamlit dashboard. Run `python train.py` first, then:  streamlit run app.py"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.bess import apply_schedule, optimize_peak_shaving
from src.config import ARTIFACT_DIR, BESS_ENERGY_MWH, BESS_POWER_MW

st.set_page_config(page_title="Delhi Load Forecasting", page_icon="⚡", layout="wide")
st.title("⚡ Delhi Electricity Demand Forecasting + BESS Peak Shaving")

pred_path = ARTIFACT_DIR / "predictions.csv"
if not pred_path.exists():
    st.error("No predictions found. Run `python train.py` first.")
    st.stop()


@st.cache_data
def load_artifacts():
    preds = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    metrics = pd.read_csv(ARTIFACT_DIR / "metrics.csv", index_col=0)
    info = json.loads((ARTIFACT_DIR / "run_info.json").read_text()) if (ARTIFACT_DIR / "run_info.json").exists() else {}
    return preds, metrics, info


preds, metrics, info = load_artifacts()
if info.get("source") == "synthetic":
    st.warning("Showing results on SYNTHETIC data. Train on real Delhi SLDC data for real numbers.")

model_cols = [c for c in preds.columns if c not in ("actual", "q10", "q90")]
default_model = info.get("best_model", model_cols[0])

with st.sidebar:
    st.header("Controls")
    model = st.selectbox("Model", model_cols, index=model_cols.index(default_model) if default_model in model_cols else 0)
    dates = sorted(set(preds.index.date))
    day = st.selectbox("Date", dates, index=len(dates) - 1)
    st.subheader("Battery (BESS)")
    power = st.slider("Power (MW)", 50, 1000, int(info.get("bess_power_mw", BESS_POWER_MW)), 50)
    energy = st.slider("Energy (MWh)", 100, 5000, int(info.get("bess_energy_mwh", BESS_ENERGY_MWH)), 100)

tab1, tab2, tab3 = st.tabs(["Forecast", "BESS peak shaving", "Model comparison"])

d = preds[preds.index.date == day]

with tab1:
    fig = go.Figure()
    if model in ("lightgbm", "ensemble", "lstm") and {"q10", "q90"} <= set(d.columns):
        fig.add_trace(go.Scatter(x=d.index, y=d["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=d.index, y=d["q10"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(31,119,180,0.2)", name="10-90% band"))
    fig.add_trace(go.Scatter(x=d.index, y=d["actual"], name="Actual", line=dict(color="black", width=3)))
    fig.add_trace(go.Scatter(x=d.index, y=d[model], name=model, line=dict(color="#1f77b4", width=2)))
    fig.update_layout(yaxis_title="Load (MW)", hovermode="x unified", height=450)
    st.plotly_chart(fig, width="stretch")

    mape_day = (abs(d["actual"] - d[model]) / d["actual"]).mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("MAPE (this day)", f"{mape_day:.2f}%")
    c2.metric("Actual peak", f"{d['actual'].max():.0f} MW", f"hour {d['actual'].idxmax().hour}:00")
    c3.metric("Forecast peak", f"{d[model].max():.0f} MW", f"hour {d[model].idxmax().hour}:00")

with tab2:
    sched = optimize_peak_shaving(d[model].values, power_mw=power, energy_mwh=energy)
    r = apply_schedule(d["actual"].values, sched)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["actual"], name="Load without BESS", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=d.index, y=r["net_load"], name="Load with BESS", line=dict(color="green", width=3)))
    fig.add_trace(go.Bar(x=d.index, y=-sched["discharge"] + sched["charge"], name="Battery (+charge / -discharge)",
                         opacity=0.35, yaxis="y2"))
    fig.update_layout(yaxis_title="Load (MW)", yaxis2=dict(overlaying="y", side="right", title="Battery MW"),
                      hovermode="x unified", height=450)
    st.plotly_chart(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak before", f"{r['peak_before']:.0f} MW")
    c2.metric("Peak after", f"{r['peak_after']:.0f} MW")
    c3.metric("Reduction", f"{r['peak_reduction_mw']:.0f} MW", f"{r['peak_reduction_pct']:.2f}%")
    st.caption("The schedule is planned on the forecast and applied to the actual load, so the result reflects forecast error.")

with tab3:
    st.dataframe(metrics.style.format("{:.2f}"), width="stretch")
    for img in ("model_comparison.png", "shap_summary.png"):
        if (ARTIFACT_DIR / img).exists():
            st.image(str(ARTIFACT_DIR / img))
