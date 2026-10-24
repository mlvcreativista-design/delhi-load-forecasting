"""Battery (BESS) peak-shaving optimisation on top of the load forecast.

For one day (24 hourly steps) we choose charge c_t and discharge d_t to minimise the
maximum net load P, using the *forecast*. We then apply that schedule to the *actual*
load to see the peak reduction you would really get.

    minimise  P
    s.t.      load_t - d_t + c_t <= P
              s_t = s_{t-1} + eta*c_t - d_t/eta      (state of charge, MWh)
              0 <= c_t, d_t <= power_mw ;  e_min <= s_t <= e_max
              s_T >= s_0                              (end the day at least as full)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .config import BESS_EFFICIENCY, BESS_ENERGY_MWH, BESS_POWER_MW


def optimize_peak_shaving(
    load_forecast,
    power_mw: float = BESS_POWER_MW,
    energy_mwh: float = BESS_ENERGY_MWH,
    eta: float = BESS_EFFICIENCY,
    soc0_frac: float = 0.5,
    soc_min_frac: float = 0.1,
    soc_max_frac: float = 0.9,
) -> dict:
    load = np.asarray(load_forecast, float)
    T = len(load)
    s0 = soc0_frac * energy_mwh
    smin, smax = soc_min_frac * energy_mwh, soc_max_frac * energy_mwh

    # variables: c[0:T], d[T:2T], s[2T:3T], P[3T]
    n = 3 * T + 1
    cost = np.zeros(n)
    cost[-1] = 1.0

    # inequality: -d_t + c_t - P <= -load_t
    A_ub = lil_matrix((T, n))
    b_ub = -load
    for t in range(T):
        A_ub[t, t] = 1.0
        A_ub[t, T + t] = -1.0
        A_ub[t, -1] = -1.0

    # equality: s_t - s_{t-1} - eta*c_t + d_t/eta = 0   (s_{-1} = s0)
    A_eq = lil_matrix((T, n))
    b_eq = np.zeros(T)
    for t in range(T):
        A_eq[t, 2 * T + t] = 1.0
        A_eq[t, t] = -eta
        A_eq[t, T + t] = 1.0 / eta
        if t == 0:
            b_eq[t] = s0
        else:
            A_eq[t, 2 * T + t - 1] = -1.0

    bounds = (
        [(0, power_mw)] * T
        + [(0, power_mw)] * T
        + [(smin, smax)] * (T - 1)
        + [(max(s0, smin), smax)]          # end-of-day SOC >= start
        + [(0, None)]
    )
    res = linprog(cost, A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"BESS optimisation failed: {res.message}")

    x = res.x
    return {
        "charge": x[:T],
        "discharge": x[T:2 * T],
        "soc": x[2 * T:3 * T],
        "planned_peak": x[-1],
    }


def apply_schedule(actual_load, schedule: dict) -> dict:
    """Apply a planned schedule to the real load and report the realised peak shaving."""
    actual = np.asarray(actual_load, float)
    net = actual - schedule["discharge"] + schedule["charge"]
    return {
        "net_load": net,
        "peak_before": float(actual.max()),
        "peak_after": float(net.max()),
        "peak_reduction_mw": float(actual.max() - net.max()),
        "peak_reduction_pct": float((actual.max() - net.max()) / actual.max() * 100),
    }


def daily_peak_shaving(pred, actual, **bess_kw):
    """Run the optimiser day by day. pred/actual: pd.Series with an hourly DatetimeIndex."""
    import pandas as pd

    rows = []
    for day, a in actual.groupby(actual.index.normalize()):
        if len(a) != 24:
            continue
        p = pred.loc[a.index]
        sched = optimize_peak_shaving(p.values, **bess_kw)
        r = apply_schedule(a.values, sched)
        rows.append({
            "date": day,
            "peak_before_mw": r["peak_before"],
            "peak_after_mw": r["peak_after"],
            "reduction_mw": r["peak_reduction_mw"],
            "reduction_pct": r["peak_reduction_pct"],
        })
    return pd.DataFrame(rows)
