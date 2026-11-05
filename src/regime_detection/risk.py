"""VaR and CVaR computation (unconditional and regime-conditional)."""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Unconditional (rolling-window) VaR
# ---------------------------------------------------------------------------

def rolling_var(df, series_col, window, quantile=0.05):
    """Compute rolling historical-simulation VaR.

    Returns a Series of the rolling *quantile* of *series_col*.
    """
    return df[series_col].rolling(window).quantile(quantile)


def add_rolling_var_columns(df, windows=None, quantile=0.05):
    """Add rolling VaR columns for WTI, Brent, Henry Hub across *windows*.

    Columns are named ``{ticker}_{window}_week_var``.
    """
    windows = windows or [52, 78, 104]
    targets = {
        "RWTC": "RWTC_log_return_stationary",
        "RBRTE": "RBRTE_log_return_stationary",
        "RNGWHHD": "RNGWHHD_log_return_stationary",
    }
    df = df.copy()
    for ticker, col in targets.items():
        for w in windows:
            df[f"{ticker}_{w}_week_var"] = rolling_var(df, col, w, quantile)
    return df


# ---------------------------------------------------------------------------
# Regime-conditional VaR / CVaR
# ---------------------------------------------------------------------------

def regime_conditional_risk(df, quantile=0.05):
    """Compute VaR and CVaR within each regime for each series.

    Expects *df* to have a 'regime' column.

    Returns
    -------
    pd.DataFrame with columns: regime, series, VaR_5pct, CVaR, n_obs,
    start_date, end_date.
    """
    targets = {
        "RWTC": "RWTC_log_return_stationary",
        "RBRTE": "RBRTE_log_return_stationary",
        "RNGWHHD": "RNGWHHD_log_return_stationary",
    }
    rows = []
    for rid in sorted(df["regime"].unique()):
        regime_data = df[df["regime"] == rid]
        for ticker, col in targets.items():
            returns = regime_data[col]
            var = returns.quantile(quantile)
            cvar = returns[returns <= var].mean()
            rows.append({
                "regime": rid,
                "series": ticker,
                "VaR_5pct": var,
                "CVaR": cvar,
                "n_obs": len(returns),
                "start_date": regime_data.index[0].date(),
                "end_date": regime_data.index[-1].date(),
            })
    return pd.DataFrame(rows)


def unconditional_risk(df, quantile=0.05):
    """Compute full-sample VaR and CVaR for each series."""
    targets = {
        "RWTC": "RWTC_log_return_stationary",
        "RBRTE": "RBRTE_log_return_stationary",
        "RNGWHHD": "RNGWHHD_log_return_stationary",
    }
    rows = []
    for ticker, col in targets.items():
        returns = df[col]
        var = returns.quantile(quantile)
        cvar = returns[returns <= var].mean()
        rows.append({"series": ticker, "VaR_5pct": var, "CVaR": cvar})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Breach rates
# ---------------------------------------------------------------------------

def compute_breach_rates(df, windows=None):
    """Compute unconditional breach rates for rolling-window VaR.

    Returns
    -------
    pd.DataFrame with columns: ticker, window, breach_rate.
    """
    windows = windows or [52, 78, 104]
    tickers = ["RWTC", "RBRTE", "RNGWHHD"]
    return_cols = {t: f"{t}_log_return_stationary" for t in tickers}
    rows = []
    for ticker in tickers:
        for w in windows:
            var_col = f"{ticker}_{w}_week_var"
            breaches = df[return_cols[ticker]] < df[var_col]
            rows.append({
                "ticker": ticker,
                "window": w,
                "breach_rate": breaches.sum() / breaches.count(),
            })
    return pd.DataFrame(rows)
