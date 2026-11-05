"""Kupiec proportion-of-failures backtest."""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_test(n_obs, n_breaches, p_expected=0.05):
    """Kupiec proportion-of-failures LR test.

    Returns
    -------
    (lr_statistic, p_value, observed_breach_rate)
    """
    if n_obs == 0:
        return np.nan, np.nan, np.nan

    p_hat = n_breaches / n_obs

    if n_breaches == 0:
        lr = -2 * (n_obs * np.log(1 - p_expected))
        return lr, chi2.sf(lr, df=1), p_hat

    if n_breaches == n_obs:
        lr = -2 * (n_obs * np.log(p_expected))
        return lr, chi2.sf(lr, df=1), p_hat

    lr = -2 * (
        (n_obs - n_breaches) * np.log(1 - p_expected)
        + n_breaches * np.log(p_expected)
        - (n_obs - n_breaches) * np.log(1 - p_hat)
        - n_breaches * np.log(p_hat)
    )
    return lr, chi2.sf(lr, df=1), p_hat


def kupiec_regime_conditional(df, quantile=0.05):
    """Run Kupiec test on regime-conditional VaR for each regime and series.

    Expects *df* to have a 'regime' column and stationary return columns.

    Returns
    -------
    pd.DataFrame with columns: type, regime, series, n_obs, n_breaches,
    breach_rate, LR_stat, p_value, reject_H0.
    """
    tickers = ["RWTC", "RBRTE", "RNGWHHD"]
    return_cols = {t: f"{t}_log_return_stationary" for t in tickers}
    rows = []

    for rid in sorted(df["regime"].unique()):
        regime_data = df[df["regime"] == rid]
        for ticker in tickers:
            returns = regime_data[return_cols[ticker]]
            regime_var = returns.quantile(quantile)
            n_obs = len(returns)
            n_breaches = int((returns < regime_var).sum())
            lr, pval, obs_rate = kupiec_test(n_obs, n_breaches, quantile)
            rows.append({
                "type": "regime-conditional",
                "regime": rid,
                "series": ticker,
                "n_obs": n_obs,
                "n_breaches": n_breaches,
                "breach_rate": obs_rate,
                "LR_stat": lr,
                "p_value": pval,
                "reject_H0": pval < 0.05 if not np.isnan(pval) else None,
            })
    return pd.DataFrame(rows)


def kupiec_unconditional(df, windows=None, quantile=0.05):
    """Run Kupiec test on rolling-window VaR for each series and window.

    Returns
    -------
    pd.DataFrame with same schema as :func:`kupiec_regime_conditional`.
    """
    windows = windows or [52, 78, 104]
    tickers = ["RWTC", "RBRTE", "RNGWHHD"]
    return_cols = {t: f"{t}_log_return_stationary" for t in tickers}
    rows = []

    for ticker in tickers:
        for w in windows:
            var_col = f"{ticker}_{w}_week_var"
            valid = df[[return_cols[ticker], var_col]].dropna()
            n_obs = len(valid)
            n_breaches = int((valid[return_cols[ticker]] < valid[var_col]).sum())
            lr, pval, obs_rate = kupiec_test(n_obs, n_breaches, quantile)
            rows.append({
                "type": f"unconditional_{w}w",
                "regime": "all",
                "series": ticker,
                "n_obs": n_obs,
                "n_breaches": n_breaches,
                "breach_rate": obs_rate,
                "LR_stat": lr,
                "p_value": pval,
                "reject_H0": pval < 0.05 if not np.isnan(pval) else None,
            })
    return pd.DataFrame(rows)
