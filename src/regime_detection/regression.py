"""OLS and Rolling OLS regression with HAC diagnostics."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson


# ---------------------------------------------------------------------------
# Static OLS
# ---------------------------------------------------------------------------

def fit_ols(y, X, hac_maxlags=4):
    """Fit OLS and HAC-corrected OLS, returning both result objects.

    Returns
    -------
    dict with keys 'ols', 'hac', 'residuals'.
    """
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    ols_fit = model.fit()
    hac_fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
    return {
        "ols": ols_fit,
        "hac": hac_fit,
        "residuals": ols_fit.resid,
    }


def fit_commodity_models(df, feature_cols, hac_maxlags=4):
    """Fit OLS for WTI, Brent, and Henry Hub returns against *feature_cols*.

    Returns
    -------
    dict[str, dict] keyed by series name ('wti', 'brent', 'henry').
    """
    targets = {
        "wti": "RWTC_log_return_stationary",
        "brent": "RBRTE_log_return_stationary",
        "henry": "RNGWHHD_log_return_stationary",
    }
    results = {}
    for name, col in targets.items():
        results[name] = fit_ols(df[col], df[feature_cols], hac_maxlags=hac_maxlags)
    return results


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def ljung_box(residuals, lags=None):
    """Return Ljung-Box test DataFrame."""
    lags = lags or [4, 12]
    return acorr_ljungbox(residuals, lags=lags, return_df=True)


def durbin_watson_stat(residuals):
    return durbin_watson(residuals)


def significant_coefficients(fit_result, alpha=0.05):
    """Return names of coefficients with |t| > critical value."""
    from scipy.stats import norm
    z_crit = norm.ppf(1 - alpha / 2)
    return [
        name for name in fit_result.model.exog_names
        if abs(fit_result.tvalues[name]) > z_crit
    ]


# ---------------------------------------------------------------------------
# Rolling OLS
# ---------------------------------------------------------------------------

def fit_rolling_ols(y, X, window=78, hac_maxlags=4):
    """Fit Rolling OLS with HAC standard errors.

    Returns
    -------
    statsmodels RollingOLSResults
    """
    X = sm.add_constant(X)
    model = RollingOLS(y, X, window=window)
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})


def fit_rolling_commodity_models(df, feature_cols, windows=None, hac_maxlags=4):
    """Fit rolling OLS across series and windows.

    Returns
    -------
    dict[(series, window)] -> RollingOLSResults
    """
    windows = windows or [52, 78, 104]
    targets = {
        "wti": "RWTC_log_return_stationary",
        "brent": "RBRTE_log_return_stationary",
        "henry": "RNGWHHD_log_return_stationary",
    }
    results = {}
    for name, col in targets.items():
        for w in windows:
            results[(name, w)] = fit_rolling_ols(
                df[col], df[feature_cols], window=w, hac_maxlags=hac_maxlags,
            )
    return results
