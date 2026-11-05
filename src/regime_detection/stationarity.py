"""Stationarity testing (ADF + KPSS) and transformation pipeline."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series, regression="c"):
    """Augmented Dickey-Fuller test.  H0: unit root (non-stationary)."""
    series = series.dropna()
    stat, pvalue, used_lag, nobs, crit, _ = adfuller(
        series, autolag="AIC", regression=regression,
    )
    return {
        "test": "adf",
        "statistic": stat,
        "pvalue": pvalue,
        "lags": used_lag,
        "nobs": nobs,
        "regression": regression,
        "is_stationary": pvalue < 0.05,
    }


def kpss_test(series, regression="c"):
    """KPSS test.  H0: stationary (or trend-stationary when regression='ct')."""
    series = series.dropna()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stat, pvalue, used_lag, crit = kpss(
            series, regression=regression, nlags="auto",
        )
        interp_warning = any("InterpolationWarning" in str(wi.category) for wi in w)
    return {
        "test": "kpss",
        "statistic": stat,
        "pvalue": pvalue,
        "lag": used_lag,
        "regression": regression,
        "is_stationary": pvalue >= 0.05,
        "pvalue_clipped": interp_warning,
    }


def classify_stationarity(series, regression="ct"):
    """Run both ADF and KPSS and return a joint verdict."""
    adf_res = adf_test(series, regression)
    kpss_res = kpss_test(series, regression)
    adf_stat = adf_res["is_stationary"]
    kpss_stat = kpss_res["is_stationary"]

    if adf_stat and kpss_stat:
        verdict = "stationary"
    elif not adf_stat and kpss_stat:
        verdict = "trend_stationary"
    elif adf_stat and not kpss_stat:
        verdict = "difference_stationary"
    else:
        verdict = "non_stationary"

    return {
        "adf_pvalue": adf_res["pvalue"],
        "adf_stationary": adf_stat,
        "kpss_pvalue": kpss_res["pvalue"],
        "kpss_stationary": kpss_stat,
        "kpss_pvalue_clipped": kpss_res["pvalue_clipped"],
        "verdict": verdict,
    }


def transform_and_revalidate(series, verdict, regression="c"):
    """Apply the appropriate transform and re-test stationarity."""
    if verdict == "stationary":
        transformed, method = series, "none"

    elif verdict == "trend_stationary":
        t = np.arange(len(series))
        fitted = sm.OLS(series, sm.add_constant(t)).fit().fitted_values
        transformed, method = series - fitted, "detrended"

    elif verdict in ("non_stationary", "difference_stationary"):
        transformed = series.diff().dropna()
        method = "differenced"

    else:
        raise ValueError(f"Unknown verdict: {verdict}")

    post = classify_stationarity(transformed, regression=regression)
    return transformed, method, post


def stationarity_pipeline(
    df,
    series_ids=None,
    regression="c",
):
    """Classify and transform each series, appending *_stationary columns."""
    series_ids = series_ids or [
        "RBRTE_log_return", "RWTC_log_return", "RNGWHHD_log_return",
        "WCESTUS1_1_w_change", "T10Y2Y", "VIXCLS", "DTWEXBGS",
    ]
    df = df.copy()

    for s in series_ids:
        pre = classify_stationarity(df[s])
        transformed, method, post = transform_and_revalidate(
            df[s], verdict=pre["verdict"], regression=regression,
        )
        print(f"{s}: {method} -> {post['verdict']}")
        df[f"{s}_stationary"] = transformed

    return df
