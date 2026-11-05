"""Feature engineering: returns, diffs, lags, rolling vol, spreads, z-scores."""

import numpy as np
import pandas as pd

from regime_detection.ingestion import get_merged_df


def compute_returns(df, series_ids=None):
    series_ids = series_ids or ["RBRTE", "RWTC", "RNGWHHD"]
    df = df.copy()
    for s in series_ids:
        df[f"{s}_log_return"] = np.log(df[s] / df[s].shift(1))
    return df


def series_diff(df, series_ids=None):
    series_ids = series_ids or ["WCESTUS1"]
    df = df.copy()
    for s in series_ids:
        df[f"{s}_1_w_change"] = df[s].diff()
    return df


def lagged_features(df, series_ids=None, lags=None):
    series_ids = series_ids or [
        "RBRTE_log_return", "RWTC_log_return",
        "RNGWHHD_log_return", "WCESTUS1_1_w_change",
    ]
    lags = lags or [1, 4, 12]
    df = df.copy()
    for s in series_ids:
        for lag in lags:
            df[f"{s}_{lag}_w_lag"] = df[s].shift(lag)
    return df


def rolling_vol(df, series_ids=None, periods=None):
    series_ids = series_ids or [
        "RBRTE_log_return", "RWTC_log_return", "RNGWHHD_log_return",
    ]
    periods = periods or [4, 12]
    df = df.copy()
    for s in series_ids:
        for p in periods:
            df[f"{s}_{p}_returns_rol_vol"] = df[s].rolling(window=p).std()
    return df


def differentials(df, series_ids=None):
    series_ids = series_ids or ["RBRTE", "RWTC"]
    df = df.copy()
    pairings = [(x, y) for i, x in enumerate(series_ids) for y in series_ids[i + 1:]]
    for a, b in pairings:
        df[f"{a}_{b}_diff"] = df[a] - df[b]
    return df


def rolling_z_scores(df, series_ids=None, window=63):
    series_ids = series_ids or [
        "RBRTE_log_return", "RWTC_log_return",
        "RNGWHHD_log_return", "RBRTE_RWTC_diff",
    ]
    df = df.copy()
    for s in series_ids:
        avg = df[s].rolling(window=window).mean()
        dev = df[s].rolling(window=window).std()
        df[f"{s}_rol_z_score"] = (df[s] - avg) / dev
    return df


def feature_pipeline(df=None):
    """Run the full feature pipeline and return the augmented DataFrame."""
    if df is None:
        df = get_merged_df()
    df = df.copy()
    df = compute_returns(df)
    df = series_diff(df)
    df = lagged_features(df)
    df = rolling_vol(df)
    df = differentials(df)
    df = rolling_z_scores(df)
    return df
