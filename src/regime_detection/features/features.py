import pandas as pd
import numpy as np
from regime_detection.data.ingestion import get_merged_df


def compute_returns(df, series_ids= ['RBRTE','RWTC', 'RNGWHHD']):
    df = df.copy()
    for series in series_ids:
        df[series + '_log_return'] = np.log(df[series]/ df[series].shift(1))
    return df


def series_diff(df, series_ids=['WCESTUS1']):
    df = df.copy()
    for series in series_ids:
        df[series + '_1_w_change'] = df[series].diff()
    return df

def lagged_features(df, series_ids= ['RBRTE_log_return','RWTC_log_return', 'RNGWHHD_log_return', 'WCESTUS1_1_w_change'], lags=[1,4,12]):
    df = df.copy()
    for series in series_ids:
        for lag in lags:
            df[series + f'_{lag}_w_lag'] = df[series].shift(lag)
    return df

def rolling_vol(df, series_ids=['RBRTE_log_return','RWTC_log_return', 'RNGWHHD_log_return'], periods=[4,12]):
    df = df.copy()
    for series in series_ids:
        for period in periods:
            df[series + f'_{period}_returns_rol_vol'] = df[series].rolling(window=period,min_periods=1).std()
    return df

def differentials(df, series_ids=['RBRTE', 'RWTC']):
    df = df.copy()
    pairings = [(x,y) for i, x in enumerate(series_ids) for y in series_ids[i+1:]] 
    for pair in pairings:
        series_a , series_b = pair 
        df[f'{series_a}_{series_b}_diff'] = df[series_a] - df[series_b]
    return df

def rolling_z_scores(df, 
                     series_ids=['RBRTE_log_return','RWTC_log_return','RNGWHHD_log_return','RBRTE_RWTC_diff'],
                     window=63,
                     min_periods=1):
    df = df.copy()
    for series in series_ids:
      avg = df[series].rolling(window=window, min_periods=min_periods).mean()
      dev = df[series].rolling(window=window, min_periods=min_periods).std()
      df[series + '_rol_z_score'] = (df[series] - avg) / dev
    return df

def feature_pipeline(df=None):
    if df is None:
        df = get_merged_df()
    df = df.copy()
    print("Computing log returns...")
    df = compute_returns(df)
    print("Computing week-on-week changes...")
    df = series_diff(df)
    print("Computing lagged values...")
    df = lagged_features(df)
    print("Computing rolling returns volatility...")
    df = rolling_vol(df)
    print("Computing commodity-to-commodity differences...")
    df = differentials(df)
    print("Computing rolling window z-scores...")
    df = rolling_z_scores(df)
    return df