import pandas as pd
import numpy as np
from regime_detection.data.ingestion import get_merged_df


def compute_returns(df, series_ids= ['RBRTE','RWTC', 'RNGWHHD']):
    df = df.copy()
    for id in series_ids:
        df[id + '_log_return'] = np.log(df[id]/ df[id].shift(1))
    return df


def series_diff(df, series_ids=['WCESTUS1']):
    df = df.copy()
    for id in series_ids:
        df[id + '_1_w_change'] = df[id].diff()
    return df

def lagged_features(df, series_ids= ['RBRTE_log_return','RWTC_log_return', 'RNGWHHD_log_return', 'WCESTUS1_1_w_change'], lags=[1,4,12]):
    df = df.copy()
    for id in series_ids:
        for lag in lags:
            df[id + f'_{lag}_w_lag'] = df[id].shift(lag)
    return df

def rolling_vol(df, series_ids=['RBRTE_log_return','RWTC_log_return', 'RNGWHHD_log_return'], periods=[4,12]):
    df = df.copy()
    for id in series_ids:
        for period in periods:
            df[id + f'_{period}_returns_rol_vol'] = df[id].rolling(window=period,min_periods=1).std()
    return df

def differentials(df, series_ids=['RBRTE', 'RWTC']):
    df = df.copy()
    pairings = [(x,y) for i, x in enumerate(series_ids) for y in series_ids[i+1:]] 
    for pair in pairings:
        id_a , id_b = pair 
        df[f'{id_a}_{id_b}_diff'] = df[id_a] - df[id_b]
    return df

def rolling_z_scores(df, 
                     series_ids=['RBRTE_log_return','RWTC_log_return','RNGWHHD_log_return','RBRTE_RWTC_diff'],
                     window=63,
                     min_periods=1):
    df = df.copy()
    for id in series_ids:
      avg = df[id].rolling(window=window, min_periods=min_periods).mean()
      dev = df[id].rolling(window=window, min_periods=min_periods).std()
      df[id + '_rol_z_score'] = (df[id] - avg) / dev
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