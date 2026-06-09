import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os


def compute_returns(df, series_ids= ['RBRTE','RWTC']):
    df = df.copy()
    for id in series_ids:
        df[id + '_log_return'] = np.log(df[id]/ df[id].shift(1))
    return df.dropna()
