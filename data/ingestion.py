import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
from typing import List

def fetch_eia_spot_prices(series_ids=['RBRTE', 'RWTC'], # ID's of series we're pulling (Brent, WTI)
                          frequency= 'weekly',# Frequency of the spot prices, either daily, weekly, or monthly.
                            length = 5000 # Timespan that we're pulling, max 5000 weeks
                            ):
    load_dotenv()
    EIA_API_KEY = os.getenv("EIA_API_KEY") #API Key to pull this data

    def series_line(series_ids) -> str: # Build the text used to specify what series we want
        text = [f"&facets[series][]={id}" for id in series_ids]
        return ''.join(text)


    URL_BASE = f"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key={EIA_API_KEY}&frequency={frequency}&data[0]=value{series_line(series_ids)}&sort[0][column]=period&sort[0][direction]=desc&length={length}"
    response = requests.get(url= URL_BASE) # Make request to EIA API
    data = response.json()

    df = pd.DataFrame(data['response']['data']).sort_index(ascending= False)
    df['period'] = pd.to_datetime(df['period']) # Clean the period to datetime type
    df['value'] = pd.to_numeric(df['value'], errors= 'coerce') # Clean the value (spot price) to numeric
    df = df[['period', 'value', 'series']]

    df_wide = df.pivot(index='period', columns = 'series', values= 'value') # Create into pivot table
    df_wide = df_wide.dropna()
    return df_wide