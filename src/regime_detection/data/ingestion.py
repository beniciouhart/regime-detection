import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from fredapi import Fred

import cloudscraper
import re
from datetime import datetime

load_dotenv()

def fetch_eia_series_pet(series_ids=['RBRTE', 'RWTC'], # ID's of series we're pulling (Brent, WTI)
                          frequency= 'weekly',# Frequency of the spot prices, either daily, weekly, or monthly.
                            length = 5000 # Timespan that we're pulling, max 5000 weeks
                            ):
    EIA_API_KEY = os.getenv("EIA_API_KEY") #API Key to pull this data

    def series_line(series_ids) -> str: # Build the text used to specify what series we want
        text = [f"&facets[series][]={series}" for series in series_ids]
        return ''.join(text)


    URL_BASE = f"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key={EIA_API_KEY}&frequency={frequency}&data[0]=value{series_line(series_ids)}&sort[0][column]=period&sort[0][direction]=desc&length={length}"
    response = requests.get(url= URL_BASE) # Make request to EIA API
    data = response.json()

    df = pd.DataFrame(data['response']['data']).sort_index(ascending= False) # Extract data from JSON output
    df['period'] = pd.to_datetime(df['period']) # Clean the period (date) to datetime type
    df['value'] = pd.to_numeric(df['value'], errors= 'coerce') # Clean the value (spot price) to numeric
    df = df[['period', 'value', 'series']] # Keep only relevant columns

    df_wide = df.pivot(index='period', columns = 'series', values= 'value') # Create into pivot table, rows indexed by date, columns are each ticker, and values are spot prices
    df_wide = df_wide.dropna()
    return df_wide

def fetch_fred_series(series_ids=['T10Y2Y','VIXCLS','DTWEXBGS'], #ID's of the series we want to pull, here 10Y-2Y spread, VIX, and DXY
                      frequency = 'W-FRI'  # How frequent we want the samples to be. (day -> "D", weekly (Friday) -> "W-FRI", monthly -> "M", yearly -> "Y"
                      ):
    
    load_dotenv()
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    fred = Fred(api_key= FRED_API_KEY)
    series_dict = {}
    for series in series_ids: # Create a dict with key as ID and values as the corresponding series
        series_dict[series] = fred.get_series(series_id=series)

    df = pd.DataFrame(series_dict) # Turn dictionary into a DF
    df.index = pd.to_datetime(df.index) # Convert index dtype to datetime
    df.index.name = 'period' # Convert index name to 'period' to match the EIA information

    df = df.resample(rule=frequency).last() # Resample to keep only the days at the end of the week

    df = df.dropna() # Drop any NaNs

    return df

def fetch_eia_stock(series_ids=['WCESTUS1'], # ID's of series we're pulling (Week-end US Crude Inventory)
                          frequency= 'weekly',# Frequency of the spot prices, either daily, weekly, or monthly.
                            length = 5000 # Timespan that we're pulling, max 5000 weeks
                            ):
    load_dotenv()
    EIA_API_KEY = os.getenv("EIA_API_KEY") #API Key to pull this data

    def series_line(series_ids) -> str: # Build the text used to specify what series we want
        text = [f"&facets[series][]={series}" for series in series_ids]
        return ''.join(text)

    URL_BASE = f'https://api.eia.gov/v2/petroleum/stoc/wstk/data/?api_key={EIA_API_KEY}&frequency={frequency}&data[0]=value{series_line(series_ids)}&sort[0][column]=period&sort[0][direction]=desc&length={length}'

    response = requests.get(url= URL_BASE) # Make request to EIA API
    data = response.json()

    df = pd.DataFrame(data['response']['data']).sort_index(ascending= False)
    df['period'] = pd.to_datetime(df['period']) # Clean the period to datetime type
    df['value'] = pd.to_numeric(df['value'], errors= 'coerce') # Clean the value (spot price) to numeric
    df = df[['period', 'value', 'series']]

    df_wide = df.pivot(index='period', columns = 'series', values= 'value') # Create into pivot table
    df_wide = df_wide.dropna()
    return df_wide

def fetch_eia_series_gas(series_ids = ["RNGWHHD"], # ID's of series we're pulling (Henry Hub)
                         frequency= 'weekly', # Frequency of the spot prices, either daily, weekly, or monthly.
                         length = 5000): # Timespan that we're pulling, max 5000 weeks
    load_dotenv()
    EIA_API_KEY = os.getenv("EIA_API_KEY") #API Key to pull this data

    def series_line(series_ids) -> str: # Build the text used to specify what series we want
        text = [f"&facets[series][]={series}" for series in series_ids]
        return ''.join(text)
    

    URL_BASE = f"https://api.eia.gov/v2/natural-gas/pri/fut/data/?api_key={EIA_API_KEY}&frequency={frequency}&data[0]=value{series_line(series_ids)}&sort[0][column]=period&sort[0][direction]=desc&length={length}"
    response = requests.get(url= URL_BASE) # Make request to EIA API
    data = response.json()

    df = pd.DataFrame(data['response']['data']).sort_index(ascending= False)
    df['period'] = pd.to_datetime(df['period']) # Clean the period to datetime type
    df['value'] = pd.to_numeric(df['value'], errors= 'coerce') # Clean the value (spot price) to numeric
    df = df[['period', 'value', 'series']]

    df_wide = df.pivot(index='period', columns = 'series', values= 'value') # Create into pivot table
    df_wide = df_wide.dropna()
    return df_wide



def get_merged_df(eia_spot_pet_df=None, eia_spot_gas_df=None, eia_stock_df=None, fred_df=None):
    #Get data
    if eia_spot_pet_df is None:
        eia_spot_pet_df = fetch_eia_series_pet()
    if eia_spot_gas_df is None:
        eia_spot_gas_df = fetch_eia_series_gas()
    if eia_stock_df is None:
        eia_stock_df = fetch_eia_stock()
    if fred_df is None:
        fred_df = fetch_fred_series()

    # Merge data




    df = pd.merge(left=eia_spot_pet_df, right=fred_df,left_index = True, right_index = True, how= 'inner')
    df = pd.merge(left=df, right=eia_stock_df, left_index = True, right_index = True, how = 'inner')
    df = pd.merge(left=df, right= eia_spot_gas_df, left_index = True, right_index = True, how = 'inner' )
    df = df.copy().reset_index().sort_values('period')

    # Get and merge OPEC, FOMC meeting dates, and the time since a meeting (these are big events for the market)
        
    opec_dates = pd.read_csv('data/opec_meetings.csv', parse_dates=['date']).sort_values('date')
    fomc_dates = pd.read_csv('data/fomc_meetings.csv', parse_dates=['date']).sort_values('date')
    
    df = pd.merge_asof(df, opec_dates.assign(last_opec=opec_dates['date']),
                    left_on='period', right_on='date', direction='backward') #merge_asof used for timeseries data where the dates don't match, finds the closest date in the rightmost df and returns that value for all rows matches, here we use backward to get the most recent date, forward would get the soonest.
    df = pd.merge_asof(df, fomc_dates.assign(last_fomc=fomc_dates['date']), left_on='period', right_on='date', direction='backward')
    df['days_since_opec'] = (df['period'] - df['last_opec']).dt.days

    df['days_since_fomc'] = (df['period'] - df['last_fomc']).dt.days
    df = df.drop(columns=['date_x','date_y', 'last_fomc','last_opec'])
    
    return df.set_index('period')


