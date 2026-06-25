import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
import warnings

def adf_test(series,regression):
    """Augmented Dickey-Fuller: H0 = Non-Stationary. 
                         Reject H0 -> Stationary"""
    series = series.dropna()

    stat, pvalue, used_lag, nobs, crit, _ = adfuller(series,autolag='AIC', 
                                                     regression=regression)
    return {
        'test' : 'adf',
        'statistic' : stat,
        'pvalue' : pvalue,
        'lags' : used_lag,
        'nobs' : nobs,
        'regression' : regression,
        'is_stationary' : pvalue < 0.05,
    }

def kpss_test(series, regression):
    """KPSS Test: H0: Stationary or Trend-Stationary (Value is a function of time). 
           Reject H0 -> Non-Stationary"""
    
    series = series.dropna()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        stat, pvalue, used_lag, crit = kpss(series, regression=regression, nlags='auto')
        interp_warning = any('InterpolationWarning' in str(wi.category) for wi in w)
    return {
        'test': "kpss",
        'statistic' : stat,
        'pvalue' : pvalue,
        'lag' : used_lag,
        'regression' : regression,
        'is_stationary' : pvalue >= 0.05, # P-Val >= 5% so that we fail to reject the null
        'pvalue_clipped' : interp_warning,
    }

def classify_stationarity(series, regression='ct'):
    adf_res = adf_test(series, regression)
    kpss_res = kpss_test(series, regression)
    adf_stat, kpss_stat = adf_res['is_stationary'], kpss_res['is_stationary']

    if adf_stat and kpss_stat:
        verdict = 'stationary'
    elif not adf_stat and kpss_stat:
        verdict = 'trend_stationary'
    elif adf_stat and not kpss_stat:
        verdict = 'difference_stationary'
    else:
        verdict = 'non_stationary'
    return {
        'adf_pvalue' : adf_res['pvalue'],
        'adf_stationary' : adf_res['is_stationary'],
        'kpss_pvalue' : kpss_res['pvalue'],
        'kpss_stationary' : kpss_res['is_stationary'],
        'kpss_pvalue_clipped' : kpss_res['pvalue_clipped'],
        'verdict' : verdict,
    }

def transform_and_revalidate(series, verdict, regression='c'):
    """ 
    Takes in a series and it's stattionarity verdict.

    It returns a series transformed with regards to the verdict,
    along with what transformation was applied.

    """
    if verdict == 'stationary':
        transformed, method = series, 'none'

    elif verdict == 'trend_stationary':
        # Create an array of the index (time) of each row in the series
        t = np.arange(len(series))
        """
        Fit an OLS model with series values as dependent variable 
        and the time as the independent var.

        Subtract the line from the actual series to 'detrend' it.
        
        """
        resid = series - sm.OLS(series, sm.add_constant(t)).fit().fitted_values
        transformed, method = resid, 'detrended'

    elif verdict in ('non_stationary', 'difference_stationary'):
        # Simply difference to remove unit root.
        transformed = series.diff().dropna()
        transformed, method =  transformed, 'differenced'

    """Here, we change the regression mode to constant (c), since,
       by differencing and detrending these series,we don't have a
       'trend' to regress on, which constant + trend (ct) focuses on."""

    post = classify_stationarity(transformed, regression=regression)

    return transformed, method, post
    



def stationarity_pipeline(df, series_ids =['RBRTE_log_return', 'RWTC_log_return',
                                       'RNGWHHD_log_return','WCESTUS1_1_w_change', 
                                       'T10Y2Y', 'VIXCLS', 'DTWEXBGS'], 
                                       regression = 'c'):
    df = df.copy()
    
    for series in series_ids:

        # Check stationarity / non-stationarity type.

        stationary = classify_stationarity(df[series])

        # Transform the series (detrend or difference) in order to make it stationary (if it's not already).

        transform, method, post = transform_and_revalidate(df[series], verdict = stationary['verdict'], regression=regression)

        # Print the result from the transformation so the user can evaluate the results.

        print(f"{series} was transformed using {method} and it's stationarity class is now {post['verdict']}")

        # Append column to dataframe.

        df[series + '_stationary'] = transform

    return df


        

    
    