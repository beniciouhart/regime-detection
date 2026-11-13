"""VaR backtests: Kupiec (breach rate) and Christoffersen (breach clustering)."""

TICKERS = ["RWTC", "RBRTE", "RNGWHHD"]

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


def christoffersen_independence(breaches):
    """Run Christoffersen Independence Test on binary breach series (0 if no VaR breach, 1 if VaR breach).

    Returns
    -------
    Likelihood Ratio Test Statistic, P-Value, Prob. breach given prev. week didn't breach VaR, Prob. breach given prev. week breached VaR
    """


    b = np.asarray(breaches, dtype=int)
    prev, curr = b[:-1], b[1:]
    n00 = np.sum((prev == 0) & (curr == 0))  
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    pi0 = n01 / (n00 + n01) # ratio of weeks with breach given previous didn't breach
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0 # ratio of weeks with breach given previous week breached
    pi = (n01 + n11) / len(prev)
    def ll(p, zeros, ones):
        return (zeros * np.log(1 - p) if zeros else 0) + (ones * np.log(p) if ones else 0)
    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    return lr, chi2.sf(lr, df=1), pi0, pi1


def backtest_table(df, windows=None, quantile=0.05):
    """Kupiec + Christoffersen for every series and rolling VaR window.

    One row per (series, window). Breaches are weeks where the return fell
    below that week's VaR.

    Returns
    -------
    pd.DataFrame with columns: series, window, n_obs, n_breaches, breach_rate,
    kupiec_p, christoffersen_p, p_breach_after_none (pi0),
    p_breach_after_breach (pi1).
    """
    windows = windows or [52, 78, 104]
    rows = []
    for t in TICKERS:
        ret = f"{t}_log_return_stationary"
        for w in windows:
            var = f"{t}_{w}_week_var"
            valid = df[[ret, var]].dropna()  # NaNs only at the start (VaR warm-up), so the weeks stay consecutive
            hits = (valid[ret] < valid[var]).astype(int)
            _, p_uc, rate = kupiec_test(len(hits), int(hits.sum()), quantile)
            _, p_ind, pi0, pi1 = christoffersen_independence(hits)
            rows.append({
                "series": t, "window": w, "n_obs": len(hits),
                "n_breaches": int(hits.sum()), "breach_rate": rate,
                "kupiec_p": p_uc, "christoffersen_p": p_ind,
                "p_breach_after_none": pi0, "p_breach_after_breach": pi1,
            })
    return pd.DataFrame(rows)


def breach_rate_by_group(df, group_col, window=52):
    """Breach rate of the rolling *window*-week VaR, split by a label column.

    *group_col* is either a single column name used for every series (e.g.
    the PELT ``"regime"`` column) or a format string with ``{t}`` in it
    (e.g. ``"{t}_vol_bucket"``) so each series is split by its own labels.
    Weeks with no label (bucket warm-up) are dropped.
    """
    rows = []
    for t in TICKERS:
        ret, var, grp = f"{t}_log_return_stationary", f"{t}_{window}_week_var", group_col.format(t=t)
        v = df[[ret, var, grp]].dropna().copy()
        v["hit"] = v[ret] < v[var]
        g = v.groupby(grp, observed=True).agg(
            weeks=("hit", "size"),
            breaches=("hit", "sum"),
            breach_rate=("hit", "mean"),
            ann_vol=(ret, lambda s: s.std() * np.sqrt(52)),
        )
        g.index.name = "group"
        g.insert(0, "series", t)
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)
