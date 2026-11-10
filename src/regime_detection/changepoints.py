"""PELT changepoint detection, CROPS penalty search, and regime labeling."""

import numpy as np
import pandas as pd
import ruptures as rpt


def run_crops(signal, model="rbf", beta_min=None, beta_max=None):
    """Run CROPS-style binary search over PELT penalties.

    Returns
    -------
    dict[float, list[int]]  mapping penalty -> list of changepoint indices.
    """
    n = len(signal)
    if beta_min is None:
        beta_min = np.log(n)
    if beta_max is None:
        beta_max = 100 * np.log(n)

    algo = rpt.Pelt(model=model).fit(signal)
    results = {}

    def _search(lo, hi):
        if hi - lo < 1e-1:
            return
        if lo not in results:
            results[lo] = algo.predict(pen=lo)
        if hi not in results:
            results[hi] = algo.predict(pen=hi)
        if len(results[lo]) == len(results[hi]):
            return
        mid = (lo + hi) / 2.0
        results[mid] = algo.predict(pen=mid)
        _search(lo, mid)
        _search(mid, hi)

    _search(beta_min, beta_max)
    return results


def select_plateau_changepoints(crops_results, min_cpts = 1):
    """Pick changepoints from the widest penalty plateau (log scale), ignoring 0-changepoint solutions."""
    pens = sorted(crops_results) # Sort the crops results by their penalty value 
    counts = [len(crops_results[p]) - 1 for p in pens] # Returns the number of changepoint indices for the given penalty (-1 since they always include 0)
    best_pen, best_width, i = None, -1, 0 
    while i < len(pens): # Loop through each penalty
        j = i
        while j + 1 < len(pens) and counts[j + 1] == counts[i]: # Keep increasing the index while it's within bounds and the number of changepoints remains the same. 
            j += 1
        right = pens[j + 1] if j + 1 < len(pens) else pens[j] # Pick the penalty associated with the first time the number of changepoints changes
        width = np.log(right) - np.log(pens[i]) # Get the difference in penalties between the first time the changepoints changed and the initial penalty 
        if counts[i] > min_cpts and width > best_width: # If more than 1 changepoint and the width is the highest we've seen, set best penalty and best width
            best_pen, best_width = pens[(i + j) // 2], width
        i = j + 1
    return sorted(crops_results[best_pen])[:-1]


def label_regimes(df, changepoints):
    """Assign integer regime labels based on *changepoints* (row indices).

    Parameters
    ----------
    df : pd.DataFrame
        Must be positionally aligned (iloc-based boundaries).
    changepoints : list[int]
        Row-index boundaries (not label-based).

    Returns
    -------
    pd.DataFrame with a 'regime' column appended.
    """
    df = df.copy()
    boundaries = [0] + sorted(changepoints) + [len(df)]
    df["regime"] = 0
    for i in range(len(boundaries) - 1):
        df.iloc[boundaries[i]:boundaries[i + 1], df.columns.get_loc("regime")] = i
    return df


def regime_summary(df):
    """Print a summary table of regimes in *df*."""
    print(f"{'Regime':<10} {'Start':<15} {'End':<15} {'Obs':<10}")
    print("-" * 50)
    for rid in sorted(df["regime"].unique()):
        subset = df[df["regime"] == rid]
        print(
            f"{rid:<10} {str(subset.index[0].date()):<15} "
            f"{str(subset.index[-1].date()):<15} {len(subset):<10}"
        )
