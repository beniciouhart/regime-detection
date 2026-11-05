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


def select_plateau_changepoints(crops_results, exclude_terminal=True):
    """Pick the changepoint set from the widest penalty plateau.

    Parameters
    ----------
    crops_results : dict[float, list[int]]
        Output of :func:`run_crops`.
    exclude_terminal : bool
        Drop the final index (PELT's end-of-series marker).

    Returns
    -------
    list[int]  Sorted changepoint indices from the widest plateau.
    """
    # Group penalties by number of changepoints
    by_n = {}
    for pen in sorted(crops_results):
        cpts = crops_results[pen]
        n_cpts = len(cpts) - (1 if exclude_terminal else 0)
        by_n.setdefault(n_cpts, []).append(pen)

    # Widest plateau = largest range of penalties with the same count
    best_n = max(by_n, key=lambda k: max(by_n[k]) - min(by_n[k]))
    # Take the changepoints from the middle of the plateau
    mid_pen = sorted(by_n[best_n])[len(by_n[best_n]) // 2]
    cpts = sorted(crops_results[mid_pen])

    if exclude_terminal:
        cpts = cpts[:-1]  # drop PELT's terminal index

    return cpts


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
