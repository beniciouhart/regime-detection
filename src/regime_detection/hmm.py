# src/regime_detection/hmm.py

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd


def fit_hmm(returns: pd.Series, n_states: int = 2, n_fits: int = 10) -> dict:
    """
    Fit a Gaussian HMM to a return series.
    
    Runs n_fits random restarts, keeps the best by log-likelihood.
    Returns dict with model, labels, and posterior probabilities.
    """
    X = returns.dropna().values.reshape(-1, 1)
    
    best_score = -np.inf
    best_model = None
    
    for i in range(n_fits):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=i,
        )
        model.fit(X)
        score = model.score(X)
        if score > best_score:
            best_score = score
            best_model = model
    
    labels = best_model.predict(X)
    posteriors = best_model.predict_proba(X)
    
    # Reorder states so state 0 = lowest volatility
    state_vols = [np.sqrt(best_model.covars_[s][0, 0]) for s in range(n_states)]
    order = np.argsort(state_vols)
    
    label_map = {old: new for new, old in enumerate(order)}
    labels = np.array([label_map[l] for l in labels])
    posteriors = posteriors[:, order]
    
    return {
        "model": best_model,
        "labels": pd.Series(labels, index=returns.dropna().index, name="hmm_regime"),
        "posteriors": pd.DataFrame(
            posteriors,
            index=returns.dropna().index,
            columns=[f"state_{i}_prob" for i in range(n_states)],
        ),
        "state_means": best_model.means_[order].flatten(),
        "state_vols": np.array(state_vols)[order],
        "transition_matrix": best_model.transmat_[np.ix_(order, order)],
        "log_likelihood": best_score,
    }


def select_n_states(returns: pd.Series, max_states: int = 4, n_fits: int = 10) -> pd.DataFrame:
    """
    Fit HMMs with 2..max_states, return BIC for each.
    Pick the K with lowest BIC.
    """
    X = returns.dropna().values.reshape(-1, 1)
    n = len(X)
    results = []
    
    for k in range(2, max_states + 1):
        result = fit_hmm(returns, n_states=k, n_fits=n_fits)
        # Number of free params: k*k transition + k means + k variances + (k-1) initial
        n_params = k * k + k + k + (k - 1)
        bic = -2 * result["log_likelihood"] + n_params * np.log(n)
        results.append({"n_states": k, "log_likelihood": result["log_likelihood"],
                        "bic": bic, "n_params": n_params})
    
    return pd.DataFrame(results)
