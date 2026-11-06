# Mixture VaR for regime-switching models (MS-GARCH)
#
# The predictive density of an MS-GARCH is NOT a single Gaussian
# with variance equal to the weighted sum of regime variances.
# It is a Gaussian mixture, one component per regime,
# weighted by the predicted regime probability.
# Computing VaR therefore requires inverting the mixture CDF,
# not just applying -z * sqrt(aggregate_var).
#
# Mixture CDF: F(x) = sum_k pi_k * Phi((x - mu) / sigma_k)
# We find x such that F(x) = alpha via bisection (brentq)

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def mixture_var(h_regimes, probs, alpha, mu=0.0, scale=100.0):
    """
    Mixture VaR, computed day by day.

    h_regimes : array (T, K) = per-regime variances, in (return * scale)^2 units
    probs     : array (T, K) = predicted regime probabilities (each row sums to ~1)
    alpha     : level (0.01 for 99%, 0.05 for 95%)
    mu        : mean (in *scale units). default 0 as in the GARCH benchmark
    scale     : return scaling factor (100 throughout this project)

    returns array (T,) of VaR at the return scale (negative values)
    """
    h_regimes = np.asarray(h_regimes, dtype=float)
    probs = np.asarray(probs, dtype=float)
    T = len(h_regimes)

    sig = np.sqrt(h_regimes)                       # per-regime std dev (scale *scale)
    w = probs / probs.sum(axis=1, keepdims=True)   # renormalize just in case

    out = np.empty(T)
    for t in range(T):
        s = sig[t]
        pw = w[t]

        # F(x) - alpha: the function whose root we seek
        def f(x):
            return np.sum(pw * norm.cdf((x - mu) / s)) - alpha

        # bracket: the root is necessarily negative (left tail)
        lo = mu - 10.0 * s.max()
        while f(lo) > 0:      # in case 10 sigma is not enough
            lo *= 2.0
        hi = mu
        out[t] = brentq(f, lo, hi, xtol=1e-10)

    return out / scale        # rescale back to return units


def var_single(vol, alpha):
    """Standard Gaussian VaR (for GARCH and GARCH-MIDAS with single predictive density).
    vol = conditional std dev at the return scale."""
    return norm.ppf(alpha) * vol      # norm.ppf(0.01) is already negative
