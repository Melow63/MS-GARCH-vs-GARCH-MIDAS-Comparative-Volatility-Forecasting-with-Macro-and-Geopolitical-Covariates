"""Forecast evaluation and VaR backtests used in notebook 05 and in scripts/strict_oos_check.py."""
import numpy as np
from scipy.stats import chi2, t as t_dist


def kupiec_test(violation_rate, expected_rate, n_obs):
    """Kupiec (1995) proportion-of-failures LR test. Returns (LR, p-value)."""
    T, v, p = n_obs, violation_rate, expected_rate
    if v == 0 or v == 1:
        return np.nan, np.nan
    lr = -2 * (np.log((1 - p) ** (T * (1 - v)) * p ** (T * v)) -
               np.log((1 - v) ** (T * (1 - v)) * v ** (T * v)))
    return lr, 1 - chi2.cdf(lr, df=1)


def christoffersen_test(returns, var_forecast):
    """Christoffersen (1998) independence test on the violation sequence. Returns (LR, p-value)."""
    violations = (returns < var_forecast).astype(int).values
    n00 = ((violations[:-1] == 0) & (violations[1:] == 0)).sum()
    n01 = ((violations[:-1] == 0) & (violations[1:] == 1)).sum()
    n10 = ((violations[:-1] == 1) & (violations[1:] == 0)).sum()
    n11 = ((violations[:-1] == 1) & (violations[1:] == 1)).sum()
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    if pi == 0 or pi == 1 or pi01 == 0 or pi11 == 0:
        return np.nan, np.nan
    lr = -2 * (np.log((1 - pi) ** (n00 + n10) * pi ** (n01 + n11)) -
               np.log((1 - pi01) ** n00 * pi01 ** n01 * (1 - pi11) ** n10 * pi11 ** n11))
    return lr, 1 - chi2.cdf(lr, df=1)


def dm_test_hac(e1, e2, h=1):
    """Diebold-Mariano test on two loss series (negative statistic: model 1 has the lower loss)."""
    d = np.asarray(e1, dtype=float) - np.asarray(e2, dtype=float)
    T = len(d)
    d_bar = d.mean()
    gamma0 = ((d - d_bar) ** 2).mean()
    gammas = [((d[i:] - d_bar) * (d[:-i] - d_bar)).mean() for i in range(1, h + 1)]
    var_d = (gamma0 + 2 * sum(gammas)) / T
    dm = d_bar / np.sqrt(var_d)
    return dm, 2 * (1 - t_dist.cdf(abs(dm), df=T - 1))


def qlike(var_decimal, realized_r2):
    """QLIKE loss (Patton, 2011) on variances in return units (not percent)."""
    return (np.log(var_decimal) + realized_r2 / var_decimal).mean()
