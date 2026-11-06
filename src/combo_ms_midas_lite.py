# MS-GARCH-MIDAS "lite" combination -- ATTEMPTED, NOT RETAINED in the final report
#
# Idea: two-step approach (not joint estimation, too heavy for the timeline)
#   1. Take the long-run component tau_t from the best GARCH-MIDAS (NFCI+EPU)
#      --> in OOS forecasting it is constant within each month (h_pred)
#   2. Apply it as a multiplicative scaling factor on the per-regime short-run variance
#      from MS-GARCH(2): h_combo(k) = tau_tilde_t * h_MS2(k)
#      (tau_tilde = tau normalized to unit mean, so it is a pure macro modulation)
#   3. Recompute RMSE / QLIKE / mixture VaR
#
# RESULT: does not work. RMSE worse than either MS-GARCH(2) alone or GARCH-MIDAS alone
# and VaR fails both Kupiec tests. The issue is structural: regime probabilities are
# frozen (not re-estimated with tau), so the macro factor and regime dynamics pull
# in incoherent directions. A proper joint estimation (Hamilton filter + MIDAS
# long-run component) would be needed left as future work See Section 6.1 of the report

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.optimize import brentq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

DATA  = os.path.join(ROOT, "data", "processed", "merged_data.csv")
MS2 = os.path.join(ROOT, "results", "ms_garch2_probabilities.csv")
MIDAS = os.path.join(ROOT, "results", "midas_oos_forecast.csv")


def mixture_var(h_regimes, probs, alpha, mu=0.0, scale=100.0):
    """Mixture VaR (same function as in 05_forecasting.ipynb, copied here
    so this script runs standalone without depending on the notebook)"""
    h_regimes = np.asarray(h_regimes, dtype=float)
    probs = np.asarray(probs, dtype=float)
    T = len(h_regimes)
    sig = np.sqrt(h_regimes)
    w = probs / probs.sum(axis=1, keepdims=True)
    out = np.empty(T)
    for t in range(T):
        s = sig[t]
        pw = w[t]

        def f(x):
            return np.sum(pw * norm.cdf((x - mu) / s)) - alpha

        lo = mu - 10.0 * s.max()
        while f(lo) > 0:
            lo *= 2.0
        out[t] = brentq(f, lo, mu, xtol=1e-10)
    return out / scale


def kupiec(v, alpha, T):
    if v == 0 or v == 1:
        return np.nan, np.nan
    lr = -2 * (np.log((1-alpha)**(T*(1-v)) * alpha**(T*v)) -
               np.log((1-v)**(T*(1-v)) * v**(T*v)))
    return lr, 1 - chi2.cdf(lr, 1)


def main():
    data = pd.read_csv(DATA, index_col="Date", parse_dates=True).dropna(subset=["log_return"])
    split = pd.Timestamp("2023-01-01")
    test = data.loc[split:]
    returns_test = test["log_return"]
    realized = returns_test**2

    ms2 = pd.read_csv(MS2, index_col="Date", parse_dates=True).reindex(data.index, method="ffill")
    h2 = ms2[["h_Regime_1", "h_Regime_2"]].loc[test.index].values
    p2 = ms2[["smoothed_Regime_1", "smoothed_Regime_2"]].loc[test.index].values

    midas = pd.read_csv(MIDAS, parse_dates=["date"]).set_index("date")
    tau = midas["h_pred"].reindex(test.index)
    tau_tilde = (tau / tau.mean()).values[:, None]   
    # macro modulation unit mean

    # combined variance per regime
    h_combo = tau_tilde * h2

    # point forecast (aggregate variance) for RMSE / QLIKE
    combo_var = pd.Series((h_combo * p2).sum(axis=1) / 10000, index=test.index)
    rmse = np.sqrt(((combo_var - realized)**2).mean()) * 1e4
    qlike = (np.log(combo_var) + realized.clip(lower=1e-12) / combo_var).mean()
    print(f"COMBO lite  RMSE = {rmse:.3f}e-4   QLIKE = {qlike:.4f}")
    print("(ref: MS-GARCH(2) RMSE~3.26, GARCH-MIDAS RMSE~3.61, GARCH RMSE~3.49)")

    # mixture VaR
    for cl, a in [("95%", 0.05), ("99%", 0.01)]:
        var = pd.Series(mixture_var(h_combo, p2, a), index=test.index)
        var_al = var.shift(1).dropna()
        ret = returns_test.reindex(var_al.index)
        nv = int((ret < var_al).sum())
        T = len(var_al)
        lr, pv = kupiec(nv/T, a, T)
        print(f"COMBO {cl}: {nv}/{T} = {nv/T*100:.2f}% Kupiec p = {pv:.3f}")

    print("\n=> conclusion: combo not valid (degraded RMSE + VaR rejected), not retained.")


if __name__ == "__main__":
    main()
