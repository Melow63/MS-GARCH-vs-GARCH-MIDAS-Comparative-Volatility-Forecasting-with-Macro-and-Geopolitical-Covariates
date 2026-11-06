"""Strict out-of-sample check.

In notebooks 04 and 05, GARCH(1,1) and MS-GARCH are estimated once on the full sample (2010-2025) and then
evaluated on 2023-2025. Here every parameter is estimated on 2010-2022 only, and the test period is filtered
forward with those fixed parameters, using one-step-ahead predicted regime probabilities (no smoothing).
GARCH-MIDAS is unchanged: it is already re-estimated each month on past data only (R/mfGARCH_forecasting.R).

Run from the repository root:  python scripts/strict_oos_check.py [--starts2 N] [--starts3 N]
Writes results/strict_oos_check.txt
"""
import argparse, sys, os, io
import numpy as np, pandas as pd
from arch import arch_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ms_garch2_from_scratch import MSGARCH2
from ms_garch3_from_scratch import MSGARCH3
from mixture_var import mixture_var, var_single
from backtests import kupiec_test, christoffersen_test, dm_test_hac, qlike

ap = argparse.ArgumentParser()
ap.add_argument("--starts2", type=int, default=25)
ap.add_argument("--starts3", type=int, default=0, help="0 skips MS-GARCH(3), which is slow (15 parameters)")
args = ap.parse_args()

SPLIT = pd.Timestamp("2023-01-01")
data = pd.read_csv(os.path.join(ROOT, "data", "processed", "merged_data.csv"),
                   index_col="Date", parse_dates=True).dropna(subset=["log_return"])
train, test = data[data.index < SPLIT], data[data.index >= SPLIT]
r_test = test["log_return"]; r2 = r_test ** 2
log = io.StringIO()
def say(s=""):
    print(s); log.write(s + "\n")

# GARCH(1,1): parameters from the training sample only, then filtered through the test sample
am = arch_model(data["log_return"] * 100, vol="GARCH", p=1, q=1, rescale=False)
res_g = arch_model(train["log_return"] * 100, vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
vol_g = pd.Series(am.fix(res_g.params).conditional_volatility, index=data.index).loc[test.index] / 100
var_g = vol_g ** 2

# MS-GARCH(2) and (3): fit on train, filter on all data with the fixed parameters
def ms_forecast(cls, n_starts, maxiter, seed):
    m_tr = cls(returns=train["log_return"].reset_index(drop=True), dates=train.index.to_series().reset_index(drop=True))
    res = m_tr.fit(n_starts=n_starts, maxiter=maxiter, random_state=seed, verbose=False)
    m_all = cls(returns=data["log_return"].reset_index(drop=True), dates=data.index.to_series().reset_index(drop=True))
    filt = m_all.hamilton_filter(res["params"])
    h = pd.DataFrame(filt["h"], index=data.index).loc[test.index].values
    p = pd.DataFrame(filt["predicted"], index=data.index).loc[test.index].values
    return res, h, p
res2, h2, p2 = ms_forecast(MSGARCH2, args.starts2, 2000, 123)
var_ms2 = pd.Series((h2 * p2).sum(axis=1) / 1e4, index=test.index)
run3 = args.starts3 > 0
if run3:
    res3, h3, p3 = ms_forecast(MSGARCH3, args.starts3, 600, 42)
    var_ms3 = pd.Series((h3 * p3).sum(axis=1) / 1e4, index=test.index)

midas = pd.read_csv(os.path.join(ROOT, "results", "midas_oos_forecast.csv"), parse_dates=["date"]).set_index("date")
var_mid = (midas["h_pred"] / 1e4).reindex(test.index)

say("Strict out-of-sample check: parameters estimated on 2010-2022, test 2023-2025")
say(f"MS-GARCH(2) train fit: loglik={res2['loglik']:.1f} p11={res2['params']['p11']:.3f} p22={res2['params']['p22']:.3f}")
if run3:
    say(f"MS-GARCH(3) train fit: loglik={res3['loglik']:.1f}  success={res3['success']}")
else:
    say("MS-GARCH(3) skipped (use --starts3 N to include it)")
say()
say("Point forecasts (variance vs squared return)")
losses = {}
models = [("GARCH(1,1)", var_g), ("MS-GARCH(2)", var_ms2)] + ([("MS-GARCH(3)", var_ms3)] if run3 else []) + [("GARCH-MIDAS", var_mid)]
for name, v in models:
    v = v.reindex(test.index)
    rmse = np.sqrt(((v - r2) ** 2).mean()) * 1e4
    losses[name] = ((v - r2) ** 2).values
    say(f"{name:12s} RMSE={rmse:.3f}e-4  QLIKE={qlike(v, r2):.4f}")
say()
say("Diebold-Mariano (HAC h=1), negative = first model has the lower squared loss")
pairs = [("MS-GARCH(2)", "GARCH(1,1)"), ("GARCH-MIDAS", "GARCH(1,1)"), ("MS-GARCH(2)", "GARCH-MIDAS")]
if run3:
    pairs += [("MS-GARCH(3)", "GARCH(1,1)"), ("MS-GARCH(3)", "GARCH-MIDAS"), ("MS-GARCH(3)", "MS-GARCH(2)")]
for a, b in pairs:
    dm, pv = dm_test_hac(losses[a], losses[b])
    say(f"{a} vs {b:12s} DM={dm:7.3f}  p={pv:.3f}")
say()
say("VaR backtest (one-step-ahead; MS-GARCH by mixture-CDF inversion with predicted probabilities)")
for cl, a in [("95%", 0.05), ("99%", 0.01)]:
    vars_ = {
        "GARCH(1,1)": var_single(vol_g, a).shift(1).dropna(),
        "MS-GARCH(2)": pd.Series(mixture_var(h2, p2, a), index=test.index).shift(1).dropna(),
        "GARCH-MIDAS": var_single(np.sqrt(var_mid), a),
    }
    if run3:
        vars_["MS-GARCH(3)"] = pd.Series(mixture_var(h3, p3, a), index=test.index).shift(1).dropna()
    for name, v in vars_.items():
        ret = r_test.reindex(v.index); nv = int((ret < v).sum()); T = len(v)
        lk, pk = kupiec_test(nv / T, a, T); lc, pc = christoffersen_test(ret, v)
        say(f"{name:12s} {cl}: {nv:3d}/{T} = {nv/T*100:5.2f}%  Kupiec p={pk:.3f}  Christoffersen p={pc:.3f}")
open(os.path.join(ROOT, "results", "strict_oos_check.txt"), "w").write(log.getvalue())
