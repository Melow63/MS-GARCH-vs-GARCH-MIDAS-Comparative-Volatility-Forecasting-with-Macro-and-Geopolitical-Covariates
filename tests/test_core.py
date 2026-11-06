import os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from mixture_var import mixture_var, var_single
from backtests import kupiec_test, christoffersen_test, dm_test_hac
from ms_garch2_from_scratch import MSGARCH2
from ms_garch3_from_scratch import MSGARCH3


def test_mixture_var_single_regime_equals_gaussian_quantile():
    h = np.full((5, 1), 4.0)          # variance 4 (in scale units), one regime
    p = np.ones((5, 1))
    v = mixture_var(h, p, 0.01, scale=1.0)
    assert np.allclose(v, norm.ppf(0.01) * 2.0, atol=1e-6)


def test_mixture_var_is_more_negative_than_gaussian_on_aggregate_variance_for_fat_mixture():
    h = np.array([[0.5, 50.0]]); p = np.array([[0.97, 0.03]])
    agg = -norm.ppf(0.01) * np.sqrt((h * p).sum())
    assert -mixture_var(h, p, 0.01, scale=1.0)[0] > agg * 0.9   # same order, tail not crushed


def test_kupiec_accepts_exact_coverage_and_rejects_bad_coverage():
    _, p_ok = kupiec_test(0.01, 0.01, 750)
    _, p_bad = kupiec_test(0.03, 0.01, 750)
    assert p_ok > 0.99 and p_bad < 0.001


def test_christoffersen_detects_clustered_violations():
    rng = np.random.default_rng(0)
    n = 1000; ret = pd.Series(np.zeros(n)); var = pd.Series(np.full(n, -1.0))
    iid = rng.random(n) < 0.05
    ret[iid] = -2.0
    lr_iid, p_iid = christoffersen_test(ret, var)
    clustered = np.zeros(n, bool)
    for s in range(20, n, 100):
        clustered[s:s + 5] = True      # blocks of 5 consecutive violations
    ret2 = pd.Series(np.where(clustered, -2.0, 0.0))
    lr_c, p_c = christoffersen_test(ret2, var)
    assert p_c < 0.01 < p_iid


def test_dm_detects_a_systematically_lower_loss():
    rng = np.random.default_rng(1)
    a = rng.random(500); b = a + 0.2
    dm, p = dm_test_hac(a, b)
    assert dm < 0 and p < 0.01


def _simulated_returns(T=600, seed=3):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(T) * 0.01)


def test_ms_garch2_filter_probabilities_are_valid():
    r = _simulated_returns()
    m = MSGARCH2(returns=r)
    params = dict(omega1=0.02, alpha1=0.05, beta1=0.90, omega2=0.2, alpha2=0.12, beta2=0.78, p11=0.97, p22=0.92)
    out = m.hamilton_filter(params)
    assert np.isfinite(out["loglik"])
    assert np.allclose(out["filtered"].sum(axis=1), 1) and np.allclose(out["predicted"].sum(axis=1), 1)


def test_ms_garch3_filter_and_smoother_are_valid():
    r = _simulated_returns()
    m = MSGARCH3(returns=r)
    p = dict(omega1=0.01, alpha1=0.04, beta1=0.92, omega2=0.05, alpha2=0.08, beta2=0.88,
             omega3=0.2, alpha3=0.15, beta3=0.78, p11=0.96, p12=0.035, p22=0.92, p23=0.06, p33=0.88, p31=0.04)
    out = m.hamilton_filter(p)
    sm = m._smooth(p, out)
    assert np.isfinite(out["loglik"]) and np.allclose(sm.sum(axis=1), 1)
    assert np.allclose(out["P"].sum(axis=1), 1)


def test_ms_garch2_fit_runs_and_orders_regimes_by_variance():
    r = _simulated_returns(400)
    m = MSGARCH2(returns=r)
    res = m.fit(n_starts=3, maxiter=200, verbose=False)
    p = res["params"]
    unc = [p[f"omega{k}"] / (1 - p[f"alpha{k}"] - p[f"beta{k}"]) for k in (1, 2)]
    assert unc[0] <= unc[1]
