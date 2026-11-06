# MS-GARCH 2 regimes from scratch -- Hamilton filter + Kim smoother
# unconstrained reparametrization for L-BFGS-B

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


EPS = 1e-12


class MSGARCH2:

    def __init__(self, returns, dates=None, scale=100.0, demean=True):
        r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()

        if dates is not None:
            d = pd.to_datetime(pd.Series(dates).loc[r.index])
            self.dates = pd.DatetimeIndex(d)
        else:
            self.dates = pd.RangeIndex(len(r))

        self.scale = float(scale)
        self.demean = bool(demean)

        x = r.to_numpy() * self.scale
        self.mean_ = float(np.mean(x)) if demean else 0.0
        self.r = x - self.mean_
        self.T = len(self.r)

        if self.T < 20:
            raise ValueError("Not enough observations.")

        self.fit_result_ = None

    def _softplus(self, x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def _unpack(self, theta):
        # theta in R^8 -> natural parameters
        # softplus ensures omega > 0, logistic + constraint ensures alpha+beta < 0.999
        omega1 = self._softplus(theta[0]) + 1e-10
        a1 = expit(theta[1])
        b1 = expit(theta[2])
        alpha1 = 0.999 * a1
        beta1 = 0.999 * (1.0 - a1) * b1

        omega2 = self._softplus(theta[3]) + 1e-10
        a2 = expit(theta[4])
        b2 = expit(theta[5])
        alpha2 = 0.999 * a2
        beta2 = 0.999 * (1.0 - a2) * b2

        p11 = 0.001 + 0.998 * expit(theta[6])
        p22 = 0.001 + 0.998 * expit(theta[7])

        return dict(omega1=omega1, alpha1=alpha1, beta1=beta1,
        omega2=omega2, alpha2=alpha2, beta2=beta2,
        p11=p11, p22=p22)

    def _theta0_from_natural(self, p):
        # rough approximation -- just for initialization, does not need to be exact
        def logit(x):
            x = float(np.clip(x, 1e-6, 1 - 1e-6))
            return np.log(x / (1 - x))

        # exact softplus inverse is log(exp(y)-1), we use log(y) as approximation
        def inv_sp_approx(y):
            y = max(float(y), 1e-8)
            return np.log(y)

        a1 = float(np.clip(p["alpha1"] / 0.999, 1e-6, 1 - 1e-6))
        b1 = float(np.clip(p["beta1"] / (0.999 * (1 - a1)), 1e-6, 1 - 1e-6))
        a2 = float(np.clip(p["alpha2"] / 0.999, 1e-6, 1 - 1e-6))
        b2 = float(np.clip(p["beta2"] / (0.999 * (1 - a2)), 1e-6, 1 - 1e-6))
        p11r = float(np.clip((p["p11"] - 0.001) / 0.998, 1e-6, 1 - 1e-6))
        p22r = float(np.clip((p["p22"] - 0.001) / 0.998, 1e-6, 1 - 1e-6))

        vals = [inv_sp_approx(p["omega1"]), logit(a1), logit(b1),
                inv_sp_approx(p["omega2"]), logit(a2), logit(b2),
                logit(p11r), logit(p22r)]
        return np.array(vals, dtype=float)

    def _compute_h(self, params):
        om = np.array([params["omega1"], params["omega2"]])
        al = np.array([params["alpha1"], params["alpha2"]])
        be = np.array([params["beta1"], params["beta2"]])

        h = np.empty((self.T, 2))
        # initialize at unconditional variance
        unc = om / np.maximum(1 - al - be, 1e-5)
        h[0] = np.where(np.isfinite(unc) & (unc > 0), unc, np.var(self.r))
        h[0] = np.clip(h[0], 1e-10, 1e8)

        for t in range(1, self.T):
            h[t] = om + al * self.r[t-1]**2 + be * h[t-1]
            h[t] = np.clip(h[t], 1e-10, 1e8)

        logdens = -0.5 * (np.log(2 * np.pi) + np.log(h) + self.r[:, None]**2 / h)
        return h, logdens

    def hamilton_filter(self, params):
        p11, p22 = params["p11"], params["p22"]
        P = np.array([[p11, 1 - p11], [1 - p22, p22]])

        # stationary distribution as initial probabilities
        denom = max((1 - p11) + (1 - p22), EPS)
        pi0 = np.array([(1 - p22) / denom, (1 - p11) / denom])

        h, logdens = self._compute_h(params)

        predicted = np.empty((self.T, 2))
        filtered = np.empty((self.T, 2))
        log_marg = np.empty(self.T)

        for t in range(self.T):
            pred_t = pi0 if t == 0 else filtered[t-1] @ P
            pred_t = np.clip(pred_t, EPS, 1.0)
            pred_t /= pred_t.sum()
            predicted[t] = pred_t

            log_joint = np.log(pred_t) + logdens[t]
            lm = logsumexp(log_joint)
            log_marg[t] = lm
            filtered[t] = np.exp(log_joint - lm)

        return {
            "predicted": predicted,
            "filtered": filtered,
            "log_marg": log_marg,
            "loglik": float(log_marg.sum()),
            "h": h,
        }

    def _neg_loglik(self, theta):
        try:
            params = self._unpack(theta)
            out = self.hamilton_filter(params)
            nll = -out["loglik"]
            return float(nll) if np.isfinite(nll) else 1e50
        except Exception:
            return 1e50

    def _smooth(self, params, filt_out):
        # Kim (1994) smoother -- backward pass
        p11, p22 = params["p11"], params["p22"]
        P = np.array([[p11, 1 - p11], [1 - p22, p22]])

        filtered = filt_out["filtered"]
        predicted = filt_out["predicted"]

        smoothed = np.empty_like(filtered)
        smoothed[-1] = filtered[-1]

        for t in range(self.T - 2, -1, -1):
            w = smoothed[t+1] / np.maximum(predicted[t+1], EPS)
            smoothed[t] = filtered[t] * (P @ w)
            smoothed[t] = np.clip(smoothed[t], EPS, 1.0)
            smoothed[t] /= smoothed[t].sum()

        return smoothed

    def _build_starts(self, n_starts, rng):
        v = max(float(np.var(self.r)), 1e-6)

        # 3 fixed starting points, manually tuned on S&P 500
        # (calm: low omega, beta close to 1 | turbulent: high omega, lower beta)
        anchors = [
            dict(omega1=0.03*v, alpha1=0.05, beta1=0.90, omega2=0.20*v, alpha2=0.12, beta2=0.78, p11=0.97, p22=0.92),
            dict(omega1=0.10*v, alpha1=0.08, beta1=0.86, omega2=0.50*v, alpha2=0.18, beta2=0.70, p11=0.95, p22=0.85),
            dict(omega1=0.50*v, alpha1=0.15, beta1=0.70, omega2=0.03*v, alpha2=0.05, beta2=0.90, p11=0.90, p22=0.97),
        ]
        starts = [self._theta0_from_natural(a) for a in anchors]

        # the rest are random
        for _ in range(max(0, n_starts - len(starts))):
            a1 = rng.uniform(0.03, 0.25)
            b1 = rng.uniform(0.60, 0.96 - a1)
            a2 = rng.uniform(0.03, 0.30)
            b2 = rng.uniform(0.50, 0.96 - a2)
            p = dict(
                omega1=v * rng.uniform(0.005, 0.40), alpha1=a1, beta1=b1,
                omega2=v * rng.uniform(0.005, 0.80), alpha2=a2, beta2=b2,
                p11=rng.uniform(0.85, 0.995), p22=rng.uniform(0.75, 0.995),
            )
            starts.append(self._theta0_from_natural(p))
        return starts

    def fit(self, n_starts=25, maxiter=2000, random_state=123, verbose=True):
        rng = np.random.default_rng(random_state)
        best = None

        for i, theta0 in enumerate(self._build_starts(n_starts, rng), 1):
            res = minimize(
                self._neg_loglik, theta0,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-5, "maxls": 50},
            )
            if verbose:
                print(f"start {i:02d}/{n_starts} | nll={res.fun:.4f} | success={res.success}")
            if best is None or res.fun < best.fun:
                best = res

        # keep the best result across all n_starts attempts
        params = self._unpack(best.x)

        # convention: regime 1 = low vol, regime 2 = high vol
        # swap labels if optimizer reversed them
        unc1 = params["omega1"] / max(1 - params["alpha1"] - params["beta1"], 1e-5)
        unc2 = params["omega2"] / max(1 - params["alpha2"] - params["beta2"], 1e-5)
        if unc1 > unc2:
            params = {
                "omega1": params["omega2"], "alpha1": params["alpha2"], "beta1": params["beta2"],
                "omega2": params["omega1"], "alpha2": params["alpha1"], "beta2": params["beta1"],
                "p11": params["p22"], "p22": params["p11"],
            }

        filt = self.hamilton_filter(params)
        smoothed = self._smooth(params, filt)
        loglik = filt["loglik"]
        k = 8
        bic = np.log(self.T) * k - 2 * loglik
        aic = 2 * k - 2 * loglik

        ix = self.dates
        self.fit_result_ = {
            "params": params,
            "loglik": loglik,
            "aic": aic,
            "bic": bic,
            "success": bool(best.success),
            "nobs": self.T,
            "filtered_probs": pd.DataFrame(filt["filtered"], index=ix, columns=["Regime_1", "Regime_2"]),
            "predicted_probs": pd.DataFrame(filt["predicted"], index=ix, columns=["Regime_1", "Regime_2"]),
            "smoothed_probs": pd.DataFrame(smoothed, index=ix, columns=["Regime_1", "Regime_2"]),
            "variances": pd.DataFrame(filt["h"], index=ix, columns=["h_Regime_1", "h_Regime_2"]),
            "optimizer": best,
        }
        return self.fit_result_

    def summary(self):
        if self.fit_result_ is None:
            raise RuntimeError("Call .fit() first.")
        p = self.fit_result_["params"]
        print(f"MS-GARCH(2)  nobs={self.fit_result_['nobs']}  loglik={self.fit_result_['loglik']:.2f}  "
              f"AIC={self.fit_result_['aic']:.2f}  BIC={self.fit_result_['bic']:.2f}")
        print(f"  success={self.fit_result_['success']}")
        for k in [1, 2]:
            unc = p[f"omega{k}"] / max(1 - p[f"alpha{k}"] - p[f"beta{k}"], 1e-5)
            print(f"  Regime {k}: omega={p[f'omega{k}']:.5f}  alpha={p[f'alpha{k}']:.4f}  "
                  f"beta={p[f'beta{k}']:.4f}  unc_var={unc:.5f}")
        print(f"  p11={p['p11']:.4f}  p22={p['p22']:.4f}")

    def plot_regime_probabilities(self, start="2010-01-01", end="2025-12-31", which="smoothed"):
        if self.fit_result_ is None:
            raise RuntimeError("Call .fit() first.")

        if which == "smoothed":
            probs = self.fit_result_["smoothed_probs"]
            title = "Smoothed probabilities -- Regime 2 (high vol)"
        else:
            probs = self.fit_result_["filtered_probs"]
            title = "Filtered probabilities -- Regime 2"

        df = probs.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.loc[start:end]

        ax = df["Regime_2"].plot(figsize=(14, 5), lw=1.2)
        ax.set_title(title)
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return ax
