# MS-GARCH 3 regimes from scratch (calm / moderate / turbulent)
# same logic as 2-regime, extended to K=3
# 15 parameters: 9 GARCH (3x omega/alpha/beta) + 6 free transition probabilities

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


EPS = 1e-12
K = 3


class MSGARCH3:

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

        if self.T < 50:
            raise ValueError("Not enough observations for 3 regimes.")

        self.fit_result_ = None

    def _softplus(self, x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def _unpack(self, theta):
        # theta R^15 -> natural parameters
        # blocks [0:3], [3:6], [6:9] for the 3 GARCH regimes
        # blocks [9:11], [11:13], [13:15] for rows of P (softmax)
        theta = np.asarray(theta, dtype=float)
        params = {}

        for k in range(K):
            j = 3 * k
            omega = self._softplus(theta[j]) + 1e-10
            a_raw = expit(theta[j+1])
            b_raw = expit(theta[j+2])
            alpha = 0.999 * a_raw
            beta = 0.999 * (1 - a_raw) * b_raw
            params[f"omega{k+1}"] = float(omega)
            params[f"alpha{k+1}"] = float(alpha)
            params[f"beta{k+1}"] = float(beta)

        # softmax for each row of P: (a, b, 0) -> 3 probabilities
        def row_softmax(a, b):
            z = np.array([a, b, 0.0])
            z -= z.max()
            e = np.exp(z)
            return e / e.sum()

        r1 = row_softmax(theta[9], theta[10])
        r2 = row_softmax(theta[11], theta[12])
        r3 = row_softmax(theta[13], theta[14])

        params["p11"], params["p12"], params["p13"] = map(float, r1)
        params["p22"], params["p23"], params["p21"] = map(float, r2)
        params["p33"], params["p31"], params["p32"] = map(float, r3)

        return params

    def _theta0_from_natural(self, p):
        # approximation to build theta0 from natural parameters
        # softplus inverse approximated by log(y), sufficient for initialization
        def logit(x):
            x = float(np.clip(x, 1e-6, 1 - 1e-6))
            return np.log(x / (1 - x))

        theta = []
        for k in range(1, K+1):
            a = float(np.clip(p[f"alpha{k}"] / 0.999, 1e-6, 1 - 1e-6))
            b = float(np.clip(p[f"beta{k}"] / (0.999 * (1 - a)), 1e-6, 1 - 1e-6))
            theta += [np.log(max(p[f"omega{k}"], 1e-8)), logit(a), logit(b)]

        # inverse of row_softmax([a,b,0]): a = log(p1/p3), b = log(p2/p3)
        # row 1 of P
        p11 = max(p["p11"], 1e-8); p12 = max(p["p12"], 1e-8)
        p13 = max(1 - p11 - p12, 1e-8)
        theta += [np.log(p11/p13), np.log(p12/p13)]

        p22 = max(p["p22"], 1e-8); p23 = max(p["p23"], 1e-8)
        p21 = max(1 - p22 - p23, 1e-8)
        theta += [np.log(p22/p21), np.log(p23/p21)]
        # row 3
        p33 = max(p["p33"], 1e-8); p31 = max(p["p31"], 1e-8)
        p32 = max(1 - p33 - p31, 1e-8)
        theta += [np.log(p33/p32), np.log(p31/p32)]

        return np.array(theta, dtype=float)

    def _build_P(self, params):
        P = np.array([
            [params["p11"], params["p12"], 1 - params["p11"] - params["p12"]],
            [1 - params["p22"] - params["p23"], params["p22"], params["p23"]],
            [params["p31"], 1 - params["p33"] - params["p31"], params["p33"]],
        ], dtype=float)
        P = np.clip(P, EPS, 1.0)
        return P / P.sum(axis=1, keepdims=True)

    def _stationary(self, P):
        # solve pi @ P = pi with sum = 1
        A = P.T - np.eye(K)
        A[-1] = 1.0
        b = np.zeros(K); b[-1] = 1.0
        pi = np.linalg.solve(A, b)
        pi = np.clip(pi, EPS, 1.0)
        return pi / pi.sum()

    def _compute_h(self, params):
        om = np.array([params[f"omega{k}"] for k in range(1, K+1)])
        al = np.array([params[f"alpha{k}"] for k in range(1, K+1)])
        be = np.array([params[f"beta{k}"] for k in range(1, K+1)])

        h = np.empty((self.T, K))
        unc = om / np.maximum(1 - al - be, 1e-5)
        h[0] = np.where(np.isfinite(unc) & (unc > 0), unc, np.var(self.r))
        h[0] = np.clip(h[0], 1e-10, 1e8)

        for t in range(1, self.T):
            h[t] = om + al * self.r[t-1]**2 + be * h[t-1]
            h[t] = np.clip(h[t], 1e-10, 1e8)

        logdens = -0.5 * (np.log(2 * np.pi) + np.log(h) + self.r[:, None]**2 / h)
        return h, logdens

    def hamilton_filter(self, params):
        P = self._build_P(params)
        pi0 = self._stationary(P)
        h, logdens = self._compute_h(params)

        predicted = np.empty((self.T, K))
        filtered = np.empty((self.T, K))
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
            "P": P,
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
        # Kim smoother, same logic as 2-regime
        P = self._build_P(params)
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

    def _reorder(self, params):
        # reorder regimes by increasing unconditional variance
        unc = []
        for k in range(1, K+1):
            unc.append(params[f"omega{k}"] / max(1 - params[f"alpha{k}"] - params[f"beta{k}"], 1e-5))

        order = np.argsort(unc)  # e.g. [2, 0, 1] if regime 3 was the calmest
        if list(order) == [0, 1, 2]:
            return params

        P_old = self._build_P(params)

        # new transition matrix after permuting rows and columns
        P_new = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                P_new[i, j] = P_old[order[i], order[j]]

        new_p = {}
        for new_k, old_idx in enumerate(order, start=1):
            old_k = int(old_idx) + 1
            new_p[f"omega{new_k}"] = params[f"omega{old_k}"]
            new_p[f"alpha{new_k}"] = params[f"alpha{old_k}"]
            new_p[f"beta{new_k}"] = params[f"beta{old_k}"]

        for i in range(K):
            for j in range(K):
                new_p[f"p{i+1}{j+1}"] = float(P_new[i, j])

        return new_p

    def _build_starts(self, n_starts, rng):
        v = max(float(np.var(self.r)), 1e-6)

        # fixed starting points, manually tuned
        anchors = [
            dict(omega1=0.01*v, alpha1=0.04, beta1=0.92,
                omega2=0.05*v, alpha2=0.08, beta2=0.88,
                omega3=0.20*v, alpha3=0.15, beta3=0.78,
                p11=0.96, p12=0.035, p22=0.92, p23=0.06, p33=0.88, p31=0.04),
            dict(omega1=0.02*v, alpha1=0.05, beta1=0.90,
                omega2=0.10*v, alpha2=0.10, beta2=0.84,
                omega3=0.50*v, alpha3=0.20, beta3=0.70,
                p11=0.97, p12=0.025, p22=0.90, p23=0.08, p33=0.85, p31=0.07),
            dict(omega1=0.005*v, alpha1=0.03, beta1=0.95,
                omega2=0.08*v, alpha2=0.12, beta2=0.80,
                omega3=0.80*v, alpha3=0.25, beta3=0.60,
                p11=0.98, p12=0.015, p22=0.94, p23=0.04, p33=0.80, p31=0.10),
        ]
        starts = [self._theta0_from_natural(a) for a in anchors]

        for _ in range(max(0, n_starts - len(starts))):
            a1 = rng.uniform(0.02, 0.12); b1 = rng.uniform(0.78, 0.97 - a1)
            a2 = rng.uniform(0.04, 0.20); b2 = rng.uniform(0.65, 0.97 - a2)
            a3 = rng.uniform(0.08, 0.35); b3 = rng.uniform(0.40, 0.97 - a3)
            p11 = rng.uniform(0.88, 0.995)
            p12 = rng.uniform(0.001, min(0.10, 0.999 - p11))
            p22 = rng.uniform(0.75, 0.97)
            p23 = rng.uniform(0.005, min(0.20, 0.999 - p22))
            p33 = rng.uniform(0.60, 0.96)
            p31 = rng.uniform(0.005, min(0.25, 0.999 - p33))
            p = dict(omega1=v*rng.uniform(0.001, 0.08), alpha1=a1, beta1=b1,
                     omega2=v*rng.uniform(0.02, 0.30), alpha2=a2, beta2=b2,
                     omega3=v*rng.uniform(0.10, 1.20), alpha3=a3, beta3=b3,
                     p11=p11, p12=p12, p22=p22, p23=p23, p33=p33, p31=p31)
            starts.append(self._theta0_from_natural(p))
        return starts

    def fit(self, n_starts=30, maxiter=2500, random_state=123, verbose=True):
        rng = np.random.default_rng(random_state)
        best = None

        # less stable than 2-regime; sometimes need to wait for the 10th or 15th start
        for i, theta0 in enumerate(self._build_starts(n_starts, rng), 1):
            res = minimize(
                self._neg_loglik, theta0,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-5, "maxls": 80},
            )
            if verbose:
                print(f"start {i:02d}/{n_starts} | nll={res.fun:.4f} | success={res.success}")
            if best is None or res.fun < best.fun:
                best = res

        params = self._unpack(best.x)
        params = self._reorder(params)  # regime 1=calm, 2=moderate, 3=turbulent

        filt = self.hamilton_filter(params)
        smoothed = self._smooth(params, filt)
        loglik = filt["loglik"]
        k_par = 15
        bic = np.log(self.T) * k_par - 2 * loglik
        aic = 2 * k_par - 2 * loglik

        ix = self.dates
        cols_p = ["Regime_1_Calm", "Regime_2_Moderate", "Regime_3_Turbulent"]
        cols_h = ["h_Regime_1", "h_Regime_2", "h_Regime_3"]

        P_df = pd.DataFrame(
            filt["P"],
            index=["From_1", "From_2", "From_3"],
            columns=["To_1", "To_2", "To_3"],
        )

        self.fit_result_ = {
            "params": params,
            "transition_matrix": P_df,
            "loglik": loglik,
            "aic": aic,
            "bic": bic,
            "success": bool(best.success),
            "nobs": self.T,
            "filtered_probs": pd.DataFrame(filt["filtered"], index=ix, columns=cols_p),
            "predicted_probs": pd.DataFrame(filt["predicted"], index=ix, columns=cols_p),
            "smoothed_probs": pd.DataFrame(smoothed, index=ix, columns=cols_p),
            "variances": pd.DataFrame(filt["h"], index=ix, columns=cols_h),
            "optimizer": best,
        }
        return self.fit_result_

    def summary(self):
        if self.fit_result_ is None:
            raise RuntimeError("Call .fit() first.")
        p = self.fit_result_["params"]
        print(f"MS-GARCH(3)  nobs={self.fit_result_['nobs']}  loglik={self.fit_result_['loglik']:.2f}  "
              f"AIC={self.fit_result_['aic']:.2f}  BIC={self.fit_result_['bic']:.2f}")
        print(f"  success={self.fit_result_['success']}")
        for k in range(1, K+1):
            unc = p[f"omega{k}"] / max(1 - p[f"alpha{k}"] - p[f"beta{k}"], 1e-5)
            print(f"  Regime {k}: omega={p[f'omega{k}']:.5f}  alpha={p[f'alpha{k}']:.4f}  "
                  f"beta={p[f'beta{k}']:.4f}  unc_var={unc:.5f}")
        print("Transition matrix:")
        print(self.fit_result_["transition_matrix"].round(4))

    def plot_regime_probabilities(self, start="2010-01-01", end="2025-12-31", which="smoothed"):
        if self.fit_result_ is None:
            raise RuntimeError("Call .fit() first.")

        df = self.fit_result_[which + "_probs"].copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.loc[start:end]

        ax = df.plot(figsize=(14, 5), lw=1.2)
        ax.set_title(f"{which.capitalize()} probabilities -- MS-GARCH 3 regimes")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return ax
