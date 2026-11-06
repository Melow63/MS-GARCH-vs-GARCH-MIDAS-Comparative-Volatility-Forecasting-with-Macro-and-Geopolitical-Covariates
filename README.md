# MS-GARCH vs GARCH-MIDAS: S&P 500 volatility forecasting

HEC Montreal supervised research project (M.Sc. Applied Financial Economics, 2026).
Four volatility models are compared on S&P 500 daily returns, January 2010 to December 2025:

| Model | Implementation |
|---|---|
| GARCH(1,1), benchmark | `arch` package |
| MS-GARCH, 2 regimes | written from scratch in Python (`src/ms_garch2_from_scratch.py`): Hamilton filter, Kim smoother, L-BFGS-B with softplus/logistic reparametrisation, multi-start |
| MS-GARCH, 3 regimes | same, extended to K = 3 (`src/ms_garch3_from_scratch.py`) |
| GARCH-MIDAS | R package `mfGARCH` (Conrad and Kleen), with NFCI (weekly), EPU and GPR (monthly) as low-frequency variables |

The MS-GARCH VaR is computed from the predictive distribution, a Gaussian mixture, by inverting its CDF numerically
(`src/mixture_var.py`), not by applying a fixed quantile to the aggregate variance.

## Main results (test period 2023-2025, 751 trading days)

Reproduced by `notebooks/05_forecasting.ipynb` (outputs are saved in the notebook).

| One-day 99% VaR | Violations | Rate | Kupiec p | Christoffersen p |
|---|---|---|---|---|
| GARCH(1,1) | 12 / 750 | 1.60% | 0.129 | 0.180 |
| MS-GARCH(2) | 8 / 750 | 1.07% | 0.856 | 0.069 |
| MS-GARCH(3) | 10 / 750 | 1.33% | 0.383 | 0.117 |
| GARCH-MIDAS | 16 / 751 | 2.13% | 0.007 | 0.346 |

Diebold-Mariano tests on squared forecast errors do not separate the models (all p-values above 0.34), so the RMSE
rankings are suggestive only. The best in-sample GARCH-MIDAS specification by BIC is NFCI + EPU (9,514 against 10,400 for GARCH(1,1)).
A two-step MS-GARCH / GARCH-MIDAS combination (`src/combo_ms_midas_lite.py`) was tried and did not work: it is kept as a documented negative result.

## What the comparison does and does not control for

Read this before interpreting the table above.

- **GARCH(1,1) and MS-GARCH parameters are estimated once on the full sample (2010-2025)** in notebooks 03 to 05, then evaluated
  on 2023-2025. Only the evaluation window is out of sample, not the estimation. GARCH-MIDAS, by contrast, is re-estimated every
  month on data up to the previous month (`R/mfGARCH_forecasting.R`), so it is the only genuinely out-of-sample model in the table.
  This favours the other three.
- The RMSE comparison for MS-GARCH uses **smoothed** regime probabilities, which use the whole sample. The VaR backtest uses
  one-step-ahead **predicted** probabilities.
- `scripts/strict_oos_check.py` repeats the comparison with every parameter estimated on 2010-2022 only. Its output is in
  `results/strict_oos_check.txt`; the summary is in the next section.
- Realised variance is proxied by the squared daily return, which is unbiased but noisy (Patton, 2011).
- A validation of the from-scratch MS-GARCH against the R `MSGARCH` package is not part of this repository.

## Strict out-of-sample check

Parameters estimated on 2010-2022 only, test period filtered forward with those fixed parameters and one-step-ahead predicted
probabilities (`results/strict_oos_check.txt`; MS-GARCH(3) is left out because its 15-parameter fit takes hours, use `--starts3`).

| | Table in notebook 05 (full-sample parameters) | Strict check (2010-2022 parameters) |
|---|---|---|
| MS-GARCH(2) 99% VaR violations | 8 / 750, Kupiec p = 0.856 | 8 / 750, Kupiec p = 0.856 |
| MS-GARCH(2) 95% VaR violations | 43 / 750, Kupiec p = 0.367 | 43 / 750, Kupiec p = 0.367 |
| GARCH(1,1) 99% / 95% violations | 12 / 37 | 12 / 39 |
| RMSE, MS-GARCH(2) against GARCH(1,1) | -6.6% with smoothed probabilities | -0.7% with predicted probabilities (DM p = 0.69) |

The VaR conclusion does not depend on estimating the parameters on the full sample. The RMSE gain of the regime-switching models
comes from the smoothed probabilities; with information available in real time it is under 1% and not significant.


## Repository layout

```
data/raw/            source series (see Data below)
data/processed/      merged daily dataset (merged_data.csv) and the file used by R
src/                 MS-GARCH (2 and 3 regimes), mixture VaR, backtests (Kupiec, Christoffersen, Diebold-Mariano, QLIKE)
notebooks/           01 data collection, 02 exploration, 03 GARCH(1,1), 04 MS-GARCH, 05 forecasting and backtests
R/                   GARCH-MIDAS estimation (mfGARCH.R) and monthly out-of-sample forecasts (mfGARCH_forecasting.R)
results/             tables, figures, regime probabilities, GARCH-MIDAS forecasts
scripts/             strict out-of-sample check
tests/               unit tests (mixture VaR, Kupiec, Christoffersen, Diebold-Mariano, Hamilton filter and smoother)
```

## Reproduce

```bash
pip install -r requirements.txt
pytest                                   # about one minute
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 02_exploratory_analysis.ipynb 03_GARCH_1_1.ipynb 05_forecasting.ipynb
cd .. && python scripts/strict_oos_check.py
```

Notebook 05 takes several minutes (it refits the 2-regime model on the training sample). Notebook 04 refits both MS-GARCH models
on the full sample (50 and 30 starting points) and takes much longer; its saved outputs and the CSV files in `results/` are
what notebook 05 reads. Notebook 01 downloads data again and needs a free FRED key in the environment variable `FRED_API_KEY`;
re-downloading will change the numbers slightly because Yahoo Finance revises history. The processed dataset is included so that
nothing needs to be downloaded to run the other notebooks.

GARCH-MIDAS (R):

```r
install.packages("mfGARCH")
```
```bash
Rscript R/mfGARCH.R               # six specifications, BIC table -> results/garch_midas_results.csv
Rscript R/mfGARCH_forecasting.R   # monthly re-estimation, forecasts -> results/midas_oos_forecast.csv
```

## Data

| Series | Source |
|---|---|
| S&P 500 daily close, VIX, Treasury yields | Yahoo Finance |
| NFCI (weekly) | Federal Reserve Bank of Chicago |
| Economic Policy Uncertainty index (monthly) | Baker, Bloom and Davis, policyuncertainty.com |
| Geopolitical Risk index (monthly) | Caldara and Iacoviello, matteoiacoviello.com/gpr.htm |
| EFFR, CPI, unemployment | FRED |

Low-frequency series are forward-filled to daily frequency for the merged dataset. Train: 2010-01-05 to 2022-12-30 (3,271 days).
Test: 2023-01-03 to 2025-12-30 (751 days).

## References

Bollerslev (1986); Hamilton (1989); Kim (1994); Haas, Mittnik and Paolella (2004); Engle, Ghysels and Sohn (2013);
Conrad and Kleen (2020); Ardia, Bluteau, Boudt and Trottier (2019); Kupiec (1995); Christoffersen (1998); Diebold and Mariano (1995);
Patton (2011); Segnon, Gupta and Wilfling (2024).
