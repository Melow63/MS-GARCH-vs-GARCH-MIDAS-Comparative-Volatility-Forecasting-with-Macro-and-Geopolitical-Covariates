# MS-GARCH-vs-GARCH-MIDAS-Comparative-Volatility-Forecasting-with-Macro-and-Geopolitical-Covariates

## Context : 
  Volatility modelling and forecasting are key steps of good market risk management. In an unstable geopolitical world, it becomes crucial for individuals and institutional investors to anticipate macroeconomic and political changes. However, traditional volatility models rely on market stability, imposing restrictive limits on regime switching and  the impact of geopolitical and macroeconomic variables.

## Problem : 
  Two recent approaches respond to these limits. MS-GARCH takes regime switching into account, allowing volatility dynamics to vary between different states. GARCH-MIDAS models integrate low-frequency variables to model the long-term volatility component. Although these  models answer to two different dimensions of the problem (volatility forecasting), it is difficult to determine which one is the most efficient regarding market state and forecasting horizon.

## Objective : 
  Compare the performance of both of these models, MS-GARCH and GARCH-MIDAS, for forecasting and modelling volatility. Evaluate to what extent integrating the specific features of each model improves forecasting quality.


## Methodology : 
  ### Models to estimate : 
    GARCH (1, 1), simple as a benchmark
    GARCH-MIDAS with GPR Index, EPU Index, NFCI, and a combination of all of them.
    MS-GARCH 2 regimes & 3 regimes
    Systematic comparison between families of models

  ### Data :
    S&P 500 daily returns (Yahoo Finance) / 2010-2025 period
    GPR index — Geopolitical Risk Index (Caldara & Iacoviello 2022), monthly
    EPU index — Economic Policy Uncertainty (Baker et al.), monthly
    NFCI — National Financial Conditions Index (Chicago Fed), weekly

  ### Model evaluation : 
    In-sample: BIC, Maximum Likelihood, variance ratio
    Out-of-sample: Diebold-Mariano test
    Risk application: Dynamic VaR at 95% & 99%
    Backtesting: Kupiec & Christoffersen tests

  ### Programming languages:
    Python
    R

  ### Deliverable:
    Reports (Word documents with graphs, tables...)
    Github (repo + code + results...)


## Repository Structure
    data/        → raw and processed datasets
    notebooks/   → exploratory analysis and estimation
    models/      → model implementations
    results/     → tables, figures, outputs
    papers/      → key references

  
