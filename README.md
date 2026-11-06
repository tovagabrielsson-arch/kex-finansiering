# Munters interest-rate fixing / ICR research prototype

**All financial inputs are hypothetical. This is not an assessment of Munters' actual risk.**

## Run

```bash
python -m pip install -r requirements.txt
python munters_risk_model.py
```

Creates `output/strategy_results.csv`, `output/static_stress.csv` and three PNG charts.

## Core logic

- 10,000 common Monte Carlo scenarios over 3 years, in quarterly steps.
- A real-world (physical-measure) **Vasicek** reference-rate process, using exact OU transition distributions.
- A correlated, mean-reverting **log EBITDA** process (annualized EBITDA stays positive).
- Floating borrowing at reference rate (floored at zero) plus a constant spread.
- Fixed exposure: fixed swap rate plus the same spread. For this version swaps are already in place at time 0 and run at least through the horizon.
- Five fixed-rate shares, from 0% to 100%; initial weighted fixing years calculated from 3-year fixed and 3-month floating exposures.
- Rolling four-quarter EBITDA / rolling four-quarter **gross interest expense**. The pre-simulation 3 quarters of history are set to constant hypothetical starting values.
- A "breach" is **at least one quarterly observation** of the rolling ICR below 3.0. It is a research threshold, **not** a confirmed Munters covenant.
- A separate *static* stress scenario with +300 basis points and -30% annual EBITDA.

## Customize

Edit the `Assumptions` dataclass at the start of `munters_risk_model.py`. All rates are decimals: 0.03 means 3%, EBITDA and debt are SEK, and `rate_sigma` is in **absolute rate units per square root of year**.

## Critical limitations

1. **Not calibrated:** `rate_kappa`, `rate_long_run`, `rate_sigma`, EBITDA parameters and correlation are illustrative. Fit to real-world data and evaluate out-of-sample before drawing conclusions.
2. **Not a contract-level debt model:** Debt, credit spread and swap rate are held constant; no maturities, refinancing, loan amortization, FX, liquidity, collateral or swap basis risk. Three-year swaps are assumed to cover the entire horizon. Their initial weighted average fixing period is **not** their changing future duration.
3. **Correlation assumption:** The parameter is the correlation of standardized *quarterly innovations*, not necessarily the historical correlation of rate levels and EBITDA, nor exactly that of underlying continuous-time shocks.
4. **Simplified benchmark:** Vasicek simulates a generic short reference rate, not the full STIBOR/EURIBOR/SOFR term structures. Reference rates are floored at zero here by a modeling choice, not by evidence of contractual floors.
5. **Accounting and timing:** End-of-quarter EBITDA annualized states are converted into quarter flows, and quarter interest is set using the **start-of-quarter** floating reference rate; 3 prior quarters are synthetic, not historical data. Actual covenant definitions may use adjusted EBITDA, cash interest, net interest, different dates or periods.
6. **Risk interpretation:** 95% confidence intervals reflect Monte Carlo sampling error only, not parameter uncertainty or model error. No claim about the optimal strategy or a bank covenant should be made from these outputs.
7. **Mean-reverting EBITDA:** This is a deliberately simple positive process, not a complete operating forecast. Other dynamics, lagged effects and scenario regimes should be examined.

## Research tasks before using with Munters

- Get Treasury approval and appropriate data; document the exact denominator and numerator of the risk metric.
- Match cash-flow dates, currencies, exposures, resets and actual hedge contracts.
- Estimate and validate rate and EBITDA parameters using relevant historical time series.
- Check full horizon with rolling calculations and examine worse stress regimes / sensitivity to correlation.
- If choosing the least-cost compliant strategy, explicitly specify a probability limit such as P(any breach) <= 5% and compare **actual expected cost**, including transaction costs and alternative swap prices.
