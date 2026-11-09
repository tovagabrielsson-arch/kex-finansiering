# Munters thesis model v2 — exposure-based EBITDA

**Academic research prototype, NOT a Munters risk assessment. All default data and borrowing terms are HYPOTHETICAL.**

Developed as a transparent *adaptation inspired by* Andrén, Jankensgård & Oxelheim (2005), *Exposure-based Cash-Flow-at-Risk under Macroeconomic Uncertainty*, IFN Working Paper 635. This is **not** their implementation and does **not** reproduce Norsk Hydro estimates.

Paper: https://www.ifn.se/media/ghmggje3/wp635.pdf

## Start (Mac/Windows/Linux)

```bash
python -m pip install -r requirements.txt
python exposure_model.py
```

The default run uses **synthetic historical data** to demonstrate the code. It then estimates relationships and runs 10,000 hypothetical three-year quarterly scenarios. Outputs are written locally to `output/`.

## Using real quarterly historical data

```bash
python exposure_model.py --data data/history.csv
```

Required CSV columns, in consecutive ascending quarters:

```csv
date,reference_rate,annualized_ebitda_sek,macro_factor
2020-Q1,0.012,1800000000,-0.5
2020-Q2,0.009,1400000000,-1.4
```

**These two sample lines are only for column-format illustration and are not Munters observations.** Do not treat them as historical data. Provide at least 20 observations (40+ preferable).

- `reference_rate`: annual decimal, e.g. `0.025` for 2.5%.
- `annualized_ebitda_sek`: positive quarterly EBITDA **multiplied by four**, or another consistently defined annualized EBITDA level; not trailing-year EBITDA unless you specifically adapt the model. Quarterly EBITDA must use the same accounting definition throughout.
- `macro_factor`: macroeconomic risk factor constructed externally from relevant observed data, such as GDP growth *standardized on a training sample*. Using `macro_factor` is optional in the broader research design, but required by this particular example program. The chosen macro variable and scaling must be reported and defended.
- Only use chronological quarterly data; the program checks dates and basic values, not data provenance or accounting comparability.

## What changed vs v1

1. Rates are projected with a fitted **AR(1)** process. For equally spaced dates this is equivalent to a discretely sampled Vasicek/OU process when estimated persistence is in (0,1); it is a *physical-measure* statistical fit, not a derivative-pricing calibration.
2. The macro factor has its own fitted AR(1), and its shocks may correlate with rate shocks.
3. EBITDA is predicted with an OLS exposure equation:

   `log(EBITDA_t / EBITDA_start) = a + phi*log(EBITDA_(t-1)/EBITDA_start) + beta_r*100*(r_t-r_(t-1)) + beta_m*macro_t + error_t`.

   **Important modeling detail:** `EBITDA_start` is the last historical EBITDA used as the forecast anchor, but OLS is estimated with the same anchor across historical observations. This makes the level and intercept comparable between training and forecast.
4. The regression residual is simulated too, capturing EBITDA variation not explained by the included macro factors; the residual is assumed normal and independent of future rate/macro innovations.
5. Reports terminal annualized-EBITDA downside shortfall at 95% vs simulated expected EBITDA (labeled an **EBITDA proxy**, not actual cash-flow CFaR).
6. Produces the original baseline, the exposure model and a no-macro sensitivity case; ICR is still computed from rolling 4-quarter EBITDA / gross interest expense.

## Output files

- `output/calibration.json`: fitted parameters, proxy CFaR, methodological warnings.
- `output/exposure_strategy_results.csv`: fixed/floating strategies and probabilities.
- `output/original_strategy_results.csv`: v1 process baseline.
- `output/no_macro_strategy_results.csv`: macro-factor sensitivity.
- `output/model_comparison.png` and `output/ebitda_bands.png`.

## Key limitations

- **The paper's full MUST method is not reproduced.** The model borrows the paper's central idea of regression-based cash-flow exposure plus risk-factor simulation and residual variability. There is a single standardized macro factor in the example. GDP, inflation, FX and input costs could be added if economically justified and observations suffice.
- Real correlation between macro factors and the rate requires appropriate data; training estimates are unstable with short quarterly histories. Stress testing and out-of-sample comparisons are necessary.
- Interest expense engine is inherited from v1: fixed debt share with one 3-year swap rate, simplified resets, constant debt and spread, no maturity/refinancing/FX, no contractual swaps.
- The current `ICR` uses **rolling 4-quarter EBITDA / rolling gross interest**, not a confirmed covenant definition. Starting rolling history is synthetic even when real historical data is provided, because v1 metrics functions are not yet adjusted to read the actual preceding three quarters. Avoid interpreting the first three forecast tests as historical TTM covenant calculations.
- The `fixed_share` slider is not a full debt cash-flow repricing schedule; the x-axis reports an *initial weighted fixing period*.
- OLS includes lagged EBITDA and contemporaneous rate/macro values. It is a predictive scenario model, not causal proof; temporal data alignment and possible endogeneity need review.
- The code does not download information, connect to bank accounts, or perform external actions. It only reads explicitly supplied CSV data and writes files to `output/`.
- Read the paper, verify data definitions, then estimate, backtest and stress parameters before claiming any real Munters result.
