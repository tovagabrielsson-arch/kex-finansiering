"""Research prototype: interest-rate fixing and EBITDA-based interest coverage.

All input values are HYPOTHETICAL and are NOT Munters' actual financial data.
Uses physical-measure Vasicek simulation with a correlated, mean-reverting
log-EBITDA process. No valuation or optimization claim is made.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Assumptions:
    debt_sek: float = 6_000_000_000
    annual_ebitda_sek: float = 1_200_000_000
    initial_reference_rate: float = 0.025
    floating_credit_spread: float = 0.015
    fixed_swap_rate: float = 0.030
    fixed_years: float = 3.0
    horizon_years: int = 3
    periods_per_year: int = 4
    simulations: int = 10_000
    # Physical (real-world) Vasicek parameters: dr=kappa(theta-r)dt+sigma_r dW
    rate_kappa: float = 0.75
    rate_long_run: float = 0.030
    rate_sigma: float = 0.018  # absolute rate units / sqrt(year)
    # d log(EBITDA) = kappa_e*(mu_e-log EBITDA)dt + sigma_e*dW_e
    ebitda_kappa: float = 0.40
    ebitda_long_run_growth_factor: float = 1.05
    ebitda_log_sigma: float = 0.24  # log units / sqrt(year)
    shock_correlation: float = 0.25
    icr_floor: float = 3.0
    seed: int = 20261105


def validate(a: Assumptions) -> None:
    if not (-1 <= a.shock_correlation <= 1):
        raise ValueError('shock_correlation must be between -1 and 1')
    if min(a.debt_sek, a.annual_ebitda_sek, a.simulations,
           a.horizon_years, a.periods_per_year, a.fixed_years) <= 0:
        raise ValueError('Debt, EBITDA, simulations, horizon, frequency and fixing years must be positive')
    if a.rate_kappa <= 0 or a.ebitda_kappa <= 0:
        raise ValueError('Mean-reversion speeds must be positive')
    if a.fixed_years < a.horizon_years:
        raise ValueError('For this version, fixed swap maturity must cover the full simulation horizon')


def simulate_joint_paths(a: Assumptions) -> tuple[np.ndarray, np.ndarray]:
    """Returns (annual reference rate, annualized EBITDA), shapes [paths, quarters+1].

    Correlation applies to standardized innovations of the two stochastic processes.
    In exact OU discretization, their quarter innovations need not have exactly that
    correlation if mean reversion speeds differ; this simplification is documented.
    """
    validate(a)
    nsteps = a.horizon_years * a.periods_per_year
    dt = 1 / a.periods_per_year
    rng = np.random.default_rng(a.seed)
    z1 = rng.standard_normal((a.simulations, nsteps))
    z_ind = rng.standard_normal((a.simulations, nsteps))
    z2 = a.shock_correlation * z1 + math.sqrt(1-a.shock_correlation**2) * z_ind

    rates = np.empty((a.simulations, nsteps+1))
    logebitda = np.empty_like(rates)
    rates[:, 0] = a.initial_reference_rate
    logebitda[:, 0] = np.log(a.annual_ebitda_sek)
    mu_log = np.log(a.annual_ebitda_sek * a.ebitda_long_run_growth_factor)

    ar = np.exp(-a.rate_kappa * dt)
    ae = np.exp(-a.ebitda_kappa * dt)
    rate_sd = a.rate_sigma * np.sqrt((1-ar**2)/(2*a.rate_kappa))
    ebitda_sd = a.ebitda_log_sigma * np.sqrt((1-ae**2)/(2*a.ebitda_kappa))

    for t in range(nsteps):
        rates[:, t+1] = a.rate_long_run + (rates[:, t] - a.rate_long_run)*ar + rate_sd*z1[:, t]
        logebitda[:, t+1] = mu_log + (logebitda[:, t]-mu_log)*ae + ebitda_sd*z2[:, t]
    return rates, np.exp(logebitda)


def quarterly_metrics(a: Assumptions, rates: np.ndarray, ebitda: np.ndarray,
                      fixed_share: float) -> dict[str, np.ndarray]:
    """Quarterly TTM interest coverage. Fixed-rate swaps hedge the reference index.

    The hypothetical initial TTM history uses 3 pre-start quarters of constant
    annualized EBITDA and quarterly interest at the initial floating index.
    For consistency, fixed hedges are assumed already active during this history.
    Interest for quarter t uses the rate reset at its START (t-1), while EBITDA
    for that quarter uses the END (t). Day count and compounding simplified to dt.
    """
    if not (0 <= fixed_share <= 1):
        raise ValueError('fixed_share must be within [0, 1]')
    q = a.periods_per_year
    dt = 1/q
    n = rates.shape[1]-1
    if rates.shape != ebitda.shape:
        raise ValueError('rates and ebitda shapes must match')
    # Disallow an unrealistic negative all-in coupon in the toy model.
    float_index = np.maximum(rates[:, :-1], 0.0)
    all_in_rate = fixed_share*a.fixed_swap_rate + (1-fixed_share)*float_index + a.floating_credit_spread
    cost = a.debt_sek * all_in_rate * dt
    realized_ebitda = ebitda[:, 1:] * dt
    initial_cost = a.debt_sek * (fixed_share*a.fixed_swap_rate +
                    (1-fixed_share)*max(a.initial_reference_rate, 0.0) +
                    a.floating_credit_spread)*dt
    history_cost = np.full((rates.shape[0], q-1), initial_cost)
    history_ebitda = np.full((rates.shape[0], q-1), a.annual_ebitda_sek*dt)
    full_cost = np.concatenate([history_cost, cost], axis=1)
    full_ebitda = np.concatenate([history_ebitda, realized_ebitda], axis=1)
    ttm_cost = np.stack([full_cost[:, t:t+q].sum(axis=1) for t in range(n)], axis=1)
    ttm_ebitda = np.stack([full_ebitda[:, t:t+q].sum(axis=1) for t in range(n)], axis=1)
    icr = np.divide(ttm_ebitda, ttm_cost, out=np.full_like(ttm_cost, np.inf), where=ttm_cost>0)
    return dict(quarterly_cost=cost, quarterly_ebitda=realized_ebitda,
                ttm_cost=ttm_cost, ttm_ebitda=ttm_ebitda, icr=icr)


def summarise(a: Assumptions, rates: np.ndarray, ebitda: np.ndarray,
              shares: tuple[float, ...] = (0, .25, .5, .75, 1.)) -> list[dict]:
    rows = []
    for share in shares:
        m = quarterly_metrics(a, rates, ebitda, share)
        # At least one breach across 12 quarterly TTM testing dates per path.
        ever = (m['icr'] < a.icr_floor).any(axis=1)
        p = float(ever.mean())
        ci = 1.96*math.sqrt(p*(1-p)/a.simulations)  # Monte Carlo error only
        rows.append({
            'fixed_share_pct': round(share*100),
            'initial_weighted_fixing_years': round(share*a.fixed_years + (1-share)/a.periods_per_year, 3),
            'mean_total_interest_msek': round(float(m['quarterly_cost'].sum(axis=1).mean()/1e6), 2),
            'p95_total_interest_msek': round(float(np.quantile(m['quarterly_cost'].sum(axis=1), .95)/1e6), 2),
            'prob_any_icr_breach_pct': round(100*p, 2),
            'mc_95ci_low_pct': round(100*max(0,p-ci), 2),
            'mc_95ci_high_pct': round(100*min(1,p+ci), 2),
            'median_min_icr': round(float(np.median(m['icr'].min(axis=1))), 2),
        })
    return rows


def static_stress(a: Assumptions, shares: tuple[float,...]=(0,.25,.5,.75,1.),
                  rate_shock: float = .03, ebitda_shock: float = -.30) -> list[dict]:
    """Instantaneous annualized shock, not a time-path or an estimated probability."""
    rows = []
    r = max(0.0, a.initial_reference_rate + rate_shock)
    e = a.annual_ebitda_sek * (1+ebitda_shock)
    for x in shares:
        interest = a.debt_sek * (x*a.fixed_swap_rate +(1-x)*r + a.floating_credit_spread)
        rows.append({'fixed_share_pct': round(100*x),
                     'stressed_interest_msek': round(interest/1e6, 2),
                     'stressed_ebitda_msek': round(e/1e6, 2),
                     'stressed_icr': round(e/interest, 2),
                     'below_floor': e/interest < a.icr_floor})
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(out: Path, a: Assumptions, rates: np.ndarray,
                 ebitda: np.ndarray, results: list[dict]) -> None:
    t = np.arange(rates.shape[1])/a.periods_per_year
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(min(35,len(rates))):
        ax.plot(t, rates[i]*100, alpha=.20, linewidth=.8)
    ax.plot(t, np.mean(rates,axis=0)*100, color='black', linewidth=2, label='Average of simulations')
    ax.set(title='Simulated reference-rate paths (hypothetical)', xlabel='Years', ylabel='Reference rate (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'simulated_rates.png',dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(min(35,len(ebitda))):
        ax.plot(t, ebitda[i]/1e6, alpha=.20, linewidth=.8)
    ax.plot(t, np.mean(ebitda,axis=0)/1e6, color='black', linewidth=2, label='Average of simulations')
    ax.set(title='Simulated annualized EBITDA (hypothetical)', xlabel='Years', ylabel='EBITDA (MSEK)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'simulated_ebitda.png',dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x=[r['initial_weighted_fixing_years'] for r in results]
    y=[r['prob_any_icr_breach_pct'] for r in results]
    ax.plot(x,y,marker='o',linewidth=2)
    ax.set(title=f'Probability of at least one TTM ICR < {a.icr_floor} over {a.horizon_years} years',
           xlabel='Initial weighted average fixing period (years)', ylabel='Breach probability (%)')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out/'breach_vs_fixing.png',dpi=160)
    plt.close(fig)


def main() -> None:
    a=Assumptions()
    out=Path(__file__).resolve().parent/'output'
    out.mkdir(parents=True,exist_ok=True)
    rates, ebitda = simulate_joint_paths(a)
    rows = summarise(a,rates,ebitda)
    stress = static_stress(a)
    save_csv(out/'strategy_results.csv',rows)
    save_csv(out/'static_stress.csv',stress)
    make_figures(out,a,rates,ebitda,rows)
    print('HYPOTHETICAL RESEARCH PROTOTYPE; NOT MUNTERS DATA')
    print(f'Scenarios: {a.simulations:,}, quarters: {a.horizon_years*a.periods_per_year}, ICR floor: {a.icr_floor:.1f}x')
    for r in rows:
        print(f"{r['fixed_share_pct']:>3}% fixed | fixing {r['initial_weighted_fixing_years']:>4.2f}y | "
              f"interest {r['mean_total_interest_msek']:>7.1f} MSEK | "
              f"P(any breach) {r['prob_any_icr_breach_pct']:>5.2f}%")
    print(f'Output files: {out}')

if __name__=='__main__':
    main()
