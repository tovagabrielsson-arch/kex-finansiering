"""Exposure-based EBITDA stress modelling inspired by Andrén et al. (2005).
Research prototype. Defaults are SYNTHETIC, not Munters observations.
"""
from __future__ import annotations
import argparse, csv, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from munters_risk_model import Assumptions, summarise, simulate_joint_paths, save_csv
ROOT=Path(__file__).resolve().parent

def synthetic_history(a, n=64):
    rng=np.random.default_rng(1337)
    r=np.full(n,0.025); m=np.zeros(n); loge=np.full(n,np.log(a.annual_ebitda_sek))
    for t in range(1,n):
        zr,zi,ze=rng.normal(size=3)
        r[t]=0.028+.87*(r[t-1]-.028)+.007*zr
        m[t]=.5*m[t-1]+.8*(.15*zr+np.sqrt(1-.15**2)*zi)
        loge[t]=np.log(a.annual_ebitda_sek)+.012+.72*(loge[t-1]-np.log(a.annual_ebitda_sek))-.025*100*(r[t]-r[t-1])+.045*m[t]+.065*ze
    return r,np.exp(loge),m

def load_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    required=['date','reference_rate','annualized_ebitda_sek','macro_factor']
    if len(rows)<20 or any(any(k not in row for k in required) for row in rows):
        raise ValueError('Need >=20 quarterly observations and columns: '+','.join(required))
    index=[]
    for row in rows:
        year,q=row['date'].split('-Q'); q=int(q)
        if q not in [1,2,3,4]:raise ValueError('Invalid quarter')
        index.append(int(year)*4+q)
    if not np.all(np.diff(index)==1):raise ValueError('Quarter dates must be consecutive and sorted')
    x=np.array([[float(z[k]) for k in required[1:]] for z in rows])
    if not np.isfinite(x).all() or np.any(x[:,1]<=0) or np.any(np.abs(x[:,0])>1):
        raise ValueError('Invalid data: EBITDA must be positive and rates in decimals')
    return x[:,0],x[:,1],x[:,2]

def ar1(x):
    X=np.column_stack((np.ones(len(x)-1),x[:-1])); intercept,phi=np.linalg.lstsq(X,x[1:],rcond=None)[0]
    resid=x[1:]-X@np.array([intercept,phi]); sd=np.std(resid,ddof=2)
    if not (0.01<phi<0.995) or sd<=0:raise ValueError(f'AR(1) estimate unstable (phi={phi:.3f}); check history')
    return {'phi':float(phi),'mean':float(intercept/(1-phi)),'sd':float(sd),'resid':resid}

def fit_exposure(r,e,m):
    rr,mm=ar1(r),ar1(m)
    rho=np.clip(np.corrcoef(rr['resid'],mm['resid'])[0,1],-.95,.95)
    # Fit log relative EBITDA to reduce numerical scale differences.
    # Calibration anchor must match simulation anchor (last historical EBITDA).
    loge=np.log(e/e[-1]); X=np.column_stack((np.ones(len(e)-1),loge[:-1],100*np.diff(r),m[1:]))
    beta=np.linalg.lstsq(X,loge[1:],rcond=None)[0]
    resid=loge[1:]-X@beta
    error_sd=np.sqrt((resid@resid)/(len(resid)-X.shape[1]))
    denom=np.sum((loge[1:]-np.mean(loge[1:]))**2)
    r2=1-(resid@resid)/denom if denom else np.nan
    return rr,mm,float(rho),beta,float(error_sd),float(r2)

def simulate(a,r,e,m,rr,mm,rho,beta,error_sd):
    steps=a.horizon_years*a.periods_per_year; rng=np.random.default_rng(a.seed)
    zr=rng.normal(size=(a.simulations,steps)); zi=rng.normal(size=zr.shape); ze=rng.normal(size=zr.shape)
    zm=rho*zr+np.sqrt(1-rho**2)*zi
    rates=np.empty((a.simulations,steps+1)); macro=np.empty_like(rates); logs=np.zeros_like(rates)
    rates[:,0]=r[-1];macro[:,0]=m[-1]
    for t in range(steps):
        rates[:,t+1]=rr['mean']+rr['phi']*(rates[:,t]-rr['mean'])+rr['sd']*zr[:,t]
        macro[:,t+1]=mm['mean']+mm['phi']*(macro[:,t]-mm['mean'])+mm['sd']*zm[:,t]
        logs[:,t+1]=beta[0]+beta[1]*logs[:,t]+beta[2]*100*(rates[:,t+1]-rates[:,t])+beta[3]*macro[:,t+1]+error_sd*ze[:,t]
    return rates,e[-1]*np.exp(np.clip(logs,-12,12))

def main(path):
    a=Assumptions(); synthetic=path is None
    r,e,m=synthetic_history(a) if synthetic else load_csv(path)
    rr,mm,rho,beta,sd,r2=fit_exposure(r,e,m)
    a=replace(a,initial_reference_rate=float(r[-1]),annual_ebitda_sek=float(e[-1]))
    rate,ebitda=simulate(a,r,e,m,rr,mm,rho,beta,sd)
    old_rate,old_ebitda=simulate_joint_paths(a)
    out=ROOT/'output';out.mkdir(exist_ok=True)
    exposure_rows=summarise(a,rate,ebitda); original_rows=summarise(a,old_rate,old_ebitda)
    save_csv(out/'exposure_strategy_results.csv',exposure_rows)
    save_csv(out/'original_strategy_results.csv',original_rows)
    no_macro=beta.copy();no_macro[3]=0
    rm,em=simulate(a,r,e,m,rr,mm,rho,no_macro,sd)
    save_csv(out/'no_macro_strategy_results.csv',summarise(a,rm,em))
    terminal=ebitda[:,-1]/1e6; target=float(np.mean(terminal));p05=float(np.quantile(terminal,.05))
    report={'data_mode':'SYNTHETIC DEMO — NOT MUNTERS DATA' if synthetic else 'USER-SUPPLIED DATA (NOT VERIFIED)',
        'n_historical_quarters':len(e),'in_sample_R2':r2,
        'EBITDA_OLS_coefficients':{'intercept':float(beta[0]),'lagged_log_EBITDA':float(beta[1]),'change_rate_in_percentage_points':float(beta[2]),'macro_factor':float(beta[3])},
        'unexplained_residual_sd':sd,'rate_macro_shock_correlation':rho,
        'terminal_annualized_EBITDA_proxy_CFaR95_MSEK':round(target-p05,2),
        'note':'This is an annualized-EBITDA downside proxy, NOT actual corporate cash flow-at-risk or a Munters covenant probability.'}
    (out/'calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,5))
    for name,rows in [('Original model',original_rows),('Exposure-based model',exposure_rows)]:
        ax.plot([x['initial_weighted_fixing_years'] for x in rows],[x['prob_any_icr_breach_pct'] for x in rows],marker='o',label=name)
    ax.set(xlabel='Initial average fixing (years)',ylabel='Probability of any ICR threshold breach (%)',title='Compare modelling approaches — hypothetical debt assumptions')
    ax.legend();fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));time=np.arange(ebitda.shape[1])/a.periods_per_year
    for quant,name in [(0.05,'5th percentile'),(.5,'Median'),(.95,'95th percentile')]:
        ax.plot(time,np.quantile(ebitda/1e6,quant,axis=0),label=name)
    ax.set(xlabel='Years',ylabel='Annualized EBITDA (MSEK)',title='Exposure model: scenario bands')
    ax.legend();fig.tight_layout();fig.savefig(out/'ebitda_bands.png',dpi=160);plt.close(fig)
    print(report['data_mode']);print('OLS R2:',round(r2,3));print('Coefficient vector:',np.round(beta,4))
    for row in exposure_rows:print(f"{row['fixed_share_pct']}% fixed : P(ICR breach)={row['prob_any_icr_breach_pct']}%")
    print('Output in',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',help='Historical quarterly CSV, otherwise synthetic demonstration')
    args=p.parse_args();main(args.data)
