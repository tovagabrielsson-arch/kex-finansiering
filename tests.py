"""Small deterministic checks for the starter model. Run: python tests.py"""
import numpy as np
from munters_risk_model import Assumptions, quarterly_metrics
from exposure_model import synthetic_history, fit_exposure, simulate, load_csv
from dataclasses import replace

a=Assumptions(simulations=1000)
r,e,m=synthetic_history(a)
rr,mm,rho,beta,sd,r2=fit_exposure(r,e,m)
assert len(r)==len(e)==len(m)==64
assert np.isfinite(beta).all() and sd>0 and 0<rr['phi']<1
x,y=simulate(a,r,e,m,rr,mm,rho,beta,sd)
assert x.shape==y.shape==(1000,13)
assert (y>0).all() and np.isfinite(x).all()
a=replace(a,initial_reference_rate=float(r[-1]),annual_ebitda_sek=float(e[-1]))
metrics=quarterly_metrics(a,x,y,.5)
assert metrics['icr'].shape==(1000,12)
assert np.isfinite(metrics['icr']).all()
print('All basic model checks passed')
