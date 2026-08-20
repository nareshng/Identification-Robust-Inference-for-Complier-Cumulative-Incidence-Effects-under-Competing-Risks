#!/usr/bin/env python3
"""Simulation for weak-IV robust complier cumulative-incidence inference.

Python 3.11+, NumPy and pandas only.  The implementation uses an
Aalen--Johansen estimator and its plug-in influence representation.
"""
from __future__ import annotations

import argparse, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def conditional_cif1(x, u, a, tau=8):
    """Cause-1 cumulative incidence through ``tau`` given latent covariates."""
    x, u, a = np.broadcast_arrays(x, u, a)
    survival = np.ones_like(x, dtype=float)
    cif = np.zeros_like(x, dtype=float)
    for k in range(1, tau + 1):
        e1 = np.exp(-3.35 + .10*k - .45*a + .35*x + .55*u)
        e2 = np.exp(-3.55 + .07*k + .25*a - .25*x + .45*u)
        den = 1 + e1 + e2
        cif += survival * e1 / den
        survival *= 1 - (e1 + e2) / den
    return cif


def quadrature_truth(nodes=60, tau=8):
    """Return the delta-invariant complier effect and compliance multiplier."""
    gh_x, gh_w = np.polynomial.hermite.hermgauss(nodes)
    x = np.sqrt(2.0) * gh_x[:, None]
    u = np.sqrt(2.0) * gh_x[None, :]
    weights = gh_w[:, None] * gh_w[None, :] / np.pi
    complier_weight = 1 - expit(-1 + .45*x + .55*u)
    contrast = conditional_cif1(x, u, 1.0, tau) - conditional_cif1(x, u, 0.0, tau)
    compliance_multiplier = np.sum(weights * complier_weight)
    truth = np.sum(weights * complier_weight * contrast) / compliance_multiplier
    return float(truth), float(compliance_multiplier)


def generate(n, delta, censoring, rng, periods=10):
    x = rng.normal(size=n); u = rng.normal(size=n); v = rng.random(n)
    z = rng.binomial(1, .5, size=n)
    q0 = expit(-1 + .45*x + .55*u)
    q1 = q0 + delta*(1-q0)
    a0 = (v <= q0).astype(int); a1 = (v <= q1).astype(int)
    a = np.where(z == 1, a1, a0)
    t = np.full(n, periods + 1, int); cause = np.zeros(n, int)
    alive = np.ones(n, bool)
    for k in range(1, periods + 1):
        idx = np.where(alive)[0]
        e1 = np.exp(-3.35 + .10*k - .45*a[idx] + .35*x[idx] + .55*u[idx])
        e2 = np.exp(-3.55 + .07*k + .25*a[idx] - .25*x[idx] + .45*u[idx])
        den = 1 + e1 + e2; p1=e1/den; p2=e2/den
        r = rng.random(idx.size); c1=r<p1; c2=(r>=p1)&(r<p1+p2)
        hit = c1|c2; who=idx[hit]
        t[who]=k; cause[who]=np.where(c1[hit],1,2); alive[who]=False
    if censoring <= 0:
        ctime=np.full(n, periods+1, int)
    else:
        # Constant discrete hazard calibrated to P(C<=periods)=censoring.
        hc=1-(1-censoring)**(1/periods)
        ctime=np.full(n, periods+1, int); unc=np.ones(n,bool)
        for k in range(1,periods+1):
            ids=np.where(unc)[0]; hit=rng.random(ids.size)<hc
            ctime[ids[hit]]=k; unc[ids[hit]]=False
    # Events win ties, an explicit convention.
    obs_t=np.minimum(t,ctime); obs_cause=np.where(t<=ctime,cause,0)
    return z,a,obs_t,obs_cause,a0,a1,t,cause


def aj_if(times, causes, tau, target=1, finite_jump=True):
    """AJ CIF and estimated influence values within one independent arm."""
    m=len(times); event_times=np.unique(times[(times<=tau)&(causes>0)])
    s=1.0; f=0.0; rows=[]
    for tt in event_times:
        risk=times>=tt; y=risk.sum()
        d1=((times==tt)&(causes==target)).sum()
        da=((times==tt)&(causes>0)).sum()
        inc=s*d1/y; f += inc
        rows.append((tt,s,f,y,d1,da,risk))
        s *= 1-da/y
    phi=np.zeros(m)
    ft=f
    for tt,sprev,fu,y,d1,da,risk in rows:
        dn1=((times==tt)&(causes==target)).astype(float)
        dna=((times==tt)&(causes>0)).astype(float)
        # Counting-process influence representation; y/m estimates H(tt).
        phi += (sprev/(y/m))*(dn1-risk*d1/y)
        # Finite-jump correction for tied/discrete event times. In the
        # continuous-time no-tie limit, 1-da/y tends to one.
        jump_survival=max(1-da/y,1e-12) if finite_jump else 1.0
        phi -= ((ft-fu)/((y/m)*jump_survival))*(dna-risk*da/y)
    return f,phi


def estimates(z,a,t,c,tau=8,finite_jump=True):
    n=len(z); out={}; IF_F={}; IF_p={}
    for zz in (0,1):
        ids=np.where(z==zz)[0]; pi=len(ids)/n
        f,phi=aj_if(t[ids],c[ids],tau,finite_jump=finite_jump)
        p=a[ids].mean(); fullf=np.zeros(n); fullp=np.zeros(n)
        fullf[ids]=phi/pi; fullp[ids]=(a[ids]-p)/pi
        out[zz]=(f,p); IF_F[zz]=fullf; IF_p[zz]=fullp
    nhat=out[1][0]-out[0][0]; khat=out[1][1]-out[0][1]
    ifn=IF_F[1]-IF_F[0]; ifk=IF_p[1]-IF_p[0]
    return nhat,khat,ifn,ifk


def score_components(nhat, khat, ifn, ifk, alpha=.05, domain=(-1.0, 1.0)):
    """Exact components of the quadratic score-inversion set."""
    n = len(ifn); crit = 1.959963984540054**2
    vnn = float(np.mean(ifn*ifn)); vnk = float(np.mean(ifn*ifk)); vkk = float(np.mean(ifk*ifk))
    coef = np.array([
        n*khat*khat - crit*vkk,
        -2*n*nhat*khat + 2*crit*vnk,
        n*nhat*nhat - crit*vnn,
    ], dtype=float)
    scale = max(float(np.max(np.abs(coef))), 1.0)
    tol = 1e-12 * scale
    lo, hi = map(float, domain)
    if abs(coef[0]) <= tol:
        roots = [] if abs(coef[1]) <= tol else [-coef[2]/coef[1]]
    else:
        disc = coef[1]*coef[1] - 4*coef[0]*coef[2]
        roots = [] if disc < -tol else [
            (-coef[1]-np.sqrt(max(disc, 0.0)))/(2*coef[0]),
            (-coef[1]+np.sqrt(max(disc, 0.0)))/(2*coef[0]),
        ]
    cuts = [lo] + sorted(r for r in roots if lo < r < hi) + [hi]
    q = lambda b: (coef[0]*b + coef[1])*b + coef[2]
    components = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        if q((left+right)/2) <= tol:
            if components and abs(components[-1][1]-left) <= 1e-10:
                components[-1] = (components[-1][0], right)
            else:
                components.append((left, right))
    for root in roots:
        if lo <= root <= hi and abs(q(root)) <= 10*tol and not any(a-1e-10 <= root <= b+1e-10 for a,b in components):
            components.append((float(root), float(root)))
    components.sort()
    return components, (vnn, vnk, vkk), tuple(coef)


def intervals(nhat,khat,ifn,ifk,truth,alpha=.05):
    zcrit=1.959963984540054; n=len(ifn)
    components, _, _ = score_components(nhat,khat,ifn,ifk,alpha)
    score=(components[0][0],components[-1][1]) if components else (np.nan,np.nan)
    vt=np.mean((ifn-truth*ifk)**2)
    truth_accepted=n*(nhat-truth*khat)**2/max(vt,1e-14)<=zcrit*zcrit
    ratio=nhat/khat if abs(khat)>1e-12 else np.nan
    if np.isfinite(ratio):
        vr=np.mean((ifn-ratio*ifk)**2)/(n*khat*khat)
        wald=(ratio-zcrit*np.sqrt(vr),ratio+zcrit*np.sqrt(vr))
    else: wald=(np.nan,np.nan)
    length=sum(b-a for a,b in components)
    full=(len(components)==1 and components[0][0] <= -1+1e-10 and components[0][1] >= 1-1e-10)
    return ratio,wald,score,length,truth_accepted,full


def true_effect(delta=None, nodes=60, tau=8):
    """Backward-compatible deterministic truth; ``delta`` cancels exactly."""
    return quadrature_truth(nodes=nodes, tau=tau)[0]


def run(args):
    start=time.time(); rows=[]
    truth, compliance_multiplier = quadrature_truth(args.quadrature_nodes)
    for delta in args.strengths:
        for n in args.sample_sizes:
            for cens in args.censoring:
                vals=[]
                for r in range(args.replications):
                    rng=np.random.default_rng(args.seed+100000*int(delta*1000)+1000*n+100*int(cens*100)+r)
                    z,a,t,c,*_=generate(n,delta,cens,rng)
                    nh,kh,ifn,ifk=estimates(z,a,t,c)
                    ratio,wald,score,length,truth_ok,full=intervals(nh,kh,ifn,ifk,truth)
                    vals.append((ratio, wald[0]<=truth<=wald[1], truth_ok,
                                 wald[1]-wald[0], length,
                                 full,kh))
                v=np.asarray(vals,float)
                rows.append(dict(delta=delta,n=n,censoring=cens,true_effect=truth,reps=args.replications,
                    wald_coverage=v[:,1].mean(),score_coverage=v[:,2].mean(),wald_mean_length=np.nanmean(v[:,3]),
                    score_mean_length=np.nanmean(v[:,4]),score_full_set=v[:,5].mean(),mean_first_stage=v[:,6].mean(),
                    ratio_bias=np.nanmean(v[:,0]-truth),ratio_rmse=np.sqrt(np.nanmean((v[:,0]-truth)**2))))
                print(rows[-1],flush=True)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'simulation_results.csv',index=False)
    meta=dict(seed=args.seed,true_effect=truth,compliance_multiplier=compliance_multiplier,
              python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
              runtime_seconds=time.time()-start,arguments=vars(args))
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results')
    p.add_argument('--replications',type=int,default=1000)
    p.add_argument('--quadrature-nodes',type=int,default=60)
    p.add_argument('--seed',type=int,default=20260819)
    p.add_argument('--sample-sizes',type=int,nargs='+',default=[500,1000])
    p.add_argument('--strengths',type=float,nargs='+',default=[.05,.15,.35])
    p.add_argument('--censoring',type=float,nargs='+',default=[0,.30])
    run(p.parse_args())
