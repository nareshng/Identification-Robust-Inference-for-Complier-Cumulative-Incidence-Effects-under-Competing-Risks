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


def aj_if(times, causes, tau, target=1):
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
        phi -= ((ft-fu)/(y/m))*(dna-risk*da/y)
    return f,phi


def estimates(z,a,t,c,tau=8):
    n=len(z); out={}; IF_F={}; IF_p={}
    for zz in (0,1):
        ids=np.where(z==zz)[0]; pi=len(ids)/n
        f,phi=aj_if(t[ids],c[ids],tau)
        p=a[ids].mean(); fullf=np.zeros(n); fullp=np.zeros(n)
        fullf[ids]=phi/pi; fullp[ids]=(a[ids]-p)/pi
        out[zz]=(f,p); IF_F[zz]=fullf; IF_p[zz]=fullp
    nhat=out[1][0]-out[0][0]; khat=out[1][1]-out[0][1]
    ifn=IF_F[1]-IF_F[0]; ifk=IF_p[1]-IF_p[0]
    return nhat,khat,ifn,ifk


def intervals(nhat,khat,ifn,ifk,truth,alpha=.05,grid_n=4001):
    zcrit=1.959963984540054
    n=len(ifn); grid=np.linspace(-1,1,grid_n)
    # Score inversion on the natural feasible parameter space.
    var=np.array([np.mean((ifn-p*ifk)**2) for p in grid])
    stat=n*(nhat-grid*khat)**2/np.maximum(var,1e-14)
    ok=stat<=zcrit*zcrit
    score=(grid[ok].min(),grid[ok].max()) if ok.any() else (np.nan,np.nan)
    vt=np.mean((ifn-truth*ifk)**2)
    truth_accepted=n*(nhat-truth*khat)**2/max(vt,1e-14)<=zcrit*zcrit
    ratio=nhat/khat if abs(khat)>1e-12 else np.nan
    if np.isfinite(ratio):
        vr=np.mean((ifn-ratio*ifk)**2)/(n*khat*khat)
        wald=(ratio-zcrit*np.sqrt(vr),ratio+zcrit*np.sqrt(vr))
    else: wald=(np.nan,np.nan)
    return ratio,wald,score,ok.mean()*2,truth_accepted,ok.all()


def true_effect(delta, seed=8675309, n_mc=2_000_000, tau=8):
    rng=np.random.default_rng(seed); batch=200_000; num=den=0.0; total=0
    while total<n_mc:
        n=min(batch,n_mc-total)
        z,a,ot,oc,a0,a1,t,c=generate(n,delta,0,rng)
        comp=a1>a0
        # Regenerate paired potential event paths with common uniforms is not
        # required: estimate E[Y^a|complier] separately with fresh hazards.
        # Use the observed structural event generator twice conditional on X,U
        # would require returning latent variables; instead exploit randomized Z:
        y=(t<=tau)&(c==1)
        num += y[(z==1)&comp].sum()/max(1,((z==1)&comp).sum()) - y[(z==0)&comp].sum()/max(1,((z==0)&comp).sum())
        den += 1; total += n
    return num/den


def run(args):
    start=time.time(); rows=[]
    for delta in args.strengths:
        truth=true_effect(delta,n_mc=args.truth_mc)
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
    meta=dict(seed=args.seed,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
              runtime_seconds=time.time()-start,arguments=vars(args))
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results')
    p.add_argument('--replications',type=int,default=1000)
    p.add_argument('--truth-mc',type=int,default=2_000_000)
    p.add_argument('--seed',type=int,default=20260819)
    p.add_argument('--sample-sizes',type=int,nargs='+',default=[500,1000])
    p.add_argument('--strengths',type=float,nargs='+',default=[.05,.15,.35])
    p.add_argument('--censoring',type=float,nargs='+',default=[0,.30])
    run(p.parse_args())
