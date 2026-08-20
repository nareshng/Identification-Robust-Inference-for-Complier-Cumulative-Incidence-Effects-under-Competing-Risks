#!/usr/bin/env python3
"""Extended simulations for weak-IV complier CIF score inference.

The script adds four targeted experiments to the core design:
1. local-to-zero compliance, sqrt(n)*kappa_n = c;
2. severe independent censoring;
3. a finite-jump variance ablation for discrete tied event times; and
4. assumption-violation stress tests (informative censoring and defiers).

Only the valid-design experiments support the coverage theorem.  Stress-test
coverage is diagnostic and is explicitly labelled as outside the theorem.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_weak_iv_cif import expit, estimates, quadrature_truth, score_components


def generate_extended(n,delta,censoring,rng,periods=10,censor_u=0.0,defier_rate=0.0):
    x=rng.normal(size=n); u=rng.normal(size=n); v=rng.random(n)
    z=rng.binomial(1,.5,size=n)
    q0=expit(-1+.45*x+.55*u); q1=q0+delta*(1-q0)
    a0=(v<=q0).astype(int); a1=(v<=q1).astype(int)
    if defier_rate>0:
        # Randomly reverse this fraction of original compliers.  Always- and
        # never-takers are unchanged; the remaining complier effect is stable.
        swap=(rng.random(n)<defier_rate)&(a1>a0)
        a0[swap],a1[swap]=a1[swap].copy(),a0[swap].copy()
    a=np.where(z==1,a1,a0)

    t=np.full(n,periods+1,int); cause=np.zeros(n,int); alive=np.ones(n,bool)
    for k in range(1,periods+1):
        idx=np.where(alive)[0]
        e1=np.exp(-3.35+.10*k-.45*a[idx]+.35*x[idx]+.55*u[idx])
        e2=np.exp(-3.55+.07*k+.25*a[idx]-.25*x[idx]+.45*u[idx])
        den=1+e1+e2; p1=e1/den; p2=e2/den
        draw=rng.random(idx.size); c1=draw<p1; c2=(draw>=p1)&(draw<p1+p2)
        hit=c1|c2; who=idx[hit]
        t[who]=k; cause[who]=np.where(c1[hit],1,2); alive[who]=False

    if censoring<=0:
        ctime=np.full(n,periods+1,int)
    else:
        base_h=1-(1-censoring)**(1/periods)
        base_logit=np.log(base_h/(1-base_h))
        ctime=np.full(n,periods+1,int); unc=np.ones(n,bool)
        for k in range(1,periods+1):
            ids=np.where(unc)[0]
            h=expit(base_logit+censor_u*u[ids])
            hit=rng.random(ids.size)<h
            ctime[ids[hit]]=k; unc[ids[hit]]=False
    obs_t=np.minimum(t,ctime); obs_cause=np.where(t<=ctime,cause,0)
    realized_censor=np.mean(ctime<=periods)
    cause1_rate=np.mean((t<=8)&(cause==1))
    return z,a,obs_t,obs_cause,realized_censor,cause1_rate


def interval_diagnostics(nhat,khat,ifn,ifk,truth,alpha=.05):
    """Wald interval and exact quadratic score-set diagnostics."""
    zcrit=1.959963984540054; n=len(ifn)
    components, variance_terms, coef = score_components(nhat,khat,ifn,ifk,alpha)
    vnn,vnk,vkk=variance_terms
    vt=vnn-2*truth*vnk+truth*truth*vkk
    score_cover=n*(nhat-truth*khat)**2/max(vt,1e-14)<=zcrit*zcrit
    score_length=sum(b-a for a,b in components)
    full=(len(components)==1 and components[0][0]<=-1+1e-10 and components[0][1]>=1-1e-10)
    ratio=nhat/khat if abs(khat)>1e-12 else np.nan
    if np.isfinite(ratio):
        vr=max(vnn-2*ratio*vnk+ratio*ratio*vkk,0.0)/(n*khat*khat)
        wald_low=ratio-zcrit*np.sqrt(vr); wald_high=ratio+zcrit*np.sqrt(vr)
    else:
        wald_low=wald_high=np.nan
    flat=[value for component in components[:2] for value in component]
    flat += [np.nan]*(4-len(flat))
    return dict(
        ratio=ratio,wald_low=wald_low,wald_high=wald_high,
        wald_length=wald_high-wald_low,wald_cover=wald_low<=truth<=wald_high,
        score_cover=score_cover,score_length=score_length,
        score_component_count=len(components),score_full_set=full,
        score_disconnected=len(components)>1,score_empty=len(components)==0,
        score_lower1=flat[0],score_upper1=flat[1],
        score_lower2=flat[2],score_upper2=flat[3],
        v_nn=vnn,v_nk=vnk,v_kk=vkk,
        q_a=coef[0],q_b=coef[1],q_c=coef[2],
        truth_score=n*(nhat-truth*khat)**2/max(vt,1e-14),
    )


def one_setting(setting,truth,reps,seed):
    records=[]
    for r in range(reps):
        ss=np.random.SeedSequence([seed,setting['seed_group'],r])
        rng=np.random.default_rng(ss)
        z,a,t,c,real_cens,event_rate=generate_extended(
            setting['n'],setting['delta'],setting['censoring'],rng,
            censor_u=setting['censor_u'],defier_rate=setting['defier_rate'])
        nh,kh,ifn,ifk=estimates(z,a,t,c,finite_jump=setting['finite_jump'])
        diag=interval_diagnostics(nh,kh,ifn,ifk,truth)
        records.append(dict(setting_id=setting['setting_id'],replicate=r,
            scenario=setting['scenario'],n=setting['n'],delta=setting['delta'],
            local_c=setting['local_c'],censoring=setting['censoring'],
            censor_u=setting['censor_u'],defier_rate=setting['defier_rate'],
            finite_jump=setting['finite_jump'],theorem_valid=setting['theorem_valid'],
            seed_group=setting['seed_group'],
            true_effect=truth,numerator=nh,first_stage=kh,
            realized_censoring=real_cens,cause1_event_rate=event_rate,**diag))
    frame=pd.DataFrame(records)
    out=dict(setting)
    out.update(
        true_effect=truth,reps=reps,
        wald_coverage=frame.wald_cover.mean(),score_coverage=frame.score_cover.mean(),
        wald_mean_length=frame.wald_length.mean(),wald_median_length=frame.wald_length.median(),
        wald_p90_length=frame.wald_length.quantile(.90),wald_p95_length=frame.wald_length.quantile(.95),
        wald_p99_length=frame.wald_length.quantile(.99),
        score_mean_length=frame.score_length.mean(),score_median_length=frame.score_length.median(),
        score_p90_length=frame.score_length.quantile(.90),
        score_full_set=frame.score_full_set.mean(),
        score_disconnected=frame.score_disconnected.mean(),score_empty=frame.score_empty.mean(),
        mean_first_stage=frame.first_stage.mean(),sd_first_stage=frame.first_stage.std(ddof=1),
        first_stage_near_zero=(frame.first_stage.abs()<.01).mean(),
        realized_censoring=frame.realized_censoring.mean(),cause1_event_rate=frame.cause1_event_rate.mean(),
        ratio_bias=(frame.ratio-truth).mean(),ratio_rmse=np.sqrt(np.nanmean((frame.ratio-truth)**2)),
        ratio_median_abs_error=(frame.ratio-truth).abs().median(),
        score_coverage_mcse=np.sqrt(frame.score_cover.mean()*(1-frame.score_cover.mean())/reps),
        wald_coverage_mcse=np.sqrt(frame.wald_cover.mean()*(1-frame.wald_cover.mean())/reps),
    )
    return out,records


def setting_task(payload):
    return one_setting(*payload)


def build_grid(compliance_multiplier):
    settings=[]
    def add(scenario,n,delta,censoring=.30,local_c=np.nan,censor_u=0.0,
            defier_rate=0.0,finite_jump=True,theorem_valid=True,seed_group=None):
        setting_id=len(settings)+1
        if seed_group is None: seed_group=setting_id
        settings.append(dict(setting_id=setting_id,scenario=scenario,n=int(n),
            delta=float(delta),local_c=local_c,censoring=float(censoring),
            censor_u=float(censor_u),defier_rate=float(defier_rate),
            finite_jump=bool(finite_jump),theorem_valid=bool(theorem_valid),
            seed_group=int(seed_group)))

    # Core design, matched to the main table.
    for delta in (.05,.15,.35):
        for n in (500,1000):
            for cens in (0,.30): add('core',n,delta,cens)

    # Direct theorem test: sqrt(n)*kappa_n=c exactly in expectation.
    for local_c in (.5,1.0,2.0,4.0):
        for n in (500,1000,2500,5000):
            delta=local_c/(compliance_multiplier*np.sqrt(n))
            add('local_to_zero',n,delta,.30,local_c=local_c)

    # Positivity/information degradation while independent censoring remains valid.
    for n in (500,1000):
        for cens in (.50,.65): add('severe_censoring',n,.15,cens)

    # Variance ablation: omission of the discrete-time finite-jump factor.
    for n in (500,1000):
        for delta in (.15,.35):
            pair_seed=1000+10*n+int(100*delta)
            add('tie_ablation_correct',n,delta,.30,finite_jump=True,seed_group=pair_seed)
            add('tie_ablation_naive',n,delta,.30,finite_jump=False,
                theorem_valid=False,seed_group=pair_seed)

    # One-factor violations; these diagnose fragility and do not support coverage.
    for gamma in (.5,1.0):
        add('informative_censoring',1000,.15,.30,censor_u=gamma,theorem_valid=False)
    for rate in (.10,.25):
        add('defiers',1000,.15,.30,defier_rate=rate,theorem_valid=False)
    return settings


def run(args):
    start=time.time()
    truth,compliance_multiplier=quadrature_truth(args.quadrature_nodes)
    truth_check,_=quadrature_truth(max(20,args.quadrature_nodes-20))
    truth_error=abs(truth-truth_check)
    settings=build_grid(compliance_multiplier)
    if args.scenarios:
        keep=set(args.scenarios); settings=[s for s in settings if s['scenario'] in keep]
    primary={'core','local_to_zero'}
    payloads=[]
    for s in settings:
        reps=(args.primary_replications if args.primary_replications is not None
              and s['scenario'] in primary else args.replications)
        payloads.append((s,truth,reps,args.seed))
    if args.workers>1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            completed=list(pool.map(setting_task,payloads))
    else:
        completed=[setting_task(p) for p in payloads]
    rows=[]; replicate_rows=[]
    for row,recs in completed:
        rows.append(row); replicate_rows.extend(recs)
        print(json.dumps({k:row[k] for k in ('setting_id','scenario','n','delta','score_coverage','wald_coverage')}),flush=True)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'extended_simulation_results.csv',index=False)
    pd.DataFrame(replicate_rows).to_csv(out/'extended_replicate_results.csv',index=False)
    meta=dict(seed=args.seed,truth_method='product Gauss-Hermite quadrature',
        quadrature_nodes=args.quadrature_nodes,true_effect=truth,
        quadrature_stability_check=truth_error,compliance_multiplier=compliance_multiplier,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        runtime_seconds=time.time()-start,arguments=vars(args),number_of_settings=len(settings),
        total_replications=sum(payload[2] for payload in payloads),
        note='Stress-test rows with theorem_valid=false are outside the coverage theorem.')
    (out/'extended_run_metadata.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='extended_results')
    p.add_argument('--replications',type=int,default=1000)
    p.add_argument('--primary-replications',type=int)
    p.add_argument('--quadrature-nodes',type=int,default=60)
    p.add_argument('--seed',type=int,default=20260819)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--scenarios',nargs='*')
    run(p.parse_args())
