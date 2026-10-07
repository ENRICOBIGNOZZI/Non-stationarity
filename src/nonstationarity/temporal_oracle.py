"""Monte Carlo checks of the exact temporal mean-estimation subclass (Section 6.6).

This is NOT a neural-network performance experiment. Oracle weights know V and
actual gamma. MU_mean is a feasible doubling selector with known Gaussian scale.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .kernels import power_oracle


def mu_gaussian_mean(y, sigma=.4, delta=.1):
    y=np.asarray(y); n=len(y); r=1
    C1=sigma*np.sqrt(2/np.pi);C2=sigma*np.sqrt(2)
    while 2*r<=n:
        S=(C1+C2*np.sqrt(2*np.log(np.pi**2/(6*delta))))/np.sqrt(r)
        S+=C2*np.sqrt(2*np.log(np.log2(r)+10)/r)
        if abs(np.mean(y[-r:])-np.mean(y[-2*r:]))>4*S:
            break
        r*=2
    return float(np.mean(y[-r:])),r


def run(out, repetitions=200):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows, paths=[],[]
    scenario=0
    for gamma in (.25,.5,1.):
      for V in (.25,1.,2.):
       for T in (128,256,512,1024,2048,4096):
        n=T-1; sigma=.4
        age=np.arange(n,0,-1,dtype=float)/T
        a=age**gamma
        mean=-V*a
        h=np.arange(1,n+1)
        moments=np.cumsum(a[::-1])/h
        mse_unif=sigma**2/h+V**2*moments**2
        best_h=int(np.argmin(mse_unif)+1)
        unif=np.zeros(n);unif[-best_h:]=1/best_h
        half=np.geomspace(1,2*n,120)
        exp=np.exp2(-np.arange(n-1,-1,-1)[None,:]/half[:,None])
        exp/=exp.sum(1,keepdims=True)
        mse_exp=sigma**2*np.sum(exp**2,axis=1)+V**2*(exp@a)**2
        ex=exp[np.argmin(mse_exp)]
        policy={"Oracle_power":power_oracle(a,sigma**2,V**2),"Oracle_uniform":unif,
                "Oracle_exponential":ex,"Expanding":np.ones(n)/n,
                "Last_observation":np.r_[np.zeros(n-1),1.]}
        for hh in (32,128):
            w=np.zeros(n);r=min(hh,n);w[-r:]=1/r
            policy[f"Fixed_{hh}"]=w
        rng=np.random.default_rng(np.random.SeedSequence([104729,scenario]))
        y=mean[None,:]+sigma*rng.normal(size=(repetitions,n))
        for name,w in policy.items():
            err=(y@w)**2;bias=V*(a@w);var=sigma**2*(w@w)
            analytic=var+bias**2
            se=np.sqrt((2*var**2+4*bias**2*var)/repetitions)
            rows.append({"gamma":gamma,"V":V,"T":T,"method":name,
                "mean_mse":float(err.mean()),"analytic_mse":float(analytic),
                "monte_carlo_se":float(se),"neff":float(1/(w@w)),
                "positive_weights":int((w>0).sum()),"replications":repetitions,
                "z_mc_error":float((err.mean()-analytic)/se)})
            for seed,val in enumerate(err):
                paths.append({"case":scenario,"gamma":gamma,"V":V,"T":T,
                              "seed":seed,"method":name,"squared_error":float(val)})
        mu=[mu_gaussian_mean(row,sigma) for row in y]
        err=np.array([v[0]**2 for v in mu])
        rows.append({"gamma":gamma,"V":V,"T":T,"method":"MU_mean",
            "mean_mse":float(err.mean()),"analytic_mse":np.nan,"monte_carlo_se":float(err.std(ddof=1)/np.sqrt(repetitions)),
            "neff":float(np.mean([v[1] for v in mu])),"positive_weights":np.nan,
            "replications":repetitions,"z_mc_error":np.nan})
        for seed,val in enumerate(err):
            paths.append({"case":scenario,"gamma":gamma,"V":V,"T":T,
                          "seed":seed,"method":"MU_mean","squared_error":float(val)})
        scenario+=1
    df=pd.DataFrame(rows); df.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(paths).to_csv(out/'paths.csv',index=False)
    slopes=[]
    for (g,v,name),group in df.groupby(['gamma','V','method']):
        slopes.append({'gamma':g,'V':v,'method':name,
            'empirical_loglog_slope':np.polyfit(np.log(group.T if False else group['T']),np.log(group.mean_mse),1)[0],
            'theoretical_oracle_slope':-2*g/(2*g+1)})
    pd.DataFrame(slopes).to_csv(out/'slopes.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'status':'complete','cases':scenario,
        'replications_per_case':repetitions,'independent_scalar_streams':scenario*repetitions,
        'methods':int(df.method.nunique()),'largest_abs_mc_z':float(df.z_mc_error.abs().max()),
        'sigma':.4,'warning':'Exact temporal mean subclass, not a neural-network benchmark.'},indent=2))
    print((out/'manifest.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--repetitions',type=int,default=200)
    a=p.parse_args();run(a.out,a.repetitions)
