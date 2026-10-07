"""Fixed-horizon sensitivity from stored causal forecasts; no network retraining.

All score horizons are reported. This is not a test-set horizon selection rule,
and the replay counts must not be counted as independent simulated datasets.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .kernels import candidate_grid
from .experiment import masks_for


def replay(root,out):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('status')!='complete':raise ValueError('Main run must be complete')
    cfg=manifest['config'];specs=candidate_grid(cfg['histories'],cfg['widths']);pools=masks_for(specs)
    methods=['Uniform_CV','Exponential_CV','Power_CV_matched','Joint_CV']
    rows=[]
    for path in sorted(root.glob('*/candidate_ledger.npz')):
        scenario,seed=path.parent.name.rsplit('__',1);data=np.load(path)
        p,y=data['predictions'],data['y'];start=cfg['start']
        losses=(p[start:]-y[start:,None])**2
        sums=np.vstack([np.zeros(p.shape[1]),np.cumsum(losses,axis=0)])
        for L in (32,64,128,256):
            for method in methods:
                ids=pools[method];risks=[]
                for t,risk in zip(data['eval_times'],data['risk']):
                    end=int(t)-start;lo=max(0,end-L)
                    score=(sums[end]-sums[lo])/max(1,end-lo)
                    best=ids[np.argmin(score[ids])]
                    tied=ids[np.isclose(score[ids],score[best],atol=1e-12,rtol=0)]
                    chosen=min(tied,key=lambda j:(specs[j].width,-specs[j].h))
                    risks.append(risk[chosen])
                rows.append({'scenario':scenario,'seed':int(seed),'method':method,'score_horizon':L,'mean_risk':float(np.mean(risks))})
    df=pd.DataFrame(rows);df.to_csv(out/'stream_sensitivity.csv',index=False)
    df.groupby(['score_horizon','method']).mean_risk.mean().reset_index().to_csv(out/'horizon_sensitivity.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'status':'complete','independent_new_streams':0,
        'replayed_method_streams':len(df),'score_horizons':[32,64,128,256],
        'warning':'Secondary fixed-horizon sensitivity; no best test horizon is selected.'},indent=2))
    return df

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(replay(a.input,a.out).groupby(['score_horizon','method']).mean_risk.mean())
