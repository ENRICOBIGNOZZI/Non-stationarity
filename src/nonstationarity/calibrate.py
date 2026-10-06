"""Select validation horizons on disjoint DEVELOPMENT streams, then freeze them.

No held-out evaluation-seed files may be placed in --development. This does not
train or tune any model on a final test target. Candidate losses were generated
prequentially. Risk probes are allowed here as simulation-development objectives.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .kernels import candidate_grid
from .experiment import masks_for
from .selectors import Prequential


def calibrate(root, horizons=(32,64,128,256)):
    root=Path(root); config=json.loads((root/'config.json').read_text())
    specs=candidate_grid(config['histories'],config['widths'])
    masks=masks_for(specs); rows=[]
    for p in sorted(root.glob('*/candidate_ledger.npz')):
        data=np.load(p)
        preds,y=data['predictions'],data['y']
        erisk={int(t):r for t,r in zip(data['eval_times'],data['risk'])}
        seed=int(p.parent.name.rsplit('__',1)[1])
        if seed>=1000:
            raise ValueError('Final test seed in development directory')
        for name,ids in masks.items():
            for H in horizons:
                selector=Prequential(specs,H,1. if name=='Joint_1SE' else 0.)
                scores=[]
                for t in range(config.get('start',64),len(y)):
                    j=selector.choose(ids)
                    if t in erisk: scores.append(erisk[t][j])
                    selector.update(preds[t],y[t])
                rows.append({'stream':p.parent.name,'method':name,'memory':H,'risk':np.mean(scores)})
    df=pd.DataFrame(rows)
    means=df.groupby(['method','memory']).risk.mean().reset_index()
    best=means.loc[means.groupby('method').risk.idxmin()]
    return {r.method:int(r.memory) for _,r in best.iterrows()},df,means

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--development',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args(); chosen,raw,means=calibrate(a.development)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'selection_memories.json').write_text(json.dumps(chosen,indent=2))
    raw.to_csv(out/'calibration_paths.csv',index=False);means.to_csv(out/'calibration_means.csv',index=False)
    print(json.dumps(chosen,indent=2))
