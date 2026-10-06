"""Paired analysis with complete streams, not time points, as replication units.

Seeds are shared across scenarios (common random numbers). Global intervals
therefore resample an entire seed across all scenarios together. No method is
removed because of poor performance. Oracle_grid is labelled infeasible.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .dgp import STRESS


def bootstrap_means(x, draws):
    return x[draws].mean(axis=1)


def analyze(root, out, n_bootstrap=10000):
    root, out = Path(root), Path(out)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest['failures']:
        raise ValueError('Refusing headline analysis of an incomplete campaign')
    config = manifest['config']
    files = sorted(root.glob('*/summary.csv'))
    if len(files) != len(config['scenarios'])*len(config['seeds']):
        raise ValueError('Unexpected number of replication files')
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    if not np.isfinite(df.mean_risk).all():
        raise ValueError('Nonfinite prediction risks')
    keys = ['scenario','seed','method']
    if df.duplicated(keys).any():
        raise ValueError('Duplicate replication')
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/'stream_summary.csv', index=False)
    rng = np.random.default_rng(602061)
    seeds = sorted(config['seeds'])
    draws = rng.integers(0, len(seeds), (n_bootstrap, len(seeds)))
    scenarios, methods = config['scenarios'], sorted(df.method.unique())
    array = np.stack([df[df.scenario==s].pivot(index='seed', columns='method', values='mean_risk')
                      .reindex(index=seeds,columns=methods).to_numpy() for s in scenarios])
    if np.isnan(array).any(): raise ValueError('Missing method/seed/scenario cell')
    idx = {m:i for i,m in enumerate(methods)}
    by_scenario = []
    for k,s in enumerate(scenarios):
        boots = bootstrap_means(array[k],draws)
        for j,m in enumerate(methods):
            by_scenario.append({'scenario':s,'method':m,'mean_risk':array[k,:,j].mean(),
                'ci_low':np.quantile(boots[:,j],.025),'ci_high':np.quantile(boots[:,j],.975),
                'seeds':len(seeds),'infeasible':m=='Oracle_grid','stress':s in STRESS})
    pd.DataFrame(by_scenario).to_csv(out/'scenario_results.csv',index=False)
    overall=[];paired=[]
    targets=['Joint_CV','Joint_1SE','Joint_FixedShare','Power_CV_matched','Joint_Lepski','Joint_Hybrid']
    groups={'all':np.arange(len(scenarios)),
            'within_sampling_assumptions':np.array([i for i,s in enumerate(scenarios) if s not in STRESS]),
            'stress':np.array([i for i,s in enumerate(scenarios) if s in STRESS])}
    for group, rows in groups.items():
        if len(rows) == 0: continue
        # Averaging scenarios FIRST keeps each bootstrap unit an independent seed.
        a = array[rows].mean(axis=0)
        boots=bootstrap_means(a,draws)
        expid=idx['NN_expanding']
        # This metric equally weights log ratios over the scenario/seed cells.
        logratio=np.log(array[rows]/array[rows,:,[expid]].reshape(len(rows),len(seeds),1)).mean(axis=0)
        for j,m in enumerate(methods):
            ratio=boots[:,j]/boots[:,expid]
            overall.append({'group':group,'method':m,'mean_risk':a[:,j].mean(),
                'ci_low':np.quantile(boots[:,j],.025),'ci_high':np.quantile(boots[:,j],.975),
                'ratio_vs_expanding':a[:,j].mean()/a[:,expid].mean(),
                'ratio_ci_low':np.quantile(ratio,.025),'ratio_ci_high':np.quantile(ratio,.975),
                'geometric_ratio_vs_expanding':np.exp(logratio[:,j].mean()),
                'scenarios':len(rows),'seed_clusters':len(seeds),'infeasible':m=='Oracle_grid'})
        for target in targets:
            for baseline in methods:
                if target==baseline: continue
                j,b=idx[target],idx[baseline]
                diff=boots[:,j]-boots[:,b]
                ratio=boots[:,j]/boots[:,b]
                per_scenario_ratio=array[rows,:,j].mean(1)/array[rows,:,b].mean(1)
                paired.append({'group':group,'target':target,'baseline':baseline,
                    'difference':a[:,j].mean()-a[:,b].mean(),
                    'difference_ci_low':np.quantile(diff,.025),'difference_ci_high':np.quantile(diff,.975),
                    'risk_ratio':a[:,j].mean()/a[:,b].mean(),
                    'ratio_ci_low':np.quantile(ratio,.025),'ratio_ci_high':np.quantile(ratio,.975),
                    'scenario_wins':int((per_scenario_ratio<1).sum()),'scenario_count':len(rows)})
    overall=pd.DataFrame(overall).sort_values(['group','mean_risk'])
    paired=pd.DataFrame(paired)
    overall.to_csv(out/'overall_results.csv',index=False)
    paired.to_csv(out/'paired_comparisons.csv',index=False)
    # Width and effective-memory diagnostics are descriptive, not drift estimators.
    trajectory=pd.concat([pd.read_csv(p) for p in sorted(root.glob('*/trajectory.csv'))],ignore_index=True)
    trajectory.groupby(['scenario','t','method'])[['excess_risk','width','neff','support']].mean().reset_index().to_csv(out/'time_profiles.csv',index=False)
    df.groupby(['scenario','method'])[['post_break_risk','prequential_mse','mean_width','mean_neff','switches']].mean().reset_index().to_csv(out/'diagnostics.csv',index=False)
    alarms=[]
    metadata=[]
    for p in sorted(root.glob('*/metadata.json')):
        m=json.loads(p.read_text());metadata.append(m)
        for detector,times in m['detector_alarms'].items():
            # All alarms in stationary/covariate-only/noise-only settings count;
            # not a calibrated false-positive probability.
            alarms.append({'scenario':m['scenario'],'seed':m['seed'],'detector':detector,
                           'alarms':len(times),'times':json.dumps(times)})
    pd.DataFrame(alarms).to_csv(out/'detector_alarms.csv',index=False)
    counts={'status':'complete','scenarios':len(scenarios),'replications_per_scenario':len(seeds),
        'neural_streams':len(metadata),'methods_including_oracle':len(methods),'feasible_methods':len(methods)-1,
        'candidate_configs':metadata[0]['candidate_count'],'neural_fits':sum(m['neural_fits'] for m in metadata),
        'stream_time_evaluations':sum(len(m['evaluation_times']) for m in metadata),
        'risk_cells':len(trajectory),'observations_per_stream':config['T'],
        'wall_seconds':manifest['wall_seconds'],
        'sum_stream_seconds':sum(m['seconds'] for m in metadata),
        'component_seconds':{k:sum(m['component_seconds'][k] for m in metadata) for k in metadata[0]['component_seconds']},
        'bootstrap_draws':n_bootstrap,'bootstrap_unit':'seed cluster across all scenarios',
        'confidence_intervals':'pointwise percentile bootstrap; not simultaneous, no multiplicity correction',
        'source_sha256':manifest['source_sha256'],'config':config}
    (out/'counts.json').write_text(json.dumps(counts,indent=2))
    return counts


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--bootstrap',type=int,default=10000)
    a=ap.parse_args();print(json.dumps(analyze(a.input,a.out,a.bootstrap),indent=2))
