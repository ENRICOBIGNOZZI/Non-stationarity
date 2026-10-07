"""Standalone figures from recorded results. Each figure is a separate plot."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate(analysis, out, temporal=None):
    analysis, out=Path(analysis),Path(out);out.mkdir(parents=True,exist_ok=True)
    overall=pd.read_csv(analysis/'overall_results.csv')
    profiles=pd.read_csv(analysis/'time_profiles.csv')
    scenarios=pd.read_csv(analysis/'scenario_results.csv')
    names={'MU_linear_tuned':'MU linear: relaxed threshold','MU_linear_conservative':'MU linear: conservative',
           'Joint_CV':'Joint selection','Joint_1SE':'Joint selection, 1-SE','Joint_FixedShare':'Joint fixed-share',
           'Power_CV_matched':'Power selection (matched)','Uniform_CV':'Uniform selection',
           'Exponential_CV':'Exponential selection','Joint_Lepski':'Joint comparison heuristic',
           'Joint_Hybrid':'Joint hybrid heuristic','Oracle_grid':'Infeasible grid oracle'}
    def label(s): return names.get(s,s.replace('_',' '))
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
    # The full feasible comparison, not just methods that beat the proposal.
    z=overall[(overall.group=='all') & (~overall.infeasible)].sort_values('mean_risk',ascending=False)
    fig,ax=plt.subplots(figsize=(8.1,9.2))
    ax.errorbar(z.ratio_vs_expanding,np.arange(len(z)),
                xerr=np.array([z.ratio_vs_expanding-z.ratio_ci_low,z.ratio_ci_high-z.ratio_vs_expanding]),fmt='o',capsize=2)
    ax.axvline(1.,linestyle='--',linewidth=1)
    ax.set_yticks(np.arange(len(z)),[label(m) for m in z.method],fontsize=8)
    ax.set_xlabel('Mean current-risk ratio versus fixed expanding NN (lower is better)')
    ax.set_title('All 28 feasible methods; paired 95% seed-cluster intervals')
    ax.set_xscale('log');save(fig,'overall_comparison')
    # Equal-search-size kernel comparison.
    methods=['Uniform_CV','Exponential_CV','Power_CV_matched','Joint_CV','Joint_1SE']
    base=scenarios[scenarios.method=='NN_expanding'].set_index('scenario').mean_risk
    fig,ax=plt.subplots(figsize=(9.2,5.5))
    sn=list(base.index); xx=np.arange(len(sn))
    for m in methods:
        y=scenarios[scenarios.method==m].set_index('scenario').mean_risk.reindex(sn)/base.reindex(sn)
        ax.plot(xx,y,marker='o',markersize=3,label=label(m))
    ax.axhline(1.,linestyle='--',linewidth=1)
    ax.set_xticks(xx,[s.replace('_',' ') for s in sn],rotation=65,ha='right',fontsize=8)
    ax.set_ylabel('Risk / fixed expanding NN risk');ax.set_title('Performance across all scenarios');ax.legend(fontsize=8,ncol=2)
    save(fig,'scenario_comparison')
    methods=['Joint_CV','Joint_1SE','Exponential_CV','Uniform_CV','ADWIN_NN','Oracle_grid']
    for field,title,ylabel in [('excess_risk','Risk around an abrupt change','Integrated current excess risk'),
                              ('neff','Effective memory around an abrupt change','Effective sample size'),
                              ('width','Selected width around an abrupt change','Hidden neurons')]:
        fig,ax=plt.subplots(figsize=(7.5,4.2))
        for m in methods:
            z=profiles[(profiles.scenario=='single_jump')&(profiles.method==m)].sort_values('t')
            ax.plot(z.t,z[field],marker='o',markersize=3,label=label(m))
        ax.axvline(384,linestyle='--',linewidth=1,label='Target jump')
        ax.set_xlabel('Time (zero-based)');ax.set_ylabel(ylabel);ax.set_title(title)
        ax.legend(fontsize=8,ncol=2);save(fig,'single_jump_'+field)
    if temporal is not None:
        df=pd.read_csv(Path(temporal)/'summary.csv')
        z=df[(df.gamma==1.)&(df.V==1.)]
        fig,ax=plt.subplots(figsize=(7.5,4.2))
        for m in ['Oracle_power','Oracle_uniform','Oracle_exponential','MU_mean','Expanding']:
            zz=z[z.method==m].sort_values('T')
            ax.loglog(zz['T'],zz.mean_mse,marker='o',label=m.replace('_',' '))
        ax.set_xlabel('Horizon T');ax.set_ylabel('Monte Carlo mean squared error')
        ax.set_title('Separate scalar mean-estimation study: gamma=1, V=1')
        ax.legend(fontsize=8);save(fig,'scalar_oracle')
        zz=df[(df.method=='Oracle_power')]
        fig,ax=plt.subplots(figsize=(7.5,4.2))
        for g in (.25,.5,1.):
            q=zz[(zz.gamma==g)&(zz.V==1.)].sort_values('T')
            ax.loglog(q['T'],q.mean_mse,marker='o',label=f'gamma={g:g}: Monte Carlo')
            ax.loglog(q['T'],q.analytic_mse,linestyle='--',label=f'gamma={g:g}: exact expectation')
        ax.set_xlabel('Horizon T');ax.set_ylabel('Power-oracle mean squared error')
        ax.set_title('Analytic expectations versus 200-replication estimates')
        ax.legend(fontsize=8,ncol=2);save(fig,'scalar_rates')
    return sorted(str(p) for p in out.glob('*.pdf'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--out',required=True);p.add_argument('--temporal')
    a=p.parse_args();print('\n'.join(generate(a.analysis,a.out,a.temporal)))
