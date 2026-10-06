"""Practical predict-then-observe API, independent of the simulation generator.

This class never receives latent targets, breakpoints, or risk-integration probes.
It trains continuous neural features; it does not certify a global ERM optimum.
"""
import numpy as np
import torch
from .kernels import candidate_grid
from .neural import NeuralBank
from .selectors import Prequential, FixedShare
from .meta import ShareGridAdaHedge, ExtendedExpertPool, sparse_vector


class AdaptiveMemoryRegressor:
    """Causal joint-memory/width selector or neural fixed-share ensemble.

    Call predict_one(x), then observe(y), exactly once per observation. The first
    warmup predictions use the clipped past mean. Subsequent experts are refitted
    from scratch every refit_stride observations. All past rows are retained;
    this reference implementation does not promise constant memory or latency.
    """
    def __init__(self, d, *, mode='select', seed=0, histories=(32,96,256,768),
                 widths=(4,12,24), warmup=64, refit_stride=32, score_horizon=256,
                 steps=60, lr=.03, bound=4.):
        if d<1 or mode not in ('select','one_se','aggregate','adaptive_aggregate','sparse3'):
            raise ValueError('invalid dimension or mode')
        if warmup<2 or refit_stride<1 or score_horizon<1 or steps<1:
            raise ValueError('invalid positive tuning parameters')
        torch.set_num_threads(1)
        self.d,self.mode,self.warmup,self.refit_stride=d,mode,warmup,refit_stride
        self.bound=bound
        self.specs=candidate_grid(histories,widths)
        self.active=np.array([i for i,s in enumerate(self.specs)
                              if s.kind in ('power','capped','expanding')])
        self.bank=NeuralBank(self.specs,d,seed,steps=steps,lr=lr,bound=bound)
        self.selector=Prequential(self.specs,score_horizon,1. if mode=='one_se' else 0.)
        self.aggregate=None
        self.ext_pool=ExtendedExpertPool(self.specs)
        self.adaptive=None
        self.X=[];self.y=[];self.pending=None
        self.last_fit=None;self.selected_params_=None

    def predict_one(self,x):
        if self.pending is not None:
            raise RuntimeError('observe the pending response before forecasting again')
        x=np.asarray(x,dtype=np.float32)
        if x.shape!=(self.d,) or not np.isfinite(x).all():
            raise ValueError('x must be a finite feature vector')
        n=len(self.y)
        if n<self.warmup:
            prediction=float(np.clip(np.mean(self.y),-self.bound,self.bound)) if n else 0.
            candidates=None
            self.selected_params_={'kind':'warmup_mean','observations':n}
        else:
            if self.aggregate is None:
                scale=4*max(.05,float(np.var(self.y[:self.warmup])))
                self.aggregate=FixedShare(len(self.specs),eta=1.,share=.01,scale=scale,active=self.active)
            if self.adaptive is None:
                scale=4*max(.05,float(np.var(self.y[:self.warmup])))
                self.adaptive=ShareGridAdaHedge(self.ext_pool.size,scale)
            if self.last_fit is None or (n-self.warmup)%self.refit_stride==0:
                self.bank.fit(np.asarray(self.X),np.asarray(self.y))
                self.last_fit=n
            candidates=self.bank.predict(x[None,:])[0]
            if self.mode in ('adaptive_aggregate','sparse3'):
                recent=float(np.clip(np.mean(self.y[-64:]),-self.bound,self.bound))
                ext=self.ext_pool.vector(candidates,np.array([0.,recent,0.,0.]))
                w=self.adaptive.vector()
                if self.mode=='sparse3': w=sparse_vector(w,3)
                prediction=float(w@ext)
                self.selected_params_={'kind':self.mode,**self.ext_pool.diagnostics(w),
                                       'fit_observations':self.last_fit}
            elif self.mode=='aggregate':
                prediction=float(self.aggregate.vector()@candidates)
                self.selected_params_={'kind':'ensemble','members':len(self.active),'fit_observations':self.last_fit}
            else:
                j=self.selector.choose(self.active)
                prediction=float(candidates[j])
                self.selected_params_={**self.specs[j].as_dict(),**self.bank.diag[j],
                                       'fit_observations':self.last_fit,'candidate_index':j}
        ext_pending=None
        if candidates is not None and self.mode in ('adaptive_aggregate','sparse3'):
            recent=float(np.clip(np.mean(self.y[-64:]),-self.bound,self.bound))
            ext_pending=self.ext_pool.vector(candidates,np.array([0.,recent,0.,0.]))
        self.pending=(x.copy(),candidates,ext_pending)
        return prediction

    def observe(self,y):
        if self.pending is None:
            raise RuntimeError('predict_one must precede observe')
        y=float(y)
        if not np.isfinite(y):raise ValueError('response must be finite')
        x,candidates,ext_pending=self.pending
        if candidates is not None:
            self.selector.update(candidates,y)
            self.aggregate.update(candidates,y)
            if ext_pending is not None: self.adaptive.update(ext_pending,y)
        self.X.append(x);self.y.append(y);self.pending=None
        return self
