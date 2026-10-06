"""Adaptive expert aggregation for nonstationary regression.

Meta algorithms use clipped, scale-normalized squared losses in [0,1].
Integrated current risk remains evaluator-only.

AdaHedge follows de Rooij et al. (2014). ShareGridAdaHedge places an AdaHedge
master over fixed-share trackers on a finite grid of learning and share rates.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def _simplex(p):
    p=np.asarray(p,dtype=float)
    if p.ndim!=1 or len(p)==0 or np.any(p<0) or not np.isfinite(p).all():
        raise ValueError("invalid probability vector")
    z=p.sum()
    if z<=0: raise ValueError("zero probability mass")
    return p/z

def clipped_losses(forecasts,y,bound):
    """Bounded convex surrogate: squared loss after clipping the observed target."""
    bound=float(bound)
    if bound<=0: raise ValueError("bound must be positive")
    f=np.clip(np.asarray(forecasts,dtype=float),-bound,bound)
    yc=float(np.clip(y,-bound,bound))
    return (f-yc)**2/(4*bound*bound)

class AdaHedge:
    """Parameter-free Hedge on bounded losses."""
    def __init__(self,size,prior=None):
        if size<1: raise ValueError("size must be positive")
        self.size=size
        self.prior=_simplex(np.ones(size) if prior is None else prior)
        self.L=np.zeros(size)
        self.delta=0.
        self.C=float(np.log(1/self.prior.min())) if size>1 else 0.
        self.rounds=0
    def eta(self):
        if self.size==1: return np.inf
        return np.inf if self.delta<=1e-15 else self.C/self.delta
    def _weights_for(self,L,eta):
        if self.size==1: return np.ones(1)
        if np.isinf(eta):
            leaders=np.isclose(L,L.min(),atol=1e-14,rtol=0)
            p=self.prior*leaders
            return p/p.sum()
        x=np.log(self.prior)-eta*(L-L.min())
        x-=x.max();p=np.exp(x);return p/p.sum()
    def vector(self):
        return self._weights_for(self.L,self.eta())
    def _mix_value(self,L,eta):
        if self.size==1:return float(L[0])
        if np.isinf(eta):return float(np.min(L))
        z=-eta*L+np.log(self.prior);m=z.max()
        return float(-(m+np.log(np.exp(z-m).sum()))/eta)
    def update_losses(self,losses):
        losses=np.asarray(losses,dtype=float)
        if losses.shape!=(self.size,) or np.any((losses<0)|(losses>1)) or not np.isfinite(losses).all():
            raise ValueError("losses must be finite in [0,1]")
        eta=self.eta();w=self.vector();hedge=float(w@losses)
        gap=max(0.,hedge-(self._mix_value(self.L+losses,eta)-self._mix_value(self.L,eta)))
        self.L+=losses;self.delta+=gap;self.rounds+=1
        return self
    def update(self,forecasts,y,bound):
        return self.update_losses(clipped_losses(forecasts,y,bound))

class ShareGridAdaHedge:
    """AdaHedge master over a finite grid of fixed-share trackers."""
    def __init__(self,size,bound=4.,eta_grid=(.25,.5,1.,2.,4.),
                 share_grid=(0.,1/512,1/128,1/32,1/8),prior=None):
        if size<1: raise ValueError("size must be positive")
        self.size=size;self.bound=float(bound)
        if self.bound<=0: raise ValueError("bound must be positive")
        self.prior=_simplex(np.ones(size) if prior is None else prior)
        grid=[(float(e),float(a)) for e in eta_grid for a in share_grid]
        if any(e<=0 or a<0 or a>=1 for e,a in grid):raise ValueError("invalid grid")
        self.eta=np.array([g[0] for g in grid]);self.share=np.array([g[1] for g in grid])
        self.P=np.repeat(self.prior[None,:],len(grid),axis=0)
        self.master=AdaHedge(len(grid));self.rounds=0
    def vector(self):
        return self.master.vector()@self.P
    def tracker_forecasts(self,forecasts):
        forecasts=np.asarray(forecasts,dtype=float)
        if forecasts.shape!=(self.size,):raise ValueError("forecast size mismatch")
        return self.P@forecasts
    def update(self,forecasts,y):
        forecasts=np.asarray(forecasts,dtype=float)
        tpred=self.tracker_forecasts(forecasts)
        self.master.update_losses(clipped_losses(tpred,y,self.bound))
        loss=clipped_losses(forecasts,y,self.bound)
        logp=np.log(np.maximum(self.P,1e-300))-self.eta[:,None]*loss[None,:]
        logp-=logp.max(axis=1,keepdims=True)
        p=np.exp(logp);p/=p.sum(axis=1,keepdims=True)
        self.P=(1-self.share[:,None])*p+self.share[:,None]*self.prior[None,:]
        self.P/=self.P.sum(axis=1,keepdims=True);self.rounds+=1
        return self
    def diagnostics(self):
        w=self.vector();mw=self.master.vector()
        return {"effective_experts":float(1/(w@w)),"top_weight":float(w.max()),
                "master_effective_trackers":float(1/(mw@mw))}

def sparse_vector(w,k):
    w=_simplex(w);k=max(1,min(int(k),len(w)))
    keep=np.argpartition(w,-k)[-k:]
    z=np.zeros_like(w);z[keep]=w[keep];z/=z.sum();return z

@dataclass
class ExtendedExpertPool:
    specs:list
    shrinkages:tuple=(.25,.5,.75,1.)
    simple_names:tuple=("zero","recent_mean","rls_099","rls_0995")
    def __post_init__(self):
        self.shrinkages=tuple(float(v) for v in self.shrinkages)
        if any(v<=0 or v>1 for v in self.shrinkages):raise ValueError("invalid shrinkage")
        self.K=len(self.specs);self.size=self.K*len(self.shrinkages)+len(self.simple_names)
        self.zero_index=self.K*len(self.shrinkages)
    def vector(self,nn_forecasts,simple_forecasts):
        p=np.asarray(nn_forecasts,dtype=float);q=np.asarray(simple_forecasts,dtype=float)
        if p.shape!=(self.K,) or q.shape!=(len(self.simple_names),):raise ValueError("pool shape")
        return np.concatenate([a*p for a in self.shrinkages]+[q])
    def matrix(self,nn_forecasts,simple_forecasts):
        p=np.asarray(nn_forecasts,dtype=float);q=np.asarray(simple_forecasts,dtype=float)
        if p.ndim!=2 or p.shape[1]!=self.K or q.shape!=(p.shape[0],len(self.simple_names)):
            raise ValueError("pool matrix shape")
        return np.column_stack([a*p for a in self.shrinkages]+[q])
    def complexity_prior(self):
        raw=[]
        for _a in self.shrinkages:
            for s in self.specs:raw.append(1/(1+float(s.width)))
        raw += [1.,.5,.5,.5]
        return _simplex(raw)
    def diagnostics(self,w):
        w=_simplex(w);neural=w[:self.K*len(self.shrinkages)]
        return {"meta_effective_experts":float(1/(w@w)),
                "meta_zero_mass":float(w[self.zero_index]),
                "meta_neural_mass":float(neural.sum()),"meta_top_weight":float(w.max())}
