"""Hierarchical mixable fixed-share master.

This is the theory-facing meta learner used in docs/ADAPTIVE_ORACLE_THEOREM.tex.
It exploits exp-concavity of bounded squared loss, giving logarithmic path
complexity rather than the generic sqrt(T) Hedge penalty.

The practical enhanced benchmark uses ShareGridAdaHedge in meta.py; this class is
kept separate so the theorem's assumptions and constants remain explicit.
"""
from __future__ import annotations
import math
import numpy as np


def _simplex(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or len(x)==0 or np.any(x<0) or not np.isfinite(x).all():
        raise ValueError("invalid probability vector")
    s=x.sum()
    if s<=0:
        raise ValueError("zero probability mass")
    return x/s


def dyadic_share_grid(horizon:int):
    """0 plus dyadic shares down to order 1/T."""
    if horizon<2:
        raise ValueError("horizon must be at least 2")
    jmax=max(1,math.ceil(math.log2(horizon)))
    vals=[0.]+[2.**(-j) for j in range(1,jmax+1)]
    return tuple(sorted(set(vals)))


def geometric_grid(maximum:float, epsilon:float):
    """Multiplicative (1+epsilon)-grid including 1 and maximum.

    Useful for memory/width discretisation in the continuum-oracle proposition.
    """
    maximum=float(maximum);epsilon=float(epsilon)
    if maximum<1 or not (0<epsilon<=1):
        raise ValueError("maximum >=1 and epsilon in (0,1] required")
    out=[1.]
    while out[-1] < maximum:
        nxt=min(maximum,out[-1]*(1+epsilon))
        if nxt<=out[-1]*(1+1e-14):
            break
        out.append(nxt)
    return np.array(out)


class HierarchicalMixableShare:
    """Mixture over fixed-share trackers at the square-loss mixability rate.

    Base forecasts are clipped to [-prediction_bound,prediction_bound]. The
    observed response is clipped to [-outcome_bound,outcome_bound]. Let
    D=prediction_bound+outcome_bound. The normalized loss is
    (y-f)^2/D^2 in [0,1]. Its exponential is concave in f for eta<=1/2.

    If the supplied outcome_bound is a high-probability Bernstein envelope, then
    on the no-clipping event the deterministic tracking inequality is an
    inequality for the original realized squared losses.
    """
    def __init__(self,size:int,horizon:int,prediction_bound:float,
                 outcome_bound:float,alpha_grid=None,expert_prior=None,
                 tracker_prior=None,eta:float=.5):
        if size<1 or horizon<2:
            raise ValueError("positive size and horizon>=2 required")
        if prediction_bound<=0 or outcome_bound<=0 or not (0<eta<=.5):
            raise ValueError("invalid bounds or eta")
        self.size=int(size);self.horizon=int(horizon)
        self.B=float(prediction_bound);self.M=float(outcome_bound)
        self.D=self.B+self.M;self.eta=float(eta)
        self.alpha=np.array(dyadic_share_grid(horizon) if alpha_grid is None else alpha_grid,dtype=float)
        if self.alpha.ndim!=1 or len(self.alpha)==0 or np.any((self.alpha<0)|(self.alpha>=1)):
            raise ValueError("alpha must lie in [0,1)")
        self.pi=_simplex(np.ones(size) if expert_prior is None else expert_prior)
        self.rho=_simplex(np.ones(len(self.alpha)) if tracker_prior is None else tracker_prior)
        self.P=np.repeat(self.pi[None,:],len(self.alpha),axis=0)
        self.R=self.rho.copy()
        self.rounds=0

    def expert_vector(self):
        """Current total mass on base experts."""
        return self.R@self.P

    def predict(self,forecasts):
        f=np.clip(np.asarray(forecasts,dtype=float),-self.B,self.B)
        if f.shape!=(self.size,):
            raise ValueError("forecast size mismatch")
        tracker=self.P@f
        return float(self.R@tracker)

    def _loss(self,forecast,y):
        yc=float(np.clip(y,-self.M,self.M))
        fc=np.clip(np.asarray(forecast,dtype=float),-self.B,self.B)
        return (fc-yc)**2/(self.D*self.D)

    def update(self,forecasts,y):
        f=np.clip(np.asarray(forecasts,dtype=float),-self.B,self.B)
        if f.shape!=(self.size,):
            raise ValueError("forecast size mismatch")
        tracker=self.P@f
        # Outer exponential aggregator over share rates.
        tloss=self._loss(tracker,y)
        z=np.log(np.maximum(self.R,1e-300))-self.eta*tloss
        z-=z.max();self.R=np.exp(z);self.R/=self.R.sum()
        # Each inner tracker performs exponential weighting then fixed share.
        eloss=self._loss(f,y)
        logp=np.log(np.maximum(self.P,1e-300))-self.eta*eloss[None,:]
        logp-=logp.max(axis=1,keepdims=True)
        post=np.exp(logp);post/=post.sum(axis=1,keepdims=True)
        self.P=(1-self.alpha[:,None])*post+self.alpha[:,None]*self.pi[None,:]
        self.P/=self.P.sum(axis=1,keepdims=True)
        self.rounds+=1
        return self

    def normalized_loss(self,forecast,y):
        return float(self._loss(float(forecast),y))

    def path_code_length(self,path,alpha_index:int):
        """-log prior mass of a base-expert path for one fixed-share tracker."""
        path=np.asarray(path,dtype=int)
        if path.ndim!=1 or len(path)==0 or np.any((path<0)|(path>=self.size)):
            raise ValueError("invalid path")
        j=int(alpha_index)
        if not (0<=j<len(self.alpha)):
            raise ValueError("bad alpha index")
        a=float(self.alpha[j])
        c=-math.log(self.pi[path[0]])
        for old,new in zip(path[:-1],path[1:]):
            tr=(1-a)*(1. if old==new else 0.)+a*self.pi[new]
            if tr<=0:
                return math.inf
            c-=math.log(tr)
        return float(c)

    def hierarchical_code_length(self,path):
        """Comparator penalty min_j{-log rho_j - log Pi_j(path)}."""
        vals=[-math.log(self.rho[j])+self.path_code_length(path,j)
              for j in range(len(self.alpha))]
        return float(min(vals))

    def comparator_bound(self,path):
        """Normalized cumulative-loss regret upper bound for any fixed path."""
        return self.hierarchical_code_length(path)/self.eta
