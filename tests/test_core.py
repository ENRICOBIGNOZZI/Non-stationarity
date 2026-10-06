import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import minimize
from nonstationarity.kernels import weights, diagnostics, power_oracle, candidate_grid
from nonstationarity.baselines import trust_region, linear_discrepancy, MazzettoUpfalLinear, RLS
from nonstationarity.dgp import Generator, SCENARIOS, tilt_uniform
from nonstationarity.neural import NeuralBank, project_l1_rows
from nonstationarity.selectors import Prequential, FixedShare
from nonstationarity.meta import AdaHedge, ShareGridAdaHedge, ExtendedExpertPool, sparse_vector

@pytest.mark.parametrize("kind", ["uniform","power","capped","exponential","expanding"])
@pytest.mark.parametrize("n", [1,7,100])
def test_weights(kind,n):
    w=weights(n,kind,16,.5,.5)
    assert np.all(w>=0) and np.isclose(w.sum(),1)
    assert np.all(np.diff(w)>=-1e-14)
    d=diagnostics(w)
    assert 1-1e-12 <= d['neff'] <= n+1e-12
    if kind in ('uniform','power','capped') and n>16:
        assert np.all(w[:-16]==0)

def test_uniform_ess():
    assert diagnostics(weights(100,'uniform',16))['neff']==16
    assert np.isclose(diagnostics(weights(1,'power',1))['neff'],1)

@pytest.mark.parametrize("seed", range(12))
def test_oracle_against_qp(seed):
    rng=np.random.default_rng(seed)
    n=int(rng.integers(2,20))
    a=np.sort(rng.uniform(.01,1,n))
    lam=10**rng.uniform(-2,1); D=10**rng.uniform(-2,2)
    w=power_oracle(a,lam,D)
    obj=lambda z: lam*(z@z)+D*(z@a)**2
    grad=lambda z: 2*lam*z+2*D*(z@a)*a
    sol=minimize(obj,np.full(n,1/n),jac=grad,bounds=[(0,1)]*n,
                 constraints={'type':'eq','fun':lambda z:z.sum()-1,'jac':lambda z:np.ones(n)},
                 method='SLSQP',options={'ftol':1e-12,'maxiter':200})
    assert sol.success
    assert abs(obj(w)-obj(sol.x)) < 1e-8

@pytest.mark.parametrize("A,b",[(np.diag([-2.,1.]),np.array([0.,.1])),
                               (np.diag([0.,2.]),np.array([0.,.1])),
                               (np.diag([1.,2.]),np.array([.1,.1])),
                               (np.array([[.3,1.],[1.,-.2]]),np.array([1.,-.2]))])
def test_trust_region(A,b):
    x,value=trust_region(A,b)
    assert np.linalg.norm(x)<=1+1e-8
    angle=np.linspace(0,2*np.pi,20001)
    z=np.column_stack([np.cos(angle),np.sin(angle)])
    grid=np.einsum('ni,ij,nj->n',z,A,z)+2*z@b
    assert value <= grid.min()+1e-7
    assert value <= 1e-10

def test_discrepancy_against_dense_grid():
    rng=np.random.default_rng(2)
    x=rng.normal(size=(40,2))/4; y=rng.uniform(-1,1,40)
    result=linear_discrepancy(x[:20],y[:20],x[20:],y[20:])
    angles=np.linspace(0,2*np.pi,2000)
    beta=np.vstack([r*np.column_stack([np.cos(angles),np.sin(angles)]) for r in np.linspace(0,1,50)])
    loss=(x@beta.T-y[:,None])**2
    brute=np.max(np.abs(loss[:20].mean(0)-loss[20:].mean(0)))/2
    assert result >= brute-1e-9 and result-brute < 3e-4

def test_bounded_output_and_representation_learning():
    torch.set_num_threads(1)
    rng=np.random.default_rng(13)
    x=rng.uniform(-1,1,(80,4)).astype('float32'); y=np.sin(x[:,0])*2
    specs=candidate_grid([32],[4,12])
    bank=NeuralBank(specs,4,seed=2,steps=30)
    bank.fit(x,y)
    probe=rng.normal(size=(100,4)).astype('float32')*100
    assert np.max(np.abs(bank.predict(probe)))<=4.00001
    assert np.isfinite(bank.train_loss).all()
    init=np.random.default_rng(np.random.SeedSequence([2,2001])).normal(0,1.3/2,(4,12))
    assert not np.allclose(bank.v.detach().numpy()[0],init)
    assert np.max(np.abs(bank.out.detach().numpy()).sum(1))<=4.00001

def test_l1_projection():
    x=torch.tensor([[4.,-3.,2.],[.1,.2,.3]])
    y=project_l1_rows(x,2.)
    assert torch.all(y.abs().sum(1)<=2.000001)
    assert torch.allclose(y[1],x[1])

def test_rls_matches_weighted_least_squares():
    rng=np.random.default_rng(9); X=rng.normal(size=(80,3)); y=rng.normal(size=80)
    model=RLS(3,.99,ridge=.01)
    for x,yy in zip(X,y): model.update(x,yy)
    Z=np.column_stack([np.ones(80),X]); w=.99**np.arange(79,-1,-1)
    beta=np.linalg.solve(Z.T@(w[:,None]*Z)+.01*.99**80*np.eye(4),Z.T@(w*y))
    assert np.allclose(model.coef,beta,atol=1e-9)

@pytest.mark.parametrize("scenario",SCENARIOS)
def test_dgp_reproducibility(scenario):
    a=Generator(scenario,192,4,2).sample(); b=Generator(scenario,192,4,2).sample()
    assert np.array_equal(a.X,b.X) and np.array_equal(a.y,b.y)
    xp,f=a.generator.probes(120,64)
    xo,fo=a.generator.probes(120,64,1)
    assert np.isfinite(f).all() and np.max(np.abs(f))<=4
    assert not np.array_equal(xp,xo)
    assert (np.abs(a.X)<=1.000001).all()

def test_tilt_density_inverse():
    u=np.linspace(0,1,1001)
    for rho in [-.6,0,.6]:
        x=tilt_uniform(u,rho)
        assert np.allclose((x+1)/2+rho*(x*x-1)/4,u,atol=1e-12)

def test_prequential_ordering():
    specs=candidate_grid([32],[4,12])
    sel=Prequential(specs,16)
    before=sel.choose(np.arange(len(specs)))
    # Merely constructing a future prediction cannot update scores.
    pred=np.arange(len(specs),dtype=float)
    assert sel.choose(np.arange(len(specs)))==before
    sel.update(pred,0.)
    assert sel.choose(np.arange(len(specs)))==0

def test_fixed_share_simplex():
    m=FixedShare(10,active=[1,3,7])
    for t in range(100):
        m.update(np.arange(10),float(t%3))
        v=m.vector()
        assert np.isclose(v.sum(),1) and (v>=0).all()
        assert np.all(v[[0,2,4,5,6,8,9]]==0)

def test_future_perturbation_does_not_change_prefix(tmp_path):
    from nonstationarity.experiment import run_stream
    conf={'T':160,'d':4,'start':64,'burnin':96,'refit_stride':32,'steps':8,
          'probes':0,'river':False,'random_forest':False,
          'histories':[32,96],'widths':[4,12]}
    a=Generator('single_jump',160,4,15).sample()
    b=copy.deepcopy(a); b.X[120:]*=-1; b.y[120:]+=100
    run_stream(conf,'single_jump',15,str(tmp_path/'a'),supplied=a)
    run_stream(conf,'single_jump',15,str(tmp_path/'b'),supplied=b)
    p=pd.read_csv(tmp_path/'a'/'single_jump__15'/'prequential.csv')
    q=pd.read_csv(tmp_path/'b'/'single_jump__15'/'prequential.csv')
    pd.testing.assert_frame_equal(p[p.t<120],q[q.t<120])


def test_adahedge_concentrates_on_persistent_winner():
    a=AdaHedge(3)
    for _ in range(200):
        a.update_losses(np.array([0.,.5,1.]))
    assert a.vector()[0] > .95
    assert np.isclose(a.vector().sum(),1)

def test_share_grid_and_sparse_simplex():
    s=ShareGridAdaHedge(5,scale=1.)
    for _ in range(100):
        s.update(np.array([0.,.3,.6,.9,1.2]),0.)
    w=s.vector()
    assert np.isclose(w.sum(),1) and (w>=0).all()
    z=sparse_vector(w,3)
    assert np.isclose(z.sum(),1) and np.count_nonzero(z)<=3

def test_extended_pool_contains_exact_zero_and_shrinkage():
    specs=candidate_grid([32],[4])
    pool=ExtendedExpertPool(specs)
    p=np.arange(len(specs),dtype=float)+1
    q=pool.vector(p,np.array([0.,2.,3.,4.]))
    assert q[pool.zero_index]==0.
    assert np.allclose(q[:len(specs)],.25*p)
    assert pool.size==4*len(specs)+4

def test_online_adaptive_aggregate_protocol():
    from nonstationarity.online import AdaptiveMemoryRegressor
    rng=np.random.default_rng(44)
    model=AdaptiveMemoryRegressor(2,mode='adaptive_aggregate',warmup=8,refit_stride=4,
                                  histories=(4,8),widths=(2,4),steps=3)
    for _ in range(20):
        x=rng.normal(size=2).astype('float32')
        pred=model.predict_one(x)
        assert np.isfinite(pred)
        model.observe(float(rng.normal()))
