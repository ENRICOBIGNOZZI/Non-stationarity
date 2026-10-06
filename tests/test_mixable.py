import itertools
import numpy as np
from nonstationarity.mixable import (
    HierarchicalMixableShare, dyadic_share_grid, geometric_grid
)


def test_geometric_grid_covers_interval():
    g=geometric_grid(100,0.1)
    assert g[0]==1 and np.isclose(g[-1],100)
    for x in np.geomspace(1,100,1000):
        hi=g[g>=x][0]
        assert hi <= 1.1*x + 1e-12


def test_dyadic_share_grid_reaches_one_over_T():
    q=dyadic_share_grid(1000)
    assert q[0]==0.
    assert q[-1] == .5
    assert min(v for v in q if v>0) <= 1/1000


def test_mixable_path_bound_by_bruteforce():
    # Small deterministic sequence. The theorem is deterministic once losses
    # are bounded, so we can check every possible expert path.
    forecasts=np.array([
        [-.8,.0,.7],
        [-.6,.1,.6],
        [-.4,.2,.5],
        [.5,.1,-.4],
        [.7,.0,-.6],
        [.8,-.1,-.7],
    ])
    y=np.array([-.7,-.5,-.3,.4,.6,.75])
    m=HierarchicalMixableShare(
        size=3,horizon=len(y),prediction_bound=1.,outcome_bound=1.,
        alpha_grid=(0.,.125,.25,.5),eta=.5
    )
    alg=0.
    for f,yy in zip(forecasts,y):
        p=m.predict(f)
        alg += m.normalized_loss(p,yy)
        m.update(f,yy)
    for path in itertools.product(range(3), repeat=len(y)):
        comp=sum((y[t]-forecasts[t,path[t]])**2/4 for t in range(len(y)))
        assert alg <= comp + m.comparator_bound(path) + 1e-10
