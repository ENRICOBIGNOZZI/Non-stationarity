import numpy as np
import pytest
from nonstationarity.online import AdaptiveMemoryRegressor

@pytest.mark.parametrize('mode',['select','one_se','aggregate'])
def test_causal_stream_api(mode):
    model=AdaptiveMemoryRegressor(2,mode=mode,histories=(4,8),widths=(4,12),
                                  warmup=8,refit_stride=4,steps=3,seed=1)
    rng=np.random.default_rng(3)
    for t in range(24):
        x=rng.uniform(-1,1,2)
        p=model.predict_one(x)
        assert np.isfinite(p) and abs(p)<=4.00001
        model.observe(float(x[0]+rng.normal()*.1))
    assert model.bank.nfits>0
    assert model.selected_params_['fit_observations']<=23

def test_no_observe_before_forecast():
    model=AdaptiveMemoryRegressor(2)
    with pytest.raises(RuntimeError):model.observe(1.)

def test_no_double_forecast_without_response():
    model=AdaptiveMemoryRegressor(2)
    model.predict_one(np.zeros(2))
    with pytest.raises(RuntimeError):model.predict_one(np.zeros(2))
