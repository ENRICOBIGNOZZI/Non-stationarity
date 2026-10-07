# Joint memory and neural complexity under nonstationarity

A reproducible, chronological simulation benchmark accompanying the research on
nonstationary weighted neural regression. It evaluates **joint selection of
training memory, temporal kernel and neural width**. It does not assume the
unknown drift exponent, drift amplitude, or changepoints are observed.

## What is implemented

The second-stage practical algorithm adds parameter-free AdaHedge and an
AdaHedge-over-fixed-share-grid master. The richer pool contains output-shrunk
copies of every neural memory together with zero, recent-mean and recursive-linear
fallbacks. The master can therefore use exponential weighting when it is best,
shrink toward zero in weak-signal periods, and track changing experts without
estimating breakpoints or a single fixed switching rate. Full, top-3 and top-5
deployment variants are benchmarked.

The meta guarantee concerns a bounded convex online surrogate; it is not relabelled
as an instantaneous L2(P_t) oracle theorem. See docs/ADAPTIVE_AGGREGATION.md, docs/META_BRIDGE.md, and docs/ADAPTIVE_ORACLE_THEOREM.tex.


* A bank of 63 genuinely trainable, one-hidden-layer tanh networks. Widths are
  4, 12 and 24; memory scales are 32, 96, 256 and 768 observations. Temporal
  weighting includes uniform windows, exponentials, power tapers, capped power
  tapers and expanding histories.
* Causal prequential selection; a paired one-standard-error heuristic;
  fixed-share and Hedge aggregation; experimental two-axis estimator comparisons.
* Fixed/selected rolling and exponential neural baselines; River ADWIN and
  Page-Hinkley neural wrappers; the Mazzetto--Upfal linear-window algorithm;
  RLS; batch random forests; River Adaptive Random Forest and Hoeffding Adaptive
  Tree; zero and recent-mean controls.
* Independent current-distribution integration probes. An explicitly infeasible
  candidate-grid oracle selects on a **second**, independent set of truth probes.
* Eighteen DGPs including gradual drift, jumps, changing temporal regularity,
  recurring targets, spatial complexity changes, covariate-only and noise-only
  changes. Three stress tests deliberately violate the main theorem assumptions.

The training objective is `sum_s w[t,s] * (Y[s] - f(X[s]))**2`. The main evaluation
metric is current integrated excess risk, `E_{X~P_t}(f_hat(X)-f_t(X))**2`, estimated
on independent evaluator-only probes. Realized prequential MSE is reported too.

## Reproduce

Python 3.13 and CPU PyTorch were used in the recorded run. Install the package:

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,report]'
python -m pytest -q
python -m nonstationarity.experiment --config configs/smoke.json --out results/smoke --jobs 1
```

For the main experiment, validation horizons are **already frozen** in
`configs/full.json`. Do not recalibrate them on the test results.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m nonstationarity.experiment --config configs/full.json --out results/full --jobs 4
python -m nonstationarity.analysis --input results/full --out results/analysis
python -m nonstationarity.temporal_oracle --out results/temporal_oracle
```

To reproduce the independent development stage:

```sh
python -m nonstationarity.experiment --config configs/development.json --out results/development --jobs 4
python -m nonstationarity.calibrate --development results/development --out results/calibration
```

Use a **new output directory** for code/config changes. `--resume` checks the
configuration hash, not historical source identity; it is only safe for an
unchanged code checkout. Each run writes source SHA-256 hashes, dependency
versions, configuration, per-stream metadata and failures. Final status is
`complete` only when every requested stream succeeds.

## Protocol and interpretation

Read [PROTOCOL.md](docs/PROTOCOL.md), [METHODS.md](docs/METHODS.md) and
[LITERATURE.md](docs/LITERATURE.md). First-campaign results are in [results/REPORT.md](results/REPORT.md). The fresh adaptive-aggregation campaign is in [results/ENHANCED_REPORT.md](results/ENHANCED_REPORT.md).

Important distinctions:

1. The reported bounds concern an explicit finite dictionary/bounded sieve and
   contain an optimization-error term. These simulations train continuous hidden
   parameters for a finite number of Adam steps. They do **not** certify global
   ERM, that optimization error, or a data-driven oracle inequality.
2. Kernel shape is a tuning parameter, not a consistent estimate of Holder
   regularity. The capped kernel is a member of the theoretical family, not an
   estimated solution of the unknown full-bound objective.
3. The two-axis comparisons and hybrid filter are **unproved heuristics**. They
   are not relabelled as Mazzetto--Upfal. That paper's linear algorithm is a
   separate baseline with an exact trust-region discrepancy calculation.
4. Fixed-share/Hedge predictions combine several neural experts. They are not a
   single network with a selected width, and their deployment cost is larger.
5. The scalar oracle Monte Carlo study is a separate mean-estimation experiment,
   not thousands of additional neural-training streams or a financial backtest.

No claim of state-of-the-art performance, financial profitability, or universal
optimality follows from this benchmark. Negative comparisons are retained.

## Use with an arbitrary observed stream

```python
from nonstationarity.online import AdaptiveMemoryRegressor

model = AdaptiveMemoryRegressor(d=4, mode="select", seed=7)
for x, y in observed_stream:
    prediction = model.predict_one(x)  # y has not been supplied to the model
    model.observe(y)
    diagnostics = model.selected_params_
```

Use `mode="one_se"` for the parsimony heuristic or `mode="aggregate"` for a
fixed-share mixture. The first 64 forecasts use a bounded past mean; scores and
fits are then updated causally. This is a reference implementation, not a
constant-memory production service. Retained history, CPU refits and ensemble
inference costs must be budgeted separately.
