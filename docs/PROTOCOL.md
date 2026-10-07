# Frozen simulation protocol

## Scope and primary estimand

At zero-based time `t`, only rows `0,...,t-1` are training data. The primary
estimand is the random integrated current risk
`E_{X~P_t}[(f_hat_t(X)-f_t(X))^2]`, conditional on the fitted predictor. The
realized squared error on one future response is a separate metric.

All candidate predictions are issued before reading `y[t]`. Selectors then see
that response to update their historical score. Training losses under different
weight vectors are never compared as a validation criterion. The true target,
changepoints and probe risks are not passed to feasible selectors.

## Main campaign

* 18 scenarios, 20 seeds (1000--1019), T=768, d=4.
* Training starts at t=64. Headline realized-loss averaging starts at t=128.
* Refit all batch methods every 32 observations. There are 22 refits per stream.
* Integrated-risk evaluation at 20 anchors t=128,160,...,736, using 256 fresh
  covariates from the *current* P_t. All methods share those evaluation covariates.
* The infeasible grid oracle uses 512 **separate** current-target probes to choose
  among the 63 trained candidates, then is scored on the common 256 probes.
  It is a finite-probe diagnostic, not the exact population oracle.
* 60 full-batch Adam steps per neural expert/refit; learning rate .03; output
  variation including the intercept constrained to an L1 ball of radius 4;
  hidden parameters restricted to [-6,6]. All hidden parameters are learned.
* Fresh fits, identical initialization across kernels at each width. No stale
  optimizer state or old fitted representation survives a hard cutoff.

There are 63 fixed candidate configurations plus two detector-driven NN wrappers
at each fit. A complete main campaign therefore performs 360 x 22 x 65 = 514,800
individual neural fits. These are **not** 514,800 independent experiments.

## Development/test separation

Validation score horizons {32,64,128,256} were selected on 12 separate development
streams (four scenarios, seeds 11--13). Frozen choices are in `configs/full.json`.
No post-test retuning or selective deletion of scenarios is permitted. Every
method receives the same data stream and eligible past. River settings, NN
optimization budget, loss normalization and heuristic thresholds are fixed.

MU_linear_tuned is a legacy code label for the **preset relaxed-threshold**
variant (multiplier .01). It is not a claim that its threshold was calibrated
on these development streams. Its confidence guarantee is not inherited.

## Scenarios

| Scenario | Target / design / noise construction |
|---|---|
| stationary_linear | fixed linear regression |
| stationary_nonlinear | fixed bounded 12-neuron tanh teacher |
| slow_smooth | slow interpolation between two teachers |
| fast_smooth | periodic teacher mixture and mean shift |
| holder_half | bounded dyadic temporal series with gamma=.5 |
| single_jump | sign reversal plus intercept jump at T/2 |
| multiple_jumps | four successive teacher blocks |
| piecewise_holder | three blocks with temporal gamma 1,.5,.25 |
| recurring | teacher sequence 0,1,0,2 |
| complexity_change | linear target followed by nonlinear interactions |
| covariate_only | changing P_t, unchanged regression function |
| joint_shift | changing design density and regression function |
| volatility_only | noise scale changes by a factor of three; target fixed |
| heteroskedastic | covariate-dependent noise scale and target drift |
| low_signal | weakened signal and higher noise |
| heavy_tails | Student-t(3) noise; Bernstein-tail stress test |
| serial_noise | AR(1) errors; independence stress test, without thinning |
| support_shift | disjoint covariate supports; overlap stress test |

The overlap-respecting changing density is (1 + rho_t*x_0)/2 on [-1,1],
with |rho_t|<=.6; the other coordinates stay uniform. The density ratio between
any two times is bounded by 4. For the dyadic finite series, gamma labels the
construction; it is not claimed to be the path's uniquely identifiable exponent.
The targets are specified in source code, not fitted using evaluation data.

## Statistical comparisons

Average integration risk across anchors **within each stream first**. Seeds are
independent within a scenario. The same seed shares underlying draws across
scenarios, so global confidence intervals resample whole seed clusters across all
18 scenarios together. Scenarios are equally weighted. Report paired differences
and ratios, plus geometric risk ratios and all per-scenario results.

Bootstrap intervals use 10,000 draws and are **pointwise**, not simultaneous and
not adjusted for the many comparisons. They describe this finite DGP collection,
not a population of financial markets. Monte Carlo integration error is present.
No individual time point is counted as an independent replication.

Post-break risk averages available evaluation anchors in the first 64 periods
after each known target jump or the separate support-shift stress event (truth
is used by the evaluator only). A zero-width
interval directly at an unknown jump cannot supply observations from the new
regime. Detector alarms in noise-only/covariate-only scenarios are recorded; their
frequency is descriptive, not a calibrated hypothesis-test error probability.

## Resource accounting and limitations

The batched expert bank is shared by several selectors. Component wall times do
not measure standalone latency of each selector or a matched-compute competition.
A fixed single model is cheaper than joint search. Fixed-share and Hedge require
multiple predictions at deployment, whereas hard selection uses one model.

This is a substantial finite-sample benchmark, not a publication-complete sweep:
only one main dimension and horizon, three widths, and finite optimization steps.
No financial data or transaction-cost model is used. The scalar oracle experiment
is separately labelled and does not establish neural minimax rates empirically.
