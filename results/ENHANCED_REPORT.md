# Enhanced adaptive aggregation benchmark

Status: **complete**. This is a fresh confirmatory campaign on seeds 2000--2019,
separate from the first 1000--1019 campaign.

The run contains 18 scenarios x 20 independent streams = 360 streams, 63 neural
memory/width configurations, 36 feasible methods plus one evaluator-only oracle,
514,800 individual neural fits and 7,200 current-risk evaluation times. The
experiment completed without failures. Global confidence intervals are paired
seed-cluster percentile bootstrap intervals with 10,000 draws; they are pointwise,
not multiplicity-adjusted.

## Main result

The best feasible method is **ShareGrid Extended Top-5**, with mean current
integrated excess risk **0.04220** and 95% interval **[0.03961, 0.04491]**.

The method uses the theory-derived neural memory bank, augments it with output
shrinkage, zero/recent-mean/RLS fallbacks, and tracks the expert pool with an
AdaHedge master over a grid of fixed-share rates. At prediction time it keeps
only the five largest current meta weights. The top-5 pruning is a deployment
heuristic; the tracking guarantee applies to the full mixture before pruning.

| Method | Mean risk | 95% interval | Ratio vs expanding NN |
|---|---:|---:|---:|
| ShareGrid Extended Top-5 | 0.04220 | [0.03961, 0.04491] | 0.446 |
| ShareGrid Extended Top-3 | 0.04279 | [0.04017, 0.04552] | 0.453 |
| AdaHedge Extended | 0.04388 | [0.04092, 0.04700] | 0.464 |
| AdaHedge Complexity Prior | 0.04423 | [0.04112, 0.04746] | 0.468 |
| ShareGrid Complexity Prior | 0.04735 | [0.04404, 0.05086] | 0.501 |
| Joint FixedShare | 0.04828 | [0.04515, 0.05161] | 0.511 |
| ShareGrid Extended | 0.04865 | [0.04530, 0.05216] | 0.515 |
| Hedge all | 0.04875 | [0.04538, 0.05240] | 0.516 |
| RLS(.99) | 0.05095 | [0.04647, 0.05591] | 0.539 |
| Exponential selection | 0.05263 | [0.04891, 0.05660] | 0.557 |
| Hard joint selection | 0.05545 | [0.05155, 0.05960] | 0.587 |

## Paired comparisons for Top-5

| Baseline | Risk ratio | 95% interval | Scenario wins |
|---|---:|---:|---:|
| Hard joint selection | 0.761 | [0.742, 0.780] | 18/18 |
| Exponential selection | 0.802 | [0.783, 0.820] | 18/18 |
| RLS(.99) | 0.828 | [0.797, 0.862] | 14/18 |
| Hedge all | 0.866 | [0.846, 0.888] | 14/18 |
| Original joint FixedShare | 0.874 | [0.851, 0.898] | 14/18 |
| AdaHedge Extended | 0.962 | [0.945, 0.980] | 8/18 |
| ShareGrid Extended Top-3 | 0.986 | [0.981, 0.991] | 16/18 |

Top-5 therefore lowers mean risk by about 19.8% relative to selected exponential
weighting, 12.6% relative to the original joint fixed-share ensemble, 13.4%
relative to ordinary Hedge, 23.9% relative to hard joint selection, and 17.2%
relative to RLS(.99). The paired risk-ratio intervals for all of these comparisons
lie below one.

## Low-signal failure is largely repaired

In the `low_signal` scenario:

- zero predictor: 0.01535;
- ShareGrid Extended Top-5: **0.02001**;
- AdaHedge Extended: 0.01835;
- RLS(.99): 0.03257;
- exponential selection: 0.08093;
- hard joint selection: 0.09366.

Adding null, shrinkage and simple experts therefore fixes most of the failure
identified in the first campaign. Zero remains better in this deliberately
weak-signal DGP.

## Robustness

For Top-5, mean risk is 0.04434 on the 15 scenarios inside the main sampling
assumptions and 0.03150 on the three deliberately misspecified stress scenarios.
The ratios to the fixed expanding NN are 0.444 and 0.465 respectively. Stress
results are descriptive and are not theoretical coverage claims.

Top-5 also improves abrupt and recurring changes: in `single_jump`, risk is
0.08033 versus 0.09268 for exponential selection; in `recurring`, 0.08378
versus 0.10625.

## Interpretation

The practical lesson is not that one universal temporal kernel dominates. The
stronger architecture is:

1. use the theory to construct a compact library of memories and neural
   complexities;
2. include simple and shrunk fallbacks so the library remains useful in
   weak-signal periods;
3. adapt online over the library rather than estimating a single drift exponent
   or changepoint rule;
4. use sparse top-k deployment to keep inference cheap.

`docs/META_BRIDGE.md` gives the bridge from fixed-share/AdaHedge tracking to an
average current-risk inequality under the manuscript assumptions. It does not
claim pointwise oracle optimality immediately after an unrestricted jump.

## Scope

The finite-step neural fits do not certify global ERM. Top-k pruning is not yet
covered by the tracking theorem. The 18 DGPs are a simulation suite, not a
population of financial markets. No financial-profitability or universal-SOTA
claim follows from these results.

Full raw results are preserved in GitHub Actions run 37526345354, artifact
`enhanced-adaptive-results`.
