# Completed simulation benchmark

Main campaign: **360/360 streams completed, no failures**. There are 18 scenarios,
20 replications per scenario, 63 neural configurations, 28 feasible methods and
one infeasible grid oracle. The campaign made **514,800 individual neural fits**;
these are not independent experiments. Main wall time: 25.4 minutes on four workers.

## Main results

Current integrated excess risk is averaged across 20 evaluation anchors per stream.
Each anchor has 256 independent current-distribution probes. The oracle selects on
a separate 512 probes. Global intervals resample 20 seed clusters across all 18
scenarios together; they are pointwise, not simultaneous.

| Method | Mean risk | 95% interval | Ratio to expanding NN |
|---|---:|---:|---:|
| Infeasible grid oracle | 0.03344 | [0.03085, 0.03608] | 0.304 |
| Joint fixed-share | 0.05310 | [0.04884, 0.05748] | 0.483 |
| Hedge all | 0.05364 | [0.04923, 0.05805] | 0.488 |
| Uniform FixedShare | 0.05426 | [0.04965, 0.05892] | 0.493 |
| Exponential selection | 0.05780 | [0.05319, 0.06244] | 0.526 |
| Joint selection, 1-SE | 0.05851 | [0.05398, 0.06294] | 0.532 |
| Power selection (matched) | 0.05946 | [0.05464, 0.06432] | 0.541 |
| Joint selection | 0.05967 | [0.05491, 0.06455] | 0.543 |
| RLS 099 | 0.05981 | [0.05280, 0.06716] | 0.544 |
| Joint hybrid heuristic | 0.06048 | [0.05562, 0.06544] | 0.550 |
| Uniform selection | 0.06611 | [0.06073, 0.07165] | 0.601 |
| ARF | 0.06924 | [0.06159, 0.07693] | 0.630 |
| RLS 0995 | 0.07288 | [0.06392, 0.08207] | 0.663 |
| Memory only CV | 0.07348 | [0.06805, 0.07898] | 0.668 |
| HAT | 0.07814 | [0.06921, 0.08725] | 0.710 |
| NN exponential | 0.08508 | [0.07631, 0.09404] | 0.774 |
| PageHinkley NN | 0.09250 | [0.08389, 0.10131] | 0.841 |
| MU linear, relaxed threshold | 0.09279 | [0.08136, 0.10456] | 0.844 |
| MU linear, conservative | 0.09528 | [0.08350, 0.10728] | 0.866 |
| ADWIN NN | 0.09624 | [0.08744, 0.10528] | 0.875 |
| NN rolling long | 0.10326 | [0.09123, 0.11581] | 0.939 |
| Width only CV | 0.10593 | [0.09401, 0.11789] | 0.963 |
| RF rolling | 0.10604 | [0.09289, 0.11967] | 0.964 |
| Joint comparison heuristic | 0.10758 | [0.09566, 0.11941] | 0.978 |
| NN expanding | 0.10998 | [0.09817, 0.12177] | 1.000 |
| RF expanding | 0.11758 | [0.10463, 0.13054] | 1.069 |
| NN rolling short | 0.13858 | [0.13260, 0.14458] | 1.260 |
| Mean recent | 0.16984 | [0.14472, 0.19906] | 1.544 |
| Zero | 0.19233 | [0.16596, 0.22293] | 1.749 |

## What the comparisons establish

* Hard joint selection reduces risk by 45.7% versus fixed expanding-NN memory,
  and by 9.7% versus the development-calibrated uniform selector. It is **3.2%
  worse than selected exponential weighting**. Joint and RLS(.99) are close.
* Joint fixed-share has the lowest observed feasible mean, but the paired interval
  does not clearly separate it from ordinary Hedge. It combines multiple models;
  it is not a single NN with a selected width. Its advantage over uniform
  fixed-share is about 2.1%, not a change of rate or a novel aggregation theorem.
* The unproved two-axis comparison heuristic performs poorly (0.10758); adding
  its filter does not improve ordinary prequential selection.
* In low signal, zero prediction has risk 0.01739, versus 0.09773 for Joint_CV and
  0.06099 for the neural candidate oracle. The bank needs a null/shrinkage option.
* Noise-only shifts produce detector alarms despite a fixed regression function.
  Blindly treating all loss-distribution change as target drift is not justified.

## Further experiments

A 20-stream paired frozen-feature ablation (four scenarios, five seeds) performs
27,720 additional fits. Learned features help the explicit complexity-change
case but hurt the fast-drift case under this budget. All paired results are kept.

A fixed-validation-horizon replay reports 32, 64, 128, 256 for four candidate
families: 5,760 method-stream replays, **not new datasets**. At the common horizon
256, matched power risk is 0.05946, uniform 0.06250, exponential 0.05780. No best
test horizon was used to revise the frozen main experiment.

The separate scalar mean-estimation experiment has 54 cases x 200 replications =
**10,800 scalar streams**, across 8 methods. Exact power-oracle expectations are
compared with Monte Carlo estimates. This is not a neural-network rate experiment.

## Files and reproducibility

Source and protocol: `src/nonstationarity`, `configs`, and `docs`. Compact results
are in `headline_results.csv`, `paired_headline.csv`, `scenario_headline.csv`,
`horizon_sensitivity.csv` and `counts.json`. Full raw forecasts, candidate ledgers,
all summary tables, integration-risk paths, development data, figures and the
12-page PDF report are in the downloadable research-session bundle. Reproduce
with the commands in the main README or the manually triggered benchmark workflow.

The final local suite passes 63 tests. The initial GitHub CI passes unit tests
and a chronological smoke run. No main-campaign core source changed after launch.
This is not a proof of adaptive oracle attainment, certified global optimization,
a financial backtest or a universal performance ranking. See METHODS and PROTOCOL
for approximations, baseline adaptations, coarse grids and unequal compute costs.
