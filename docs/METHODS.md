# Method mapping and implementation details

## Candidate memory family

For `n` past rows, age is `j=n,...,1`. Compact power weights are proportional to
`[1-(j/(h+1))^shape]_+`. The `h+1` convention makes `h` the maximum number of
strictly positive weights and works for h=1. A capped candidate uses
`min(.5,[1-(j/(h+1))^shape]_+)`. Exponential weights are proportional to
`2^(-2*(j-1)/h)`: h/2 is the half-life and the past is not hard-truncated.

The power and capped forms are motivated by Sections 4--5 of the accompanying
unpublished bounds manuscript. Unknown constants in that oracle problem are not
estimated here. Selecting among these kernels is not solving the unknown oracle.
The reported exponent `shape` is a tuning parameter, not an estimate of gamma_k.

## Neural fitting

Each expert has one tanh hidden layer (4,12 or24 units) and an intercept.
Normalization 1/m is absorbed into the output coefficients. Adam jointly learns
hidden and output parameters. Output/intercept coefficients are projected onto an
L1 ball of radius4; hence outputs are bounded for all inputs. The minimum
*training-objective* checkpoint among60 steps is retained. Selection between
experts uses only previous one-step-ahead errors. There is no certified global
optimization-error bound and no exhaustive finite-dictionary enumeration.

## Selectors and controls

| Code label | Meaning |
|---|---|
| Joint_CV | hard prequential selection among39 power/capped/expanding candidates |
| Joint_1SE | same candidates; paired one-standard-error parsimony heuristic |
| Joint_Lepski | experimental two-axis comparison score, no adaptation theorem |
| Joint_Hybrid | experimental comparison filter followed by prequential selection |
| Power_CV_matched | power shape1 plus expanding;15 candidates |
| Uniform_CV | uniform plus expanding;15 candidates |
| Exponential_CV | exponential plus expanding;15 candidates |
| Width_only_CV | three widths, expanding training histories only |
| Memory_only_CV |13 temporal kernels at fixed width12 |
| Joint_FixedShare | exponential loss update plus uniform sharing over39 candidates |
| Uniform_FixedShare | the same update over15 uniform/expanding candidates |
| Hedge_all | no-sharing exponential aggregation over all63 candidates |
| NN_expanding | fixed width12, all observed training data |
| NN_rolling_short | fixed width12, last32 training rows |
| NN_rolling_long | fixed width12, last256 training rows |
| NN_exponential | fixed width12, half-life128 |
| ADWIN_NN | width12 NN, uniform history length set by River ADWIN width |
| PageHinkley_NN | width12 NN, restart history after River Page-Hinkley alarm |
| MU_linear_conservative | Algorithm1 of Mazzetto--Upfal with conservative constants |
| MU_linear_tuned | preset .01 threshold relaxation; no inherited guarantee |
| RLS_099 / RLS_0995 | recursive linear least squares, forgetting .99/.995 |
| RF_expanding / RF_rolling |32-tree random forests, all past/last256 rows |
| ARF | River Adaptive Random Forest regressor,10 trees |
| HAT | River Hoeffding Adaptive Tree regressor |
| Mean_recent / Zero | last64 response average / zero predictor |
| Oracle_grid | infeasible current-target choice on independent selection probes |

Power_CV_matched, Uniform_CV and Exponential_CV have the same number of candidate
configurations. Their score horizons are allowed to differ because each was chosen
using the same development procedure. The larger Joint family has an additional
search cost. The point of these matched controls is to distinguish kernel shape
from the benefit or cost of searching more configurations.

## Literature algorithms versus adaptations

Mazzetto--Upfal: the implementation computes the **exact** supremum of absolute
squared-loss discrepancy for the bounded Euclidean linear class, using global
trust-region solves (including the hard case). It compares the latest r and2r
observations and stops when discrepancy exceeds4S, otherwise doubles r. X with an
intercept is divided by sqrt(d+1); Y is clipped after division by4. Conservative
uniform-convergence constants are C1=16, C2=2sqrt(2), delta=.1. The paper states
O(1) constants for this class, not a calibrated implementation recommendation.
The .01 variant is explicitly a relaxed-threshold adaptation. Neither baseline
is a generic joint neural architecture selector. Clipping changes the regression
target, and its bounded-label guarantee does not directly cover original Gaussian
responses or an unseen jump at the next prediction time.

ADWIN and Page-Hinkley: detectors are the actual River0.22 implementations.
Their inputs are clipped normalized prequential squared errors from their own
neural wrapper. The NN wrappers are our adaptations, not the original papers'
complete algorithms. Refit timing is shared with the neural bank. A noise-variance
change may trigger them even when the regression function has not changed.

Fixed-share: implements exponential loss weighting followed by a1% uniform-share
step. Losses are normalized by4 times the initial observed response variance and
clipped to[0,1]. This is an ensemble prediction, not one chosen model. Its generic
expert-regret interpretation is not a proof of the manuscript's instantaneous
L2(P_t) oracle inequality. No theoretical claim is made for original unbounded
squared errors using that clipped update.

## Deliberately not claimed

Brock--Nagler supply weighted learning theory, not a deployable benchmark algorithm
with a specified data-driven selector. We do not invent one and label it theirs.
The empirical two-axis comparison rule is not a proved neural extension of
Mazzetto--Upfal or a proved Goldenshluger--Lepski procedure. No SOTA or novelty claim
is inferred from the current comparisons.


## Second-stage adaptive aggregation

The enhanced campaign retains the 63 trained neural candidates and adds four
deterministic output scales (0.25, 0.5, 0.75, 1) for every candidate, plus zero,
recent mean and RLS(.99/.995). This produces 256 expert forecasts with no
additional neural fitting.

AdaHedge_NN uses the 63 original neural experts. AdaHedge_Extended uses all 256.
ShareGrid_NN and ShareGrid_Extended run 25 fixed-share trackers formed by
eta in {0.25,0.5,1,2,4} crossed with share in
{0,1/512,1/128,1/32,1/8}; an AdaHedge master combines their forecasts. The grid
is frozen before the test campaign. Complexity-prior variants are separately
labelled empirical ablations. Top-3 and top-5 variants prune only the final
deployment vector.

The meta loss clips the observed response to [-4,4] and uses squared error divided
by 64. Expert forecasts are already bounded in the same interval. Hence the
surrogate is in [0,1] and convex in the issued prediction. Current integrated
risk is still evaluated on unobserved current-distribution probes and is not
fed to the meta learner. Detailed regret scope is in ADAPTIVE_AGGREGATION.md.
