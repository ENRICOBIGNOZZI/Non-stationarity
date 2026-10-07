# Adaptive aggregation layer

The oracle theory determines useful expert structure: temporal kernels, memory
scales and neural widths. This note states what the practical master algorithm
does and what its online guarantee does and does not imply.

## Bounded online surrogate

Every base predictor is clipped to [-B,B] and the observed response is clipped to
the same interval only for the meta update. Define

\[
\widetilde\ell_{t,i}
=
\frac{(\widehat f_{t,i}(X_t)-[Y_t]_{[-B,B]})^2}{4B^2}
\in[0,1].
\]

For any probability vector p, squared-loss convexity gives

\[
\widetilde\ell_t\!\left(\sum_i p_i\widehat f_{t,i}\right)
\le \sum_i p_i\widetilde\ell_{t,i}.
\]

The current-distribution integrated risk from the manuscript is still the
simulation evaluation metric. It is never used by a feasible meta update.

## AdaHedge

For cumulative expert losses L_{t-1,i}, AdaHedge plays exponential weights with

\[
\eta_t=\frac{\log K}{\Delta_{t-1}},
\]

with the infinite-learning-rate limit at Delta=0. Delta is the observed cumulative
mixability gap. Thus there is no learning-rate tuning on the final sample.
The standard AdaHedge result gives parameter-free expert regret with worst-case
order sqrt(T log K), while adapting more aggressively on easy sequences. Because
our issued prediction is the convex mixture above, its bounded squared surrogate
is no larger than the corresponding Hedge loss.

## Fixed share with unknown switching rate

A fixed-share tracker first applies exponential weights with learning rate eta and
then sends a fraction alpha of its mass back to a base prior. This is equivalent
to a Markov prior over expert paths. For a uniform base prior and a path with S
expert switches, a standard exponential-weights calculation gives the surrogate
tracking bound

\[
L_T^{(\eta,\alpha)}
-
L_T(i_{1:T})
\le
\frac{
 \log K
 + S\log(K/\alpha)
 + (T-1-S)\log(1/(1-\alpha))
}{\eta}
+\frac{\eta T}{8},
\]

using the lower bounds alpha/K for a switch transition and 1-alpha for a stay.
The displayed expression is deliberately conservative.

Our practical algorithm does not estimate S. It runs a frozen grid

\[
\eta\in\{0.25,0.5,1,2,4\},\qquad
\alpha\in\{0,1/512,1/128,1/32,1/8\},
\]

and an AdaHedge master combines the 25 tracker forecasts. Therefore the outer
master pays only the usual finite-expert aggregation overhead relative to the
best tracker in this frozen grid. The result is a cumulative tracking statement
for the bounded surrogate. It is not an instantaneous oracle inequality for
L2(P_t), and no such claim is made.

## Rich expert pool

The original 63 neural experts are retained. Without new neural fitting, the
enhanced pool adds output scales 0.25, 0.5, 0.75 and 1 for every neural expert,
plus zero, a recent mean and RLS(.99/.995). There are therefore 256 expert
forecasts. The zero/shrinkage candidates address the explicit low-signal failure
observed in the first campaign.

The principal method uses a uniform prior. A separately labelled complexity-prior
ablation mildly favours smaller widths and simple predictors. Top-3 and top-5
versions prune the final deployment mixture after the adaptive state is computed;
they test whether most ensemble robustness can be retained at low prediction cost.

## Relation to the oracle paper

The oracle theory and the aggregation theorem answer different questions.

* The oracle bound explains which memories and model complexities are statistically
  sensible for a given local drift structure.
* The expert layer competes online with changing members of a finite library
  without estimating V_k, gamma_k, alpha, breakpoints or the number of switches.
* Combining the two does not by itself prove attainment of the pointwise
  current-risk oracle. A separate bridge theorem would be needed.

Primary references: Herbster and Warmuth (1998) for fixed share; de Rooij,
van Erven, Grunwald and Koolen (2014) for AdaHedge.
