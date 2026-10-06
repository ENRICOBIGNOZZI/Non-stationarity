# Bridge from online aggregation to average current risk

This note records the bridge theorem needed to connect the practical expert
master to the manuscript's current-risk bounds. It is deliberately an
**average-risk** statement. It does not claim a pointwise adaptive oracle
inequality.

## Setup

At time t, expert i issues a past-measurable function f_{t,i}: X -> [-B,B].
The master uses a past-measurable probability vector p_t and predicts

\[
\bar f_t(x)=\sum_{i=1}^K p_{t,i}f_{t,i}(x).
\]

The data satisfy the manuscript model

\[
Y_t=f_t^\star(X_t)+\varepsilon_t,\qquad
\mathbb E[\varepsilon_t\mid X_t]=0,
\]

with independent, non-identically distributed pairs, |f_t^*| <= B and the
conditional Bernstein moment condition. Write

\[
\mathcal E_t(f)=R_t(f)-R_t(f_t^\star)
=\|f-f_t^\star\|_{L^2(P_t)}^2.
\]

An expert path i_{1:T} may be deterministic or chosen from the deterministic
DGP/oracle quantities. It must not depend on future realised noise.

## Online tracking term

Suppose first that |Y_t| <= M almost surely. Normalize square loss by
G^2=(M+B)^2. The normalized losses lie in [0,1], and convexity gives

\[
\ell_t(\bar f_t)\leq \sum_i p_{t,i}\ell_t(f_{t,i}).
\]

For a fixed-share tracker with learning rate eta, share alpha and base prior pi,
the standard exponential-weights path argument yields

\[
\sum_{t=1}^T \ell_t(\bar f_t)
-
\sum_{t=1}^T \ell_t(f_{t,i_t})
\leq
\frac{C_\pi(i_{1:T};\alpha)}{\eta}
+\frac{\eta T}{8},
\]

where

\[
C_\pi(i_{1:T};\alpha)
=
-\log\pi_{i_1}
-
\sum_{t=2}^T
\log\left(
(1-\alpha)\mathbf 1\{i_t=i_{t-1}\}
+\alpha\pi_{i_t}
\right).
\]

For a uniform prior and a path with S switches,

\[
C_\pi
\leq
\log K
+S\log(K/\alpha)
+(T-1-S)\log(1/(1-\alpha)).
\]

The implementation runs a finite grid of (eta,alpha) trackers and an AdaHedge
master over their issued forecasts. If J trackers are used, denote the outer
AdaHedge regret by R_AH(T,J); the standard bound is of order
sqrt(T log J). Hence the normalized cumulative loss is within

\[
\min_{(\eta,\alpha)\in\mathcal G}
\left\{
C_\pi/\eta+\eta T/8
\right\}
+R_{AH}(T,J)
\]

of every comparator path.

## Removing bounded outcomes under Bernstein noise

The paper does not assume bounded noise. Let u_T(delta) be any deterministic
level satisfying

\[
\Pr\left(\max_{t\leq T}|\varepsilon_t|>u_T(\delta)\right)
\leq \delta.
\]

The conditional Bernstein assumption supplies such a level of logarithmic
order; the manuscript already uses the explicit Bernstein envelope in its
finite-sample proof. On this event, |Y_t| <= B+u_T. Therefore the preceding
tracking inequality applies to **unclipped realised squared losses** after
normalization by

\[
G_T^2=(2B+u_T)^2.
\]

The practical code instead clips the response in the meta update at a fixed
robust scale. Its deterministic online guarantee is therefore for that bounded
surrogate. The theorem above describes a theory-calibrated variant when the
noise envelope is known or conservatively bounded.

## From realised loss to current integrated risk

For two past-measurable forecasts f_t,g_t in [-B,B], define

\[
Z_t=
\{(Y_t-f_t(X_t))^2-(Y_t-g_t(X_t))^2\}
-
\{R_t(f_t)-R_t(g_t)\}.
\]

Then (Z_t) is a martingale-difference sequence. Expanding
Y_t=f_t^*(X_t)+epsilon_t shows that Z_t is the sum of a bounded design term and
a conditionally Bernstein noise term. Consequently a martingale Bernstein
inequality gives, with probability at least 1-delta,

\[
\left|\sum_{t=1}^T Z_t\right|
\lesssim
B(B+\sigma)\sqrt{T\log(1/\delta)}
+
(B^2+Bb)\log(1/\delta).
\]

Thus, simultaneously with the online tracking event,

\[
\frac1T\sum_{t=1}^T \mathcal E_t(\bar f_t)
\leq
\frac1T\sum_{t=1}^T \mathcal E_t(f_{t,i_t})
+
\frac{G_T^2}{T}
\left[
\min_{(\eta,\alpha)\in\mathcal G}
\{C_\pi/\eta+\eta T/8\}
+R_{AH}(T,J)
\right]
+
\operatorname{Mart}_T(\delta),
\]

where Mart_T(delta)=O(sqrt(log(1/delta)/T)) up to the displayed fixed signal and
noise scales.

The irreducible conditional noise variance cancels in the risk difference.
This is the key reason the online expert guarantee can be connected to the
statistical target of the manuscript.

## Combining with the neural oracle bound

For every predetermined expert (m,w), Theorem 3.2 of the manuscript supplies a
high-probability bound of the form

\[
\mathcal E_t(m,w)
\lesssim
A m^{-2\alpha}
+
\frac{b_Tm}{n_{\rm eff}(w)}
+
V_k^2 A_{\gamma_k,t}(w)^2
+
\text{remainder}.
\]

Choose any deterministic path through the finite expert library, for example
the best library approximation to the blockwise oracle
(m_t^*,w_t^*). Summing these bounds and inserting them above gives an
**adaptive average-risk inequality**:

\[
\frac1T\sum_t \mathcal E_t(\bar f_t)
\lesssim
\inf_{i_{1:T}}
\frac1T\sum_t
\Big[
A_{i_t}+
S_{i_t}+
D_{i_t}
\Big]
+
\text{library approximation}
+
\text{switching/aggregation penalty}
+
\text{martingale and optimisation remainders}.
\]

For K_block piecewise-Hölder blocks, a comparator that changes expert only when
the locally optimal library member changes has O(K_block) structural switches.
The fixed-share term then vanishes per observation whenever that switching
complexity grows slowly relative to T.

This is materially stronger than saying that a uniform bound remains valid
after data-dependent selection. It explains why aggregation can adapt without
estimating V_k, gamma_k, alpha, breakpoints or the number of switches.

## What remains distinct

1. The result is an **average current-risk** guarantee, not pointwise optimality
   immediately after an unrestricted jump. The manuscript proves why the latter
   is impossible without new-regime observations.
2. The practical fixed clipping used in the simulations robustifies the master
   but changes the exact theoretical loss on rare large-noise observations.
3. The finite library must contain useful approximations to the blockwise oracle;
   the theorem cannot rescue a missing model class. This is why the enhanced pool
   adds zero, shrinkage and linear experts.
4. Top-k pruning is a deployment heuristic. The tracking theorem applies to the
   full mixture before pruning.

Primary references for the aggregation step are Herbster--Warmuth (fixed share)
and de Rooij--van Erven--Gruenwald--Koolen (AdaHedge). The neural current-risk
bound is the companion manuscript in this project.
