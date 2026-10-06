# Adaptive aggregation to continuum-oracle risk

The detailed proof is in `docs/ADAPTIVE_ORACLE_THEOREM.tex`.

The key improvement over the earlier bridge is to exploit **mixability of
bounded square loss**, rather than a generic Hedge bound. After normalizing by a
high-probability Bernstein envelope, square loss is (1/2)-exp-concave. A
hierarchical fixed-share master therefore pays a path code length, not a
`sqrt(T)` regret term.

For a path (i_{1:T}) with (S) switches through a finite expert library
(mathcal I_T), the realized-loss overhead is of order

[
D_T^2left[
log|mathcal I_T|
+Slog!rac{e|mathcal I_T|T}{S}
+log|mathcal Q_T|
ight].
]

A localized martingale Bernstein argument converts this to the manuscript's
current integrated risk without reintroducing a generic square-root-in-(T)
penalty. The resulting high-probability inequality is

[
rac1Tsum_t mathcal E_t(widehat f_t)
le
inf_{i_{1:T}}
left{
(1+zeta)rac1Tsum_t mathcal B_{t,i_t}
+
rac{C_zetaXi_T^2}{T}
left[
mathcal C_T(i_{1:T})+log(1/delta)
ight]
ight},
]

where (mathcal B_{t,i}) is the finite-sample neural bound already proved in
the manuscript.

The second step is a geometric sieve over neural width (m), memory (H), and
kernel shape (g). With grid resolution (epsilon_T=1/log T), the library has
only (O(log^5T)) members while approximating the continuum
((m,H,g)) oracle within (1+o(1)) in the interior. The blockwise oracle's
quantized path changes only (O(Klog^2T)) times. Hence

[
operatorname{AdaptCost}(K,T)
=
O!left(rac{K,operatorname{polylog}T}{T}ight).
]

This is asymptotically smaller than the nonparametric oracle terms for fixed
piecewise-Hölder complexity (K). Therefore the adaptive algorithm recovers
the same polynomial Barron/Sobolev rates, up to logarithms, without estimating
(alpha,V_k,gamma_k), or the breakpoints.

This statement is an average-current-risk theorem. Pointwise optimality at the
first observation after an unrestricted jump remains impossible, consistently
with Section 6.1 of the manuscript.
