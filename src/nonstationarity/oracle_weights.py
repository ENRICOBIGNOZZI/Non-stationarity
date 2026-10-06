"""Oracle optimisation for the nonstationary neural-network risk bounds.

The inputs are population/oracle constants. This module is NOT an adaptive
forecaster, does not train a neural network, and does not estimate drift.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize


@dataclass(frozen=True)
class WeightSolution:
    weights: NDArray[np.float64]
    objective: float
    effective_sample_size: float
    support_size: int
    threshold: float | None = None


def _ages(values: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.all(np.isfinite(a)):
        raise ValueError("ages must be a nonempty finite vector")
    if np.any(a <= 0):
        raise ValueError("ages must be strictly positive (the training data are past observations)")
    return a


def _result(w: NDArray[np.float64], a: NDArray[np.float64], variance_cost: float,
            drift_cost: float, max_weight_cost: float = 0.,
            threshold: float | None = None) -> WeightSolution:
    q = float(w @ w)
    val = variance_cost*q + drift_cost*float(w @ a)**2 + max_weight_cost*float(w.max())
    return WeightSolution(w, val, 1./q, int(np.count_nonzero(w > 1e-12)), threshold)


def optimal_power_weights(ages: ArrayLike, variance_cost: float,
                          drift_cost: float) -> WeightSolution:
    """Exact active-set solution of c ||w||_2^2 + d (a'w)^2 on the simplex.

    Formula: w_j = (z - a_j)_+ / sum_i (z - a_i)_+.
    The threshold solves sum_j a_j (z - a_j)_+ = c/d.
    """
    a = _ages(ages)
    c, d = float(variance_cost), float(drift_cost)
    if not np.isfinite(c) or c <= 0 or not np.isfinite(d) or d < 0:
        raise ValueError("variance_cost must be positive and drift_cost nonnegative")
    n = len(a)
    if d == 0 or np.all(a == a[0]):
        return _result(np.full(n, 1./n), a, c, d)
    order = np.argsort(a, kind="stable")
    aa = a[order].astype(np.longdouble)
    p1, p2 = np.cumsum(aa), np.cumsum(aa*aa)
    ratio = np.longdouble(c)/np.longdouble(d)
    z = None
    support = n
    for h in range(1, n+1):
        candidate = (p2[h-1] + ratio)/p1[h-1]
        if h == n or candidate <= aa[h]:
            z, support = candidate, h
            break
    assert z is not None
    raw = np.zeros(n, dtype=np.longdouble)
    if support == 1:
        raw[0] = 1
    else:
        raw[:support] = np.maximum(z-aa[:support], 0)
        if raw.sum() <= 0:
            raise ArithmeticError("loss of precision in active-set normalisation")
        raw /= raw.sum()
    w = np.empty(n)
    w[order] = np.asarray(raw, dtype=float)
    w /= w.sum()
    return _result(w, a, c, d, threshold=float(z))


def optimal_full_bound_weights(ages: ArrayLike, variance_cost: float,
                               drift_cost: float, max_weight_cost: float,
                               *, tolerance: float = 1e-12) -> WeightSolution:
    """Numerically solve the full convex epigraph problem, including ||w||_inf.

    The minimiser has a flat head followed by a truncated affine function of
    ages. SLSQP is used here as a small reproducible reference solver.
    """
    a = _ages(ages)
    c, d, e = map(float, (variance_cost, drift_cost, max_weight_cost))
    if not (np.isfinite(c) and c > 0 and np.isfinite(d) and d >= 0
            and np.isfinite(e) and e >= 0):
        raise ValueError("invalid nonnegative objective coefficients")
    if e == 0:
        return optimal_power_weights(a, c, d)
    n = len(a)
    scale = max(c, d*float(a.max())**2, e, 1e-100)
    cc, dd, ee = c/scale, d/scale, e/scale
    start = optimal_power_weights(a, c, d).weights
    x0 = np.r_[start, start.max()]
    def fun(x: NDArray[np.float64]) -> float:
        w = x[:-1]
        return cc*float(w @ w) + dd*float(a @ w)**2 + ee*x[-1]
    def jac(x: NDArray[np.float64]) -> NDArray[np.float64]:
        w = x[:-1]
        return np.r_[2*cc*w + 2*dd*float(a @ w)*a, ee]
    cap_jac = np.c_[-np.eye(n), np.ones(n)]
    constraints = [
        {"type": "eq", "fun": lambda x: x[:-1].sum()-1.,
         "jac": lambda x: np.r_[np.ones(n), 0.]},
        {"type": "ineq", "fun": lambda x: x[-1]-x[:-1],
         "jac": lambda x: cap_jac},
    ]
    out = minimize(fun, x0, jac=jac, method="SLSQP", bounds=[(0., 1.)]*(n+1),
                   constraints=constraints,
                   options={"ftol": tolerance, "maxiter": 3000})
    if not out.success:
        raise RuntimeError(f"convex optimiser failed: {out.message}")
    w = np.maximum(out.x[:-1], 0.)
    w /= w.sum()
    return _result(w, a, c, d, e)


def joint_discrete_oracle(n: int, horizon: int, gamma: float, drift_cost: float,
                         alpha: float, approximation_cost: float,
                         complexity_cost: float, neuron_counts: ArrayLike) -> dict:
    """Globally enumerate integer m; solve its weight subproblem exactly.

    Minimises A*m**(-2*alpha) + complexity_cost*m*sum(w**2)
               + drift_cost*(sum(w*(j/horizon)**gamma))**2.
    """
    if not (1 <= n < horizon and 0 < gamma <= 1 and alpha > 0):
        raise ValueError("need 1 <= n < horizon, gamma in (0,1], alpha > 0")
    if approximation_cost <= 0 or complexity_cost <= 0 or drift_cost < 0:
        raise ValueError("invalid costs")
    counts = np.asarray(neuron_counts)
    if counts.ndim != 1 or len(counts) == 0 or np.any(counts < 1) or np.any(counts != counts.astype(int)):
        raise ValueError("neuron_counts must contain positive integers")
    a = (np.arange(1, n+1, dtype=float)/horizon)**gamma
    best = None
    for m in np.unique(counts.astype(int)):
        sol = optimal_power_weights(a, complexity_cost*int(m), drift_cost)
        value = approximation_cost*int(m)**(-2*alpha) + sol.objective
        if best is None or value < best["objective"]:
            best = {"m": int(m), "objective": float(value), "weights": sol.weights,
                    "support_size": sol.support_size,
                    "effective_sample_size": sol.effective_sample_size}
    assert best is not None
    return best


def continuous_joint_oracle(horizon: float, alpha: float, gamma: float,
                            approximation_cost: float, complexity_cost: float,
                            drift_cost: float) -> dict:
    """Interior continuous proxy, not clipped to the current segment length.

    Uses the normalised power kernel and retains all its integration constants.
    """
    if min(horizon, alpha, gamma, approximation_cost, complexity_cost, drift_cost) <= 0 or gamma > 1:
        raise ValueError("all inputs must be positive, gamma <= 1")
    R = 2*(gamma+1)/(2*gamma+1)
    mu = 1/(2*gamma+1)
    Delta = alpha + gamma*(2*alpha+1)
    log_m = ((2*gamma+1)*math.log(2*alpha*approximation_cost)
             + 2*gamma*math.log(horizon)
             - 2*gamma*math.log(complexity_cost*R)
             - math.log(2*gamma*drift_cost*mu*mu))/(2*Delta)
    m = math.exp(log_m)
    H = complexity_cost*R/(2*alpha*approximation_cost)*m**(2*alpha+1)
    approx = approximation_cost*m**(-2*alpha)
    stat = complexity_cost*R*m/H
    drift = drift_cost*mu*mu*(H/horizon)**(2*gamma)
    return {"m": m, "H": H, "approximation": approx, "statistical": stat,
            "drift": drift, "objective": approx+stat+drift,
            "Delta": Delta, "kernel_R": R, "kernel_moment": mu}
