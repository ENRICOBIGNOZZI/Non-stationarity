"""Deterministic candidate memories derived from the manuscript's oracle families.

`h` is the maximum number of positive weights for compact kernels. Its real
bandwidth is h + 1, avoiding the off-by-one/zero-denominator case at h=1.
The exponent `shape` indexes a candidate, not an estimate of true Holder gamma.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

@dataclass(frozen=True)
class ExpertSpec:
    kind: str
    h: int
    shape: float
    width: int
    cap: float = 1.0

    @property
    def name(self):
        return f"{self.kind}:h{self.h}:g{self.shape:g}:m{self.width}:c{self.cap:g}"

    def as_dict(self):
        return asdict(self)


def weights(n: int, kind: str, h: int = 0, shape: float = 1.0,
            cap: float = 1.0) -> np.ndarray:
    """Return chronological weights for n already observed samples only."""
    if n < 1 or shape <= 0 or not (0 < cap <= 1):
        raise ValueError("n >= 1, shape > 0 and 0 < cap <= 1 are required")
    age = np.arange(n, 0, -1, dtype=float)
    if kind == "expanding":
        raw = np.ones(n)
    elif h < 1:
        raise ValueError("a positive memory scale is required")
    elif kind == "uniform":
        raw = (age <= h).astype(float)
    elif kind == "exponential":
        # h is twice the half-life; unlike compact kernels, every past row remains.
        raw = np.exp2(-2 * (age - 1) / h)
    elif kind in ("power", "capped"):
        raw = np.maximum(1 - (age / (h + 1)) ** shape, 0)
        if kind == "capped":
            raw = np.minimum(raw, cap)
    else:
        raise ValueError(f"Unknown kernel {kind}")
    z = raw.sum()
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Invalid normalization")
    return raw / z


def diagnostics(w: np.ndarray) -> dict:
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or len(w) == 0 or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError("w must be a normalized simplex vector")
    q = float(w @ w)
    return {"neff": 1 / q, "q": q, "v": float(w.max()),
            "v_over_q": float(w.max() / q), "support": int((w > 0).sum()),
            "mean_age": float(w @ np.arange(len(w), 0, -1))}


def candidate_grid(histories=(32, 96, 256, 768), widths=(4, 12, 24)):
    kernels = [("uniform", 1., 1.), ("exponential", 1., 1.),
               ("power", .5, 1.), ("power", 1., 1.), ("capped", 1., .5)]
    specs = [ExpertSpec(kind, int(h), g, int(m), cap)
             for kind, g, cap in kernels for h in histories for m in widths]
    specs += [ExpertSpec("expanding", 0, 1., int(m)) for m in widths]
    return specs


def power_oracle(ages: np.ndarray, statistical_cost: float, drift_squared: float):
    """Exact active-set solution of lambda * ||w||² + D * (a'w)².

    Input ages need not be chronological. This function uses no observed outcomes
    and is an oracle diagnostic only unless the supplied costs are actually known.
    """
    a = np.asarray(ages, dtype=float)
    if a.ndim != 1 or not len(a) or (a <= 0).any():
        raise ValueError("ages must be a nonempty positive vector")
    if statistical_cost <= 0 or drift_squared < 0:
        raise ValueError("invalid costs")
    if drift_squared == 0:
        return np.full(len(a), 1 / len(a))
    order = np.argsort(a)
    z = a[order]
    sums, sums2 = np.cumsum(z), np.cumsum(z*z)
    theta = None
    for h in range(1, len(z) + 1):
        v = (sums2[h-1] + statistical_cost / drift_squared) / sums[h-1]
        if h == len(z) or v <= z[h]:
            theta = v
            break
    raw = np.maximum(theta - a, 0)
    return raw / raw.sum()
