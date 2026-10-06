"""Causal selection and aggregation. All update methods consume past outcomes only.

The estimator-comparison filters are experimental heuristics, NOT an application
of the Mazzetto--Upfal theorem. The kernel shape is a tuning parameter, not a
consistent estimator of temporal smoothness.
"""
from collections import deque
import numpy as np


class Prequential:
    def __init__(self, specs, memory=96, parsimony=0.):
        self.specs = specs
        self.memory, self.parsimony = memory, parsimony
        self.losses = deque(maxlen=memory)

    def update(self, predictions, y):
        losses = (np.asarray(predictions)-y)**2
        if not np.isfinite(losses).all():
            raise FloatingPointError("invalid candidate forecast")
        self.losses.append(losses)

    def scores(self):
        if not self.losses:
            return np.zeros(len(self.specs))
        return np.mean(self.losses, axis=0)

    def choose(self, indices):
        ids = np.asarray(indices, dtype=int)
        if not len(ids):
            raise ValueError("empty candidate family")
        s = self.scores()
        best = ids[np.argmin(s[ids])]
        if self.parsimony > 0 and len(self.losses) >= 8:
            # Paired one-standard-error heuristic, no finite-sample validity claim.
            l = np.asarray(self.losses)
            diff = l[:, ids] - l[:, best, None]
            se = diff.std(axis=0, ddof=1) / np.sqrt(len(l))
            allowed = ids[(s[ids]-s[best]) <= self.parsimony*se]
            return min(allowed, key=lambda j: (self.specs[j].width, -self.specs[j].h))
        # Stable deterministic tie-break. No access to true drift or probe risk.
        ties = ids[np.isclose(s[ids], s[best], atol=1e-12, rtol=0)]
        return int(min(ties, key=lambda j: (self.specs[j].width, -self.specs[j].h)))


class FixedShare:
    """Exponentiated loss update followed by uniform fixed sharing.

    Inspired by Herbster & Warmuth (1998); clipped/normalized scoring is explicit.
    beta=0 gives ordinary exponential-weight aggregation (Hedge).
    """
    def __init__(self, size, eta=1., share=.01, scale=1., active=None):
        self.active = np.arange(size) if active is None else np.asarray(active, dtype=int)
        self.p = np.full(len(self.active), 1/len(self.active))
        self.eta, self.share, self.scale = eta, share, max(float(scale), 1e-8)
        self.size = size

    def vector(self):
        v = np.zeros(self.size)
        v[self.active] = self.p
        return v

    def update(self, forecasts, y):
        loss = np.clip((np.asarray(forecasts)[self.active]-y)**2/self.scale, 0., 1.)
        logp = np.log(np.maximum(self.p, 1e-300)) - self.eta*loss
        logp -= logp.max()
        p = np.exp(logp)
        p /= p.sum()
        self.p = (1-self.share)*p + self.share/len(p)


def comparison_penalties(specs, predictions, diagnostics, variance_scale=1., threshold=.25):
    """Two-axis comparison diagnostic evaluated on past covariates only."""
    K = len(specs)
    dist = np.mean((predictions[:, :, None]-predictions[:, None, :])**2, axis=0)
    radius = variance_scale*np.array([(d["q"]+d["v"])*(s.width+np.log(1+K))
                                      for s, d in zip(specs, diagnostics)])
    D, A = np.zeros(K), np.zeros(K)
    for i, s in enumerate(specs):
        shorter = [j for j, r in enumerate(specs) if r.kind == s.kind and r.shape == s.shape
                   and r.cap == s.cap and r.width == s.width and 0 < r.h < s.h]
        larger = [j for j, r in enumerate(specs) if r.kind == s.kind and r.shape == s.shape
                  and r.cap == s.cap and r.h == s.h and r.width > s.width]
        if shorter:
            D[i] = max(0., np.max(dist[i, shorter]-threshold*(radius[i]+radius[shorter])))
        if larger:
            A[i] = max(0., np.max(dist[i, larger]-threshold*(radius[i]+radius[larger])))
    return D, A, radius
