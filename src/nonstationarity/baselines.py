"""Published baselines, with adaptations and guarantees labelled explicitly."""
import math
import numpy as np
from scipy.optimize import brentq


def trust_region(A, b, radius=1.):
    """Global min x'Ax + 2b'x on ||x||<=radius, including the hard case."""
    A = (np.asarray(A, dtype=float)+np.asarray(A, dtype=float).T)/2
    b = np.asarray(b, dtype=float)
    ev, Q = np.linalg.eigh(A)
    q = Q.T @ b
    tol = 1e-11*max(1., np.max(np.abs(ev)), np.linalg.norm(b))
    # Positive-semidefinite unconstrained case (including a singular null space).
    if ev[0] >= -tol and np.all(np.abs(q[ev <= tol]) <= tol):
        z = np.zeros_like(q)
        good = ev > tol
        z[good] = -q[good]/ev[good]
        if np.linalg.norm(z) <= radius:
            x = Q @ z
            return x, float(x@A@x+2*b@x)
    lower = max(0., -ev[0])
    den = ev+lower
    small = den <= tol
    z = np.zeros_like(q)
    z[~small] = -q[~small]/den[~small]
    if np.all(np.abs(q[small]) <= tol) and np.linalg.norm(z) <= radius:
        # Indefinite hard case: add a minimum-eigenvalue component.
        if lower > 0 and small.any():
            z[np.flatnonzero(small)[0]] = np.sqrt(max(0., radius**2-z@z))
    else:
        def norm_minus(lam):
            return np.linalg.norm(q/(ev+lam))-radius
        lo = lower + max(1e-13, tol*1e-3)
        hi = max(1., lower+1.)
        while norm_minus(hi) > 0:
            hi = lower+2*(hi-lower)
        lam = brentq(norm_minus, lo, hi, xtol=1e-13)
        z = -q/(ev+lam)
    x = Q@z
    return x, float(x@A@x+2*b@x)


def linear_discrepancy(X_recent, y_recent, X_old, y_old):
    """Exact loss discrepancy: latest r versus latest 2r for an L2 unit ball."""
    r = len(y_recent)
    if len(y_old) != r or r < 1:
        raise ValueError("equally sized nonempty blocks required")
    A = (X_recent.T@X_recent-X_old.T@X_old)/(2*r)
    b = -(X_recent.T@y_recent-X_old.T@y_old)/(2*r)
    c = (y_recent@y_recent-y_old@y_old)/(2*r)
    _, low = trust_region(A, b)
    _, neg = trust_region(-A, -b)
    return max(0., c-neg, -c-low)


class MazzettoUpfalLinear:
    """Algorithm 1 with exact trust-region discrepancy for Section 6's class.

    Features and responses are mapped into unit balls using a fixed label bound.
    'theory' uses conservative sufficient uniform-convergence constants (not
    numerically specified by the paper's O(1) statement). 'tuned' uses a small
    threshold multiplier and carries NO inherited confidence guarantee.
    At time t it trains through t-1: next-step drift is not observable.
    """
    def __init__(self, d, delta=.1, threshold_scale=1., label_bound=4.):
        self.d, self.delta, self.threshold_scale = d, delta, threshold_scale
        self.label_bound = label_bound
        self.coef = np.zeros(d+1)
        self.window = 0

    def phi(self, X):
        return np.column_stack([np.ones(len(X)), X])/np.sqrt(self.d+1)

    def fit(self, X, y):
        z = self.phi(X)
        target = np.clip(y/self.label_bound, -1, 1)
        r, n = 1, len(y)
        C1, C2 = 16., 2*np.sqrt(2.)
        while 2*r <= n:
            disc = linear_discrepancy(z[-r:], target[-r:], z[-2*r:-r], target[-2*r:-r])
            S = (C1+C2*np.sqrt(2*np.log(np.pi**2/(6*self.delta))))/np.sqrt(r)
            S += C2*np.sqrt(2*np.log(np.log2(r)+10)/r)
            if disc > 4*self.threshold_scale*S:
                break
            r *= 2
        self.window = r
        a, _ = trust_region(z[-r:].T@z[-r:]/r, -(z[-r:].T@target[-r:])/r)
        self.coef = a
        return self

    def predict(self, X):
        return self.label_bound*(self.phi(X)@self.coef)


class RLS:
    """Recursive least squares with a fixed forgetting factor and ridge prior."""
    def __init__(self, d, forgetting=.99, ridge=.01, bound=4.):
        self.coef = np.zeros(d+1)
        self.P = np.eye(d+1)/ridge
        self.forgetting, self.bound = forgetting, bound

    def predict(self, X):
        z = np.column_stack([np.ones(len(X)), X])
        return np.clip(z@self.coef, -self.bound, self.bound)

    def update(self, x, y):
        z = np.r_[1., x]
        Pz = self.P@z
        gain = Pz/(self.forgetting+z@Pz)
        self.coef += gain*(y-z@self.coef)
        self.P = (self.P - np.outer(gain, Pz))/self.forgetting
        self.P = (self.P+self.P.T)/2


class DriftMonitors:
    """River detectors on clipped prequential squared errors.

    ADWIN-window NN uses the retained detector width as a training lookback.
    Page-Hinkley-reset NN starts a fresh history after an alarm. These NN wrappers
    are adaptations; the underlying detector implementations are River's.
    """
    def __init__(self, min_history=16):
        from river import drift
        self.adwin = drift.ADWIN(delta=.01, clock=16, min_window_length=5, grace_period=20)
        self.ph = drift.PageHinkley(min_instances=30, delta=.005, threshold=10., alpha=.9999)
        self.last_reset = 0
        self.count = 0
        self.min_history = min_history
        self.alarms = {"adwin": [], "page_hinkley": []}

    def update(self, t, adwin_loss, ph_loss, scale):
        self.adwin.update(float(np.clip(adwin_loss/scale, 0, 1)))
        self.ph.update(float(np.clip(ph_loss/scale, 0, 1)))
        self.count += 1
        if self.adwin.drift_detected:
            self.alarms["adwin"].append(int(t))
        if self.ph.drift_detected:
            self.last_reset = t+1
            self.alarms["page_hinkley"].append(int(t))

    def windows(self, t):
        return max(self.min_history, int(self.adwin.width)), max(1, t-self.last_reset)


class RiverModels:
    def __init__(self, seed):
        from river import forest, tree
        self.models = {
            "ARF": forest.ARFRegressor(n_models=10, seed=seed, max_depth=12, grace_period=50),
            "HAT": tree.HoeffdingAdaptiveTreeRegressor(seed=seed, max_depth=12, grace_period=50),
        }

    def predict(self, X):
        rows = [dict(enumerate(map(float, row))) for row in X]
        return {name: np.array([model.predict_one(row) or 0. for row in rows])
                for name, model in self.models.items()}

    def update(self, x, y):
        row = dict(enumerate(map(float, x)))
        for model in self.models.values():
            model.learn_one(row, float(y))
