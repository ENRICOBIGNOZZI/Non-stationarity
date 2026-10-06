"""Independent streams with bounded targets, plus explicitly labelled stress tests.

The learner receives X/Y only. Target functions, changepoints, probe samples,
and oracle losses are evaluator-only data. All randomness has separate streams.
"""
from dataclasses import dataclass
import numpy as np

SCENARIOS = (
    "stationary_linear", "stationary_nonlinear", "slow_smooth", "fast_smooth",
    "holder_half", "single_jump", "multiple_jumps", "piecewise_holder",
    "recurring", "complexity_change", "covariate_only", "joint_shift",
    "volatility_only", "heteroskedastic", "low_signal", "heavy_tails",
    "serial_noise", "support_shift",
)
STRESS = {"heavy_tails": "Student-t(3) noise violates conditional Bernstein tails",
          "serial_noise": "AR(1) errors violate pair independence; no thinning used",
          "support_shift": "disjoint design supports violate design comparability"}

@dataclass
class Stream:
    X: np.ndarray
    y: np.ndarray
    name: str
    seed: int
    breakpoints: list
    noise_scale: np.ndarray
    generator: object


def tilt_uniform(u: np.ndarray, rho: float):
    """Inverse CDF of density (1 + rho*x)/2 on [-1,1], stable at rho=0."""
    return (4*u - 2 + rho) / (np.sqrt((1-rho)**2 + 4*rho*u) + 1)


class Generator:
    def __init__(self, name: str, T: int, d: int, seed: int):
        if name not in SCENARIOS or d < 2 or T < 64:
            raise ValueError("unknown scenario, d<2, or T<64")
        self.name, self.T, self.d, self.seed = name, T, d, seed
        # Teachers are sampled once, independently of all X and noise draws.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1729]))
        self.v = rng.normal(0, 1.8 / np.sqrt(d), (4, d, 12))
        self.b = rng.uniform(-.6, .6, (4, 12))
        self.a = rng.choice([-1., 1.], (4, 12)) * np.linspace(1., .4, 12)
        self.a /= np.abs(self.a).sum(1, keepdims=True)
        self.phases = rng.uniform(0, 2*np.pi, 12)
        self.breakpoints = self._breakpoints()

    def _breakpoints(self):
        n = self.name
        if n in ("single_jump", "complexity_change", "support_shift"):
            return [self.T//2]
        if n in ("multiple_jumps", "recurring"):
            return [self.T//4, self.T//2, 3*self.T//4]
        if n == "piecewise_holder":
            return [self.T//3, 2*self.T//3]
        # A covariate/noise-only change is not a target-function break.
        return []

    def draw_X(self, t: int, n: int, rng):
        x = rng.uniform(-1, 1, (n, self.d))
        u = t / max(1, self.T - 1)
        if self.name in ("covariate_only", "joint_shift"):
            rho = .6 * np.sin(2*np.pi*u)
            x[:, 0] = tilt_uniform((x[:, 0]+1)/2, rho)
        if self.name == "support_shift":
            x[:, 0] = (-.6 if t < self.T//2 else .6) + .4*x[:, 0]
        return x

    def _teacher(self, x, k):
        return 2.2 * np.tanh(x @ self.v[k] + self.b[k]) @ self.a[k]

    def rough(self, u, gamma=.5):
        # A dyadic Holder series: constant independent of truncation for gamma<1.
        j = np.arange(10)
        amplitudes = 2. ** (-gamma*j)
        return float(np.sum(amplitudes*np.sin(2*np.pi*(2.**j)*u + self.phases[:10])) /
                     amplitudes.sum())

    def target(self, t: int, x: np.ndarray):
        u = t / max(1, self.T-1)
        n = self.name
        linear = .9*x[:, 0] - .5*x[:, 1]
        a, b = self._teacher(x, 0), self._teacher(x, 1)
        if n == "stationary_linear":
            return linear
        if n in ("stationary_nonlinear", "covariate_only", "volatility_only", "support_shift"):
            return a
        if n == "slow_smooth":
            return (1-u)*a + u*b
        if n in ("fast_smooth", "joint_shift", "heteroskedastic", "heavy_tails", "serial_noise"):
            z = .5 + .5*np.sin(4*np.pi*u)
            return (1-z)*a + z*b + .25*np.sin(2*np.pi*u)
        if n == "low_signal":
            z = .5 + .5*np.sin(4*np.pi*u)
            return .35*((1-z)*a + z*b)
        if n == "holder_half":
            z = .5 + .45*self.rough(u, .5)
            return (1-z)*a + z*b
        if n == "single_jump":
            return a if t < self.T//2 else -a + .3
        if n == "multiple_jumps":
            k = min(3, 4*t//self.T)
            return self._teacher(x, k) + (-.3 if k % 2 else .3)
        if n == "recurring":
            k = (0, 1, 0, 2)[min(3, 4*t//self.T)]
            return self._teacher(x, k)
        if n == "piecewise_holder":
            k = min(2, 3*t//self.T)
            local = (t - k*self.T/3) / (self.T/3)
            gamma = (1., .5, .25)[k]
            z = local if k == 0 else .5 + .45*self.rough(local, gamma)
            return (1-z)*self._teacher(x, k) + z*self._teacher(x, k+1)
        if n == "complexity_change":
            return linear if t < self.T//2 else a + .6*np.sin(3*x[:, 0])*np.sin(3*x[:, 1])
        raise RuntimeError(n)

    def sigma(self, t, x):
        sigma = np.full(len(x), .35)
        if self.name == "volatility_only":
            sigma *= 3 if self.T//3 <= t < 2*self.T//3 else 1
        if self.name == "heteroskedastic":
            sigma = .15 + .6*np.abs(x[:, 0])
        if self.name == "low_signal":
            sigma[:] = 1.0
        return sigma

    def sample(self):
        rng_x = np.random.default_rng(np.random.SeedSequence([self.seed, 1001]))
        rng_e = np.random.default_rng(np.random.SeedSequence([self.seed, 1002]))
        X = np.vstack([self.draw_X(t, 1, rng_x) for t in range(self.T)])
        f = np.array([self.target(t, X[t:t+1])[0] for t in range(self.T)])
        sig = np.array([self.sigma(t, X[t:t+1])[0] for t in range(self.T)])
        e = rng_e.normal(size=self.T)
        if self.name == "heavy_tails":
            e = rng_e.standard_t(3, self.T) / np.sqrt(3)
        if self.name == "serial_noise":
            for t in range(1, self.T):
                e[t] = .8*e[t-1] + .6*e[t]
        return Stream(X.astype(np.float32), (f+sig*e).astype(np.float32), self.name,
                      self.seed, self.breakpoints, sig, self)

    def probes(self, t, n, stream=0):
        # Independent from the training stream AND from previous evaluation times.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 99191, int(t), int(stream)]))
        x = self.draw_X(t, n, rng)
        return x.astype(np.float32), self.target(t, x)
