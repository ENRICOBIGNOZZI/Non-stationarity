"""Batched, genuinely trainable one-hidden-layer neural experts.

All hidden and output parameters are learned. Models are freshly initialized at
each refit: obsolete data cannot persist through optimizer momentum. The same
initialization is used across memories at each width. Bounded output variation
ensures |f(x)| <= B for every x, not merely the training points. This is a
continuous-neuron practical implementation, NOT exhaustive finite-dictionary ERM.
"""
import numpy as np
import torch
from .kernels import ExpertSpec, weights, diagnostics


def project_l1_rows(z, radius):
    """Euclidean projection of each row onto an L1 ball (including intercept)."""
    u = z.abs().sort(dim=1, descending=True).values
    css = u.cumsum(1) - radius
    j = torch.arange(1, z.shape[1]+1, dtype=z.dtype, device=z.device)[None, :]
    pos = u > css/j
    rho = pos.sum(1).clamp(min=1) - 1
    theta = (css.gather(1, rho[:, None]) / (rho[:, None]+1)).clamp(min=0)
    return z.sign()*(z.abs()-theta).clamp(min=0)


class NeuralBank:
    def __init__(self, specs, d, seed, steps=60, lr=.03, bound=4., hidden_bound=6.,
                 freeze_hidden=False):
        self.specs, self.d, self.seed = list(specs), d, seed
        self.steps, self.lr = steps, lr
        self.bound, self.hidden_bound = bound, hidden_bound
        self.freeze_hidden = freeze_hidden
        self.nfits = 0
        self.train_loss = None
        self.diag = []

    def fit(self, X, y, extra_memories=None):
        if len(X) != len(y) or len(y) == 0 or X.shape[1] != self.d:
            raise ValueError("invalid history")
        n, K = len(y), len(self.specs)
        M = max(s.width for s in self.specs)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2001]))
        init_v = rng.normal(0, 1.3/np.sqrt(self.d), (self.d, M))
        init_b = rng.uniform(-.5, .5, M)
        init_a = rng.normal(0, .05, M)
        mask = torch.tensor([[j < s.width for j in range(M)] for s in self.specs], dtype=torch.float32)
        self.mask = mask
        self.v = torch.nn.Parameter(torch.tensor(np.broadcast_to(init_v, (K, self.d, M)).copy(), dtype=torch.float32), requires_grad=not self.freeze_hidden)
        self.hb = torch.nn.Parameter(torch.tensor(np.broadcast_to(init_b, (K, M)).copy(), dtype=torch.float32), requires_grad=not self.freeze_hidden)
        self.out = torch.nn.Parameter(torch.tensor(np.column_stack([np.broadcast_to(init_a,(K,M)), np.zeros(K)]).copy(), dtype=torch.float32))
        W = []
        self.diag = []
        for i, spec in enumerate(self.specs):
            if extra_memories and i in extra_memories:
                w = weights(n, "uniform", max(1, int(extra_memories[i])))
            else:
                w = weights(n, spec.kind, spec.h, spec.shape, spec.cap)
            W.append(w)
            self.diag.append(diagnostics(w))
        wt = torch.tensor(np.array(W), dtype=torch.float32)
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        params = [self.out] if self.freeze_hidden else [self.v, self.hb, self.out]
        opt = torch.optim.Adam(params, lr=self.lr)
        best_loss = torch.full((K,), torch.inf)
        best = [p.detach().clone() for p in [self.v, self.hb, self.out]]
        # Fixed, shared optimization budget. Best training objective checkpoint only.
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            h = torch.tanh(torch.einsum('nd,kdm->knm', x, self.v) + self.hb[:,None,:])
            p = (h*(self.out[:,:M]*mask)[:,None,:]).sum(2) + self.out[:,-1,None]
            losses = (wt*(p-yt)**2).sum(1)
            with torch.no_grad():
                take = losses < best_loss
                best_loss[take] = losses[take]
                for dst, src in zip(best, [self.v, self.hb, self.out]):
                    dst[take] = src[take]
            losses.sum().backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=100.)
            opt.step()
            with torch.no_grad():
                self.v.clamp_(-self.hidden_bound, self.hidden_bound)
                self.hb.clamp_(-self.hidden_bound, self.hidden_bound)
                self.out[:,:M].mul_(mask)
                self.out.copy_(project_l1_rows(self.out, self.bound))
        with torch.no_grad():
            for dst, src in zip([self.v, self.hb, self.out], best):
                dst.copy_(src)
        self.train_loss = best_loss.numpy()
        self.nfits += K
        if not np.isfinite(self.train_loss).all():
            raise FloatingPointError("nonfinite neural fit")
        return self

    def predict(self, X):
        if self.train_loss is None:
            raise RuntimeError("fit must precede predict")
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
            h = torch.tanh(torch.einsum('nd,kdm->knm', x, self.v) + self.hb[:,None,:])
            p = (h*(self.out[:,:-1]*self.mask)[:,None,:]).sum(2) + self.out[:,-1,None]
            return p.numpy().T.copy()
