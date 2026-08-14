"""Muon optimizer (plan §5) — momentum SGD + Newton-Schulz orthogonalization.

Highest learning-per-line file in the repo. Muon has won every nanoGPT
speedrun record since Oct 2024 on sample efficiency; used at Kimi-K2 scale.

Idea: SGD gives an update matrix G. Muon replaces G with its nearest
semi-orthogonal matrix (U V^T from G's SVD) — computed cheaply via a 5-step
Newton-Schulz iteration in bf16, no actual SVD. Orthogonal updates spread
learning across all singular directions instead of chasing the dominant one.

Applies ONLY to 2-D hidden weight matrices. Embeddings, unembedding, and
norms go to AdamW (see train.make_optimizers). Split is the exact speedrun/
Kimi-K2 recipe.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate G -> U V^T (orthogonalization). Quintic coeffs tuned so the
    iteration pushes singular values toward 1; runs in bf16 for speed."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:                       # keep the "tall" dim as columns
        X = X.T
    X = X / (X.norm() + 1e-7)            # spectral norm <= 1 so the iteration converges
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, nesterov, ns = group["lr"], group["momentum"], group["nesterov"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(g))
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nesterov else buf
                g = zeropower_via_newtonschulz5(g, ns)
                # shape correction: non-square matrices get scaled so update RMS matches
                p.add_(g, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)


def _selfcheck():
    torch.manual_seed(0)
    # 1) Newton-Schulz really orthogonalizes: singular values pushed near 1.
    G = torch.randn(64, 32)
    s = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, 5).float())
    assert 0.6 < s.min() and s.max() < 1.4, f"NS singular values off: {s.min():.2f}..{s.max():.2f}"

    # 2) Optimizer actually descends a real 2-D matrix loss (linear regression).
    X = torch.randn(256, 16)
    W_true = torch.randn(16, 8)
    Y = X @ W_true
    W = torch.zeros(16, 8, requires_grad=True)
    opt = Muon([W], lr=0.05)
    l0 = None
    for _ in range(200):
        loss = ((X @ W - Y) ** 2).mean()
        l0 = l0 or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.2 * l0, f"Muon failed to descend: {l0:.3f} -> {loss.item():.3f}"
    print(f"muon selfcheck PASS | NS svals {s.min():.2f}..{s.max():.2f} | loss {l0:.3f} -> {loss.item():.3f}")


if __name__ == "__main__":
    _selfcheck()
