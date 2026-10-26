"""Measured-smoothness probes used for the robustness claim.

These produce the numbers behind manuscript Figures 4-5. The robustness claim
rests on these MEASURED quantities, not on the loose worst-case Lipschitz
bound of Theorem 1.

  * power_iteration_lipschitz : operator-norm (local Lipschitz) estimate of a
    module around a batch of inputs.
  * loss_landscape_1d         : loss along the adversarial-gradient direction,
    averaged over a batch (KAN basin vs MLP valley).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _normalize(v):
    return v / (v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1))) + 1e-12)


def power_iteration_lipschitz(module, x, iters: int = 20) -> float:
    """Estimate the local operator norm (Lipschitz constant) of `module` at `x`.

    Uses the Jacobian-vector / vector-Jacobian products via autograd; returns the
    largest singular value estimate averaged over the batch.
    """
    x = x.clone().detach().requires_grad_(True)
    v = torch.randn_like(x)
    v = _normalize(v)
    sigma = 0.0
    for _ in range(iters):
        x_ = x.clone().detach().requires_grad_(True)
        out = module(x_)
        # Jacobian-vector product via double backprop
        u = torch.autograd.functional.jvp(
            lambda z: module(z), (x_,), (v,), create_graph=False
        )[1]
        u_norm = u.flatten(1).norm(dim=1)
        # back-project
        g = torch.autograd.grad(
            (module(x_) * _normalize(u).detach()).sum(), x_, retain_graph=False
        )[0]
        v = _normalize(g).detach()
        sigma = float(u_norm.mean())
    return sigma


def global_lipschitz(encoder, x, iters: int = 20) -> float:
    """Convenience wrapper: operator-norm estimate of a full encoder stack."""
    return power_iteration_lipschitz(encoder, x, iters)


def loss_landscape_1d(model, x, y, n: int = 41, span: float = 1.0):
    """Return (alphas, mean_loss) along v = grad_x L / ||grad_x L||.

    Averaged over the batch; this is the empirical probe of curvature.
    """
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    g = torch.autograd.grad(loss, x)[0]
    v = g / (g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
    alphas = torch.linspace(-span, span, n, device=x.device)
    curve = []
    with torch.no_grad():
        for a in alphas:
            xa = torch.clamp(x + a * v, 0, 1)
            curve.append(float(F.cross_entropy(model(xa), y, reduction="mean")))
    return alphas.cpu().tolist(), curve
