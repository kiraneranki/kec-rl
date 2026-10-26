"""Matched-MLP encoder baseline (for KAN-vs-MLP comparison and MLP+CR+AT).

Provides:
  * MLPEncoder           -- same widths as the KAN encoder, standard activations.
  * curvature_penalty    -- optional finite-difference curvature regulariser
                            in the style of Moosavi-Dezfooli et al. (2019),
                            used by the MLP+CR+AT baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, widths: tuple[int, ...] = (2048, 1024, 512, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.out_dim = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def curvature_regulariser(
    model: nn.Module, x: torch.Tensor, loss_fn, y: torch.Tensor, h: float = 1e-2
) -> torch.Tensor:
    """Finite-difference curvature penalty (Moosavi-Dezfooli et al., 2019).

    Approximates ||Hessian . v|| along the gradient direction by a finite
    difference of gradients, and penalises its norm.
    """
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    loss = loss_fn(out, y)
    grad = torch.autograd.grad(loss, x, create_graph=True)[0]
    v = grad / (grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
    out2 = model(x + h * v)
    loss2 = loss_fn(out2, y)
    grad2 = torch.autograd.grad(loss2, x, create_graph=True)[0]
    curv = (grad2 - grad).flatten(1).norm(dim=1) / h
    return curv.mean()
