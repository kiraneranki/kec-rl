"""Kolmogorov-Arnold Network (KAN) encoder with B-spline edges.

The encoder places a learnable univariate B-spline on every edge, following
Liu et al. (2024, arXiv:2404.19756). Its purpose in KEC-RL is adversarial
robustness: spline edges of order k >= 3 are C^{k-1} continuous, which the
curvature regulariser `spline_l2_penalty` drives toward a low realised
Lipschitz constant (see kecrl/analysis/lipschitz.py).

This is a clean, dependency-light reference implementation. It is correct and
runnable; it is not tuned for maximum throughput. For large-scale training you
may prefer an optimised kernel (e.g. the `efficient-kan` package), which this
module is API-compatible with at the layer level.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """A single KAN layer: every (in, out) edge carries a B-spline + a base path.

    phi(x) = w_base * silu(x) + w_spline * sum_i c_i B_i(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        base_activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Static knot vector (extended for the spline order), registered as buffer.
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        )
        grid = grid.expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            nn.init.normal_(self.spline_weight, std=0.1 / (self.grid_size + 1))
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the B-spline basis at x.

        x:   (batch, in_features)
        out: (batch, in_features, grid_size + spline_order)
        """
        grid = self.grid  # (in_features, n_knots)
        x = x.unsqueeze(-1)  # (batch, in_features, 1)
        # order-0 basis (indicator of each knot interval)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # Cox-de Boor recursion up to spline_order
        for k in range(1, self.spline_order + 1):
            left_den = grid[:, k:-1] - grid[:, : -(k + 1)]
            right_den = grid[:, k + 1 :] - grid[:, 1:(-k)]
            left = (x - grid[:, : -(k + 1)]) / left_den * bases[:, :, :-1]
            right = (grid[:, k + 1 :] - x) / right_den * bases[:, :, 1:]
            bases = left + right
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(self.base_activation(x), self.base_weight)
        spline_basis = self.b_splines(x)  # (batch, in, coeff)
        scaled = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline = torch.einsum("bic,oic->bo", spline_basis, scaled)
        return base + spline

    def spline_l2_penalty(self) -> torch.Tensor:
        """Omega(psi): L2 norm of spline coefficients, a proxy for edge slope.

        Penalising this reduces the realised derivative magnitude beta and hence
        the measured Lipschitz constant (manuscript Eq. for Omega).
        """
        return (self.spline_weight * self.spline_scaler.unsqueeze(-1)).pow(2).mean()


class KANEncoder(nn.Module):
    """Stack of KAN layers mapping backbone features to a robust embedding.

    Default widths match the manuscript: 2048 -> 1024 -> 512 -> 256.
    """

    def __init__(
        self,
        widths: tuple[int, ...] = (2048, 1024, 512, 256),
        grid_size: int = 8,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            KANLinear(widths[i], widths[i + 1], grid_size, spline_order)
            for i in range(len(widths) - 1)
        )
        self.out_dim = widths[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def curvature_penalty(self) -> torch.Tensor:
        return sum(layer.spline_l2_penalty() for layer in self.layers)
