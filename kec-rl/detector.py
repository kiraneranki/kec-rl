"""Deepfake detector: backbone trunk -> KAN encoder -> linear head.

Supports an inference-time Gaussian randomisation level `nu`, which the RL
controller sets and over which the adaptive attacker takes
expectation-over-transformation (see kecrl/attacks/).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .kan import KANEncoder


def build_backbone(name: str = "xception", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return (feature_extractor, feature_dim).

    Uses timm so backbones (xception, efficientnet_b4, convnext_tiny,
    vit_base_patch16_224) can be swapped for the cross-architecture study.
    """
    import timm

    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    feat_dim = model.num_features
    return model, feat_dim


class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        backbone: str = "xception",
        kan_widths: tuple[int, ...] | None = None,
        grid_size: int = 8,
        spline_order: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.trunk, feat_dim = build_backbone(backbone, pretrained)
        if kan_widths is None:
            # project backbone features down through the KAN stack
            kan_widths = (feat_dim, 1024, 512, 256)
        else:
            kan_widths = (feat_dim,) + tuple(kan_widths)
        self.encoder = KANEncoder(kan_widths, grid_size, spline_order)
        self.head = nn.Linear(self.encoder.out_dim, 2)
        self.nu = 0.0  # randomisation level, set by the RL controller

    def set_randomisation(self, nu: float) -> None:
        self.nu = float(max(0.0, nu))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if self.nu > 0.0 and self.training is False:
            x = x + torch.randn_like(x) * self.nu
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.features(x))
        return self.head(z)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.features(x))
