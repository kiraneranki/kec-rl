"""Smoke tests for the KEC-RL components. Run: pytest -q"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kecrl.models.kan import KANEncoder, KANLinear
from kecrl.models.ppo_controller import PPO, PPOConfig
from kecrl.analysis.lipschitz import loss_landscape_1d


def test_kan_layer_shapes():
    layer = KANLinear(16, 8, grid_size=5, spline_order=3)
    x = torch.randn(4, 16)
    out = layer(x)
    assert out.shape == (4, 8)
    assert layer.spline_l2_penalty().item() >= 0


def test_kan_encoder_forward_and_penalty():
    enc = KANEncoder(widths=(32, 16, 8), grid_size=5)
    x = torch.randn(3, 32)
    assert enc(x).shape == (3, 8)
    assert enc.curvature_penalty().item() >= 0


def test_ppo_update_runs():
    ppo = PPO(PPOConfig(epochs=1), state_dim=272, action_dim=3)
    s = torch.randn(10, 272)
    a, logp, v = ppo.net.act(s)
    batch = {
        "states": s, "actions": a.detach(), "logps": logp.detach(),
        "advantages": torch.randn(10), "returns": torch.randn(10),
    }
    stats = ppo.update(batch)
    assert "policy_loss" in stats


def test_state_dim():
    from kecrl.env import build_state
    emb = torch.randn(256); stats = torch.randn(12)
    s = build_state(emb, stats, nu=0.01, lam=1e-2, t=5, device="cpu")
    assert s.numel() == 272


def test_loss_landscape_shapes():
    model = torch.nn.Sequential(torch.nn.Flatten(),
                                torch.nn.Linear(3 * 8 * 8, 2))
    x = torch.rand(4, 3, 8, 8); y = torch.randint(0, 2, (4,))
    alphas, curve = loss_landscape_1d(model, x, y, n=11, span=0.5)
    assert len(alphas) == len(curve) == 11
