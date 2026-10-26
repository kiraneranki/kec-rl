"""Markov-game environment for the KEC-RL defender.

The defender observes a 272-d state, selects a 3-d action (robust-loss mixing,
randomisation level nu, commitment frequency), and receives the reward of
manuscript Eq. (11):

    r_t = a1 * accuracy
        - a2 * robust_loss
        + a3 * commitment_valid
        - a4 * cost

The attacker plays an inner best-response (PGD) each step. This file defines the
reward and the state featurisation; the optimisation loop lives in train.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import torch
import torch.nn.functional as F

from .attacks.harness import pgd_attack


@dataclass
class RewardWeights:
    a1: float = 1.0   # accuracy
    a2: float = 1.0   # robustness
    a3: float = 0.2   # integrity (no effect on accuracy by construction)
    a4: float = 0.05  # latency / compute cost


def build_state(embedding, attack_stats, nu, lam, t, device):
    """Assemble the 272-d state vector.

    embedding:     (256,) pooled KAN embedding (batch-averaged)
    attack_stats:  (12,) running attack statistics
    nu, lam:       scalars; t: episode index
    """
    pieces = [
        embedding.detach().flatten()[:256],
        attack_stats.detach().flatten()[:12],
        torch.tensor([nu, lam, math.log1p(t)], device=device),
    ]
    s = torch.cat(pieces)
    assert s.numel() == 272, f"state dim {s.numel()} != 272"
    return s


def map_action(a_raw):
    """Map tanh-squashed action in (-1,1)^3 to environment ranges.

    returns (mix, nu, commit_freq)
      mix        in [0,1]   -- clean/adversarial loss mixing coefficient
      nu         in [0,0.05]-- inference randomisation level
      commit     in [0,1]   -- soft commitment frequency
    """
    mix = (a_raw[0] + 1) / 2
    nu = (a_raw[1] + 1) / 2 * 0.05
    commit = (a_raw[2] + 1) / 2
    return float(mix), float(nu), float(commit)


def step_reward(model, x, y, action, weights: RewardWeights, eps=0.03,
                chain_ok: bool = True):
    """Run one environment step; return (reward, info)."""
    mix, nu, commit = map_action(action)
    model.set_randomisation(nu)

    # defender's current accuracy on a clean batch
    with torch.no_grad():
        clean_acc = (model(x).argmax(1) == y).float().mean()

    # attacker inner best-response (cheap PGD for the training inner loop)
    x_adv = pgd_attack(model, x, y, eps=eps, steps=10, restarts=1, eot_samples=4)
    robust_loss = F.cross_entropy(model(x_adv), y)

    cost = commit * 0.7e-3 + 1.9e-3  # ms-scale compute proxy (KAN + commit)

    r = (weights.a1 * float(clean_acc)
         - weights.a2 * float(robust_loss)
         + weights.a3 * (1.0 if chain_ok else 0.0)
         - weights.a4 * cost)
    info = {"clean_acc": float(clean_acc), "robust_loss": float(robust_loss),
            "nu": nu, "mix": mix, "commit": commit}
    return r, info, x_adv, mix
