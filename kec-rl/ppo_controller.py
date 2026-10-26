"""PPO controller that treats attacker-defender interaction as a Markov game.

State  s_t in R^272 (see build_state in kecrl/train.py):
    256-d pooled KAN embedding + 12-d attack statistics + nu + lambda + log(1+t)
Action a_t in R^3 (squashed to ranges in the environment):
    (a) robust-loss mixing coefficient
    (b) randomisation level nu in [0, 0.05]
    (c) soft commitment-frequency parameter

This module implements the policy/value networks and the clipped-surrogate
update of Schulman et al. (2017), with GAE (Schulman et al., 2016).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 272, action_dim: int = 3, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.value = nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor):
        h = self.body(s)
        mu = torch.tanh(self.mu(h))  # in (-1, 1); env rescales to ranges
        std = self.log_std.exp().expand_as(mu)
        return mu, std, self.value(h).squeeze(-1)

    def act(self, s: torch.Tensor):
        mu, std, value = self(s)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), value

    def evaluate(self, s: torch.Tensor, a: torch.Tensor):
        mu, std, value = self(s)
        dist = torch.distributions.Normal(mu, std)
        return dist.log_prob(a).sum(-1), dist.entropy().sum(-1), value


@dataclass
class PPOConfig:
    clip: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class PPO:
    def __init__(self, cfg: PPOConfig, state_dim: int = 272, action_dim: int = 3,
                 device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self.net = ActorCritic(state_dim, action_dim).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    def compute_gae(self, rewards, values, dones, last_value):
        adv, gae = [], 0.0
        values = values + [last_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones[t]) * gae
            adv.insert(0, gae)
        returns = [a + v for a, v in zip(adv, values[:-1])]
        return adv, returns

    def update(self, batch: dict) -> dict:
        s = batch["states"].to(self.device)
        a = batch["actions"].to(self.device)
        old_logp = batch["logps"].to(self.device)
        adv = batch["advantages"].to(self.device)
        ret = batch["returns"].to(self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        stats = {}
        for _ in range(self.cfg.epochs):
            logp, entropy, value = self.net.evaluate(s, a)
            ratio = (logp - old_logp).exp()
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(value, ret)
            loss = (policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy.mean())
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
            self.opt.step()
            stats = {"policy_loss": float(policy_loss),
                     "value_loss": float(value_loss),
                     "entropy": float(entropy.mean())}
        return stats
