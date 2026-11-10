"""KEC-RL training loop (Algorithm 1 in the manuscript).

Alternates:
  * detector/encoder update minimising the robust objective + curvature penalty,
  * PPO controller update maximising the Markov-game return,
  * EC commitment of each verdict (auxiliary; does not affect accuracy).

Run via scripts/train.py, which parses configs and datasets. This module holds
the reusable training step so it can be unit-tested on synthetic data.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .env import RewardWeights, build_state, map_action, step_reward
from .models.ppo_controller import PPO, PPOConfig
from .crypto.ec_commit import ProvenanceChain, SigningKey


@dataclass
class TrainConfig:
    episodes: int = 500
    steps_per_episode: int = 512
    eps: float = 0.03
    lambda_curv: float = 1e-2
    lr: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


def _digest(z: torch.Tensor) -> bytes:
    return hashlib.sha256(z.detach().cpu().numpy().tobytes()).digest()


def train_kecrl(model, data_iter, cfg: TrainConfig,
                weights: RewardWeights | None = None, log=print):
    """Train KEC-RL. `data_iter` yields (x, y) batches on the right device.

    Returns the trained model, the PPO controller, and the provenance chain.
    """
    torch.manual_seed(cfg.seed)
    weights = weights or RewardWeights()
    device = cfg.device
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ppo = PPO(PPOConfig(), state_dim=272, action_dim=3, device=device)
    chain = ProvenanceChain(SigningKey.generate())

    attack_stats = torch.zeros(13, device=device)  # 13-d; see manuscript Sec. 6.3
    returns_hist = []

    for ep in range(cfg.episodes):
        states, actions, logps, rewards, values, dones = [], [], [], [], [], []
        ep_return = 0.0

        for t in range(cfg.steps_per_episode):
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                emb = model.embedding(x).mean(0)
            s = build_state(emb, attack_stats, model.nu, cfg.lambda_curv,
                            ep, device)
            a, logp, value = ppo.net.act(s.unsqueeze(0))
            a = a.squeeze(0)

            r, info, x_adv, mix = step_reward(model, x, y, a, weights, cfg.eps)

            # ---- detector/encoder update: robust objective + curvature penalty
            model.train()
            logits_clean = model(x)
            logits_adv = model(x_adv.detach())
            loss = (mix * F.cross_entropy(logits_adv, y)
                    + (1 - mix) * F.cross_entropy(logits_clean, y)
                    + cfg.lambda_curv * model.encoder.curvature_penalty())
            opt.zero_grad(); loss.backward(); opt.step()
            model.eval()

            # ---- EC commitment of the verdict (auxiliary)
            with torch.no_grad():
                z = model.embedding(x).mean(0)
                verdict = int(model(x).argmax(1).float().mean().round())
                score = float(model(x).softmax(1)[:, 1].mean())
            entry = chain.commit_verdict(_digest(z), verdict, score)
            chain_ok = chain.verify()

            # update running attack statistics (12-d)
            attack_stats = 0.95 * attack_stats
            attack_stats[0] = info["robust_loss"]
            attack_stats[1] = 1.0 - info["clean_acc"]

            states.append(s); actions.append(a.detach())
            logps.append(logp.detach()); rewards.append(r)
            values.append(float(value)); dones.append(0.0)
            ep_return += r

        # ---- PPO update at episode end
        with torch.no_grad():
            last_value = float(ppo.net(states[-1].unsqueeze(0))[2])
        adv, rets = ppo.compute_gae(rewards, values, dones, last_value)
        batch = {
            "states": torch.stack(states),
            "actions": torch.stack(actions),
            "logps": torch.stack(logps),
            "advantages": torch.tensor(adv, device=device),
            "returns": torch.tensor(rets, device=device),
        }
        ppo.update(batch)
        returns_hist.append(ep_return / cfg.steps_per_episode)
        if ep % 10 == 0:
            log(f"[ep {ep:4d}] return={returns_hist[-1]:+.3f} "
                f"nu={model.nu:.3f} chain_ok={chain_ok}")

    return model, ppo, chain, returns_hist
