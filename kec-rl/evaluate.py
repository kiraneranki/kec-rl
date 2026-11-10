"""Evaluation: clean/robust accuracy, AUC, and adaptive ASR.

These functions produce the raw numbers that populate Tables 3-5 and the
figures. They do NOT hard-code any result; every value comes from the model
you pass in, evaluated on the data you pass in.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .attacks.harness import pgd_attack, ensemble_asr


@torch.no_grad()
def clean_accuracy(model, loader, device) -> float:
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def robust_accuracy(model, loader, device, eps=0.03, steps=20,
                    restarts=1, eot_samples=10) -> float:
    """Robust accuracy under l_inf PGD at budget eps."""
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd_attack(model, x, y, eps=eps, steps=steps,
                           restarts=restarts, eot_samples=eot_samples)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


@torch.no_grad()
def auc_score(model, loader, device) -> float:
    """ROC-AUC over the test set (fake = positive)."""
    scores, labels = [], []
    for x, y in loader:
        x = x.to(device)
        p = model(x).softmax(1)[:, 1]
        scores.append(p.cpu()); labels.append(y)
    scores = torch.cat(scores); labels = torch.cat(labels)
    # rank-based AUC (Mann-Whitney U)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float)
    auc = (ranks[labels == 1].sum() - pos.numel() * (pos.numel() + 1) / 2) \
        / (pos.numel() * neg.numel())
    return float(auc)


def epsilon_sweep(model, loader, device, budgets=(0.01, 0.03, 0.06)) -> dict:
    return {eps: robust_accuracy(model, loader, device, eps=eps) for eps in budgets}


def adaptive_asr_table(model, x_fake, y_fake) -> dict:
    """Per-attack-type and ensemble ASR with the paper's attack configs."""
    configs = {
        "pgd": dict(eps=0.03, steps=100, restarts=10, eot_samples=30),
        "cw": dict(kappa=5.0, lr=5e-3, steps=1000, search_steps=9, eot_samples=30),
        "boundary": dict(queries=10000),
    }
    return ensemble_asr(model, x_fake, y_fake, configs)
