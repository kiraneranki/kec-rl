"""Adaptive-attack harness: PGD, C&W, and a decision-based boundary attack.

All gradient attacks support expectation-over-transformation (EoT) with
`eot_samples` draws per step, so the attacker is not fooled by the defender's
inference-time randomisation (closes the gradient-masking concern).

The attacker is assumed to have white-box access to the detector (encoder +
head) but NOT to the RL policy parameters; it sees only the deployed
randomisation level.

Attack-success rate (ASR) is defined (see `attack_success_rate`) as the
fraction of initially-correctly-detected fakes that are flipped to "real".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _eot_grad(model, x, y, eot_samples):
    """Mean loss gradient over EoT draws (defeats stochastic pre-processing)."""
    g = torch.zeros_like(x)
    for _ in range(max(1, eot_samples)):
        x_ = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_), y)
        g = g + torch.autograd.grad(loss, x_)[0]
    return g / max(1, eot_samples)


def pgd_attack(model, x, y, eps=0.03, steps=100, step_size=None,
               restarts=10, eot_samples=30):
    """l_inf PGD with random restarts and EoT. Returns best adversarial x."""
    step_size = step_size or eps / 10.0
    best = x.clone()
    best_loss = torch.full((x.size(0),), -1e9, device=x.device)
    for _ in range(restarts):
        delta = (torch.rand_like(x) * 2 - 1) * eps
        x_adv = torch.clamp(x + delta, 0, 1)
        for _ in range(steps):
            g = _eot_grad(model, x_adv, y, eot_samples)
            x_adv = x_adv + step_size * g.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.clamp(x_adv, 0, 1).detach()
        with torch.no_grad():
            loss = F.cross_entropy(model(x_adv), y, reduction="none")
        improve = loss > best_loss
        best[improve] = x_adv[improve]
        best_loss[improve] = loss[improve]
    return best.detach()


def cw_attack(model, x, y, c=1.0, kappa=5.0, lr=5e-3, steps=1000,
              search_steps=9, eot_samples=30):
    """l_2 Carlini-Wagner attack (untargeted) with EoT and binary search on c."""
    x = x.clone().detach()
    lo = torch.zeros(x.size(0), device=x.device)
    hi = torch.full((x.size(0),), 1e4, device=x.device)
    c_t = torch.full((x.size(0),), c, device=x.device)
    best = x.clone()
    best_l2 = torch.full((x.size(0),), 1e9, device=x.device)

    # change of variables to keep x in [0,1]
    def to_w(z):
        z = torch.clamp(z, 1e-6, 1 - 1e-6)
        return torch.atanh(2 * z - 1)

    for _ in range(search_steps):
        w = to_w(x).detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(steps):
            x_adv = (torch.tanh(w) + 1) / 2
            logits = torch.zeros(x.size(0), 2, device=x.device)
            for _ in range(max(1, eot_samples // 6)):  # lighter EoT inside CW
                logits = logits + model(x_adv)
            logits = logits / max(1, eot_samples // 6)
            real = logits.gather(1, y.view(-1, 1)).squeeze(1)
            other = logits.scatter(1, y.view(-1, 1), -1e9).max(1).values
            f = torch.clamp(real - other + kappa, min=0)
            l2 = ((x_adv - x) ** 2).flatten(1).sum(1)
            loss = (l2 + c_t * f).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pred = model(x_adv).argmax(1)
                success = pred != y
                improved = success & (l2 < best_l2)
                best[improved] = x_adv[improved].detach()
                best_l2[improved] = l2[improved].detach()
        # binary search on c
        with torch.no_grad():
            succ = (model(best).argmax(1) != y)
            hi = torch.where(succ, c_t, hi)
            lo = torch.where(~succ, c_t, lo)
            c_t = torch.where(hi < 1e4, (lo + hi) / 2, c_t * 10)
    return best.detach()


def boundary_attack(model, x, y, queries=10000, eot_samples=1):
    """Decision-based boundary attack (gradient-free). Returns best adversarial x.

    Simplified reference version: random-walk toward the boundary while staying
    adversarial, shrinking step size over the query budget.
    """
    x_adv = torch.rand_like(x)  # start from a clearly adversarial point
    with torch.no_grad():
        for _ in range(200):  # ensure start is adversarial
            if (model(x_adv).argmax(1) != y).all():
                break
            x_adv = torch.rand_like(x)
    step = 0.1
    with torch.no_grad():
        for q in range(queries):
            perturb = torch.randn_like(x) * step
            cand = torch.clamp(x_adv + perturb, 0, 1)
            # move candidate toward the clean image
            cand = cand + 0.1 * (x - cand)
            still_adv = model(cand).argmax(1) != y
            closer = ((cand - x) ** 2).flatten(1).sum(1) < ((x_adv - x) ** 2).flatten(1).sum(1)
            take = still_adv & closer
            x_adv[take] = cand[take]
            if q % 1000 == 0:
                step *= 0.9
    return x_adv.detach()


ATTACKS = {"pgd": pgd_attack, "cw": cw_attack, "boundary": boundary_attack}


def attack_success_rate(model, x_fake, y_fake, attack="pgd", **kwargs):
    """ASR = fraction of initially-correct fakes flipped to 'real' (label 0).

    `y_fake` must be all-ones (fake). Only inputs the model initially gets right
    count toward the denominator.
    """
    with torch.no_grad():
        initially_correct = model(x_fake).argmax(1) == y_fake
    if initially_correct.sum() == 0:
        return 0.0
    xs, ys = x_fake[initially_correct], y_fake[initially_correct]
    x_adv = ATTACKS[attack](model, xs, ys, **kwargs)
    with torch.no_grad():
        flipped = model(x_adv).argmax(1) != ys
    return float(flipped.float().mean())


def ensemble_asr(model, x_fake, y_fake, configs: dict) -> dict:
    """Per-attack ASR plus the ensemble (strongest per-sample) ASR."""
    with torch.no_grad():
        initially_correct = model(x_fake).argmax(1) == y_fake
    xs, ys = x_fake[initially_correct], y_fake[initially_correct]
    if xs.size(0) == 0:
        return {"ensemble": 0.0}
    flipped_any = torch.zeros(xs.size(0), dtype=torch.bool, device=xs.device)
    out = {}
    for name, kw in configs.items():
        x_adv = ATTACKS[name](model, xs, ys, **kw)
        with torch.no_grad():
            flipped = model(x_adv).argmax(1) != ys
        out[name] = float(flipped.float().mean())
        flipped_any |= flipped
    out["ensemble"] = float(flipped_any.float().mean())
    return out
