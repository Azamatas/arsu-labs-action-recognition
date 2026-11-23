"""Training loops shared by every protocol and model.

Models take a tuple of input tensors (e.g. (x, stats) for the FIN classifier, (x,) for TinyHAR) and return logits.
A model may define penalty() -> scalar added to the loss (L2 of the Ignatov CNN). Adam uses eps = 1e-7 as Keras did.
All training data is moved to the GPU once; batches are drawn with a numpy generator seeded per run.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from . import targets as tg

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def tensors(*arrays):
    return tuple(torch.as_tensor(np.ascontiguousarray(a), device=DEV) for a in arrays)


def adam(model, lr):
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, eps=1e-7)


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


@torch.no_grad()
def predict(model, inputs, batch=4096):
    """Arg-max predictions (numpy) for a tuple of input tensors, dropout off."""
    was = model.training
    model.eval()
    out = torch.cat([model(*(a[i:i + batch] for a in inputs)).argmax(1) for i in range(0, len(inputs[0]), batch)])
    model.train(was)
    return out.cpu().numpy()


def _set_mode(model, frozen):
    model.train()
    for m in frozen:
        m.eval()


def pretrain_fin(fin, axes, max_lag, seed, epochs=60, lr=1e-3, batch=256):
    """Imitation pretraining: the FIN learns the 3 z-scored targets of single-axis windows (n, T). Returns the MSE of
    the last epoch (mean over its batches)."""
    y = tg.imitated(axes, max_lag)
    y = ((y - y.mean(0)) / y.std(0)).astype(np.float32)
    x, y = tensors(axes.astype(np.float32), y)
    fin.to(DEV).train()
    opt = adam(fin, lr)
    rng = np.random.default_rng(seed)
    loss = None
    for _ in range(epochs):
        losses = []
        for idx in np.array_split(rng.permutation(len(x)), int(np.ceil(len(x) / batch))):
            idx = torch.as_tensor(idx, device=DEV)
            opt.zero_grad()
            mse = nn.functional.mse_loss(fin(x[idx]), y[idx])
            mse.backward()
            opt.step()
            losses.append(mse.item() * len(idx))
        loss = sum(losses) / len(x)
    return loss


def fit_epochs(model, inputs, y, epochs, lr, seed, batch=256, evals=None, select=None, frozen=()):
    """Mini-batch training for a number of epochs (reshuffled every epoch).

    evals: {name: (inputs, y)} scored after every epoch (accuracy curves). select: the name in evals whose best
    epoch is kept as 'best' (its predictions on every eval set are returned); the last epoch is always returned.
    frozen: sub-modules kept in eval mode (a frozen FIN)."""
    evals = evals or {}
    y = torch.as_tensor(y, device=DEV, dtype=torch.long)
    opt = adam(model, lr)
    rng = np.random.default_rng(seed)
    curves = {name: [] for name in evals}
    best = {"epoch": 0, "score": -1.0, "state": None}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        _set_mode(model, frozen)
        for idx in np.array_split(rng.permutation(len(y)), int(np.ceil(len(y) / batch))):
            idx = torch.as_tensor(idx, device=DEV)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(*(a[idx] for a in inputs)), y[idx])
            if hasattr(model, "penalty"):
                loss = loss + model.penalty()
            loss.backward()
            opt.step()
        for name, (xs, ys) in evals.items():
            curves[name].append(float(np.mean(predict(model, xs) == ys)))
        if select and curves[select][-1] > best["score"]:
            best = {"epoch": epoch, "score": curves[select][-1],
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    out = {"curves": curves, "last": {name: predict(model, xs) for name, (xs, _) in evals.items()},
           "seconds": time.time() - t0}
    if select:
        model.load_state_dict(best["state"])
        out["best"] = {name: predict(model, xs) for name, (xs, _) in evals.items()}
        out["best_epoch"] = best["epoch"]
    return out


def fit_steps(model, domains, steps, lr, seed, per_domain, val, eval_every=100, frozen=()):
    """Domain-balanced training: every step draws per_domain windows (with replacement) from each source domain.
    domains: list of (inputs, y tensor). val: (inputs, y numpy); the weights with the best validation accuracy
    (checked every eval_every steps) are restored at the end. Returns (best validation accuracy, best step)."""
    opt = adam(model, lr)
    rng = np.random.default_rng(seed)
    best = (-1.0, None, 0)
    _set_mode(model, frozen)
    for step in range(1, steps + 1):
        idx = [torch.as_tensor(rng.integers(0, len(y), per_domain), device=DEV) for _, y in domains]
        xb = tuple(torch.cat([d[0][k][i] for d, i in zip(domains, idx)]) for k in range(len(domains[0][0])))
        yb = torch.cat([d[1][i] for d, i in zip(domains, idx)])
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(*xb), yb)
        if hasattr(model, "penalty"):
            loss = loss + model.penalty()
        loss.backward()
        opt.step()
        if step % eval_every == 0:
            acc = float(np.mean(predict(model, val[0]) == val[1]))
            _set_mode(model, frozen)
            if acc > best[0]:
                best = (acc, {k: v.detach().clone() for k, v in model.state_dict().items()}, step)
    model.load_state_dict(best[1])
    return best[0], best[2]
