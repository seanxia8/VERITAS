"""Frozen-checkpoint readouts for the November 17 Round-2 repair campaign."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import lucid_arms as la


def _weights(features, active):
    hit = active & (features[..., 8] > 0.5)
    has_hit = hit.any(axis=1, keepdims=True)
    hit_or_active = np.where(has_hit, hit, active)
    prompt = hit & (features[..., 7] >= 0)
    charge = features[..., 6]
    time = features[..., 7]
    first = np.min(time, axis=1, keepdims=True, where=prompt, initial=np.inf)
    delay = time - np.where(np.isfinite(first), first, 0.0)
    log_charge = 3.0 * charge
    log_early = log_charge - 4.0 * delay

    def normalized(log_weight, mask):
        maximum = np.max(log_weight, axis=1, keepdims=True, where=mask, initial=-np.inf)
        maximum = np.where(np.isfinite(maximum), maximum, 0.0)
        weight = np.exp(np.where(mask, log_weight - maximum, -np.inf))
        total = weight.sum(axis=1, keepdims=True)
        return weight / np.where(total > 0.0, total, 1.0)

    charge_weight = normalized(log_charge, hit_or_active)
    uniform_active = normalized(np.zeros_like(charge), active)
    charge_weight = np.where(has_hit, charge_weight, uniform_active)
    early_weight = np.where(prompt.any(axis=1, keepdims=True),
                            normalized(log_early, prompt), charge_weight)
    return hit, hit_or_active, prompt, charge_weight, early_weight


def _weighted_mean(tokens, weights):
    return np.einsum("bnd,bn->bd", tokens, weights)


def pool_tokens(tokens, features, mode, active=None):
    """Pool an aligned batch, returning zero on an empty active view.

    Charge and early-time weighting use the active mean on no-hit events.
    When all hit times are negative, early-time weighting uses the charge mean.
    """
    if mode == "all_mean":
        return la._mean_tokens(tokens, active)
    active = np.ones(features.shape[:2], dtype=bool) if active is None else active
    _, hit_or_active, _, charge_weight, early_weight = _weights(features, active)
    if mode == "hit_mean":
        return la._mean_tokens(tokens, hit_or_active)
    if mode == "charge_weighted":
        return _weighted_mean(tokens, charge_weight)
    if mode == "early_time_weighted":
        return _weighted_mean(tokens, early_weight)
    raise ValueError(f"unknown pooling mode {mode!r}")


def moment_features(tokens, features, active=None):
    """Return three weighted means, centered cross-moments and 15 raw summaries."""
    active = np.ones(features.shape[:2], dtype=bool) if active is None else active
    hit, hit_or_active, prompt, charge_weight, early_weight = _weights(features, active)
    pos = features[..., :3]
    hit_mean = la._mean_tokens(tokens, hit_or_active)
    hit_pos = la._mean_tokens(pos, hit)
    means = np.concatenate(
        [hit_mean, _weighted_mean(tokens, charge_weight), _weighted_mean(tokens, early_weight)],
        axis=1,
    )
    hit_count = hit.sum(axis=1)
    active_count = active.sum(axis=1)
    hit_weights = hit / np.maximum(hit_count, 1)[:, None]
    # Since the hit-weighted position is centered, subtracting the token mean
    # explicitly gives the same covariance and would allocate a B x N x D copy.
    centered_pos = (pos - hit_pos[:, None, :]) * hit_weights[..., None]
    cross = centered_pos.transpose(0, 2, 1) @ tokens
    latent = np.concatenate([means, cross.reshape(len(tokens), -1)], axis=1)

    def moments(values, mask):
        mean = la._mean_tokens(values[..., None], mask)
        variance = la._mean_tokens((values - mean)[:, :, None] ** 2, mask)
        return np.concatenate([mean, np.sqrt(variance)], axis=1)

    summary = np.concatenate(
        [
            (hit_count / np.maximum(active_count, 1))[:, None],
            np.log(np.maximum(hit_count, 1))[:, None],
            moments(features[..., 6], hit), moments(features[..., 7], prompt),
            hit_pos, _weighted_mean(pos, charge_weight), _weighted_mean(pos, early_weight),
        ],
        axis=1,
    )
    return latent, np.concatenate([latent, summary], axis=1)


def fit_ridge(train_x, train_y, groups, test_x, *, form, target_names):
    """Use the existing standardized run-grouped ridge and return predictions."""
    train = la.ProbeDataset(
        kind="repair", form=form, x=train_x, y=train_y, run_ids=groups,
        target_names=target_names, meta={},
    )
    test = la.ProbeDataset(
        kind="repair", form=form, x=test_x, y=np.zeros((len(test_x), len(target_names))),
        run_ids=np.empty(len(test_x), dtype=object), target_names=target_names, meta={},
    )
    fitted = la.fit_probe(train)
    return la.probe_predictions(fitted, test), [
        fit.get("alpha") for fit in fitted["fits"]
    ]


def fit_mlp(train_x, train_y, groups, test_x, *, seed, device, epochs=200, patience=25):
    """Fit the prescribed MLP using a held-out set of complete training runs."""
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    if len(unique) < 2:
        raise ValueError("run-grouped early stopping requires at least two training runs")
    val_runs = unique[:max(1, round(0.1 * len(unique)))]
    validation = np.isin(groups, val_runs)
    mean = train_x[~validation].mean(axis=0)
    scale = train_x[~validation].std(axis=0)
    scale[scale < 1e-9] = 1.0

    def tensor(array):
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    x = tensor((train_x - mean) / scale)
    y = tensor(train_y)
    val_x, val_y = x[validation], y[validation]
    train_indices = np.flatnonzero(~validation)
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(x.shape[-1], 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(64, y.shape[-1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_state = None
    best_score, best_epoch = float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        for rows in np.array_split(rng.permutation(train_indices),
                                   max(1, (len(train_indices) + 127) // 128)):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x[rows]), y[rows])
            loss.backward()
            optimizer.step()
        schedule.step()
        model.eval()
        with torch.no_grad():
            score = float(torch.linalg.vector_norm(model(val_x) - val_y, dim=-1).mean())
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("MLP training did not produce a finite validation score")
    model.load_state_dict(best_state)
    with torch.no_grad():
        prediction = model(tensor((test_x - mean) / scale)).cpu().numpy()
    return prediction, {"best_epoch": best_epoch, "best_validation_m": best_score,
                        "validation_runs": val_runs.tolist()}
