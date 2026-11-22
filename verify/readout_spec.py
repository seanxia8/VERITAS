"""Runbook §1 readouts, implemented from the text of §1 (r2-tier1v).

(a) all-mean, (c) spatial-moment latent [m1, m2, m3, vec(C)] (768), (d) raw baseline (33),
f_event inputs, and the Ridge protocol (alpha in {1e-6, 1e-4, 1e-2, 1, 100}, chosen per target
coordinate by 5-fold CV with folds grouped by run; features standardised on the training split).
"""
from __future__ import annotations

import numpy as np
import torch

ALPHAS = np.array([1e-6, 1e-4, 1e-2, 1.0, 100.0])


def _masked_softmax(logits, mask):
    neg = torch.finfo(logits.dtype).min
    l = torch.where(mask, logits, torch.full_like(logits, neg))
    m = l.max(1, keepdim=True).values
    w = torch.where(mask, torch.exp(l - m), torch.zeros_like(l))
    return w / w.sum(1, keepdim=True).clamp_min(1e-300)


def weights(f):
    """f: (B, N, 9) float64 tensor -> (wh, wq, we, H) each (B, N); §1 masks and weights."""
    hit = f[..., 8] > 0.5
    empty = ~hit.any(1, keepdim=True)
    H = hit | empty                                  # all tokens if H is empty
    ell, tau = f[..., 6], f[..., 7]
    wh = H.double() / H.double().sum(1, keepdim=True)
    wq = _masked_softmax(3.0 * ell, H)
    P = H & (tau >= 0)
    inf = torch.full_like(tau, float("inf"))
    tmin = torch.where(P, tau, inf).min(1, keepdim=True).values
    tmin = torch.where(torch.isfinite(tmin), tmin, torch.zeros_like(tmin))
    we = _masked_softmax(3.0 * ell - 4.0 * (tau - tmin), P)
    we = torch.where(P.any(1, keepdim=True), we, wq)  # P empty -> m3 = m2
    return wh, wq, we, H, P


def spatial_moments(tok, f):
    """(c) construction on per-token vectors tok (B, N, D): (B, 3D + 3D) = [m1, m2, m3, vec(C)]."""
    wh, wq, we, H, _ = weights(f)
    r = f[..., 0:3]
    rbar = (wh[..., None] * r).sum(1)                                   # (B, 3)
    m1 = torch.einsum("bn,bnd->bd", wh, tok)
    m2 = torch.einsum("bn,bnd->bd", wq, tok)
    m3 = torch.einsum("bn,bnd->bd", we, tok)
    C = torch.einsum("bn,bnc,bnd->bcd", wh, r - rbar[:, None], tok)     # (1/|H|) sum_H (r - rbar) z^T
    return torch.cat([m1, m2, m3, C.reshape(len(f), -1)], 1), (wh, wq, we, H, rbar)


def raw_baseline(f):
    """(d): (c) on (ell, tau, h) + 15 summaries -> (B, 33)."""
    tok = f[..., 6:9]
    base, (wh, wq, we, H, rbar) = spatial_moments(tok, f)
    hit = f[..., 8] > 0.5
    nh = hit.double().sum(1)
    N = f.shape[1]
    _, _, _, _, P = weights(f)

    def ms(v, m):
        c = m.double().sum(1).clamp_min(1)
        mu = (v * m).sum(1) / c
        return [mu, torch.sqrt(((v - mu[:, None]) ** 2 * m).sum(1) / c)]
    r = f[..., 0:3]
    s = [nh / N, torch.log(nh.clamp_min(1))] + ms(f[..., 6], H.double()) + ms(f[..., 7], P.double())
    summ = torch.cat([torch.stack(s, 1), rbar, torch.einsum("bn,bnc->bc", wq, r),
                      torch.einsum("bn,bnc->bc", we, r)], 1)
    return torch.cat([base, summ], 1)


# ------------------------------------------------------------------ Ridge protocol
def _path(X, Y, alphas):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y
    return np.stack([Vt.T @ ((s / (s * s + a))[:, None] * UtY) for a in alphas])  # (A, D, T)


def fold_ids(groups, k=5):
    ug = np.unique(groups)
    f = {g: i % k for i, g in enumerate(ug)}
    return np.array([f[g] for g in groups])


def ridge_fit_predict(Xtr, Ytr, Xtests, groups, alphas=ALPHAS):
    """Standardise on train; per target column choose alpha by 5-fold run-grouped CV (SSE);
    refit on all train with intercept.  Returns list of predictions for Xtests and the alphas."""
    Ytr = np.asarray(Ytr, float).reshape(len(Ytr), -1)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    Z = (Xtr - mu) / sd
    fold = fold_ids(groups)
    sse = np.zeros((len(alphas), Ytr.shape[1]))
    for k in range(5):
        tr, va = fold != k, fold == k
        xm, ym = Z[tr].mean(0), Ytr[tr].mean(0)
        B = _path(Z[tr] - xm, Ytr[tr] - ym, alphas)
        pred = np.einsum("nd,adt->ant", Z[va] - xm, B) + ym
        sse += ((pred - Ytr[va]) ** 2).sum(1)
    best = sse.argmin(0)
    xm, ym = Z.mean(0), Ytr.mean(0)
    B = _path(Z - xm, Ytr - ym, alphas)
    coef = np.stack([B[best[t], :, t] for t in range(Ytr.shape[1])], 1)
    preds = [((X - mu) / sd - xm) @ coef + ym for X in Xtests]
    return preds, [float(alphas[b]) for b in best]


def vertex_error(pred, y):
    return float(np.linalg.norm(pred - y, axis=1).mean())
