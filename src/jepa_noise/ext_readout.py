# SPDX-License-Identifier: MIT
"""Q1/Q2 readout extension helpers (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-01_READOUT.md`` (revision 2).
Extension name: ``readout_diagnostic_2026-11-01``.

This module holds the numpy/torch helpers the runbook assigns to
``src/jepa_noise/ext_readout.py``: the five readout forms of §2 on a tokenised
batch, the token sampling, the ridge/MLP probes, the CKA wrapper, the
layout/event fractions of §6 and the §4.2 reproduction block.  It only imports
existing modules (``lucid_arms``, ``lucid_run``, ``point_encoder``) read-only;
no registered file, config, criterion or verdict is touched.

Forms (§2)::

    full_input    arm.jepa.encode(batch.features)
    half_pass     HalfPassEncoder(arm.jepa, within_target_mask(geometry))
    propagated    encode(features, hidden_mask=batch.hidden), masked positions
    context_only  the same pass, context positions
    raw_input     batch.features (no encoder)

``form_representations`` returns ``(event_repr, token_repr, token_mask)``: the
runbook's fourth tuple element (``y_vertex``) has no source inside a tokenised
batch, so the vertex truth is carried by :class:`FormStream` rows instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import lucid_arms as la
from . import lucid_run as lr
from . import point_encoder as pe

EXTENSION = "readout_diagnostic_2026-11-01"
#: §3.5 exact item groups (matrix_a / matrix_b).
GROUP_A = ("init9000", "old9000", "old9001", "fix9000")
GROUP_B = ("fix9001", "mae9000", "across9000", "across9001")
#: §2 forms.
FORMS = ("full_input", "half_pass", "propagated", "context_only", "raw_input")
#: half_pass is within-only (§2: "across is per-event random, so record n/a").
WITHIN_FORMS = ("full_input", "half_pass", "propagated", "context_only", "raw_input")
ACROSS_FORMS = ("full_input", "propagated", "context_only", "raw_input")
#: The runbook's alpha grid for the Part-C-comparable token ridge (§2).
RIDGE_ALPHAS = (1e-4, 1e-2, 1.0, 100.0)
#: §4.2 recorded Part C full-input photon vertex resolutions (m).
REPRODUCTION = {"old9000": 0.264401, "old9001": 0.233759, "fix9000": 0.219258,
                "fix9001": 0.204281, "mae9000": 0.104949, "init9000": 0.160072}
REPRODUCTION_TOLERANCE_M = 1e-3
#: §2 sampling: at most 4 000 events, 32 hit tokens/event, seed 0.
MAX_EVENTS = 4000
TOKENS_PER_EVENT = 32
SAMPLE_SEED = 0
#: §3.5 primary domain (release note: "photon domain only here").
PRIMARY_DOMAIN = "photon"
#: FEATURES[0:6]: pos_x, pos_y, pos_z, dir_x, dir_y, dir_z (point_encoder.py:46-47)
GEOMETRY_COLUMNS = slice(0, 6)
_HALF_MASK_CACHE: dict = {}


# --------------------------------------------------------------------- geometry
def within_target_mask(geometry) -> np.ndarray:
    """The registered fixed ``within`` target set: odd 1-based pmt_id_in_mpmt.

    Same definition as ``scripts/jepa_pregrid_checks.py:within_target_mask`` and
    ``lucid_views.within_mask`` (even = context, odd = target).
    """
    digest = getattr(geometry, "digest", None)
    if digest is not None and digest in _HALF_MASK_CACHE:
        return _HALF_MASK_CACHE[digest].copy()
    mask = (np.asarray(geometry.pmt_id_in_mpmt) % 2) == 1
    if digest is not None:
        _HALF_MASK_CACHE[digest] = mask.copy()
    return mask


class HalfPassEncoder(torch.nn.Module):
    """Encodes each token in the pass it saw during training (within split only).

    ``encode(x)`` with no mask returns context-position outputs from the
    context-only pass and target-position outputs from the target-only pass;
    an explicit ``hidden_mask`` is passed through unchanged (the ``propagated``
    form).  Signature per runbook revision 2: ``HalfPassEncoder(base, target_mask)``.
    """

    def __init__(self, base, target_mask):
        super().__init__()
        self.base = base
        self.register_buffer("target", torch.as_tensor(np.asarray(target_mask, bool)))

    def encode(self, features, hidden_mask=None):
        if hidden_mask is not None:
            return self.base.encode(features, hidden_mask=hidden_mask)
        t = self.target.to(features.device).unsqueeze(0).expand(features.shape[0], -1)
        z_ctx = self.base.encode(features, hidden_mask=t)
        z_tgt = self.base.encode(features, hidden_mask=~t)
        return torch.where(t.unsqueeze(-1), z_tgt, z_ctx)


# --------------------------------------------------------------- runtime hooks
def geometry_blind(features: torch.Tensor) -> torch.Tensor:
    """A clone of ``features`` with the six geometry columns zeroed (§3.1 E2)."""
    if features.size(-1) < GEOMETRY_COLUMNS.stop:
        raise ValueError("geometry_blind needs at least six feature columns")
    out = features.clone()
    out[..., GEOMETRY_COLUMNS] = 0.0
    return out


def install_geometry_blind(model):
    """Wrap ``model.encode`` so every pass sees zero geometry features.

    Returns the original bound method so the caller can restore it.  The wrapper
    never mutates its input: it zeroes a clone (E2 acceptance test §4.3).
    """
    original = model.encode

    def encode(features, hidden_mask=None):
        return original(geometry_blind(features), hidden_mask=hidden_mask)

    model.encode = encode
    return original


def restore_encode(model, original) -> None:
    """Remove a previously installed :func:`install_geometry_blind` wrapper."""
    model.encode = original


# --------------------------------------------------------------- representations
def form_representations(arm, batch, form, *, half_encoder=None):
    """Return ``(event_repr, token_repr, token_mask)`` for one batch and one form.

    ``event_repr``  ``(B, d)`` float32 mean of the form's tokens (§2).
    ``token_repr``  ``(B, N, d)`` float32 token output of the form's own pass.
    ``token_mask``  ``(B, N)`` bool positions the form reads.

    ``full_input``/``half_pass``/``raw_input`` read every PMT token; ``propagated``
    reads the masked/target positions and ``context_only`` the context positions
    of the same context pass (their union is the context-pass output).
    """
    form = str(form)
    if form not in FORMS:
        raise ValueError(f"unknown readout form {form!r}; expected {FORMS}")
    features = batch.features
    ones = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
    with torch.no_grad():
        if form == "full_input":
            token_t = arm.jepa.encode(features)
            mask = ones
        elif form == "half_pass":
            if half_encoder is None:
                raise ValueError("half_pass needs a HalfPassEncoder built with the within mask")
            token_t = half_encoder.encode(features)
            mask = ones
        elif form in ("propagated", "context_only"):
            token_t = arm.jepa.encode(features, hidden_mask=batch.hidden)
            mask = batch.hidden if form == "propagated" else ~batch.hidden
        else:  # raw_input
            token_t = features
            mask = ones
    if token_t is None:
        raise ValueError(f"form {form!r} produced no token output")
    tokens = token_t.detach().to(torch.float32).cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    event = (la._mean_tokens(tokens, None) if form in ("full_input", "half_pass", "raw_input")
             else la._mean_tokens(tokens, mask_np))
    return np.asarray(event, np.float32), tokens, mask_np


# ------------------------------------------------------------------- statistics
def ridge_r2(x_train, y_train, x_test, y_test, alphas=RIDGE_ALPHAS) -> dict:
    """Standardised ridge, alpha chosen on the last 20% of train, scored on test.

    Copy of ``scripts/jepa_pregrid_checks.py:ridge_r2`` (the Part-C-comparable
    token protocol of §2); returns ``{r2, alpha, n_train, n_test}``.
    """
    xtr, xte = np.asarray(x_train, float), np.asarray(x_test, float)
    ytr, yte = np.asarray(y_train, float).ravel(), np.asarray(y_test, float).ravel()
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd == 0] = 1
    ztr, zte = (xtr - mu) / sd, (xte - mu) / sd
    cut = int(0.8 * len(ztr))

    def fit(z, y, a):
        gram = z.T @ z + a * np.eye(z.shape[1])
        return np.linalg.solve(gram, z.T @ (y - y.mean())), y.mean()

    best = None
    for a in alphas:
        coef, off = fit(ztr[:cut], ytr[:cut], a)
        err = float(((ztr[cut:] @ coef + off - ytr[cut:]) ** 2).mean())
        if best is None or err < best[1]:
            best = (a, err)
    coef, off = fit(ztr, ytr, best[0])
    pred = zte @ coef + off
    sst = float(((yte - yte.mean()) ** 2).sum())
    return {"r2": (float(1 - ((yte - pred) ** 2).sum() / sst) if sst > 0 else None),
            "alpha": best[0], "n_train": int(len(ytr)), "n_test": int(len(yte))}


def effective_rank(x: np.ndarray) -> dict:
    """Effective rank and participation ratio (``scripts/jepa_pregrid_checks.py:114-133``)."""
    x = np.asarray(x, float) - np.asarray(x, float).mean(0)
    w = np.clip(np.linalg.eigvalsh(x.T @ x / max(len(x) - 1, 1)), 0, None)
    if w.sum() <= 0:
        return {"effective_rank": 0.0, "participation_ratio": 0.0}
    p = w / w.sum()
    p = p[p > 0]
    return {"effective_rank": float(np.exp(-(p * np.log(p)).sum())),
            "participation_ratio": float(w.sum() ** 2 / (w ** 2).sum())}


def epps_pulley(x: np.ndarray, n_slices: int = 256, seed: int = 0) -> float:
    """Numpy replica of the pinned SIGReg statistic (17 nodes on [0, 3], times N)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(x.shape[1], n_slices))
    dirs /= np.linalg.norm(dirs, axis=0)
    proj = x @ dirs
    t = np.linspace(0, 3, 17)
    dt = 3 / 16
    w = np.full(17, 2 * dt)
    w[[0, -1]] = dt
    phi = np.exp(-.5 * t ** 2)
    w = w * phi
    xt = proj[..., None] * t
    err = (np.cos(xt).mean(0) - phi) ** 2 + np.sin(xt).mean(0) ** 2
    return float((err @ w * len(x)).mean())


def mlp_probe(x_train, y_train, x_test, y_test, *, seed: int = 0, hidden: int = 256,
              epochs: int = 200, patience: int = 10, batch: int = 4096,
              metric: str = "r2") -> dict:
    """2 x 256 MLP probe on standardised inputs (runbook §3.6).

    Early stopping on the last 10 % of the training rows, fixed seed, Adam 1e-3.
    Training is minibatched (``batch`` rows, order drawn from the seeded generator)
    instead of ``lr._fit_mlp``'s single full-batch step: the token tables are up to
    ~1e5 rows and the full-batch activations would exceed the CPU memory budget.
    The architecture, patience, validation split and seed are the runbook's.

    Returns ``{status, r2_mean, r2_per_target, resolution, n_train, n_test, seed}``;
    ``resolution`` is the mean Euclidean error (vertex probe, §Q2-c) and is only
    meaningful for a 3-column target.
    """
    x = np.asarray(x_train, float)
    y = np.asarray(y_train, float)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("mlp_probe needs matching (n, d) features and (n, k) targets")
    if x.shape[0] < 10:
        return {"status": "not_evaluable", "reason": "too few rows", "n_train": int(x.shape[0])}
    centre, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1.0
    y_centre = y.mean(0, keepdims=True)
    torch.manual_seed(int(seed))
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, y.shape[1]))
    xt = torch.as_tensor((x - centre) / scale, dtype=torch.float32)
    yt = torch.as_tensor(y - y_centre, dtype=torch.float32)
    cut = max(1, int(0.9 * x.shape[0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.functional.mse_loss
    generator = torch.Generator().manual_seed(int(seed) + 1)
    best, state, bad, epochs_run = math.inf, None, 0, 0
    for epoch in range(int(epochs)):
        epochs_run = epoch + 1
        model.train()
        order = torch.randperm(cut, generator=generator)
        for start in range(0, cut, int(batch)):
            index = order[start:start + int(batch)]
            optimizer.zero_grad()
            loss_fn(model(xt[index]), yt[index]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(loss_fn(model(xt[cut:]), yt[cut:]))
        if validation < best:
            best, state, bad = validation, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor((np.asarray(x_test, float) - centre) / scale,
                                     dtype=torch.float32)).numpy() + y_centre
    y_test = np.asarray(y_test, float)
    if y_test.ndim == 1:
        y_test = y_test[:, None]
    per_target = []
    for column in range(y_test.shape[1]):
        sst = float(((y_test[:, column] - y_test[:, column].mean()) ** 2).sum())
        per_target.append(float(1 - ((y_test[:, column] - pred[:, column]) ** 2).sum() / sst)
                          if sst > 0 else None)
    finite = [value for value in per_target if value is not None]
    out = {"status": "evaluated", "r2_per_target": per_target,
           "r2_mean": (float(np.mean(finite)) if finite else None),
           "n_train": int(x.shape[0]), "n_test": int(y_test.shape[0]),
           "seed": int(seed), "epochs_run": int(epochs_run)}
    if metric == "resolution":
        errors = np.linalg.norm(pred - y_test, axis=1)
        out["resolution"] = float(errors.mean())
        out["resolution_median"] = float(np.median(errors))
    return out


def cka(x, y) -> float:
    """Linear CKA (``lr.linear_cka``, the only CKA in the repo)."""
    return lr.linear_cka(x, y)


# ------------------------------------------------------- layout/event fractions
def layout_state(n_pmt: int, dim: int) -> dict:
    """Accumulator for the §6 Q1-e/Q1-f fractions (never materialises all tokens)."""
    return {"n_pmt": int(n_pmt), "dim": int(dim),
            "sum_pd": np.zeros((int(n_pmt), int(dim)), np.float64),
            "sum_sq": np.zeros(int(dim), np.float64),
            "sum_me": np.zeros(int(dim), np.float64),
            "sum_me_sq": np.zeros(int(dim), np.float64),
            "n_events": 0, "n_tokens": 0}


def layout_add(state: dict, tokens) -> None:
    """Add one chunk of full-input token outputs ``(E, N, d)`` to the accumulator."""
    z = np.asarray(tokens, np.float64)
    if z.ndim != 3 or z.shape[1:] != (state["n_pmt"], state["dim"]):
        raise ValueError("layout_add: tokens must be (E, N_pmt, d)")
    state["sum_pd"] += z.sum(axis=0)
    state["sum_sq"] += (z * z).sum(axis=(0, 1))
    event_mean = z.mean(axis=1)
    state["sum_me"] += event_mean.sum(axis=0)
    state["sum_me_sq"] += (event_mean * event_mean).sum(axis=0)
    state["n_events"] += int(z.shape[0])
    state["n_tokens"] += int(z.shape[0] * z.shape[1])


def layout_fractions(state: dict) -> dict:
    """The two §6 fractions from :func:`layout_state`.

    ``f_layout`` = per-dimension average of ``Var_p(m_p) / Var_{p,e}(z)`` with
    ``m_p`` the event-mean token output of PMT ``p``; ``f_event`` = same with
    ``Var_e(m_e)`` where ``m_e`` is the PMT-mean (event) embedding.  Dimensions
    with a zero denominator are dropped and counted in ``n_dropped``.
    """
    n_events, n_tokens = int(state["n_events"]), int(state["n_tokens"])
    if n_events < 2 or n_tokens < 2:
        return {"f_layout": None, "f_event": None, "n_dropped": None,
                "reason": "fewer than two rows"}
    m_p = state["sum_pd"] / n_events
    mean_all = state["sum_pd"].sum(axis=0) / n_tokens
    var_pe = (state["sum_sq"] - n_tokens * mean_all * mean_all) / (n_tokens - 1)
    var_p = m_p.var(axis=0, ddof=1)
    m_e = state["sum_me"] / n_events
    var_e = (state["sum_me_sq"] - n_events * m_e * m_e) / (n_events - 1)
    keep = var_pe > 0
    n_dropped = int((~keep).sum())
    if not keep.any():
        return {"f_layout": None, "f_event": None, "n_dropped": n_dropped,
                "reason": "constant representation (§2: report n/a)"}
    return {"f_layout": float(np.mean(var_p[keep] / var_pe[keep])),
            "f_event": float(np.mean(var_e[keep] / var_pe[keep])),
            "n_dropped": n_dropped, "n_events": n_events, "n_tokens": n_tokens,
            "f_layout_per_dim": (var_p[keep] / var_pe[keep]).tolist()[:128],
            "f_event_per_dim": (var_e[keep] / var_pe[keep]).tolist()[:128]}


def layout_and_event_fractions(event_repr, token_repr) -> dict:
    """Array-form of :func:`layout_fractions` (runbook §3.6 signature)."""
    z = np.asarray(token_repr, float)
    if z.ndim != 3:
        raise ValueError("token_repr must be (E, N, d) full-input token outputs")
    state = layout_state(z.shape[1], z.shape[2])
    layout_add(state, z)
    return layout_fractions(state)


# ------------------------------------------------------------------ form streams
@dataclass
class FormStream:
    """One readout form's streamed rows for one probe split."""

    form: str
    events: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    run_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    vertex: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float64))
    log_yield: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    tokens: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    token_y: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    token_run_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    layout: dict | None = None

    @property
    def n_events(self) -> int:
        return int(self.events.shape[0])

    @property
    def n_token_rows(self) -> int:
        return int(self.tokens.shape[0])


def _empty_lists() -> dict:
    return {"events": [], "runs": [], "vertex": [], "log_yield": [],
            "tokens": [], "token_y": [], "token_runs": []}


def _finish_stream(form, acc, layout) -> FormStream:
    def cat(key, dtype=None):
        if not acc[key]:
            return None
        return np.concatenate(acc[key]).astype(dtype) if dtype is not None else np.concatenate(acc[key])
    events = cat("events")
    tokens = cat("tokens")
    return FormStream(
        form=form,
        events=(events if events is not None else np.zeros((0, 0), np.float32)),
        run_ids=(cat("runs") if acc["runs"] else np.zeros(0, dtype=object)),
        vertex=(cat("vertex") if acc["vertex"] else np.zeros((0, 3), np.float64)),
        log_yield=(cat("log_yield") if acc["log_yield"] else np.zeros(0, np.float64)),
        tokens=(tokens if tokens is not None else np.zeros((0, 0), np.float32)),
        token_y=(cat("token_y") if acc["token_y"] else np.zeros(0, np.float64)),
        token_run_ids=(cat("token_runs") if acc["token_runs"] else np.zeros(0, dtype=object)),
        layout=layout)


def _module_gain_accumulators(forms) -> dict:
    out = {}
    if "full_input" in forms or "half_pass" in forms:
        out["run_mean"] = la.ProbeAccumulator("module_gain", "run_mean")
    if "propagated" in forms:
        out["propagated"] = la.ProbeAccumulator("module_gain", "propagated")
    return out


def stream_model_forms(model, probes, forms, geometry, *, split, mask_seed, device,
                       chunk_events: int = 8, max_events: int | None = None,
                       domain: str | None = PRIMARY_DOMAIN, sample_seed: int = SAMPLE_SEED,
                       tokens_per_event: int = TOKENS_PER_EVENT,
                       zero_geometry: bool = False) -> dict:
    """One streaming pass over probe-train/probe-test for several forms at once.

    ``probes`` is ``{"train": Dataset, "test": Dataset}``.  Returns::

        {"train": {form: FormStream}, "test": {form: FormStream},
         "module_gain": {"train": {form: ProbeAccumulator}, "test": {...}}}

    The pass computes each event chunk once per representation: the full pass for
    ``full_input``/``half_pass``/``raw_input`` and the masked pass (shared by
    ``propagated`` and ``context_only``).  ``zero_geometry=True`` zeroes the six
    geometry columns of every probe batch before any form function (H12 E2).
    """
    forms = [str(form) for form in forms]
    for form in forms:
        if form not in FORMS:
            raise ValueError(f"unknown readout form {form!r}; expected {FORMS}")
    scaling = pe.FeatureScaling.from_geometry(geometry)
    module_index = np.asarray(geometry.module_index, np.int64)
    needs_full = any(form in ("full_input", "half_pass") for form in forms)
    needs_context = any(form in ("propagated", "context_only") for form in forms)
    run_mean_form = ("half_pass" if "half_pass" in forms and "full_input" not in forms
                     else ("full_input" if "full_input" in forms else None))
    out = {"train": {}, "test": {},
           "module_gain": {"train": _module_gain_accumulators(forms),
                           "test": _module_gain_accumulators(forms)}}
    accumulators = out["module_gain"]
    acc = {which: {form: _empty_lists() for form in forms} for which in ("train", "test")}
    layout = {which: {form: (layout_state(geometry.n_pmt, 0) if form == "full_input" else None)
                      for form in forms} for which in ("train", "test")}
    for which in ("train", "test"):
        dataset = probes[which]
        rng = np.random.default_rng(sample_seed)
        for row in dataset.runs:
            source = dataset.directory / "runs" / row["run_dir"]
            truth = lr._run_truth(row, dataset, geometry)
            ids = lr._run_event_ids(source)
            if domain not in (None, "all"):
                allowed = lr._domain_event_ids(source, domain)
                if allowed is not None:
                    ids = np.intersect1d(ids, allowed)
            if ids.size == 0:
                continue
            if max_events is not None:
                ids = ids[:int(max_events)]
            partials = {} if run_mean_form is not None else None
            for start in range(0, int(ids.size), int(chunk_events)):
                batch = lr._probe_batch(source, geometry, split=split, mask_seed=mask_seed,
                                        occupancy="natural", scaling=scaling,
                                        event_ids=ids[start:start + int(chunk_events)])
                if batch.event_ids.size == 0:
                    continue
                batch = lr._to_device(batch, device)
                if zero_geometry:
                    batch = _replace_features(batch, geometry_blind(batch.features))
                selected_truth = la._select_truth(truth, np.asarray(batch.event_ids))
                event_ids = np.asarray(batch.event_ids)
                hit = np.asarray(batch.hit_flags, bool)
                log_charge = np.asarray(batch.log_charge, float)
                hidden = np.asarray(batch.hidden.detach().cpu(), bool)
                tokens = context = None
                with torch.no_grad():
                    if needs_full:
                        tokens = model.encode(batch.features).detach().to(torch.float32) \
                            .cpu().numpy()
                    if needs_context:
                        context = model.encode(batch.features, hidden_mask=batch.hidden) \
                            .detach().to(torch.float32).cpu().numpy()
                data = la.RunProbeData(
                    run_id=str(row["run_id"]), event_ids=event_ids, tokens=tokens,
                    context_tokens=context, raw=None, hit=hit, hidden=hidden,
                    module_index=module_index, truth=selected_truth)
                for form in forms:
                    if form == "full_input":
                        token_np, mask = tokens, np.ones_like(hit)
                    elif form == "half_pass":
                        token_np, mask = tokens, np.ones_like(hit)
                    elif form == "propagated":
                        token_np, mask = context, hidden
                    elif form == "context_only":
                        token_np, mask = context, ~hidden
                    else:  # raw_input
                        token_np = batch.features.detach().to(torch.float32).cpu().numpy()
                        mask = np.ones_like(hit)
                    if token_np is None:
                        raise RuntimeError(f"form {form!r} has no representation in this pass")
                    event = (la._mean_tokens(token_np, None)
                             if form in ("full_input", "half_pass", "raw_input")
                             else la._mean_tokens(token_np, mask))
                    bucket = acc[which][form]
                    bucket["events"].append(np.asarray(event, np.float32))
                    bucket["runs"].append(np.full(event.shape[0], row["run_id"], dtype=object))
                    bucket["vertex"].append(np.asarray(selected_truth["vertex"], float))
                    bucket["log_yield"].append(np.asarray(selected_truth["log_yield"], float))
                    if form == "full_input":
                        state = layout[which][form]
                        if state["dim"] == 0:
                            state = layout_state(geometry.n_pmt, token_np.shape[-1])
                            layout[which][form] = state
                        layout_add(state, token_np)
                    if form == "raw_input":
                        continue
                    candidate = hit & mask
                    for event_index in range(candidate.shape[0]):
                        positions = np.flatnonzero(candidate[event_index])
                        if positions.size == 0:
                            continue
                        take = min(int(tokens_per_event), int(positions.size))
                        pick = (rng.choice(positions, size=take, replace=False)
                                if positions.size > take else positions)
                        bucket["tokens"].append(token_np[event_index, pick])
                        bucket["token_y"].append(log_charge[event_index, pick])
                        bucket["token_runs"].append(np.full(take, row["run_id"], dtype=object))
                if partials is not None:
                    part = la.probe_run_mean_partial(data, "module_gain")
                    partials[run_mean_form] = (part if run_mean_form not in partials
                                               else la._merge_run_mean_partial(
                                                   partials[run_mean_form], part))
                if "propagated" in accumulators[which]:
                    features, targets, _key, names = la.probe_rows(data, "module_gain",
                                                                  "propagated")
                    if features.shape[0]:
                        accumulators[which]["propagated"].target_names = names
                        accumulators[which]["propagated"].add(row["run_id"], features, targets)
                del data
            if partials is not None and run_mean_form in partials:
                features, targets, _key, names = la.finalise_run_mean(partials[run_mean_form])
                if features.shape[0]:
                    accumulators[which]["run_mean"].target_names = names
                    accumulators[which]["run_mean"].add(row["run_id"], features, targets)
    for which in ("train", "test"):
        for form in forms:
            state = layout[which][form]
            summary = (layout_fractions(state) if state is not None and state["dim"] else None)
            out[which][form] = _finish_stream(form, acc[which][form], summary)
    return out


def _replace_features(batch, features):
    """A LucidBatch with a substituted feature tensor (same masks/indices/truth)."""
    from dataclasses import replace
    return replace(batch, features=features)


# ------------------------------------------------------------------- form scores
def _probe_dataset(stream: FormStream) -> la.ProbeDataset:
    return la.ProbeDataset(kind="vertex", form=stream.form,
                           x=np.asarray(stream.events, np.float64),
                           y=np.asarray(stream.vertex, np.float64),
                           run_ids=np.asarray(stream.run_ids, dtype=object),
                           target_names=("x", "y", "z"), meta={})


def vertex_resolution(train: FormStream, test: FormStream) -> dict:
    """Part-C physics probe on the form's event-level representation (§2, §3.6)."""
    if train.n_events < 2 or test.n_events < 2:
        return {"status": "not_evaluable", "reason": "too few events",
                "resolution": None, "median": None, "n": int(test.n_events),
                "failure_rate": None, "alphas": None}
    fitted = la.fit_probe(_probe_dataset(train))
    if fitted.get("status") != "evaluated":
        return {"status": "not_evaluable", "reason": "probe fit failed",
                "resolution": None, "median": None, "n": int(test.n_events),
                "failure_rate": None, "alphas": None}
    predictions = la.probe_predictions(fitted, _probe_dataset(test))
    scored = lr._vertex_error(predictions, test.vertex)
    scored["alphas"] = [row["alpha"] for row in fitted["fits"] if row["status"] == "fit"]
    scored["n_train"] = int(train.n_events)
    scored["method"] = "fit_probe/probe_predictions/lr._vertex_error"
    return scored


def token_log_charge_r2(train: FormStream, test: FormStream) -> dict:
    """§2 token ln q R² (``ridge_r2``, seed-0 32-token sampling)."""
    if train.n_token_rows == 0 or test.n_token_rows == 0:
        return {"status": "n/a", "reason": "raw_input has no encoder token rows (§2)",
                "r2": None}
    out = ridge_r2(train.tokens, train.token_y, test.tokens, test.token_y)
    out["status"] = "evaluated"
    return out


def module_gain_r2(module_gain: dict, form: str) -> dict:
    """``module_gain::run_mean``/``::propagated`` R² from the streamed accumulators."""
    key = "run_mean" if form in ("full_input", "half_pass") else (
        "propagated" if form == "propagated" else None)
    if key is None or key not in module_gain["train"]:
        return {"status": "n/a", "value": None,
                "reason": "module gain is n/a outside full_input/half_pass/propagated (§2)"}
    train, test = module_gain["train"][key], module_gain["test"][key]
    if train.n_rows == 0 or test.n_rows == 0:
        return {"status": "n/a", "value": None, "reason": "no module-gain rows"}
    scored = la.probe_r2_streamed(train, test)
    return {"status": scored.get("status"), "value": scored.get("r2_mean"),
            "source": f"module_gain::{key}", "r2": scored.get("r2"),
            "n_train": scored.get("n_train"), "n_test": scored.get("n_test")}


def form_metrics(stream_train: FormStream, stream_test: FormStream, *,
                 module_gain: dict | None = None) -> dict:
    """Every §2 metric for one form."""
    vertex = vertex_resolution(stream_train, stream_test)
    token = token_log_charge_r2(stream_train, stream_test)
    rank = effective_rank(stream_test.events) if stream_test.n_events >= 2 else {}
    sigreg = epps_pulley(stream_test.events) if stream_test.n_events >= 2 else None
    gain = (module_gain_r2(module_gain, stream_test.form) if module_gain is not None
            else {"status": "n/a", "value": None, "reason": "not computed"})
    return {"vertex_resolution_m": vertex.get("resolution"),
            "vertex": vertex,
            "token_log_charge_r2": token,
            "effective_rank": rank.get("effective_rank"),
            "participation_ratio": rank.get("participation_ratio"),
            "event_sigreg": sigreg,
            "module_gain_r2": gain,
            "layout": stream_test.layout if stream_test.form == "full_input" else None,
            "n_events": {"train": stream_train.n_events, "test": stream_test.n_events},
            "n_token_rows": {"train": stream_train.n_token_rows,
                             "test": stream_test.n_token_rows}}


# ------------------------------------------------------------- reproduction §4.2
def reproduction_block(items: dict, *, expected: dict | None = None,
                       tolerance: float = REPRODUCTION_TOLERANCE_M) -> dict:
    """Compare full-input photon vertex resolutions with the recorded Part C values."""
    expected = dict(REPRODUCTION if expected is None else expected)
    rows, ok = {}, True
    for label, target in expected.items():
        row = (items.get(label) or {}).get("forms", {}).get("full_input", {})
        got = row.get("vertex_resolution_m")
        diff = None if got is None else float(abs(got - target))
        passed = bool(got is not None and diff <= float(tolerance))
        ok = ok and passed
        rows[label] = {"expected_m": float(target), "got_m": got, "abs_diff_m": diff,
                       "pass": passed}
    return {"registered": False, "extension": EXTENSION, "check": "§4.2 full-input photon vertex",
            "tolerance_m": float(tolerance), "items": rows, "reproduction_pass": bool(ok)}


# ---------------------------------------------------------------- item loading
def item_cell(config, split: str):
    """The evaluation cell for one item: the dev s=0.05 cell with the item's split."""
    import dataclasses
    cell = lr.dev_cell(config)
    if split == "within":
        return cell
    return dataclasses.replace(cell, cell_id=f"ext_dev_s005_{split}", split=str(split))


def item_arm(item: dict, config, arms_config, geometry, *, device="cpu"):
    """Build one manifest item's arm (random-init rebuild or checkpoint load)."""
    cell = item_cell(config, item.get("split", "within"))
    seed = int(item.get("seed", 9000))
    torch.manual_seed(seed)
    np.random.seed(seed % 2 ** 32)
    arm = lr.build_arm(item["arm"], geometry, arms_config, cell, seed,
                       beta=item.get("beta"))
    if item.get("source"):
        payload = torch.load(item["source"], map_location="cpu", weights_only=False)
        identity = payload.get("identity") or {}
        if identity.get("arm") not in (None, item["arm"]):
            raise RuntimeError(f"{item.get('label')}: checkpoint arm {identity.get('arm')!r} "
                               f"does not match manifest arm {item['arm']!r}")
        arm.load_state_dict(payload["model"])
    arm.eval()
    return arm.to(device)


# --------------------------------------------------------- high-level evaluation
def evaluate_item(item: dict, probes: dict, geometry, config, arms_config, *,
                  device, forms=("full_input", "half_pass"), chunk_events: int = 8,
                  max_events: int | None = None, domain: str | None = PRIMARY_DOMAIN,
                  zero_geometry: bool = False) -> dict:
    """The full Q1/Q2 readout protocol for one item and one set of forms."""
    arm = item_arm(item, config, arms_config, geometry, device=device)
    split = item.get("split", "within")
    requested = [form for form in forms if form in FORMS]
    if split == "across" and "half_pass" in requested:
        requested.remove("half_pass")
    full_forms = [form for form in requested if form != "half_pass"]
    half_forms = [form for form in requested if form == "half_pass"]
    mask_seed = int(config["views"]["mask_seed"])
    results = {"label": item.get("label"), "arm": item["arm"], "split": split,
               "seed": int(item.get("seed", 9000)), "forms": {}}
    if full_forms:
        full = stream_model_forms(arm.jepa, probes, full_forms, geometry, split=split,
                                  mask_seed=mask_seed, device=device,
                                  chunk_events=chunk_events, max_events=max_events,
                                  domain=domain, zero_geometry=zero_geometry)
        for form in full_forms:
            results["forms"][form] = form_metrics(full["train"][form],
                                                  full["test"][form],
                                                  module_gain=full["module_gain"])
    if half_forms:
        half = HalfPassEncoder(arm.jepa, within_target_mask(geometry)).to(device)
        streamed = stream_model_forms(half, probes, ["half_pass"], geometry, split=split,
                                      mask_seed=mask_seed, device=device,
                                      chunk_events=chunk_events, max_events=max_events,
                                      domain=domain, zero_geometry=zero_geometry)
        for form in half_forms:
            results["forms"][form] = form_metrics(streamed["train"][form],
                                                  streamed["test"][form],
                                                  module_gain=streamed["module_gain"])
    for form in requested:
        if form not in results["forms"]:
            results["forms"][form] = {
                "vertex_resolution_m": None, "vertex": {"status": "n/a"},
                "token_log_charge_r2": {"status": "n/a", "reason": "not applicable"},
                "effective_rank": None, "participation_ratio": None,
                "event_sigreg": None, "module_gain_r2": {"status": "n/a", "value": None,
                                                        "reason": "split has no fixed within mask"},
                "n_events": {"train": 0, "test": 0}, "n_token_rows": {"train": 0, "test": 0}}
    return results
