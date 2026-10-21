# Issue updates for Junjie — ready to paste (21 October 2026)

**Status: prepared, not posted.** Post in this order: #5 → #4 → #3 → #6.
Everything below is from the local repo state after the 21 Oct merge of
`codex/junjie-pmt-response` into `JEPA` (PMT docs now current; `src/` and
`configs/` unchanged by that merge) and after local `main` was fast-forwarded to
`dev`.

Numbers referenced here are development evidence on synthetic response values,
not calibrated WCTE results. Do not re-post the old draft
[PMT_ISSUE5_REPLY_DRAFT.md](PMT_ISSUE5_REPLY_DRAFT.md): it is superseded by the
text below (1,300+ words, stale in places, and it re-asks decisions Junjie has
already answered).

---

## 1. Issue #5 — reply to Junjie (paste as a new comment)

https://github.com/seanxia8/VERITAS/issues/5

```markdown
Hi Junjie,

Licence first: you were right — the missing LICENSE file never blocked running LUCiD here, and the research-use clarification is enough. We have since run the pinned LUCiD (eaf2e89) prompt transport end to end on WCTE; licence-file/redistribution is only a release question.

**What changed, briefly.** Wavelength-dependent QE (QE(λ)·CE; the exact QE×2/CE÷2 control is bit-identical, so no QE/CE/gain identifiability claim); a fixed 0.25 PE threshold with training-time walk calibration (timing-only cells shift digit times, not charge); a synthetic module-level B-field gain+width proxy; 105 mPMT module groups, not crates; prompt-only uniform-AA transport with a source-truth local emission-angle closure (median 0.46°, no reconstructed cone/Eloss); thesis used for stage conventions and the Eq. 6.2/6.3/5.13 caveats. No claim experiment has run — all synthetic development; vertex still fails on tracks (0.81–0.88 × constant vs 0.8).

**What I need, with the detail that matters:**

1. **Energy endpoint (D-E).** It blocks K_energy, the energy gates and WP7. Default (A): deposited energy in the instrumented volume, primary endpoint on contained μ⁻ at 100–300 MeV kinetic. Basis: straight-track range containment is 100% up to 250 MeV, 94% at 300, 69% at 350, 35% at 400 (multiple scattering ignored, so a lower bound). On escaping tracks the endpoint is essentially path length inside the volume (~340–370 MeV whatever the energy), so we keep it diagnostic only. Agree with (A)? Or (B) drop energy and keep vertex+direction, or (C) another endpoint? If (A), is the primary muon's kinetic loss the right truth, or should secondaries' deposits be included?

2. **Target QE/CE and dark rate (D-QE).** QE, CE and gain are degenerate in situ, so we need the **effective QE·CE product**, not separate curves — at least 3 wavelength points over roughly 200–800 nm. Current values are synthetic (5 anchors, peak 0.5 at 400 nm; CE=0.8; dark 1 kHz). Please send the effective product, the WCTE 3-inch dark rate, and whether it depends on temperature or channel. Separately: the original per-PMT QE spread is **not** implemented (one global curve). Should the baseline include it (a training-time spread would be part of the reference, not N)? If yes, give the sigma/distribution; if not, it stays 0.0 and out of the reference.

3. **Walk form and per-channel Q–T (D-TQ).** At fixed threshold we use walk(q)=a+b/√q, a=0, b=2 ns, q_sat=50 PE (4 ns at 0.25 PE, 0.5 ns at 16 PE). This departs from your "polynomial approximation": the old polynomial was nearly flat over 0.25–20 PE, which is why I changed it. Do you accept 1/√q, or prefer a polynomial / an Eq. 5.13-type curve? If Eq. 5.13, please give the fit with its charge-coordinate conversion. Also, the reference has one walk curve for all channels; do you want a per-channel Q–T spread (what sigma) or measured per-channel curves? That turns the cell from a global slope shift into per-channel time shifts.

4. **Crate map (D-CR).** We group by the 105 mPMT modules from the pinned geometry, so per-crate effects are not modelled. A channel→crate (or crate→cable) map would let us switch grouping and the gain/width factors to crates. Is one available? If not, is module-level grouping acceptable for this iteration?

Portfolio: I read your "Okies" and "agreed with the rest" as leaving the transfer arm unstarted and keeping M excluded; I will keep those defaults unless you object.

Thanks,
Dowling
```

---

## 2. Issue #4 — status comment (paste after the #5 comment; link to it)

https://github.com/seanxia8/VERITAS/issues/4

```markdown
Status after the 12 October re-scope, so this thread is not left as a stale ack. The substantive reply to your corrections is in #5.

- **LUCiD electronics noise:** frozen; enters no claim cell.
- **Optical N → detector response:** five families (efficiency, gain, timing, fixed-threshold T–Q, module loss) in `noise_module_lucid/response.py`, with a declared readout chain and consequence K. Toy development evidence only.
- **Angle/medium:** angular-acceptance and medium model changes are classed M (default: excluded, no new label; one-line confirmation pending).
- **TIDMAD:** controlled-N injection implemented on the 208 noise-only files (`tidmad_transformer/n_injection.py`); tests only, not yet run on real data. S family and held-out split still open (D-S).
- **DELight:** no simulation chain; the HeST → QP → `noise_module` chain exists but was demoted; conditional on Phase B.
- **LIGO:** future work, no arm built.
- **LArTPC/JAXTPC:** replacement candidate for the transfer arm, not an addition; no build before D-T.
- **Training rule:** baseline only, ≥3 seeds, response variants applied at test time (frozen-model design).

Still open with Junjie: D-E (energy endpoint), D-L (LUCiD role), D-T (transfer arm), D-S (TIDMAD S family); D-M has a recorded default. No claim comparison has run; everything above is development and non-citable.
```

---

## 3. Issue #3 — close as completed

https://github.com/seanxia8/VERITAS/issues/3

```markdown
Closing as completed. `noise_module_lucid` 0.5.0 on `dev` implements this package (declared readout chain, five response families, consequence K, tests): 68 tests pass on a clean `dev` export, and `notebooks/noise_module_lucid_tutorial.ipynb` imports it. Follow-ups are tracked separately: merge `dev` into the default branch, and the LUCiD licence-file/redistribution question (research use is settled in #5). No new work is tracked here.
```

---

## 4. Issue #6 — close / convert to a design note

https://github.com/seanxia8/VERITAS/issues/6

```markdown
Converting this from an implementation issue to a design note. The five items are a model lineup/spec (CRATE-MAE, ToST, GLS-FISTA, frequency-resolved whitener, subject integration); none is implemented in this repo, and the referenced `src/whitebox_subjects/` is not on any branch. Current state is recorded in `docs/REFERENCE_UPGRADES_2026-10-08.md` (TR17/TR20/TR21 are conditional) and `docs/TODO.md`; the repository has the Kronecker whitener and the linear/cumulant controls. A scoped implementation issue should be opened only if D-T/D-L select a white-box subject. Closing as a spec.
```

---

## Posting notes

- Post #5 first. Use the permalink of the new #5 comment in the #4 comment ("The substantive reply to your corrections is in #5" — add the permalink).
- After #5, update the repository's status docs if any number changes; the reply deliberately contains no new numbers beyond the already-recorded R-4c gate failure.
- Do not paste the 18 Oct draft. If it is kept in the tree, mark it superseded by this file.
- If Junjie's answer forces a default change, follow [DECISION_REGISTER_2026-10-21.md](DECISION_REGISTER_2026-10-21.md) §4: record the message, edit
  `configs/decisions/open_decisions_v1.json`, bump the revision, and re-freeze the affected contract. Do not silently edit a default.
