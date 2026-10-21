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

Thanks for the corrections, and for pinning the LUCiD managers. You were right about the licence: the missing LICENSE file never blocked running it here — the research-use clarification is enough for our use, and the licence-file/redistribution question stays separate. We have since run the pinned LUCiD (eaf2e89) prompt transport end to end on WCTE.

Taking your points in order:

**QE, CE and gain.** QE is now wavelength-dependent (5 anchors, 200–800 nm) and enters as QE(λ)·CE per photon; an exact QE×2/CE÷2 control gives bit-identical digits, so we make no identifiability claim between QE, CE and gain — they are one effective factor, as you said. What we still need is the target effective QE·CE at ≥3 wavelengths, not separate curves. The original per-PMT QE spread is not built: the reference has one global QE curve. If it belongs in the baseline, give me a number; otherwise I can add it as a documented 5% lognormal default, changeable later.

**Fixed threshold and T–Q.** Taken: the discriminator is fixed at 0.25 PE, the walk is calibrated at training time, and a changed Q–T relation shifts digit times instead of losing charge; charge and hit membership are unchanged by the timing-only cells. Two gaps: (i) the reference has one walk curve for all channels — per-channel Q–T is not implemented; I can add it behind a switch with a documented default, or use your measured spread; (ii) I replaced the polynomial with walk(q) = a + b/√q (b = 2 ns, q_sat = 50 PE), because the old polynomial was almost flat over 0.25–20 PE. That is a deviation from your "polynomial approximation" — you may prefer the polynomial or an Eq. 5.13 fit, and I will switch if so.

**B-field / modules / crates.** B-field is a synthetic module-level gain+width proxy, not a field model. Groups are the 105 mPMT modules from the pinned geometry, not crates; with a crate/cable map we can switch the grouping to crates.

**Angular acceptance and cone.** Deferred as you suggested: prompt-only, uniform AA, no concentrator, scattering/reflection off. We have a source-truth local emission-angle closure (median 0.46°); a reconstructed ToF-vertex cone and an Eloss comparison are not implemented.

**Thesis.** Used for the stage conventions and the Eq. 6.2/6.3/5.13 caveats; Eq. 5.13 is not converted into the target curve yet.

**Status.** No claim experiment has run. Everything above is development evidence with synthetic response values; no model has been trained with any response variant, and no Claim-1/2 comparison exists. Vertex still fails on tracks (0.81–0.88 × constant against a 0.8 gate), which is why the energy endpoint decision matters.

Four things from you would unblock the physics: (1) D-E: deposited energy on contained μ⁻ at 100–300 MeV as the default, or drop energy; (2) effective QE·CE at ≥3 wavelengths, the dark rate (with temperature/channel dependence), and whether per-PMT QE spread belongs in the baseline; (3) a measured/thesis T–Q curve — or is a + b/√q acceptable as the default; (4) the crate/cable map (module grouping meanwhile).

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
