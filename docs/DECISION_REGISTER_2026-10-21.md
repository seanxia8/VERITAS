# Open decisions and defaults — how to keep moving without freezing a claim

**21 October 2026.** Companion to [ISSUE_UPDATES_2026-10-21.md](ISSUE_UPDATES_2026-10-21.md).
The machine-readable register is
[`configs/decisions/open_decisions_v1.json`](../configs/decisions/open_decisions_v1.json);
the loader and claim gate are
[`src/noise_module_lucid/decision_registry.py`](../src/noise_module_lucid/decision_registry.py).

The problem this solves: decisions are spread across `PMT_DECISIONS_LOG.md`,
`PREREGISTRATION.md`, `PMT_EXPERIMENT_CONTRACT.md`, runbooks and issue threads,
and several "adopted" settings are author decisions that Junjie has not
reviewed. A default written in one place can silently become a claim. The
register gives every open item one ID, one default, one owner and one switch.

---

## 1. Status vocabulary (use these words exactly)

| status | meaning | may be used in a claim run? |
|---|---|---|
| `pending` | waiting on Junjie (or on the author for launch items) | no |
| `provisional_default` | a default is in force for development; not a physics sign-off | no, unless a dated waiver says so |
| `adopted` | binding internal decision by the author | not on its own: it is not Junjie's review |
| `confirmed` | explicitly confirmed by the named physics owner, or by a dated amendment | yes |
| `retired` | superseded; kept for history | n/a |

**Claim run** means WP7 or any Claim-1/Claim-2 comparison. The gate checks every
entry with `"claim_gate": true`:

```bash
PYTHONPATH=src .venv/bin/python -m noise_module_lucid.decision_registry
PYTHONPATH=src .venv/bin/python -m noise_module_lucid.decision_registry --claim-gate
```

The second command exits 1 while `D-E`, `D-TQ`, `D-QE` or `D-WP7` is unresolved.
A waiver must be a dated amendment; it does not change the status field.

---

## 2. The open items, their defaults, and what each one blocks

| ID | owner | default now | blocks | switch cost |
|---|---|---|---|---|
| **D-E** energy endpoint | Junjie | deposited energy in volume; contained μ⁻ 100–300 MeV primary; escaping tracks diagnostic only | K_energy, energy gates, Claim-2 energy cells, WP7 | config/contract edit; no code unless choice (C) |
| **D-T** transfer arm | Junjie | no arm; NuRadioMC provisional; JAXTPC replacement candidate | transfer-arm build | documentation + arm registry |
| **D-L** LUCiD role in Paper 3 | Junjie | Claim-1 null control + Claim-2 replication; JEPA independent | Paper-3 arm list | documentation |
| **D-M** category M | Junjie | excluded from both claims; no origin label | taxonomy finalisation | cell-contract edit |
| **D-S** TIDMAD S family | Junjie | no S family; 20-file evaluation; controlled N on 208 noise-only files | TIDMAD claim cell | define S + split |
| **D-K** K aggregation | author | K = max(K_vertex, K_energy); κ_m = 1.10 provisional | Claim-2 threshold | amendment + precision study |
| **D-TQ** walk form/values | Junjie+author | walk(q)=a+b/√q, a=0, b=2 ns, q_sat=50 PE, synthetic_dev; per-channel slope sigma 0.0 | timing family | manifest v3 + amendment |
| **D-QE** effective QE·CE / dark rate | Junjie | 5-anchor synthetic QE, CE=0.8, dark 1 kHz | target calibration, response values, WP7 | manifest v3 |
| **D-QESPREAD** per-PMT QE variance | Junjie | **0.0** (off; current outputs unchanged) | baseline QE realism | code switch + manifest v3 |
| **D-CH** per-channel Q–T | Junjie | **0.0** (one global curve; current outputs unchanged) | per-channel T–Q family | code switch (ChannelWalkCurve) |
| **D-CR** grouping | Junjie | **mpmt_modules** (105); no crate map | per-crate cells | pass a crate map; no code change |
| **D-AA** angular acceptance | Junjie | uniform, no concentrator; model change classed M | acceptance family | transport + classification |
| **D-CONE** cone / Eloss | Junjie | local emission-angle closure only; no reconstructed cone/Eloss | exploratory diagnostic | new estimator |
| **D-WP7** launch | author | **false** | WP7, any claim comparison | explicit "launch WP7" |
| **D-LIC** licence/redistribution | author | research use; licence file absent; no redistribution | release only | upstream licence |

Notes on the two ambiguous ones:

- **D-M:** Junjie's "Agreed with the rest" is read as accepting exclusion. Keep the
  default; do not reopen it as a fresh question. Ask for one-line confirmation
  in the reply instead.
- **D-T:** Junjie's "Okies" does not select JAXTPC. The default is no arm; the
  issue reply says "I keep those defaults unless you object", not "we are
  building JAXTPC".

---

## 3. What is actually blocking what

```
D-E ─┬─> K_energy ──> Claim-2 energy cells
     └─> WP7
D-QE ─┬─> target response values ──> Claim-2 response cells
      └─> WP7
D-TQ ─┬─> timing cells ──> Claim-2
      └─> WP7 (if the form is not accepted)
D-WP7 ──> the comparison itself (author gate)
```

Everything else (D-T, D-L, D-M, D-S, D-CR, D-AA, D-CONE, D-LIC) can stay at its
default while engineering continues; it does not authorise or block a claim run.
D-K and D-TQ values are already used in development and are labelled
`synthetic_dev` / `provisional_dev`.

**Cost of waiting** is low for D-T/D-L/D-M/D-S (paper-scope and arms), medium for
D-CR/AA/cone (family breadth), and high for D-E and D-QE: WP7 cannot start
without them, and every additional development run on synthetic values has to be
labelled non-citable.

---

## 4. How a default changes (the only allowed path)

1. **Record the source message.** Paste Junjie's (or the author's) message
   verbatim into `docs/PMT_DECISIONS_LOG.md`, with date, or add a dated amendment
   to the relevant contract.
2. **Edit the register.** Change the entry's `default` and/or `status`, and bump
   the `revision` field for that entry (add it if missing). Keep the old value in
   `notes` or a `history` list. Do not edit silently.
3. **Apply the value where the entry points.** Config keys and contracts are
   listed in `config_refs`/`config_keys`. Physics values in
   `configs/lucid_pmt/reference.json` require a **new manifest version** (v3)
   with new `source`, `status` and `sha256` — do not edit a frozen manifest in
   place; old results keep their own hashes.
4. **Re-run the affected tests** and, for a claim-gate change, the register test:
   ```bash
   PYTHONPATH=src .venv/bin/python -m pytest src/noise_module_lucid/tests/test_decision_registry.py -q
   PYTHONPATH=src .venv/bin/python -m noise_module_lucid.decision_registry --claim-gate
   ```

The register is data, so a default change is reviewable as a diff. The test suite
fails if an entry loses its owner/default/switch or if a claim-gate entry is
changed without the gate clearing.

---

## 5. Implementation plan for the three physics switches (defaults are non-breaking)

These are the capabilities the issue #5 reply says are missing. None is
implemented in this pass, because each one either needs Junjie's value or would
change the frozen `reference.json` hash. The defaults below reproduce today's
outputs exactly; enabling them later is a config edit plus a new manifest.

1. **Per-PMT QE spread (`D-QESPREAD`, default 0.0).**
   - Add `qe_scale: Any = 1.0` to `ReadoutParams` (`src/noise_module_lucid/readout.py`, around
     the per-channel fields) and multiply the detection probability by it in the
     efficiency stage (`detect = detect * qe_scale[ids]`).
   - `response_manifest.reference_response(..., qe_spread_sigma=0.0)` draws a
     deterministic per-channel lognormal factor, keyed on channel and a declared
     seed, so twins stay paired. `0.0` → all factors 1 and bit-identical outputs.
   - When Junjie gives a value, add `parameters.per_pmt_qe_sigma` to a **v3**
     manifest with `source`/`status`/`sha256`; the fixture and WCTE pilots then
     pick it up through the single builder.
2. **Per-channel Q–T (`D-CH`, default 0.0).**
   - `ChannelWalkCurve` already exists (`src/noise_module_lucid/timing_response.py`) but is
     unused in the reference. Build it from `InverseSqrtWalk` per channel with
     lognormal slope factors; add `ChannelWalkCurve.scaled()` so the existing
     slope-cell construction keeps working.
   - `reference_response(..., walk_slope_sigma=0.0)`; `0.0` keeps the single
     global curve and today's outputs. Enabling it makes test cells shift digit
     times per channel (charge and hit membership still unchanged), which is the
     behaviour Junjie asked for.
3. **Crate grouping (`D-CR`, default mPMT modules).**
   - No code change: `reference_response(n_channels, channel_group)` already
     accepts any group map. A physical crate/cable map is passed as
     `channel_group` instead of the mPMT map; until the map exists, attempting
     crate grouping should fail closed rather than invent a map.
4. **Energy endpoint (`D-E`) and target QE/CE/dark rate (`D-QE`).**
   - Config/contract edits only. The R-3 gate already fails closed unless the run
     declares `deposited_energy_in_volume`; choosing (B) "drop energy" is the one
     choice that needs a code path that skips the energy gate and reports the
     reduced claim scope.

The implementation above is deliberately small: each switch is one constructor
argument plus one manifest key, and the default reproduces the current frozen
reference. That is what makes the defaults genuinely reversible.

---

## 6. Recommended sequence

1. Post the #5 reply, then the #4 status comment, close #3 and #6 (texts in
   [ISSUE_UPDATES_2026-10-21.md](ISSUE_UPDATES_2026-10-21.md)).
2. Collect Junjie's four answers (D-E, effective QE·CE/dark rate/per-PMT spread,
   T–Q curve or 1/√q, crate map).
3. Apply the answers through §4: values into a v3 manifest, decisions to
   `confirmed`, contracts re-frozen with dated amendments.
4. Only then launch WP7. Until then the register's claim gate is the single
   place that answers "can we run the comparison yet?".
