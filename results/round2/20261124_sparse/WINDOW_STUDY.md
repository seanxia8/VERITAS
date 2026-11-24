# r2-sparse W0: readout-window study (cached LUCiD digits)

Source: cached LUCiD digits of `lucid_v1` `dev_probe_train` (LUCiD e651e836), first 2,000 photon bombs, 500 muons, 500 electrons per detector in run order; NHits trigger recomputed exactly as the packs (earliest digit per PMT, `time_rule` N and 20 ns). Signal = LUCiD digit with a physics photoelectron. Script `scripts/sparse_window_study.py`; numbers in `WINDOW_STUDY.json`.

## Summary per detector

| | WCTE | SK | JUNO | HK |
|---|---|---|---|---|
| current window [ns] | [-20, 60] | [-20, 270] | [-20, 200] | [-20, 460] |
| cache truncated by LUCiD trigger | no | yes | no | yes |
| events (no NHits trigger) | 3000 (0) | 3000 (0) | 3000 (0) | 3000 (0) |
| cached digit span rel. trigger, median min / max [ns] | 0 / 37 | -269 / 746 | -74 / 579 | -274 / 945 |
| events with signal before trigger - 20 ns | 25.2 % | 55.3 % | 41.3 % | 50.6 % |
| signal digits before trigger - 20 ns | 0.03 % | 0.05 % | 0.03 % | 0.03 % |
| trigger - earliest signal, median / q95 / q99.5 [ns] | -0 / 83 / 97 | 40 / 155 / 182 | -0 / 88 / 98 | 22 / 146 / 178 |
| first-per-PMT signal beyond window end | 0.00 % | 0.79 % | 3.55 % | 1.34 % |
| all signal digits beyond window end | 0.00 % | 1.29 % | 5.07 % | 2.02 % |
| first-per-PMT signal rel. trigger q99 / q99.5 [ns] | 29 / 32 | 258 / 294 | 274 / 316 | 488 / 558 |
| PMTs with > 1 digit | 0.00 % | 0.97 % | 1.75 % | 3.06 % |
| later-digit delay median / q95 [ns] | n/a / n/a | 252 / 603 | 253 / 454 | 288 / 649 |
| dark digits / event in current window | 0.33 | 7.89 | 6.38 | 32.39 |

## WCTE window by the registered rule (W1)

- q99.5(trigger − earliest signal digit) = 97.2 ns → P = max(50, 97.2 + 10) rounded up to 10 ns = **110 ns**.
- Smallest multiple of 50 ns with ≥ 99.5 % of first-per-PMT signal digits in [−P, A]: **A = 50 ns** (100.00 % inside).
- WCTE window: **[−110 ns, +50 ns]** around the NHits trigger (old: [−20, +60] ns).

- Coverage by the cached dark-noise span at this window: pre-trigger side uncovered in 74.3 % of events (mean 8.0 ns), post-trigger side in 0.0 % (mean 0.0 ns). W2 fills the uncovered parts with LUCiD dark noise.
- Dark digits per event inside the window from the cache (before extension): 0.60; events fully covered without extension: 25.7 %; largest P covered in >= 99 % of events without extension: 84 ns.
- A = 50 ns is shorter than the old end (+60 ns): first-per-PMT signal digits in (50, 60] ns are 0.0041 % of all; no WCTE PMT has a second digit (the 200 ns integration window spans the whole event), so there is no separate late light.

## What the early "signal" digits are (WCTE)

LUCiD flags a digit as physics when its 200 ns integration window holds any physics photoelectron. A dark photoelectron that reaches a PMT before that PMT's light therefore opens the window and sets the digit time, and the digit is still flagged as signal. Expected number of such digits earlier than trigger − 20 ns, from the dark rate (4.2 kHz), the number of hit PMTs and the cached dark span:

- observed early signal digits per event: 0.309; expected from dark-opened windows: 0.317;
- events with at least one: observed 25.2 %, expected 25.4 %;
- (trigger − early digit) per 10 ns from 20 ns: [122, 132, 112, 132, 112, 128, 106, 84, 0, 0] (flat up to the 100 ns dark pad, as for dark noise);
- median charge of the early digits / median of all first digits: 1.58 (normal hits, not single-PE).

So on WCTE the early digits are almost entirely dark-opened windows on PMTs that are hit later: no early direct light is lost. What the old window does lose is the whole PMT (its charge and later light), because the readout keeps each PMT's earliest digit and then applies the window (F3). P = 110 ns keeps these PMTs; their time is the dark photoelectron's time. The registered W1 rule is applied unchanged; this reading is recorded as a W0 finding.

## Dark noise uniformity (WCTE, digits per 100 ns rel. trigger, all events)

| bin start [ns] | -600 | -500 | -400 | -300 | -200 | -100 | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dark digits | 0 | 0 | 0 | 0 | 0 | 945 | 1464 | 456 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The cached dark noise spans only [first deposit − 100 ns, last deposit + 100 ns] (WCTE, JUNO), so the counts fall off outside it; SK and HK caches end at LUCiD's trigger gate.

## True time range of the current samples (co-author's question)

- **WCTE**: cached digits span median [0, 37] ns around the NHits trigger (no LUCiD trigger: every digit kept); the packs keep [-20, 60] ns.
- **SK**: cached digits span median [-269, 746] ns around the NHits trigger (LUCiD trigger gate (200 ns window + 300 ns each side) applied before caching); the packs keep [-20, 270] ns.
- **JUNO**: cached digits span median [-74, 579] ns around the NHits trigger (no LUCiD trigger: every digit kept); the packs keep [-20, 200] ns.
- **HK**: cached digits span median [-274, 945] ns around the NHits trigger (LUCiD trigger gate (200 ns window + 300 ns each side) applied before caching); the packs keep [-20, 460] ns.
