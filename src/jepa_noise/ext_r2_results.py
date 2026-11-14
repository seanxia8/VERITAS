"""Round-2 results writer (runbook §10): runs.json, readings.json, READINGS_SUMMARY.md, figures.

Reads the returns root (ledger, run records, evaluations) and writes the committed result
files.  Descriptive only: every reading is computed by ``ext_r2`` with the §7 rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import ext_r2 as x
from . import r2_geometry as rg

#: Round-1 committed WCTE values (full-input vertex, m) for the cross-detector table.
WCTE_REFERENCE = {
    "init": {"9000": 0.160072, "source": "results/jepa/20261101_readout/matrix.json items/init9000"},
    "mae_mse": {"9000": 0.104916,
                "source": "results/jepa/20261031_mae_dejitter/summary.json (mae_within_trigger9000)"},
    "target_posfree": {"9000": 0.126005, "9001": 0.140544,
                       "source": "results/jepa/20261107_shortcut/README.md (R-2)"},
    "sigreg_jepa": {"9000": 0.219259, "9001": 0.204281,
                    "source": "results/jepa/20261101_readout/matrix.json items/fix9000, fix9001"},
}
WCTE_FIDUCIAL_RADIUS_M = None  # filled from the WCTE geometry at run time
ARM_LABEL = {"sigreg_jepa": "fixed-label JEPA", "target_posfree": "position-free JEPA",
             "mae_mse": "MAE", "init": "random init", "mae_mse_rf": "MAE (RF-matched)"}


#: Where each detector's evaluations live (SK: deepthought2 L40S; JUNO, HK: TOpAS A100 batch).
EVAL_ROOT = {"sk": "", "juno": "batch", "hk": "batch"}


def detector_root(root, detector: str) -> Path:
    return Path(root) / EVAL_ROOT.get(detector, "")


def runs_table(root, detector: str) -> dict:
    """Per run: final record (batch record preferred when a run moved to batch), GPU hours from
    every ledger line it used, and the hardware of each segment."""
    root = Path(root)
    led = {}
    for name in ("RESOURCE_LEDGER_R2.json", "RESOURCE_LEDGER_R2_BATCH.json"):
        p = root / "ledger" / name
        led[name] = json.loads(p.read_text())["entries"] if p.exists() else []
    names = sorted({d.name for base in (root / "runs", root / "batch" / "runs") if base.exists()
                    for d in base.glob(f"{detector}__*")})
    rows = []
    for n in names:
        recs = [(where, base / n / "run_record.json") for where, base in
                (("deepthought2", root / "runs"), ("batch", root / "batch" / "runs"))
                if (base / n / "run_record.json").exists()]
        if not recs:
            continue
        final_where, final_p = recs[-1]
        r = json.loads(final_p.read_text())
        hours = {k: sum(float(e.get("gpu_h", 0.0)) for e in v if e.get("kind") == "train"
                        and e.get("detector") == detector and e.get("arm") == r["arm"]
                        and int(e.get("seed", -1)) == int(r["seed"])) for k, v in led.items()}
        rows.append({"detector": detector, "arm": r["arm"], "seed": r["seed"], "steps": r["steps"],
                     "complete": r["complete"], "micro_batch": r["micro_batch"],
                     "l40s_gpu_h": hours["RESOURCE_LEDGER_R2.json"],
                     "batch_gpu_h": hours["RESOURCE_LEDGER_R2_BATCH.json"],
                     "hardware": [w for w, _ in recs], "final_gpu_model": r.get("gpu_model"),
                     "final_site": r.get("site"), "peak_memory_gib": r["peak_memory_gib"],
                     "seconds_per_step_last": r["seconds_per_step"], "config_sha256": r["config_sha256"]})
    smokes = []
    for base, hw in ((root / "smokes", "L40S (deepthought2)"), (root / "batch" / "smokes", "A100 (TOpAS)")):
        for p in sorted(base.glob(f"{detector}__*/run_record.json")):
            r = json.loads(p.read_text())
            smokes.append({"arm": r["arm"], "micro_batch": r["micro_batch"], "hardware": hw,
                           "seconds_per_step": r["seconds_per_step"],
                           "projected_h": r["seconds_per_step"] * x.STEPS / 3600.0,
                           "peak_memory_gib": r["peak_memory_gib"]})
    return {"runs": rows, "smokes": smokes}


def time_feature_stats(h5_path, detector: str) -> dict:
    """§6: std of the time feature over hit tokens of dev_train."""
    import h5py
    from . import r2_generator as gen
    geometry = rg.load_r2_geometry(detector)
    scale = gen.time_rule_for(geometry).time_scale_ns
    with h5py.File(h5_path, "r") as f:
        t = f["dev_train"]["time_ns"][:]
    return {"time_scale_ns": scale, "time_feature_std": x.time_feature_std(t, scale),
            "n_hits": int(t.size)}


def wcte_time_feature_stats() -> dict:
    """The same statistic on the round-1 WCTE dev_train (100 runs, time scale 100 ns)."""
    root = Path("/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/lucid") / \
        "dev_train__F1__0.05__s20261017__n100000x1000" / "runs"
    ts = []
    for d in sorted(root.iterdir()):
        with np.load(d / "hits.npz", allow_pickle=False) as h:
            ts.append(np.asarray(h["time_ns"], np.float64))
    t = np.concatenate(ts)
    return {"time_scale_ns": 100.0, "time_feature_std": x.time_feature_std(t, 100.0),
            "n_hits": int(t.size), "runs": len(ts)}


ROUND1_MAE_LOG = Path("/ceph/dwong/jepa/returns/ext_mae_dejitter_20261031/train/runs/"
                      "ext_dev_s005_within_trigger__mae_mse__s9000/train_log.jsonl")


def wcte_loss_shares() -> dict:
    """WCTE round-1 MAE loss shares. Round 1 logged every 100 steps, so start = the step-100
    row and end = the last row (an approximation of the round-2 100-step window means)."""
    rows = [json.loads(l) for l in ROUND1_MAE_LOG.read_text().splitlines() if l.strip()]
    out = {}
    for label, r in (("start", rows[0]), ("end", rows[-1])):
        terms = {t: float(r["metrics"][t]) for t in x.LOSS_TERMS}
        tot = sum(terms.values())
        out[label] = {"step": int(r["step"]), "shares": {t: v / tot for t, v in terms.items()}}
    out["note"] = "round-1 log every 100 steps: single rows, not 100-step means"
    return out


def time_share_ratio(det_shares: dict, wcte: dict) -> dict:
    """Ratio of the time-term share to WCTE's at start and end; flag a move > 2x."""
    out = {}
    for label in ("start", "end"):
        r = det_shares[label]["shares"]["time"] / wcte[label]["shares"]["time"]
        out[label] = {"ratio": r, "moved_more_than_2x": bool(r > 2 or r < 0.5)}
    return out


def detector_bundle(root, detector: str, h5_path=None) -> dict:
    droot = detector_root(root, detector)
    out = {"detector": detector, "extension": x.EXTENSION, "registered": False,
           "evaluation_hardware": "L40S (deepthought2)" if detector == "sk" else "A100-40GB (TOpAS batch)",
           "readings": x.readings_both_versions(droot, detector),
           "loss_shares": x.loss_share_diagnostics(droot, detector)}
    if h5_path is not None and Path(h5_path).is_file():
        out["time_feature"] = time_feature_stats(h5_path, detector)
    return out


def _fmt(v, nd=4):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1e-3 and v != 0 else f"{v:.{nd}f}"
    return str(v)


def summary_markdown(bundles: dict, wcte_time: dict | None, seed9002: dict, gatev: dict) -> str:
    L = ["# Round 2 readings summary", "",
         "Extension `r2_multidetector_2026-11-11`, `registered: false`. Gate V is informational "
         "(addendum `RUNBOOK_2026-11-12_R2_GATEV_CEILING.md`): it failed with the round-1 absolute "
         "thresholds on SK and JUNO, so the analytic model is not validated against PhotonSim + "
         "LUCiD at these scales and every number below carries that caveat.", ""]
    L += ["## Vertex error (full input, m), per arm and seed", "",
          "| Detector | " + " | ".join(ARM_LABEL[a] for a in x.WCTE_ORDER) + " | MAE (RF) 9000 |",
          "|---|" + "---|" * (len(x.WCTE_ORDER) + 1)]
    L.append("| WCTE (round 1) | " + " | ".join(
        "/".join(_fmt(WCTE_REFERENCE[a][s]) for s in ("9000", "9001") if s in WCTE_REFERENCE[a])
        for a in x.WCTE_ORDER) + " | – |")
    for det, b in bundles.items():
        two = b["readings"]["two_seed"]
        if two.get("status") != "evaluated":
            L.append(f"| {det.upper()} | " + " | ".join("–" for _ in x.WCTE_ORDER) + " | – |")
            continue
        m = two["metrics"]
        cells = ["/".join(_fmt(m[f"{a}:{s}"]["vertex_full_m"]) for s in (9000, 9001))
                 for a in x.WCTE_ORDER]
        rf = m.get("mae_mse_rf:9000")
        L.append(f"| {det.upper()} | " + " | ".join(cells) + f" | {_fmt(rf['vertex_full_m']) if rf else '–'} |")
    L += ["", "## Readings R2-a to R2-e (two seeds; three-seed version where seed 9002 ran)", ""]
    for det, b in bundles.items():
        two = b["readings"]["two_seed"]
        if two.get("status") != "evaluated":
            L.append(f"- **{det.upper()}**: not evaluated ({two.get('status')}).")
            continue
        m = two["metrics"]
        fid = m["init:9000"]["fiducial_radius_m"]
        L.append(f"### {det.upper()} (fiducial radius {fid:.2f} m)")
        a = two["R2-a"]
        L.append(f"- R2-a fixed-label collapse: **{a['class']}** (f_event "
                 + ", ".join(f"{s}: {_fmt(v)}" for s, v in a["f_event"].items()) + ").")
        bb = two["R2-b"]
        L.append(f"- R2-b position-free recovery: **{bb['class']}** (" + "; ".join(
            f"{s}: vertex {_fmt(v['vertex'])} vs init {_fmt(v['init_vertex'])}, token R² "
            f"{_fmt(v['token_r2'])}, f_event {_fmt(v['f_event'])}" for s, v in bb["per_seed"].items()) + ").")
        c = two["R2-c"]
        L.append(f"- R2-c MAE above floor: **{c['class']}** (" + "; ".join(
            f"{s}: MAE − init {_fmt(v['mean_difference_m'])} m, CI [{_fmt(v['ci95_m'][0])}, "
            f"{_fmt(v['ci95_m'][1])}]" for s, v in c["per_seed"].items()) + ").")
        d = two["R2-d"]
        L.append(f"- R2-d ranking: {' < '.join(d['order'])}; Kendall τ vs WCTE = {d['kendall_tau']:.3f}.")
        e = two["R2-e"]
        if e.get("status") == "not run":
            L.append("- R2-e receptive field: not run.")
        else:
            v = e["vertex"]
            L.append(f"- R2-e receptive field (RF − MAE, seed 9000): vertex {_fmt(v['mean_difference_m'])} m, "
                     f"CI [{_fmt(v['ci95_m'][0])}, {_fmt(v['ci95_m'][1])}]; token R² "
                     f"{_fmt(e['token_r2_difference'])}; f_event {_fmt(e['f_event_difference'])}; "
                     + ("confounded by receptive field by "
                        f"{abs(e['by_m']):.3f} m." if e["confounded_by_receptive_field"]
                        else "not flagged as a receptive-field confound."))
        fb = two["floor_band"]
        if "vertex_full_m" in fb:
            v = fb["vertex_full_m"]
            L.append(f"- Floor band ({v['n']} init seeds): vertex mean {_fmt(v['mean'])} m, range "
                     f"[{_fmt(v['min'])}, {_fmt(v['max'])}]; token R² mean "
                     f"{_fmt(fb.get('token_log_charge_r2', {}).get('mean'))}.")
        three = b["readings"].get("three_seed")
        if three:
            L.append(f"- Three-seed version: R2-a {three['R2-a']['class']}, R2-b {three['R2-b']['class']}, "
                     f"R2-c {three['R2-c']['class']}, R2-d {' < '.join(three['R2-d']['order'])} "
                     f"(seeds used {three['seeds_used']}).")
        s9 = seed9002.get(det)
        if s9:
            L.append("- Seed 9002: " + "; ".join(
                f"{p['arm']} ({', '.join(p['triggers'])}): {p['decision']}"
                + (f" — {p['reason']}" if p.get("reason") else "") for p in s9) + ".")
        else:
            L.append("- Seed 9002: no trigger.")
        L.append("")
    L += ["## Particle tasks (seed 9000)", "",
          "| Detector | Arm | μ/e AUC | e direction (deg) | μ direction (deg) | e energy R² | μ energy R² |",
          "|---|---|---|---|---|---|---|"]
    for det, b in bundles.items():
        two = b["readings"]["two_seed"]
        if two.get("status") != "evaluated":
            continue
        for a in x.WCTE_ORDER + ("mae_mse_rf",):
            mm = two["metrics"].get(f"{a}:9000")
            if not mm:
                continue
            p = mm.get("particle", {})
            keys = list(p)
            pick = lambda *subs: next((p[k] for k in keys if all(s in k for s in subs)
                                       and isinstance(p[k], (int, float))), None)
            L.append(f"| {det.upper()} | {ARM_LABEL[a]} | {_fmt(p.get('mue_auc'))} | "
                     f"{_fmt(p.get('electron_direction_mlp_median_deg'))} | "
                     f"{_fmt(p.get('muon_direction_mlp_median_deg'))} | {_fmt(p.get('electron_energy_r2'))} | "
                     f"{_fmt(p.get('muon_energy_r2'))} |")
    L += ["", "## Time-scale diagnostics (§6, logged, no threshold)", ""]
    wls = None
    try:
        wls = wcte_loss_shares()
    except OSError:
        pass
    if wls:
        L.append("- WCTE round-1 MAE loss shares (step 100 / step 10,000 rows): " + ", ".join(
            f"{t} {wls['start']['shares'][t]:.3f} → {wls['end']['shares'][t]:.3f}"
            for t in wls["start"]["shares"]) + ".")
    if wcte_time:
        L.append(f"- WCTE round 1: time-feature std {wcte_time['time_feature_std']:.4f} "
                 f"(scale {wcte_time['time_scale_ns']:.0f} ns).")
    for det, b in bundles.items():
        tf = b.get("time_feature")
        if tf:
            L.append(f"- {det.upper()}: time-feature std {tf['time_feature_std']:.4f} (scale "
                     f"{tf['time_scale_ns']:.1f} ns).")
        for arm, sh in (b.get("loss_shares") or {}).items():
            st, en = sh["start"]["shares"], sh["end"]["shares"]
            L.append(f"  - {arm} loss shares (first / last 100 steps): " + ", ".join(
                f"{t} {st[t]:.3f} → {en[t]:.3f}" for t in st) + ".")
            if wls:
                r = time_share_ratio(sh, wls)
                moved = r["start"]["moved_more_than_2x"] or r["end"]["moved_more_than_2x"]
                L.append(f"    The time term's share is {r['start']['ratio']:.2f}× (start) and "
                         f"{r['end']['ratio']:.2f}× (end) the WCTE share: it "
                         + ("moved by more than 2× relative to WCTE." if moved
                            else "did not move by more than 2× relative to WCTE."))
    if gatev:
        L += ["", "## Gate V (informational) and the photon-statistics ceiling", "",
              "| Detector | Module median | Log-charge median | Absolute gate | Module ceiling C | Log ceiling C | ρ̃ module | ρ̃ log |",
              "|---|---|---|---|---|---|---|---|"]
        for det, g in gatev.items():
            L.append(f"| {det.upper()} | {_fmt(g.get('module'))} | {_fmt(g.get('log'))} | {g.get('verdict')} | "
                     f"{_fmt(g.get('C_module'))} | {_fmt(g.get('C_log'))} | {_fmt(g.get('rho_module'))} | "
                     f"{_fmt(g.get('rho_log'))} |")
    return "\n".join(L) + "\n"


def figures(bundles: dict, out_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    dets = [d for d, b in bundles.items() if b["readings"]["two_seed"].get("status") == "evaluated"]
    if not dets:
        return made
    arms = list(x.WCTE_ORDER)
    fig, axes = plt.subplots(1, len(dets) + 1, figsize=(4 * (len(dets) + 1), 3.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, det in zip(axes, ["wcte"] + dets):
        for i, a in enumerate(arms):
            if det == "wcte":
                vals = [WCTE_REFERENCE[a][s] for s in ("9000", "9001") if s in WCTE_REFERENCE[a]]
                fid = None
            else:
                m = bundles[det]["readings"]["two_seed"]["metrics"]
                vals = [m[f"{a}:{s}"]["vertex_full_m"] for s in (9000, 9001)]
            ax.scatter([i] * len(vals), vals, color="C0", zorder=3)
        if det != "wcte":
            fb = bundles[det]["readings"]["two_seed"]["floor_band"].get("vertex_full_m")
            if fb:
                ax.axhspan(fb["min"], fb["max"], color="0.85", zorder=1, label="init floor band (5 seeds)")
            rf = bundles[det]["readings"]["two_seed"]["metrics"].get("mae_mse_rf:9000")
            if rf:
                ax.scatter([len(arms)], [rf["vertex_full_m"]], color="C1", zorder=3)
        ax.set_xticks(range(len(arms) + (det != "wcte")))
        ax.set_xticklabels([ARM_LABEL[a] for a in arms] + (["MAE RF"] if det != "wcte" else []),
                           rotation=35, ha="right", fontsize=8)
        ax.set_title(det.upper())
        ax.set_ylabel("vertex error (m)")
    fig.tight_layout()
    p = out_dir / "vertex_per_arm.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p.name)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for j, det in enumerate(dets):
        m = bundles[det]["readings"]["two_seed"]["metrics"]
        for i, a in enumerate(arms):
            vals = [max(1e-6, m[f"{a}:{s}"]["f_event"]) for s in (9000, 9001)]
            ax.scatter([i + 0.15 * j] * 2, vals, color=f"C{j}", label=det.upper() if i == 0 else None)
    ax.set_yscale("log")
    ax.axhline(1e-3, ls=":", color="0.5")
    ax.axhline(1e-2, ls=":", color="0.5")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=20, fontsize=8)
    ax.set_ylabel("full-input f_event")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "f_event_per_arm.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p.name)
    return made


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(x.RETURNS_ROOT))
    p.add_argument("--out", required=True, help="results/round2 directory in a repo clone")
    p.add_argument("--date", default="20261112")
    p.add_argument("--detectors", default="sk,juno,hk")
    p.add_argument("--gatev", default=None, help="gatev summary JSON (per detector)")
    p.add_argument("--seed9002", default=None, help="seed-9002 decisions JSON (per detector)")
    p.add_argument("--wcte-time", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out)
    bundles, runs = {}, {}
    for det in a.detectors.split(","):
        h5 = x.DATA_ROOT / det / f"{det}_r2.h5"
        bundles[det] = detector_bundle(a.root, det, h5)
        runs[det] = runs_table(a.root, det)
    ledger = x.ledger_load(a.root)
    train_dir, eval_dir = out / f"{a.date}_train", out / f"{a.date}_eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "runs.json").write_text(json.dumps(
        {"extension": x.EXTENSION, "ledger_line": x.LEDGER_LINE, "cap_l40s_h": x.GPU_CAP_H,
         "booked_l40s_h": x.booked_gpu_h(ledger), "hk_fallback": ledger.get("hk_fallback"),
         "detectors": runs}, indent=1, default=float) + "\n")
    wcte_time = wcte_time_feature_stats() if a.wcte_time else None
    s9 = json.loads(Path(a.seed9002).read_text()) if a.seed9002 else {}
    gv = json.loads(Path(a.gatev).read_text()) if a.gatev else {}
    (eval_dir / "readings.json").write_text(json.dumps(
        {"extension": x.EXTENSION, "registered": False, "wcte_reference": WCTE_REFERENCE,
         "wcte_time_feature": wcte_time, "detectors": bundles, "seed9002": s9},
        indent=1, sort_keys=True, default=float) + "\n")
    (eval_dir / "READINGS_SUMMARY.md").write_text(summary_markdown(bundles, wcte_time, s9, gv))
    made = figures(bundles, eval_dir / "figures")
    print(json.dumps({"wrote": [str(train_dir / "runs.json"), str(eval_dir / "readings.json"),
                                str(eval_dir / "READINGS_SUMMARY.md")], "figures": made}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
