"""Build ``configs/lucid_pmt/reference_v3.json`` (R-7 v2) from a pinned WCSim source file.

R-7 v2 (author message, 23 Oct 2026) adopts WCTE's 3-inch Hamamatsu R14374 response
from WCSim (github.com/WCSim/WCSim, commit c5ed0b1, ``src/WCSimPMTObject.cc``, class
``PMT3inchR14374_WCTE``). Author decisions, 3 Nov 2026: sample the SPE table directly
(the Gaussian+exponential fit misses the KS tolerance declared before fitting), and
take the timing width from WCSim's spline (0.760 ns at 0.25 PE, not 0.77).

Usage (CPU, seconds)::

    git clone --filter=blob:none --no-checkout https://github.com/WCSim/WCSim wcsim
    git -C wcsim checkout c5ed0b1 -- src/WCSimPMTObject.cc
    PYTHONPATH=src python scripts/pmt_r7v2_wcsim_manifest.py wcsim/src/WCSimPMTObject.cc

The script refuses a source file whose sha256 differs from the pinned one, starts
from the thesis cross-check manifest (so every unchanged parameter keeps its value,
source and status), replaces the R-7 v2 parameters, recomputes their value hashes,
and writes the manifest. Re-running it on the pinned file reproduces the committed
manifest byte for byte.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "configs" / "lucid_pmt" / "reference_v3_thesis_xcheck.json"
OUT = ROOT / "configs" / "lucid_pmt" / "reference_v3.json"

WCSIM_REPO = "https://github.com/WCSim/WCSim"
WCSIM_COMMIT = "c5ed0b127fd4ef13e5c34436c2c5385fcc9b26b3"
WCSIM_FILE = "src/WCSimPMTObject.cc"
WCSIM_SHA256 = "caa9ad51f1e59f236905927102ff775c33d11aeb5b51a320c884b3392bf18b9f"
CLASS = "PMT3inchR14374_WCTE"
#: rn1pe (the SPE sampler) lives in WCSimWCPMT.cc at the same commit.
RN1PE = "src/WCSimWCPMT.cc (WCSimWCPMT::rn1pe, lines 55-78; file sha256 91c7b9b92a69b01769a964f9b60080bc5caf6efd764e9b27cdc7f5b9dce91efa)"
SPE_FIT_KS_TOL = 0.02          # fixed before fitting in the 23-25 Oct extraction script (outside the repo), R-7 v2 item 4
QE_EDGE_NM = 0.01              # QE is zero outside [300, 700] nm, as the author instructs (R-7 v2 item 2)
STATUS = "literature_proxy_wcsim_wcte_3in"
SRC = f"WCSim {WCSIM_REPO} commit {WCSIM_COMMIT[:7]}, {WCSIM_FILE} (sha256 {WCSIM_SHA256}), class {CLASS}"
#: Other WCSim c5ed0b1 files read for what WCSim does around the PMT object (recorded, not applied).
WCSIM_CONTEXT = {
    "macros/WCTE.mac": "cef1689ae14454186aca9d598b95279cab5837ffe9d23228827fb99534aea64d",
    "src/WCSimPMTQE.cc": "eb804f9301f23034c7beee4d8e34e483b6c39d48c041fdd7c1b63fb7e3032624",
    "src/WCSimWCSD.cc": "a29a6ed0e5f527906256cb53a8fb4d96701d55b2e7becb2c44efe706539892ef",
    "src/WCSimWCDigitizer.cc": "14740450b4b7be63da6eacbbf4b1a40c7b5dd74f44715ba98e37245b006ebcf4",
}


def value_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def class_block(src):
    start = src.index(f"{CLASS}::{CLASS}()")
    return src[start: src.index("COMBINED PMTS CLASS", start)]


def array_after(block, fn):
    i = block.index(f"::{fn}(")
    body = re.sub(r"//[^\n]*", "", block[i: block.index("return", i)])
    nums = re.search(r"\[\d+\]\s*=\s*\{(.*?)\}", body, re.S).group(1).replace("*correctionFactor", "")
    return [float(x) for x in nums.split(",") if x.strip()]


def ctor_array(block, name):
    return [float(x) for x in re.search(rf"double {name}\[\d+\]\s*=\s*\{{(.*?)\}};", block, re.S).group(1).split(",")]


def scalar(block, fn, pattern):
    i = block.index(f"::{fn}(")
    return re.search(pattern, block[i: block.index("}", i)]).group(1)


def qpe_table(block):
    i = block.index("::Getqpe()")
    body = block[i: block.index("return qpe0", i)]
    body = re.sub(r"//[^\n]*", "", body[body.index("{", body.index("qpe0[")) + 1: body.rindex("}")])
    vals = [float(x) for x in body.split(",") if x.strip()]
    if len(vals) != 501 or vals[-1] != 0.0:
        raise ValueError("expected 500 CDF values plus WCSim's trailing dummy 0.0")
    return vals[:500]


def spe_fit_record(cdf):
    """The Gaussian+exponential fit R-7 v2 item 4 asked for, scored against the declared KS tolerance.

    Recorded, not adopted: KS 0.031 exceeds 0.02, so the author chose direct table sampling (3 Nov 2026).
    """
    c = np.asarray(cdf)
    i = np.arange(len(c))
    lo, hi = (i - 50) / 22.83, (i - 49) / 22.83
    pmf = np.diff(np.concatenate([[0.0], c]))

    def F(p, q):
        w, mu, sg, tau = p
        return w * (1 - np.exp(-np.clip(q, 0, None) / tau)) + (1 - w) * norm.cdf((q - mu) / sg)

    r = least_squares(lambda p: F(p, hi) - c, [0.2, 1.0, 0.35, 0.3], bounds=([0, 0.3, 0.05, 0.01], [0.9, 2.0, 1.5, 3.0]))
    qq = np.linspace(lo[0], hi[-1], 200001)
    ks = float(np.max(np.abs(F(r.x, qq) - np.interp(qq, np.concatenate([[lo[0]], hi]), np.concatenate([[0.0], c])))))
    w, mu, sg, tau = (float(x) for x in r.x)
    mean = w * tau + (1 - w) * mu
    eff = float(np.sqrt(w * 2 * tau ** 2 + (1 - w) * (sg ** 2 + mu ** 2) - mean ** 2) / mean)
    return {"form": "w*Exp(tau) + (1-w)*Normal(mu, sigma) (Bellamy 1994), fitted to the table CDF at bin upper edges",
            "w": round(w, 4), "mu_pe": round(mu, 4), "sigma_pe": round(sg, 4), "tau_pe": round(tau, 4),
            "fit_mean_pe": round(mean, 4), "fit_relative_sigma_effective": round(eff, 4),
            "ks_distance_to_table": round(ks, 4), "ks_tolerance_declared_before_fit": SPE_FIT_KS_TOL,
            "ks_pass": ks <= SPE_FIT_KS_TOL, "adopted": False,
            "ks_objective_note": ("the KS value depends on the fit objective: this is a least-squares fit to the CDF; a fit "
                                  "minimising the KS distance directly reaches 0.024, still above the tolerance"),
            "tolerance_provenance": ("0.02 was fixed before any fit in the 23-25 Oct WCSim extraction script "
                                     "(_patches/pmt_r7v2/extract_v3.py, KS_TOL = 0.02, outside the repository)")}


def build(src_path):
    raw = Path(src_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != WCSIM_SHA256:
        raise SystemExit(f"{src_path}: sha256 does not match the pinned WCSim file {WCSIM_SHA256}")
    b = class_block(raw.decode())
    charge, resol = ctor_array(b, "charge"), ctor_array(b, "resol")
    wl, qe = array_after(b, "GetQEWavelength"), array_after(b, "GetQE")
    ce = array_after(b, "GetCollectionEfficiencyArray")
    n_qe_defined = int(scalar(b, "GetNbOfQEDefined", r"factor = (\d+)"))
    dr_factor = float(scalar(b, "GetDarkRateConversionFactor", r"factor = ([\d.]+)"))
    cdf = qpe_table(b)
    assert len(charge) == len(resol) == 14 and len(wl) == len(qe) == 21 and set(ce) == {95.0}
    assert "1.0*CLHEP::kilohertz" in b and "pmt_tts = 1.5" in b and "if (Q>4.0) Q = 4.0" in b and 'Eval(Q,0,"S")' in b

    from noise_module_lucid.spe_charge import TabulatedSPE
    spe = TabulatedSPE(tuple(cdf))
    p_below_thr_raw = float(spe.cdf_at(0.25 / spe.raw_mean_pe))      # raw WCSim charge < 0.25 PE
    p_below_thr = float(spe.cdf_at(0.25))                            # normalised charge < 0.25 PE
    fit = spe_fit_record(cdf)

    m = json.loads(THESIS.read_text())
    thesis_sha = hashlib.sha256(THESIS.read_bytes()).hexdigest()
    out = {}
    for k in ("schema",):
        out[k] = m[k]
    out["status"] = STATUS
    out["status_note"] = (f"R-7 v2 (author, 23 Oct 2026; decisions 3 Nov 2026): qe_curve, collection_efficiency, spe_charge, "
                          f"timing_width and dark_rate_hz are WCTE 3-inch R14374 values from WCSim (status {STATUS}); "
                          "threshold_pe keeps 0.25 PE (literature_sk) and now acts on the digitised charge; walk, charge "
                          "domain, per-module factor rules and every other parameter keep their synthetic_dev value and provenance")
    out["supersedes"] = ("pmt-response-v2 (configs/lucid_pmt/reference.json, sha256 " + m["supersedes"].split("sha256 ")[1].split(")")[0] +
                         "), retained byte-identical as the label of past results (R-4, R-4b, R-4c) and of the pinned JEPA SHA 5f4ecc0; "
                         f"and the first R-7 manifest (20-inch thesis proxies, committed in d2f179f as reference_v3.json, sha256 {thesis_sha}), "
                         "superseded by R-7 v2 before any result used it and kept byte-identical as configs/lucid_pmt/reference_v3_thesis_xcheck.json, a cross-check only")
    out["builder"] = m["builder"]
    tc = dict(m["target_candidate"])
    tc["pmt_type"] = "Hamamatsu R14374 3-inch (WCSim PMT3inchR14374_WCTE); LUCiD geometry sensor_radius 0.04 m, 105 mPMT modules x 19 PMTs"
    out["target_candidate"] = tc
    out["units"], out["photon_stage"] = m["units"], m["photon_stage"]
    so = dict(m["stage_owners"])
    so["charge_gain"] = "readout: one tabulated SPE charge per photoelectron (WCSim Getqpe/rn1pe distribution, mean normalised to 1 PE), summed per PMT, then the per-module gain"
    so["timing_width"] = ("readout Gaussian digit kernel, sigma(Q) = sqrt(1.5^2 + R(min(Q,4))^2)/2.355 ns (WCSim HitTimeSmearing, "
                          "R spline-interpolated as TGraph::Eval(Q,0,\"S\")), evaluated at the digitised charge Q with no 0.5 PE floor "
                          "(author decision 3 Nov 2026; WCSim's SKI digitizer passes max(Q, 0.5), recorded, not applied), with a per-module "
                          "factor multiplying sigma; upstream TTS forbidden")
    so["dark_counts"] = so["dark_counts"].replace("the same SPE charge", "the same tabulated SPE charge")
    out["stage_owners"] = so
    out["transport"] = m["transport"]
    cv = dict(m["conventions"])
    cv["charge"] = ("q_PMT = sum over its photoelectrons of x_k, x_k drawn from the WCSim Getqpe table: bin i with probability "
                    "cdf[i]-cdf[i-1], uniform inside the bin, raw charge (i-50+u)/22.83 (WCSim rn1pe), divided by the table mean "
                    f"{spe.raw_mean_pe:.6f} so the mean SPE charge is 1 PE; clipped at 0; then gain")
    cv["timing"] = ("forward ns; D-TQ walk(q)=a+b/sqrt(min(q,q_sat)) on the charge domain; frozen reference subtraction; digit width is a "
                    "symmetric Gaussian about photon arrival + walk with sigma(Q) = sqrt(1.5^2 + R(min(Q,4))^2)/2.355 ns, R the WCSim "
                    "14-point FWHM table interpolated with a not-a-knot cubic spline (ROOT TSpline3 without end conditions, as "
                    "TGraph::Eval(Q,0,\"S\")); mean shift 0; digits earlier than photon arrival counted as digits_before_arrival")
    cv["rng"] = cv["rng"].replace("seed/event/channel for charge, timing and dark counts",
                                  "seed/event/persistent photon ID for the source-PE SPE charge (spe_charge), seed/event/(channel, rank) "
                                  "for the dark-PE SPE charge (dark_spe_charge), seed/event/channel for timing and dark counts")
    cv["discriminator"] = ("T-Q mode: continuous PE peak, hard threshold .25 PE applied to the digitised (SPE-smeared, gain-scaled) charge "
                           "(author, R-7 v2 item 6); WCSim's SKI digitizer instead accepts with a probabilistic turn-on (recorded, not "
                           "applied); legacy threshold_walk mode retains sampled-template normalization")
    out["conventions"] = cv
    out["fit_config"], out["fixture"] = m["fit_config"], m["fixture"]
    out["wcsim_provenance"] = {
        "repository": WCSIM_REPO, "commit": WCSIM_COMMIT, "commit_date": "2026-10-06", "file": WCSIM_FILE,
        "file_sha256": WCSIM_SHA256, "class": CLASS, "lines": "3809-4020", "spe_sampler": RN1PE,
        "adopted": {
            "timing_width": "HitTimeSmearing: pmt_tts = 1.5 ns FWHM; 14-point resol table (constructor); Q clamped at 4 PE; Eval(Q,0,\"S\"); /2.355",
            "qe_curve": "GetQEWavelength/GetQE: 21 points 300-700 nm, raw values (the 1/0.73 correctionFactor not applied); zero outside 300-700 nm as the author instructs (edges at 299.99 and 700.01 nm; v2 domain anchors 200 and 800 nm kept)",
            "collection_efficiency": "GetCollectionEfficiencyArray: 95% in every angular bin",
            "spe_charge": "Getqpe: 500-value cumulative table (trailing dummy dropped), sampled as rn1pe does, mean divided out",
            "dark_rate_hz": "GetDarkRate: 1.0 kHz (WCSim: 'Realistic/Optimistic value ... ToDo')",
        },
        "recorded_not_applied": {
            "qe_correction_factor": "1/0.73 (WCSim: July 2015 workaround, PR #98, 'will be removed'); open question for Junjie",
            "dark_rate_conversion_factor": dr_factor,
            "GetmaxQE": "0.271 x 1/0.73",
            "qe_method_ratio": ("WCTE.mac sets /WCSim/PMTQEMethod SensitiveDetector_Only (method 3); WCSimWCSD.cc then calls GetPMTQE with "
                                "ratio = 1/(1-0.25), which multiplies QE, so WCSim's effective WCTE QE is raw x 1/0.73 x 4/3 (peak 0.492); "
                                "not applied; open question for Junjie"),
            "qe_outside_table": ("WCSimPMTQE.cc GetPMTQE (flag 1) returns 0 only at <= 200 nm or >= 660 nm; between 200 and 300 nm it "
                                 "extends the 300-320 nm segment linearly (QE 0.026 at 290 nm, positive down to 285.0 nm, negative and "
                                 "so in effect 0 below), and it cuts at 660 nm inside the table; not applied: the author asks for zero "
                                 "outside 300-700 nm"),
            "digitizer_ski": {
                "selected_by": "WCTE.mac /DAQ/Digitizer SKI",
                "threshold": ("WCSimWCDigitizerSKI::Threshold: probabilistic turn-on polynomial in pe+0.1 (about 54% acceptance at "
                              "0.25 PE, 100% at >= 1.0 PE); with this Getqpe table about 6.9% of 1-PE hits are lost, against 2.96% "
                              "for the hard 0.25 PE cut adopted here"),
                "charge_error_pe": "N(0, 0.03) added to accepted charge",
                "charge_efficiency": "0.985 multiplies the accepted charge",
                "timing_charge_floor_pe": ("HitTimeSmearing is called with Q = max(q, 0.5), so WCSim's digit sigma is flat at "
                                           "0.670 ns below 0.5 PE; here sigma follows q down to 0.25 PE (0.760 ns), per the author's 3 Nov 2026 decision"),
            },
        },
        "context_files_sha256": WCSIM_CONTEXT,
        "GetNbOfQEDefined": {"value": n_qe_defined, "table_points": len(qe),
                             "reading": "WCSimPMTQE.cc loops i < n and reads wavelength[i+1] ('off by one bug fix'), so n counts intervals and all 21 points are used; no point is dropped"},
        "checks": {
            "peak_qe": max(qe), "peak_qe_nm": wl[int(np.argmax(qe))], "collection_efficiency": ce[0] / 100,
            "peak_qe_times_ce": round(max(qe) * ce[0] / 100, 6), "required_band": [0.24, 0.28],
            "sigma_ns": {"0.2": None, "0.25": None, "0.5": None, "1.0": None, "4.0": None},
            "spe_table_raw_mean_pe": round(spe.raw_mean_pe, 6), "spe_relative_sigma": round(spe.relative_sigma, 6),
            "spe_fraction_below_0.25pe": {"normalised_charge": round(p_below_thr, 6), "raw_wcsim_charge": round(p_below_thr_raw, 6)},
        },
        "spe_fit_record": fit,
        "author_decisions_3_nov_2026": {
            "spe": "sample the WCSim 501-point table directly; do not fit",
            "sigma_at_0.25pe": "take WCSim's spline value 0.760 ns, not 0.77",
        },
        "wcte_fork_differences": {
            "repository": "https://github.com/WCTE/WCSim", "branch": "WCTE_mPMT", "commit": "c1ec455",
            "class": ("PMT3inchR14374 (no PMT3inchR14374_WCTE class on any branch). Main WCSim c5ed0b1 also contains this same "
                      "plain PMT3inchR14374 class; WCTE.mac selects PMT3inchR14374_WCTE (/WCSim/mPMT/PMTtype_inner), while "
                      "WCSimDetectorConfigs.cc defaults mPMTs to the plain class, so the real difference is _WCTE vs plain class, not main vs fork"),
            "file_sha256": "1cfbd68111d5609cf1711df749c0d2dfbd515e5e265164ccacf52d7fc4279106",
            "timing": "constant Gaussian sigma 0.6 ns at every charge (B. Quilain: TTS 1.4 ns at 1 PE), no charge dependence",
            "spe_table": "different Getqpe table: raw mean 1.0197 PE, relative sigma 0.433 (main WCSim: 1.0648, 0.4525); max |dCDF| 0.069",
            "dark_rate_conversion_factor": 1.110,
            "identical": "QE table and 1/0.73 factor, CE 95%, dark rate 1 kHz, GetNbOfQEDefined 20",
            "decision": "reported, not chosen between (R-7 v2)",
        },
    }
    out["cross_check"] = {"manifest": "configs/lucid_pmt/reference_v3_thesis_xcheck.json", "sha256": thesis_sha,
                          "content": "first R-7: Xia (2019) 20-inch proxies, split-normal Eq. 6.3, HK R12860 QE, CE 0.87, spe_sigma 0.30; never used by a result"}
    out["open_questions_for_junjie"] = [
        "whether WCTE production applies WCSim's 1/0.73 QE correction factor",
        "which PMT class WCTE production uses: PMT3inchR14374_WCTE (selected by WCSim's WCTE.mac, adopted here) or the plain PMT3inchR14374 (WCTE fork and WCSim config default: constant 0.6 ns, different SPE table, dark-rate conversion 1.110)",
        "whether WCTE production applies the extra QE ratio 1/(1-0.25) of PMTQEMethod SensitiveDetector_Only",
        "whether to adopt WCSim's SKI digitizer behaviour (soft threshold turn-on, 0.03 PE charge error, 0.985 charge factor, 0.5 PE floor on the timing charge) instead of the hard 0.25 PE cut",
        "measured WCTE values (T-Q walk, QE/CE, dark rate) to replace these simulation-level values",
    ]

    P = copy.deepcopy(m["parameters"])
    del P["spe_sigma"]

    def put(name, value, source, status=STATUS):
        P[name] = {"value": value, "source": source, "status": status, "sha256": value_sha256(value)}

    put("qe_curve", {"wavelength_nm": [200.0, 300.0 - QE_EDGE_NM] + wl + [700.0 + QE_EDGE_NM, 800.0],
                     "efficiency": [0.0, 0.0] + qe + [0.0, 0.0]},
        f"{SRC}, GetQEWavelength/GetQE (21 points 300-700 nm, raw peak 0.2695 at 380 nm); the legacy 1/0.73 correctionFactor "
        f"is recorded, not applied (open question for Junjie), and so is the PMTQEMethod ratio 1/(1-0.25); zero outside 300-700 nm as "
        f"the author instructs ({QE_EDGE_NM} nm edges; WCSim itself extends the first segment down to 285 nm and cuts at 660 nm, recorded); "
        f"v2 zero anchors at 200 and 800 nm keep the 200-800 nm domain; piecewise-linear inside the table as WCSimPMTQE interpolates")
    put("collection_efficiency", ce[0] / 100, f"{SRC}, GetCollectionEfficiencyArray (95% in every bin); peak QE x CE = {max(qe)} x {ce[0]/100} = {max(qe)*ce[0]/100:.4f}")
    put("spe_charge", {"kernel": "wcsim_qpe_table", "cdf": cdf, "bin_offset": 50, "bins_per_pe": 22.83, "normalise_mean": True},
        f"{SRC}, Getqpe (500 cumulative values, trailing dummy dropped), sampled as {RN1PE.split(' (')[0]} rn1pe does and divided by the "
        f"table mean {spe.raw_mean_pe:.4f} PE (relative sigma {spe.relative_sigma:.4f}); author decision 3 Nov 2026: sample directly because the "
        f"Gaussian+exponential fit missed the declared KS tolerance ({fit['ks_distance_to_table']} > {SPE_FIT_KS_TOL})")
    put("timing_width", {"kernel": "wcsim_tts_spline", "charge_pe": charge, "resolution_fwhm_ns": resol, "tts_fwhm_ns": 1.5,
                         "q_clamp_pe": 4.0, "fwhm_to_sigma": 2.355, "spline": "not-a-knot cubic (ROOT TSpline3 without end conditions)"},
        f"{SRC}, HitTimeSmearing: sigma(Q) = sqrt(1.5^2 + R(min(Q,4))^2)/2.355 ns, R the 14-point constructor table interpolated as "
        "TGraph::Eval(Q,0,\"S\") (not-a-knot cubic spline), evaluated at the mean-normalised digitised charge; author decision 3 Nov 2026: "
        "0.760 ns at 0.25 PE (the spline), not 0.77; WCSim's SKI digitizer would pass max(Q, 0.5) (0.670 ns below 0.5 PE), recorded, not applied")
    put("dark_rate_hz", 1000.0, f"{SRC}, GetDarkRate = 1.0 kHz (WCSim marks it 'Realistic/Optimistic ... ToDo'); value unchanged from R-6; "
        f"WCSim's dark-rate conversion factor {dr_factor} recorded, not applied")
    thr = dict(P["threshold_pe"])
    thr["source"] = (thr["source"] + "; R-7 v2 item 6: a hard cut applied to the digitised (SPE-smeared, gain-scaled) charge (readout "
                     "stage 5 already cut on that charge, so no code change); WCSim's SKI digitizer uses a probabilistic turn-on instead "
                     "(wcsim_provenance.recorded_not_applied.digitizer_ski)")
    P["threshold_pe"] = thr
    order = list(m["parameters"])
    out["parameters"] = {("spe_charge" if k == "spe_sigma" else k): P["spe_charge" if k == "spe_sigma" else k] for k in order}

    # sigma checks from the very kernel the readout uses
    from noise_module_lucid.timing_response import WCSimTTSWidth
    tw = WCSimTTSWidth(tuple(charge), tuple(resol), 1.5, 4.0, (0.25, 1000.0))
    out["wcsim_provenance"]["checks"]["sigma_ns"] = {"0.25": round(float(tw(0.25)), 4), "0.5": round(float(tw(0.5)), 4),
                                                     "1.0": round(float(tw(1.0)), 4), "4.0": round(float(tw(4.0)), 4),
                                                     "10.0": round(float(tw(10.0)), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wcsim_pmt_object", help="path to WCSim src/WCSimPMTObject.cc at commit c5ed0b1")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    text = json.dumps(build(a.wcsim_pmt_object), indent=2) + "\n"
    Path(a.out).write_text(text)
    print(f"wrote {a.out} sha256 {hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
