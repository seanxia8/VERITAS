"""Bounded prompt-only development driver. Never trains or marks evidence citable.

Run: python -m noise_module_lucid.pmt_development --out NEW_DIRECTORY
All IDs, tolerances and fixture sizes are fixed before validation.
"""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import numpy as np
from .calibration import fit_reference_calibration, event_digits, digest
from .consequence import FitConfig
from .dataset import ResponseCellSpec, run_response_matrix
from .group_response import GroupMap, group_intervention
from . import response_manifest
from .toy import cylinder_detector, prompt_source_event
from .timing_response import SplitNormalWidth


def qe_drift_shape(wavelength_nm):
    """The v1/v2 QE-shape anchors [200,300,400,600,800] -> [0,-1,0,1,0], resampled on any table."""
    return np.interp(np.asarray(wavelength_nm, dtype=float),
                     [200.0, 300.0, 400.0, 600.0, 800.0], [0.0, -1.0, 0.0, 1.0, 0.0])


def fixture():
    spec = response_manifest.load_manifest()["fixture"]
    det = cylinder_detector(4, 12, 2)
    mapping = GroupMap(tuple(np.arange(det.n_channels)//spec["channels_per_group"]), spec["group_provenance"], spec["group_source"])
    ref = response_manifest.reference_response(det.n_channels, np.asarray(mapping.channel_group))
    n = spec["photons_per_event"]
    fit = [prompt_source_event(det,10000+i,n_photons=n) for i in range(32)]
    closure = [prompt_source_event(det,20000+i,n_photons=n) for i in range(16)]
    evaluation = [prompt_source_event(det,30000+i,n_photons=n) for i in range(24)]
    return det,mapping,ref,fit,closure,evaluation


def run(out):
    out=Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start=time.monotonic()
    det,mapping,ref,fit,closure,events=fixture()
    cfg=FitConfig(grid=5)
    calibration=fit_reference_calibration(fit,det.positions,ref,fit_config=cfg)
    calibration.save(out/"reference_calibration.json")
    cfg=FitConfig(grid=5)
    base=ResponseCellSpec("reference","reference",None,ref,origin="clean")
    closure_result=run_response_matrix([base],closure,det.positions,out/"closure",fit_config=cfg,n_boot=0,calibration=calibration)
    cq=closure_result["cells"][0]
    # Predeclared engineering closure: 0 failures, <0.3 m mean error and <10% energy error.
    checks={"closure_no_failures":cq["K_vertex"]["n_failed_cell"]==0,
            "closure_vertex_under_0_3m":cq["K_vertex"]["mean_error_cell"]<.3,
            "closure_energy_under_10pct":cq["K_energy"]["mean_error_cell"]<.1}
    # D-TQ intervention: slope b*(1 +/- 0.2); the offset-only cell is a separate control.
    steeper,shallower,offset=ref.walk_curve.scaled(1.2),ref.walk_curve.scaled(0.8),ref.walk_curve.scaled(offset_ns=1.0)
    coupled=group_intervention(ref,mapping,[1,3],gain_factor=.8,width_factor=1.5)
    gain_only=group_intervention(ref,mapping,[1,3],gain_factor=.8)
    width_only=group_intervention(ref,mapping,[1,3],width_factor=1.5)
    # Exact product control changes QE and CE jointly but no operational origin class.
    exact=response_manifest.exact_product_control(ref)
    cells=[base,
        ResponseCellSpec("identity","identity",0,ref),
        ResponseCellSpec("qe_shape","qe_shape",.3,replace(ref,qe_curve=ref.qe_curve.drift(.3,qe_drift_shape(ref.qe_curve.wavelength_nm)))),
        ResponseCellSpec("fixed_tq_slope_up","fixed_tq",1.2,replace(ref,walk_curve=steeper)),
        ResponseCellSpec("fixed_tq_slope_down","fixed_tq",.8,replace(ref,walk_curve=shallower)),
        ResponseCellSpec("tq_offset_control","tq_offset_control",1.0,replace(ref,walk_curve=offset)),
        ResponseCellSpec("timing_width","timing_width",1.5,replace(ref,timing_width_factor=tuple(1.5*np.array(ref.timing_width_factor)))),
        ResponseCellSpec("group_coupled","group_response",1,coupled,affected_groups=(1,3),changed_fields=("gain","timing_width_factor"),causal_control="synthetic-group-strength"),
        ResponseCellSpec("group_gain","gain_drift",.8,gain_only,affected_groups=(1,3)),
        ResponseCellSpec("group_width","timing_width",1.5,width_only,affected_groups=(1,3))]
    summary=run_response_matrix(cells,events,det.positions,out/"matrix",fit_config=cfg,n_boot=100,calibration=calibration)
    spectrum=np.linspace(300,600,301)
    qe_integrals={"spectrum":"uniform 300..600 nm", "reference_probability":float(np.trapezoid(ref.qe_curve(spectrum)*ref.collection_efficiency,spectrum)/300),
                  "shape_drift_probability":float(np.trapezoid(cells[2].params.qe_curve(spectrum)*ref.collection_efficiency,spectrum)/300)}
    exact_equal=True
    for ev in events:
        a,b=event_digits(ev,det.n_channels,ref),event_digits(ev,det.n_channels,exact)
        exact_equal &= all(np.array_equal(getattr(a,k),getattr(b,k)) for k in ("sensor","charge","time"))
    checks["exact_degeneracy_equal"]=bool(exact_equal)
    # Weaker control: fit ONLY a charge-sum match on reference-fit data; evaluate residuals separately.
    qe_loss=replace(ref,collection_efficiency=.8*ref.collection_efficiency)
    q_ref=sum(event_digits(e,det.n_channels,ref).charge.sum() for e in fit)
    q_loss=sum(event_digits(e,det.n_channels,qe_loss).charge.sum() for e in fit)
    gain_matched=replace(ref,gain=tuple(np.array(ref.gain)*(q_loss/q_ref)))
    rows=[]
    spatial_a=np.zeros(det.n_channels);spatial_b=spatial_a.copy()
    for ev in events:
        a,b=event_digits(ev,det.n_channels,qe_loss),event_digits(ev,det.n_channels,gain_matched)
        common,ia,ib=np.intersect1d(a.sensor,b.sensor,return_indices=True)
        spatial_a[a.sensor]+=a.charge;spatial_b[b.sensor]+=b.charge
        rows.append({"event_id":ev.event_id,"charge_difference":float(a.charge.sum()-b.charge.sum()),
            "occupancy_difference":len(a)-len(b),"charge_std_difference":float(a.charge.std()-b.charge.std()),
            "common_time_rms_difference_ns":float(np.sqrt(np.mean((a.time[ia]-b.time[ib])**2)))})
    weak={"label":"charge_matched_not_exact", "fit_event_ids":list(calibration.event_ids),"gain_factor":float(q_loss/q_ref),
          "evaluation":rows,"spatial_charge_l1":float(np.abs(spatial_a-spatial_b).sum()),"exact_degeneracy_claim":False}
    (out/"controls.json").write_text(json.dumps({"exact_product_equal":exact_equal,"integrated_detection_probability":qe_integrals,"charge_matched":weak},indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    w=np.linspace(300,600,100);q=np.linspace(.25,10,100)
    axes[0,0].plot(w,ref.qe_curve(w),label="reference");axes[0,0].plot(w,cells[2].params.qe_curve(w),label="shape drift")
    axes[0,0].set(xlabel="Wavelength (nm)",ylabel="QE");axes[0,0].legend()
    axes[0,1].plot(q,ref.walk_curve(q),label="reference");axes[0,1].plot(q,steeper(q),label="slope x1.2");axes[0,1].plot(q,offset(q),label="offset control")
    axes[0,1].set(xlabel="Charge (PE)",ylabel="Raw walk location (ns)");axes[0,1].legend()
    tw=ref.timing_width
    if isinstance(tw,SplitNormalWidth):
        axes[0,2].plot(q,tw.sigma_e(q),label="early sigma_e");axes[0,2].plot(q,tw.sigma_l(q),label="late sigma_l");axes[0,2].legend()
    else:
        axes[0,2].plot(q,tw(q))
    axes[0,2].set(xlabel="Charge (PE)",ylabel="Digit-width sigma (ns)")
    axes[1,0].plot(coupled.gain,label="gain");axes[1,0].plot(coupled.timing_width_factor,label="width factor")
    axes[1,0].set(xlabel="Synthetic channel",ylabel="Factor");axes[1,0].legend()
    axes[1,1].bar(["vertex (m)","energy (relative)"],[cq["K_vertex"]["mean_error_cell"],cq["K_energy"]["mean_error_cell"]]);axes[1,1].set_title("Independent reference closure")
    axes[1,2].hist([row["charge_difference"] for row in rows],bins=8);axes[1,2].set(xlabel="Held-out charge difference (PE)",ylabel="Events",title="Charge match is not exact")
    fig.suptitle("Synthetic PMT engineering development — non-citable")
    fig.tight_layout();fig.savefig(out/"diagnostics.png",dpi=130);plt.close(fig)
    files=list(Path("src/noise_module_lucid").rglob("*.py"))+list(Path("configs/lucid_pmt").glob("*.json"))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    report={"checks":checks,"evidence_stage":"development","non_citable":True,"source_kind":"synthetic_prompt_flasher",
        "physics_review_status":"pending","protocol_freeze_id":None,"training_run":False,"claim_comparisons":"not_run",
        "partitions":{"reference_fit":list(calibration.event_ids),"closure":[e.event_id for e in closure],"evaluation":[e.event_id for e in events]},
        "reference":ref.to_dict(),"response_manifest":response_manifest.provenance(),"group_map":asdict(mapping),"fit_config":asdict(cfg),"code_config_sha256":hashes,
        "head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "python":sys.executable,"python_version":platform.python_version(),"numpy":np.__version__,
        "command":sys.argv,"elapsed_seconds":time.monotonic()-start}
    (out/"validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"checks":checks,"elapsed_seconds":report["elapsed_seconds"]}))
    if not all(checks.values()):
        raise RuntimeError("predeclared engineering validation failed; preserve evidence")
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)
