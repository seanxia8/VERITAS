"""R-1 acceptance: one response manifest; no module builds a response from literals."""
import ast
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from noise_module_lucid import response_manifest
from noise_module_lucid.readout import ReadoutParams
from noise_module_lucid.timing_response import InverseSqrtWalk, SplitNormalWidth

PACKAGE = Path(response_manifest.__file__).parent
CTORS = {"ReadoutParams", "WalkCurve", "InverseSqrtWalk", "WidthCurve", "SplitNormalWidth", "QECurve", "ChannelWalkCurve"}
RESPONSE_FIELDS = set(ReadoutParams.__dataclass_fields__)
# Modules that define the response types or build the reference from the manifest.
BUILDERS = {"response_manifest.py", "timing_response.py", "quantum_efficiency.py"}
# Bare ReadoutParams() is the legacy toy chain (no QE, T-Q or width model), kept for historical results.
LEGACY_DEFAULT = {"dataset.py", "readout.py"}
# Declared cell/control constructions outside a ResponseCellSpec(...) call; each is an intervention, not a reference.
DECLARED = {("pmt_development.py", "collection_efficiency"): "charge-matched weak control: 20% CE loss",
            ("multienergy_data.py", "dark_rate_hz"): "R-4b 2x2 diagnostic dark-counts-off arm (author, 18 Oct 2026)",
            **{("dataset.py", f): "legacy response_cells: identity value (1.0/0.0) for unaffected groups inside a cell"
               for f in ("survival", "gain", "timing_offsets_ns")}}


def _numeric(node):
    return any(isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) for n in ast.walk(node))


def literal_responses(source, filename="<src>"):
    """Response constructors outside the builders, and literal-valued response-field replacements
    outside a ``ResponseCellSpec(...)`` cell definition or a declared construction.

    Known blind spot: a literal first bound to a name and then passed in is not traced.
    """
    tree = ast.parse(source)
    in_cell = {id(n) for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "ResponseCellSpec"
               for n in ast.walk(c) if n is not c}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", getattr(node.func, "attr", None))
        if name in CTORS and filename not in BUILDERS:
            if not (name == "ReadoutParams" and not node.args and not node.keywords and filename in LEGACY_DEFAULT):
                found.append((filename, node.lineno, name))
        if name in ("replace", "with_change") and id(node) not in in_cell:
            for k in node.keywords:
                if k.arg in RESPONSE_FIELDS and _numeric(k.value) and (filename, k.arg) not in DECLARED:
                    found.append((filename, node.lineno, f"{name}:{k.arg}"))
    return found


def test_no_module_constructs_a_response_from_literals():
    found = [hit for p in sorted(PACKAGE.glob("*.py")) for hit in literal_responses(p.read_text(), p.name)]
    assert found == []


@pytest.mark.parametrize("source", [
    "ReadoutParams(qe_curve=QECurve((300,400,600),(.25,.5,.25)))",
    "WalkCurve((2+.05*(i%3),-1),(.25,1000))",
    "InverseSqrtWalk()",
    "ReadoutParams(gain=tuple(1+.03*(g%2) for g in mapping.channel_group))",
    "QECurve(ref.qe_curve.wavelength_nm,tuple(2*np.asarray(ref.qe_curve.efficiency)),ref.qe_curve.source)",
    "replace(ref,gain=tuple(1+.03*(inverse%2)))",
    "replace(ref,timing_width_factor=tuple(1+.05*(np.unique(g.detector.mpmt_id,return_inverse=True)[1]%2)))",
    "replace(ref,collection_efficiency=.64)",
    "with_change(ref,gain=0.8)",
    "replace(ref,dead_channels=(1,2))",
])
def test_scan_detects_literal_responses(source):
    assert literal_responses(source)


@pytest.mark.parametrize("source", [
    "ResponseCellSpec('w','timing_width',1.5,replace(ref,timing_width_factor=tuple(1.5*np.array(ref.timing_width_factor))))",
    "replace(ref,walk_curve=ref.walk_curve.scaled(factor))",
    "response_manifest.exact_product_control(ref)",
])
def test_scan_allows_cells_and_manifest_derivations(source):
    assert not literal_responses(source)


def test_pilots_build_from_the_manifest():
    for name in ("lucid_integration", "lucid_controls", "multienergy_data", "pmt_development"):
        text = (PACKAGE / f"{name}.py").read_text()
        assert "response_manifest" in text, name


def test_manifest_hashes_domain_and_fixture():
    m = response_manifest.load_manifest()
    # R-7 (23 Oct 2026): v3 is the default; v2 stays loadable and hash-pinned.
    assert m["schema"] == "pmt-response-v3" and m["status"] == "literature_proxy_dev"
    assert m["parameters"]["charge_domain_pe"]["value"] == [0.25, 1000.0]
    assert m["parameters"]["walk"]["value"] == {"form": "inverse_sqrt", "a_ns": 0.0, "b_ns": 2.0, "q_sat_pe": 50.0}
    assert m["fixture"]["photons_per_event"] == 4000
    assert all(e["status"] and e["source"] for e in m["parameters"].values())


def test_tampered_or_incomplete_manifest_is_rejected(tmp_path):
    m = json.loads(response_manifest.DEFAULT_PATH.read_text())
    bad = json.loads(json.dumps(m)); bad["parameters"]["walk"]["value"]["b_ns"] = 3.0
    (tmp_path / "a.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="hash"): response_manifest.load_manifest(tmp_path / "a.json")
    bad = json.loads(json.dumps(m)); del bad["parameters"]["gain"]
    (tmp_path / "b.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="exactly"): response_manifest.load_manifest(tmp_path / "b.json")
    bad = json.loads(json.dumps(m)); bad["schema"] = "pmt-response-v1"
    (tmp_path / "c.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="schema"): response_manifest.load_manifest(tmp_path / "c.json")


def test_reference_response_any_size_and_group_rule():
    for n, group in ((76, np.arange(76)//12), (1995, response_manifest.module_groups(np.repeat(np.arange(105)*7, 19)))):
        p = response_manifest.reference_response(n, group)
        assert isinstance(p, ReadoutParams) and len(p.gain) == n and p.charge_domain_pe == (0.25, 1000.0)
        assert isinstance(p.timing_width, SplitNormalWidth)
        np.testing.assert_allclose(p.gain, 1+.03*(np.asarray(group) % 2))
        np.testing.assert_allclose(p.timing_width_factor, 1+.05*(np.asarray(group) % 2))
        assert p.walk_curve == p.reference_walk == InverseSqrtWalk(0.0, 2.0, 50.0, (0.25, 1000.0))
    with pytest.raises(ValueError): response_manifest.reference_response(5, np.arange(4))
    with pytest.raises(ValueError): response_manifest.reference_response(2, np.array([0.0, 1.0]))


def test_fixture_reference_is_the_manifest_response():
    from noise_module_lucid.pmt_development import fixture
    from noise_module_lucid.toy import prompt_source_event
    det, mapping, ref, fit, closure, evaluation = fixture()
    assert ref == response_manifest.reference_response(det.n_channels, np.asarray(mapping.channel_group))
    expected = prompt_source_event(det, fit[0].event_id, n_photons=4000)
    assert det.n_channels == 76
    np.testing.assert_array_equal(fit[0].photon_batch.photon_id, expected.photon_batch.photon_id)
