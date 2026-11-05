"""Two-part fit checks: a case with a snap-fit lid, built correctly and with the
two mistakes that motivated this feature (tabs on the wrong side, lid mis-sized)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_model import FakeClient  # noqa: E402

from foreman.agent.loop import Session  # noqa: E402
from foreman.kernel import assembly as asm  # noqa: E402
from foreman.kernel.dsl import SCRIPT_NAMESPACE  # noqa: E402


def case_script(tabs="out", clear=0.2):
    ys = "(-P['tab_d'], P['W'])" if tabs == "out" else "(P['t'], P['W'] - P['t'] - P['tab_d'])"
    return f"""
P = dict(L=60, W=40, H=20, t=2, lid_h=10, lid_t=2, clear={clear}, tab_w=10, tab_h=2, tab_d=0.6, tab_z=15)
shell_box = box((P['L'], P['W'], P['H']))
base = shell(shell_box, P['t'], faces(shell_box, normal='+z'))
base = union(base, [box((P['tab_w'], P['tab_d'], P['tab_h']), at=(P['L']/2 - P['tab_w']/2, y, P['tab_z'])) for y in {ys}])
Li, Wi = P['L'] + 2*P['clear'], P['W'] + 2*P['clear']
outer = box((Li + 2*P['lid_t'], Wi + 2*P['lid_t'], P['lid_h']), at=(-P['clear'] - P['lid_t'], -P['clear'] - P['lid_t'], P['H'] - P['lid_h'] + P['lid_t']))
lid = difference(outer, box((Li, Wi, P['lid_h']), at=(-P['clear'], -P['clear'], P['H'] - P['lid_h'])))
assembly = dict(base=base, lid=lid)
part = base
"""


def run(script):
    ns = dict(SCRIPT_NAMESPACE)
    exec(script, ns)
    return ns


def test_correct_snap_fit_interferes_only_at_the_tabs():
    ns = run(case_script("out", 0.2))
    f = asm.fit(ns["base"], ns["lid"])
    assert f["relationship"] == "interpenetrating"
    # two tabs, each 10 x 2 mm, overlapping the lid wall by 0.6 - 0.2 = 0.4 mm
    assert f["interference_mm3"] == pytest.approx(2 * 10 * 2 * 0.4, rel=1e-3)
    bb = f["interference_bbox"]
    assert bb["min"][2] == pytest.approx(15) and bb["max"][2] == pytest.approx(17)  # at tab height only
    assert bb["min"][0] == pytest.approx(25) and bb["max"][0] == pytest.approx(35)  # at tab width only


def test_tabs_on_the_wrong_side_never_engage():
    ns = run(case_script("in", 0.2))
    f = asm.fit(ns["base"], ns["lid"])
    assert f["relationship"] in ("separate", "touching")
    assert f["interference_mm3"] == 0.0


def test_undersized_lid_interferes_along_the_whole_wall():
    ns = run(case_script("out", -0.3))
    f = asm.fit(ns["base"], ns["lid"])
    assert f["interference_mm3"] > 400
    bb = f["interference_bbox"]
    assert bb["max"][0] - bb["min"][0] == pytest.approx(60, abs=0.01)  # full wall length, not just the tabs


def test_assembly_build_reports_fit_and_warns_the_model(tmp_path):
    s = Session(client=FakeClient([[{"type": "tool_use", "name": "build", "input": {"script": case_script("in", 0.2), "note": "tabs inside"}}],
                                   [{"type": "tool_use", "name": "measure", "input": {"kind": "fit", "a": "base", "b": "lid"}}],
                                   [{"type": "text", "text": "."}]]), model="fake")
    try:
        events = list(s.run("snap-fit case"))
        b = [e for e in events if e["type"] == "build"][0]
        assert b["ok"] and b["metrics"]["fits"][0]["interference_mm3"] == 0.0
        assert [o["name"] for o in b["viewer"]["others"]] == ["lid"]
        first_result = [m for m in s.messages if m["role"] == "user"][1]["content"][0]["content"][0]["text"]
        assert "FIT between bodies" in first_result
        assert "snap features are not engaging" in first_result
        m = [e for e in events if e["type"] == "measure"][0]
        assert m["result"]["relationship"] in ("separate", "touching")
        out = tmp_path / "case.usda"
        s.export_usd(str(out))
        import usd_brep as sm
        assert len(sm.usd.import_breps(str(out))) == 2  # both bodies exported as exact B-reps
    finally:
        s.close()


def test_part_must_be_in_assembly():
    with pytest.raises(ValueError, match="must be one of the bodies"):
        b = SCRIPT_NAMESPACE["box"]((1, 1, 1))
        asm.validate({"a": b}, SCRIPT_NAMESPACE["box"]((2, 2, 2)))
