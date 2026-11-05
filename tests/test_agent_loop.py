import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_model import SCRIPTED_BRACKET, FakeClient  # noqa: E402

from foreman.agent.loop import Session  # noqa: E402


@pytest.fixture(scope="module")
def session():
    s = Session(client=FakeClient(SCRIPTED_BRACKET), model="fake")
    yield s
    s.close()


def test_scripted_bracket_session(session, tmp_path):
    events = list(session.run("Make the bracket."))
    types = [e["type"] for e in events]
    builds = [e for e in events if e["type"] == "build"]
    assert len(builds) == 2 and all(b["ok"] for b in builds)
    # the first (buggy) build has a blind hole -> material along the axis
    measures = [e for e in events if e["type"] == "measure"]
    assert measures[0]["result"]["material_spans_mm"], "blind hole should show material along the axis"
    assert measures[-1]["result"]["material_spans_mm"] == [], "fixed hole must be through"
    images = [e for e in events if e["type"] == "image"]
    assert len(images) >= 3  # two build renders + one section
    assert all(base64.b64decode(i["png_b64"])[:8] == b"\x89PNG\r\n\x1a\n" for i in images)
    assert "finish" in types and types[-1] == "done"
    assert session.finished["report"][1]["pass"] is True
    # tool results were fed back to the model with images, older ones pruned
    n_img = sum(1 for m in session.messages if m["role"] == "user"
                for item in m["content"] if item.get("type") == "tool_result"
                for b in item["content"] if b.get("type") == "image")
    assert n_img <= 2
    # export the verified part as USD with BrepArray prims
    out = tmp_path / "part.usda"
    session.export_usd(str(out))
    txt = out.read_text()
    assert "BrepArray" in txt and "Mesh" in txt


def test_build_failure_is_reported():
    s = Session(client=FakeClient([[{"type": "tool_use", "name": "build", "input": {"script": "part = fillet(box((10,10,10)), edges(box((10,10,10))), 9)", "note": "too big"}}],
                                   [{"type": "text", "text": "ok"}]]), model="fake")
    try:
        events = list(s.run("x"))
        b = [e for e in events if e["type"] == "build"][0]
        assert b["ok"] is False and "implausible" in b["error"]
        # the model is told about the failure
        last_tool_result = [m for m in s.messages if m["role"] == "user"][-1]["content"][0]
        assert "BUILD FAILED" in last_tool_result["content"][0]["text"]
    finally:
        s.close()


def test_review_mode_imports_part_and_briefs_model(tmp_path):
    import usd_brep as sm
    path = str(tmp_path / "in.usda")
    sm.usd.export_brep(sm.boolean_difference(sm.create_box((0, 0, 0), 40, 30, 10), sm.create_cylinder((20, 15, -1), 4, 20)), path)
    s = Session(client=FakeClient([[{"type": "tool_use", "name": "build", "input": {
        "script": "part = fillet(imported, edges(imported, direction='z', straight=True), 3)", "note": "round verticals"}}],
        [{"type": "text", "text": "done"}]]), model="fake")
    try:
        events = s.load_usd(path, display_name="in.usda")
        assert [e["type"] for e in events] == ["build", "image"]
        assert events[0]["metrics"]["faces"] == 7
        assert "in.usda" in s.messages[0]["content"][0]["text"]
        out = list(s.run("round the vertical edges"))
        b = [e for e in out if e["type"] == "build"][0]
        assert b["ok"] and b["metrics"]["faces"] == 11
    finally:
        s.close()
