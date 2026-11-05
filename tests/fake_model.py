"""A stand-in for the Anthropic client that replays scripted turns, so the
agent loop, event stream and server can be tested offline."""
from __future__ import annotations

import itertools
from types import SimpleNamespace


class _Block(SimpleNamespace):
    pass


class _Stream:
    def __init__(self, turn):
        self.turn = turn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for b in self.turn:
            if b["type"] == "text":
                yield SimpleNamespace(type="text", text=b["text"])

    def get_final_message(self):
        blocks = []
        for i, b in enumerate(self.turn):
            if b["type"] == "text":
                blocks.append(_Block(type="text", text=b["text"]))
            else:
                blocks.append(_Block(type="tool_use", id=f"tu_{id(self)}_{i}", name=b["name"], input=b["input"]))
        stop = "tool_use" if any(b["type"] == "tool_use" for b in self.turn) else "end_turn"
        return SimpleNamespace(content=blocks, stop_reason=stop)


class FakeClient:
    """`turns` is a list of turns; each turn is a list of blocks:
    {"type": "text", "text": ...} or {"type": "tool_use", "name": ..., "input": {...}}."""

    def __init__(self, turns):
        self._turns = itertools.chain(turns, itertools.repeat([{"type": "text", "text": "(no more scripted turns)"}]))
        self.messages = self
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _Stream(next(self._turns))


BRACKET_WRONG = """
P = dict(L=60, W=40, T=8, boss_r=8, boss_h=18, hole_d=8)
base = box((P['L'], P['W'], P['T']))
boss = cylinder(P['boss_r'], P['boss_h'], at=(P['L']/2, P['W']/2, P['T']))
body = union(base, boss)
# BUG: hole only 10 mm deep, does not go through
body = difference(body, cylinder(P['hole_d']/2, 10, at=(P['L']/2, P['W']/2, P['T']+P['boss_h']-10)))
part = body
"""

BRACKET_FIXED = """
P = dict(L=60, W=40, T=8, boss_r=8, boss_h=18, hole_d=8, r_in=2)
base = box((P['L'], P['W'], P['T']))
boss = cylinder(P['boss_r'], P['boss_h'], at=(P['L']/2, P['W']/2, P['T']))
body = union(base, boss)
body = difference(body, cylinder(P['hole_d']/2, 60, at=(P['L']/2, P['W']/2, -1)))
body = fillet(body, edges(body, convex=False, circular=True), P['r_in'])
part = body
"""

SCRIPTED_BRACKET = [
    [{"type": "text", "text": "I'll model the bracket and verify it against a checklist."},
     {"type": "tool_use", "name": "plan", "input": {"requirements": ["plate 60 x 40 x 8 mm", "boss Ø16 x 18 mm on the plate centre", "Ø8 hole through the boss and plate", "2 mm fillet at the boss-to-plate junction"], "assumptions": ["part sits on Z=0", "hole is a plain through hole"]}}],
    [{"type": "tool_use", "name": "build", "input": {"script": BRACKET_WRONG, "note": "first attempt"}}],
    [{"type": "text", "text": "Let me verify the hole goes through."},
     {"type": "tool_use", "name": "measure", "input": {"kind": "thickness", "point": [30, 20, -1], "direction": [0, 0, 1]}}],
    [{"type": "text", "text": "The ray hits material: the hole is blind. Fixing."},
     {"type": "tool_use", "name": "build", "input": {"script": BRACKET_FIXED, "note": "through hole + fillet"}}],
    [{"type": "tool_use", "name": "look", "input": {"section_point": [30, 20, 0], "section_normal": [0, 1, 0]}}],
    [{"type": "tool_use", "name": "measure", "input": {"kind": "thickness", "point": [30, 20, -1], "direction": [0, 0, 1]}}],
    [{"type": "tool_use", "name": "finish", "input": {"summary": "A 60 x 40 x 8 mm plate with a central Ø16 boss, Ø8 through hole and a 2 mm junction fillet. The first attempt had a blind hole, caught by a thickness ray and fixed.", "report": [
        {"id": "R1", "requirement": "plate 60 x 40 x 8 mm", "method": "metrics bbox", "result": "60 x 40 x 26 overall, plate faces F0/F1 at z=0 and z=8", "pass": True},
        {"id": "R2", "requirement": "boss Ø16 x 18 mm on the plate centre", "method": "census: cylindrical face r=8 axis (0,0,1) centre (30,20)", "result": "r=8, top at z=26", "pass": True},
        {"id": "R3", "requirement": "Ø8 hole through the boss and plate", "method": "measure thickness ray along the hole axis", "result": "no material hits", "pass": True},
        {"id": "R4", "requirement": "2 mm fillet at the boss-to-plate junction", "method": "census: concave circular edge replaced by toroidal face", "result": "fillet present", "pass": True}]}}],
    [{"type": "text", "text": "Done: the bracket is verified."}],
]
