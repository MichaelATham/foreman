"""
The Foreman agent loop.

A `Session` owns one kernel worker, the conversation with the model, and the
version history of the feature script. `run(user_message)` is a generator that
yields UI events while the model plans, builds, inspects and verifies:

    {"type": "text", "text": ...}                      streamed assistant prose
    {"type": "tool_call", "name", "input", "id"}       the model decided to act
    {"type": "build", "version", "metrics", "viewer", "script", "note", "error"}
    {"type": "image", "png_b64", "caption"}            what the model is looking at
    {"type": "measure", "result"}
    {"type": "finish", "summary", "report"}
    {"type": "done"} / {"type": "error", "message"}

Context hygiene: every build returns an image (vision is the point), but images
are the expensive part of the context, so after each turn we replace images in
tool results older than the two most recent ones with a short text stub.
"""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Iterator

from ..kernel.worker import KernelError, KernelTimeout, KernelWorker
from . import prompts
from .providers import AnthropicProvider, make_provider
from .tools import TOOLS

MAX_STEPS = int(os.environ.get("FOREMAN_MAX_STEPS", "14"))


def _sanity_warnings(met: dict, census: dict) -> list[str]:
    """Cheap automatic checks appended to every build result."""
    out = []
    if not met.get("manifold_solid"):
        out.append("the result is NOT a closed manifold solid - a boolean or sweep produced an open/invalid body")
    if met.get("bbox_min") and met["bbox_min"][2] < -1e-3:
        out.append(f"the part extends below Z=0 (min z = {met['bbox_min'][2]})")
    tiny = [f["id"] for f in census["faces"] if f["area"] < 0.05]
    if tiny:
        out.append(f"{len(tiny)} sliver face(s) with area < 0.05 mm2 ({', '.join(tiny[:6])}) - often a boolean of nearly-coincident faces")
    for f in met.get("fits") or []:
        if f.get("interference_mm3"):
            bb = f.get("interference_bbox", {})
            out.append(f"{f['a']} and {f['b']} INTERFERE by {f['interference_mm3']} mm3 in bbox {bb.get('min')} -> {bb.get('max')}. "
                       "That is only correct if it is a deliberate snap/press feature at exactly that location.")
        elif f["relationship"] in ("separate", "touching"):
            out.append(f"{f['a']} and {f['b']} do not interfere ({f['relationship']}, min gap {f['min_gap_mm']} mm). "
                       "If they are meant to snap or press together, the snap features are not engaging.")
    concave = [e for e in census["edges"] if e.get("convex") is False]
    if concave:
        desc = ", ".join(f"{e['id']} ({e['type']}" + (f" r={e.get('radius')}" if e.get("radius") is not None else "") + f" at z={e['mid'][2]})" for e in concave[:8])
        more = f" and {len(concave) - 8} more" if len(concave) > 8 else ""
        out.append(f"SHARP INSIDE CORNERS remain at {len(concave)} concave edge(s): {desc}{more}. "
                   "If the brief asks for a fillet at a junction, these edges must disappear (replaced by a blend face).")
    return out


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _image_block(png: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.b64encode(png).decode()}}


class Session:
    def __init__(self, *, model: str | None = None, client=None, provider=None,
                 worker: KernelWorker | None = None, quality: str = "medium"):
        # `client` keeps the test seam: any object with .messages.stream(...)
        if provider is None:
            provider = AnthropicProvider(model or "fake", client=client) if client is not None else make_provider(model)
        self.provider = provider
        self.model = provider.model
        self.worker = worker or KernelWorker()
        self.quality = quality
        self.messages: list[dict] = []
        self.versions: list[dict] = []  # {version, script, note, metrics, ok, error}
        self.system = prompts.system_prompt()
        self.finished: dict | None = None
        self.plan: dict | None = None

    # ------------------------------------------------------------------ tools
    def _tool_build(self, inp: dict) -> tuple[list[dict], list[dict]]:
        script = inp.get("script", "")
        note = inp.get("note", "")
        version = len(self.versions) + 1
        events: list[dict] = []
        t0 = time.time()
        try:
            res = self.worker.build(script, quality=self.quality)
        except (KernelError, KernelTimeout) as e:
            err = str(e)
            self.versions.append({"version": version, "script": script, "note": note, "ok": False, "error": err})
            events.append({"type": "build", "version": version, "ok": False, "error": err, "script": script, "note": note,
                           "seconds": round(time.time() - t0, 2)})
            tb = getattr(e, "traceback_text", "")
            tb = ("\n" + tb[-1200:]) if tb and "feature-script" in tb else ""
            return [_text_block(f"BUILD FAILED (v{version}):\n{err}{tb}\n\nFix the script and call build again.")], events

        met = res["metrics"]
        warnings = _sanity_warnings(met, res["census"])
        caption = f"v{version}  size {met.get('size')} mm  volume {met.get('volume_mm3')} mm3  manifold={met.get('manifold_solid')}"
        try:
            png = self.worker.render(caption=caption)
        except Exception as e:  # noqa: BLE001
            png = None
            caption += f"  (render failed: {e})"
        self.versions.append({"version": version, "script": script, "note": note, "ok": True, "metrics": met})
        events.append({"type": "build", "version": version, "ok": True, "metrics": met, "viewer": res["viewer"],
                       "script": script, "note": note, "seconds": round(time.time() - t0, 2),
                       "census": res["census"]})
        if png:
            events.append({"type": "image", "png_b64": base64.b64encode(png).decode(), "caption": caption, "version": version})
        text = (
            f"BUILD OK (v{version}) in {res['build_seconds']}s\n"
            f"metrics: {json.dumps(met)}\n"
            + ("".join(f"WARNING: {w}\n" for w in warnings))
            + (f"stdout:\n{res['stdout']}\n" if res["stdout"].strip() else "")
            + "census:\n" + res["census_text"]
            + "\n\nThe image shows iso / front / top / right views with face ids. Now verify against your checklist."
        )
        blocks = [_text_block(text)]
        if png:
            blocks.append(_image_block(png))
        return blocks, events

    def _tool_look(self, inp: dict) -> tuple[list[dict], list[dict]]:
        events: list[dict] = []
        blocks: list[dict] = []
        if inp.get("section_point") and inp.get("section_normal"):
            try:
                res = self.worker.section(inp["section_point"], inp["section_normal"])
            except (KernelError, KernelTimeout) as e:
                return [_text_block(f"section failed: {e}")], events
            cap = f"section through {inp['section_point']} normal {inp['section_normal']}"
            events.append({"type": "image", "png_b64": base64.b64encode(res["png"]).decode(), "caption": cap})
            blocks.append(_text_block(f"Section ({res['segments']} segments). Grid lines every 5 mm. The outline is the "
                                      "material boundary in the cutting plane; closed loops inside the outer loop are voids."))
            blocks.append(_image_block(res["png"]))
            return blocks, events
        try:
            png = self.worker.render(highlight=inp.get("highlight_faces"), views=inp.get("views"),
                                     caption="highlight: " + ", ".join(inp.get("highlight_faces") or []) if inp.get("highlight_faces") else "")
        except (KernelError, KernelTimeout) as e:
            return [_text_block(f"render failed: {e}")], events
        events.append({"type": "image", "png_b64": base64.b64encode(png).decode(),
                       "caption": "look: " + ", ".join(inp.get("highlight_faces") or inp.get("views") or ["all"])})
        return [_text_block("Rendered."), _image_block(png)], events

    def _tool_measure(self, inp: dict) -> tuple[list[dict], list[dict]]:
        kind = inp.get("kind")
        kw = {k: v for k, v in inp.items() if k != "kind" and v is not None}
        try:
            res = self.worker.measure(kind, **kw)
        except (KernelError, KernelTimeout) as e:
            res = {"error": str(e)}
        return [_text_block(json.dumps(res))], [{"type": "measure", "kind": kind, "input": kw, "result": res}]

    def _tool_plan(self, inp: dict) -> tuple[list[dict], list[dict]]:
        reqs = [str(r) for r in inp.get("requirements", [])]
        self.plan = {"requirements": [{"id": f"R{i + 1}", "text": r} for i, r in enumerate(reqs)],
                     "assumptions": [str(a) for a in inp.get("assumptions", [])]}
        listing = "\n".join(f"  {r['id']}: {r['text']}" for r in self.plan["requirements"])
        return [_text_block(f"Checklist recorded:\n{listing}\nNow build the part.")], [{"type": "plan", **self.plan}]

    def _tool_finish(self, inp: dict) -> tuple[list[dict], list[dict]]:
        self.finished = {"summary": inp.get("summary", ""), "report": inp.get("report", [])}
        return [_text_block("Report recorded. Reply to the user with a brief wrap-up.")], [
            {"type": "finish", **self.finished, "model": f"{self.provider.name}/{self.model}"}]

    # ------------------------------------------------------------------- loop
    def run(self, user_message: str) -> Iterator[dict]:
        self.finished = None
        self.messages.append({"role": "user", "content": [_text_block(user_message)]})
        for step in range(MAX_STEPS):
            content_blocks: list[dict] = []
            tool_uses: list[dict] = []
            try:
                with self.provider.stream(self.system, self.messages, TOOLS) as stream:
                    for event in stream:
                        if event.type == "text":
                            yield {"type": "text", "text": event.text}
                    final = stream.get_final_message()
            except Exception as e:  # noqa: BLE001  (provider API errors)
                yield {"type": "error", "message": f"model API error ({self.provider.name}): {e}"}
                return
            for block in final.content:
                if block.type == "text":
                    content_blocks.append(_text_block(block.text))
                elif block.type == "tool_use":
                    content_blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
                    tool_uses.append({"id": block.id, "name": block.name, "input": block.input})
            self.messages.append({"role": "assistant", "content": content_blocks})
            if not tool_uses:
                break
            results = []
            for tu in tool_uses:
                yield {"type": "tool_call", "id": tu["id"], "name": tu["name"],
                       "input": {k: v for k, v in tu["input"].items() if k != "script"} | (
                           {"script": tu["input"]["script"]} if "script" in tu["input"] else {})}
                handler = {"plan": self._tool_plan, "build": self._tool_build, "look": self._tool_look,
                           "measure": self._tool_measure, "finish": self._tool_finish}.get(tu["name"])
                if handler is None:
                    blocks, events = [_text_block(f"unknown tool {tu['name']}")], []
                else:
                    blocks, events = handler(tu["input"])
                for ev in events:
                    yield ev
                results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": blocks})
            self.messages.append({"role": "user", "content": results})
            self._prune_images(keep=2)
            if self.finished is not None and final.stop_reason == "tool_use":
                # let the model write its wrap-up, then stop
                continue
        yield {"type": "done", "finished": self.finished}

    def _prune_images(self, keep: int = 2):
        """Replace all but the last `keep` images in tool results with a stub."""
        seen = 0
        for msg in reversed(self.messages):
            if msg["role"] != "user":
                continue
            for item in msg["content"]:
                if item.get("type") != "tool_result":
                    continue
                for i, blk in enumerate(item["content"]):
                    if blk.get("type") == "image":
                        seen += 1
                        if seen > keep:
                            item["content"][i] = _text_block("[image removed from context to save space - call look to re-render]")

    # ---------------------------------------------------------------- import
    def load_usd(self, path: str, display_name: str | None = None) -> list[dict]:
        """Import an existing USD BRep part as the current part (review mode).
        Returns UI events; the model is briefed with census + image so a follow-up
        like "audit this for 2 mm minimum walls" can start immediately."""
        name = display_name or os.path.basename(path)
        info = self.worker.load_usd(path)
        script = f"# loaded from {name} ({info['bodies']} body/bodies); edit below to modify it\npart = imported\n"
        blocks, events = self._tool_build({"script": script, "note": f"imported {name}"})
        text = "".join(b["text"] for b in blocks if b.get("type") == "text")
        img = [b for b in blocks if b.get("type") == "image"]
        self.messages.append({"role": "user", "content": [_text_block(
            f"[The user loaded an existing part from USD file '{name}' into the session. It is now the current part, "
            f"available to scripts as `imported` (all bodies: `imported_bodies`). You did not design it. "
            f"Here is its build result:\n{text}\nWait for the user's request (e.g. an audit, or a modification) before acting.]")] + img})
        self.messages.append({"role": "assistant", "content": [_text_block(
            f"Loaded '{name}'. I have its metrics, census and views. Tell me what to check or change.")]})
        self._prune_images(keep=2)
        return events

    # ---------------------------------------------------------------- helpers
    @property
    def current_script(self) -> str | None:
        for v in reversed(self.versions):
            if v["ok"]:
                return v["script"]
        return None

    def export_usd(self, path: str) -> dict:
        return self.worker.export_usd(path)

    def close(self):
        self.worker.stop()
