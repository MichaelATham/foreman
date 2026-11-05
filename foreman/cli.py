"""Headless runner: `python -m foreman.cli "<brief>" --out out/`.

Writes the final USD (BrepArray + mesh), every render the agent looked at,
the feature script of each version and a markdown report of the session.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def main(argv=None):
    load_dotenv()
    ap = argparse.ArgumentParser(description="Foreman - self-verifying AI CAD agent (headless)")
    ap.add_argument("brief", help="design brief, or @file.txt")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--model", default=None)
    ap.add_argument("--follow-up", action="append", default=[], help="additional edit requests after the first result")
    ap.add_argument("--audit", metavar="PART.usda", help="load an existing USD BRep part first (review mode); the brief is then the audit/edit request")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    from .agent.loop import Session

    brief = args.brief
    if brief.startswith("@"):
        brief = Path(brief[1:]).read_text()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = Session(model=args.model)
    log = []
    n_img = 0
    t0 = time.time()

    def say(*a):
        if not args.quiet:
            print(*a, flush=True)

    try:
        if args.audit:
            say(f"=== loading {args.audit}")
            for ev in s.load_usd(args.audit):
                if ev["type"] == "build":
                    say(f"[imported] {json.dumps(ev.get('metrics'))}" if ev["ok"] else f"[import failed] {ev['error']}")
                elif ev["type"] == "image":
                    n_img += 1
                    (out / f"look_{n_img:02d}.png").write_bytes(base64.b64decode(ev["png_b64"]))
        for msg in [brief, *args.follow_up]:
            say(f"\n=== USER: {msg}\n")
            for ev in s.run(msg):
                log.append({k: v for k, v in ev.items() if k not in ("png_b64", "viewer", "census")})
                t = ev["type"]
                if t == "text":
                    say(ev["text"], end="") if not args.quiet else None
                    sys.stdout.flush()
                elif t == "tool_call":
                    say(f"\n[tool] {ev['name']} {json.dumps({k: v for k, v in ev['input'].items() if k != 'script'})}")
                    if "script" in ev["input"]:
                        say("------ script ------\n" + ev["input"]["script"] + "\n--------------------")
                elif t == "build":
                    (out / f"v{ev['version']}.py").write_text(ev["script"])
                    say(f"[build v{ev['version']}] " + ("OK " + json.dumps(ev["metrics"]) if ev["ok"] else "FAILED: " + ev["error"]))
                elif t == "image":
                    n_img += 1
                    p = out / f"look_{n_img:02d}.png"
                    p.write_bytes(base64.b64decode(ev["png_b64"]))
                    say(f"[image] {ev.get('caption','')} -> {p}")
                elif t == "measure":
                    say(f"[measure] {json.dumps(ev['result'])}")
                elif t == "finish":
                    say("\n[verification report]")
                    for r in ev["report"]:
                        say(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['requirement']}  —  {r['method']}: {r['result']}")
                elif t == "error":
                    say(f"[error] {ev['message']}")
            say("")
        if s.current_script:
            res = s.export_usd(str(out / "part.usda"))
            say(f"\nwrote {res['path']} ({res['bytes']} bytes)")
            (out / "part.py").write_text(s.current_script)
        _write_report(out, brief, s, log, time.time() - t0)
        say(f"report: {out / 'report.md'}   ({time.time() - t0:.0f}s)")
    finally:
        s.close()


def _write_report(out: Path, brief: str, s, log: list, seconds: float):
    lines = [f"# Foreman session report", "", f"**Brief:** {brief}", "", f"Model: `{s.model}` · {seconds:.0f}s · {len(s.versions)} build(s)", ""]
    for v in s.versions:
        lines.append(f"## v{v['version']} — {v['note']}  " + ("✅" if v["ok"] else "❌"))
        if v["ok"]:
            lines.append(f"`{json.dumps(v['metrics'])}`")
        else:
            lines.append(f"error: `{v['error']}`")
        lines += ["", "```python", v["script"], "```", ""]
    if s.finished:
        lines += ["## Verification report", "", s.finished["summary"], "", "| | requirement | method | result |", "|---|---|---|---|"]
        for r in s.finished["report"]:
            lines.append(f"| {'✅' if r['pass'] else '❌'} | {r['requirement']} | {r['method']} | {r['result']} |")
    (out / "report.md").write_text("\n".join(lines))
    (out / "events.json").write_text(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
