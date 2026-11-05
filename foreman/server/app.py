"""FastAPI server: one kernel worker + agent session per browser session,
events streamed as Server-Sent Events."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from ..agent.loop import Session  # noqa: E402
from ..kernel.worker import KernelError, KernelTimeout  # noqa: E402

app = FastAPI(title="Foreman")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
_sessions: dict[str, Session] = {}
_locks: dict[str, threading.Lock] = {}


class Message(BaseModel):
    text: str


class Script(BaseModel):
    script: str
    note: str = "manual edit"


def _get(sid: str) -> Session:
    s = _sessions.get(sid)
    if s is None:
        raise HTTPException(404, "unknown session")
    return s


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.get("/api/health")
def health():
    from ..agent.providers import make_provider
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("FOREMAN_FAKE_MODEL"))
    try:
        p = make_provider()
        model = f"{p.name}/{p.model}"
    except Exception:  # noqa: BLE001
        model = "?"
    return {"ok": True, "has_api_key": has_key, "model": model}


def _new_session() -> Session:
    if os.environ.get("FOREMAN_FAKE_MODEL"):
        # Offline demo/test mode: replay the scripted session from tests/fake_model.py
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
        from fake_model import SCRIPTED_BRACKET, FakeClient  # type: ignore
        return Session(client=FakeClient(SCRIPTED_BRACKET), model="scripted-fake-model")
    return Session()


@app.post("/api/session")
def create_session():
    sid = uuid.uuid4().hex[:12]
    _sessions[sid] = _new_session()
    _locks[sid] = threading.Lock()
    return {"session": sid, "model": _sessions[sid].model}


@app.get("/api/session/{sid}")
def session_state(sid: str):
    s = _get(sid)
    return {"versions": [{k: v for k, v in ver.items() if k != "script"} | {"script": ver["script"]} for ver in s.versions],
            "finished": s.finished, "model": s.model}


@app.post("/api/session/{sid}/message")
def message(sid: str, msg: Message):
    s = _get(sid)
    lock = _locks[sid]
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "the agent is still working on the previous message")

    def gen():
        try:
            for ev in s.run(msg.text):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            lock.release()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/session/{sid}/build")
def manual_build(sid: str, body: Script):
    """Rebuild from a hand-edited script (no model involved). The edit is also
    recorded as a version and shown to the model on its next turn."""
    s = _get(sid)
    lock = _locks[sid]
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "the agent is still working")
    try:
        version = len(s.versions) + 1
        try:
            res = s.worker.build(body.script, quality=s.quality)
        except (KernelError, KernelTimeout) as e:
            s.versions.append({"version": version, "script": body.script, "note": body.note, "ok": False, "error": str(e)})
            return {"ok": False, "version": version, "error": str(e)}
        s.versions.append({"version": version, "script": body.script, "note": body.note, "ok": True, "metrics": res["metrics"]})
        s.messages.append({"role": "user", "content": [{"type": "text", "text":
            f"[The user edited the feature script by hand and rebuilt it as v{version} ({body.note}). "
            f"Current script:\n```python\n{body.script}\n```\nmetrics: {json.dumps(res['metrics'])}]"}]})
        s.messages.append({"role": "assistant", "content": [{"type": "text", "text": "Noted - I will continue from the hand-edited v%d." % version}]})
        return {"ok": True, "version": version, "metrics": res["metrics"], "viewer": res["viewer"], "census": res["census"]}
    finally:
        lock.release()


@app.post("/api/session/{sid}/import")
async def import_usd(sid: str, file: UploadFile = File(...)):
    """Load an existing USD BRep part into the session (review mode)."""
    s = _get(sid)
    lock = _locks[sid]
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "the agent is still working")
    try:
        suffix = Path(file.filename or "part.usda").suffix.lower() or ".usda"
        if suffix not in (".usda", ".usdc", ".usd"):
            raise HTTPException(400, "please upload a .usda / .usdc / .usd file")
        path = os.path.join(tempfile.gettempdir(), f"foreman_import_{sid}{suffix}")
        with open(path, "wb") as fh:
            fh.write(await file.read())
        try:
            events = s.load_usd(path, display_name=file.filename)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"could not import: {e}")
        return {"ok": True, "events": events}
    finally:
        lock.release()


@app.get("/api/session/{sid}/export.usda")
def export(sid: str):
    s = _get(sid)
    if s.current_script is None:
        raise HTTPException(400, "nothing built yet")
    path = os.path.join(tempfile.gettempdir(), f"foreman_{sid}.usda")
    s.export_usd(path)
    return FileResponse(path, media_type="text/plain", filename="foreman_part.usda")


@app.get("/api/session/{sid}/render.png")
def render(sid: str):
    s = _get(sid)
    png = s.worker.render(caption="")
    return StreamingResponse(iter([png]), media_type="image/png")


def main():
    import uvicorn
    uvicorn.run("foreman.server.app:app", host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
