"""
Kernel worker: runs feature scripts in a separate process.

The SMLib kernel is native code; a pathological fillet or boolean can take
minutes or (rarely) crash the interpreter. Running it in a child process gives
us hard timeouts and crash isolation, and because the *script* is the model
(not kernel handles), a crashed worker is restarted and the last good script
simply replayed. Nothing but text crosses the process boundary, plus numpy
arrays for the mesh and PNG bytes for renders.
"""
from __future__ import annotations

import io
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from typing import Any

# --------------------------------------------------------------------------- #
# Child-side
# --------------------------------------------------------------------------- #

_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list",
        "map", "filter", "max", "min", "print", "range", "reversed", "round", "set", "sorted",
        "str", "sum", "tuple", "zip", "isinstance", "ValueError", "RuntimeError", "Exception",
        "True", "False", "None",
    ) if (k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k))
}


class _State:
    part = None
    script = ""
    census = None
    mesh = None
    imported = None
    imported_bodies = None
    assembly = None
    assembly_fits = None
    part_mesh = None
    body_meshes = None


def _child_main(conn):
    # Quieten USD's plugin warnings on stderr (the wheel bundles the runtime).
    os.environ.setdefault("TF_DEBUG", "")
    import usd_brep as sm  # noqa: F401  (import here so the parent never loads the kernel)
    from . import assembly as asm, census as census_mod, dsl, mesh as mesh_mod, render

    st = _State()

    def load_usd(path: str, heal: bool = True):
        """Import every BrepArray prim from a USD file; the first body becomes
        `imported` in scripts (all of them are in `imported_bodies`)."""
        bodies = list(sm.usd.import_breps(path, heal=heal))
        if not bodies:
            raise ValueError("no BrepArray prims found in the file")
        st.imported_bodies = bodies
        st.imported = bodies[0]
        return {"ok": True, "bodies": len(bodies),
                "metrics": [census_mod.metrics(b) for b in bodies]}

    def build(code: str, quality: str = "medium"):
        ns = dict(dsl.SCRIPT_NAMESPACE)
        ns["__builtins__"] = _SAFE_BUILTINS
        ns["imported"] = st.imported
        ns["imported_bodies"] = st.imported_bodies
        out = io.StringIO()
        t0 = time.time()
        with redirect_stdout(out):
            exec(compile(code, "<feature-script>", "exec"), ns)
        part = ns.get("part")
        if part is None:
            raise ValueError("the script must assign the final body to a variable named `part`")
        if not hasattr(part, "faces"):
            raise ValueError(f"`part` must be a Brep body, got {type(part).__name__}")
        assembly = ns.get("assembly")
        if assembly is not None:
            asm.validate(assembly, part)
        t_build = time.time() - t0
        st.part = part
        st.script = code
        st.census = census_mod.census(part)
        st.part_mesh = mesh_mod.tessellate(part, quality=quality)
        st.part_mesh["edge_faces"] = {e["id"]: set(int(f[1:]) for f in e["faces"]) for e in st.census["edges"]}
        st.mesh = st.part_mesh
        st.assembly, st.assembly_fits, st.body_meshes = None, None, None
        met = census_mod.metrics(part)
        viewer = mesh_mod.to_viewer_json(st.part_mesh)
        census_text = census_mod.summarize(st.census)
        if assembly is not None:
            others = [(n, b) for n, b in assembly.items() if b is not part]
            other_meshes = [mesh_mod.tessellate(b, quality=quality) for _, b in others]
            st.body_meshes = {n: (st.part_mesh if b is part else other_meshes[[o for o, _ in others].index(n)])
                              for n, b in assembly.items()}
            st.mesh = asm.merge_meshes(st.part_mesh, other_meshes)
            st.mesh["edge_faces"] = st.part_mesh["edge_faces"]
            st.assembly = dict(assembly)
            st.assembly_fits = asm.pairwise(st.assembly)
            met["assembly"] = [{"name": n, **census_mod.metrics(b)} for n, b in assembly.items()]
            met["fits"] = st.assembly_fits
            viewer["others"] = [dict(mesh_mod.to_viewer_json(m), name=n) for (n, _), m in zip(others, other_meshes)]
            census_text = asm.summarize(st.assembly, part, st.assembly_fits) + "\n" + census_text
        return {
            "ok": True,
            "stdout": out.getvalue()[-4000:],
            "build_seconds": round(t_build, 3),
            "metrics": met,
            "census": st.census,
            "census_text": census_text,
            "viewer": viewer,
        }

    def render_sheet(label_faces=True, highlight=None, caption="", views=None, size=420):
        if st.mesh is None:
            raise ValueError("nothing built yet")
        labels = None
        if label_faces:
            faces = st.census["faces"]
            # label at most ~40 faces: the largest ones
            top = sorted(faces, key=lambda f: -f["area"])[:40]
            labels = {int(f["id"][1:]): f["id"] for f in top}
        hl = set(int(h[1:]) for h in (highlight or []) if isinstance(h, str) and h.startswith("F"))
        kwargs = {}
        if views:
            kwargs["views"] = tuple(views)
        img = render.contact_sheet(st.mesh, labels=labels, highlight=hl or None, caption=caption, size=size, **kwargs)
        return {"ok": True, "png": render.png_bytes(img)}

    def section(plane_point, plane_normal, size=520):
        if st.part is None:
            raise ValueError("nothing built yet")
        from PIL import Image, ImageDraw
        import numpy as np
        bodies = list(st.assembly.values()) if st.assembly else [st.part]
        segs = []
        for body in bodies:
            poly = sm.tessellate(body, **{**sm.tessellation_defaults(), "chord_height_tolerance": 0.02,
                                          "surface_angle_tolerance_deg": 8.0})
            segs += list(poly.section_plane(tuple(map(float, plane_point)), tuple(map(float, plane_normal)), 1e-4))
        if not segs:
            raise ValueError("the plane does not intersect the part")
        n = np.asarray(plane_normal, dtype=float)
        n /= np.linalg.norm(n)
        # 2D basis in the plane
        a = np.cross(n, (0, 0, 1.0))
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(n, (1.0, 0, 0))
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        if b[2] < -1e-9 or (abs(b[2]) < 1e-9 and b.sum() < 0):
            b = -b  # keep "up" pointing up in the drawing
        if a[np.argmax(np.abs(a))] < 0:
            a = -a
        P = np.asarray([[s[0], s[1]] for s in segs], dtype=float)  # (S,2,3)
        X = P @ a
        Y = P @ b
        ext = max(X.max() - X.min(), Y.max() - Y.min(), 1e-6)
        sc = size * 0.8 / ext
        cx, cy = (X.max() + X.min()) / 2, (Y.max() + Y.min()) / 2
        img = Image.new("RGB", (size, size), (255, 255, 255))
        d = ImageDraw.Draw(img)
        # light grid every 5 mm
        step = 5.0
        gx = math.floor((cx - ext / 2) / step) * step
        while gx < cx + ext / 2 + step:
            px = size / 2 + (gx - cx) * sc
            d.line([(px, 0), (px, size)], fill=(235, 235, 240))
            gx += step
        gy = math.floor((cy - ext / 2) / step) * step
        while gy < cy + ext / 2 + step:
            py = size / 2 - (gy - cy) * sc
            d.line([(0, py), (size, py)], fill=(235, 235, 240))
            gy += step
        for (x0, x1), (y0, y1) in zip(X, Y):
            d.line([(size / 2 + (x0 - cx) * sc, size / 2 - (y0 - cy) * sc),
                    (size / 2 + (x1 - cx) * sc, size / 2 - (y1 - cy) * sc)], fill=(20, 20, 30), width=2)
        d.text((8, 6), f"section  point={tuple(round(float(v), 2) for v in plane_point)}  normal={tuple(round(float(v), 2) for v in n)}  grid=5mm",
               fill=(40, 40, 40), font=render._font(13))
        d.text((8, size - 20), f"axes: horizontal={tuple(round(float(v), 2) + 0.0 for v in a)} vertical={tuple(round(float(v), 2) + 0.0 for v in b)}",
               fill=(90, 90, 90), font=render._font(11))
        return {"ok": True, "png": render.png_bytes(img), "segments": len(segs)}

    def v3(v):
        if hasattr(v, "x"):
            v = (v.x, v.y, v.z)
        return (round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4))

    def measure(kind: str, **kw):
        if st.part is None:
            raise ValueError("nothing built yet")
        part = st.part
        faces = part.faces()
        edges = part.edges()

        def face(fid):
            return faces[int(fid[1:])]

        if kind == "probe":
            p = tuple(map(float, kw["point"]))
            cp, dist = sm.brep_closest_point(part, p)
            probe = sm.create_sphere(p, 1e-4)
            rel = part.relationship_to(probe)["relationship"]
            return {"point": p, "inside_material": rel == "a_contains_b", "distance_to_surface": round(dist, 4),
                    "closest_surface_point": v3(cp)}
        if kind == "distance":
            a, b = kw["a"], kw["b"]
            if isinstance(a, str) and isinstance(b, str):
                fa, fb = face(a), face(b)
                # min over sampled points of fa to fb (centroid + corners)
                best = None
                for pt in [fa.centroid()] + [v.point() for v in fa.vertices()]:
                    cp, d = fb.closest_point(pt)
                    if best is None or d < best[1]:
                        best = (cp, d)
                return {"from": a, "to": b, "distance": round(best[1], 4)}
            if isinstance(a, str):
                cp, d = face(a).closest_point(tuple(map(float, b)))
                return {"from": a, "to": b, "distance": round(d, 4), "closest_point": v3(cp)}
            if isinstance(b, str):
                cp, d = face(b).closest_point(tuple(map(float, a)))
                return {"from": a, "to": b, "distance": round(d, 4), "closest_point": v3(cp)}
            return {"distance": round(math.dist(a, b), 4)}
        if kind == "thickness":
            p = tuple(map(float, kw["point"]))
            dvec = tuple(map(float, kw["direction"]))
            def spans_of(mesh):
                ts = _ray_hits(mesh, p, dvec)
                return ([round(t, 4) for t in ts], [round(ts[i + 1] - ts[i], 4) for i in range(0, len(ts) - 1, 2)])
            if st.body_meshes:
                per = {}
                for name, m in st.body_meshes.items():
                    hits, spans = spans_of(m)
                    per[name] = {"surface_hits_at_distance": hits, "material_spans_mm": spans}
                return {"ray_from": p, "ray_direction": dvec, "per_body": per}
            hits, spans = spans_of(st.part_mesh)
            return {"ray_from": p, "ray_direction": dvec, "surface_hits_at_distance": hits, "material_spans_mm": spans}
        if kind == "fit":
            bodies = dict(st.assembly or {})
            if not bodies:
                raise ValueError("no `assembly` in the current script; declare assembly = {'base': base, 'lid': lid}")
            a, b = kw.get("a"), kw.get("b")
            if a not in bodies or b not in bodies:
                raise ValueError(f"fit needs two assembly names, got a={a!r} b={b!r}; available: {sorted(bodies)}")
            return {"a": a, "b": b, **asm.fit(bodies[a], bodies[b])}
        if kind == "face":
            f = face(kw["id"])
            info = next(x for x in st.census["faces"] if x["id"] == kw["id"])
            return dict(info, vertices=[tuple(round(float(c), 3) for c in v.point()) for v in f.vertices()])
        if kind == "edge":
            info = next(x for x in st.census["edges"] if x["id"] == kw["id"])
            return info
        raise ValueError(f"unknown measurement {kind!r}")

    def export_usd(path: str, with_mesh: bool = True):
        if st.part is None:
            raise ValueError("nothing built yet")
        bodies = list(st.assembly.values()) if st.assembly else [st.part]
        sm.usd.export_breps(bodies, path)
        if with_mesh:
            polys = [sm.tessellate(b, **{**sm.tessellation_defaults(), "chord_height_tolerance": 0.01,
                                        "surface_angle_tolerance_deg": 8.0}) for b in bodies]
            sm.usd.append_meshes(polys, path)
        return {"ok": True, "path": path, "bytes": os.path.getsize(path)}

    handlers = {"build": build, "render": render_sheet, "section": section, "measure": measure,
                "export_usd": export_usd, "load_usd": load_usd, "ping": lambda: {"ok": True}}

    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            return
        if msg is None:
            return
        cmd, kwargs = msg
        try:
            res = handlers[cmd](**kwargs)
            conn.send(("ok", res))
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc(limit=6)
            conn.send(("err", {"error": f"{type(e).__name__}: {e}", "traceback": tb[-3000:]}))


def _ray_hits(mesh, origin, direction):
    """Moller-Trumbore ray/triangle test over the tessellation; returns sorted
    hit distances (merged within 1e-6)."""
    import numpy as np
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    P = mesh["points"].astype(np.float64)
    T = mesh["tris"]
    v0, v1, v2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    h = np.cross(np.broadcast_to(d, e2.shape), e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-12
    f = np.zeros_like(a)
    f[ok] = 1.0 / a[ok]
    s_ = o - v0
    u = f * np.einsum("ij,ij->i", s_, h)
    q = np.cross(s_, e1)
    v = f * (q @ d)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
    ts = np.sort(t[hit])
    out = []
    for x in ts:
        if not out or x - out[-1] > 1e-6:
            out.append(float(x))
    return out


# --------------------------------------------------------------------------- #
# Parent-side
# --------------------------------------------------------------------------- #

class KernelTimeout(RuntimeError):
    pass


class KernelError(RuntimeError):
    def __init__(self, error: str, traceback_text: str = ""):
        super().__init__(error)
        self.error = error
        self.traceback_text = traceback_text


class KernelWorker:
    """Owns one child process. Thread-safe enough for a single-session server;
    the agent loop calls it sequentially."""

    def __init__(self, default_timeout: float = 90.0):
        self.default_timeout = default_timeout
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None
        self.last_script: str | None = None
        self.last_import: str | None = None
        self.start()

    def start(self):
        self.stop()
        parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_child_main, args=(child,), daemon=True)
        self._proc.start()
        self._conn = parent

    def stop(self):
        if self._proc is not None and self._proc.is_alive():
            try:
                self._conn.send(None)
                self._proc.join(timeout=1.0)
            except Exception:
                pass
            if self._proc.is_alive():
                self._proc.kill()
        self._proc = None
        self._conn = None

    def _call(self, cmd: str, timeout: float | None = None, **kwargs) -> Any:
        timeout = timeout or self.default_timeout
        if self._proc is None or not self._proc.is_alive():
            self.start()
            if self.last_import and cmd != "load_usd":
                self._call("load_usd", path=self.last_import)
            if self.last_script and cmd != "build":
                self._call("build", code=self.last_script)
        self._conn.send((cmd, kwargs))
        if not self._conn.poll(timeout):
            self._proc.kill()
            self._proc = None
            raise KernelTimeout(f"kernel operation '{cmd}' exceeded {timeout:.0f}s and was killed; "
                                "simplify the operation (smaller fillet radius, fewer edges, avoid tangent faces)")
        try:
            status, payload = self._conn.recv()
        except EOFError:
            self._proc = None
            raise KernelError("the kernel process crashed during this operation")
        if status == "err":
            raise KernelError(payload["error"], payload.get("traceback", ""))
        return payload

    # public API ------------------------------------------------------------
    def build(self, code: str, quality: str = "medium", timeout: float | None = None) -> dict:
        res = self._call("build", timeout=timeout, code=code, quality=quality)
        self.last_script = code
        return res

    def render(self, **kw) -> bytes:
        return self._call("render", **kw)["png"]

    def section(self, plane_point, plane_normal) -> dict:
        return self._call("section", plane_point=plane_point, plane_normal=plane_normal)

    def measure(self, kind: str, **kw) -> dict:
        return self._call("measure", kind=kind, **kw)

    def export_usd(self, path: str, with_mesh: bool = True) -> dict:
        return self._call("export_usd", path=path, with_mesh=with_mesh)

    def load_usd(self, path: str, heal: bool = True) -> dict:
        res = self._call("load_usd", path=path, heal=heal)
        self.last_import = path
        return res
