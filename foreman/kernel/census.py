"""
Topology census: a compact, LLM-readable description of a BRep's exact
geometry, computed with kernel queries (not from the tessellation).

Every face and edge gets a stable-within-this-build id (`F3`, `E12`) that is
the index in `brep.faces()` / `brep.edges()`. The same ids are used by the
DSL selectors, by the renderer's labels and by the viewer's highlighting, so
the agent can reason about "F3" across text, image and UI.
"""
from __future__ import annotations

import math
from functools import lru_cache

import usd_brep as sm

_EPS = 1e-6


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _len(v):
    return math.sqrt(_dot(v, v))


def _unit(v):
    l = _len(v)
    return (v[0] / l, v[1] / l, v[2] / l) if l > 1e-12 else (0.0, 0.0, 0.0)


def _r(v, n=3):
    if isinstance(v, (tuple, list)):
        return tuple(round(float(c), n) for c in v)
    return round(float(v), n)


def _classify_surface(face) -> tuple[str, dict]:
    """Classify a face's surface by sampling normals on a UV grid.

    planar      : all normals equal
    cylindrical : all normals perpendicular to one axis, constant distance to it
    conical     : normals perpendicular-ish to a common axis but varying angle
    spherical   : all normals point away from one centre
    toroidal/freeform otherwise
    """
    surf = face.surface()
    extra: dict = {}
    try:
        if surf.is_planar():
            return "planar", extra
    except Exception:
        pass
    (u0, v0), (u1, v1) = face.uv_domain()
    n = 4
    samples = []
    for i in range(n):
        for j in range(n):
            u = u0 + (u1 - u0) * (i + 0.5) / n
            v = v0 + (v1 - v0) * (j + 0.5) / n
            try:
                p = surf.evaluate(u, v)
                nn = surf.normal(u, v)
            except Exception:
                continue
            samples.append((p, _unit(nn)))
    if len(samples) < 4:
        return "freeform", extra
    normals = [nn for _, nn in samples]
    # spherical: lines p - t*n all pass through a common centre
    # Solve least-squares centre c minimising sum |(p - c) x n|^2  -> cheap check:
    # for a sphere of radius R, p - R*n == c for every sample.
    # Estimate R from two samples: |p_i - p_j| / |n_i - n_j|
    pts = [p for p, _ in samples]
    # cylinder/cone test: find axis as the direction most perpendicular to all normals
    axis = _principal_perp(normals)
    if axis is not None:
        perp = max(abs(_dot(axis, nn)) for nn in normals)
        # radial distances from the axis line through the mean point
        mean = tuple(sum(p[k] for p in pts) / len(pts) for k in range(3))
        rads = []
        for p in pts:
            d = _sub(p, mean)
            d_par = _dot(d, axis)
            r = _len(_sub(d, (axis[0] * d_par, axis[1] * d_par, axis[2] * d_par)))
            rads.append(r)
        if perp < 0.02:
            # normals perpendicular to axis: cylinder. Radius from the point
            # p - R*n lying on the axis for every sample: R = chord / normal-chord.
            r_est = _estimate_radius(samples)
            extra["axis"] = _r(axis)
            if r_est is not None:
                extra["radius"] = _r(r_est)
                c = [_sub(p, (r_est * nn[0], r_est * nn[1], r_est * nn[2])) for p, nn in samples]
                mean = tuple(sum(q[k] for q in c) / len(c) for k in range(3))
                extra["axis_point"] = _r(mean)
            return "cylindrical", extra
        if perp < 0.7 and _is_conical(samples, axis):
            extra["axis"] = _r(axis)
            return "conical", extra
    sph = _sphere_fit(samples)
    if sph is not None:
        c, rad = sph
        extra["center"] = _r(c)
        extra["radius"] = _r(rad)
        return "spherical", extra
    if surf.is_periodic_u() or surf.is_periodic_v():
        return "toroidal", extra
    return "freeform", extra


def _principal_perp(normals):
    """Direction minimising sum (n . a)^2 : smallest eigenvector of the normal
    scatter matrix. Tiny 3x3 power-iteration on (I*trace - M) to avoid numpy."""
    m = [[0.0] * 3 for _ in range(3)]
    for nn in normals:
        for i in range(3):
            for j in range(3):
                m[i][j] += nn[i] * nn[j]
    tr = m[0][0] + m[1][1] + m[2][2]
    a = [[(tr if i == j else 0.0) - m[i][j] for j in range(3)] for i in range(3)]
    v = (0.3, 0.5, 0.8)
    for _ in range(60):
        w = tuple(sum(a[i][j] * v[j] for j in range(3)) for i in range(3))
        l = _len(w)
        if l < 1e-12:
            return None
        v = (w[0] / l, w[1] / l, w[2] / l)
    return v


def _estimate_radius(samples):
    """Radius R such that p - R*n is (nearly) the same axis point for all
    samples: for a pair, (p-q) . (n1-n2) = R |n1-n2|^2 when both lie on a
    common circle of the cylinder."""
    best = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            p, n1 = samples[i]
            q, n2 = samples[j]
            dn = _sub(n1, n2)
            l2 = _dot(dn, dn)
            if l2 > 0.0025:
                best.append(_dot(_sub(p, q), dn) / l2)
    if not best:
        return None
    best.sort()
    r = best[len(best) // 2]
    return abs(r)


def _is_conical(samples, axis):
    angs = [math.degrees(math.acos(max(-1.0, min(1.0, _dot(nn, axis))))) for _, nn in samples]
    return (max(angs) - min(angs)) < 2.0 and abs(angs[0] - 90) > 2.0


def _sphere_fit(samples):
    # c = p - R n for all samples; R from pairs
    rs = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            p, n1 = samples[i]
            q, n2 = samples[j]
            dn = _sub(n1, n2)
            l2 = _dot(dn, dn)
            if l2 < 1e-6:
                continue
            rs.append(_dot(_sub(p, q), dn) / l2)
    if len(rs) < 3:
        return None
    rs.sort()
    R = rs[len(rs) // 2]
    if R <= 0:
        return None
    cs = [_sub(p, (R * nn[0], R * nn[1], R * nn[2])) for p, nn in samples]
    mean = tuple(sum(c[k] for c in cs) / len(cs) for k in range(3))
    spread = max(_len(_sub(c, mean)) for c in cs)
    if spread < 1e-3 * max(1.0, R):
        return mean, R
    return None


def _edge_type(edge) -> tuple[str, dict]:
    """Classify an edge curve. Native kernel Line/Circle objects are trusted;
    anything else (e.g. NURBS edges read back from USD, where every edge is a
    B-spline) is classified geometrically by sampling: collinear -> line,
    constant distance from a fitted centre -> circle (with radius and centre)."""
    c = edge.curve()
    name = type(c).__name__
    extra = {}
    if name == "Line":
        return "line", extra
    if name == "Circle":
        try:
            extra["radius"] = _r(c.radius())
            extra["center"] = _r(c.center())
        except Exception:
            pass
        return "circle", extra
    try:
        t0, t1 = edge.parameter_range()
        pts = [edge.point_at(t0 + (t1 - t0) * k / 10) for k in range(11)]
    except Exception:
        return "curve", extra
    p0, p1 = pts[0], pts[-1]
    chord = _sub(p1, p0)
    L = _len(chord)
    size = max(L, max(_len(_sub(p, p0)) for p in pts), 1e-9)
    if L > 1e-9:
        d = _unit(chord)
        dev = max(_len(_sub(_sub(p, p0), tuple(d[k] * _dot(_sub(p, p0), d) for k in range(3)))) for p in pts)
        if dev < 1e-5 * size + 1e-7:
            return "line", extra
    # circle fit through three samples
    a, b, cc = pts[0], pts[5], pts[-1] if L > 1e-9 else pts[7]
    try:
        centre, radius = _circle_through(a, b, cc)
    except Exception:
        return "curve", extra
    if radius is not None and all(abs(_len(_sub(p, centre)) - radius) < 1e-4 * max(radius, 1.0) for p in pts):
        extra["radius"] = _r(radius)
        extra["center"] = _r(centre)
        return "circle", extra
    return "curve", extra


def _circle_through(a, b, c):
    ab = _sub(b, a)
    ac = _sub(c, a)
    n = _cross(ab, ac)
    n2 = _dot(n, n)
    if n2 < 1e-18:
        return None, None
    # circumcentre of triangle a,b,c:  a + [ (|ac|^2 ab - |ab|^2 ac) x n ] / (2 |n|^2)... with ab x ac = n:
    # centre = a + ( |ac|^2 (n x ab) + |ab|^2 (ac x n) ) / (2 |n|^2)
    t1 = tuple(_dot(ac, ac) * x for x in _cross(n, ab))
    t2 = tuple(_dot(ab, ab) * x for x in _cross(ac, n))
    centre = tuple(a[k] + (t1[k] + t2[k]) / (2 * n2) for k in range(3))
    return centre, _len(_sub(a, centre))


def _edge_convexity(brep, edge, faces_info, face_index):
    """True for a convex (outside) edge, False for concave (inside corner),
    None if unknown. Uses the kernel: offset the edge midpoint along the mean
    of the adjacent outward normals and ask whether the point is inside."""
    try:
        fs = edge.faces()
        if len(fs) != 2:
            return None
        t0, t1 = edge.parameter_range()
        mid = edge.point_at(0.5 * (t0 + t1))
        n1 = _outward_normal_near(fs[0], mid)
        n2 = _outward_normal_near(fs[1], mid)
        d = _sub(n1, n2)
        if _len(d) < 0.05:
            return None  # tangent-continuous (smooth) edge: neither convex nor concave
        d = _unit(d)
        bb = brep.bounding_box()
        diag = _len(_sub(bb[1], bb[0]))
        eps = max(2e-3 * diag, 1e-3)
        # For a convex edge the point mid + eps*(n1 - n2) lies in the air; for a
        # concave (inside-corner) edge it lies inside the material.
        probe = (mid[0] + d[0] * eps, mid[1] + d[1] * eps, mid[2] + d[2] * eps)
        return not _point_inside(brep, probe, eps * 0.1)
    except Exception:
        return None


def _outward_normal_near(face, point):
    """Outward normal of `face` at the surface point closest to `point`."""
    surf = face.surface()
    try:
        _, uvs, _ = sm.surface_closest_point(surf, tuple(point))
        if uvs:
            uv = uvs[0]
            u, v = (uv[0], uv[1]) if hasattr(uv, "__getitem__") else (uv.x, uv.y)
            return _unit(face.outward_normal_at_uv(u, v))
    except Exception:
        pass
    (u0, v0), (u1, v1) = face.uv_domain()
    return _unit(face.outward_normal_at_uv(0.5 * (u0 + u1), 0.5 * (v0 + v1)))


def _point_inside(brep, p, r):
    probe = sm.create_sphere(p, r)
    rel = brep.relationship_to(probe)["relationship"]
    return rel == "a_contains_b"


def census(brep, *, convexity: bool = True) -> dict:
    """Full census of a BRep. Cached per-handle by the worker, not here."""
    faces = brep.faces()
    edges = brep.edges()
    face_index = {}
    faces_info = []
    for i, f in enumerate(faces):
        face_index[id(f)] = i
    # map kernel face handles -> index via equality (pybind identity)
    def fidx(fh):
        for i, f in enumerate(faces):
            if f == fh:
                return i
        return -1

    for i, f in enumerate(faces):
        kind, extra = _classify_surface(f)
        try:
            centre = f.centroid()
        except Exception:
            centre = f.interior_point()
        normal = None
        if kind == "planar":
            try:
                normal = _r(_outward_normal_near(f, centre))
            except Exception:
                normal = None
        mn, mx = f.bounding_box()
        info = {
            "id": f"F{i}",
            "type": kind,
            "area": _r(f.area(), 2),
            "center": _r(centre),
            "normal": normal,
            "bbox_min": _r(mn),
            "bbox_max": _r(mx),
            "edges": [],
        }
        info.update(extra)
        faces_info.append(info)

    edges_info = []
    for j, e in enumerate(edges):
        kind, extra = _edge_type(e)
        t0, t1 = e.parameter_range()
        p0 = e.start_vertex().point()
        p1 = e.end_vertex().point()
        mid = e.point_at(0.5 * (t0 + t1))
        adj = [fidx(fh) for fh in e.faces()]
        adj_ids = [f"F{k}" for k in adj if k >= 0]
        info = {
            "id": f"E{j}",
            "type": kind,
            "length": _r(e.length(), 2),
            "start": _r(p0),
            "end": _r(p1),
            "mid": _r(mid),
            "faces": adj_ids,
            "convex": None,
            "seam": len(set(adj_ids)) < 2,
        }
        if kind == "line":
            info["direction"] = _r(_unit(_sub(p1, p0)))
        info.update(extra)
        if convexity and not info["seam"]:
            info["convex"] = _edge_convexity(brep, e, faces_info, face_index)
        for k in adj:
            if k >= 0:
                faces_info[k]["edges"].append(f"E{j}")
        edges_info.append(info)

    return {"faces": faces_info, "edges": edges_info}


def metrics(brep) -> dict:
    info = brep.info()
    out = {
        "manifold_solid": bool(info.get("manifold")),
        "faces": info.get("faces"),
        "edges": info.get("edges"),
        "vertices": info.get("vertices"),
    }
    try:
        mn, mx = brep.bounding_box(tight=True)
        out["bbox_min"] = _r(mn)
        out["bbox_max"] = _r(mx)
        out["size"] = _r(tuple(b - a for a, b in zip(mn, mx)))
    except Exception:
        pass
    try:
        if out["manifold_solid"]:
            mp = brep.mass_properties()
            out["volume_mm3"] = _r(mp["volume"], 1)
            out["surface_area_mm2"] = _r(mp["area"], 1)
            out["centroid"] = _r(mp["centroid"])
    except Exception:
        pass
    return out


def summarize(c: dict, limit: int = 60) -> str:
    """Compact textual census for the LLM context window."""
    lines = []
    fs = c["faces"]
    es = c["edges"]
    lines.append(f"FACES ({len(fs)}):")
    for f in fs[:limit]:
        extra = ""
        if f["type"] in ("cylindrical", "spherical") and f.get("radius") is not None:
            extra = f" r={f['radius']}"
        if f["type"] in ("cylindrical", "conical") and f.get("axis") is not None:
            extra += f" axis={f['axis']}"
        nrm = f" normal={f['normal']}" if f["type"] == "planar" else ""
        lines.append(f"  {f['id']}: {f['type']}{extra} area={f['area']} center={f['center']}{nrm} edges={','.join(f['edges'])}")
    if len(fs) > limit:
        lines.append(f"  ... {len(fs) - limit} more faces")
    lines.append(f"EDGES ({len(es)}):")
    for e in es[:limit]:
        conv = "seam" if e.get("seam") else {True: "convex", False: "concave", None: "smooth"}[e["convex"]]
        geo = ""
        if e["type"] == "line":
            geo = f" dir={e['direction']}"
        elif e["type"] == "circle":
            geo = f" r={e.get('radius')} center={e.get('center')}"
        lines.append(f"  {e['id']}: {e['type']}{geo} len={e['length']} mid={e['mid']} {conv} faces={','.join(e['faces'])}")
    if len(es) > limit:
        lines.append(f"  ... {len(es) - limit} more edges")
    return "\n".join(lines)
