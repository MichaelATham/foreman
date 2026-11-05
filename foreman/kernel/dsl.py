"""
Feature-script DSL: a thin, safe, *replayable* layer over the SMLib kernel
(`usd_brep`).

Design goals
------------
* Every function is pure with respect to its inputs (we copy before any
  consuming kernel call), so a script can reuse a body freely. This hides the
  kernel's "booleans consume their operands" ownership contract, which is the
  #1 footgun for an LLM writing kernel code.
* Units are millimetres, Z is up, boxes/cylinders are created at an explicit
  `at=` position so the model never has to reason about hidden origins.
* Selectors (`faces()`, `edges()`) are expressed over the same *census* dicts
  the agent sees in its tool results, so "fillet the top edges" maps 1:1 to
  `edges(part, z='max', straight=True)`.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import usd_brep as sm

from . import census as _census

Vec = tuple[float, float, float]

AXES = {
    "x": (1.0, 0.0, 0.0), "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0), "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


def _vec(v) -> Vec:
    if isinstance(v, str):
        return AXES[v.lower()]
    v = tuple(float(c) for c in v)
    if len(v) != 3:
        raise ValueError(f"expected a 3-vector, got {v}")
    return v


def _norm(v: Vec) -> Vec:
    l = math.sqrt(sum(c * c for c in v))
    if l < 1e-12:
        raise ValueError("zero-length vector")
    return (v[0] / l, v[1] / l, v[2] / l)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _align_z_to(brep, axis, about: Vec):
    """Rotate a body built along +Z so that +Z maps onto `axis`."""
    a = _norm(_vec(axis))
    z = (0.0, 0.0, 1.0)
    d = max(-1.0, min(1.0, _dot(z, a)))
    if d > 1 - 1e-9:
        return brep
    if d < -1 + 1e-9:
        return sm.rotate(brep, about, (1.0, 0.0, 0.0), 180.0)
    axis_rot = _norm(_cross(z, a))
    return sm.rotate(brep, about, axis_rot, math.degrees(math.acos(d)))


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

def box(size: Sequence[float], at: Sequence[float] = (0, 0, 0), center: bool = False):
    """Axis-aligned box. `at` is the min corner, or the centre if center=True."""
    l, w, h = (float(s) for s in size)
    x, y, z = (float(c) for c in at)
    if center:
        x, y, z = x - l / 2, y - w / 2, z - h / 2
    return sm.create_box((x, y, z), l, w, h)


def cylinder(r: float, h: float, at: Sequence[float] = (0, 0, 0), axis="z"):
    """Cylinder of radius r, height h. `at` is the centre of the base disc; it
    extends `h` along `axis`."""
    at = tuple(float(c) for c in at)
    b = sm.create_cylinder((0.0, 0.0, 0.0), float(r), float(h))
    b = _align_z_to(b, axis, (0.0, 0.0, 0.0))
    return sm.translate(b, at)


def cone(r_base: float, r_top: float, h: float, at: Sequence[float] = (0, 0, 0), axis="z"):
    """Frustum (r_top=0 for a pointed cone), base centred at `at`, along `axis`."""
    at = tuple(float(c) for c in at)
    b = sm.create_cone((0.0, 0.0, 0.0), float(r_base), float(r_top), float(h))
    b = _align_z_to(b, axis, (0.0, 0.0, 0.0))
    return sm.translate(b, at)


def sphere(r: float, at: Sequence[float] = (0, 0, 0)):
    return sm.create_sphere(tuple(float(c) for c in at), float(r))


def torus(r_major: float, r_minor: float, at: Sequence[float] = (0, 0, 0), axis="z"):
    """Torus centred at `at`, ring in the plane perpendicular to `axis`."""
    at = tuple(float(c) for c in at)
    b = sm.create_torus((0.0, 0.0, 0.0), float(r_major), float(r_minor))
    b = _align_z_to(b, axis, (0.0, 0.0, 0.0))
    return sm.translate(b, at)


# --------------------------------------------------------------------------- #
# Profiles and sweeps
# --------------------------------------------------------------------------- #

def _profile_curves(profile, z: float = 0.0):
    """Turn a profile spec into closed kernel curves in the plane Z=z.

    profile may be:
      * a list of (x, y) points -> closed polygon
      * ("circle", r) or ("circle", r, (cx, cy))
      * ("rect", l, w) or ("rect", l, w, (cx, cy))
      * ("polygon", n, r) or ("polygon", n, r, (cx, cy))
    """
    if isinstance(profile, (list, tuple)) and profile and isinstance(profile[0], str):
        kind = profile[0].lower()
        if kind == "circle":
            r = float(profile[1])
            c = profile[2] if len(profile) > 2 else (0.0, 0.0)
            return [sm.create_circle((float(c[0]), float(c[1]), z), r)]
        if kind == "rect":
            l, w = float(profile[1]), float(profile[2])
            c = profile[3] if len(profile) > 3 else (0.0, 0.0)
            return sm.create_rectangle((float(c[0]), float(c[1]), z), l, w)
        if kind == "polygon":
            n, r = int(profile[1]), float(profile[2])
            c = profile[3] if len(profile) > 3 else (0.0, 0.0)
            return sm.create_regular_polygon((float(c[0]), float(c[1]), z), n, r)
        raise ValueError(f"unknown profile kind {kind!r}")
    pts = [(float(p[0]), float(p[1])) for p in profile]
    if len(pts) < 3:
        raise ValueError("a polygon profile needs at least 3 points")
    if math.dist(pts[0], pts[-1]) < 1e-9:
        pts = pts[:-1]
    curves = []
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        if math.dist(p, q) < 1e-9:
            continue
        curves.append(sm.create_line_segment((p[0], p[1], z), (q[0], q[1], z)))
    return curves


def extrude(profile, h: float, at: Sequence[float] = (0, 0, 0), axis="z"):
    """Extrude a planar profile by `h` along `axis` (a prism). Profile
    coordinates (u, v) are relative to `at`, and the profile plane is
    perpendicular to the axis:  axis z: (u,v) -> (x,y);  axis y: (u,v) -> (x,z);
    axis x: (u,v) -> (y,z). For '-x' etc. the prism extends the other way."""
    curves = _profile_curves(profile, 0.0)
    b = sm.linear_sweep(curves, (0.0, 0.0, 1.0), float(h), cap_ends=True)
    h = float(h)
    key = axis.lower() if isinstance(axis, str) else None
    if key in ("z", "+z"):
        pass
    elif key == "-z":
        b = sm.translate(b, (0.0, 0.0, -h))
    elif key in ("x", "+x", "-x"):
        b = sm.rotate(b, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 120.0)  # z->x, x->y, y->z
        if key == "-x":
            b = sm.translate(b, (-h, 0.0, 0.0))
    elif key in ("y", "+y", "-y"):
        b = sm.rotate(b, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 90.0)  # y->z, z->-y
        if key != "-y":
            b = sm.translate(b, (0.0, h, 0.0))
    else:
        b = _align_z_to(b, axis, (0.0, 0.0, 0.0))
    return sm.translate(b, tuple(float(c) for c in at))


def revolve(profile_rz: Sequence[Sequence[float]], angle: float = 360.0,
            at: Sequence[float] = (0, 0, 0), axis="z"):
    """Revolve a closed polygon given as (radius, height) pairs about `axis`.
    Radii must be >= 0 (the profile lives in the half-plane x>=0)."""
    pts = [(float(r), float(zz)) for r, zz in profile_rz]
    if any(r < -1e-9 for r, _ in pts):
        raise ValueError("revolve profile radii must be >= 0")
    if math.dist(pts[0], pts[-1]) < 1e-9:
        pts = pts[:-1]
    curves = []
    for i, (r0, z0) in enumerate(pts):
        r1, z1 = pts[(i + 1) % len(pts)]
        if math.dist((r0, z0), (r1, z1)) < 1e-9:
            continue
        curves.append(sm.create_line_segment((r0, 0.0, z0), (r1, 0.0, z1)))
    b = sm.rotational_sweep(curves, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), float(angle), cap_ends=True)
    b = _align_z_to(b, axis, (0.0, 0.0, 0.0))
    return sm.translate(b, tuple(float(c) for c in at))


def sweep_pipe(path_points: Sequence[Sequence[float]], r: float):
    """Circular tube of radius r along a smooth B-spline through >= 4 points."""
    pts = [tuple(float(c) for c in p) for p in path_points]
    if len(pts) < 4:
        raise ValueError("sweep_pipe needs at least 4 path points")
    path = sm.create_curve(pts)
    return sm.pipe_sweep(path, radius=float(r), cap_ends=True)


def loft(profiles, heights: Sequence[float], at: Sequence[float] = (0, 0, 0), axis="z", smooth: bool = False):
    """Loft a solid through 2+ planar profiles stacked along `axis` at the given
    heights (same profile specs as extrude: ('circle', r), ('rect', l, w),
    ('polygon', n, r) or a point list). smooth=True blends with a cubic skin
    through intermediate sections instead of ruled (straight) faces."""
    if len(profiles) != len(heights) or len(profiles) < 2:
        raise ValueError("loft() needs the same number of profiles and heights (at least 2)")
    counts, curves = [], []
    for prof, h in zip(profiles, heights):
        cs = _profile_curves(prof, float(h))
        counts.append(len(cs))
        curves.extend(cs)
    degree = 3 if (smooth and len(profiles) >= 4) else (2 if smooth and len(profiles) == 3 else 1)
    b = sm.create_skin_primitive(counts, curves, degree=degree, cap_ends=True)
    b = _align_z_to(b, axis, (0.0, 0.0, 0.0))
    return sm.translate(b, tuple(float(c) for c in at))


def draft_extrude(profile, h: float, angle: float, at: Sequence[float] = (0, 0, 0), axis="z",
                  round_corners: bool = False):
    """Extrude a profile by h with a draft angle (degrees): positive narrows the
    section towards the far end (moulding draft), negative widens it. Same
    profile specs and placement as extrude(); round_corners blends expanding
    corners."""
    curves = _profile_curves(profile, 0.0)
    b = sm.draft_sweep(curves, float(h), float(angle), fillet_corners=bool(round_corners), cap_ends=sm.CapEnds.BOTH)
    b = _align_z_to(b, axis, (0.0, 0.0, 0.0))
    return sm.translate(b, tuple(float(c) for c in at))


def sweep(profile, path_points: Sequence[Sequence[float]]):
    """Sweep a planar profile along a smooth spline through >= 4 path points.
    The profile (same specs as extrude, drawn around its own origin) is placed
    perpendicular to the path at its first point. Use sweep_pipe() for a plain
    round tube."""
    pts = [tuple(float(c) for c in p) for p in path_points]
    if len(pts) < 4:
        raise ValueError("sweep() needs at least 4 path points")
    path = sm.create_curve_interp_points(pts)
    t0, _ = path.parameter_range()
    p0 = path.evaluate(t0)
    tangent = _norm(_sub3(path.evaluate(t0 + 1e-4 * (path.parameter_range()[1] - t0)), p0))
    curves = _profile_curves(profile, 0.0)
    for c in curves:
        _align_z_to(c, tangent, (0.0, 0.0, 0.0))
        sm.translate(c, (p0[0], p0[1], p0[2]))
    return sm.curve_sweep(curves, path, cap_ends=True)


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def offset(part, d: float):
    """Grow (d > 0) or shrink (d < 0) a body by offsetting all its faces."""
    return sm.offset_brep(part, float(d))


# --------------------------------------------------------------------------- #
# Booleans (non-consuming)
# --------------------------------------------------------------------------- #

def union(*parts):
    parts = _flatten(parts)
    if not parts:
        raise ValueError("union() needs at least one body")
    acc = parts[0].copy()
    for p in parts[1:]:
        acc = sm.boolean_union(acc, p.copy())
    return acc


def difference(base, *tools):
    tools = _flatten(tools)
    acc = base.copy()
    for t in tools:
        acc = sm.boolean_difference(acc, t.copy())
    return acc


def intersection(a, b):
    return sm.boolean_intersection(a.copy(), b.copy())


def _flatten(items):
    out = []
    for it in items:
        if isinstance(it, (list, tuple)):
            out.extend(_flatten(it))
        else:
            out.append(it)
    return out


# --------------------------------------------------------------------------- #
# Edits
# --------------------------------------------------------------------------- #

def _blend_guard(before, after, op: str, r: float):
    """The kernel can return a 'successful' but geometrically absurd body when a
    fillet/chamfer radius exceeds what the adjacent faces can accommodate
    (e.g. r=9 on a 10 mm cube yields a manifold solid with *more* volume than
    the cube). Rounding or bevelling convex edges must remove material and
    reduce area; blending concave edges adds material. We check that the
    change is physically plausible and raise a clear error otherwise."""
    if not after.is_manifold_solid():
        raise ValueError(f"{op}(r={r}): result is not a closed solid - the radius is too large for "
                         "the adjacent faces or the edges collide; try a smaller radius or fewer edges")
    v0, v1 = float(before.volume()), float(after.volume())
    a0, a1 = float(before.mass_properties()["area"]), float(after.mass_properties()["area"])
    if a1 > a0 * 1.02 or v1 > v0 * 1.25 or v1 < v0 * 0.35:
        raise ValueError(f"{op}(r={r}): result is geometrically implausible (area {a0:.0f}->{a1:.0f} mm2, "
                         f"volume {v0:.0f}->{v1:.0f} mm3); the radius is too large for the adjacent faces. "
                         "Rule of thumb: r <= 1/3 of the smallest adjacent face dimension.")


def fillet(part, edge_sel, r: float, *, smooth: bool = False):
    """Round the given edges (an edge-id list or the result of `edges()`).
    Returns a new body; the input is untouched. `smooth=True` uses a G2 blend."""
    b = part.copy()
    es = _resolve_edges(b, edge_sel)
    if not es:
        raise ValueError("fillet(): no edges matched the selection")
    if smooth:
        sm.fillet_edges(b, es, radius=float(r), xsect_type=2, continuity=2)
    else:
        sm.fillet_edges(b, es, radius=float(r), xsect_type=1)
    _blend_guard(part, b, "fillet", r)
    return b


def chamfer(part, edge_sel, d: float):
    """Bevel the given edges with a flat cut of width d. Returns a new body."""
    b = part.copy()
    es = _resolve_edges(b, edge_sel)
    if not es:
        raise ValueError("chamfer(): no edges matched the selection")
    sm.fillet_edges(b, es, radius=float(d), xsect_type=0)
    _blend_guard(part, b, "chamfer", d)
    return b


def fillet_all(part, r: float):
    """Round every edge of the body with radius r."""
    b = part.copy()
    sm.circular_fillet(b, float(r))
    _blend_guard(part, b, "fillet_all", r)
    return b


def chamfer_all(part, d: float):
    """Bevel every edge of the body with width d."""
    b = part.copy()
    sm.chamfer_fillet(b, float(d))
    _blend_guard(part, b, "chamfer_all", d)
    return b


def shell(part, thickness: float, open_faces=None):
    """Hollow the body leaving walls of `thickness`. `open_faces` (face ids or
    a `faces()` result) are removed to create openings, e.g. the top face of a
    cup. Returns a new body."""
    fs = _resolve_faces(part, open_faces) if open_faces else []
    return sm.shell_brep(part, -abs(float(thickness)), faces_to_shell=fs)


def move(part, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
    return sm.translate(part.copy(), (float(dx), float(dy), float(dz)))


def rotate(part, axis, angle: float, about: Sequence[float] = (0, 0, 0)):
    return sm.rotate(part.copy(), tuple(float(c) for c in about), _vec(axis), float(angle))


def cut_plane(part, point, normal):
    """Keep the half of the body on the side of the plane that `normal` points to."""
    return sm.cut(part.copy(), tuple(float(c) for c in point), _vec(normal))


def copy(part):
    return part.copy()


def pattern_linear(part, n: int, step: Sequence[float]):
    """n copies of `part` offset by `step` each; returns a list of bodies."""
    out = []
    for i in range(int(n)):
        out.append(move(part, step[0] * i, step[1] * i, step[2] * i))
    return out


def pattern_polar(part, n: int, axis="z", about: Sequence[float] = (0, 0, 0)):
    """n copies rotated evenly about `axis`; returns a list of bodies."""
    return [rotate(part, axis, 360.0 * i / int(n), about) for i in range(int(n))]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def _match_axis(vec, want, tol=0.05):
    if want is None:
        return True
    w = _norm(_vec(want))
    v = vec
    l = math.sqrt(_dot(v, v))
    if l < 1e-12:
        return False
    v = (v[0] / l, v[1] / l, v[2] / l)
    if isinstance(want, str) and want.lower() in ("x", "y", "z"):
        return abs(abs(_dot(v, w)) - 1.0) < tol  # either sign
    return abs(_dot(v, w) - 1.0) < tol


def _match_extreme(items, key, spec, tol=1e-4):
    if spec is None:
        return [True] * len(items)
    vals = [key(it) for it in items]
    if isinstance(spec, str):
        if spec == "max":
            target = max(vals)
        elif spec == "min":
            target = min(vals)
        else:
            raise ValueError(f"bad extreme spec {spec!r}")
    else:
        target = float(spec)
    return [abs(v - target) < tol for v in vals]


def faces(part, *, normal=None, type: str | None = None, z=None, x=None, y=None,
          area_min: float | None = None, area_max: float | None = None,
          where: Callable | None = None):
    """Select faces by geometric predicates. Returns a list of face ids like
    ['F3', 'F7'] that can be passed to shell() / measure tools.

    normal: 'z' (either sign), '+z', '-z', '+x', ... or a vector.
    type:   'planar' | 'cylindrical' | 'conical' | 'spherical' | 'toroidal' | 'freeform'
    z/x/y:  'max' | 'min' | number   -> face centre coordinate
    area_min/area_max: area bounds in mm^2
    where:  callable(info_dict) -> bool for anything else
    """
    c = _census.census(part)
    fs = c["faces"]
    sel = []
    mz = _match_extreme(fs, lambda f: f["center"][2], z)
    mx = _match_extreme(fs, lambda f: f["center"][0], x)
    my = _match_extreme(fs, lambda f: f["center"][1], y)
    for i, f in enumerate(fs):
        if normal is not None and (f["normal"] is None or not _match_axis(f["normal"], normal)):
            continue
        if type is not None and f["type"] != type:
            continue
        if not (mz[i] and mx[i] and my[i]):
            continue
        if area_min is not None and f["area"] < area_min:
            continue
        if area_max is not None and f["area"] > area_max:
            continue
        if where is not None and not where(f):
            continue
        sel.append(f["id"])
    return sel


def edges(part, *, direction=None, straight: bool | None = None, circular: bool | None = None,
          z=None, x=None, y=None, convex: bool | None = None,
          length_min: float | None = None, length_max: float | None = None,
          radius: float | None = None, on_face=None, where: Callable | None = None,
          include_seams: bool = False):
    """Select edges by geometric predicates. Returns a list of edge ids like
    ['E1', 'E4'] for fillet()/chamfer().

    direction: 'z'|'x'|'y' (either sign) or a vector — for straight edges only.
    straight / circular: edge curve type filters.
    z/x/y:  'max' | 'min' | number -> edge midpoint coordinate.
    convex: True = outside (rounding) edges, False = inside corners.
    radius: circular edges of this radius (±1e-3).
    on_face: a face id (or list) — keep edges bounding those faces.
    where:  callable(info_dict) -> bool.
    include_seams: periodic-surface seam edges (one adjacent face) are
            topological artifacts and are excluded unless this is True.
    """
    c = _census.census(part)
    es = c["edges"]
    mz = _match_extreme(es, lambda e: e["mid"][2], z)
    mx = _match_extreme(es, lambda e: e["mid"][0], x)
    my = _match_extreme(es, lambda e: e["mid"][1], y)
    faces_set = None
    if on_face is not None:
        faces_set = set([on_face] if isinstance(on_face, str) else on_face)
    sel = []
    for i, e in enumerate(es):
        if e.get("seam") and not include_seams:
            continue
        if straight is not None and (e["type"] == "line") != straight:
            continue
        if circular is not None and (e["type"] == "circle") != circular:
            continue
        if direction is not None:
            if e["type"] != "line" or not _match_axis(e["direction"], direction):
                continue
        if not (mz[i] and mx[i] and my[i]):
            continue
        if convex is not None and e["convex"] != convex:
            continue
        if length_min is not None and e["length"] < length_min:
            continue
        if length_max is not None and e["length"] > length_max:
            continue
        if radius is not None and (e.get("radius") is None or abs(e["radius"] - radius) > 1e-3):
            continue
        if faces_set is not None and not (set(e["faces"]) & faces_set):
            continue
        if where is not None and not where(e):
            continue
        sel.append(e["id"])
    return sel


def _ids(sel) -> list[str]:
    if isinstance(sel, str):
        return [sel]
    return [s if isinstance(s, str) else s["id"] for s in sel]


def _resolve_edges(brep, sel):
    ids = _ids(sel)
    all_edges = brep.edges()
    out = []
    for i in ids:
        if not i.startswith("E"):
            raise ValueError(f"{i!r} is not an edge id")
        out.append(all_edges[int(i[1:])])
    return out


def _resolve_faces(brep, sel):
    ids = _ids(sel)
    all_faces = brep.faces()
    out = []
    for i in ids:
        if not i.startswith("F"):
            raise ValueError(f"{i!r} is not a face id")
        out.append(all_faces[int(i[1:])])
    return out


# --------------------------------------------------------------------------- #
# Measurement helpers available inside scripts
# --------------------------------------------------------------------------- #

def volume(part) -> float:
    """Exact volume of a closed body in mm^3."""
    return float(part.volume())


def bbox(part):
    """Tight bounding box: {'min': (x,y,z), 'max': (x,y,z), 'size': (l,w,h)}."""
    mn, mx = part.bounding_box(tight=True)
    return {"min": tuple(round(v, 4) for v in mn), "max": tuple(round(v, 4) for v in mx),
            "size": tuple(round(b - a, 4) for a, b in zip(mn, mx))}


def is_solid(part) -> bool:
    """True if the body is a closed manifold solid."""
    return bool(part.is_manifold_solid())


SCRIPT_NAMESPACE = {
    # primitives
    "box": box, "cylinder": cylinder, "cone": cone, "sphere": sphere, "torus": torus,
    "extrude": extrude, "revolve": revolve, "sweep_pipe": sweep_pipe,
    "loft": loft, "draft_extrude": draft_extrude, "sweep": sweep, "offset": offset,
    # booleans
    "union": union, "difference": difference, "intersection": intersection,
    # edits
    "fillet": fillet, "chamfer": chamfer, "fillet_all": fillet_all, "chamfer_all": chamfer_all,
    "shell": shell, "move": move, "rotate": rotate, "cut_plane": cut_plane, "copy": copy,
    "pattern_linear": pattern_linear, "pattern_polar": pattern_polar,
    # selection + measurement
    "faces": faces, "edges": edges, "volume": volume, "bbox": bbox, "is_solid": is_solid,
    # escape hatch to the raw kernel
    "sm": sm, "math": math,
}
