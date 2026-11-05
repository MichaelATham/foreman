"""
Multi-part fit checks.

A script can declare `assembly = {"base": base, "lid": lid}` with every body in its
assembled position. `part` stays the primary body (census, face ids, measurements);
the other bodies are rendered, exported and, most importantly, checked against each
other after every build:

    relationship  separate / touching / interpenetrating / a_contains_b / b_contains_a
    min_gap_mm    smallest clearance between the two bodies (0 when touching/overlapping)
    interference  volume and bounding box of the overlap (boolean intersection)

That is enough to answer "does the lid fit the case?" exactly: a sliding fit should be
separate with a small positive gap; a press/snap fit should show a small, deliberate
interference located at the tabs, not across a whole wall.
"""
from __future__ import annotations

import numpy as np
import usd_brep as sm

# face_ids for non-primary bodies in the merged render mesh: -10, -11, ...
OTHER_BODY_ID_BASE = -10


def _r(v, n=3):
    if isinstance(v, (tuple, list)):
        return tuple(round(float(c), n) for c in v)
    return round(float(v), n)


def validate(assembly, part) -> dict:
    if not isinstance(assembly, dict) or not assembly:
        raise ValueError("`assembly` must be a non-empty dict like {'base': base, 'lid': lid}")
    for name, body in assembly.items():
        if not isinstance(name, str):
            raise ValueError("assembly keys must be part names (strings)")
        if not hasattr(body, "faces"):
            raise ValueError(f"assembly['{name}'] must be a body, got {type(body).__name__}")
    if not any(b is part for b in assembly.values()):
        raise ValueError("`part` must be one of the bodies in `assembly` (it is the primary body "
                         "that gets face ids); e.g. part = assembly['base']")
    return assembly


def fit(a, b) -> dict:
    """Exact fit between two bodies."""
    rel = a.relationship_to(b)
    out = {"relationship": rel["relationship"], "min_gap_mm": _r(rel["distance"], 4)}
    if rel["relationship"] in ("interpenetrating", "a_contains_b", "b_contains_a", "touching"):
        try:
            inter = sm.boolean_intersection(a.copy(), b.copy())
            if inter.face_count():
                v = float(inter.volume())
                if v > 1e-6:
                    mn, mx = inter.bounding_box(tight=True)
                    out["interference_mm3"] = _r(v, 3)
                    out["interference_bbox"] = {"min": _r(mn), "max": _r(mx)}
        except Exception as e:  # noqa: BLE001
            out["interference_note"] = f"could not compute overlap volume: {e}"
    out.setdefault("interference_mm3", 0.0)
    return out


def pairwise(assembly: dict) -> list[dict]:
    names = list(assembly)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = fit(assembly[names[i]], assembly[names[j]])
            rows.append({"a": names[i], "b": names[j], **r})
    return rows


def summarize(assembly: dict, part, fits: list[dict]) -> str:
    lines = ["ASSEMBLY (all bodies in assembled position):"]
    for name, body in assembly.items():
        mn, mx = body.bounding_box(tight=True)
        size = tuple(round(b - a, 3) for a, b in zip(mn, mx))
        primary = "  <- primary `part` (face ids / census refer to this body)" if body is part else ""
        vol = round(float(body.volume()), 1) if body.is_manifold_solid() else "n/a (not a closed solid)"
        lines.append(f"  {name}: bbox {_r(mn)} -> {_r(mx)} size {size} volume {vol}{primary}")
    lines.append("FIT between bodies (exact, from the kernel):")
    for f in fits:
        s = f"  {f['a']} vs {f['b']}: {f['relationship']}, min gap {f['min_gap_mm']} mm"
        if f.get("interference_mm3"):
            s += f", INTERFERENCE {f['interference_mm3']} mm3 in bbox {f['interference_bbox']['min']} -> {f['interference_bbox']['max']}"
        lines.append(s)
    return "\n".join(lines)


def merge_meshes(primary: dict, others: list[dict]) -> dict:
    """Combine the primary mesh with the other bodies' meshes for rendering.
    Other bodies get face ids OTHER_BODY_ID_BASE - k so they are drawn in their own
    colour and never labelled or highlighted as primary faces."""
    pts = [primary["points"]]
    tris = [primary["tris"]]
    nrm = [primary["tri_normals"]]
    fids = [primary["face_ids"]]
    edges = list(primary["edges"])
    offset = len(primary["points"])
    for k, m in enumerate(others):
        pts.append(m["points"])
        tris.append(m["tris"] + offset)
        nrm.append(m["tri_normals"])
        fids.append(np.full(len(m["tris"]), OTHER_BODY_ID_BASE - k, dtype=np.int32))
        edges += [{"id": f"X{k}_{e['id']}", "points": e["points"]} for e in m["edges"]]
        offset += len(m["points"])
    merged = dict(primary)
    merged.update(points=np.concatenate(pts), tris=np.concatenate(tris),
                  tri_normals=np.concatenate(nrm), face_ids=np.concatenate(fids), edges=edges)
    return merged
