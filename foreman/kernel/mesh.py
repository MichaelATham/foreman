"""
Tessellation for the viewer and the renderer.

The kernel tessellates the exact BRep into a `PolyBrep`; we turn that into flat
numpy arrays and, crucially, map every triangle back to the BRep face it came
from (`face_ids`), using the kernel's `find_face` on triangle centroids. That
mapping is what lets the viewer highlight "F7" and the renderer label faces.
"""
from __future__ import annotations

import numpy as np
import usd_brep as sm


def tessellate(brep, *, quality: str = "medium") -> dict:
    q = {
        "low": dict(chord_height_tolerance=0.2, surface_angle_tolerance_deg=30.0, curve_angle_tolerance_deg=30.0),
        "medium": dict(chord_height_tolerance=0.05, surface_angle_tolerance_deg=12.0, curve_angle_tolerance_deg=12.0),
        "high": dict(chord_height_tolerance=0.01, surface_angle_tolerance_deg=6.0, curve_angle_tolerance_deg=6.0),
    }[quality]
    poly = sm.tessellate(brep, **{**sm.tessellation_defaults(), **q})
    arr = poly.to_numpy_arrays(face_varying_normals=True)
    points = np.asarray(arr["points"], dtype=np.float32)
    stream = np.asarray(arr["faces"], dtype=np.int32)
    tris = _triangulate_stream(stream)
    # map triangles -> BRep faces
    faces = brep.faces()
    face_ids = np.full(len(tris), -1, dtype=np.int32)
    if len(tris):
        cents = points[tris].mean(axis=1)
        for i, c in enumerate(cents):
            try:
                fh = sm.find_face(brep, (float(c[0]), float(c[1]), float(c[2])))
            except Exception:
                continue
            for k, f in enumerate(faces):
                if f == fh:
                    face_ids[i] = k
                    break
    # flat-shaded normals per triangle
    p0, p1, p2 = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    l = np.linalg.norm(n, axis=1, keepdims=True)
    l[l == 0] = 1
    n = n / l
    # BRep edges as polylines, for the viewer and renderer
    edge_lines = boundary_polylines(brep)
    return {
        "points": points,
        "tris": tris,
        "tri_normals": n.astype(np.float32),
        "face_ids": face_ids,
        "edges": edge_lines,
    }


def _triangulate_stream(stream: np.ndarray) -> np.ndarray:
    """Flat PyVista-style stream [n, i0..in-1, n, ...] -> (T,3) triangles (fan)."""
    tris = []
    i = 0
    n_total = len(stream)
    while i < n_total:
        n = int(stream[i])
        idx = stream[i + 1:i + 1 + n]
        for k in range(1, n - 1):
            tris.append((idx[0], idx[k], idx[k + 1]))
        i += 1 + n
    return np.asarray(tris, dtype=np.int32).reshape(-1, 3)


def boundary_polylines(brep, angle_deg: float = 8.0) -> list[dict]:
    """Sample every BRep edge as a polyline: [{'id': 'E3', 'points': (N,3)}]."""
    out = []
    try:
        pts, counts, edges = sm.tessellate_boundaries(brep, angle_tolerance_deg=angle_deg, allow_partial=True)
    except Exception:
        return out
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    all_edges = brep.edges()
    offset = 0
    for j, cnt in enumerate(counts):
        cnt = int(cnt)
        seg = pts[offset:offset + cnt]
        offset += cnt
        eid = j
        try:
            for k, e in enumerate(all_edges):
                if e == edges[j]:
                    eid = k
                    break
        except Exception:
            pass
        if cnt >= 2:
            out.append({"id": f"E{eid}", "points": seg})
    return out


def to_viewer_json(mesh: dict) -> dict:
    """Compact JSON-able payload for the Three.js viewer."""
    pts = mesh["points"]
    tris = mesh["tris"]
    # Unroll to non-indexed triangles so each triangle carries its own face id
    # (lets the viewer colour/highlight per BRep face with a single attribute).
    flat = pts[tris].reshape(-1, 3)
    normals = np.repeat(mesh["tri_normals"], 3, axis=0)
    fids = np.repeat(mesh["face_ids"], 3)
    return {
        "positions": flat.astype(np.float32).ravel().round(4).tolist(),
        "normals": normals.astype(np.float32).ravel().round(3).tolist(),
        "faceIds": fids.tolist(),
        "edges": [{"id": e["id"], "points": e["points"].ravel().round(4).tolist()} for e in mesh["edges"]],
        "triangleCount": int(len(tris)),
    }
