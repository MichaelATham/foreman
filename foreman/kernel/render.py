"""
A tiny dependency-free software renderer (numpy + Pillow) that turns the
tessellated part into a labelled multi-view "contact sheet" PNG.

Why not a real renderer? The agent runs headless inside a container and needs
images synchronously after every build step. Painter's-algorithm flat shading
over ~50k triangles takes well under a second with Pillow, needs no GPU or
GL context, and - importantly - lets us stamp the BRep face ids (F3, F7 ...)
straight onto the image so the vision model can correlate what it *sees* with
the exact-geometry census it *reads*.
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEWS = {
    # name: (camera direction the camera looks *along*, up vector)
    "iso": ((-1.0, -1.0, -0.8), (0.0, 0.0, 1.0)),
    "front (-Y)": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "top (+Z)": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "right (+X)": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}

_PALETTE = np.array([
    (120, 170, 220), (235, 160, 90), (130, 200, 140), (210, 120, 150),
    (170, 150, 220), (220, 200, 100), (120, 200, 200), (200, 140, 100),
    (160, 190, 120), (190, 120, 190), (100, 160, 180), (230, 180, 150),
], dtype=np.float32)


_OTHER_BODIES = np.array([(150, 150, 160), (120, 130, 150), (175, 165, 140)], dtype=np.float32)


def _font(size=13):
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _basis(view_dir, up):
    f = np.asarray(view_dir, dtype=np.float64)
    f /= np.linalg.norm(f)
    u = np.asarray(up, dtype=np.float64)
    r = np.cross(f, u)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, (1.0, 0.0, 0.0))
    r /= np.linalg.norm(r)
    u2 = np.cross(r, f)
    return r, u2, f  # right, up, forward


def render_view(mesh: dict, view: str, size: int = 420, *, labels: dict | None = None,
                highlight: set | None = None, color_by_face: bool = True) -> Image.Image:
    """Render a single orthographic view. `labels` maps face index -> text to
    stamp at the face's (projected) centre when the face is visible."""
    pts = mesh["points"].astype(np.float64)
    tris = mesh["tris"]
    fids = mesh["face_ids"]
    vd, up = VIEWS[view]
    r, u, f = _basis(vd, up)
    img = Image.new("RGB", (size, size), (250, 250, 252))
    draw = ImageDraw.Draw(img)
    if len(tris) == 0:
        return img
    # project
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
    P = pts - centre
    X = P @ r
    Y = P @ u
    Z = P @ f  # depth along view: larger = farther
    extent = max(X.max() - X.min(), Y.max() - Y.min(), 1e-6)
    margin = 0.12
    scale = size * (1 - 2 * margin) / extent
    sx = size / 2 + X * scale
    sy = size / 2 - Y * scale

    # z-buffer rasterisation (2x supersampled), so interpenetrating or nested bodies
    # (e.g. a lid over a case) are drawn correctly per pixel
    normals = mesh["tri_normals"].astype(np.float64)
    light = np.array([-0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    facing = normals @ (-f)
    ndl = np.clip(normals @ light, 0, 1)
    shade = 0.45 + 0.55 * ndl
    cols = np.empty((len(tris), 3), dtype=np.float32)
    for t in range(len(tris)):
        fid = int(fids[t])
        if fid <= -10:  # another body of an assembly
            base = _OTHER_BODIES[(-10 - fid) % len(_OTHER_BODIES)]
        elif color_by_face and fid >= 0:
            base = _PALETTE[fid % len(_PALETTE)]
        else:
            base = np.array((170, 180, 200), dtype=np.float32)
        if highlight and fid in highlight:
            base = np.array((255, 90, 60), dtype=np.float32)
        cols[t] = np.clip(base * shade[t], 0, 255)
    ss = 2
    W = size * ss
    zbuf = np.full((W, W), np.inf)
    cbuf = np.empty((W, W, 3), dtype=np.float32)
    cbuf[:] = (250, 250, 252)
    SX, SY = sx * ss, sy * ss
    for t in np.where(facing > 0)[0]:
        a, b, c = tris[t]
        x0, y0, x1, y1, x2, y2 = SX[a], SY[a], SX[b], SY[b], SX[c], SY[c]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))), W - 1)
        if xmin > xmax or ymin > ymax:
            continue
        px, py = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5)
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        z = w0 * Z[a] + w1 * Z[b] + w2 * Z[c]
        sub = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        win = inside & (z < sub)
        sub[win] = z[win]
        cbuf[ymin:ymax + 1, xmin:xmax + 1][win] = cols[t]
    img = Image.fromarray(cbuf.astype(np.uint8), "RGB").resize((size, size), Image.LANCZOS)
    draw = ImageDraw.Draw(img)

    # BRep edges with hidden-line removal: sample each polyline and keep only the
    # samples not occluded by any triangle along the view direction.
    occl = _Occluder(pts, tris, f)
    vis_faces = set(int(x) for x in np.unique(fids[facing > 0.05]))
    for e in mesh["edges"]:
        q = e["points"].astype(np.float64)
        q = _resample(q, max_step=extent / 120.0)
        vis = occl.visible(q)
        ex = size / 2 + ((q - centre) @ r) * scale
        ey = size / 2 - ((q - centre) @ u) * scale
        run = []
        for i in range(len(q)):
            if vis[i]:
                run.append((ex[i], ey[i]))
            if (not vis[i] or i == len(q) - 1) and len(run) >= 2:
                draw.line(run, fill=(30, 30, 40), width=1)
                run = [] if not vis[i] else run
            elif not vis[i]:
                run = []

    # labels on visible faces
    if labels:
        font = _font(12)
        for fid, text in labels.items():
            if fid not in vis_faces:
                continue
            tsel = np.where((fids == fid) & (facing > 0.05))[0]
            if len(tsel) == 0:
                continue
            # anchor = centroid of the front-facing triangle nearest the face's mean
            cents3 = pts[tris[tsel]].mean(axis=1)
            mean3 = cents3.mean(axis=0)
            order_c = np.argsort(np.linalg.norm(cents3 - mean3, axis=1))[:12]
            vis_c = occl.visible(cents3[order_c])
            if not vis_c.any():
                continue
            anchor = cents3[order_c[int(np.argmax(vis_c))]]
            cx = size / 2 + float((anchor - centre) @ r) * scale
            cy = size / 2 - float((anchor - centre) @ u) * scale
            w = draw.textlength(text, font=font)
            draw.rectangle([cx - w / 2 - 2, cy - 8, cx + w / 2 + 2, cy + 8], fill=(255, 255, 255), outline=(60, 60, 60))
            draw.text((cx - w / 2, cy - 7), text, fill=(0, 0, 0), font=font)

    # axis gizmo + caption
    _draw_gizmo(draw, r, u, size)
    font = _font(13)
    draw.text((8, 6), view, fill=(40, 40, 40), font=font)
    return img


def _resample(q: np.ndarray, max_step: float) -> np.ndarray:
    """Insert points so no polyline segment is longer than max_step."""
    out = [q[0]]
    for a, b in zip(q[:-1], q[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / max_step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.asarray(out)


class _Occluder:
    """Orthographic occlusion test: a point is visible when no triangle lies
    between it and the camera along the view direction (vectorised
    Moller-Trumbore in chunks)."""

    def __init__(self, pts, tris, view_dir):
        self.d = np.asarray(view_dir, dtype=np.float64)
        self.d /= np.linalg.norm(self.d)
        self.v0 = pts[tris[:, 0]]
        self.e1 = pts[tris[:, 1]] - self.v0
        self.e2 = pts[tris[:, 2]] - self.v0
        self.h = np.cross(self.d, self.e2)
        a = np.einsum("ij,ij->i", self.e1, self.h)
        self.ok = np.abs(a) > 1e-12
        self.f = np.where(self.ok, 1.0 / np.where(self.ok, a, 1.0), 0.0)
        self.q_e1 = None
        self.back = 1e4

    def visible(self, points: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        out = np.ones(len(points), dtype=bool)
        chunk = max(1, int(2_000_000 // max(1, len(self.v0))))
        for s0 in range(0, len(points), chunk):
            P = points[s0:s0 + chunk]
            o = P - self.d * self.back                       # (S,3) ray origins far in front
            sv = o[:, None, :] - self.v0[None, :, :]         # (S,T,3)
            uu = self.f[None, :] * np.einsum("stj,tj->st", sv, self.h)
            qv = np.cross(sv, self.e1[None, :, :])
            vv = self.f[None, :] * (qv @ self.d)
            tt = self.f[None, :] * np.einsum("stj,tj->st", qv, self.e2)
            hit = self.ok[None, :] & (uu >= -1e-9) & (vv >= -1e-9) & (uu + vv <= 1 + 1e-9) & (tt > 1e-9) & (tt < self.back - eps)
            out[s0:s0 + chunk] = ~hit.any(axis=1)
        return out


def _draw_gizmo(draw, r, u, size):
    o = (36, size - 36)
    font = _font(11)
    for name, vec, col in (("X", (1, 0, 0), (200, 60, 60)), ("Y", (0, 1, 0), (60, 160, 60)), ("Z", (0, 0, 1), (60, 90, 220))):
        v = np.asarray(vec, dtype=np.float64)
        dx, dy = float(v @ r) * 24, -float(v @ u) * 24
        draw.line([o, (o[0] + dx, o[1] + dy)], fill=col, width=2)
        draw.text((o[0] + dx * 1.25 - 3, o[1] + dy * 1.25 - 6), name, fill=col, font=font)


def contact_sheet(mesh: dict, *, labels: dict | None = None, highlight: set | None = None,
                  caption: str = "", size: int = 420, views=("iso", "front (-Y)", "top (+Z)", "right (+X)")) -> Image.Image:
    tiles = [render_view(mesh, v, size, labels=labels, highlight=highlight) for v in views]
    cols = 2
    rows = math.ceil(len(tiles) / cols)
    cap_h = 24 if caption else 0
    sheet = Image.new("RGB", (cols * size, rows * size + cap_h), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * size, (i // cols) * size + cap_h))
    if caption:
        ImageDraw.Draw(sheet).text((8, 4), caption, fill=(20, 20, 20), font=_font(13))
    return sheet


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
