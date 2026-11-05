"""Kernel-layer tests: run in-process (no worker) for speed."""
import math

import pytest

from foreman.kernel import census, dsl, mesh

ns = dsl.SCRIPT_NAMESPACE


def bracket():
    base = ns["box"]((60, 40, 10))
    boss = ns["cylinder"](8, 20, at=(30, 20, 10))
    body = ns["union"](base, boss)
    return ns["difference"](body, ns["cylinder"](4, 60, at=(30, 20, -1)), ns["cylinder"](2.5, 30, at=(8, 8, -1)))


def test_primitives_volume():
    assert math.isclose(ns["volume"](ns["box"]((10, 20, 5))), 1000)
    assert math.isclose(ns["volume"](ns["cylinder"](5, 10)), math.pi * 25 * 10, rel_tol=1e-6)
    assert math.isclose(ns["volume"](ns["sphere"](3)), 4 / 3 * math.pi * 27, rel_tol=1e-3)


def test_cylinder_axis_and_position():
    c = ns["cylinder"](5, 10, at=(1, 2, 3), axis="-y")
    b = ns["bbox"](c)
    assert b["min"] == pytest.approx((-4, -8, -2), abs=1e-3)
    assert b["max"] == pytest.approx((6, 2, 8), abs=1e-3)


def test_booleans_are_non_consuming():
    a = ns["box"]((10, 10, 10))
    b = ns["cylinder"](3, 20, at=(5, 5, -5))
    r1 = ns["difference"](a, b)
    r2 = ns["difference"](a, b)  # inputs still usable
    assert math.isclose(ns["volume"](r1), ns["volume"](r2))
    assert ns["volume"](r1) < 1000


def test_census_classifies_and_orients():
    c = census.census(bracket())
    types = {f["type"] for f in c["faces"]}
    assert "planar" in types and "cylindrical" in types
    cyl = [f for f in c["faces"] if f["type"] == "cylindrical"]
    assert {round(f["radius"], 2) for f in cyl} == {8.0, 4.0, 2.5}
    top = [f for f in c["faces"] if f["type"] == "planar" and f["center"][2] == pytest.approx(30)]
    assert top and top[0]["normal"] == pytest.approx((0, 0, 1))
    # boss-to-plate junction is concave, outer plate edges convex, hole rims convex
    conv = {e["id"]: e["convex"] for e in c["edges"]}
    junction = [e for e in c["edges"] if e["type"] == "circle" and e.get("radius") == 8 and e["mid"][2] == pytest.approx(10)]
    assert junction and conv[junction[0]["id"]] is False
    outer = [e for e in c["edges"] if e["type"] == "line" and e["length"] == 60]
    assert all(conv[e["id"]] for e in outer)
    seams = [e for e in c["edges"] if e["seam"]]
    assert len(seams) == 3


def test_selectors():
    p = bracket()
    assert len(ns["edges"](p, direction="z", straight=True, convex=True)) == 4
    assert len(ns["edges"](p, convex=False, circular=True)) == 1
    assert ns["faces"](p, normal="+z", z="max") == [f["id"] for f in census.census(p)["faces"] if f["center"][2] == pytest.approx(30)]
    assert len(ns["faces"](p, type="cylindrical")) == 3


def test_fillet_and_guard():
    p = bracket()
    f = ns["fillet"](p, ns["edges"](p, direction="z", straight=True, convex=True), 5)
    assert ns["is_solid"](f) and ns["volume"](f) < ns["volume"](p)
    with pytest.raises(ValueError, match="implausible|not a closed"):
        ns["fillet"](ns["box"]((10, 10, 10)), ns["edges"](ns["box"]((10, 10, 10))), 9)
    with pytest.raises(ValueError, match="no edges"):
        ns["fillet"](p, [], 1)


def test_shell_extrude_revolve():
    cup = ns["shell"](ns["cylinder"](20, 30), 2, ns["faces"](ns["cylinder"](20, 30), normal="+z"))
    assert ns["is_solid"](cup)
    assert math.isclose(ns["volume"](cup), math.pi * 400 * 30 - math.pi * 18 ** 2 * 28, rel_tol=1e-4)
    ex = ns["extrude"]([(0, 0), (30, 0), (30, 20), (0, 20)], 5, axis="x", at=(5, 0, 0))
    assert ns["bbox"](ex)["min"] == pytest.approx((5, 0, 0), abs=1e-6)
    assert ns["bbox"](ex)["size"] == pytest.approx((5, 30, 20), abs=1e-6)
    ey = ns["extrude"]([(0, 0), (30, 0), (30, 20), (0, 20)], 5, axis="-y", at=(0, 0, 0))
    assert ns["bbox"](ey)["min"] == pytest.approx((0, -5, 0), abs=1e-6)
    assert ns["bbox"](ey)["size"] == pytest.approx((30, 5, 20), abs=1e-6)
    rev = ns["revolve"]([(0, 0), (10, 0), (10, 30), (0, 30)])
    assert math.isclose(ns["volume"](rev), math.pi * 100 * 30, rel_tol=1e-6)
    hexnut = ns["extrude"](("polygon", 6, 17 / math.sqrt(3)), 8)
    assert ns["bbox"](hexnut)["size"][2] == pytest.approx(8)


def test_tessellation_face_map_and_viewer_payload():
    p = bracket()
    m = mesh.tessellate(p)
    assert len(m["tris"]) > 100
    assert (m["face_ids"] >= 0).all()
    assert len(set(m["face_ids"].tolist())) == p.face_count()
    v = mesh.to_viewer_json(m)
    assert len(v["positions"]) == len(m["tris"]) * 9 and len(v["faceIds"]) == len(m["tris"]) * 3
    assert len(v["edges"]) == p.edge_count()


def test_loft_draft_sweep_offset():
    l = ns["loft"]([("circle", 10), ("rect", 12, 12)], [0, 30])
    assert ns["is_solid"](l) and ns["bbox"](l)["size"] == pytest.approx((20, 20, 30), abs=1e-3)
    l3 = ns["loft"]([("circle", 10), ("circle", 6), ("circle", 9)], [0, 20, 40], smooth=True)
    assert ns["is_solid"](l3) and ns["volume"](l3) < math.pi * 100 * 40
    d = ns["draft_extrude"](("rect", 40, 30), 20, 10)
    assert ns["is_solid"](d) and ns["volume"](d) < 40 * 30 * 20
    sw = ns["sweep"](("circle", 3), [(0, 0, 0), (20, 0, 5), (40, 10, 10), (60, 30, 10)])
    assert ns["is_solid"](sw) and 1800 < ns["volume"](sw) < 2300  # ~ pi r^2 * path length
    assert math.isclose(ns["volume"](ns["offset"](ns["box"]((10, 10, 10)), 1)), 12 ** 3, rel_tol=1e-6)


def test_usd_roundtrip_census_on_imported_part(tmp_path):
    import usd_brep as sm
    p = bracket()
    path = str(tmp_path / "p.usda")
    sm.usd.export_brep(p, path)
    (b,) = sm.usd.import_breps(path, heal=True)
    c = census.census(b)
    # edges come back as NURBS; the census still classifies them geometrically
    assert sum(e["type"] == "line" for e in c["edges"]) >= 12  # the 12 box edges (plus seams)
    circles = [e for e in c["edges"] if e["type"] == "circle"]
    assert {round(e["radius"], 2) for e in circles} == {8.0, 4.0, 2.5}
    assert len(ns["edges"](b, direction="z", straight=True, convex=True)) == 4
    assert len(ns["edges"](b, convex=False, circular=True)) == 1
