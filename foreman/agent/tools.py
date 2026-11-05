"""Tool definitions exposed to the model (Anthropic tool-use schema)."""

TOOLS = [
    {
        "name": "plan",
        "description": (
            "Declare the checklist of concrete, verifiable requirements you extracted from the brief "
            "(plus your stated assumptions) BEFORE the first build. Each item gets an id R1, R2, ... that "
            "you must reference in the finish report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "requirements": {"type": "array", "items": {"type": "string"},
                                 "description": "One short, measurable statement each, e.g. 'plate 80 x 50 x 8 mm'."},
                "assumptions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["requirements"],
        },
    },
    {
        "name": "build",
        "description": (
            "Run a feature script in the CAD kernel and replace the current part with its result. "
            "Returns kernel metrics, a topology census and a rendered 4-view image with face ids. "
            "The whole script is re-run each time, so always send the complete script."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "Complete feature script ending with `part = ...`."},
                "note": {"type": "string", "description": "One line: what changed and why (shown in the version history)."},
            },
            "required": ["script", "note"],
        },
    },
    {
        "name": "look",
        "description": (
            "Render the current part again: optionally highlight faces (by id) in orange, choose views, "
            "or cut a section through the part with a plane (shows the cross-section as a 2D drawing on a 5 mm grid - "
            "the right way to verify wall thickness, hollow interiors and hole depths)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "highlight_faces": {"type": "array", "items": {"type": "string"}, "description": "Face ids to highlight, e.g. ['F3']."},
                "views": {"type": "array", "items": {"type": "string", "enum": ["iso", "front (-Y)", "top (+Z)", "right (+X)"]}},
                "section_point": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                                  "description": "A point on the section plane (mm)."},
                "section_normal": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                                   "description": "Normal of the section plane, e.g. [0,1,0] for an XZ section."},
            },
        },
    },
    {
        "name": "measure",
        "description": (
            "Exact measurements from the kernel on the current part. kinds: "
            "'probe' {point} -> is the point inside material, distance to the nearest surface; "
            "'distance' {a, b} -> minimum distance between two face ids, or a face id and a point, or two points; "
            "'thickness' {point, direction} -> fire a ray and list where it enters/exits material and the material spans (wall thicknesses) along it; "
            "'face' {id} / 'edge' {id} -> full details of one census entry; "
            "'fit' {a, b} -> exact fit between two named bodies of the `assembly` (relationship, min gap, "
            "interference volume and where it is). With an assembly, 'thickness' reports spans per body."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["probe", "distance", "thickness", "face", "edge", "fit"]},
                "point": {"type": "array", "items": {"type": "number"}},
                "direction": {"type": "array", "items": {"type": "number"}},
                "a": {"description": "face id or [x,y,z]; for 'fit', an assembly body name"},
                "b": {"description": "face id or [x,y,z]; for 'fit', an assembly body name"},
                "id": {"type": "string"},
            },
            "required": ["kind"],
        },
    },
    {
        "name": "finish",
        "description": "Hand the part over with a verification report. Call this only when every requirement has been checked.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "2-4 sentences describing the design and key assumptions."},
                "report": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "R1, R2, ... from plan (omit for extra checks)"},
                            "requirement": {"type": "string"},
                            "method": {"type": "string", "description": "How it was verified (metric, census, measure, image, section)."},
                            "result": {"type": "string", "description": "The measured/observed value."},
                            "pass": {"type": "boolean"},
                        },
                        "required": ["requirement", "method", "result", "pass"],
                    },
                },
            },
            "required": ["summary", "report"],
        },
    },
]
