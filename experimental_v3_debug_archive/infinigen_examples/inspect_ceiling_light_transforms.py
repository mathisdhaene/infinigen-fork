import json
import math
from pathlib import Path

import bpy
import mathutils


def _vec(value):
    return [float(value[0]), float(value[1]), float(value[2])]


def _deg(euler):
    return [float(math.degrees(euler.x)), float(math.degrees(euler.y)), float(math.degrees(euler.z))]


def _world_bbox(obj):
    mw = obj.matrix_world.copy()
    pts = []
    for corner in obj.bound_box:
        p = mw @ mathutils.Vector(corner)
        pts.append(_vec(p))
    return pts


def _record(obj):
    mw = obj.matrix_world.copy()
    return {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "children": [c.name for c in obj.children],
        "location_xyz_m": _vec(mw.to_translation()),
        "rotation_euler_xyz_deg": _deg(mw.to_euler("XYZ")),
        "dimensions_xyz_m": _vec(getattr(obj, "dimensions", (0.0, 0.0, 0.0))),
        "bound_box_world_xyz_m": _world_bbox(obj),
    }


def main():
    records = []
    for obj in bpy.data.objects:
        if "CeilingLightFactory" in obj.name or "PointLampFactory" in obj.name:
            records.append(_record(obj))
    out = Path("/tmp/inspect_ceiling_light_transforms.json")
    out.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
