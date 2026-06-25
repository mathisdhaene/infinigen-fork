import argparse
import json
from pathlib import Path
import sys

import bpy


def _vector3(value):
    return [float(value[0]), float(value[1]), float(value[2])]


def _euler_deg(rot):
    return [float(v) for v in (rot.x * 180.0 / 3.141592653589793, rot.y * 180.0 / 3.141592653589793, rot.z * 180.0 / 3.141592653589793)]


def _collection_names(obj):
    names = []
    for collection in getattr(obj, "users_collection", []) or []:
        try:
            names.append(str(collection.name))
        except Exception:
            continue
    return names


def _room_hint(obj):
    tokens = []
    for value in [obj.name, *( _collection_names(obj) ), str(obj.parent.name) if obj.parent else ""]:
        lower = str(value).lower()
        for token in ("kitchen", "bedroom", "living-room", "dining-room", "bathroom", "hallway", "corridor"):
            if token in lower and token not in tokens:
                tokens.append(token)
    return tokens[0] if tokens else None


def _light_record(obj):
    data = obj.data
    matrix = obj.matrix_world.copy()
    loc = matrix.to_translation()
    rot = matrix.to_euler("XYZ")
    color = getattr(data, "color", None)

    rec = {
        "name": str(obj.name),
        "data_name": str(data.name),
        "type": str(data.type),
        "energy": float(getattr(data, "energy", 0.0)),
        "color": _vector3(color) if color is not None else None,
        "use_shadow": bool(getattr(data, "use_shadow", False)) if hasattr(data, "use_shadow") else None,
        "shadow_soft_size": float(getattr(data, "shadow_soft_size", 0.0)) if hasattr(data, "shadow_soft_size") else None,
        "spot_size": float(getattr(data, "spot_size", 0.0)) if hasattr(data, "spot_size") else None,
        "spot_blend": float(getattr(data, "spot_blend", 0.0)) if hasattr(data, "spot_blend") else None,
        "size": float(getattr(data, "size", 0.0)) if hasattr(data, "size") else None,
        "size_y": float(getattr(data, "size_y", 0.0)) if hasattr(data, "size_y") else None,
        "angle": float(getattr(data, "angle", 0.0)) if hasattr(data, "angle") else None,
        "location_xyz_m": _vector3(loc),
        "rotation_euler_xyz_deg": _euler_deg(rot),
        "parent_name": str(obj.parent.name) if obj.parent else None,
        "collection_names": _collection_names(obj),
        "room_hint": _room_hint(obj),
    }

    cycles = getattr(data, "cycles", None)
    if cycles is not None:
        rec["cycles"] = {
            "cast_shadow": bool(getattr(cycles, "cast_shadow", False)) if hasattr(cycles, "cast_shadow") else None,
            "max_bounces": int(getattr(cycles, "max_bounces", 0)) if hasattr(cycles, "max_bounces") else None,
            "use_multiple_importance_sampling": bool(getattr(cycles, "use_multiple_importance_sampling", False))
            if hasattr(cycles, "use_multiple_importance_sampling")
            else None,
            "is_portal": bool(getattr(cycles, "is_portal", False)) if hasattr(cycles, "is_portal") else None,
        }
    else:
        rec["cycles"] = None

    return rec


def export_blender_light_manifest(output_path: Path):
    lights = [_light_record(obj) for obj in bpy.data.objects if obj.type == "LIGHT"]
    payload = {
        "blend_path": str(bpy.data.filepath),
        "scene_name": str(bpy.context.scene.name) if bpy.context.scene else None,
        "light_count": len(lights),
        "lights": lights,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, required=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    args = parser.parse_args(argv)
    export_blender_light_manifest(args.output_path)


if __name__ == "__main__":
    main()
