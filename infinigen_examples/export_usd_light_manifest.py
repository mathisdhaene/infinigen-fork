import argparse
import json
from pathlib import Path


def _quat_to_euler_xyz_deg(w, x, y, z):
    import math

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch_y = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return [math.degrees(roll_x), math.degrees(pitch_y), math.degrees(yaw_z)]


def export_usd_light_manifest(stage_path: Path, output_path: Path):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {stage_path}")

    xform_cache = UsdGeom.XformCache()
    lights = []
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name not in {"SphereLight", "RectLight", "DomeLight", "DistantLight"}:
            continue
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        quat = matrix.ExtractRotationQuat()
        imag = quat.GetImaginary()
        attrs = {}
        for attr_name in (
            "intensity",
            "exposure",
            "radius",
            "width",
            "height",
            "length",
            "color",
            "enableColorTemperature",
            "colorTemperature",
        ):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.HasValue():
                value = attr.Get()
                if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                    value = [float(x) for x in value]
                attrs[attr_name] = value
        x_axis = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        y_axis = matrix.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
        z_axis = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
        lights.append(
            {
                "prim_path": prim.GetPath().pathString,
                "prim_name": prim.GetName(),
                "type": type_name,
                "translation_xyz_m": [float(translation[0]), float(translation[1]), float(translation[2])],
                "rotation_euler_xyz_deg": _quat_to_euler_xyz_deg(
                    float(quat.GetReal()),
                    float(imag[0]),
                    float(imag[1]),
                    float(imag[2]),
                ),
                "x_axis": [float(x_axis[0]), float(x_axis[1]), float(x_axis[2])],
                "y_axis": [float(y_axis[0]), float(y_axis[1]), float(y_axis[2])],
                "z_axis": [float(z_axis[0]), float(z_axis[1]), float(z_axis[2])],
                "attrs": attrs,
            }
        )

    payload = {
        "stage_path": str(stage_path),
        "light_count": len(lights),
        "lights": lights,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    export_usd_light_manifest(args.stage_path, args.output_path)


if __name__ == "__main__":
    main()
