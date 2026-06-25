import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import unreal


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bedlam360_mini_validation as mini

mini = importlib.reload(mini)


DEFAULT_OUTPUT_ROOT = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_erp_convention_calibration")
DEFAULT_CAMERA_POSE = {
    "x": 0.0,
    "y": 0.0,
    "z": 140.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
}
DEFAULT_MARKER_RADIUS_CM = 300.0
DEFAULT_MARKER_PREFIX = "BEDLAM360_CALIB_"
POST_SCRIPT_PATH = SCRIPT_DIR / "bedlam360_erp_convention_calibration_post.py"

MARKER_SPECS = [
    {"name": "plus_x", "label": "+X", "offset_cm": (1.0, 0.0, 0.0), "rgb": (255, 0, 0), "bgr": (0, 0, 255)},
    {"name": "minus_x", "label": "-X", "offset_cm": (-1.0, 0.0, 0.0), "rgb": (0, 255, 255), "bgr": (255, 255, 0)},
    {"name": "plus_y", "label": "+Y", "offset_cm": (0.0, 1.0, 0.0), "rgb": (0, 255, 0), "bgr": (0, 255, 0)},
    {"name": "minus_y", "label": "-Y", "offset_cm": (0.0, -1.0, 0.0), "rgb": (255, 0, 255), "bgr": (255, 0, 255)},
    {"name": "plus_z", "label": "+Z", "offset_cm": (0.0, 0.0, 1.0), "rgb": (0, 128, 255), "bgr": (255, 128, 0)},
    {"name": "minus_z", "label": "-Z", "offset_cm": (0.0, 0.0, -1.0), "rgb": (255, 255, 0), "bgr": (0, 255, 255)},
]


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path, payload):
    path = Path(path)
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _make_transform_matrix(x_cm, y_cm, z_cm, yaw_deg, pitch_deg, roll_deg):
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    rot = _mat3_mul(_mat3_mul(rz, ry), rx)
    return [
        [rot[0][0], rot[0][1], rot[0][2], float(x_cm)],
        [rot[1][0], rot[1][1], rot[1][2], float(y_cm)],
        [rot[2][0], rot[2][1], rot[2][2], float(z_cm)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat3_mul(a, b):
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(sum(a[i][k] * b[k][j] for k in range(3)))
        out.append(row)
    return out


def _mat3_vec_mul(a, v):
    return [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]


def _transform_point(point_xyz, matrix4x4):
    x, y, z = point_xyz
    return [
        matrix4x4[0][0] * x + matrix4x4[0][1] * y + matrix4x4[0][2] * z + matrix4x4[0][3],
        matrix4x4[1][0] * x + matrix4x4[1][1] * y + matrix4x4[1][2] * z + matrix4x4[1][3],
        matrix4x4[2][0] * x + matrix4x4[2][1] * y + matrix4x4[2][2] * z + matrix4x4[2][3],
    ]


def _inverse_rigid_transform(matrix4x4):
    rot = [
        [matrix4x4[0][0], matrix4x4[0][1], matrix4x4[0][2]],
        [matrix4x4[1][0], matrix4x4[1][1], matrix4x4[1][2]],
        [matrix4x4[2][0], matrix4x4[2][1], matrix4x4[2][2]],
    ]
    trans = [matrix4x4[0][3], matrix4x4[1][3], matrix4x4[2][3]]
    rot_t = [
        [rot[0][0], rot[1][0], rot[2][0]],
        [rot[0][1], rot[1][1], rot[2][1]],
        [rot[0][2], rot[1][2], rot[2][2]],
    ]
    inv_t = _mat3_vec_mul(rot_t, [-trans[0], -trans[1], -trans[2]])
    return [
        [rot_t[0][0], rot_t[0][1], rot_t[0][2], inv_t[0]],
        [rot_t[1][0], rot_t[1][1], rot_t[1][2], inv_t[1]],
        [rot_t[2][0], rot_t[2][1], rot_t[2][2], inv_t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _project_camera_xyz_to_erp(points_cam_xyz, width, height):
    x, y, z = points_cam_xyz
    radius = math.sqrt(x * x + y * y + z * z)
    valid = math.isfinite(radius) and radius > 1e-8
    lon = math.atan2(y, x)
    lat = math.atan2(z, math.sqrt(x * x + y * y))
    px = ((lon / (2.0 * math.pi)) + 0.5) * float(width)
    py = (0.5 - (lat / math.pi)) * float(height)
    px = px % float(width)
    py = min(max(py, 0.0), float(height - 1))
    return [float(px), float(py)], bool(valid), float(lon), float(lat)


def _look_at_rotation(source_loc, target_loc):
    dx = target_loc["x"] - source_loc["x"]
    dy = target_loc["y"] - source_loc["y"]
    dz = target_loc["z"] - source_loc["z"]
    yaw = math.degrees(math.atan2(dy, dx))
    horizontal = max(1e-6, math.hypot(dx, dy))
    pitch = math.degrees(math.atan2(dz, horizontal))
    return {"pitch": pitch, "yaw": yaw, "roll": 0.0}


def _clear_existing_markers(prefix=DEFAULT_MARKER_PREFIX):
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cleared = 0
    for actor in actor_subsystem.get_all_level_actors():
        try:
            label = actor.get_actor_label()
        except Exception:
            continue
        if not str(label).startswith(prefix):
            continue
        try:
            actor_subsystem.destroy_actor(actor)
            cleared += 1
        except Exception:
            pass
    unreal.log(f"[BEDLAM360] Cleared {cleared} ERP calibration actors with prefix '{prefix}'")


def _find_text_component(actor):
    components = actor.get_components_by_class(unreal.TextRenderComponent)
    return components[0] if components else None


def _spawn_text_marker(marker_spec, camera_pose, radius_cm):
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    dx, dy, dz = marker_spec["offset_cm"]
    location = {
        "x": float(camera_pose["x"] + dx * radius_cm),
        "y": float(camera_pose["y"] + dy * radius_cm),
        "z": float(camera_pose["z"] + dz * radius_cm),
    }
    facing = _look_at_rotation(location, camera_pose)
    actor = actor_subsystem.spawn_actor_from_class(
        unreal.TextRenderActor,
        unreal.Vector(location["x"], location["y"], location["z"]),
        unreal.Rotator(facing["pitch"], facing["yaw"], facing["roll"]),
    )
    actor.set_actor_label(f"{DEFAULT_MARKER_PREFIX}{marker_spec['name']}")
    component = _find_text_component(actor)
    if component is None:
        raise RuntimeError(f"Spawned TextRenderActor has no TextRenderComponent: {marker_spec['name']}")
    component.set_editor_property("text", marker_spec["label"])
    component.set_editor_property("world_size", 120.0)
    component.set_editor_property("horizontal_alignment", unreal.HorizTextAligment.EHTA_CENTER)
    component.set_editor_property("vertical_alignment", unreal.VerticalTextAligment.EVRTA_TEXT_CENTER)
    color = unreal.Color(int(marker_spec["rgb"][0]), int(marker_spec["rgb"][1]), int(marker_spec["rgb"][2]), 255)
    component.set_text_render_color(color)
    component.set_editor_property("x_scale", 2.0)
    component.set_editor_property("y_scale", 2.0)
    component.set_editor_property("hidden_in_game", False)
    return {
        "name": marker_spec["name"],
        "label": marker_spec["label"],
        "rgb": marker_spec["rgb"],
        "bgr": marker_spec["bgr"],
        "world_location_cm": location,
        "actor_label": actor.get_actor_label(),
    }


def _predicted_marker_pixels(camera_pose, markers, width, height):
    camera_to_world = _make_transform_matrix(
        camera_pose["x"],
        camera_pose["y"],
        camera_pose["z"],
        camera_pose["yaw"],
        camera_pose["pitch"],
        camera_pose["roll"],
    )
    world_to_camera = _inverse_rigid_transform(camera_to_world)
    predictions = {}
    for marker in markers:
        world = [
            marker["world_location_cm"]["x"],
            marker["world_location_cm"]["y"],
            marker["world_location_cm"]["z"],
        ]
        camera = _transform_point(world, world_to_camera)
        points_2d, valid, lon, lat = _project_camera_xyz_to_erp(camera, width=width, height=height)
        predictions[marker["name"]] = {
            "camera_xyz_cm": camera,
            "predicted_pixel_xy": None if not valid else points_2d,
            "predicted_longitude_deg": math.degrees(lon),
            "predicted_latitude_deg": math.degrees(lat),
        }
    return predictions


def run_erp_convention_calibration(output_root=DEFAULT_OUTPUT_ROOT, marker_radius_cm=DEFAULT_MARKER_RADIUS_CM):
    output_root = _ensure_dir(output_root)
    images_dir = _ensure_dir(output_root / "images")
    metadata_dir = _ensure_dir(output_root / "metadata")
    overlays_dir = _ensure_dir(output_root / "overlays")

    _clear_existing_markers()
    mini.reconstruct_full_bedlam_scene.clear_existing_bedlam_bodies()

    markers = [_spawn_text_marker(spec, DEFAULT_CAMERA_POSE, marker_radius_cm) for spec in MARKER_SPECS]

    actor = mini.capture_scene_cube.find_scene_capture_cube(mini.DEFAULT_ACTOR_LABEL)
    component = mini.capture_scene_cube.get_capture_component(actor)
    texture_target = mini.capture_scene_cube.get_texture_target(component)
    export_lib = mini.unreal.BEDLAM360ExportLibrary
    mini.capture_scene_cube.set_actor_pose(
        actor,
        DEFAULT_CAMERA_POSE["x"],
        DEFAULT_CAMERA_POSE["y"],
        DEFAULT_CAMERA_POSE["z"],
        DEFAULT_CAMERA_POSE["pitch"],
        DEFAULT_CAMERA_POSE["yaw"],
        DEFAULT_CAMERA_POSE["roll"],
    )

    frame_name = "erp_convention_calibration"
    hdr_path = images_dir / f"{frame_name}.hdr"
    exr_path = images_dir / f"{frame_name}.exr"
    capture_result = mini.stabilized_capture_and_export(
        actor=actor,
        component=component,
        texture_target=texture_target,
        export_lib=export_lib,
        frame_name=frame_name,
        hdr_path=hdr_path,
        exr_path=exr_path,
        faces_dir=None,
    )
    png_path = images_dir / f"{frame_name}.png"
    metadata_path = metadata_dir / f"{frame_name}.json"

    predicted = _predicted_marker_pixels(DEFAULT_CAMERA_POSE, markers, width=2048, height=1024)
    preview_status = mini._run_preview_frame(exr_path, png_path, metadata_path, overlay=False)

    metadata_payload = {
        "frame_name": frame_name,
        "camera_pose_cm_deg": DEFAULT_CAMERA_POSE,
        "marker_radius_cm": marker_radius_cm,
        "markers": markers,
        "predicted_markers": predicted,
        "capture_result": capture_result,
        "preview_status": preview_status,
        "renderer_projection_assumption": {
            "predicted_center_axis": "+X",
            "predicted_right_axis": "+Y",
            "predicted_up_axis": "+Z",
        },
        "coordinate_system": "Unreal: centimeters, X forward, Y right, Z up",
    }
    _write_json(metadata_path, metadata_payload)

    if not preview_status.get("success"):
        raise RuntimeError(f"Calibration PNG generation failed: {preview_status}")

    if not POST_SCRIPT_PATH.is_file():
        raise RuntimeError(f"Calibration post-process script not found: {POST_SCRIPT_PATH}")
    report_path = metadata_dir / "erp_convention_calibration_report.json"
    overlay_path = overlays_dir / f"{frame_name}_predicted_vs_actual.png"
    command = ["python3", str(POST_SCRIPT_PATH), str(metadata_path), str(png_path), str(report_path), str(overlay_path)]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to launch ERP convention calibration post-process: {exc}") from exc
    if completed.stdout:
        unreal.log(f"[BEDLAM360] ERP calibration post stdout:\n{completed.stdout}")
    if completed.stderr:
        unreal.log_warning(f"[BEDLAM360] ERP calibration post stderr:\n{completed.stderr}")
    if completed.returncode != 0:
        raise RuntimeError(f"ERP convention calibration post-process failed with code {completed.returncode}")

    unreal.log(f"[BEDLAM360] ERP convention calibration report: {report_path}")
    unreal.log(f"[BEDLAM360] ERP convention calibration overlay: {overlay_path}")
    print(f"[BEDLAM360] ERP convention calibration report: {report_path}")
    print(f"[BEDLAM360] ERP convention calibration overlay: {overlay_path}")
    return report_path


if __name__ == "__main__":
    run_erp_convention_calibration()
