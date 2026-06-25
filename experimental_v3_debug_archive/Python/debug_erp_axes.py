import argparse
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--analyze-root")
    parser.add_argument("--png-path")
    parser.add_argument("--report-path")
    parser.add_argument("--overlay-path")
    parser.add_argument("--axis-name")
    parser.add_argument("--pred-x", type=float)
    parser.add_argument("--pred-y", type=float)
    parser.add_argument("--summary-path")
    return parser


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


def _project_camera_xyz_to_erp(point_cam_xyz, width, height):
    x, y, z = point_cam_xyz
    lon = math.atan2(y, x)
    lat = math.atan2(z, math.sqrt(x * x + y * y))
    px = ((lon / (2.0 * math.pi)) + 0.5) * float(width)
    py = (0.5 - (lat / math.pi)) * float(height)
    px = px % float(width)
    py = min(max(py, 0.0), float(height - 1))
    return [float(px), float(py)], float(math.degrees(lon)), float(math.degrees(lat))


def _predicted_marker_pixel(camera_pose, world_location_cm, width, height):
    camera_to_world = _make_transform_matrix(
        camera_pose["x"],
        camera_pose["y"],
        camera_pose["z"],
        camera_pose["yaw"],
        camera_pose["pitch"],
        camera_pose["roll"],
    )
    world_to_camera = _inverse_rigid_transform(camera_to_world)
    point_cam = _transform_point(world_location_cm, world_to_camera)
    pixel, lon_deg, lat_deg = _project_camera_xyz_to_erp(point_cam, width=width, height=height)
    return {
        "camera_xyz_cm": point_cam,
        "predicted_pixel_xy": pixel,
        "predicted_longitude_deg": lon_deg,
        "predicted_latitude_deg": lat_deg,
    }


def _run_postprocess(png_path, report_path, overlay_path, axis_name, pred_x, pred_y):
    import cv2
    import numpy as np

    image = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load PNG: {png_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_idx = None
    best_area = 0
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_idx = idx

    actual_bbox = None
    actual_centroid = None
    if best_idx is not None:
        x = int(stats[best_idx, cv2.CC_STAT_LEFT])
        y = int(stats[best_idx, cv2.CC_STAT_TOP])
        w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
        actual_bbox = {"min_x": x, "min_y": y, "max_x": x + w, "max_y": y + h}
        actual_centroid = [float(centroids[best_idx][0]), float(centroids[best_idx][1])]

    overlay = image.copy()
    px = int(round(float(pred_x)))
    py = int(round(float(pred_y)))
    cv2.drawMarker(overlay, (px, py), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.putText(overlay, f"Pred {axis_name}", (px + 8, max(20, py - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    if actual_bbox is not None:
        cv2.rectangle(
            overlay,
            (actual_bbox["min_x"], actual_bbox["min_y"]),
            (actual_bbox["max_x"], actual_bbox["max_y"]),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    if actual_centroid is not None:
        cx = int(round(actual_centroid[0]))
        cy = int(round(actual_centroid[1]))
        cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(overlay, f"Actual {axis_name}", (cx + 8, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(overlay_path), overlay)

    report = {
        "axis_name": axis_name,
        "predicted_pixel_xy": [float(pred_x), float(pred_y)],
        "actual_bbox": actual_bbox,
        "actual_centroid": actual_centroid,
        "actual_detected_area_px": int(best_area),
        "overlay_png": str(overlay_path),
        "png_path": str(png_path),
    }
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _wrap_angle_deg(value_deg):
    return ((float(value_deg) + 180.0) % 360.0) - 180.0


def _circular_mean_x(pixel_xs, width):
    import cv2
    import numpy as np

    if pixel_xs.size == 0:
        return None
    theta = (pixel_xs.astype(np.float64) / float(width)) * (2.0 * math.pi)
    mean_sin = float(np.mean(np.sin(theta)))
    mean_cos = float(np.mean(np.cos(theta)))
    if abs(mean_sin) < 1e-12 and abs(mean_cos) < 1e-12:
        return None
    mean_theta = math.atan2(mean_sin, mean_cos)
    x = ((mean_theta / (2.0 * math.pi)) % 1.0) * float(width)
    return float(x)


def _analyze_existing_root(root_path, summary_path=None):
    import cv2
    import numpy as np

    root = Path(root_path)
    images_dir = root / "images"
    metadata_dir = root / "metadata"
    overlays_dir = root / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    axis_json_paths = sorted(metadata_dir.glob("erp_axis_*.json"))
    axis_entries = []
    for meta_path in axis_json_paths:
        if meta_path.name.endswith("_report.json"):
            continue
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        axis_name = payload["axis_name"]
        png_path = images_dir / f"erp_axis_{axis_name}.png"
        image = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image: {png_path}")
        axis_entries.append(
            {
                "axis_name": axis_name,
                "meta_path": meta_path,
                "png_path": png_path,
                "payload": payload,
                "image": image,
            }
        )

    if not axis_entries:
        raise RuntimeError(f"No axis metadata found in {metadata_dir}")

    stack = np.stack([entry["image"].astype(np.float32) for entry in axis_entries], axis=0)
    background = np.median(stack, axis=0).astype(np.uint8)
    height, width = background.shape[:2]

    report_paths = []
    horizontal_deltas = []
    horizontal_axis_names = {"plus_x", "minus_x", "plus_y", "minus_y"}

    for entry in axis_entries:
        axis_name = entry["axis_name"]
        payload = entry["payload"]
        image = entry["image"]
        diff = cv2.absdiff(image, background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

        ys, xs = np.where(mask > 0)
        actual_centroid = None
        actual_bbox = None
        actual_area = int(xs.size)
        if xs.size > 0:
            actual_x = _circular_mean_x(xs, width=width)
            actual_y = float(np.mean(ys.astype(np.float64)))
            actual_centroid = [float(actual_x), float(actual_y)]
            actual_bbox = {
                "min_x": int(np.min(xs)),
                "min_y": int(np.min(ys)),
                "max_x": int(np.max(xs)) + 1,
                "max_y": int(np.max(ys)) + 1,
            }

        pred_x, pred_y = payload["predicted"]["predicted_pixel_xy"]
        pred_lon = float(payload["predicted"]["predicted_longitude_deg"])
        actual_lon = None if actual_centroid is None else _wrap_angle_deg(((actual_centroid[0] / float(width)) - 0.5) * 360.0)
        yaw_delta = None if actual_lon is None else _wrap_angle_deg(actual_lon - pred_lon)

        overlay = image.copy()
        cv2.drawMarker(overlay, (int(round(pred_x)), int(round(pred_y))), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
        cv2.putText(overlay, f"Pred {axis_name}", (int(round(pred_x)) + 8, max(20, int(round(pred_y)) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        if actual_bbox is not None:
            cv2.rectangle(overlay, (actual_bbox["min_x"], actual_bbox["min_y"]), (actual_bbox["max_x"], actual_bbox["max_y"]), (0, 255, 0), 2, cv2.LINE_AA)
        if actual_centroid is not None:
            cx = int(round(actual_centroid[0])) % width
            cy = int(round(actual_centroid[1]))
            cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(overlay, f"Actual {axis_name}", (min(width - 220, cx + 8), min(height - 10, cy + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        overlay_path = overlays_dir / f"erp_axis_{axis_name}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

        report = {
            "axis_name": axis_name,
            "predicted_pixel_xy": [float(pred_x), float(pred_y)],
            "predicted_longitude_deg": pred_lon,
            "actual_bbox": actual_bbox,
            "actual_centroid": actual_centroid,
            "actual_detected_area_px": actual_area,
            "actual_longitude_deg": actual_lon,
            "renderer_yaw_delta_deg": yaw_delta,
            "overlay_png": str(overlay_path),
            "png_path": str(entry["png_path"]),
            "analysis_method": "median_background_subtraction",
        }
        report_path = metadata_dir / f"erp_axis_{axis_name}_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_paths.append(str(report_path))
        if axis_name in horizontal_axis_names and yaw_delta is not None:
            horizontal_deltas.append({"axis_name": axis_name, "renderer_yaw_delta_deg": float(yaw_delta)})

    validated = False
    derived_offset = None
    circular_std_deg = None
    if horizontal_deltas:
        angles_rad = np.deg2rad(np.array([row["renderer_yaw_delta_deg"] for row in horizontal_deltas], dtype=np.float64))
        mean_sin = float(np.mean(np.sin(angles_rad)))
        mean_cos = float(np.mean(np.cos(angles_rad)))
        derived_offset = _wrap_angle_deg(math.degrees(math.atan2(mean_sin, mean_cos)))
        resultant = math.sqrt(mean_sin * mean_sin + mean_cos * mean_cos)
        if resultant > 1e-12:
            circular_std_deg = float(np.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant)))))
        validated = (len(horizontal_deltas) >= 4) and (abs(_wrap_angle_deg(derived_offset - 90.0)) <= 10.0) and (circular_std_deg is not None and circular_std_deg <= 15.0)

    convention = {
        "validated": bool(validated),
        "erp_yaw_zero_offset_deg": None if derived_offset is None else float(derived_offset),
        "expected_confirmed_value_deg": 90.0,
        "tolerance_deg": 10.0,
        "circular_std_deg": circular_std_deg,
        "horizontal_axes_used": horizontal_deltas,
        "source": "synthetic_axis_marker_calibration",
        "analysis_method": "median_background_subtraction",
        "notes": [
            "The six axis renders are compared against their per-pixel median image to suppress static scene content and isolate the moving calibration sphere.",
            "The derived ERP yaw-zero offset is the circular mean of actual-minus-predicted longitudes for +X, -X, +Y, -Y.",
        ],
    }
    convention_path = metadata_dir / "erp_projection_convention.json"
    convention_path.write_text(json.dumps(convention, indent=2), encoding="utf-8")

    summary = {
        "reports": report_paths,
        "erp_projection_convention_json": str(convention_path),
        "erp_projection_convention": convention,
    }
    summary_path = Path(summary_path) if summary_path else (metadata_dir / "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _run_unreal_capture():
    import bedlam360_mini_validation as mini

    mini = importlib.reload(mini)
    import unreal

    output_root = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_erp_axes_debug")
    images_dir = output_root / "images"
    metadata_dir = output_root / "metadata"
    overlays_dir = output_root / "overlays"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    camera_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    radius_cm = 300.0
    axes = [
        ("plus_x", "+X", (radius_cm, 0.0, 0.0)),
        ("minus_x", "-X", (-radius_cm, 0.0, 0.0)),
        ("plus_y", "+Y", (0.0, radius_cm, 0.0)),
        ("minus_y", "-Y", (0.0, -radius_cm, 0.0)),
        ("plus_z", "+Z", (0.0, 0.0, radius_cm)),
        ("minus_z", "-Z", (0.0, 0.0, -radius_cm)),
    ]
    prefix = "BEDLAM360_ERPAXIS_"

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in actor_subsystem.get_all_level_actors():
        try:
            label = actor.get_actor_label()
        except Exception:
            continue
        if str(label).startswith(prefix):
            try:
                actor_subsystem.destroy_actor(actor)
            except Exception:
                pass

    camera_actor = mini.capture_scene_cube.find_scene_capture_cube(mini.DEFAULT_ACTOR_LABEL)
    component = mini.capture_scene_cube.get_capture_component(camera_actor)
    texture_target = mini.capture_scene_cube.get_texture_target(component)
    export_lib = mini.unreal.BEDLAM360ExportLibrary
    mini.capture_scene_cube.set_actor_pose(
        camera_actor,
        camera_pose["x"],
        camera_pose["y"],
        camera_pose["z"],
        camera_pose["pitch"],
        camera_pose["yaw"],
        camera_pose["roll"],
    )

    sphere_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Sphere.Sphere")
    if sphere_mesh is None:
        raise RuntimeError("Could not load /Engine/BasicShapes/Sphere.Sphere")

    all_reports = []
    for axis_name, axis_label, offset in axes:
        for actor in actor_subsystem.get_all_level_actors():
            try:
                label = actor.get_actor_label()
            except Exception:
                continue
            if str(label).startswith(prefix):
                try:
                    actor_subsystem.destroy_actor(actor)
                except Exception:
                    pass

        location = unreal.Vector(offset[0], offset[1], offset[2])
        sphere_actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor, location, unreal.Rotator(0.0, 0.0, 0.0))
        sphere_actor.set_actor_label(f"{prefix}{axis_name}")
        mesh_component = sphere_actor.static_mesh_component
        mesh_component.set_editor_property("static_mesh", sphere_mesh)
        sphere_actor.set_actor_scale3d(unreal.Vector(1.5, 1.5, 1.5))

        stem = f"erp_axis_{axis_name}"
        hdr_path = images_dir / f"{stem}.hdr"
        exr_path = images_dir / f"{stem}.exr"
        png_path = images_dir / f"{stem}.png"
        capture_result = mini.stabilized_capture_and_export(
            actor=camera_actor,
            component=component,
            texture_target=texture_target,
            export_lib=export_lib,
            frame_name=stem,
            hdr_path=hdr_path,
            exr_path=exr_path,
            faces_dir=None,
        )

        predicted = _predicted_marker_pixel(camera_pose, [offset[0], offset[1], offset[2]], width=2048, height=1024)
        metadata_path = metadata_dir / f"{stem}.json"
        metadata = {
            "axis_name": axis_name,
            "axis_label": axis_label,
            "world_location_cm": {"x": offset[0], "y": offset[1], "z": offset[2]},
            "camera_pose_cm_deg": camera_pose,
            "predicted": predicted,
            "capture_result": capture_result,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        preview_status = mini._run_preview_frame(exr_path, png_path, metadata_path, overlay=False)
        if not preview_status.get("success"):
            raise RuntimeError(f"PNG generation failed for {axis_name}: {preview_status}")

        report_path = metadata_dir / f"{stem}_report.json"
        overlay_path = overlays_dir / f"{stem}_overlay.png"
        command = [
            "python3",
            str(Path(__file__).resolve()),
            "--postprocess",
            "--png-path",
            str(png_path),
            "--report-path",
            str(report_path),
            "--overlay-path",
            str(overlay_path),
            "--axis-name",
            axis_name,
            "--pred-x",
            str(predicted["predicted_pixel_xy"][0]),
            "--pred-y",
            str(predicted["predicted_pixel_xy"][1]),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.stdout:
            unreal.log(f"[BEDLAM360] ERP axis post stdout ({axis_name}):\n{completed.stdout}")
        if completed.stderr:
            unreal.log_warning(f"[BEDLAM360] ERP axis post stderr ({axis_name}):\n{completed.stderr}")
        if completed.returncode != 0:
            raise RuntimeError(f"Postprocess failed for {axis_name} with code {completed.returncode}")
        all_reports.append(str(report_path))

    summary_path = metadata_dir / "summary.json"
    summary_path.write_text(json.dumps({"reports": all_reports}, indent=2), encoding="utf-8")
    print(f"[BEDLAM360] ERP axes debug summary: {summary_path}")


def main():
    args = _build_arg_parser().parse_args()
    if args.postprocess:
        _run_postprocess(
            png_path=Path(args.png_path),
            report_path=Path(args.report_path),
            overlay_path=Path(args.overlay_path),
            axis_name=args.axis_name,
            pred_x=float(args.pred_x),
            pred_y=float(args.pred_y),
        )
    elif args.analyze_root:
        _analyze_existing_root(root_path=Path(args.analyze_root), summary_path=args.summary_path)
    else:
        _run_unreal_capture()


if __name__ == "__main__":
    main()
