import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import bedlam360_gt_erp_alignment as gt_alignment  # type: ignore
import bedlam360_benchmark_export as benchmark_export  # type: ignore


CALIBRATED_ERP_YAW_ZERO_OFFSET_DEG = 90.22704672681925
DEFAULT_VARIANT_OFFSETS = (
    CALIBRATED_ERP_YAW_ZERO_OFFSET_DEG,
    -CALIBRATED_ERP_YAW_ZERO_OFFSET_DEG,
    0.0,
    180.0,
)
FORWARD_AXIS_OPTIONS = ("+X", "-X", "+Y", "-Y")
UP_AXIS_OPTIONS = ("+Z", "-Z")


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    path = Path(path)
    _ensure_dir(path.parent)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return path


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _sha1_file(path):
    path = Path(path)
    digest = hashlib.sha1()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_report(path):
    path = Path(path)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "sha1": None,
            "mtime_utc": None,
            "size_bytes": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "sha1": _sha1_file(path),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": int(stat.st_size),
    }


def _sequence_manifest_path(run_root, sequence_id):
    return Path(run_root) / "raw" / "sequences" / sequence_id / "manifest.json"


def _load_sequence_context(run_root, sequence_id):
    render_manifest = _read_json(Path(run_root) / "render_manifest.json")
    sequence_manifest = _read_json(_sequence_manifest_path(run_root, sequence_id))
    return render_manifest, sequence_manifest


def _sequence_specs_by_id(render_manifest):
    return {item["sequence_id"]: item for item in render_manifest.get("sequence_specs", [])}


def _find_sequence_frame_record(sequence_manifest, sequence_local_frame_index):
    frame_records = sequence_manifest["range_result"]["frame_records"]
    if sequence_local_frame_index < 0 or sequence_local_frame_index >= len(frame_records):
        raise RuntimeError(
            f"sequence_local_frame_index={sequence_local_frame_index} out of range for {sequence_manifest['sequence_id']}"
        )
    return frame_records[sequence_local_frame_index]


def _load_global_frame_context(run_root, sequence_id, sequence_local_frame_index):
    raw_root = Path(run_root) / "raw"
    frames = _read_json(raw_root / "metadata" / "frames.json")
    frame_mapping = _read_json(raw_root / "metadata" / "frame_mapping.json")
    for global_frame_index, item in enumerate(frames):
        if item["sequence_id"] == sequence_id and int(item["sequence_local_frame_index"]) == int(sequence_local_frame_index):
            return global_frame_index, item, frame_mapping[global_frame_index]
    raise RuntimeError(
        f"Could not find global frame for sequence_id={sequence_id} sequence_local_frame_index={sequence_local_frame_index}"
    )


def _camera_pose_from_matrix(camera_to_world):
    mat = np.asarray(camera_to_world, dtype=np.float64)
    x = float(mat[0, 3])
    y = float(mat[1, 3])
    z = float(mat[2, 3])
    yaw = math.degrees(math.atan2(mat[1, 0], mat[0, 0]))
    pitch = math.degrees(math.atan2(-mat[2, 0], math.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2)))
    roll = math.degrees(math.atan2(mat[2, 1], mat[2, 2]))
    return {
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }


def _camera_world_from_pose_variant(camera_pose, pitch_sign=1.0, roll_sign=1.0, rotation_order="zyx"):
    x = float(camera_pose["x"])
    y = float(camera_pose["y"])
    z = float(camera_pose["z"])
    yaw_deg = float(camera_pose["yaw"])
    pitch_deg = float(camera_pose["pitch"]) * float(pitch_sign)
    roll_deg = float(camera_pose["roll"]) * float(roll_sign)

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)

    if rotation_order == "zyx":
        rot = rz @ ry @ rx
    elif rotation_order == "zxy":
        rot = rz @ rx @ ry
    elif rotation_order == "yxz":
        rot = ry @ rx @ rz
    else:
        raise ValueError(f"Unsupported rotation_order: {rotation_order}")

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rot
    matrix[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return matrix


def _translation_only_camera_world(camera_to_world):
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray(camera_to_world, dtype=np.float32)[:3, 3]
    return matrix


def _matrix_diff_report(matrix_a, matrix_b):
    a = np.asarray(matrix_a, dtype=np.float64)
    b = np.asarray(matrix_b, dtype=np.float64)
    diff = a - b
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "frobenius_norm": float(np.linalg.norm(diff)),
    }


def _axis_vector(label):
    mapping = {
        "+X": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "+Y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "+Z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    return mapping[label]


def _camera_axis_basis(forward_axis="+X", up_axis="+Z"):
    forward = _axis_vector(forward_axis)
    up_seed = _axis_vector(up_axis)
    right = np.cross(up_seed, forward)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        raise ValueError(f"Invalid forward/up combination: {forward_axis}, {up_axis}")
    right = right / right_norm
    up = np.cross(forward, right)
    up = up / max(1e-6, float(np.linalg.norm(up)))
    basis = np.stack([forward, right, up], axis=0).astype(np.float32)
    return {
        "forward": forward.tolist(),
        "right": right.tolist(),
        "up": up.tolist(),
        "matrix": basis,
    }


def _remap_camera_frame_points(points_cam_xyz, forward_axis="+X", up_axis="+Z"):
    basis = _camera_axis_basis(forward_axis=forward_axis, up_axis=up_axis)
    remapped = (basis["matrix"] @ np.asarray(points_cam_xyz, dtype=np.float32).T).T
    return remapped, basis


def _make_tiled_contact_sheet(image_paths, labels, output_path, cols=4, tile_margin=8):
    images = [gt_alignment._load_image(path) for path in image_paths]  # pylint: disable=protected-access
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    rows = int(math.ceil(float(len(images)) / float(cols)))
    canvas = np.zeros(
        (
            rows * height + max(0, rows - 1) * tile_margin,
            cols * width + max(0, cols - 1) * tile_margin,
            3,
        ),
        dtype=np.uint8,
    )
    for idx, (image, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        y0 = row * (height + tile_margin)
        x0 = col * (width + tile_margin)
        tile = np.zeros((height, width, 3), dtype=np.uint8)
        tile[: image.shape[0], : image.shape[1]] = image
        cv2.putText(tile, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        canvas[y0 : y0 + height, x0 : x0 + width] = tile
    cv2.imwrite(str(output_path), canvas)
    return output_path


def _draw_bbox(image, bbox, color):
    if bbox is None:
        return image
    cv2.rectangle(
        image,
        (int(round(bbox["min_x"])), int(round(bbox["min_y"]))),
        (int(round(bbox["max_x"])), int(round(bbox["max_y"]))),
        color,
        1,
        cv2.LINE_AA,
    )
    return image


def _overlay_variant(
    dataset_root,
    frame_i,
    frame_meta,
    body_map,
    joints3d,
    joints_available,
    camera_inv,
    body_world,
    width,
    height,
    yaw_offset_deg,
    output_path,
):
    image = gt_alignment._load_image(dataset_root / "images" / frame_meta["image_png"])  # pylint: disable=protected-access
    rows = []
    for body_i in range(min(int(joints3d.shape[1]), int(len(body_map)))):
        asset_id = body_map[body_i].get("asset_id", f"body_{body_i}")
        if not joints_available[frame_i, body_i]:
            rows.append(
                {
                    "body_index": int(body_i),
                    "asset_id": asset_id,
                    "projected_joint_count": 0,
                    "bbox": None,
                    "suspected_coordinate_mismatch": ["joints_unavailable"],
                }
            )
            continue
        joints_local_m = joints3d[frame_i, body_i]
        joints_unreal_local_cm = (gt_alignment.SMPL_TO_UNREAL_LOCAL_CM @ joints_local_m.T).T
        joints_world_cm = gt_alignment._transform_points(joints_unreal_local_cm, body_world[frame_i, body_i])  # pylint: disable=protected-access
        joints_camera_cm = gt_alignment._transform_points(joints_world_cm, camera_inv)  # pylint: disable=protected-access
        joints_2d, valid_mask = gt_alignment._project_camera_xyz_to_erp_with_yaw_offset(  # pylint: disable=protected-access
            joints_camera_cm,
            width=width,
            height=height,
            yaw_offset_deg=yaw_offset_deg,
        )
        bbox = gt_alignment._bbox_from_points(joints_2d, valid_mask)  # pylint: disable=protected-access
        reasons = gt_alignment._suspected_mismatch(int(valid_mask.sum()), bbox, width, height)  # pylint: disable=protected-access
        color = gt_alignment.BODY_COLORS_BGR[body_i % len(gt_alignment.BODY_COLORS_BGR)]
        image = gt_alignment._draw_body_joints(image, joints_2d, valid_mask, color, asset_id)  # pylint: disable=protected-access
        image = _draw_bbox(image, bbox, color)
        rows.append(
            {
                "body_index": int(body_i),
                "asset_id": asset_id,
                "projected_joint_count": int(valid_mask.sum()),
                "bbox": bbox,
                "suspected_coordinate_mismatch": reasons,
            }
        )
    cv2.putText(
        image,
        f"yaw_offset={yaw_offset_deg:+.3f} | timeline_frame={frame_meta['timeline_frame_index']}",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), image)
    return rows


def _overlay_variant_with_camera_inv(
    dataset_root,
    frame_i,
    frame_meta,
    body_map,
    joints3d,
    joints_available,
    camera_inv,
    body_world,
    width,
    height,
    yaw_offset_deg,
    output_path,
):
    image = gt_alignment._load_image(dataset_root / "images" / frame_meta["image_png"])  # pylint: disable=protected-access
    rows = []
    for body_i in range(min(int(joints3d.shape[1]), int(len(body_map)))):
        asset_id = body_map[body_i].get("asset_id", f"body_{body_i}")
        if not joints_available[frame_i, body_i]:
            rows.append(
                {
                    "body_index": int(body_i),
                    "asset_id": asset_id,
                    "projected_joint_count": 0,
                    "bbox": None,
                    "suspected_coordinate_mismatch": ["joints_unavailable"],
                }
            )
            continue
        joints_local_m = joints3d[frame_i, body_i]
        joints_unreal_local_cm = (gt_alignment.SMPL_TO_UNREAL_LOCAL_CM @ joints_local_m.T).T
        joints_world_cm = gt_alignment._transform_points(joints_unreal_local_cm, body_world[frame_i, body_i])  # pylint: disable=protected-access
        joints_camera_cm = gt_alignment._transform_points(joints_world_cm, camera_inv)  # pylint: disable=protected-access
        joints_2d, valid_mask = gt_alignment._project_camera_xyz_to_erp_with_yaw_offset(  # pylint: disable=protected-access
            joints_camera_cm,
            width=width,
            height=height,
            yaw_offset_deg=yaw_offset_deg,
        )
        bbox = gt_alignment._bbox_from_points(joints_2d, valid_mask)  # pylint: disable=protected-access
        reasons = gt_alignment._suspected_mismatch(int(valid_mask.sum()), bbox, width, height)  # pylint: disable=protected-access
        color = gt_alignment.BODY_COLORS_BGR[body_i % len(gt_alignment.BODY_COLORS_BGR)]
        image = gt_alignment._draw_body_joints(image, joints_2d, valid_mask, color, asset_id)  # pylint: disable=protected-access
        image = _draw_bbox(image, bbox, color)
        rows.append(
            {
                "body_index": int(body_i),
                "asset_id": asset_id,
                "projected_joint_count": int(valid_mask.sum()),
                "bbox": bbox,
                "suspected_coordinate_mismatch": reasons,
            }
        )
    cv2.imwrite(str(output_path), image)
    return rows


def _overlay_variant_with_camera_model(
    dataset_root,
    frame_i,
    frame_meta,
    body_map,
    joints3d,
    joints_available,
    camera_to_world,
    body_world,
    width,
    height,
    yaw_offset_deg,
    output_path,
    forward_axis="+X",
    up_axis="+Z",
):
    image = gt_alignment._load_image(dataset_root / "images" / frame_meta["image_png"])  # pylint: disable=protected-access
    camera_inv = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float32)).astype(np.float32)
    rows = []
    for body_i in range(min(int(joints3d.shape[1]), int(len(body_map)))):
        asset_id = body_map[body_i].get("asset_id", f"body_{body_i}")
        if not joints_available[frame_i, body_i]:
            rows.append(
                {
                    "body_index": int(body_i),
                    "asset_id": asset_id,
                    "projected_joint_count": 0,
                    "bbox": None,
                    "suspected_coordinate_mismatch": ["joints_unavailable"],
                }
            )
            continue
        joints_local_m = joints3d[frame_i, body_i]
        joints_unreal_local_cm = (gt_alignment.SMPL_TO_UNREAL_LOCAL_CM @ joints_local_m.T).T
        joints_world_cm = gt_alignment._transform_points(joints_unreal_local_cm, body_world[frame_i, body_i])  # pylint: disable=protected-access
        joints_camera_cm = gt_alignment._transform_points(joints_world_cm, camera_inv)  # pylint: disable=protected-access
        joints_camera_cm, basis = _remap_camera_frame_points(
            joints_camera_cm,
            forward_axis=forward_axis,
            up_axis=up_axis,
        )
        joints_2d, valid_mask = gt_alignment._project_camera_xyz_to_erp_with_yaw_offset(  # pylint: disable=protected-access
            joints_camera_cm,
            width=width,
            height=height,
            yaw_offset_deg=yaw_offset_deg,
        )
        bbox = gt_alignment._bbox_from_points(joints_2d, valid_mask)  # pylint: disable=protected-access
        reasons = gt_alignment._suspected_mismatch(int(valid_mask.sum()), bbox, width, height)  # pylint: disable=protected-access
        color = gt_alignment.BODY_COLORS_BGR[body_i % len(gt_alignment.BODY_COLORS_BGR)]
        image = gt_alignment._draw_body_joints(image, joints_2d, valid_mask, color, asset_id)  # pylint: disable=protected-access
        image = _draw_bbox(image, bbox, color)
        rows.append(
            {
                "body_index": int(body_i),
                "asset_id": asset_id,
                "projected_joint_count": int(valid_mask.sum()),
                "bbox": bbox,
                "suspected_coordinate_mismatch": reasons,
                "basis": basis,
            }
        )
    cv2.imwrite(str(output_path), image)
    return rows


def _synthetic_camera_frame_markers(camera_to_world, width, height, yaw_offset_deg):
    rotation = np.asarray(camera_to_world, dtype=np.float32)[:3, :3]
    camera_origin = np.asarray(camera_to_world, dtype=np.float32)[:3, 3]
    markers = {
        "front": np.array([100.0, 0.0, 0.0], dtype=np.float32),
        "right": np.array([0.0, 100.0, 0.0], dtype=np.float32),
        "back": np.array([-100.0, 0.0, 0.0], dtype=np.float32),
        "left": np.array([0.0, -100.0, 0.0], dtype=np.float32),
        "up": np.array([0.0, 0.0, 100.0], dtype=np.float32),
        "down": np.array([0.0, 0.0, -100.0], dtype=np.float32),
    }
    rows = []
    for label, cam_vec in markers.items():
        world_point = camera_origin + (rotation @ cam_vec)
        camera_point = cam_vec.reshape(1, 3)
        pts2d, valid = gt_alignment._project_camera_xyz_to_erp_with_yaw_offset(  # pylint: disable=protected-access
            camera_point,
            width=width,
            height=height,
            yaw_offset_deg=yaw_offset_deg,
        )
        rows.append(
            {
                "label": label,
                "camera_frame_point_cm": cam_vec.tolist(),
                "world_point_cm": world_point.tolist(),
                "predicted_erp_xy": None if not bool(valid[0]) else [float(pts2d[0][0]), float(pts2d[0][1])],
            }
        )
    return rows


def _synthetic_candidate_markers(width, height, yaw_offset_deg, forward_axis="+X", up_axis="+Z"):
    basis = _camera_axis_basis(forward_axis=forward_axis, up_axis=up_axis)
    markers = {
        "front": np.asarray(basis["forward"], dtype=np.float32) * 100.0,
        "right": np.asarray(basis["right"], dtype=np.float32) * 100.0,
        "up": np.asarray(basis["up"], dtype=np.float32) * 100.0,
    }
    rows = []
    for label, cam_vec in markers.items():
        pts2d, valid = gt_alignment._project_camera_xyz_to_erp_with_yaw_offset(  # pylint: disable=protected-access
            cam_vec.reshape(1, 3),
            width=width,
            height=height,
            yaw_offset_deg=yaw_offset_deg,
        )
        rows.append(
            {
                "label": label,
                "camera_frame_point_cm": cam_vec.tolist(),
                "predicted_erp_xy": None if not bool(valid[0]) else [float(pts2d[0][0]), float(pts2d[0][1])],
            }
        )
    return rows


def _make_marker_overlay(image_path, markers, output_path):
    image = gt_alignment._load_image(image_path)  # pylint: disable=protected-access
    colors = {
        "front": (0, 255, 0),
        "right": (255, 0, 0),
        "back": (0, 0, 255),
        "left": (255, 255, 0),
        "up": (255, 0, 255),
        "down": (0, 255, 255),
    }
    for item in markers:
        pt = item.get("predicted_erp_xy")
        if pt is None:
            continue
        image = gt_alignment._draw_direction_marker(  # pylint: disable=protected-access
            image,
            pt,
            item["label"],
            colors.get(item["label"], (255, 255, 255)),
        )
    cv2.imwrite(str(output_path), image)
    return output_path


def audit_v1_frame(run_root, sequence_id, frame_index, focused_translation_only=False):
    run_root = Path(run_root)
    dataset_root = run_root / "raw"
    render_manifest, sequence_manifest = _load_sequence_context(run_root, sequence_id)
    frame_record = _find_sequence_frame_record(sequence_manifest, int(frame_index))
    pose_json = _read_json(frame_record["pose_json_path"])
    global_frame_index, frame_meta, global_mapping = _load_global_frame_context(run_root, sequence_id, int(frame_index))

    image_path = dataset_root / "images" / frame_meta["image_png"]
    image = gt_alignment._load_image(image_path)  # pylint: disable=protected-access
    height, width = image.shape[:2]
    joints_npz = np.load(dataset_root / "joints3d" / "joints3d.npz")
    joints3d = joints_npz["joints3d"]
    joints_available = joints_npz["available"]
    camera_world = np.load(dataset_root / "metadata" / "camera_world_transforms.npy")
    body_world = np.load(dataset_root / "metadata" / "body_world_transforms.npy")
    camera_to_world = np.asarray(camera_world[global_frame_index], dtype=np.float32)
    world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
    convention = gt_alignment._load_erp_projection_convention(dataset_root)  # pylint: disable=protected-access

    audit_root = run_root / "projection_audit" / sequence_id / f"frame_{int(frame_index):04d}"
    if audit_root.exists():
        shutil.rmtree(audit_root)
    _ensure_dir(audit_root)
    variants_dir = _ensure_dir(audit_root / "variants")

    overlay_dir = dataset_root / "projections2d" / "erp_alignment" / "overlays"
    current_overlay_name = f"{Path(frame_meta['image_png']).stem}_joints_overlay.png"
    current_overlay_path = overlay_dir / current_overlay_name
    current_overlay_before = _file_report(current_overlay_path)

    variant_rows = []
    variant_paths = []
    variant_labels = []
    for yaw_offset_deg in DEFAULT_VARIANT_OFFSETS:
        label = f"yaw_{yaw_offset_deg:+.3f}".replace("+", "p").replace("-", "m")
        overlay_path = variants_dir / f"{label}.png"
        rows = _overlay_variant(
            dataset_root=dataset_root,
            frame_i=global_frame_index,
            frame_meta=frame_meta,
            body_map=global_mapping["body_frame_mapping"],
            joints3d=joints3d,
            joints_available=joints_available,
            camera_inv=world_to_camera,
            body_world=body_world,
            width=width,
            height=height,
            yaw_offset_deg=float(yaw_offset_deg),
            output_path=overlay_path,
        )
        variant_rows.append(
            {
                "erp_yaw_zero_offset_deg": float(yaw_offset_deg),
                "overlay": _file_report(overlay_path),
                "rows": rows,
            }
        )
        variant_paths.append(overlay_path)
        variant_labels.append(f"{yaw_offset_deg:+.3f} deg")

    contact_sheet_path = audit_root / "projection_variants_contact_sheet.png"
    gt_alignment._make_contact_sheet(variant_paths, variant_labels, contact_sheet_path)  # pylint: disable=protected-access

    current_overlay_regenerated_path = audit_root / "regenerated_current_convention_overlay.png"
    if current_overlay_path.exists():
        current_overlay_path.unlink()
    regenerated_current_rows = _overlay_variant(
        dataset_root=dataset_root,
        frame_i=global_frame_index,
        frame_meta=frame_meta,
        body_map=global_mapping["body_frame_mapping"],
        joints3d=joints3d,
        joints_available=joints_available,
        camera_inv=world_to_camera,
        body_world=body_world,
        width=width,
        height=height,
        yaw_offset_deg=float(convention["erp_yaw_zero_offset_deg"]),
        output_path=current_overlay_path,
    )
    shutil.copy2(current_overlay_path, current_overlay_regenerated_path)

    synthetic_markers = _synthetic_camera_frame_markers(
        camera_to_world=camera_to_world,
        width=width,
        height=height,
        yaw_offset_deg=float(convention["erp_yaw_zero_offset_deg"]),
    )
    synthetic_overlay_path = audit_root / "synthetic_markers_overlay.png"
    _make_marker_overlay(image_path, synthetic_markers, synthetic_overlay_path)

    translation_only_dir = _ensure_dir(audit_root / "translation_only_variants")
    translation_only_camera_to_world = _translation_only_camera_world(camera_to_world)
    translation_only_camera_inv = np.linalg.inv(translation_only_camera_to_world).astype(np.float32)
    full_rotation_camera_inv = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float32)).astype(np.float32)
    translation_only_defs = [
        {
            "name": "full_rotation__yaw_calibrated",
            "camera_to_world": camera_to_world,
            "camera_inv": full_rotation_camera_inv,
            "yaw_offset_deg": float(convention["erp_yaw_zero_offset_deg"]),
            "note": "Current GT path: camera translation plus camera rotation plus calibrated ERP yaw offset.",
        },
        {
            "name": "translation_only__yaw_calibrated",
            "camera_to_world": translation_only_camera_to_world,
            "camera_inv": translation_only_camera_inv,
            "yaw_offset_deg": float(convention["erp_yaw_zero_offset_deg"]),
            "note": "Hypothesis test: ignore camera yaw/pitch/roll entirely, keep only translation plus calibrated ERP yaw offset.",
        },
        {
            "name": "translation_only__yaw_zero",
            "camera_to_world": translation_only_camera_to_world,
            "camera_inv": translation_only_camera_inv,
            "yaw_offset_deg": 0.0,
            "note": "Hypothesis test: translation only with zero ERP yaw offset.",
        },
        {
            "name": "translation_only__yaw_p180",
            "camera_to_world": translation_only_camera_to_world,
            "camera_inv": translation_only_camera_inv,
            "yaw_offset_deg": 180.0,
            "note": "Hypothesis test: translation only with 180-degree ERP yaw offset.",
        },
        {
            "name": "translation_only__yaw_p90_227",
            "camera_to_world": translation_only_camera_to_world,
            "camera_inv": translation_only_camera_inv,
            "yaw_offset_deg": CALIBRATED_ERP_YAW_ZERO_OFFSET_DEG,
            "note": "Same as translation_only__yaw_calibrated, included with explicit BEDLAM360 calibration label for side-by-side inspection.",
        },
    ]
    translation_only_rows = []
    translation_only_paths = []
    translation_only_labels = []
    for variant in translation_only_defs:
        overlay_path = translation_only_dir / f"{variant['name']}.png"
        rows = _overlay_variant_with_camera_inv(
            dataset_root=dataset_root,
            frame_i=global_frame_index,
            frame_meta=frame_meta,
            body_map=global_mapping["body_frame_mapping"],
            joints3d=joints3d,
            joints_available=joints_available,
            camera_inv=variant["camera_inv"],
            body_world=body_world,
            width=width,
            height=height,
            yaw_offset_deg=float(variant["yaw_offset_deg"]),
            output_path=overlay_path,
        )
        translation_only_rows.append(
            {
                "name": variant["name"],
                "note": variant["note"],
                "yaw_offset_deg_used": float(variant["yaw_offset_deg"]),
                "camera_rotation_applied": bool(variant["name"].startswith("full_rotation")),
                "camera_to_world_matrix": np.asarray(variant["camera_to_world"], dtype=np.float32).tolist(),
                "world_to_camera_matrix": np.asarray(variant["camera_inv"], dtype=np.float32).tolist(),
                "matrix_diff_vs_dataset_camera_world": _matrix_diff_report(variant["camera_to_world"], camera_to_world),
                "overlay": _file_report(overlay_path),
                "rows": rows,
            }
        )
        translation_only_paths.append(overlay_path)
        translation_only_labels.append(variant["name"])
    translation_only_contact_sheet_path = audit_root / "translation_only_vs_full_rotation_contact_sheet.png"
    gt_alignment._make_contact_sheet(translation_only_paths, translation_only_labels, translation_only_contact_sheet_path)  # pylint: disable=protected-access

    pitch_variant_rows = []
    pitch_contact_sheet_path = None
    camera_model_rows = []
    grouped_variant_sheets = []
    selected_yaw_for_pitch_audit = 180.0
    if not focused_translation_only:
        pitch_variants_dir = _ensure_dir(audit_root / "pitch_variants")
        pitch_variant_defs = [
            {
                "name": "saved_camera_world",
                "camera_to_world": camera_to_world,
                "note": "Uses stored camera_world matrix from dataset export.",
            },
            {
                "name": "metadata_rebuilt_zyx",
                "camera_to_world": _camera_world_from_pose_variant(pose_json["camera_pose_cm_deg"], pitch_sign=1.0, roll_sign=1.0, rotation_order="zyx"),
                "note": "Rebuilt from metadata using current yaw(Z)-pitch(Y)-roll(X) order.",
            },
            {
                "name": "metadata_pitch_negated_zyx",
                "camera_to_world": _camera_world_from_pose_variant(pose_json["camera_pose_cm_deg"], pitch_sign=-1.0, roll_sign=1.0, rotation_order="zyx"),
                "note": "Rebuilt from metadata with pitch sign flipped.",
            },
            {
                "name": "metadata_roll_negated_zyx",
                "camera_to_world": _camera_world_from_pose_variant(pose_json["camera_pose_cm_deg"], pitch_sign=1.0, roll_sign=-1.0, rotation_order="zyx"),
                "note": "Rebuilt from metadata with roll sign flipped.",
            },
            {
                "name": "metadata_rebuilt_zxy",
                "camera_to_world": _camera_world_from_pose_variant(pose_json["camera_pose_cm_deg"], pitch_sign=1.0, roll_sign=1.0, rotation_order="zxy"),
                "note": "Rebuilt from metadata with yaw-roll-pitch multiplication order.",
            },
        ]
        pitch_variant_paths = []
        pitch_variant_labels = []
        for variant in pitch_variant_defs:
            variant_path = pitch_variants_dir / f"{variant['name']}.png"
            variant_camera_inv = np.linalg.inv(np.asarray(variant["camera_to_world"], dtype=np.float32)).astype(np.float32)
            rows = _overlay_variant_with_camera_inv(
                dataset_root=dataset_root,
                frame_i=global_frame_index,
                frame_meta=frame_meta,
                body_map=global_mapping["body_frame_mapping"],
                joints3d=joints3d,
                joints_available=joints_available,
                camera_inv=variant_camera_inv,
                body_world=body_world,
                width=width,
                height=height,
                yaw_offset_deg=selected_yaw_for_pitch_audit,
                output_path=variant_path,
            )
            pitch_variant_rows.append(
                {
                    "name": variant["name"],
                    "note": variant["note"],
                    "camera_to_world_matrix": np.asarray(variant["camera_to_world"], dtype=np.float32).tolist(),
                    "world_to_camera_matrix": variant_camera_inv.tolist(),
                    "matrix_diff_vs_dataset_camera_world": _matrix_diff_report(variant["camera_to_world"], camera_to_world),
                    "overlay": _file_report(variant_path),
                    "rows": rows,
                    "yaw_offset_deg_used": float(selected_yaw_for_pitch_audit),
                }
            )
            pitch_variant_paths.append(variant_path)
            pitch_variant_labels.append(variant["name"])
        pitch_contact_sheet_path = audit_root / "pitch_variants_contact_sheet.png"
        gt_alignment._make_contact_sheet(pitch_variant_paths, pitch_variant_labels, pitch_contact_sheet_path)  # pylint: disable=protected-access

        camera_model_variants_dir = _ensure_dir(audit_root / "camera_model_variants")
        camera_model_marker_dir = _ensure_dir(audit_root / "camera_model_marker_variants")
        for yaw_offset_deg in DEFAULT_VARIANT_OFFSETS:
            yaw_tag = f"yaw_{yaw_offset_deg:+.3f}".replace("+", "p").replace("-", "m")
            sheet_paths = []
            sheet_labels = []
            marker_sheet_paths = []
            marker_sheet_labels = []
            for pitch_sign in (1.0, -1.0):
                for roll_sign in (1.0, -1.0):
                    candidate_camera_to_world = _camera_world_from_pose_variant(
                        pose_json["camera_pose_cm_deg"],
                        pitch_sign=pitch_sign,
                        roll_sign=roll_sign,
                        rotation_order="zyx",
                    )
                    for forward_axis in FORWARD_AXIS_OPTIONS:
                        for up_axis in UP_AXIS_OPTIONS:
                            candidate_name = (
                                f"{yaw_tag}__pitch_{'p' if pitch_sign > 0 else 'm'}__roll_{'p' if roll_sign > 0 else 'm'}"
                                f"__fwd_{forward_axis.replace('+','p').replace('-','m')}__up_{up_axis.replace('+','p').replace('-','m')}"
                            )
                            overlay_path = camera_model_variants_dir / f"{candidate_name}.png"
                            rows = _overlay_variant_with_camera_model(
                                dataset_root=dataset_root,
                                frame_i=global_frame_index,
                                frame_meta=frame_meta,
                                body_map=global_mapping["body_frame_mapping"],
                                joints3d=joints3d,
                                joints_available=joints_available,
                                camera_to_world=candidate_camera_to_world,
                                body_world=body_world,
                                width=width,
                                height=height,
                                yaw_offset_deg=float(yaw_offset_deg),
                                output_path=overlay_path,
                                forward_axis=forward_axis,
                                up_axis=up_axis,
                            )
                            marker_rows = _synthetic_candidate_markers(
                                width=width,
                                height=height,
                                yaw_offset_deg=float(yaw_offset_deg),
                                forward_axis=forward_axis,
                                up_axis=up_axis,
                            )
                            marker_overlay_path = camera_model_marker_dir / f"{candidate_name}.png"
                            _make_marker_overlay(image_path, marker_rows, marker_overlay_path)
                            camera_model_rows.append(
                                {
                                    "candidate_name": candidate_name,
                                    "yaw_offset_deg": float(yaw_offset_deg),
                                    "pitch_sign": float(pitch_sign),
                                    "roll_sign": float(roll_sign),
                                    "forward_axis_assumption": forward_axis,
                                    "up_axis_assumption": up_axis,
                                    "camera_to_world_matrix": np.asarray(candidate_camera_to_world, dtype=np.float32).tolist(),
                                    "world_to_camera_matrix": np.linalg.inv(np.asarray(candidate_camera_to_world, dtype=np.float32)).astype(np.float32).tolist(),
                                    "matrix_diff_vs_dataset_camera_world": _matrix_diff_report(candidate_camera_to_world, camera_to_world),
                                    "overlay": _file_report(overlay_path),
                                    "marker_overlay": _file_report(marker_overlay_path),
                                    "rows": rows,
                                    "synthetic_markers": marker_rows,
                                }
                            )
                            sheet_paths.append(overlay_path)
                            sheet_labels.append(
                                f"p{'+' if pitch_sign > 0 else '-'} r{'+' if roll_sign > 0 else '-'} {forward_axis}/{up_axis}"
                            )
                            marker_sheet_paths.append(marker_overlay_path)
                            marker_sheet_labels.append(
                                f"p{'+' if pitch_sign > 0 else '-'} r{'+' if roll_sign > 0 else '-'} {forward_axis}/{up_axis}"
                            )
            body_sheet_path = audit_root / f"{yaw_tag}__camera_model_contact_sheet.png"
            marker_sheet_path = audit_root / f"{yaw_tag}__camera_model_markers_contact_sheet.png"
            _make_tiled_contact_sheet(sheet_paths, sheet_labels, body_sheet_path, cols=4)
            _make_tiled_contact_sheet(marker_sheet_paths, marker_sheet_labels, marker_sheet_path, cols=4)
            grouped_variant_sheets.append(
                {
                    "yaw_offset_deg": float(yaw_offset_deg),
                    "body_contact_sheet": _file_report(body_sheet_path),
                    "marker_contact_sheet": _file_report(marker_sheet_path),
                }
            )

    metadata_camera_pose = pose_json.get("camera_pose_cm_deg") or {}
    reconstructed_camera_world = benchmark_export._make_transform_matrix(
        float(metadata_camera_pose.get("x", 0.0)),
        float(metadata_camera_pose.get("y", 0.0)),
        float(metadata_camera_pose.get("z", 0.0)),
        float(metadata_camera_pose.get("yaw", 0.0)),
        float(metadata_camera_pose.get("pitch", 0.0)),
        float(metadata_camera_pose.get("roll", 0.0)),
    )

    audit_report = {
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "sequence_id": sequence_id,
        "sequence_name": sequence_manifest["sequence_name"],
        "sequence_local_frame_index": int(frame_index),
        "global_frame_index": int(global_frame_index),
        "timeline_frame_index": int(frame_meta["timeline_frame_index"]),
        "image_used": _file_report(image_path),
        "projection_overlay_existing": _file_report(current_overlay_path),
        "projection_overlay_existing_before_regeneration": current_overlay_before,
        "projection_overlay_regenerated_in_place": _file_report(current_overlay_path),
        "projection_overlay_regenerated_copy": _file_report(current_overlay_regenerated_path),
        "pose_json_used": _file_report(frame_record["pose_json_path"]),
        "camera_pose_used_for_render": pose_json.get("camera_pose_cm_deg"),
        "camera_pose_used_for_gt_projection_metadata": global_mapping.get("camera_pose_cm_deg"),
        "camera_pose_reconstructed_from_camera_world": _camera_pose_from_matrix(camera_to_world),
        "pose_json_contains_full_camera_matrix": bool("camera_to_world_matrix" in pose_json or "world_to_camera_matrix" in pose_json),
        "current_projection_matrix_source": {
            "uses_euler_reconstruction": True,
            "uses_dataset_camera_world_matrix": True,
            "pose_json_full_matrix_present": bool("camera_to_world_matrix" in pose_json or "world_to_camera_matrix" in pose_json),
            "note": "Current GT projection uses camera_world_transforms.npy. That file is produced during postprocess from camera pose Euler metadata, not from a full camera matrix stored in pose JSON.",
        },
        "metadata_euler_reconstruction_camera_to_world": np.asarray(reconstructed_camera_world, dtype=np.float32).tolist(),
        "metadata_euler_reconstruction_diff_vs_dataset_camera_world": _matrix_diff_report(reconstructed_camera_world, camera_to_world),
        "yaw_pitch_roll_metadata": {
            "render_pose": pose_json.get("camera_pose_cm_deg"),
            "gt_projection_pose": global_mapping.get("camera_pose_cm_deg"),
        },
        "camera_to_world_matrix": camera_to_world.tolist(),
        "world_to_camera_matrix": world_to_camera.tolist(),
        "erp_yaw_zero_offset_deg_used": float(convention["erp_yaw_zero_offset_deg"]),
        "erp_projection_convention_source": convention["path"],
        "erp_projection_convention_payload": convention["payload"],
        "pitch_roll_usage_audit": {
            "pitch_ignored": False,
            "roll_ignored": False,
            "note": "benchmark_export._make_transform_matrix uses yaw, pitch, and roll. GT projection consumes camera_world_transforms.npy derived from those Euler values.",
        },
        "render_manifest_scene_signature": next(
            (item.get("scene_signature") for item in render_manifest.get("sequence_specs", []) if item.get("sequence_id") == sequence_id),
            None,
        ),
        "generated_overlay_variants": variant_rows,
        "projection_variants_contact_sheet": _file_report(contact_sheet_path),
        "rotation_vs_translation_only_variants": translation_only_rows,
        "rotation_vs_translation_only_contact_sheet": _file_report(translation_only_contact_sheet_path),
        "translation_only_hypothesis_audit": {
            "question": "Does the ERP renderer behave as if it ignores camera yaw/pitch/roll, making translation-only projection align better than full camera-relative projection?",
            "current_projection_applies_camera_rotation": True,
            "translation_only_projection_applies_camera_rotation": False,
            "manual_review_required": True,
            "best_horizontal_alignment": "Inspect translation_only_vs_full_rotation_contact_sheet.png",
            "best_vertical_alignment": "Inspect translation_only_vs_full_rotation_contact_sheet.png",
        },
        "pitch_variant_yaw_offset_deg_used": float(selected_yaw_for_pitch_audit),
        "pitch_variants": pitch_variant_rows,
        "pitch_variants_contact_sheet": None if pitch_contact_sheet_path is None else _file_report(pitch_contact_sheet_path),
        "camera_model_variants": camera_model_rows,
        "camera_model_variant_contact_sheets": grouped_variant_sheets,
        "synthetic_markers": synthetic_markers,
        "synthetic_marker_overlay": _file_report(synthetic_overlay_path),
        "body_frame_mapping": global_mapping.get("body_frame_mapping"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    audit_report_path = audit_root / "projection_audit_report.json"
    _write_json(audit_report_path, audit_report)
    print(f"[BEDLAM360][V1_AUDIT] Audit report: {audit_report_path}")
    print(f"[BEDLAM360][V1_AUDIT] Contact sheet: {contact_sheet_path}")
    print(f"[BEDLAM360][V1_AUDIT] Rotation vs translation-only contact sheet: {translation_only_contact_sheet_path}")
    print(f"[BEDLAM360][V1_AUDIT] Synthetic markers overlay: {synthetic_overlay_path}")
    print(f"[BEDLAM360][V1_AUDIT] Current overlay regenerated in place: {current_overlay_path}")
    print(f"[BEDLAM360][V1_AUDIT] Camera model contact sheets: {[item['body_contact_sheet']['path'] for item in grouped_variant_sheets]}")
    print(f"[BEDLAM360][V1_AUDIT] Image path used: {image_path}")
    print(f"[BEDLAM360][V1_AUDIT] Existing overlay path: {current_overlay_path}")
    print(f"[BEDLAM360][V1_AUDIT] ERP yaw offset used: {float(convention['erp_yaw_zero_offset_deg'])}")
    print(f"[BEDLAM360][V1_AUDIT] Convention source: {convention['path']}")
    return audit_root


def _frame_camera_pose(frame_record):
    pose_json = _read_json(frame_record["pose_json_path"])
    return dict(pose_json.get("camera_pose_cm_deg") or {})


def _select_frames_for_verification(sequence_manifest, frames_per_sequence):
    frame_records = list(sequence_manifest["range_result"]["frame_records"])
    if not frame_records:
        return []
    anchor = _frame_camera_pose(frame_records[0])
    ax = float(anchor.get("x", 0.0))
    ay = float(anchor.get("y", 0.0))
    az = float(anchor.get("z", 0.0))
    scored = []
    for idx, frame_record in enumerate(frame_records):
        pose = _frame_camera_pose(frame_record)
        dx = float(pose.get("x", 0.0)) - ax
        dy = float(pose.get("y", 0.0)) - ay
        dz = float(pose.get("z", 0.0)) - az
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        scored.append({"frame_index": idx, "translation_distance_cm": float(distance)})
    chosen = []
    ordered_candidates = []
    if scored:
        ordered_candidates.append(scored[0])
        ordered_candidates.append(max(scored, key=lambda item: item["translation_distance_cm"]))
        ordered_candidates.append(scored[len(scored) // 2])
        ordered_candidates.extend(sorted(scored, key=lambda item: item["translation_distance_cm"], reverse=True))
    for item in ordered_candidates:
        if item["frame_index"] not in chosen:
            chosen.append(item["frame_index"])
        if len(chosen) >= int(frames_per_sequence):
            break
    return chosen


def verify_projection_mode_across_frames(run_root, sequence_ids=None, frames_per_sequence=3):
    run_root = Path(run_root)
    render_manifest = _read_json(run_root / "render_manifest.json")
    sequence_specs = render_manifest.get("sequence_specs", [])
    if sequence_ids:
        wanted = set(sequence_ids)
        selected_sequence_ids = [item["sequence_id"] for item in sequence_specs if item["sequence_id"] in wanted]
    else:
        selected_sequence_ids = [item["sequence_id"] for item in sequence_specs[:3]]
    if len(selected_sequence_ids) < 3:
        print(f"[BEDLAM360][V1_AUDIT] Warning: only {len(selected_sequence_ids)} sequence(s) available for verification.")

    verify_root = _ensure_dir(run_root / "projection_mode_verification")
    summary_rows = []
    per_sequence_reports = []
    for sequence_id in selected_sequence_ids:
        _render_manifest, sequence_manifest = _load_sequence_context(run_root, sequence_id)
        selected_frames = _select_frames_for_verification(sequence_manifest, frames_per_sequence=int(frames_per_sequence))
        sequence_rows = []
        for frame_index in selected_frames:
            audit_root = audit_v1_frame(
                run_root=run_root,
                sequence_id=sequence_id,
                frame_index=int(frame_index),
                focused_translation_only=True,
            )
            audit_report = _read_json(audit_root / "projection_audit_report.json")
            frame_record = _find_sequence_frame_record(sequence_manifest, int(frame_index))
            pose = _frame_camera_pose(frame_record)
            anchor_pose = _frame_camera_pose(sequence_manifest["range_result"]["frame_records"][0])
            dx = float(pose.get("x", 0.0)) - float(anchor_pose.get("x", 0.0))
            dy = float(pose.get("y", 0.0)) - float(anchor_pose.get("y", 0.0))
            dz = float(pose.get("z", 0.0)) - float(anchor_pose.get("z", 0.0))
            translation_distance_cm = math.sqrt(dx * dx + dy * dy + dz * dz)
            row = {
                "sequence_id": sequence_id,
                "frame_index": int(frame_index),
                "timeline_frame_index": int(audit_report["timeline_frame_index"]),
                "translation_distance_cm": float(translation_distance_cm),
                "contact_sheet_path": audit_report["rotation_vs_translation_only_contact_sheet"]["path"],
                "full_rotation_overlay_path": next(
                    item["overlay"]["path"]
                    for item in audit_report["rotation_vs_translation_only_variants"]
                    if item["name"] == "full_rotation__yaw_calibrated"
                ),
                "translation_only_p180_overlay_path": next(
                    item["overlay"]["path"]
                    for item in audit_report["rotation_vs_translation_only_variants"]
                    if item["name"] == "translation_only__yaw_p180"
                ),
                "manual_best_alignment": "",
                "notes": "Fill manual_best_alignment after visual inspection: full_rotation__yaw_calibrated or translation_only__yaw_p180.",
            }
            summary_rows.append(row)
            sequence_rows.append(row)
        per_sequence_reports.append(
            {
                "sequence_id": sequence_id,
                "selected_frame_indices": [int(item["frame_index"]) for item in sequence_rows],
                "rows": sequence_rows,
            }
        )

    summary_json = {
        "run_root": str(run_root),
        "verification_kind": "v1_rotation_vs_translation_only",
        "selected_sequence_ids": selected_sequence_ids,
        "frames_per_sequence": int(frames_per_sequence),
        "rows": summary_rows,
        "per_sequence_reports": per_sequence_reports,
        "report_note": "Manual visual review is still required to declare a consistent winner across sequences.",
    }
    _write_json(verify_root / "verification_report.json", summary_json)
    with (verify_root / "verification_report.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "sequence_id",
                "frame_index",
                "timeline_frame_index",
                "translation_distance_cm",
                "contact_sheet_path",
                "full_rotation_overlay_path",
                "translation_only_p180_overlay_path",
                "manual_best_alignment",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[BEDLAM360][V1_AUDIT] Verification report: {verify_root / 'verification_report.json'}")
    print(f"[BEDLAM360][V1_AUDIT] Verification CSV: {verify_root / 'verification_report.csv'}")
    return verify_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--sequence-id", default="")
    parser.add_argument("--frame-index", type=int, default=0, help="Sequence-local frame index.")
    parser.add_argument("--focused-translation-only", action="store_true", help="Skip the heavier pitch/rotation-order and camera-model sweeps.")
    parser.add_argument("--verify-multi-frame", action="store_true", help="Audit at least three sequences and several frames using the focused translation-only comparison.")
    parser.add_argument("--frames-per-sequence", type=int, default=3)
    parser.add_argument("--sequence-ids", default="", help="Optional comma-separated sequence ids for multi-frame verification.")
    args = parser.parse_args()
    if args.verify_multi_frame:
        sequence_ids = [part.strip() for part in str(args.sequence_ids).split(",") if part.strip()]
        verify_projection_mode_across_frames(
            args.run_root,
            sequence_ids=sequence_ids or None,
            frames_per_sequence=int(args.frames_per_sequence),
        )
    else:
        if not args.sequence_id:
            raise RuntimeError("--sequence-id is required unless --verify-multi-frame is used")
        audit_v1_frame(
            args.run_root,
            args.sequence_id,
            args.frame_index,
            focused_translation_only=bool(args.focused_translation_only),
        )


if __name__ == "__main__":
    main()
