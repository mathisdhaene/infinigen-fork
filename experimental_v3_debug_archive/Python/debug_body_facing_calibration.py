import argparse
import importlib
import json
import math
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import unreal


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bedlam360_canonical_validation as canonical
import bedlam360_mini_validation as mini

canonical = importlib.reload(canonical)
mini = importlib.reload(mini)


DEFAULT_OUTPUT_ROOT = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_body_facing_calibration")
DEFAULT_LATEST_RUN_JSON = DEFAULT_OUTPUT_ROOT / "LATEST_RUN.json"
DEFAULT_FRAME_INDEX = 12
DEFAULT_CAMERA_POSE = dict(canonical.DEFAULT_CAMERA_POSE)
DEFAULT_POSITIONS = {
    "pos_px": {"x": 220.0, "y": 0.0, "z": 0.0},
    "pos_nx": {"x": -220.0, "y": 0.0, "z": 0.0},
    "pos_py": {"x": 0.0, "y": 220.0, "z": 0.0},
    "pos_ny": {"x": 0.0, "y": -220.0, "z": 0.0},
}
DEFAULT_YAW_OFFSETS_DEG = [-180.0, -90.0, 0.0, 90.0, 180.0]
DEFAULT_ASSET_CONFIGS = [
    {
        "asset_id": "it_4052_3XL_2406",
        **dict(canonical.DEFAULT_FULL_APPEARANCE_FIELDS_BY_SLOT[0]),
    },
    {
        "asset_id": "it_4049_2XL_2400",
        **dict(canonical.DEFAULT_FULL_APPEARANCE_FIELDS_BY_SLOT[1]),
    },
    {
        "asset_id": "it_4029_L_2402",
        **dict(canonical.DEFAULT_FULL_APPEARANCE_FIELDS_BY_SLOT[2]),
    },
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


def _utc_now():
    return datetime.now(timezone.utc)


def _make_run_id():
    return f"{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"


def _log(message):
    unreal.log(f"[BEDLAM360][BODY_FACING_CALIBRATION] {message}")


def _normalize_yaw_deg(yaw_deg):
    yaw = float(yaw_deg)
    while yaw > 180.0:
        yaw -= 360.0
    while yaw <= -180.0:
        yaw += 360.0
    return yaw


def _computed_face_camera_yaw_deg(body_position_cm, camera_pose_cm_deg):
    dx = float(camera_pose_cm_deg["x"]) - float(body_position_cm["x"])
    dy = float(camera_pose_cm_deg["y"]) - float(body_position_cm["y"])
    return _normalize_yaw_deg(math.degrees(math.atan2(dy, dx)))


def _body_spec_for_candidate(asset_config, position_name, position_cm, yaw_offset_deg, camera_pose_cm_deg):
    computed_yaw = _computed_face_camera_yaw_deg(position_cm, camera_pose_cm_deg)
    final_yaw = _normalize_yaw_deg(computed_yaw + float(yaw_offset_deg))
    return {
        "asset_id": asset_config["asset_id"],
        "x": float(position_cm["x"]),
        "y": float(position_cm["y"]),
        "z": float(position_cm["z"]),
        "yaw": float(final_yaw),
        "pitch": 0.0,
        "roll": 0.0,
        "start_frame": 1,
        "texture_body": asset_config.get("texture_body"),
        "texture_clothing": asset_config.get("texture_clothing"),
        "texture_clothing_overlay": asset_config.get("texture_clothing_overlay"),
        "hair": asset_config.get("hair"),
        "haircolor": asset_config.get("haircolor"),
        "shoe": asset_config.get("shoe"),
        "shoe_offset": asset_config.get("shoe_offset"),
        "body_position_cm": {
            "x": float(position_cm["x"]),
            "y": float(position_cm["y"]),
            "z": float(position_cm["z"]),
        },
        "camera_position_cm": {
            "x": float(camera_pose_cm_deg["x"]),
            "y": float(camera_pose_cm_deg["y"]),
            "z": float(camera_pose_cm_deg["z"]),
        },
        "direction_vector_to_camera_cm": {
            "x": float(camera_pose_cm_deg["x"] - position_cm["x"]),
            "y": float(camera_pose_cm_deg["y"] - position_cm["y"]),
            "z": float(camera_pose_cm_deg["z"] - position_cm["z"]),
        },
        "computed_face_camera_yaw_deg": float(computed_yaw),
        "applied_yaw_offset_deg": float(yaw_offset_deg),
        "final_yaw_deg": float(final_yaw),
        "yaw_mode": "face_camera_plus_constant_offset",
        "position_name": position_name,
    }


def _overwrite_preview_overlay(candidate_metadata, range_result):
    pose_json_path = Path(range_result["frame_records"][0]["pose_json_path"])
    frame_record = json.loads(pose_json_path.read_text(encoding="utf-8"))
    frame_record["overlay_lines"] = [
        f"asset: {candidate_metadata['asset_id']}",
        f"position: {candidate_metadata['position_name']}",
        f"face_yaw: {candidate_metadata['computed_face_camera_yaw_deg']:.1f}",
        f"yaw_offset: {candidate_metadata['applied_yaw_offset_deg']:+.1f}",
        f"final_yaw: {candidate_metadata['final_yaw_deg']:.1f}",
    ]
    pose_json_path.write_text(json.dumps(frame_record, indent=2), encoding="utf-8")
    preview_status = mini._run_preview_frame(
        image_path=range_result["frame_records"][0]["exr_path"],
        output_png_path=range_result["frame_records"][0]["png_path"],
        metadata_json_path=str(pose_json_path),
        overlay=True,
    )
    return preview_status


def _run_contact_sheet(output_path, image_paths, cols=5):
    image_paths = [str(Path(path)) for path in image_paths if Path(path).is_file()]
    if not image_paths:
        return {"attempted": False, "success": False, "reason": "no_images"}
    command = [
        "python3",
        str(mini.PREVIEW_SCRIPT_PATH),
        "contact-sheet",
        str(output_path),
        *image_paths,
        "--cols",
        str(int(cols)),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "attempted": True,
        "success": completed.returncode == 0,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_path": str(output_path),
    }


def _write_selected_offset_convention(run_root, selected_offset_deg):
    payload = {
        "kind": "BEDLAM360-body-forward-yaw-convention",
        "body_forward_yaw_offset_deg": float(selected_offset_deg),
        "discovered_by": "manual_visual_calibration",
        "created_at_utc": _utc_now().isoformat(),
    }
    path = Path(run_root) / "body_forward_yaw_convention.json"
    _write_json(path, payload)
    return path


def render_body_facing_calibration(
    output_root=DEFAULT_OUTPUT_ROOT,
    frame_index=DEFAULT_FRAME_INDEX,
    selected_offset_deg=None,
):
    run_id = _make_run_id()
    run_root = _ensure_dir(Path(output_root) / run_id)
    camera_pose = dict(DEFAULT_CAMERA_POSE)
    all_candidates = []
    all_preview_paths = []
    canonical._validate_assets([{"asset_id": cfg["asset_id"]} for cfg in DEFAULT_ASSET_CONFIGS])

    for asset_index, asset_config in enumerate(DEFAULT_ASSET_CONFIGS):
        asset_root = _ensure_dir(run_root / asset_config["asset_id"])
        asset_preview_paths = []
        asset_candidates = []
        for position_name, position_cm in DEFAULT_POSITIONS.items():
            for yaw_offset_deg in DEFAULT_YAW_OFFSETS_DEG:
                body_spec = _body_spec_for_candidate(
                    asset_config=asset_config,
                    position_name=position_name,
                    position_cm=position_cm,
                    yaw_offset_deg=yaw_offset_deg,
                    camera_pose_cm_deg=camera_pose,
                )
                candidate_id = (
                    f"{asset_index:02d}_{asset_config['asset_id']}__{position_name}__yaw_"
                    f"{int(yaw_offset_deg):+d}".replace("+", "p").replace("-", "m")
                )
                candidate_root = _ensure_dir(asset_root / candidate_id)
                sequence_name = f"body_facing_calibration_{asset_index:02d}_{position_name}_{candidate_id.split('__')[-1]}"
                _log(
                    f"Rendering asset={asset_config['asset_id']} position={position_name} "
                    f"face_yaw={body_spec['computed_face_camera_yaw_deg']:.1f} "
                    f"offset={yaw_offset_deg:+.1f} final_yaw={body_spec['final_yaw_deg']:.1f}"
                )
                result = canonical.render_full_appearance_sequence_to_root(
                    run_id=run_id,
                    run_root=candidate_root,
                    sequence_name=sequence_name,
                    frame_start=int(frame_index),
                    frame_end=int(frame_index),
                    body_specs=[body_spec],
                    camera_pose=camera_pose,
                )
                preview_status = _overwrite_preview_overlay(body_spec, result["range_result"])
                candidate_manifest = {
                    "candidate_id": candidate_id,
                    "asset_id": asset_config["asset_id"],
                    "position_name": position_name,
                    "body_position_cm": body_spec["body_position_cm"],
                    "camera_position_cm": body_spec["camera_position_cm"],
                    "direction_vector_to_camera_cm": body_spec["direction_vector_to_camera_cm"],
                    "computed_face_camera_yaw_deg": body_spec["computed_face_camera_yaw_deg"],
                    "applied_yaw_offset_deg": body_spec["applied_yaw_offset_deg"],
                    "final_yaw_deg": body_spec["final_yaw_deg"],
                    "yaw_mode": body_spec["yaw_mode"],
                    "appearance_debug_by_body": result["appearance_debug_by_body"],
                    "range_result": result["range_result"],
                    "overlay_refresh_status": preview_status,
                }
                manifest_path = _write_json(candidate_root / "candidate_manifest.json", candidate_manifest)
                png_path = Path(result["range_result"]["frame_records"][0]["png_path"])
                asset_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "manifest_path": str(manifest_path),
                        "png_path": str(png_path),
                        "position_name": position_name,
                        "applied_yaw_offset_deg": float(yaw_offset_deg),
                        "computed_face_camera_yaw_deg": float(body_spec["computed_face_camera_yaw_deg"]),
                        "final_yaw_deg": float(body_spec["final_yaw_deg"]),
                    }
                )
                asset_preview_paths.append(png_path)
                all_preview_paths.append(png_path)

        contact_sheet_status = _run_contact_sheet(
            output_path=asset_root / "contact_sheet.png",
            image_paths=asset_preview_paths,
            cols=len(DEFAULT_YAW_OFFSETS_DEG),
        )
        asset_manifest = {
            "asset_id": asset_config["asset_id"],
            "frame_index": int(frame_index),
            "camera_pose_cm_deg": camera_pose,
            "yaw_offsets_deg": list(DEFAULT_YAW_OFFSETS_DEG),
            "positions": DEFAULT_POSITIONS,
            "candidates": asset_candidates,
            "contact_sheet_status": contact_sheet_status,
        }
        _write_json(asset_root / "asset_manifest.json", asset_manifest)
        all_candidates.extend(asset_candidates)

    overall_contact_sheet_status = _run_contact_sheet(
        output_path=run_root / "contact_sheet_all_assets.png",
        image_paths=all_preview_paths,
        cols=len(DEFAULT_YAW_OFFSETS_DEG),
    )
    selected_offset_path = None
    if selected_offset_deg is not None:
        selected_offset_path = _write_selected_offset_convention(run_root, selected_offset_deg)

    run_manifest = {
        "kind": "BEDLAM360-body-facing-calibration",
        "version": 0,
        "run_id": run_id,
        "created_at_utc": _utc_now().isoformat(),
        "run_root": str(run_root),
        "frame_index": int(frame_index),
        "camera_pose_cm_deg": camera_pose,
        "tested_asset_ids": [item["asset_id"] for item in DEFAULT_ASSET_CONFIGS],
        "positions": DEFAULT_POSITIONS,
        "yaw_offsets_deg": list(DEFAULT_YAW_OFFSETS_DEG),
        "candidate_count": int(len(all_candidates)),
        "candidates": all_candidates,
        "overall_contact_sheet_status": overall_contact_sheet_status,
        "selected_body_forward_yaw_offset_deg": None if selected_offset_deg is None else float(selected_offset_deg),
        "selected_body_forward_yaw_offset_path": None if selected_offset_path is None else str(selected_offset_path),
        "manual_review_required": selected_offset_deg is None,
        "hypothesis": "BEDLAM GeometryCache local forward axis requires a constant yaw offset relative to the face-camera yaw formula.",
    }
    run_manifest_path = _write_json(run_root / "manifest.json", run_manifest)
    _write_json(
        DEFAULT_LATEST_RUN_JSON,
        {
            "run_id": run_id,
            "run_root": str(run_root),
            "manifest_path": str(run_manifest_path),
            "created_at_utc": run_manifest["created_at_utc"],
        },
    )
    _log(f"Wrote calibration manifest: {run_manifest_path}")
    _log(f"Overall contact sheet: {run_root / 'contact_sheet_all_assets.png'}")
    if selected_offset_path is not None:
        _log(f"Stored selected body forward yaw offset: {selected_offset_path}")
    return run_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--frame-index", type=int, default=DEFAULT_FRAME_INDEX)
    parser.add_argument(
        "--selected-offset-deg",
        type=float,
        default=None,
        help="Optional manual selection after visual review; if provided, save body_forward_yaw_convention.json.",
    )
    args = parser.parse_args()
    run_root = render_body_facing_calibration(
        output_root=Path(args.output_root),
        frame_index=int(args.frame_index),
        selected_offset_deg=args.selected_offset_deg,
    )
    print(run_root)


if __name__ == "__main__":
    main()
