import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import unreal  # type: ignore
except Exception:  # pragma: no cover - offline python path
    unreal = None


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bedlam360_pipeline.manifest_versioning import build_pipeline_versions  # type: ignore
from bedlam360_pipeline.qa_summary import write_dataset_html_summary  # type: ignore
from bedlam360_pipeline.quality_filters import default_quality_filter_spec  # type: ignore


DEFAULT_OUTPUT_ROOT = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_v0_dataset")
DEFAULT_LATEST_RUN_JSON = DEFAULT_OUTPUT_ROOT / "LATEST_RUN.json"
DEFAULT_TARGET_FRAMES = 500
DEFAULT_FRAMES_PER_SEQUENCE = 100
DEFAULT_FIXED_SEED = 3601001
DEFAULT_SMPLX_MODEL_ROOT = Path("/media/mathis/PANO/BEDLAM2/models/smplx")
DEFAULT_PREVIEW_CONTACT_SHEET = "dataset_contact_sheet.png"
DEFAULT_BODY_FORWARD_YAW_OFFSET_DEG = -90.0
DEFAULT_NPZ_SEARCH_ROOTS = [
    Path("/media/mathis/PANO/BEDLAM2/animations/training"),
    Path("/media/mathis/PANO/BEDLAM2/animations"),
    Path("/media/mathis/PANO/BEDLAM2"),
]
UNREAL_ANIMATIONS_ROOT = "/Engine/PS/Bedlam/SMPLX_LH_animations"
LOCAL_ANIMATIONS_ROOT = Path("/media/mathis/PANO/BEDLAM2/smpl/SMPLX_LH_animations")


def _load_postprocess_modules():
    import numpy as np  # type: ignore
    import bedlam360_benchmark_export as benchmark_export  # type: ignore
    import bedlam360_gt_erp_alignment as gt_alignment  # type: ignore
    import bedlam360_preview_tools as preview_tools  # type: ignore

    return np, benchmark_export, gt_alignment, preview_tools


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
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _utc_now():
    return datetime.now(timezone.utc)


def _make_run_id():
    return f"{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"


def _log(message):
    text = f"[BEDLAM360][V0_DATASET] {message}"
    if unreal is not None:
        unreal.log(text)
    else:
        print(text)


def _range_tag(frame_start, frame_end):
    return f"frames_{int(frame_start):04d}_{int(frame_end):04d}"


def _split_asset_id(asset_id):
    parts = str(asset_id).split("_")
    if len(parts) < 4:
        return str(asset_id), ""
    return "_".join(parts[:-1]), parts[-1]


def _local_animation_asset_path(asset_id):
    identity, _motion = _split_asset_id(asset_id)
    path = LOCAL_ANIMATIONS_ROOT / identity / f"{asset_id}.uasset"
    return str(path) if path.is_file() else None


def _unreal_animation_asset_path(asset_id):
    identity, _motion = _split_asset_id(asset_id)
    return f"{UNREAL_ANIMATIONS_ROOT}/{identity}/{asset_id}.{asset_id}"


_NPZ_PATH_CACHE = None


def _scan_npz_path_map():
    global _NPZ_PATH_CACHE
    if _NPZ_PATH_CACHE is not None:
        return _NPZ_PATH_CACHE
    mapping = {}
    for root in DEFAULT_NPZ_SEARCH_ROOTS:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*.npz"):
            mapping.setdefault(path.stem, str(path))
    _NPZ_PATH_CACHE = mapping
    return mapping


def _source_npz_path(asset_id):
    return _scan_npz_path_map().get(str(asset_id))


def _safe_symlink_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    _ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
    try:
        os.symlink(src, dst)
    except Exception:
        import shutil

        shutil.copy2(src, dst)
    return dst


def _build_scene_body_specs(base_layout, slot_fields, asset_ids):
    body_specs = []
    for index, item in enumerate(base_layout):
        spec = dict(item)
        spec["asset_id"] = asset_ids[index]
        spec.update(slot_fields.get(index, {}))
        spec["body_slot"] = index
        spec["start_frame"] = 1
        body_specs.append(spec)
    return body_specs


def _rotate_slot_fields(slot_fields, offset):
    ordered_keys = sorted(slot_fields.keys())
    ordered_values = [dict(slot_fields[key]) for key in ordered_keys]
    rotated = {}
    for index, key in enumerate(ordered_keys):
        rotated[key] = dict(ordered_values[(index + offset) % len(ordered_values)])
    return rotated


def _transform_layout(base_layout, slot_transforms=None):
    slot_transforms = slot_transforms or {}
    transformed = []
    for index, item in enumerate(base_layout):
        spec = dict(item)
        delta = slot_transforms.get(index, {})
        for key in ("x", "y", "z", "yaw", "pitch", "roll"):
            spec[key] = float(spec.get(key, 0.0)) + float(delta.get(key, 0.0))
        transformed.append(spec)
    return transformed


def _normalize_yaw_deg(yaw_deg):
    yaw = float(yaw_deg)
    while yaw > 180.0:
        yaw -= 360.0
    while yaw <= -180.0:
        yaw += 360.0
    return yaw


def _face_camera_yaw_deg(body_spec, camera_pose_cm_deg):
    dx = float(camera_pose_cm_deg["x"]) - float(body_spec["x"])
    dy = float(camera_pose_cm_deg["y"]) - float(body_spec["y"])
    return _normalize_yaw_deg(math.degrees(math.atan2(dy, dx)))


def _yaw_mode_offset_deg(yaw_mode, slot_index, base_offset_deg=DEFAULT_BODY_FORWARD_YAW_OFFSET_DEG):
    deterministic_small_offsets = [0.0, -12.5, 14.0]
    if yaw_mode == "face_camera":
        return float(base_offset_deg) + deterministic_small_offsets[slot_index % len(deterministic_small_offsets)]
    if yaw_mode == "front_three_quarter":
        return float(base_offset_deg) + (30.0 if slot_index % 2 == 0 else -30.0)
    if yaw_mode == "side_left":
        return float(base_offset_deg) - 90.0
    if yaw_mode == "side_right":
        return float(base_offset_deg) + 90.0
    if yaw_mode == "back_to_camera":
        return float(base_offset_deg) + 180.0
    if yaw_mode == "random_yaw":
        deterministic = [-135.0, -25.0, 80.0]
        return float(base_offset_deg) + deterministic[slot_index % len(deterministic)]
    raise ValueError(f"Unsupported yaw mode: {yaw_mode}")


def _apply_yaw_modes(body_specs, camera_pose_cm_deg, yaw_modes=None):
    yaw_modes = yaw_modes or ["face_camera"] * len(body_specs)
    updated = []
    for index, spec in enumerate(body_specs):
        item = dict(spec)
        yaw_mode = yaw_modes[index] if index < len(yaw_modes) else yaw_modes[-1]
        computed_face_yaw = _face_camera_yaw_deg(item, camera_pose_cm_deg)
        calibrated_forward_offset = float(item.get("body_forward_yaw_offset_deg", DEFAULT_BODY_FORWARD_YAW_OFFSET_DEG))
        yaw_offset = _yaw_mode_offset_deg(
            yaw_mode,
            index,
            float(item.get("yaw_offset_deg", calibrated_forward_offset)),
        )
        final_yaw = _normalize_yaw_deg(computed_face_yaw + yaw_offset)
        item["body_forward_yaw_offset_deg"] = float(calibrated_forward_offset)
        item["computed_face_camera_yaw_deg"] = float(computed_face_yaw)
        item["yaw_offset_deg"] = float(yaw_offset)
        item["final_yaw_deg"] = float(final_yaw)
        item["yaw_mode"] = str(yaw_mode)
        item["camera_position_cm"] = {
            "x": float(camera_pose_cm_deg["x"]),
            "y": float(camera_pose_cm_deg["y"]),
            "z": float(camera_pose_cm_deg["z"]),
        }
        item["body_position_cm"] = {
            "x": float(item["x"]),
            "y": float(item["y"]),
            "z": float(item["z"]),
        }
        item["yaw"] = float(final_yaw)
        updated.append(item)
    return updated


def _scene_signature_payload(scene):
    body_specs = []
    for spec in scene["body_specs"]:
        body_specs.append(
            {
                "asset_id": spec["asset_id"],
                "body_slot": int(spec["body_slot"]),
                "pose_cm_deg": {
                    "x": float(spec["x"]),
                    "y": float(spec["y"]),
                    "z": float(spec["z"]),
                    "yaw": float(spec["yaw"]),
                    "pitch": float(spec["pitch"]),
                    "roll": float(spec["roll"]),
                },
                "appearance": {
                    "texture_body": spec.get("texture_body"),
                    "texture_clothing": spec.get("texture_clothing"),
                    "texture_clothing_overlay": spec.get("texture_clothing_overlay"),
                    "hair": spec.get("hair"),
                    "haircolor": spec.get("haircolor"),
                    "shoe": spec.get("shoe"),
                    "shoe_offset": spec.get("shoe_offset"),
                },
            }
        )
    return {
        "scene_id": scene["scene_id"],
        "camera_pose_cm_deg": scene["camera_pose_cm_deg"],
        "body_specs": body_specs,
    }


def _scene_signature(scene):
    payload = _scene_signature_payload(scene)
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical_payload.encode("utf-8")).hexdigest()[:12]
    return f"{scene['scene_id']}__{digest}"


def _motion_triplet_payload(asset_ids):
    payload = []
    for asset_id in asset_ids:
        payload.append(
            {
                "asset_id": str(asset_id),
                "unreal_animation_asset_path": _unreal_animation_asset_path(asset_id),
                "local_animation_asset_path": _local_animation_asset_path(asset_id),
                "source_npz_path": _source_npz_path(asset_id),
            }
        )
    return payload


def _motion_triplet_signature(asset_ids):
    payload = _motion_triplet_payload(asset_ids)
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical_payload.encode("utf-8")).hexdigest()[:12]
    return f"motion_triplet__{digest}"


def _build_scene_preset(scene_id, label, camera_pose_cm_deg, base_layout, slot_fields, asset_ids, yaw_modes=None):
    body_specs = _build_scene_body_specs(base_layout, slot_fields, asset_ids)
    body_specs = _apply_yaw_modes(body_specs, camera_pose_cm_deg, yaw_modes=yaw_modes)
    scene = {
        "scene_id": scene_id,
        "label": label,
        "camera_pose_cm_deg": dict(camera_pose_cm_deg),
        "body_specs": body_specs,
        "body_forward_yaw_offset_deg": float(DEFAULT_BODY_FORWARD_YAW_OFFSET_DEG),
        "motion_triplet_asset_ids": list(asset_ids),
        "motion_triplet_payload": _motion_triplet_payload(asset_ids),
    }
    scene["scene_signature"] = _scene_signature(scene)
    scene["motion_triplet_signature"] = _motion_triplet_signature(asset_ids)
    return scene


def _diversity_report_from_scenes(scene_specs):
    scene_signatures = [scene.get("scene_signature") for scene in scene_specs]
    motion_triplet_signatures = [scene.get("motion_triplet_signature") for scene in scene_specs]
    body_triplets = [tuple(spec["asset_id"] for spec in scene.get("body_specs", [])) for scene in scene_specs]
    motion_ids = sorted({asset_id for triplet in body_triplets for asset_id in triplet})
    appearance_combinations = []
    duplicate_signatures = {}
    duplicate_motion_triplets = {}
    for signature in scene_signatures:
        duplicate_signatures[signature] = duplicate_signatures.get(signature, 0) + 1
    for signature in motion_triplet_signatures:
        duplicate_motion_triplets[signature] = duplicate_motion_triplets.get(signature, 0) + 1
    for scene in scene_specs:
        combo = []
        for spec in scene.get("body_specs", []):
            combo.append(
                (
                    spec["asset_id"],
                    spec.get("texture_body"),
                    spec.get("texture_clothing"),
                    spec.get("hair"),
                    spec.get("haircolor"),
                    spec.get("shoe"),
                )
            )
        appearance_combinations.append(tuple(combo))
    return {
        "unique_scene_signatures": int(len(set(scene_signatures))),
        "unique_motion_triplet_count": int(len(set(motion_triplet_signatures))),
        "unique_body_asset_triplets": int(len(set(body_triplets))),
        "unique_motion_ids": int(len(set(motion_ids))),
        "unique_appearance_combinations": int(len(set(appearance_combinations))),
        "scene_signature_count": int(len(scene_signatures)),
        "duplicate_scene_signatures": {key: value for key, value in duplicate_signatures.items() if value > 1},
        "duplicate_motion_triplets": {key: value for key, value in duplicate_motion_triplets.items() if value > 1},
        "sequence_motion_summary": [
            {
                "scene_id": scene.get("scene_id"),
                "scene_signature": scene.get("scene_signature"),
                "motion_triplet_signature": scene.get("motion_triplet_signature"),
                "asset_ids": [spec["asset_id"] for spec in scene.get("body_specs", [])],
                "source_npz_paths": [item.get("source_npz_path") for item in scene.get("motion_triplet_payload", [])],
                "frame_range": [int(scene.get("frame_start", 0)), int(scene.get("frame_end", 0))] if scene.get("frame_end") is not None else None,
            }
            for scene in scene_specs
        ],
    }


def _scene_presets(canonical):
    base_camera = dict(canonical.DEFAULT_CAMERA_POSE)
    base_layout = list(canonical.DEFAULT_BODY_LAYOUT)
    base_fields = canonical.DEFAULT_FULL_APPEARANCE_FIELDS_BY_SLOT
    return [
        _build_scene_preset(
            "scene_a_front",
            "motion_triplet_a_front",
            base_camera,
            base_layout,
            base_fields,
            ["it_4052_3XL_2406", "it_4049_2XL_2400", "it_4029_L_2402"],
            yaw_modes=["face_camera", "face_camera", "front_three_quarter"],
        ),
        _build_scene_preset(
            "scene_b_front",
            "motion_triplet_b_front",
            base_camera,
            base_layout,
            _rotate_slot_fields(base_fields, 1),
            ["it_4052_3XL_2408", "it_4083_2XL_2408", "it_4201_L_2403"],
            yaw_modes=["face_camera", "front_three_quarter", "face_camera"],
        ),
        _build_scene_preset(
            "scene_c_left",
            "motion_triplet_c_left",
            dict(base_camera, x=float(base_camera["x"]) - 80.0, y=float(base_camera["y"]) + 35.0, yaw=float(base_camera["yaw"]) - 18.0),
            _transform_layout(
                base_layout,
                {
                    0: {"x": -25.0, "y": 30.0, "yaw": -10.0},
                    1: {"x": 35.0, "y": -20.0, "yaw": 12.0},
                    2: {"x": -10.0, "y": -35.0, "yaw": 6.0},
                },
            ),
            _rotate_slot_fields(base_fields, 2),
            ["it_4052_3XL_2403", "it_4083_2XL_2400", "it_4201_L_2401"],
            yaw_modes=["front_three_quarter", "face_camera", "face_camera"],
        ),
        _build_scene_preset(
            "scene_d_right",
            "motion_triplet_d_right",
            dict(base_camera, x=float(base_camera["x"]) + 95.0, y=float(base_camera["y"]) - 40.0, yaw=float(base_camera["yaw"]) + 22.0),
            _transform_layout(
                base_layout,
                {
                    0: {"x": 20.0, "y": -28.0, "yaw": 14.0},
                    1: {"x": -30.0, "y": 22.0, "yaw": -16.0},
                    2: {"x": 18.0, "y": 26.0, "yaw": 9.0},
                },
            ),
            base_fields,
            ["it_4052_3XL_2410", "it_4083_2XL_2405", "it_4029_L_2400"],
            yaw_modes=["face_camera", "side_left", "face_camera"],
        ),
        _build_scene_preset(
            "scene_e_tight",
            "motion_triplet_e_tight",
            dict(base_camera, x=float(base_camera["x"]) - 45.0, z=float(base_camera["z"]) + 15.0, yaw=float(base_camera["yaw"]) + 12.0),
            _transform_layout(
                base_layout,
                {
                    0: {"x": -12.0, "y": 14.0, "yaw": 4.0},
                    1: {"x": 12.0, "y": -8.0, "yaw": -8.0},
                    2: {"x": 0.0, "y": 18.0, "yaw": 10.0},
                },
            ),
            _rotate_slot_fields(base_fields, 1),
            ["it_4052_3XL_2400", "it_4083_2XL_2409", "it_4201_L_2400"],
            yaw_modes=["face_camera", "face_camera", "side_right"],
        ),
    ]


def _sequence_specs(target_frames, frames_per_sequence, scene_presets, allow_duplicate_motion_triplets=False):
    if target_frames <= 0:
        raise ValueError("target_frames must be > 0")
    if frames_per_sequence <= 0:
        raise ValueError("frames_per_sequence must be > 0")

    # The validated canonical full-appearance clip spans timeline frames 0..120 inclusive.
    max_valid_frame = 120
    scene_frame_capacity = max_valid_frame + 1
    per_scene_frame_budget = min(int(frames_per_sequence), int(scene_frame_capacity))
    requested_sequence_count = int(math.ceil(float(target_frames) / float(frames_per_sequence)))
    available_scene_count = int(len(scene_presets))
    usable_sequence_count = min(requested_sequence_count, available_scene_count)
    max_total_frames = usable_sequence_count * per_scene_frame_budget
    actual_target_frames = min(int(target_frames), int(max_total_frames))
    remaining = int(actual_target_frames)
    specs = []
    for sequence_index, scene in enumerate(scene_presets[:usable_sequence_count]):
        if remaining <= 0:
            break
        this_count = min(int(frames_per_sequence), remaining)
        if this_count > scene_frame_capacity:
            raise RuntimeError(
                f"Requested sequence length {this_count} exceeds validated canonical clip capacity {scene_frame_capacity}."
            )
        frame_start = 0
        frame_end = frame_start + this_count - 1
        specs.append(
            {
                "sequence_index": int(sequence_index),
                "sequence_id": f"seq_{sequence_index:04d}",
                "sequence_name": f"bedlam360_v0_{scene['scene_id']}_seq_{sequence_index:04d}",
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "frame_count": int(this_count),
                "camera_pose_cm_deg": dict(scene["camera_pose_cm_deg"]),
                "scene_id": scene["scene_id"],
                "scene_label": scene["label"],
                "scene_signature": scene["scene_signature"],
                "motion_triplet_signature": scene["motion_triplet_signature"],
                "motion_triplet_payload": scene["motion_triplet_payload"],
                "body_specs": scene["body_specs"],
            }
        )
        remaining -= this_count
    duplicate_motion_triplets = _diversity_report_from_scenes(specs)["duplicate_motion_triplets"]
    if duplicate_motion_triplets and not allow_duplicate_motion_triplets:
        raise RuntimeError(
            "Duplicate motion triplets detected in sequence planning: "
            f"{duplicate_motion_triplets}. Re-run with --allow-duplicate-motion-triplets only if this is intentional."
        )
    planning = {
        "requested_target_frames": int(target_frames),
        "requested_sequence_count": int(requested_sequence_count),
        "available_scene_count": int(available_scene_count),
        "usable_sequence_count": int(len(specs)),
        "scene_frame_capacity": int(scene_frame_capacity),
        "per_scene_frame_budget": int(per_scene_frame_budget),
        "actual_target_frames": int(actual_target_frames),
        "reduced_target_frames": bool(actual_target_frames < int(target_frames)),
        "reduction_reason": None
        if int(actual_target_frames) == int(target_frames)
        else (
            f"Requested {int(target_frames)} frames but only {int(len(specs))} unique curated scenes "
            f"are configured with {int(per_scene_frame_budget)} frames each."
        ),
        "allow_duplicate_motion_triplets": bool(allow_duplicate_motion_triplets),
    }
    return specs, planning


def _appearance_ids_by_body_slot(body_specs):
    results = []
    for spec in body_specs:
        results.append(
            {
                "body_slot": int(spec["body_slot"]),
                "asset_id": spec["asset_id"],
                "texture_body": spec.get("texture_body"),
                "texture_clothing": spec.get("texture_clothing"),
                "texture_clothing_overlay": spec.get("texture_clothing_overlay"),
                "hair": spec.get("hair"),
                "haircolor": spec.get("haircolor"),
                "shoe": spec.get("shoe"),
                "shoe_offset": spec.get("shoe_offset"),
            }
        )
    return results


def _load_canonical_module():
    if unreal is None:
        raise RuntimeError("Render stage requires Unreal Python.")
    import bedlam360_canonical_validation as canonical_validation  # type: ignore

    return importlib.reload(canonical_validation)


def _render_sequence(canonical, run_id, run_root, sequence_spec, body_specs):
    sequence_root = _ensure_dir(run_root / "raw" / "sequences" / sequence_spec["sequence_id"])
    camera_pose = dict(canonical.DEFAULT_CAMERA_POSE)
    if sequence_spec.get("camera_pose_cm_deg"):
        camera_pose.update(sequence_spec["camera_pose_cm_deg"])
    unreal_sequence_name = f"{sequence_spec['sequence_name']}_{run_id}"
    result = canonical.render_full_appearance_sequence_to_root(
        run_id=run_id,
        run_root=sequence_root,
        sequence_name=unreal_sequence_name,
        frame_start=int(sequence_spec["frame_start"]),
        frame_end=int(sequence_spec["frame_end"]),
        body_specs=body_specs,
        camera_pose=camera_pose,
    )
    sequence_manifest = {
        "sequence_id": sequence_spec["sequence_id"],
        "sequence_name": sequence_spec["sequence_name"],
        "unreal_sequence_name": unreal_sequence_name,
        "sequence_index": int(sequence_spec["sequence_index"]),
        "frame_start": int(sequence_spec["frame_start"]),
        "frame_end": int(sequence_spec["frame_end"]),
        "frame_count": int(sequence_spec["frame_count"]),
        "camera_pose_cm_deg": camera_pose,
        "scene_id": sequence_spec.get("scene_id"),
        "scene_label": sequence_spec.get("scene_label"),
        "scene_signature": sequence_spec.get("scene_signature"),
        "motion_triplet_signature": sequence_spec.get("motion_triplet_signature"),
        "motion_triplet_payload": sequence_spec.get("motion_triplet_payload"),
        "body_specs": result["body_specs"],
        "spawned_roles": result["spawned_roles"],
        "appearance_debug_by_body": result["appearance_debug_by_body"],
        "level_sequence": result["level_sequence"],
        "range_result": result["range_result"],
        "paths": {
            "sequence_root": str(sequence_root),
            "images_dir": str(sequence_root / result["range_result"]["range_tag"] / "images"),
            "metadata_dir": str(sequence_root / result["range_result"]["range_tag"] / "metadata"),
            "previews_dir": str(sequence_root / result["range_result"]["range_tag"] / "previews"),
            "preview_mp4_path": str(result["range_result"]["preview_mp4_path"]),
        },
        "appearance_status_by_body": [
            {
                "asset_id": item["asset_id"],
                "render_role": item.get("render_role", "body"),
                "body_material_applied": bool(((item.get("appearance_metadata") or {}).get("body_material") or {}).get("applied", False)),
                "clothing_actor_spawned": bool(
                    item.get("render_role") == "clothing"
                    or ((item.get("appearance_metadata") or {}).get("clothing") or {}).get("applied", False)
                ),
                "clothing_material_applied": bool(((item.get("appearance_metadata") or {}).get("clothing") or {}).get("material_applied", False)),
                "hair_applied": bool(((item.get("appearance_metadata") or {}).get("hair") or {}).get("applied", False)),
                "shoe_applied": bool(((item.get("appearance_metadata") or {}).get("shoe") or {}).get("applied", False)),
            }
            for item in result["spawned_roles"]
        ],
    }
    manifest_path = sequence_root / "manifest.json"
    _write_json(manifest_path, sequence_manifest)
    _log(
        f"Rendered {sequence_spec['sequence_id']} frames={sequence_spec['frame_start']}..{sequence_spec['frame_end']} "
        f"unreal_sequence={unreal_sequence_name} "
        f"scene_id={sequence_spec.get('scene_id')} scene_signature={sequence_spec.get('scene_signature')} "
        f"preview={result['range_result']['preview_mp4_path']}"
    )
    _log(f"Sequence manifest: {manifest_path}")
    _log(f"Sequence images dir: {sequence_manifest['paths']['images_dir']}")
    _log(f"Sequence metadata dir: {sequence_manifest['paths']['metadata_dir']}")
    _log(f"Sequence preview mp4: {sequence_manifest['paths']['preview_mp4_path']}")
    return {
        "sequence_id": sequence_spec["sequence_id"],
        "sequence_root": str(sequence_root),
        "manifest_path": str(manifest_path),
        "frame_count": int(sequence_spec["frame_count"]),
        "range_result": result["range_result"],
    }


def render_v0_dataset(
    output_root=DEFAULT_OUTPUT_ROOT,
    target_frames=DEFAULT_TARGET_FRAMES,
    frames_per_sequence=DEFAULT_FRAMES_PER_SEQUENCE,
    fixed_seed=DEFAULT_FIXED_SEED,
    auto_postprocess=False,
    smplx_model_roots=None,
    allow_duplicate_motion_triplets=False,
):
    canonical = _load_canonical_module()
    run_id = _make_run_id()
    run_root = _ensure_dir(Path(output_root) / run_id)
    script_path = Path(__file__).resolve()
    script_stat = script_path.stat()
    scene_presets = _scene_presets(canonical)
    sequence_specs, planning = _sequence_specs(
        target_frames=int(target_frames),
        frames_per_sequence=int(frames_per_sequence),
        scene_presets=scene_presets,
        allow_duplicate_motion_triplets=bool(allow_duplicate_motion_triplets),
    )
    rendered_sequences = []
    for sequence_spec in sequence_specs:
        rendered_sequences.append(
            _render_sequence(
                canonical=canonical,
                run_id=run_id,
                run_root=run_root,
                sequence_spec=sequence_spec,
                body_specs=sequence_spec["body_specs"],
            )
        )
    diversity_report = _diversity_report_from_scenes(sequence_specs)
    duplicate_scene_signatures = render_manifest_duplicates = diversity_report["duplicate_scene_signatures"]
    if duplicate_scene_signatures:
        _log(f"WARNING duplicate scene signatures detected: {duplicate_scene_signatures}")
    if diversity_report["duplicate_motion_triplets"]:
        _log(f"WARNING duplicate motion triplets detected: {diversity_report['duplicate_motion_triplets']}")

    render_manifest = {
        "kind": "BEDLAM360-v0-render",
        "version": 0,
        "run_id": run_id,
        "created_at_utc": _utc_now().isoformat(),
        "dataset_root": str(run_root),
        "fixed_seed": int(fixed_seed),
        "target_frames_requested": int(target_frames),
        "target_frames": int(planning["actual_target_frames"]),
        "frames_per_sequence": int(frames_per_sequence),
        "sequence_count": int(len(rendered_sequences)),
        "body_count": 3,
        "body_forward_yaw_offset_deg": float(DEFAULT_BODY_FORWARD_YAW_OFFSET_DEG),
        "scene_presets": scene_presets,
        "scene_planning": planning,
        "diversity_report": diversity_report,
        "duplicate_scene_signature_warning": render_manifest_duplicates,
        "duplicate_motion_triplet_warning": diversity_report["duplicate_motion_triplets"],
        "camera_model": {
            "projection": "equirectangular_360",
            "render_pipeline": "sequencer_geometrycache_scene_capture_cube",
        },
        "sequence_specs": sequence_specs,
        "rendered_sequences": rendered_sequences,
        "script_path": str(script_path),
        "script_mtime_utc": datetime.fromtimestamp(script_stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    render_manifest_path = run_root / "render_manifest.json"
    _write_json(render_manifest_path, render_manifest)
    _write_json(
        DEFAULT_LATEST_RUN_JSON,
        {
            "run_id": run_id,
            "run_root": str(run_root),
            "render_manifest_path": str(render_manifest_path),
            "created_at_utc": render_manifest["created_at_utc"],
        },
    )
    _log(f"Wrote render manifest: {render_manifest_path}")

    if auto_postprocess:
        command = [
            "python3",
            str(script_path),
            "--stage",
            "postprocess",
            "--run-root",
            str(run_root),
        ]
        for model_root in smplx_model_roots or []:
            command.extend(["--smplx-model-root", str(model_root)])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        _write_json(
            run_root / "postprocess_status.json",
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
        _log(f"Postprocess returncode={completed.returncode}")
    return run_root


def render_v0_debug_sequence(
    output_root=DEFAULT_OUTPUT_ROOT,
    fixed_seed=DEFAULT_FIXED_SEED,
    allow_duplicate_motion_triplets=False,
):
    canonical = _load_canonical_module()
    scene_presets = _scene_presets(canonical)
    run_id = _make_run_id()
    run_root = _ensure_dir(Path(output_root) / f"{run_id}_debug")
    scene = scene_presets[0]
    body_specs = scene["body_specs"]
    sequence_spec = {
        "sequence_index": 0,
        "sequence_id": "seq_debug",
        "sequence_name": f"bedlam360_v0_{scene['scene_id']}_debug_seq",
        "frame_start": 0,
        "frame_end": 9,
        "frame_count": 10,
        "camera_pose_cm_deg": dict(scene["camera_pose_cm_deg"]),
        "scene_id": scene["scene_id"],
        "scene_label": scene["label"],
        "scene_signature": scene["scene_signature"],
        "body_specs": body_specs,
    }
    rendered = _render_sequence(
        canonical=canonical,
        run_id=run_id,
        run_root=run_root,
        sequence_spec=sequence_spec,
        body_specs=body_specs,
    )
    manifest = {
        "kind": "BEDLAM360-v0-render-debug",
        "version": 0,
        "run_id": run_id,
        "created_at_utc": _utc_now().isoformat(),
        "dataset_root": str(run_root),
        "fixed_seed": int(fixed_seed),
        "sequence_count": 1,
        "body_count": int(len(body_specs)),
        "body_forward_yaw_offset_deg": float(DEFAULT_BODY_FORWARD_YAW_OFFSET_DEG),
        "asset_ids": [spec["asset_id"] for spec in body_specs],
        "motion_triplet_signature": sequence_spec["motion_triplet_signature"],
        "motion_triplet_payload": sequence_spec["motion_triplet_payload"],
        "body_specs": body_specs,
        "appearance_ids_by_body_slot": _appearance_ids_by_body_slot(body_specs),
        "scene_presets": scene_presets,
        "sequence_specs": [sequence_spec],
        "rendered_sequences": [rendered],
        "diversity_report": _diversity_report_from_scenes([sequence_spec]),
        "compare_against": {
            "known_working_entrypoint": "bedlam360_canonical_validation.py --render-full-appearance",
            "known_working_range": "frames 0..120",
            "debug_range": "frames 0..9",
        },
        "paths": {
            "run_root": str(run_root),
            "render_manifest_path": str(run_root / "render_manifest.json"),
            "sequence_manifest_path": str(Path(rendered["manifest_path"])),
        },
    }
    render_manifest_path = run_root / "render_manifest.json"
    _write_json(render_manifest_path, manifest)
    _log(f"Debug run root: {run_root}")
    _log(f"Debug render manifest: {render_manifest_path}")
    _log(f"Debug sequence manifest: {rendered['manifest_path']}")
    return run_root


def render_v0_debug_two_scenes(
    output_root=DEFAULT_OUTPUT_ROOT,
    fixed_seed=DEFAULT_FIXED_SEED,
    allow_duplicate_motion_triplets=False,
):
    canonical = _load_canonical_module()
    scene_presets = _scene_presets(canonical)
    run_id = _make_run_id()
    run_root = _ensure_dir(Path(output_root) / f"{run_id}_debug_two_scenes")
    rendered_sequences = []
    sequence_specs = []
    for index, scene in enumerate(scene_presets[:2]):
        sequence_spec = {
            "sequence_index": int(index),
            "sequence_id": f"seq_debug_{scene['scene_id']}",
            "sequence_name": f"bedlam360_v0_{scene['scene_id']}_debug_seq",
            "frame_start": 0,
            "frame_end": 9,
            "frame_count": 10,
            "camera_pose_cm_deg": dict(scene["camera_pose_cm_deg"]),
            "scene_id": scene["scene_id"],
            "scene_label": scene["label"],
            "scene_signature": scene["scene_signature"],
            "body_specs": scene["body_specs"],
        }
        rendered_sequences.append(
            _render_sequence(
                canonical=canonical,
                run_id=run_id,
                run_root=run_root,
                sequence_spec=sequence_spec,
                body_specs=scene["body_specs"],
            )
        )
        sequence_specs.append(sequence_spec)
    manifest = {
        "kind": "BEDLAM360-v0-render-debug-two-scenes",
        "version": 0,
        "run_id": run_id,
        "created_at_utc": _utc_now().isoformat(),
        "dataset_root": str(run_root),
        "fixed_seed": int(fixed_seed),
        "sequence_count": int(len(rendered_sequences)),
        "body_forward_yaw_offset_deg": float(DEFAULT_BODY_FORWARD_YAW_OFFSET_DEG),
        "scene_presets": scene_presets,
        "sequence_specs": sequence_specs,
        "rendered_sequences": rendered_sequences,
        "diversity_report": _diversity_report_from_scenes(sequence_specs),
    }
    render_manifest_path = run_root / "render_manifest.json"
    _write_json(render_manifest_path, manifest)
    _log(f"Debug-two-scenes run root: {run_root}")
    _log(f"Debug-two-scenes render manifest: {render_manifest_path}")
    return run_root


def _load_sequence_manifests(run_root):
    render_manifest = _read_json(Path(run_root) / "render_manifest.json")
    sequence_manifests = []
    for item in render_manifest["rendered_sequences"]:
        sequence_manifest = _read_json(item["manifest_path"])
        sequence_manifests.append(sequence_manifest)
    return render_manifest, sequence_manifests


def _copy_render_outputs(sequence_manifest, dataset_root):
    images_dir = _ensure_dir(dataset_root / "images")
    metadata_frames_dir = _ensure_dir(dataset_root / "metadata" / "frames")
    previews_dir = _ensure_dir(dataset_root / "previews")

    sequence_id = sequence_manifest["sequence_id"]
    range_result = sequence_manifest["range_result"]
    frame_records = []
    for local_frame_index, record in enumerate(range_result["frame_records"]):
        src_png = Path(record["png_path"])
        src_exr = Path(record["exr_path"])
        src_pose = Path(record["pose_json_path"])
        global_prefix = f"{sequence_id}__{src_png.stem}"
        dst_png = images_dir / f"{global_prefix}.png"
        dst_exr = images_dir / f"{global_prefix}.exr"
        dst_pose = metadata_frames_dir / f"{global_prefix}.json"
        _safe_symlink_or_copy(src_png, dst_png)
        if src_exr.exists():
            _safe_symlink_or_copy(src_exr, dst_exr)
        _safe_symlink_or_copy(src_pose, dst_pose)
        frame_records.append(
            {
                "sequence_id": sequence_id,
                "sequence_name": sequence_manifest["sequence_name"],
                "local_frame_index": int(local_frame_index),
                "source_frame_record": record,
                "global_png_name": dst_png.name,
                "global_exr_name": dst_exr.name if dst_exr.exists() else None,
                "global_pose_name": dst_pose.name,
                "pose_json_path": str(dst_pose),
            }
        )

    preview_status = range_result.get("preview_mp4_status") or {}
    preview_mp4_path = preview_status.get("mp4_path") or range_result.get("preview_mp4_path")
    if preview_mp4_path and Path(preview_mp4_path).exists():
        dst_mp4 = previews_dir / f"{sequence_id}_preview.mp4"
        _safe_symlink_or_copy(preview_mp4_path, dst_mp4)
    return frame_records


def _material_report_ok(report):
    if not isinstance(report, dict):
        return True
    for key, value in report.items():
        if isinstance(value, dict) and value.get("requested") and not value.get("applied", False):
            return False
    return True


def _extract_body_quality_meta(appearance_metadata, body_spec):
    checks = {
        "body_material_requested": bool(body_spec.get("texture_body")),
        "body_material_applied": True,
        "hair_requested": bool(body_spec.get("hair")),
        "hair_applied": True,
        "clothing_requested": bool(body_spec.get("texture_clothing") or body_spec.get("texture_clothing_overlay")),
        "clothing_applied": True,
        "clothing_metadata_supported": True,
        "shoe_requested": bool(body_spec.get("shoe")),
        "shoe_applied": True,
    }
    body_material = (appearance_metadata or {}).get("body_material") or {}
    hair = (appearance_metadata or {}).get("hair") or {}
    clothing = (appearance_metadata or {}).get("clothing") or {}
    shoe = (appearance_metadata or {}).get("shoe") or {}
    if checks["body_material_requested"]:
        checks["body_material_applied"] = bool(body_material.get("applied"))
    if checks["hair_requested"]:
        checks["hair_applied"] = bool(hair.get("applied"))
    if checks["clothing_requested"]:
        if clothing.get("supported") is False and clothing.get("reason") == "not_implemented_yet":
            checks["clothing_metadata_supported"] = False
            checks["clothing_applied"] = True
        else:
            checks["clothing_applied"] = bool(clothing.get("applied"))
    if checks["shoe_requested"]:
        checks["shoe_applied"] = bool(shoe.get("applied"))
    checks["appearance_success"] = all(
        checks[key]
        for key in (
            "body_material_applied",
            "hair_applied",
            "clothing_applied",
            "shoe_applied",
        )
        if checks[key.replace("_applied", "_requested")]  # type: ignore[index]
    ) if any(
        checks[key] for key in ("body_material_requested", "hair_requested", "clothing_requested", "shoe_requested")
    ) else True
    return checks


def _camera_body_distance_cm(camera_pose, body_row):
    origin = (body_row.get("bounds") or {}).get("origin") or {}
    radius_cm = float(((body_row.get("bounds") or {}).get("radius_cm")) or 0.0)
    dx = float(origin.get("x", 0.0)) - float(camera_pose["x"])
    dy = float(origin.get("y", 0.0)) - float(camera_pose["y"])
    dz = float(origin.get("z", 0.0)) - float(camera_pose["z"])
    distance_cm = math.sqrt(dx * dx + dy * dy + dz * dz)
    return distance_cm, max(0.0, distance_cm - radius_cm), radius_cm


def _refresh_has_error(refresh_records):
    for record in refresh_records or []:
        for group in ("component_methods", "actor_methods"):
            for value in (record.get(group) or {}).values():
                if isinstance(value, str) and value.startswith("error:"):
                    return True
    return False


def _alignment_rows_by_frame_body(alignment_report):
    mapping = {}
    for row in alignment_report.get("frames", []):
        mapping[(int(row["frame_index"]), int(row["body_index"]))] = row
    return mapping


def _sequence_spec_by_id(render_manifest):
    return {item["sequence_id"]: item for item in render_manifest.get("sequence_specs", [])}


def _frame_overlay_path(raw_root, frame_meta):
    stem = Path(frame_meta["image_png"]).stem
    return raw_root / "projections2d" / "erp_alignment" / "overlays" / f"{stem}_joints_overlay.png"


def _gt_index_summary(frame_map):
    parts = []
    for item in frame_map.get("body_frame_mapping") or []:
        gt = item.get("ground_truth") or {}
        asset_id = item.get("asset_id", "body")
        npz_idx = gt.get("npz_frame_index")
        if npz_idx is not None:
            parts.append(f"{asset_id}:{int(npz_idx)}")
    return "|".join(parts)


def _write_dataset_summary_exports(run_root, raw_root, rows, frames_json, frame_mapping, render_manifest, dataset_manifest):
    run_root = Path(run_root)
    raw_root = Path(raw_root)
    seq_specs = _sequence_spec_by_id(render_manifest)
    frames_by_index = {int(item["frame_index"]): item for item in frames_json}
    frame_map_by_index = {int(item["frame_index"]): item for item in frame_mapping}

    warning_rows = []
    clean_rows = []
    warning_reason_counts = {}
    distance_values = []
    camera_policy_counts = {}
    distance_regime_counts = {}
    unique_scene_signatures = set()
    unique_motion_triplets = set()

    for spec in render_manifest.get("sequence_specs", []):
        camera_policy = spec.get("camera_policy")
        distance_regime = spec.get("distance_regime")
        scene_signature = spec.get("scene_signature")
        motion_triplet_signature = spec.get("motion_triplet_signature")
        if camera_policy:
            camera_policy_counts[camera_policy] = camera_policy_counts.get(camera_policy, 0) + 1
        if distance_regime:
            distance_regime_counts[distance_regime] = distance_regime_counts.get(distance_regime, 0) + 1
        if scene_signature:
            unique_scene_signatures.add(scene_signature)
        if motion_triplet_signature:
            unique_motion_triplets.add(motion_triplet_signature)

    for row in rows:
        frame_index = int(row["frame_index"])
        frame_meta = frames_by_index.get(frame_index, {})
        frame_map = frame_map_by_index.get(frame_index, {})
        overlay_path = _frame_overlay_path(raw_root, frame_meta)
        image_path = raw_root / "images" / str(frame_meta.get("image_png", ""))
        warning_reasons = [part for part in str(row.get("warning_reasons", "")).split("|") if part]
        for reason in warning_reasons:
            warning_reason_counts[reason] = warning_reason_counts.get(reason, 0) + 1
        min_distance = row.get("min_joint_distance_to_camera_cm")
        if min_distance not in ("", None):
            try:
                distance_values.append(float(min_distance))
            except Exception:
                pass
        spec = seq_specs.get(row["sequence_id"], {})
        base = {
            "sequence_id": row["sequence_id"],
            "sequence_name": row["sequence_name"],
            "frame_index": frame_index,
            "sequence_local_frame_index": int(row["sequence_local_frame_index"]),
            "timeline_frame_index": int(row["timeline_frame_index"]),
            "image_path": str(image_path),
            "overlay_path": str(overlay_path),
            "warning_reasons": "|".join(warning_reasons),
            "camera_policy": spec.get("camera_policy"),
            "distance_regime": spec.get("distance_regime"),
            "scene_signature": spec.get("scene_signature"),
            "motion_triplet_signature": spec.get("motion_triplet_signature"),
            "gt_frame_mapping_summary": _gt_index_summary(frame_map),
        }
        if row.get("warning_classification") == "accepted_warning":
            warning_rows.append(
                {
                    **base,
                    "joints_near_seam_count": int(row.get("joints_near_seam_count") or 0),
                    "joints_near_pole_count": int(row.get("joints_near_pole_count") or 0),
                    "min_joint_distance_to_camera_cm": row.get("min_joint_distance_to_camera_cm"),
                    "max_abs_latitude_deg": row.get("max_abs_latitude_deg"),
                }
            )
        elif row.get("warning_classification") == "accepted":
            clean_row = dict(base)
            clean_row.pop("warning_reasons", None)
            clean_rows.append(clean_row)

    warning_csv = run_root / "warning_frames.csv"
    clean_csv = run_root / "clean_frames.csv"
    _write_csv(
        warning_csv,
        warning_rows,
        fieldnames=[
            "sequence_id",
            "sequence_name",
            "frame_index",
            "sequence_local_frame_index",
            "timeline_frame_index",
            "image_path",
            "overlay_path",
            "warning_reasons",
            "joints_near_seam_count",
            "joints_near_pole_count",
            "min_joint_distance_to_camera_cm",
            "max_abs_latitude_deg",
            "camera_policy",
            "distance_regime",
            "scene_signature",
            "motion_triplet_signature",
            "gt_frame_mapping_summary",
        ],
    )
    _write_csv(
        clean_csv,
        clean_rows,
        fieldnames=[
            "sequence_id",
            "sequence_name",
            "frame_index",
            "sequence_local_frame_index",
            "timeline_frame_index",
            "image_path",
            "overlay_path",
            "camera_policy",
            "distance_regime",
            "scene_signature",
            "motion_triplet_signature",
            "gt_frame_mapping_summary",
        ],
    )

    distance_summary = {
        "min_camera_body_distance_cm": None,
        "mean_camera_body_distance_cm": None,
        "max_camera_body_distance_cm": None,
    }
    if distance_values:
        distance_summary = {
            "min_camera_body_distance_cm": float(min(distance_values)),
            "mean_camera_body_distance_cm": float(sum(distance_values) / float(len(distance_values))),
            "max_camera_body_distance_cm": float(max(distance_values)),
        }

    summary = {
        "total_frames": int(len(rows)),
        "accepted_clean": int(sum(1 for row in rows if row.get("warning_classification") == "accepted")),
        "accepted_warning": int(sum(1 for row in rows if row.get("warning_classification") == "accepted_warning")),
        "rejected": int(sum(1 for row in rows if row.get("warning_classification") == "rejected")),
        "warning_reason_counts": warning_reason_counts,
        "projection_mode": dataset_manifest.get("erp_projection_mode"),
        "erp_yaw_zero_offset_deg_used": dataset_manifest.get("erp_yaw_zero_offset_deg_used"),
        "camera_policy_distribution": camera_policy_counts,
        "distance_regime_distribution": distance_regime_counts,
        "unique_scene_count": int(len(unique_scene_signatures)),
        "unique_motion_triplet_count": int(len(unique_motion_triplets)),
        **distance_summary,
        "warning_frames_csv": str(warning_csv),
        "clean_frames_csv": str(clean_csv),
    }
    html_summary_path = write_dataset_html_summary(run_root / "dataset_summary.html", summary, rows)
    summary["html_summary_path"] = str(html_summary_path)
    summary_path = run_root / "dataset_summary.json"
    _write_json(summary_path, summary)
    return summary_path, warning_csv, clean_csv, html_summary_path


def _build_quality_report(raw_root, dataset_manifest, render_manifest, sequence_manifests, output_csv_path):
    raw_root = Path(raw_root)
    frames_json = _read_json(raw_root / "metadata" / "frames.json")
    frame_mapping = _read_json(raw_root / "metadata" / "frame_mapping.json")
    alignment_report_path = raw_root / "projections2d" / "erp_alignment" / "alignment_report.json"
    alignment_report = {"frames": []} if not alignment_report_path.is_file() else _read_json(alignment_report_path)
    alignment_map = _alignment_rows_by_frame_body(alignment_report)
    sequence_spawned_meta = {}
    sequence_body_specs = {}
    for sequence_manifest in sequence_manifests:
        sequence_spawned_meta[sequence_manifest["sequence_id"]] = {
            item["asset_id"]: item.get("appearance_metadata") or {}
            for item in sequence_manifest.get("spawned_roles", [])
        }
        sequence_body_specs[sequence_manifest["sequence_id"]] = list(sequence_manifest.get("body_specs", []))

    rows = []
    for frame_meta, frame_map in zip(frames_json, frame_mapping):
        pose_json = _read_json(frame_meta["pose_json_path"])
        sequence_id = frame_meta["sequence_id"]
        body_rows = pose_json.get("body_evaluations") or []
        body_rows_by_asset = {row["asset_id"]: row for row in body_rows}

        appearance_success = True
        projected_mismatch = False
        cropped_body = False
        camera_intersection = False
        extreme_proximity = False
        geometrycache_warning = False
        material_warning = False
        min_surface_clearance_cm = None
        projected_joint_min = None
        max_abs_latitude_deg = None
        joints_near_seam_count = 0
        joints_near_pole_count = 0
        min_joint_distance_to_camera_cm = None
        close_body_flag = False
        extreme_erp_distortion_flag = False
        warning_reasons = []
        body_material_success = True
        clothing_success = True
        hair_success = True
        shoe_success = True
        projected_joints_valid = True
        projected_joints_inside_image = True
        rejection_reasons = []

        current_body_specs = sequence_body_specs.get(sequence_id) or []
        for body_index, body_spec in enumerate(current_body_specs):
            asset_id = body_spec["asset_id"]
            body_row = body_rows_by_asset.get(asset_id, {})
            appearance_meta = (sequence_spawned_meta.get(sequence_id) or {}).get(asset_id) or {}
            appearance_checks = _extract_body_quality_meta(appearance_meta, body_spec)
            appearance_success = appearance_success and bool(appearance_checks["appearance_success"])
            body_material_success = body_material_success and bool(appearance_checks["body_material_applied"])
            clothing_success = clothing_success and bool(appearance_checks["clothing_applied"])
            hair_success = hair_success and bool(appearance_checks["hair_applied"])
            shoe_success = shoe_success and bool(appearance_checks["shoe_applied"])
            distance_cm, clearance_cm, radius_cm = _camera_body_distance_cm(pose_json["camera_pose_cm_deg"], body_row)
            min_surface_clearance_cm = clearance_cm if min_surface_clearance_cm is None else min(min_surface_clearance_cm, clearance_cm)
            camera_intersection = camera_intersection or (distance_cm <= radius_cm + 1e-4)
            extreme_proximity = extreme_proximity or (clearance_cm < 35.0)
            if body_row.get("sample_frame_index") is None or body_row.get("section_end_frame") is None:
                geometrycache_warning = True

            alignment_row = alignment_map.get((int(frame_meta["frame_index"]), int(body_index))) or {}
            reasons = alignment_row.get("suspected_coordinate_mismatch") or []
            if isinstance(reasons, str):
                reasons = [part for part in reasons.split("|") if part]
            projected_mismatch = projected_mismatch or bool(reasons)
            joint_count = int(alignment_row.get("projected_joint_count") or 0)
            projected_joints_valid = projected_joints_valid and bool(alignment_row) and joint_count >= 32
            projected_joint_min = joint_count if projected_joint_min is None else min(projected_joint_min, joint_count)
            row_max_abs_lat = alignment_row.get("max_abs_latitude_deg")
            if row_max_abs_lat is not None:
                row_max_abs_lat = float(row_max_abs_lat)
                max_abs_latitude_deg = row_max_abs_lat if max_abs_latitude_deg is None else max(max_abs_latitude_deg, row_max_abs_lat)
            joints_near_seam_count += int(alignment_row.get("joints_near_seam_count") or 0)
            joints_near_pole_count += int(alignment_row.get("joints_near_pole_count") or 0)
            row_min_joint_distance = alignment_row.get("min_joint_distance_to_camera_cm")
            if row_min_joint_distance is not None:
                row_min_joint_distance = float(row_min_joint_distance)
                min_joint_distance_to_camera_cm = (
                    row_min_joint_distance
                    if min_joint_distance_to_camera_cm is None
                    else min(min_joint_distance_to_camera_cm, row_min_joint_distance)
                )
            close_body_flag = close_body_flag or bool(alignment_row.get("close_body_flag"))
            extreme_erp_distortion_flag = extreme_erp_distortion_flag or bool(alignment_row.get("extreme_erp_distortion_flag"))
            row_warning_labels = alignment_row.get("warning_labels") or []
            if isinstance(row_warning_labels, str):
                row_warning_labels = [part for part in row_warning_labels.split("|") if part]
            for label in row_warning_labels:
                if label not in warning_reasons:
                    warning_reasons.append(label)
            bbox_min_y = alignment_row.get("bbox_min_y")
            bbox_max_y = alignment_row.get("bbox_max_y")
            bbox_min_x = alignment_row.get("bbox_min_x")
            bbox_max_x = alignment_row.get("bbox_max_x")
            image_h = dataset_manifest["image_size"]["height"]
            image_w = dataset_manifest["image_size"]["width"]
            if alignment_row:
                if joint_count < 32:
                    cropped_body = True
                if bbox_min_y is not None and float(bbox_min_y) <= 2.0:
                    cropped_body = True
                if image_h is not None and bbox_max_y is not None and float(bbox_max_y) >= float(image_h - 3):
                    cropped_body = True
                if bbox_min_x is not None and float(bbox_min_x) <= 1.0:
                    projected_joints_inside_image = False
                if image_w is not None and bbox_max_x is not None and float(bbox_max_x) >= float(image_w - 2):
                    projected_joints_inside_image = False

        for report in pose_json.get("post_warmup_material_reports") or []:
            resolved = report.get("resolved_body_material_path")
            current = ((report.get("current_component_material") or {}).get("material_path"))
            if resolved and current and resolved != current:
                material_warning = True

        render_success = bool(pose_json.get("hdr_ok")) or bool(pose_json.get("exr_ok"))
        preview_png_ok = bool(pose_json.get("preview_png_ok"))
        geometrycache_warning = geometrycache_warning or _refresh_has_error(pose_json.get("geometrycache_render_refresh"))
        severe_projection_failure = (not projected_joints_valid) or (not projected_joints_inside_image and joints_near_seam_count == 0)
        rejected = (
            (not render_success)
            or (not preview_png_ok)
            or (not body_material_success)
            or (not hair_success)
            or (not shoe_success)
            or camera_intersection
            or severe_projection_failure
            or cropped_body
        )
        accepted = not rejected
        if not render_success:
            rejection_reasons.append("render_failure")
        if not preview_png_ok:
            rejection_reasons.append("preview_failure")
        if not body_material_success:
            rejection_reasons.append("body_material_failure")
        if not clothing_success:
            warning_reasons.append("clothing_metadata_failure")
        if not hair_success:
            rejection_reasons.append("hair_failure")
        if not shoe_success:
            rejection_reasons.append("shoe_failure")
        if camera_intersection:
            rejection_reasons.append("camera_body_intersection")
        if extreme_proximity:
            if "extreme-close-risk" not in warning_reasons:
                warning_reasons.append("extreme-close-risk")
        if not projected_joints_valid:
            rejection_reasons.append("projected_joints_invalid")
        if not projected_joints_inside_image:
            if joints_near_seam_count > 0:
                if "seam-risk" not in warning_reasons:
                    warning_reasons.append("seam-risk")
            else:
                rejection_reasons.append("projected_joints_outside_image")
        if cropped_body:
            rejection_reasons.append("missing_or_cropped_body")
        if material_warning:
            if "material_warning" not in warning_reasons:
                warning_reasons.append("material_warning")
        if extreme_erp_distortion_flag and "globally_aligned_but_distorted" not in warning_reasons:
            warning_reasons.append("globally_aligned_but_distorted")
        warning_classification = "accepted"
        if rejected:
            warning_classification = "rejected"
        elif warning_reasons:
            warning_classification = "accepted_warning"
        rows.append(
            {
                "frame_index": int(frame_meta["frame_index"]),
                "sequence_id": sequence_id,
                "sequence_name": frame_meta["sequence_name"],
                "sequence_local_frame_index": int(frame_meta["sequence_local_frame_index"]),
                "timeline_frame_index": int(frame_meta["timeline_frame_index"]),
                "image_png": frame_meta["image_png"],
                "render_success": bool(render_success),
                "preview_png_ok": bool(preview_png_ok),
                "hdr_ok": bool(pose_json.get("hdr_ok")),
                "exr_ok": bool(pose_json.get("exr_ok")),
                "appearance_success": bool(appearance_success),
                "body_material_success": bool(body_material_success),
                "clothing_success": bool(clothing_success),
                "hair_success": bool(hair_success),
                "shoe_success": bool(shoe_success),
                "camera_body_intersection": bool(camera_intersection),
                "extreme_close": bool(extreme_proximity),
                "min_surface_clearance_cm": None if min_surface_clearance_cm is None else float(min_surface_clearance_cm),
                "projected_joints_valid": bool(projected_joints_valid),
                "projected_joints_inside_image": bool(projected_joints_inside_image),
                "missing_or_cropped_body_heuristic": bool(cropped_body),
                "projected_joint_min": None if projected_joint_min is None else int(projected_joint_min),
                "max_abs_latitude_deg": None if max_abs_latitude_deg is None else float(max_abs_latitude_deg),
                "joints_near_seam_count": int(joints_near_seam_count),
                "joints_near_pole_count": int(joints_near_pole_count),
                "min_joint_distance_to_camera_cm": None if min_joint_distance_to_camera_cm is None else float(min_joint_distance_to_camera_cm),
                "close_body_flag": bool(close_body_flag),
                "extreme_erp_distortion_flag": bool(extreme_erp_distortion_flag),
                "geometrycache_warning": bool(geometrycache_warning),
                "material_warning": bool(material_warning),
                "movement_score": float(pose_json.get("movement_score") or 0.0),
                "warning_classification": warning_classification,
                "warning_reasons": "|".join(warning_reasons),
                "accepted": bool(accepted),
                "rejection_reasons": "|".join(rejection_reasons),
            }
        )

    csv_path = Path(output_csv_path)
    _write_csv(
        csv_path,
        rows,
        fieldnames=[
            "frame_index",
            "sequence_id",
            "sequence_name",
            "sequence_local_frame_index",
            "timeline_frame_index",
            "image_png",
            "render_success",
            "preview_png_ok",
            "hdr_ok",
            "exr_ok",
            "appearance_success",
            "body_material_success",
            "clothing_success",
            "hair_success",
            "shoe_success",
            "camera_body_intersection",
            "extreme_close",
            "min_surface_clearance_cm",
            "projected_joints_valid",
            "projected_joints_inside_image",
            "missing_or_cropped_body_heuristic",
            "projected_joint_min",
            "max_abs_latitude_deg",
            "joints_near_seam_count",
            "joints_near_pole_count",
            "min_joint_distance_to_camera_cm",
            "close_body_flag",
            "extreme_erp_distortion_flag",
            "geometrycache_warning",
            "material_warning",
            "movement_score",
            "warning_classification",
            "warning_reasons",
            "accepted",
            "rejection_reasons",
        ],
    )
    rejection_reason_counts = {}
    for row in rows:
        for reason in str(row["rejection_reasons"]).split("|"):
            if not reason:
                continue
            rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
    summary = {
        "frame_count": int(len(rows)),
        "render_success_count": int(sum(1 for row in rows if row["render_success"])),
        "appearance_success_count": int(sum(1 for row in rows if row["appearance_success"])),
        "accepted_frame_count": int(sum(1 for row in rows if row["accepted"])),
        "accepted_warning_frame_count": int(sum(1 for row in rows if row["warning_classification"] == "accepted_warning")),
        "rejected_frame_count": int(sum(1 for row in rows if not row["accepted"])),
        "camera_intersection_count": int(sum(1 for row in rows if row["camera_body_intersection"])),
        "extreme_close_count": int(sum(1 for row in rows if row["extreme_close"])),
        "extreme_erp_distortion_count": int(sum(1 for row in rows if row["extreme_erp_distortion_flag"])),
        "seam_risk_frame_count": int(sum(1 for row in rows if int(row["joints_near_seam_count"]) > 0)),
        "pole_risk_frame_count": int(sum(1 for row in rows if int(row["joints_near_pole_count"]) > 0)),
        "rejection_reason_counts": rejection_reason_counts,
    }
    _write_json(raw_root / "metadata" / "quality_summary.json", summary)
    return csv_path, summary


def postprocess_v0_dataset(
    run_root,
    extra_npz_roots=None,
    smplx_model_roots=None,
    erp_projection_mode="camera_rotation_aware",
    erp_post_rotation_config=None,
):
    np, benchmark_export, gt_alignment, preview_tools = _load_postprocess_modules()
    run_root = Path(run_root)
    render_manifest, sequence_manifests = _load_sequence_manifests(run_root)
    raw_root = _ensure_dir(run_root / "raw")
    images_dir = _ensure_dir(raw_root / "images")
    metadata_dir = _ensure_dir(raw_root / "metadata")
    smplx_dir = _ensure_dir(raw_root / "smplx")
    joints3d_dir = _ensure_dir(raw_root / "joints3d")
    vertices_dir = _ensure_dir(raw_root / "vertices")
    projections2d_dir = _ensure_dir(raw_root / "projections2d")
    previews_dir = _ensure_dir(run_root / "previews")
    _ensure_dir(metadata_dir / "frames")

    sequence_body_specs_by_id = {manifest["sequence_id"]: list(manifest["body_specs"]) for manifest in sequence_manifests}
    unique_asset_ids = []
    for seq_specs in sequence_body_specs_by_id.values():
        for item in seq_specs:
            if item["asset_id"] not in unique_asset_ids:
                unique_asset_ids.append(item["asset_id"])
    num_bodies = len(next(iter(sequence_body_specs_by_id.values())))
    resolved_assets = benchmark_export._resolved_asset_records(unique_asset_ids)
    npz_report, npz_scanned_roots = benchmark_export._select_npz_paths(
        unique_asset_ids,
        benchmark_export._npz_search_roots(extra_roots=extra_npz_roots),
    )
    npz_payloads = {}
    for asset_id, report in npz_report.items():
        if report["selected_path"]:
            npz_payloads[asset_id] = benchmark_export._extract_npz_payload(
                benchmark_export._load_npz(report["selected_path"])
            )
    runtime_status = benchmark_export._try_import_smplx_runtime()
    model_scan = benchmark_export._scan_smplx_model_files(extra_roots=smplx_model_roots)

    global_frame_records = []
    for sequence_manifest in sequence_manifests:
        global_frame_records.extend(_copy_render_outputs(sequence_manifest, raw_root))

        preview_mp4 = sequence_manifest["range_result"].get("preview_mp4_path")
        if preview_mp4 and Path(preview_mp4).exists():
            dst_preview = previews_dir / f"{sequence_manifest['sequence_id']}_preview.mp4"
            _safe_symlink_or_copy(preview_mp4, dst_preview)

    num_frames = len(global_frame_records)
    max_poses_dim = max((payload["poses"].shape[1] for payload in npz_payloads.values()), default=0)
    global_orient_slice, body_pose_slice = benchmark_export._infer_pose_slices(max_poses_dim)
    global_orient = np.full((num_frames, num_bodies, global_orient_slice.stop - global_orient_slice.start), np.nan, dtype=np.float32)
    body_pose = np.full((num_frames, num_bodies, body_pose_slice.stop - body_pose_slice.start), np.nan, dtype=np.float32)
    expression_dim = max(
        (0 if payload["expression"] is None else int(payload["expression"].shape[-1]) for payload in npz_payloads.values()),
        default=0,
    )
    expression = (
        np.full((num_frames, num_bodies, expression_dim), np.nan, dtype=np.float32)
        if expression_dim > 0
        else np.empty((num_frames, num_bodies, 0), dtype=np.float32)
    )
    max_betas_dim = max((payload["betas"].shape[0] for payload in npz_payloads.values()), default=0)
    betas = np.full((num_frames, num_bodies, max_betas_dim), np.nan, dtype=np.float32) if max_betas_dim > 0 else np.empty((num_frames, num_bodies, 0), dtype=np.float32)
    genders = np.array([["unknown"] * num_bodies for _ in range(num_frames)], dtype=object)

    trans_world = np.full((num_frames, num_bodies, 3), np.nan, dtype=np.float32)
    available = np.zeros((num_frames, num_bodies), dtype=bool)
    camera_world = np.zeros((num_frames, 4, 4), dtype=np.float32)
    body_world = np.zeros((num_frames, num_bodies, 4, 4), dtype=np.float32)
    frame_mapping = []
    frames_json = []

    per_sequence_body_rows = {}
    for sequence_manifest in sequence_manifests:
        range_result = sequence_manifest["range_result"]
        per_body_rows = list(csv.DictReader(open(range_result["per_body_csv_path"], "r", encoding="utf-8")))
        per_sequence_body_rows[sequence_manifest["sequence_id"]] = {
            (int(row["frame_index"]), row["asset_id"]): row
            for row in per_body_rows
        }

    for global_frame_index, item in enumerate(global_frame_records):
        pose = _read_json(item["pose_json_path"])
        sequence_id = item["sequence_id"]
        sequence_manifest = next(man for man in sequence_manifests if man["sequence_id"] == sequence_id)
        current_body_specs = sequence_body_specs_by_id[sequence_id]
        body_rows_by_key = per_sequence_body_rows[sequence_id]
        cam = benchmark_export._camera_record_from_pose(pose)
        camera_world[global_frame_index] = benchmark_export._make_transform_matrix(
            cam["x_cm"], cam["y_cm"], cam["z_cm"], cam["yaw_deg"], cam["pitch_deg"], cam["roll_deg"]
        )

        frame_body_map = []
        for body_i, spec in enumerate(current_body_specs):
            asset_id = spec["asset_id"]
            body_world[global_frame_index, body_i] = benchmark_export._make_transform_matrix(
                float(spec["x"]),
                float(spec["y"]),
                float(spec["z"]),
                float(spec["yaw"]),
                float(spec["pitch"]),
                float(spec["roll"]),
            )
            row = body_rows_by_key.get((int(pose["timeline_frame_index"]), asset_id), {})
            npz_frame_index = int(pose["timeline_frame_index"]) + 1
            gt_status = {
                "npz_frame_index": int(npz_frame_index),
                "npz_available": False,
                "npz_path": npz_report[asset_id]["selected_path"],
                "npz_frame_valid": False,
            }
            payload = npz_payloads.get(asset_id)
            if payload is not None:
                poses_arr = payload["poses"]
                trans_arr = payload["trans"]
                beta_take = min(betas.shape[2], payload["betas"].shape[0]) if betas.ndim == 3 else 0
                if beta_take > 0:
                    betas[global_frame_index, body_i, :beta_take] = payload["betas"][:beta_take]
                if payload["gender"] is not None:
                    genders[global_frame_index, body_i] = payload["gender"]
                if 0 <= npz_frame_index < poses_arr.shape[0] and 0 <= npz_frame_index < trans_arr.shape[0]:
                    pose_vec = poses_arr[npz_frame_index]
                    trans_vec = trans_arr[npz_frame_index]
                    go_take = min(global_orient.shape[-1], pose_vec[global_orient_slice].shape[0])
                    bp_take = min(body_pose.shape[-1], pose_vec[body_pose_slice].shape[0])
                    if go_take > 0:
                        global_orient[global_frame_index, body_i, :go_take] = pose_vec[global_orient_slice][:go_take]
                    if bp_take > 0:
                        body_pose[global_frame_index, body_i, :bp_take] = pose_vec[body_pose_slice][:bp_take]
                    trans_world[global_frame_index, body_i] = trans_vec[:3]
                    if expression_dim > 0 and payload["expression"] is not None and npz_frame_index < payload["expression"].shape[0]:
                        exp_vec = payload["expression"][npz_frame_index]
                        take_exp = min(expression_dim, exp_vec.shape[0])
                        expression[global_frame_index, body_i, :take_exp] = exp_vec[:take_exp]
                    available[global_frame_index, body_i] = True
                    gt_status["npz_available"] = True
                    gt_status["npz_frame_valid"] = True
                gt_status["npz_num_frames"] = int(payload["poses"].shape[0])
                gt_status["npz_poses_dim"] = int(payload["poses"].shape[1])

            frame_body_map.append(
                {
                    "asset_id": asset_id,
                    "sequence_id": sequence_id,
                    "sequence_timeline_frame_index": int(pose["timeline_frame_index"]),
                    "sequence_timeline_time_seconds": float(pose["timeline_time_seconds"]),
                    "geometrycache_local_time_seconds": None if row.get("sample_time_seconds", "") == "" else float(row["sample_time_seconds"]),
                    "geometrycache_local_frame_index": None if row.get("sample_frame_index", "") == "" else int(row["sample_frame_index"]),
                    "geometrycache_section_end_frame": None if row.get("section_end_frame", "") == "" else int(float(row["section_end_frame"])),
                    "geometrycache_section_play_rate": None if row.get("section_play_rate", "") == "" else float(row["section_play_rate"]),
                    "ground_truth": gt_status,
                }
            )

        frame_mapping.append(
            {
                "frame_index": int(global_frame_index),
                "sequence_id": sequence_id,
                "sequence_name": item["sequence_name"],
                "sequence_local_frame_index": int(item["local_frame_index"]),
                "frame_name": pose["frame_name"],
                "source_pose_json_path": item["source_frame_record"]["pose_json_path"],
                "source_png_path": item["source_frame_record"]["png_path"],
                "source_exr_path": item["source_frame_record"]["exr_path"],
                "camera_pose_cm_deg": pose["camera_pose_cm_deg"],
                "body_frame_mapping": frame_body_map,
            }
        )
        frames_json.append(
            {
                "frame_index": int(global_frame_index),
                "sequence_id": sequence_id,
                "sequence_name": item["sequence_name"],
                "sequence_local_frame_index": int(item["local_frame_index"]),
                "frame_name": pose["frame_name"],
                "timeline_frame_index": int(pose["timeline_frame_index"]),
                "timeline_time_seconds": float(pose["timeline_time_seconds"]),
                "image_png": item["global_png_name"],
                "image_exr": item["global_exr_name"],
                "pose_json_path": item["pose_json_path"],
            }
        )

    np.save(metadata_dir / "camera_world_transforms.npy", camera_world)
    np.save(metadata_dir / "body_world_transforms.npy", body_world)
    _write_json(metadata_dir / "frame_mapping.json", frame_mapping)
    _write_json(metadata_dir / "frames.json", frames_json)

    smplx_forward = benchmark_export._forward_smplx_if_available(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        transl_world=trans_world,
        genders=genders,
        runtime_status=runtime_status,
        model_scan=model_scan,
        body_labels=[f"body_slot_{i}" for i in range(num_bodies)],
        available_mask=available,
    )

    np.savez_compressed(
        smplx_dir / "parameters.npz",
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        transl_world=trans_world,
        expression=expression,
        genders=genders,
        available=available,
    )
    if smplx_forward["available"]:
        np.savez_compressed(joints3d_dir / "joints3d.npz", joints3d=smplx_forward["joints3d"], available=available)
        np.savez_compressed(vertices_dir / "vertices.npz", vertices=smplx_forward["vertices"], available=available)
        smplx_saved_vs_direct = benchmark_export.validate_saved_vs_direct_smplx(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl_world=trans_world,
            genders=genders,
            saved_joints3d=smplx_forward["joints3d"],
            saved_vertices=smplx_forward["vertices"],
            available=available,
            runtime_status=runtime_status,
            model_scan=model_scan,
            preferred_frame_index=129 if num_frames > 129 else None,
            preferred_body_index=1 if num_bodies > 1 else None,
        )
    else:
        np.savez_compressed(
            joints3d_dir / "joints3d.npz",
            joints3d=np.empty((num_frames, num_bodies, 0, 3), dtype=np.float32),
            available=np.zeros((num_frames, num_bodies), dtype=bool),
        )
        np.savez_compressed(
            vertices_dir / "vertices.npz",
            vertices=np.empty((num_frames, num_bodies, 0, 3), dtype=np.float32),
            available=np.zeros((num_frames, num_bodies), dtype=bool),
        )
        smplx_saved_vs_direct = {
            "available": False,
            "reason": "SMPL-X forward pass unavailable.",
        }
    np.savez_compressed(
        projections2d_dir / "projections2d_erp.npz",
        joints2d=np.empty((num_frames, num_bodies, 0, 2), dtype=np.float32),
        vertices2d=np.empty((num_frames, num_bodies, 0, 2), dtype=np.float32),
        available=np.zeros((num_frames, num_bodies), dtype=bool),
    )

    missing_assets = [asset_id for asset_id, report in npz_report.items() if not report["present"]]
    gt_unavailable_reason = (
        "SMPL-X joints/vertices are unavailable until BEDLAM2 motion NPZs and optional SMPL-X model files are provided."
    )
    unavailable_hits = np.argwhere(~available)
    first_unavailable = None
    if unavailable_hits.size:
        fi, bi = (int(unavailable_hits[0][0]), int(unavailable_hits[0][1]))
        first_unavailable = {
            "frame_index": fi,
            "body_index": bi,
            "asset_id": unique_asset_ids[bi] if bi < len(unique_asset_ids) else f"body_slot_{bi}",
            "reason": "NPZ out of range",
        }
    gt_availability_report = {
        "configured_npz_root": str(benchmark_export.DEFAULT_BEDLAM2_NPZ_ROOT),
        "scanned_roots": npz_scanned_roots,
        "assets": list(npz_report.values()),
        "missing_assets": missing_assets,
        "download_hint": benchmark_export.DOWNLOAD_HINT,
        "smplx_runtime": {
            "python_runtime_available": runtime_status["python_runtime_available"],
            "smplx_importable": runtime_status["smplx_importable"],
            "torch_importable": runtime_status["torch_importable"],
            "smplx_version": runtime_status["smplx_version"],
            "torch_version": runtime_status["torch_version"],
            "errors": runtime_status["errors"],
        },
        "smplx_model_scan": model_scan,
        "smplx_forward": {
            "available": smplx_forward["available"],
            "reason": smplx_forward["reason"],
            "joint_count": smplx_forward["joint_count"],
            "vertex_count": smplx_forward["vertex_count"],
            "traceback": smplx_forward.get("traceback"),
        },
        "smplx_saved_vs_direct_validation": smplx_saved_vs_direct,
        "availability_summary": {
            "valid_frame_body_count": int(np.count_nonzero(available)),
            "unavailable_frame_body_count": int(available.size - np.count_nonzero(available)),
            "first_unavailable": first_unavailable,
            "gt_forward_succeeded_for_available_frames": bool(smplx_forward["available"]),
        },
    }
    _write_json(metadata_dir / "gt_availability_report.json", gt_availability_report)

    convention = gt_alignment._load_erp_projection_convention(run_root)
    if convention is not None:
        _write_json(metadata_dir / "erp_projection_convention.json", convention["payload"])

    source_asset_records = []
    for asset_id in unique_asset_ids:
        resolved = resolved_assets[asset_id]
        source_asset_records.append(
            {
                "asset_id": asset_id,
                "resolved_asset_id": resolved.get("resolved_asset_id"),
                "asset_class": resolved.get("asset_class"),
                "local_animation_asset_path": resolved.get("local_asset_path"),
                "unreal_animation_asset_path": resolved.get("unreal_asset_path"),
                "unreal_geometrycache_asset_path": resolved.get("body_geometry_cache_path"),
                "body_geometry_cache_source": resolved.get("body_geometry_cache_source"),
                "motion_npz_selected_path": npz_report[asset_id]["selected_path"],
                "motion_npz_present": npz_report[asset_id]["present"],
                "smplx_ground_truth_available": bool(npz_report[asset_id]["present"]),
                "smplx_ground_truth_reason": None if npz_report[asset_id]["present"] else gt_unavailable_reason,
            }
        )

    image0 = None
    if frames_json:
        try:
            import cv2  # type: ignore

            image0 = cv2.imread(str(raw_root / "images" / frames_json[0]["image_png"]), cv2.IMREAD_COLOR)
        except Exception:
            image0 = None
    image_size = {"width": None, "height": None}
    if image0 is not None:
        image_size = {"width": int(image0.shape[1]), "height": int(image0.shape[0])}

    fallback_target_frames = render_manifest.get("target_frames")
    if fallback_target_frames is None:
        fallback_target_frames = render_manifest.get("frame_count")
    if fallback_target_frames is None:
        fallback_target_frames = sum(int(item.get("frame_count", 0)) for item in render_manifest.get("sequence_specs", []))
    fallback_target_frames_requested = render_manifest.get("target_frames_requested")
    if fallback_target_frames_requested is None:
        fallback_target_frames_requested = fallback_target_frames

    dataset_manifest = {
        "kind": "BEDLAM360-v0-dataset",
        "version": 0,
        **build_pipeline_versions(),
        "run_id": render_manifest["run_id"],
        "created_at_utc": render_manifest["created_at_utc"],
        "dataset_root": str(run_root),
        "render_manifest_path": str(run_root / "render_manifest.json"),
        "config_bundle": render_manifest.get("config_bundle"),
        "config_bundle_path": render_manifest.get("config_bundle_path"),
        "resume_state": render_manifest.get("resume_state"),
        "target_frames_requested": int(fallback_target_frames_requested),
        "target_frames": int(fallback_target_frames),
        "frame_count": int(num_frames),
        "sequence_count": int(len(sequence_manifests)),
        "body_count": int(num_bodies),
        "asset_ids": unique_asset_ids,
        "scene_presets": render_manifest.get("scene_presets", []),
        "body_forward_yaw_offset_deg": float(render_manifest.get("body_forward_yaw_offset_deg", DEFAULT_BODY_FORWARD_YAW_OFFSET_DEG)),
        "scene_planning": render_manifest.get("scene_planning", {}),
        "diversity_report": render_manifest.get("diversity_report", {}),
        "duplicate_scene_signature_warning": render_manifest.get("duplicate_scene_signature_warning", {}),
        "duplicate_motion_triplet_warning": render_manifest.get("duplicate_motion_triplet_warning", {}),
        "source_assets": source_asset_records,
        "camera_model": {
            "projection": "equirectangular_360",
            "camera_pose_units": "Unreal centimeters / degrees",
        },
        "erp_projection_mode": erp_projection_mode,
        "scene_capturecube_rotation_effective": bool(erp_projection_mode == "camera_rotation_aware"),
        "erp_yaw_zero_offset_deg_used": None if convention is None else float(convention["erp_yaw_zero_offset_deg"]),
        "erp_projection_note": (
            "SceneCaptureCube long-lat ERP output was empirically observed to be orientation-invariant in BEDLAM360 v1; "
            "translation_only_world_aligned ignores camera yaw/pitch/roll during ERP GT reprojection while preserving camera translation."
            if erp_projection_mode == "translation_only_world_aligned"
            else "Legacy ERP GT reprojection mode that applies full camera rotation and translation."
        ),
        "erp_projection_convention": None if convention is None else convention["payload"],
        "erp_post_rotation": None if not erp_post_rotation_config else dict(erp_post_rotation_config),
        "quality_filter_spec": default_quality_filter_spec(),
        "image_size": image_size,
        "ground_truth_availability": {
            "smplx_parameters": bool(npz_payloads),
            "joints3d": bool(smplx_forward["available"]),
            "vertices": bool(smplx_forward["available"]),
            "projections2d": bool(smplx_forward["available"]),
            "reason": gt_unavailable_reason if not smplx_forward["available"] else None,
        },
        "sequences": [
            {
                "sequence_id": item["sequence_id"],
                "sequence_name": item["sequence_name"],
                "frame_start": int(item["frame_start"]),
                "frame_end": int(item["frame_end"]),
                "frame_count": int(item["frame_count"]),
                "camera_pose_cm_deg": item["camera_pose_cm_deg"],
                "scene_id": item.get("scene_id"),
                "scene_label": item.get("scene_label"),
                "scene_signature": item.get("scene_signature"),
                "motion_triplet_signature": item.get("motion_triplet_signature"),
                "asset_ids": [spec["asset_id"] for spec in item.get("body_specs", [])],
                "motion_ids": [spec["asset_id"] for spec in item.get("body_specs", [])],
                "source_npz_paths": [payload.get("source_npz_path") for payload in item.get("motion_triplet_payload", [])],
                "unreal_animation_asset_paths": [payload.get("unreal_animation_asset_path") for payload in item.get("motion_triplet_payload", [])],
                "appearance_ids_by_body_slot": _appearance_ids_by_body_slot(item.get("body_specs", [])),
                "preview_mp4_path": str(previews_dir / f"{item['sequence_id']}_preview.mp4"),
            }
            for item in render_manifest["sequence_specs"]
        ],
        "files": {
            "raw_camera_world_transforms_npy": str(metadata_dir / "camera_world_transforms.npy"),
            "raw_body_world_transforms_npy": str(metadata_dir / "body_world_transforms.npy"),
            "raw_frame_mapping_json": str(metadata_dir / "frame_mapping.json"),
            "raw_frames_json": str(metadata_dir / "frames.json"),
            "raw_gt_availability_report_json": str(metadata_dir / "gt_availability_report.json"),
            "raw_smplx_parameters_npz": str(smplx_dir / "parameters.npz"),
            "raw_joints3d_npz": str(joints3d_dir / "joints3d.npz"),
            "raw_vertices_npz": str(vertices_dir / "vertices.npz"),
            "raw_projections2d_npz": str(projections2d_dir / "projections2d_erp.npz"),
            "quality_report_csv": str(run_root / "quality_report.csv"),
            "dataset_summary_html": str(run_root / "dataset_summary.html"),
        },
    }
    _write_json(run_root / "manifest.json", dataset_manifest)
    _write_json(raw_root / "manifest.json", dataset_manifest)

    if smplx_forward["available"]:
        gt_alignment.export_gt_erp_alignment(
            raw_root,
            output_subdir="projections2d/erp_alignment",
            projection_mode=erp_projection_mode,
        )
        if erp_post_rotation_config and bool(erp_post_rotation_config.get("enabled")):
            post_rotation_exports = gt_alignment.export_erp_post_rotation_package(
                raw_root,
                output_subdir="projections2d/erp_post_rotation",
                erp_yaw_zero_offset_deg=0.0 if convention is None else float(convention["erp_yaw_zero_offset_deg"]),
                projection_mode=erp_projection_mode,
                post_rotation_config=erp_post_rotation_config,
            )
            dataset_manifest["erp_post_rotation_exports"] = post_rotation_exports

    quality_csv_path, quality_summary = _build_quality_report(
        raw_root,
        dataset_manifest,
        render_manifest,
        sequence_manifests,
        run_root / "quality_report.csv",
    )
    dataset_manifest["quality_summary"] = quality_summary
    quality_rows = list(csv.DictReader(open(quality_csv_path, "r", encoding="utf-8")))
    summary_path, warning_csv, clean_csv, html_summary_path = _write_dataset_summary_exports(
        run_root,
        raw_root,
        quality_rows,
        frames_json,
        frame_mapping,
        render_manifest,
        dataset_manifest,
    )
    dataset_manifest["summary_exports"] = {
        "warning_frames_csv": str(warning_csv),
        "clean_frames_csv": str(clean_csv),
        "dataset_summary_json": str(summary_path),
        "dataset_summary_html": str(html_summary_path),
    }

    accepted_indices = [
        int(row["frame_index"])
        for row in quality_rows
        if str(row["accepted"]).lower() == "true"
    ]
    accepted_root = _ensure_dir(run_root / "accepted")
    accepted_images_dir = _ensure_dir(accepted_root / "images")
    accepted_metadata_frames_dir = _ensure_dir(accepted_root / "metadata" / "frames")
    top_metadata_dir = _ensure_dir(run_root / "metadata")
    top_smplx_dir = _ensure_dir(run_root / "smplx")
    top_joints3d_dir = _ensure_dir(run_root / "joints3d")
    top_vertices_dir = _ensure_dir(run_root / "vertices")
    top_projections2d_dir = _ensure_dir(run_root / "projections2d")

    accepted_frames = [frames_json[index] for index in accepted_indices]
    accepted_mapping = [frame_mapping[index] for index in accepted_indices]
    accepted_available = available[accepted_indices]
    accepted_global_orient = global_orient[accepted_indices]
    accepted_body_pose = body_pose[accepted_indices]
    accepted_trans_world = trans_world[accepted_indices]
    accepted_expression = expression[accepted_indices]
    accepted_camera_world = camera_world[accepted_indices]
    accepted_body_world = body_world[accepted_indices]
    accepted_joints = None if not smplx_forward["available"] else smplx_forward["joints3d"][accepted_indices]
    accepted_vertices = None if not smplx_forward["available"] else smplx_forward["vertices"][accepted_indices]
    raw_proj_npz = np.load(raw_root / "projections2d" / "projections2d_erp.npz")
    raw_joints2d = raw_proj_npz["joints2d"]
    raw_proj_available = raw_proj_npz["available"]
    accepted_joints2d = raw_joints2d[accepted_indices]
    accepted_proj_available = raw_proj_available[accepted_indices]

    for frame in accepted_frames:
        src_png = raw_root / "images" / frame["image_png"]
        if src_png.exists():
            _safe_symlink_or_copy(src_png, accepted_images_dir / frame["image_png"])
        if frame.get("pose_json_path"):
            src_pose = Path(frame["pose_json_path"])
            if src_pose.exists():
                _safe_symlink_or_copy(src_pose, accepted_metadata_frames_dir / src_pose.name)

    _write_json(top_metadata_dir / "frames.json", accepted_frames)
    _write_json(top_metadata_dir / "frame_mapping.json", accepted_mapping)
    np.save(top_metadata_dir / "camera_world_transforms.npy", accepted_camera_world)
    np.save(top_metadata_dir / "body_world_transforms.npy", accepted_body_world)
    if (raw_root / "metadata" / "erp_projection_convention.json").exists():
        _safe_symlink_or_copy(raw_root / "metadata" / "erp_projection_convention.json", top_metadata_dir / "erp_projection_convention.json")
    _safe_symlink_or_copy(raw_root / "metadata" / "gt_availability_report.json", top_metadata_dir / "gt_availability_report.json")

    np.savez_compressed(
        top_smplx_dir / "parameters.npz",
        global_orient=accepted_global_orient,
        body_pose=accepted_body_pose,
        betas=betas,
        transl_world=accepted_trans_world,
        expression=accepted_expression,
        genders=genders,
        available=accepted_available,
    )
    if accepted_joints is not None and accepted_vertices is not None:
        np.savez_compressed(top_joints3d_dir / "joints3d.npz", joints3d=accepted_joints, available=accepted_available)
        np.savez_compressed(top_vertices_dir / "vertices.npz", vertices=accepted_vertices, available=accepted_available)
    else:
        np.savez_compressed(top_joints3d_dir / "joints3d.npz", joints3d=np.empty((len(accepted_indices), num_bodies, 0, 3), dtype=np.float32), available=np.zeros((len(accepted_indices), num_bodies), dtype=bool))
        np.savez_compressed(top_vertices_dir / "vertices.npz", vertices=np.empty((len(accepted_indices), num_bodies, 0, 3), dtype=np.float32), available=np.zeros((len(accepted_indices), num_bodies), dtype=bool))
    np.savez_compressed(
        top_projections2d_dir / "projections2d_erp.npz",
        joints2d=accepted_joints2d,
        vertices2d=np.empty((len(accepted_indices), num_bodies, 0, 2), dtype=np.float32),
        available=accepted_proj_available,
    )

    dataset_manifest["accepted_root"] = str(accepted_root)
    dataset_manifest["raw_root"] = str(raw_root)
    dataset_manifest["accepted_frame_count"] = int(len(accepted_indices))
    dataset_manifest["rejected_frame_count"] = int(num_frames - len(accepted_indices))
    dataset_manifest["files"].update(
        {
            "camera_world_transforms_npy": str(top_metadata_dir / "camera_world_transforms.npy"),
            "body_world_transforms_npy": str(top_metadata_dir / "body_world_transforms.npy"),
            "frame_mapping_json": str(top_metadata_dir / "frame_mapping.json"),
            "frames_json": str(top_metadata_dir / "frames.json"),
            "gt_availability_report_json": str(top_metadata_dir / "gt_availability_report.json"),
            "smplx_parameters_npz": str(top_smplx_dir / "parameters.npz"),
            "joints3d_npz": str(top_joints3d_dir / "joints3d.npz"),
            "vertices_npz": str(top_vertices_dir / "vertices.npz"),
            "projections2d_npz": str(top_projections2d_dir / "projections2d_erp.npz"),
        }
    )
    if dataset_manifest.get("erp_post_rotation_exports"):
        dataset_manifest["files"]["raw_erp_post_rotation_root"] = str(raw_root / "projections2d" / "erp_post_rotation")
    _write_json(run_root / "manifest.json", dataset_manifest)

    preview_pngs = []
    for sequence_manifest in sequence_manifests:
        first_record = sequence_manifest["range_result"]["frame_records"][0]
        first_png_name = f"{sequence_manifest['sequence_id']}__{Path(first_record['png_path']).stem}.png"
        first_png_path = raw_root / "images" / first_png_name
        if first_png_path.exists():
            preview_pngs.append(first_png_path)
    if preview_pngs:
        try:
            preview_tools.export_contact_sheet(
                [str(path) for path in preview_pngs],
                run_root / "previews" / DEFAULT_PREVIEW_CONTACT_SHEET,
                cols=min(3, len(preview_pngs)),
            )
        except Exception:
            _write_json(
                run_root / "previews" / "contact_sheet_error.json",
                {"traceback": traceback.format_exc()},
            )

    _log(f"Wrote dataset manifest: {run_root / 'manifest.json'}")
    _log(f"Wrote quality report: {quality_csv_path}")
    return run_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("render", "postprocess"), default="render")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-root", default="", help="Existing dataset run root for postprocess stage.")
    parser.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)
    parser.add_argument("--frames-per-sequence", type=int, default=DEFAULT_FRAMES_PER_SEQUENCE)
    parser.add_argument("--fixed-seed", type=int, default=DEFAULT_FIXED_SEED)
    parser.add_argument("--auto-postprocess", action="store_true", help="Render stage: invoke this same entry point with --stage postprocess via system python3.")
    parser.add_argument("--debug-single-sequence", action="store_true", help="Render one 10-frame full-appearance sequence for appearance propagation debugging.")
    parser.add_argument("--debug-two-scenes", action="store_true", help="Render 10 frames from each curated scene preset.")
    parser.add_argument("--allow-duplicate-motion-triplets", action="store_true", help="Allow repeated motion triplets across sequences. By default this is treated as a planning error.")
    parser.add_argument("--npz-root", action="append", default=[])
    parser.add_argument("--smplx-model-root", action="append", default=[str(DEFAULT_SMPLX_MODEL_ROOT)])
    parser.add_argument(
        "--erp-projection-mode",
        choices=("camera_rotation_aware", "translation_only_world_aligned"),
        default="camera_rotation_aware",
        help="Postprocess stage: choose whether ERP GT projection uses camera rotation or translation only.",
    )
    args = parser.parse_args()

    if args.stage == "render":
        if args.debug_single_sequence:
            run_root = render_v0_debug_sequence(
                output_root=Path(args.output_root),
                fixed_seed=int(args.fixed_seed),
                allow_duplicate_motion_triplets=bool(args.allow_duplicate_motion_triplets),
            )
        elif args.debug_two_scenes:
            run_root = render_v0_debug_two_scenes(
                output_root=Path(args.output_root),
                fixed_seed=int(args.fixed_seed),
                allow_duplicate_motion_triplets=bool(args.allow_duplicate_motion_triplets),
            )
        else:
            run_root = render_v0_dataset(
                output_root=Path(args.output_root),
                target_frames=int(args.target_frames),
                frames_per_sequence=int(args.frames_per_sequence),
                fixed_seed=int(args.fixed_seed),
                auto_postprocess=bool(args.auto_postprocess),
                smplx_model_roots=args.smplx_model_root,
                allow_duplicate_motion_triplets=bool(args.allow_duplicate_motion_triplets),
            )
        print(run_root)
    else:
        if not args.run_root:
            raise RuntimeError("--run-root is required for --stage postprocess")
        out = postprocess_v0_dataset(
            run_root=Path(args.run_root),
            extra_npz_roots=args.npz_root,
            smplx_model_roots=args.smplx_model_root,
            erp_projection_mode=args.erp_projection_mode,
        )
        print(out)


if __name__ == "__main__":
    main()
