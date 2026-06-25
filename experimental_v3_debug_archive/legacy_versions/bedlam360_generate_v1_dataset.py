import argparse
import csv
import importlib
import json
import math
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    import unreal  # type: ignore
except Exception:  # pragma: no cover
    unreal = None


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bedlam360_generate_v0_dataset as v0  # type: ignore
import bedlam360_pipeline.config_loader as _pipeline_config_loader  # type: ignore
import bedlam360_pipeline.manifest_versioning as _pipeline_manifest_versioning  # type: ignore
import bedlam360_pipeline.resume as _pipeline_resume  # type: ignore
import bedlam360_pipeline.seed_manager as _pipeline_seed_manager  # type: ignore

_pipeline_config_loader = importlib.reload(_pipeline_config_loader)
_pipeline_manifest_versioning = importlib.reload(_pipeline_manifest_versioning)
_pipeline_resume = importlib.reload(_pipeline_resume)
_pipeline_seed_manager = importlib.reload(_pipeline_seed_manager)

load_config_bundle = _pipeline_config_loader.load_config_bundle
build_pipeline_versions = _pipeline_manifest_versioning.build_pipeline_versions
build_resume_state = _pipeline_resume.build_resume_state
collect_completed_sequence_ids = _pipeline_resume.collect_completed_sequence_ids
build_seed_bundle = _pipeline_seed_manager.build_seed_bundle


DEFAULT_OUTPUT_ROOT = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_v1_dataset")
DEFAULT_LATEST_RUN_JSON = DEFAULT_OUTPUT_ROOT / "LATEST_RUN.json"
DEFAULT_CONFIG_ROOT = SCRIPT_DIR.parent / "configs"
DEFAULT_TARGET_FRAMES = 500
DEFAULT_FRAMES_PER_SEQUENCE = 100
DEFAULT_FIXED_SEED = 3601002
DEFAULT_SEQUENCE_DIR = "/Game/BEDLAM360_Debug"
DEFAULT_CAPTURE_FPS = 30
DEFAULT_MAX_TRAJECTORY_ATTEMPTS = 8
DEFAULT_DEBUG_FRAME_COUNT = 20
DEFAULT_SMPLX_MODEL_ROOT = Path("/media/mathis/PANO/BEDLAM2/models/smplx")
DEFAULT_ERP_POST_ROTATION_YAW_AMPLITUDE_DEG = 0.0
DEFAULT_ERP_POST_ROTATION_PITCH_AMPLITUDE_DEG = 0.0
DEFAULT_ERP_POST_ROTATION_ROLL_AMPLITUDE_DEG = 0.0

CAMERA_POLICIES = (
    "static",
    "slow_orbit",
    "small_translation",
    "handheld_micro_motion",
    "close_static",
    "close_orbit",
)

DISTANCE_REGIMES_CM = {
    "extreme_close": (18.0, 35.0),
    "close": (35.0, 90.0),
    "medium": (90.0, 220.0),
    "far": (220.0, 500.0),
}

SAFE_CLEARANCE_THRESHOLDS_CM = {
    "extreme_close": 8.0,
    "close": 20.0,
    "medium": 35.0,
    "far": 45.0,
}


def _log(message):
    text = f"[BEDLAM360][V1_DATASET] {message}"
    if unreal is not None:
        unreal.log(text)
    else:
        print(text)


def _load_canonical_module():
    if unreal is None:
        raise RuntimeError("Render stage requires Unreal Python.")
    import bedlam360_canonical_validation as canonical_validation  # type: ignore

    return importlib.reload(canonical_validation)


def _load_mini_module():
    if unreal is None:
        raise RuntimeError("Render stage requires Unreal Python.")
    import bedlam360_mini_validation as mini_validation  # type: ignore

    return importlib.reload(mini_validation)


def _ensure_dir(path):
    return v0._ensure_dir(path)


def _write_json(path, payload):
    return v0._write_json(path, payload)


def _write_csv(path, rows, fieldnames):
    return v0._write_csv(path, rows, fieldnames)


def _utc_now():
    return datetime.now(timezone.utc)


def _make_run_id():
    return v0._make_run_id()


def _normalize_yaw_deg(yaw_deg):
    return v0._normalize_yaw_deg(yaw_deg)


def _erp_post_rotation_config(
    enabled=False,
    yaw_amplitude_deg=DEFAULT_ERP_POST_ROTATION_YAW_AMPLITUDE_DEG,
    pitch_amplitude_deg=DEFAULT_ERP_POST_ROTATION_PITCH_AMPLITUDE_DEG,
    roll_amplitude_deg=DEFAULT_ERP_POST_ROTATION_ROLL_AMPLITUDE_DEG,
    draw_debug_text=True,
):
    return {
        "enabled": bool(enabled),
        "mode": "spherical_post_rotation",
        "yaw_amplitude_deg": float(yaw_amplitude_deg),
        "pitch_amplitude_deg": float(pitch_amplitude_deg),
        "roll_amplitude_deg": float(roll_amplitude_deg),
        "draw_debug_text": bool(draw_debug_text),
        "yaw_cycles": 1.0,
        "pitch_cycles": 0.75,
        "roll_cycles": 1.25,
        "scene_capturecube_rotation_effective": False,
        "note": (
            "Physical SceneCaptureCube translation is rendered in Unreal; camera orientation motion is synthesized offline "
            "as ERP-domain spherical post-rotation to preserve validated GT alignment."
        ),
    }


def _load_v2_config_bundle(config_path=None):
    return load_config_bundle(DEFAULT_CONFIG_ROOT, overrides_path=config_path)


def _resolve_runtime_settings(
    *,
    config_bundle,
    output_root,
    target_frames,
    frames_per_sequence,
    fixed_seed,
    erp_post_rotation_config,
):
    dataset_cfg = config_bundle.dataset
    post_cfg = config_bundle.post_rotation
    resolved_output_root = output_root
    if str(output_root) == str(DEFAULT_OUTPUT_ROOT):
        resolved_output_root = str(Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360") / str(dataset_cfg.output_root))
    resolved_target_frames = int(target_frames if int(target_frames) != int(DEFAULT_TARGET_FRAMES) else dataset_cfg.target_frames)
    resolved_frames_per_sequence = int(
        frames_per_sequence if int(frames_per_sequence) != int(DEFAULT_FRAMES_PER_SEQUENCE) else dataset_cfg.frames_per_sequence
    )
    resolved_fixed_seed = int(fixed_seed if int(fixed_seed) != int(DEFAULT_FIXED_SEED) else dataset_cfg.global_seed)
    resolved_post_rotation = dict(erp_post_rotation_config)
    if not bool(resolved_post_rotation.get("enabled")) and bool(post_cfg.enabled):
        resolved_post_rotation["enabled"] = True
    for key, default_value in [
        ("yaw_amplitude_deg", DEFAULT_ERP_POST_ROTATION_YAW_AMPLITUDE_DEG),
        ("pitch_amplitude_deg", DEFAULT_ERP_POST_ROTATION_PITCH_AMPLITUDE_DEG),
        ("roll_amplitude_deg", DEFAULT_ERP_POST_ROTATION_ROLL_AMPLITUDE_DEG),
    ]:
        if float(resolved_post_rotation.get(key, default_value)) == float(default_value):
            resolved_post_rotation[key] = float(getattr(post_cfg, key))
    if bool(resolved_post_rotation.get("draw_debug_text", True)) == True:
        resolved_post_rotation["draw_debug_text"] = bool(post_cfg.draw_debug_text)
    return {
        "output_root": str(resolved_output_root),
        "target_frames": resolved_target_frames,
        "frames_per_sequence": resolved_frames_per_sequence,
        "fixed_seed": resolved_fixed_seed,
        "erp_post_rotation_config": resolved_post_rotation,
    }


def _apply_sequence_seed_bundles(sequence_specs, fixed_seed):
    updated = []
    for item in sequence_specs:
        spec = dict(item)
        spec["seed_bundle"] = build_seed_bundle(int(fixed_seed), sequence_id=str(spec["sequence_id"]))
        updated.append(spec)
    return updated


def _scene_presets(canonical):
    return list(v0._scene_presets(canonical))


def _distance_regime_for_policy(camera_policy):
    if camera_policy in {"close_static", "close_orbit"}:
        return "close"
    return {
        "static": "medium",
        "slow_orbit": "medium",
        "small_translation": "medium",
        "handheld_micro_motion": "close",
    }.get(camera_policy, "medium")


def _camera_variants():
    return [
        {"camera_policy": "static", "distance_regime": "medium"},
        {"camera_policy": "slow_orbit", "distance_regime": "medium"},
        {"camera_policy": "small_translation", "distance_regime": "medium"},
        {"camera_policy": "handheld_micro_motion", "distance_regime": "close"},
        {"camera_policy": "close_static", "distance_regime": "extreme_close"},
        {"camera_policy": "close_orbit", "distance_regime": "close"},
    ]


def _scene_signature_v1(sequence_payload):
    digest_source = json.dumps(sequence_payload, sort_keys=True, separators=(",", ":"))
    return f"{sequence_payload['scene_id']}__{v0.hashlib.sha1(digest_source.encode('utf-8')).hexdigest()[:12]}"


def _body_triplet_signature(body_specs):
    return tuple(spec["asset_id"] for spec in body_specs)


def _sequence_specs_v1(target_frames, frames_per_sequence, canonical):
    scene_presets = _scene_presets(canonical)
    requested_sequence_count = int(math.ceil(float(target_frames) / float(frames_per_sequence)))
    variants = _camera_variants()
    specs = []
    remaining = int(target_frames)
    for scene_index, scene in enumerate(scene_presets):
        if remaining <= 0:
            break
        variant = variants[scene_index % len(variants)]
        frame_count = min(int(frames_per_sequence), remaining, 121)
        scene_payload = {
            "scene_id": scene["scene_id"],
            "motion_triplet_signature": scene["motion_triplet_signature"],
            "camera_policy": variant["camera_policy"],
            "distance_regime": variant["distance_regime"],
            "body_specs": scene["body_specs"],
            "camera_pose_cm_deg": scene["camera_pose_cm_deg"],
        }
        specs.append(
            {
                "sequence_index": int(len(specs)),
                "sequence_id": f"seq_{len(specs):04d}",
                "sequence_name": f"bedlam360_v1_{scene['scene_id']}_{variant['camera_policy']}",
                "frame_start": 0,
                "frame_end": int(frame_count - 1),
                "frame_count": int(frame_count),
                "scene_id": scene["scene_id"],
                "scene_label": scene["label"],
                "scene_signature": _scene_signature_v1(scene_payload),
                "motion_triplet_signature": scene["motion_triplet_signature"],
                "motion_triplet_payload": scene["motion_triplet_payload"],
                "body_specs": list(scene["body_specs"]),
                "camera_policy": variant["camera_policy"],
                "distance_regime": variant["distance_regime"],
                "camera_pose_cm_deg": dict(scene["camera_pose_cm_deg"]),
                "body_forward_yaw_offset_deg": scene.get("body_forward_yaw_offset_deg", v0.DEFAULT_BODY_FORWARD_YAW_OFFSET_DEG),
            }
        )
        remaining -= frame_count
        if len(specs) >= requested_sequence_count:
            break
    planning = {
        "requested_target_frames": int(target_frames),
        "frames_per_sequence": int(frames_per_sequence),
        "requested_sequence_count": int(requested_sequence_count),
        "actual_sequence_count": int(len(specs)),
        "actual_target_frames": int(sum(item["frame_count"] for item in specs)),
        "reduced_target_frames": bool(sum(item["frame_count"] for item in specs) < int(target_frames)),
        "reduction_reason": None if sum(item["frame_count"] for item in specs) >= int(target_frames) else "Insufficient curated v1 scene/policy combinations for requested frame count.",
    }
    return specs, planning


def _sequence_diversity_report(sequence_specs):
    scene_signatures = [item["scene_signature"] for item in sequence_specs]
    motion_signatures = [item["motion_triplet_signature"] for item in sequence_specs]
    appearance_combinations = []
    for item in sequence_specs:
        appearance_combinations.append(
            tuple(
                (
                    spec["asset_id"],
                    spec.get("texture_body"),
                    spec.get("texture_clothing"),
                    spec.get("hair"),
                    spec.get("haircolor"),
                    spec.get("shoe"),
                )
                for spec in item["body_specs"]
            )
        )
    duplicate_motion_groups = {}
    for signature in motion_signatures:
        duplicate_motion_groups[signature] = duplicate_motion_groups.get(signature, 0) + 1
    return {
        "unique_scene_signature_count": int(len(set(scene_signatures))),
        "unique_motion_triplet_count": int(len(set(motion_signatures))),
        "unique_body_asset_triplets": int(len(set(_body_triplet_signature(item["body_specs"]) for item in sequence_specs))),
        "unique_appearance_combinations": int(len(set(appearance_combinations))),
        "camera_policies": sorted({item["camera_policy"] for item in sequence_specs}),
        "distance_regimes": sorted({item["distance_regime"] for item in sequence_specs}),
        "duplicate_motion_triplet_groups": {key: value for key, value in duplicate_motion_groups.items() if value > 1},
        "sequence_motion_summary": [
            {
                "sequence_id": item["sequence_id"],
                "scene_id": item["scene_id"],
                "scene_signature": item["scene_signature"],
                "motion_triplet_signature": item["motion_triplet_signature"],
                "camera_policy": item["camera_policy"],
                "distance_regime": item["distance_regime"],
                "asset_ids": [spec["asset_id"] for spec in item["body_specs"]],
                "source_npz_paths": [payload.get("source_npz_path") for payload in item["motion_triplet_payload"]],
                "frame_range": [int(item["frame_start"]), int(item["frame_end"])],
            }
            for item in sequence_specs
        ],
    }


def _centroid_from_body_specs(body_specs):
    count = max(1, len(body_specs))
    return {
        "x": sum(float(spec["x"]) for spec in body_specs) / float(count),
        "y": sum(float(spec["y"]) for spec in body_specs) / float(count),
        "z": sum(float(spec["z"]) for spec in body_specs) / float(count),
    }


def _look_at_rotation(camera_loc, target_loc):
    dx = float(target_loc["x"]) - float(camera_loc["x"])
    dy = float(target_loc["y"]) - float(camera_loc["y"])
    dz = float(target_loc["z"]) - float(camera_loc["z"])
    yaw = math.degrees(math.atan2(dy, dx))
    planar = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, max(1e-6, planar)))
    return float(pitch), float(_normalize_yaw_deg(yaw)), 0.0


def _smooth_signal(progress, amplitude, phase, cycles=1.0):
    return float(amplitude) * math.sin((2.0 * math.pi * float(cycles) * float(progress)) + float(phase))


def _sample_camera_trajectory(sequence_spec, fixed_seed, attempt_index):
    frame_count = int(sequence_spec["frame_count"])
    base_camera = dict(sequence_spec["camera_pose_cm_deg"])
    body_specs = list(sequence_spec["body_specs"])
    centroid = _centroid_from_body_specs(body_specs)
    target_loc = {"x": centroid["x"], "y": centroid["y"], "z": centroid["z"] + 120.0}
    rng = random.Random(int(fixed_seed) + (sequence_spec["sequence_index"] * 1009) + (attempt_index * 97))
    camera_policy = str(sequence_spec["camera_policy"])
    distance_regime = str(sequence_spec["distance_regime"])
    clearance_min_cm, clearance_max_cm = DISTANCE_REGIMES_CM[distance_regime]
    if camera_policy == "close_static":
        clearance_min_cm, clearance_max_cm = DISTANCE_REGIMES_CM["extreme_close"]
    elif camera_policy == "close_orbit":
        clearance_min_cm, clearance_max_cm = DISTANCE_REGIMES_CM["close"]
    clearance_cm = rng.uniform(clearance_min_cm, clearance_max_cm)
    anchor_radius_cm = max(
        90.0,
        max(math.sqrt(float(spec["x"]) ** 2 + float(spec["y"]) ** 2) for spec in body_specs) * 0.30,
    )
    radial_distance_cm = anchor_radius_cm + clearance_cm
    base_azimuth_deg = rng.uniform(-180.0, 180.0)
    base_height_cm = float(base_camera.get("z", 160.0)) + rng.uniform(-10.0, 10.0)
    phase_a = rng.uniform(0.0, 2.0 * math.pi)
    phase_b = rng.uniform(0.0, 2.0 * math.pi)
    phase_c = rng.uniform(0.0, 2.0 * math.pi)

    poses = []
    metadata = []
    for frame_offset in range(frame_count):
        progress = 0.0 if frame_count <= 1 else float(frame_offset) / float(frame_count - 1)
        azimuth_deg = base_azimuth_deg
        radial_delta_cm = 0.0
        x_micro = 0.0
        y_micro = 0.0
        z_micro = 0.0
        yaw_extra = 0.0
        pitch_extra = 0.0
        roll_extra = 0.0

        if camera_policy == "slow_orbit":
            azimuth_deg += -18.0 + (36.0 * progress)
            yaw_extra += _smooth_signal(progress, 2.0, phase_a, cycles=1.0)
        elif camera_policy == "small_translation":
            x_micro += _smooth_signal(progress, 18.0, phase_a, cycles=1.0)
            y_micro += _smooth_signal(progress, 15.0, phase_b, cycles=1.5)
            z_micro += _smooth_signal(progress, 8.0, phase_c, cycles=0.5)
            yaw_extra += _smooth_signal(progress, 2.5, phase_a, cycles=1.0)
            pitch_extra += _smooth_signal(progress, 1.5, phase_b, cycles=1.0)
        elif camera_policy == "handheld_micro_motion":
            x_micro += _smooth_signal(progress, 10.0, phase_a, cycles=1.7)
            y_micro += _smooth_signal(progress, 12.0, phase_b, cycles=1.3)
            z_micro += _smooth_signal(progress, 6.0, phase_c, cycles=1.9)
            yaw_extra += _smooth_signal(progress, 3.0, phase_a, cycles=1.5)
            pitch_extra += _smooth_signal(progress, 2.0, phase_b, cycles=1.3)
            roll_extra += _smooth_signal(progress, 1.5, phase_c, cycles=1.8)
        elif camera_policy == "close_orbit":
            azimuth_deg += -14.0 + (28.0 * progress)
            radial_delta_cm += _smooth_signal(progress, 6.0, phase_a, cycles=1.0)
            yaw_extra += _smooth_signal(progress, 2.0, phase_b, cycles=1.0)
            pitch_extra += _smooth_signal(progress, 1.5, phase_c, cycles=0.75)
        elif camera_policy == "close_static":
            x_micro += _smooth_signal(progress, 6.0, phase_a, cycles=1.0)
            y_micro += _smooth_signal(progress, 4.0, phase_b, cycles=1.0)
            z_micro += _smooth_signal(progress, 4.0, phase_c, cycles=0.6)
            yaw_extra += _smooth_signal(progress, 1.4, phase_b, cycles=1.0)
            pitch_extra += _smooth_signal(progress, 1.2, phase_c, cycles=0.8)
        else:  # static
            yaw_extra += _smooth_signal(progress, 1.0, phase_a, cycles=0.5)
            pitch_extra += _smooth_signal(progress, 0.8, phase_b, cycles=0.5)

        radians_azimuth = math.radians(azimuth_deg)
        effective_distance_cm = max(30.0, radial_distance_cm + radial_delta_cm)
        camera_loc = {
            "x": centroid["x"] + math.cos(radians_azimuth) * effective_distance_cm + x_micro,
            "y": centroid["y"] + math.sin(radians_azimuth) * effective_distance_cm + y_micro,
            "z": base_height_cm + z_micro,
        }
        pitch_deg, yaw_deg, _ = _look_at_rotation(camera_loc, target_loc)
        final_pose = {
            "x": float(camera_loc["x"]),
            "y": float(camera_loc["y"]),
            "z": float(camera_loc["z"]),
            "yaw": float(_normalize_yaw_deg(yaw_deg + yaw_extra)),
            "pitch": float(max(-35.0, min(35.0, pitch_deg + pitch_extra))),
            "roll": float(max(-8.0, min(8.0, roll_extra))),
        }
        poses.append(final_pose)
        metadata.append(
            {
                "camera_policy": camera_policy,
                "distance_regime": distance_regime,
                "frame_offset": int(frame_offset),
                "progress": float(progress),
                "target_loc_cm": dict(target_loc),
                "base_clearance_cm": float(clearance_cm),
                "base_azimuth_deg": float(base_azimuth_deg),
                "azimuth_deg": float(azimuth_deg),
                "radial_distance_cm": float(effective_distance_cm),
                "yaw_variation_deg": float(yaw_extra),
                "pitch_variation_deg": float(pitch_extra),
                "roll_variation_deg": float(roll_extra),
                "camera_pose_cm_deg": dict(final_pose),
            }
        )
    return {
        "poses": poses,
        "metadata": metadata,
        "summary": {
            "camera_policy": camera_policy,
            "distance_regime": distance_regime,
            "attempt_index": int(attempt_index),
            "base_clearance_cm": float(clearance_cm),
            "base_azimuth_deg": float(base_azimuth_deg),
            "base_height_cm": float(base_height_cm),
        },
    }


def _merge_clearance_samples(proximity_samples, bounds_clearances):
    bounds_by_asset = {item["asset_id"]: item for item in bounds_clearances}
    merged = []
    for sample in proximity_samples:
        bounds_sample = bounds_by_asset.get(sample["asset_id"])
        actor_bounds_clearance_cm = None if bounds_sample is None else bounds_sample.get("surface_clearance_cm")
        effective_clearance_cm = float(sample["surface_clearance_cm"])
        if actor_bounds_clearance_cm is not None:
            effective_clearance_cm = min(effective_clearance_cm, float(actor_bounds_clearance_cm))
        merged.append(
            {
                **sample,
                "actor_bounds_clearance_cm": actor_bounds_clearance_cm,
                "effective_clearance_cm": effective_clearance_cm,
                "effective_clearance_m": effective_clearance_cm / 100.0,
            }
        )
    return merged


def _validate_camera_trajectory(mini, level_sequence_info, frame_start, frame_end, trajectory_bundle, distance_regime):
    frame_numbers = list(range(int(frame_start), int(frame_end) + 1))
    per_frame = []
    min_effective_clearance_cm = None
    closest_frame = None
    closest_body = None
    camera_inside_body = False
    rejection_reasons = set()
    safe_threshold_cm = float(SAFE_CLEARANCE_THRESHOLDS_CM[distance_regime])

    for local_index, frame_index in enumerate(frame_numbers):
        mini._evaluate_level_sequence_frame(
            level_sequence_info["level_sequence"],
            target_frame_index=frame_index,
            warmup_frames=mini.DEFAULT_LEVEL_SEQUENCE_WARMUP_FRAMES,
        )
        live_bodies = mini._sync_sequence_bound_bodies(level_sequence_info["bodies"])
        camera_pose = trajectory_bundle["poses"][local_index]
        proximity_samples = mini._camera_to_actor_proximity(camera_pose, live_bodies)
        bounds_clearances = mini._camera_to_bounds_clearances(camera_pose, live_bodies)
        merged = _merge_clearance_samples(proximity_samples, bounds_clearances)
        nearest = min(merged, key=lambda item: item["effective_clearance_cm"]) if merged else None
        frame_min_clearance_cm = None if nearest is None else float(nearest["effective_clearance_cm"])
        intersects = bool(frame_min_clearance_cm is not None and frame_min_clearance_cm < 0.0)
        if intersects:
            camera_inside_body = True
            rejection_reasons.add("camera_inside_body")
            rejection_reasons.add("body_bounding_volume_intersection")
        if frame_min_clearance_cm is not None and frame_min_clearance_cm < safe_threshold_cm:
            rejection_reasons.add("minimum_surface_clearance_below_threshold")
        if min_effective_clearance_cm is None or (frame_min_clearance_cm is not None and frame_min_clearance_cm < min_effective_clearance_cm):
            min_effective_clearance_cm = frame_min_clearance_cm
            closest_frame = int(frame_index)
            closest_body = None if nearest is None else nearest["asset_id"]
        per_frame.append(
            {
                "frame_index": int(frame_index),
                "camera_pose_cm_deg": dict(camera_pose),
                "yaw_deg": float(camera_pose["yaw"]),
                "pitch_deg": float(camera_pose["pitch"]),
                "roll_deg": float(camera_pose["roll"]),
                "min_effective_clearance_cm": frame_min_clearance_cm,
                "closest_body_asset_id": None if nearest is None else nearest["asset_id"],
                "closest_body_actor_label": None if nearest is None else nearest["actor_label"],
                "camera_inside_body": bool(intersects),
                "proximity_category": None if nearest is None else mini._proximity_category_from_surface_clearance_m(float(nearest["effective_clearance_m"])),
                "proximity_samples": merged,
            }
        )

    accepted = not rejection_reasons
    return {
        "accepted": bool(accepted),
        "distance_regime": distance_regime,
        "minimum_surface_clearance_cm": None if min_effective_clearance_cm is None else float(min_effective_clearance_cm),
        "minimum_surface_clearance_threshold_cm": float(safe_threshold_cm),
        "closest_body_asset_id": closest_body,
        "closest_frame_index": closest_frame,
        "camera_inside_body_detected": bool(camera_inside_body),
        "rejection_reasons": sorted(rejection_reasons),
        "per_frame": per_frame,
    }


def _trajectory_reports_root(run_root):
    root = _ensure_dir(Path(run_root) / "trajectory_reports")
    _ensure_dir(root / "accepted")
    _ensure_dir(root / "rejected")
    return root


def _select_validated_trajectory(mini, level_sequence_info, sequence_spec, fixed_seed, reports_root):
    attempt_reports = []
    accepted_report = None
    accepted_bundle = None
    for attempt_index in range(DEFAULT_MAX_TRAJECTORY_ATTEMPTS):
        bundle = _sample_camera_trajectory(sequence_spec, fixed_seed, attempt_index)
        validation = _validate_camera_trajectory(
            mini=mini,
            level_sequence_info=level_sequence_info,
            frame_start=sequence_spec["frame_start"],
            frame_end=sequence_spec["frame_end"],
            trajectory_bundle=bundle,
            distance_regime=sequence_spec["distance_regime"],
        )
        report = {
            "sequence_id": sequence_spec["sequence_id"],
            "scene_id": sequence_spec["scene_id"],
            "camera_policy": sequence_spec["camera_policy"],
            "distance_regime": sequence_spec["distance_regime"],
            "attempt_index": int(attempt_index),
            "trajectory_summary": bundle["summary"],
            "validation": validation,
            "poses": bundle["poses"],
            "metadata": bundle["metadata"],
        }
        attempt_reports.append(report)
        report_path = reports_root / ("accepted" if validation["accepted"] else "rejected") / f"{sequence_spec['sequence_id']}_attempt_{attempt_index:02d}.json"
        _write_json(report_path, report)
        if validation["accepted"]:
            accepted_report = report
            accepted_bundle = bundle
            break
    return accepted_bundle, accepted_report, attempt_reports


def _write_trajectory_summary_reports(run_root, rendered_sequences):
    summary_rows = []
    rejection_rows = []
    accepted_root = _ensure_dir(Path(run_root) / "trajectory_reports" / "accepted")
    rejected_root = _ensure_dir(Path(run_root) / "trajectory_reports" / "rejected")
    for rendered in rendered_sequences:
        report_path_value = rendered.get("trajectory_report_path", "")
        report_path = Path(report_path_value) if report_path_value else None
        if report_path is not None and report_path.exists():
            report = v0._read_json(report_path)
            validation = report.get("validation") or {}
            summary_rows.append(
                {
                    "sequence_id": report.get("sequence_id"),
                    "scene_id": report.get("scene_id"),
                    "camera_policy": report.get("camera_policy"),
                    "distance_regime": report.get("distance_regime"),
                    "attempt_index": report.get("attempt_index"),
                    "accepted": bool(validation.get("accepted")),
                    "minimum_surface_clearance_cm": validation.get("minimum_surface_clearance_cm"),
                    "closest_body_asset_id": validation.get("closest_body_asset_id"),
                    "closest_frame_index": validation.get("closest_frame_index"),
                    "camera_inside_body_detected": bool(validation.get("camera_inside_body_detected")),
                    "rejection_reasons": ",".join(validation.get("rejection_reasons") or []),
                    "report_path": str(report_path),
                }
            )
        sequence_root = Path(rendered["sequence_root"])
        for rejected_path in sorted(rejected_root.glob(f"{rendered['sequence_id']}_attempt_*.json")):
            report = v0._read_json(rejected_path)
            validation = report.get("validation") or {}
            rejection_rows.append(
                {
                    "sequence_id": report.get("sequence_id"),
                    "scene_id": report.get("scene_id"),
                    "camera_policy": report.get("camera_policy"),
                    "distance_regime": report.get("distance_regime"),
                    "attempt_index": report.get("attempt_index"),
                    "minimum_surface_clearance_cm": validation.get("minimum_surface_clearance_cm"),
                    "closest_body_asset_id": validation.get("closest_body_asset_id"),
                    "closest_frame_index": validation.get("closest_frame_index"),
                    "camera_inside_body_detected": bool(validation.get("camera_inside_body_detected")),
                    "rejection_reasons": ",".join(validation.get("rejection_reasons") or []),
                    "report_path": str(rejected_path),
                    "sequence_root": str(sequence_root),
                }
            )
    accepted_csv = Path(run_root) / "trajectory_reports" / "accepted_trajectories.csv"
    rejected_csv = Path(run_root) / "trajectory_reports" / "rejected_trajectories.csv"
    _write_csv(
        accepted_csv,
        summary_rows,
        [
            "sequence_id",
            "scene_id",
            "camera_policy",
            "distance_regime",
            "attempt_index",
            "accepted",
            "minimum_surface_clearance_cm",
            "closest_body_asset_id",
            "closest_frame_index",
            "camera_inside_body_detected",
            "rejection_reasons",
            "report_path",
        ],
    )
    _write_csv(
        rejected_csv,
        rejection_rows,
        [
            "sequence_id",
            "scene_id",
            "camera_policy",
            "distance_regime",
            "attempt_index",
            "minimum_surface_clearance_cm",
            "closest_body_asset_id",
            "closest_frame_index",
            "camera_inside_body_detected",
            "rejection_reasons",
            "report_path",
            "sequence_root",
        ],
    )
    return {
        "accepted_csv_path": str(accepted_csv),
        "rejected_csv_path": str(rejected_csv),
        "accepted_count": int(len(summary_rows)),
        "rejected_attempt_count": int(len(rejection_rows)),
    }


def _render_sequence_v1(canonical, mini, run_id, run_root, sequence_spec, fixed_seed):
    sequence_root = _ensure_dir(Path(run_root) / "raw" / "sequences" / sequence_spec["sequence_id"])
    reports_root = _trajectory_reports_root(run_root)
    body_specs = list(sequence_spec["body_specs"])
    unreal_sequence_name = f"{sequence_spec['sequence_name']}_{run_id}"
    spawned = canonical._append_clothing_actors_for_specs(  # pylint: disable=protected-access
        canonical._spawn_canonical_bodies(unreal_sequence_name, body_specs)  # pylint: disable=protected-access
    )
    actor = mini.capture_scene_cube.find_scene_capture_cube(mini.DEFAULT_ACTOR_LABEL)
    if actor is None:
        raise RuntimeError(f"SceneCaptureCube actor not found: {mini.DEFAULT_ACTOR_LABEL}")
    component = mini.capture_scene_cube.get_capture_component(actor)
    texture_target = mini.capture_scene_cube.get_texture_target(component)
    export_lib = mini.unreal.BEDLAM360ExportLibrary
    level_sequence_info = mini._create_level_sequence_for_sequence_bodies(  # pylint: disable=protected-access
        sequence_name=unreal_sequence_name,
        spawned_bodies=spawned,
        sequence_frame_count=(int(sequence_spec["frame_end"]) + 1),
        level_sequence_dir=DEFAULT_SEQUENCE_DIR,
        display_fps=DEFAULT_CAPTURE_FPS,
        warmup_frames=mini.DEFAULT_LEVEL_SEQUENCE_WARMUP_FRAMES,
        use_natural_timing=canonical.DEFAULT_USE_NATURAL_TIMING,
    )

    trajectory_bundle, accepted_report, attempt_reports = _select_validated_trajectory(
        mini=mini,
        level_sequence_info=level_sequence_info,
        sequence_spec=sequence_spec,
        fixed_seed=fixed_seed,
        reports_root=reports_root,
    )
    if trajectory_bundle is None or accepted_report is None:
        raise RuntimeError(
            f"No safe trajectory found for {sequence_spec['sequence_id']} "
            f"policy={sequence_spec['camera_policy']} regime={sequence_spec['distance_regime']} "
            f"after {DEFAULT_MAX_TRAJECTORY_ATTEMPTS} attempts."
        )

    range_result = canonical._render_range(  # pylint: disable=protected-access
        run_id=run_id,
        run_root=sequence_root,
        actor=actor,
        component=component,
        texture_target=texture_target,
        export_lib=export_lib,
        level_sequence_info=level_sequence_info,
        camera_pose=dict(sequence_spec["camera_pose_cm_deg"]),
        camera_trajectory=trajectory_bundle["poses"],
        camera_trajectory_metadata=trajectory_bundle["metadata"],
        frame_start=sequence_spec["frame_start"],
        frame_end=sequence_spec["frame_end"],
    )
    appearance_debug_by_body = [canonical._appearance_debug_entry(body) for body in spawned]  # pylint: disable=protected-access
    for item in appearance_debug_by_body:
        unreal.log(
            "[BEDLAM360][V1_APPEARANCE_DEBUG] "
            f"asset={item['asset_id']} texture_body={item['texture_body']} "
            f"texture_clothing={item['texture_clothing']} hair={item['hair']} "
            f"haircolor={item['haircolor']} shoe={item['shoe']} "
            f"body_material={item['body_material_path']} clothing_asset={item['clothing_asset_path']} "
            f"clothing_material={item['clothing_material_path']} hair_applied={item['hair_applied']}"
        )

    sequence_manifest = {
        "sequence_id": sequence_spec["sequence_id"],
        "sequence_name": sequence_spec["sequence_name"],
        "unreal_sequence_name": unreal_sequence_name,
        "sequence_index": int(sequence_spec["sequence_index"]),
        "frame_start": int(sequence_spec["frame_start"]),
        "frame_end": int(sequence_spec["frame_end"]),
        "frame_count": int(sequence_spec["frame_count"]),
        "scene_id": sequence_spec["scene_id"],
        "scene_label": sequence_spec["scene_label"],
        "scene_signature": sequence_spec["scene_signature"],
        "motion_triplet_signature": sequence_spec["motion_triplet_signature"],
        "motion_triplet_payload": sequence_spec["motion_triplet_payload"],
        "camera_policy": sequence_spec["camera_policy"],
        "distance_regime": sequence_spec["distance_regime"],
        "seed_bundle": sequence_spec.get("seed_bundle"),
        "body_specs": body_specs,
        "spawned_roles": [
            {
                "asset_id": body["body_pose"]["asset_id"],
                "actor_label": body["actor_label"],
                "appearance_metadata": body.get("appearance_metadata"),
            }
            for body in spawned
        ],
        "appearance_debug_by_body": appearance_debug_by_body,
        "trajectory_validation": accepted_report["validation"],
        "trajectory_summary": accepted_report["trajectory_summary"],
        "trajectory_attempt_count": int(len(attempt_reports)),
        "trajectory_report_path": str(reports_root / "accepted" / f"{sequence_spec['sequence_id']}_attempt_{accepted_report['attempt_index']:02d}.json"),
        "level_sequence": {
            "asset_path": level_sequence_info["sequence_asset_path"],
            "display_fps": level_sequence_info["display_fps"],
            "warmup_frames": level_sequence_info["warmup_frames"],
            "timeline_frame_count": level_sequence_info["timeline_frame_count"],
        },
        "range_result": range_result,
        "paths": {
            "sequence_root": str(sequence_root),
            "images_dir": str(sequence_root / range_result["range_tag"] / "images"),
            "metadata_dir": str(sequence_root / range_result["range_tag"] / "metadata"),
            "previews_dir": str(sequence_root / range_result["range_tag"] / "previews"),
            "preview_mp4_path": str(range_result["preview_mp4_path"]),
        },
    }
    _write_json(sequence_root / "trajectory.json", {"poses": trajectory_bundle["poses"], "metadata": trajectory_bundle["metadata"]})
    _write_csv(
        sequence_root / "trajectory.csv",
        [
            {
                "frame_index": int(sequence_spec["frame_start"] + idx),
                "x": pose["x"],
                "y": pose["y"],
                "z": pose["z"],
                "yaw": pose["yaw"],
                "pitch": pose["pitch"],
                "roll": pose["roll"],
                "camera_policy": sequence_spec["camera_policy"],
                "distance_regime": sequence_spec["distance_regime"],
            }
            for idx, pose in enumerate(trajectory_bundle["poses"])
        ],
        fieldnames=["frame_index", "x", "y", "z", "yaw", "pitch", "roll", "camera_policy", "distance_regime"],
    )
    manifest_path = sequence_root / "manifest.json"
    _write_json(manifest_path, sequence_manifest)
    _log(
        f"Rendered {sequence_spec['sequence_id']} policy={sequence_spec['camera_policy']} "
        f"regime={sequence_spec['distance_regime']} preview={range_result['preview_mp4_path']}"
    )
    _log(f"Sequence manifest: {manifest_path}")
    return {
        "sequence_id": sequence_spec["sequence_id"],
        "sequence_root": str(sequence_root),
        "manifest_path": str(manifest_path),
        "frame_count": int(sequence_spec["frame_count"]),
        "trajectory_report_path": sequence_manifest["trajectory_report_path"],
        "range_result": range_result,
    }


def render_v1_dataset(
    output_root=DEFAULT_OUTPUT_ROOT,
    target_frames=DEFAULT_TARGET_FRAMES,
    frames_per_sequence=DEFAULT_FRAMES_PER_SEQUENCE,
    fixed_seed=DEFAULT_FIXED_SEED,
    erp_post_rotation_config=None,
    config_path=None,
    resume_run_root="",
):
    canonical = _load_canonical_module()
    mini = _load_mini_module()
    config_bundle = _load_v2_config_bundle(config_path)
    runtime = _resolve_runtime_settings(
        config_bundle=config_bundle,
        output_root=output_root,
        target_frames=target_frames,
        frames_per_sequence=frames_per_sequence,
        fixed_seed=fixed_seed,
        erp_post_rotation_config=erp_post_rotation_config,
    )
    output_root = runtime["output_root"]
    target_frames = runtime["target_frames"]
    frames_per_sequence = runtime["frames_per_sequence"]
    fixed_seed = runtime["fixed_seed"]
    erp_post_rotation_config = runtime["erp_post_rotation_config"]
    run_id = _make_run_id() if not resume_run_root else Path(resume_run_root).name
    run_root = _ensure_dir(Path(resume_run_root) if resume_run_root else (Path(output_root) / run_id))
    script_path = Path(__file__).resolve()
    script_stat = script_path.stat()
    sequence_specs, planning = _sequence_specs_v1(int(target_frames), int(frames_per_sequence), canonical)
    sequence_specs = _apply_sequence_seed_bundles(sequence_specs, fixed_seed)
    diversity_report = _sequence_diversity_report(sequence_specs)
    completed_sequence_ids = collect_completed_sequence_ids(run_root)
    rendered_sequences = []
    for sequence_spec in sequence_specs:
        if sequence_spec["sequence_id"] in completed_sequence_ids:
            existing_manifest = Path(run_root) / "raw" / "sequences" / sequence_spec["sequence_id"] / "manifest.json"
            rendered_sequences.append(
                {
                    "sequence_id": sequence_spec["sequence_id"],
                    "sequence_root": str(Path(run_root) / "raw" / "sequences" / sequence_spec["sequence_id"]),
                    "manifest_path": str(existing_manifest),
                    "frame_count": int(sequence_spec["frame_count"]),
                    "trajectory_report_path": "",
                    "range_result": {"resumed": True},
                    "resume_reused": True,
                }
            )
            continue
        rendered_sequences.append(
            _render_sequence_v1(
                canonical=canonical,
                mini=mini,
                run_id=run_id,
                run_root=run_root,
                sequence_spec=sequence_spec,
                fixed_seed=int(fixed_seed),
            )
        )
    trajectory_report_summary = _write_trajectory_summary_reports(run_root, rendered_sequences)
    render_manifest = {
        "kind": "BEDLAM360-v1-render",
        "version": 1,
        **build_pipeline_versions(),
        "run_id": run_id,
        "created_at_utc": _utc_now().isoformat(),
        "dataset_root": str(run_root),
        "fixed_seed": int(fixed_seed),
        "seed_bundle": build_seed_bundle(int(fixed_seed)),
        "target_frames_requested": int(target_frames),
        "target_frames": int(planning["actual_target_frames"]),
        "frames_per_sequence": int(frames_per_sequence),
        "sequence_count": int(len(rendered_sequences)),
        "body_count": 3,
        "camera_policies": list(CAMERA_POLICIES),
        "distance_regimes": list(DISTANCE_REGIMES_CM.keys()),
        "scene_planning": planning,
        "diversity_report": diversity_report,
        "sequence_specs": sequence_specs,
        "rendered_sequences": rendered_sequences,
        "trajectory_report_summary": trajectory_report_summary,
        "erp_post_rotation_config": erp_post_rotation_config,
        "config_bundle": config_bundle.to_dict(),
        "config_bundle_path": None if not config_path else str(config_path),
        "resume_state": build_resume_state(run_root),
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
    return run_root


def render_v1_debug_two_scenes(
    output_root=DEFAULT_OUTPUT_ROOT,
    fixed_seed=DEFAULT_FIXED_SEED,
    erp_post_rotation_config=None,
    config_path=None,
):
    canonical = _load_canonical_module()
    config_bundle = _load_v2_config_bundle(config_path)
    runtime = _resolve_runtime_settings(
        config_bundle=config_bundle,
        output_root=output_root,
        target_frames=2 * DEFAULT_DEBUG_FRAME_COUNT,
        frames_per_sequence=DEFAULT_DEBUG_FRAME_COUNT,
        fixed_seed=fixed_seed,
        erp_post_rotation_config=erp_post_rotation_config,
    )
    output_root = runtime["output_root"]
    fixed_seed = runtime["fixed_seed"]
    erp_post_rotation_config = runtime["erp_post_rotation_config"]
    specs, _planning = _sequence_specs_v1(target_frames=2 * DEFAULT_DEBUG_FRAME_COUNT, frames_per_sequence=DEFAULT_DEBUG_FRAME_COUNT, canonical=canonical)
    specs = _apply_sequence_seed_bundles(specs, fixed_seed)
    if len(specs) > 2:
        specs = specs[:2]
    mini = _load_mini_module()
    run_id = _make_run_id()
    run_root = _ensure_dir(Path(output_root) / f"{run_id}_debug_two_scenes")
    rendered_sequences = []
    for spec in specs:
        rendered_sequences.append(
            _render_sequence_v1(
                canonical=canonical,
                mini=mini,
                run_id=run_id,
                run_root=run_root,
                sequence_spec=spec,
                fixed_seed=int(fixed_seed),
            )
        )
    trajectory_report_summary = _write_trajectory_summary_reports(run_root, rendered_sequences)
    manifest = {
        "kind": "BEDLAM360-v1-render-debug-two-scenes",
        "version": 1,
        **build_pipeline_versions(),
        "run_id": run_id,
        "created_at_utc": _utc_now().isoformat(),
        "dataset_root": str(run_root),
        "fixed_seed": int(fixed_seed),
        "seed_bundle": build_seed_bundle(int(fixed_seed)),
        "sequence_specs": specs,
        "rendered_sequences": rendered_sequences,
        "trajectory_report_summary": trajectory_report_summary,
        "diversity_report": _sequence_diversity_report(specs),
        "erp_post_rotation_config": erp_post_rotation_config,
        "config_bundle": config_bundle.to_dict(),
        "config_bundle_path": None if not config_path else str(config_path),
    }
    render_manifest_path = run_root / "render_manifest.json"
    _write_json(render_manifest_path, manifest)
    _log(f"Debug-two-scenes run root: {run_root}")
    _log(f"Debug-two-scenes render manifest: {render_manifest_path}")
    return run_root


def postprocess_v1_dataset(
    run_root,
    extra_npz_roots=None,
    smplx_model_roots=None,
    erp_projection_mode="translation_only_world_aligned",
    erp_post_rotation_config=None,
):
    if erp_post_rotation_config is None:
        render_manifest_path = Path(run_root) / "render_manifest.json"
        if render_manifest_path.exists():
            render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
            erp_post_rotation_config = render_manifest.get("erp_post_rotation_config")
    return v0.postprocess_v0_dataset(
        run_root,
        extra_npz_roots=extra_npz_roots,
        smplx_model_roots=smplx_model_roots,
        erp_projection_mode=erp_projection_mode,
        erp_post_rotation_config=erp_post_rotation_config,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("render", "postprocess"), default="render")
    parser.add_argument("--config", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-root", default="")
    parser.add_argument("--resume-run-root", default="")
    parser.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)
    parser.add_argument("--frames-per-sequence", type=int, default=DEFAULT_FRAMES_PER_SEQUENCE)
    parser.add_argument("--fixed-seed", type=int, default=DEFAULT_FIXED_SEED)
    parser.add_argument("--debug-two-scenes", action="store_true")
    parser.add_argument("--enable-erp-post-rotation", action="store_true")
    parser.add_argument("--erp-post-rotation-yaw-amplitude", type=float, default=DEFAULT_ERP_POST_ROTATION_YAW_AMPLITUDE_DEG)
    parser.add_argument("--erp-post-rotation-pitch-amplitude", type=float, default=DEFAULT_ERP_POST_ROTATION_PITCH_AMPLITUDE_DEG)
    parser.add_argument("--erp-post-rotation-roll-amplitude", type=float, default=DEFAULT_ERP_POST_ROTATION_ROLL_AMPLITUDE_DEG)
    parser.add_argument("--no-debug-text", action="store_true")
    parser.add_argument("--npz-root", action="append", default=[])
    parser.add_argument("--smplx-model-root", action="append", default=[str(DEFAULT_SMPLX_MODEL_ROOT)])
    parser.add_argument(
        "--erp-projection-mode",
        choices=("camera_rotation_aware", "translation_only_world_aligned"),
        default="translation_only_world_aligned",
    )
    args = parser.parse_args()
    erp_post_rotation_config = _erp_post_rotation_config(
        enabled=bool(args.enable_erp_post_rotation),
        yaw_amplitude_deg=float(args.erp_post_rotation_yaw_amplitude),
        pitch_amplitude_deg=float(args.erp_post_rotation_pitch_amplitude),
        roll_amplitude_deg=float(args.erp_post_rotation_roll_amplitude),
        draw_debug_text=not bool(args.no_debug_text),
    )

    if args.stage == "render":
        if args.debug_two_scenes:
            run_root = render_v1_debug_two_scenes(
                output_root=args.output_root,
                fixed_seed=args.fixed_seed,
                erp_post_rotation_config=erp_post_rotation_config,
                config_path=(None if not args.config else args.config),
            )
        else:
            run_root = render_v1_dataset(
                output_root=args.output_root,
                target_frames=args.target_frames,
                frames_per_sequence=args.frames_per_sequence,
                fixed_seed=args.fixed_seed,
                erp_post_rotation_config=erp_post_rotation_config,
                config_path=(None if not args.config else args.config),
                resume_run_root=args.resume_run_root,
            )
    else:
        if not args.run_root:
            raise RuntimeError("--run-root is required for --stage postprocess")
        run_root = postprocess_v1_dataset(
            run_root=args.run_root,
            extra_npz_roots=args.npz_root,
            smplx_model_roots=args.smplx_model_root,
            erp_projection_mode=args.erp_projection_mode,
            erp_post_rotation_config=erp_post_rotation_config if args.enable_erp_post_rotation else None,
        )
    _log(f"Completed stage={args.stage} run_root={run_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if unreal is not None:
            unreal.log_error(traceback.format_exc())
        raise
