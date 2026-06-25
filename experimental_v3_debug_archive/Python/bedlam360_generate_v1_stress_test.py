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
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    import unreal  # type: ignore
except Exception:  # pragma: no cover
    unreal = None


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bedlam360_generate_v0_dataset as v0  # type: ignore
import bedlam360_generate_v1_dataset as v1  # type: ignore


DEFAULT_OUTPUT_ROOT = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_v1_stress_tests")
DEFAULT_LATEST_RUN_JSON = DEFAULT_OUTPUT_ROOT / "LATEST_RUN.json"
DEFAULT_FIXED_SEED = 3601911
DEFAULT_FRAME_COUNT = 180
DEFAULT_SEQUENCE_DIR = v1.DEFAULT_SEQUENCE_DIR
DEFAULT_CAPTURE_FPS = v1.DEFAULT_CAPTURE_FPS
DEFAULT_SMPLX_MODEL_ROOT = v1.DEFAULT_SMPLX_MODEL_ROOT
DEFAULT_ERP_PROJECTION_MODE = "translation_only_world_aligned"
DEFAULT_STRESS_TEST_NAME = "extreme_close_orbit_pushin"
DEFAULT_KNOWN_GOOD_ASSET_ID = "it_4083_2XL_2408"
DEFAULT_SCENE_ID = "scene_b_front"


def _log(message):
    text = f"[BEDLAM360][V1_STRESS] {message}"
    if unreal is not None:
        unreal.log(text)
    else:
        print(text)


def _ensure_dir(path):
    return v0._ensure_dir(path)


def _write_json(path, payload):
    return v0._write_json(path, payload)


def _write_csv(path, rows, fieldnames):
    return v0._write_csv(path, rows, fieldnames)


def _utc_now():
    return datetime.now(timezone.utc)


def _load_canonical_module():
    return v1._load_canonical_module()


def _load_mini_module():
    return v1._load_mini_module()


def _load_postprocess_modules():
    return v0._load_postprocess_modules()


def _load_stress_postprocess_modules():
    import bedlam360_gt_erp_alignment as gt_alignment  # type: ignore
    import bedlam360_validate_erp_projection as validator  # type: ignore

    return gt_alignment, validator


def _require_postprocess_deps():
    if cv2 is None or np is None:
        raise RuntimeError("Postprocess stage requires numpy and opencv-python (cv2) in the current Python environment.")


def _make_run_id():
    return v0._make_run_id()


def _find_known_good_single_body_scene(canonical, scene_id=DEFAULT_SCENE_ID, asset_id=DEFAULT_KNOWN_GOOD_ASSET_ID):
    for scene in v0._scene_presets(canonical):
        if str(scene["scene_id"]) != str(scene_id):
            continue
        for body_index, body_spec in enumerate(scene["body_specs"]):
            if str(body_spec["asset_id"]) != str(asset_id):
                continue
            return {
                "scene": scene,
                "body_index": int(body_index),
                "body_spec": dict(body_spec),
                "motion_payload": dict(scene["motion_triplet_payload"][body_index]),
            }
    raise RuntimeError(f"Could not find asset {asset_id} in scene {scene_id}.")


def _stress_post_rotation_config(
    yaw_amplitude_deg=45.0,
    pitch_amplitude_deg=12.0,
    roll_amplitude_deg=9.0,
    draw_debug_text=False,
):
    return v1._erp_post_rotation_config(
        enabled=True,
        yaw_amplitude_deg=float(yaw_amplitude_deg),
        pitch_amplitude_deg=float(pitch_amplitude_deg),
        roll_amplitude_deg=float(roll_amplitude_deg),
        draw_debug_text=bool(draw_debug_text),
    )


def _build_stress_sequence_spec(canonical, frame_count, fixed_seed):
    selection = _find_known_good_single_body_scene(canonical)
    scene = selection["scene"]
    body_spec = dict(selection["body_spec"])
    body_spec["x"] = 0.0
    body_spec["y"] = 0.0
    body_spec["z"] = 0.0
    body_spec["body_slot"] = 0
    motion_payload = dict(selection["motion_payload"])
    scene_payload = {
        "scene_id": scene["scene_id"],
        "stress_test_name": DEFAULT_STRESS_TEST_NAME,
        "asset_id": body_spec["asset_id"],
        "motion_payload": motion_payload,
    }
    digest_source = json.dumps(scene_payload, sort_keys=True, separators=(",", ":"))
    return {
        "sequence_index": 0,
        "sequence_id": "seq_0000",
        "sequence_name": f"bedlam360_v1_stress_{DEFAULT_STRESS_TEST_NAME}",
        "frame_start": 0,
        "frame_end": int(frame_count - 1),
        "frame_count": int(frame_count),
        "scene_id": scene["scene_id"],
        "scene_label": f"stress_{scene['label']}",
        "scene_signature": f"{scene['scene_id']}__{v0.hashlib.sha1(digest_source.encode('utf-8')).hexdigest()[:12]}",
        "motion_triplet_signature": f"stress_single__{body_spec['asset_id']}",
        "motion_triplet_payload": [motion_payload],
        "body_specs": [body_spec],
        "camera_policy": DEFAULT_STRESS_TEST_NAME,
        "distance_regime": "extreme_close",
        "camera_pose_cm_deg": {"x": 0.0, "y": -360.0, "z": 155.0, "yaw": 90.0, "pitch": 0.0, "roll": 0.0},
        "body_forward_yaw_offset_deg": scene.get("body_forward_yaw_offset_deg", v0.DEFAULT_BODY_FORWARD_YAW_OFFSET_DEG),
        "seed_bundle": v1.build_seed_bundle(int(fixed_seed), sequence_id="seq_0000"),
        "stress_test_name": DEFAULT_STRESS_TEST_NAME,
        "known_good_asset_id": body_spec["asset_id"],
    }


def _sample_extreme_close_orbit_pushin_trajectory(sequence_spec, fixed_seed, attempt_index):
    body_spec = dict(sequence_spec["body_specs"][0])
    centroid = {
        "x": float(body_spec["x"]),
        "y": float(body_spec["y"]),
        "z": float(body_spec["z"]) + 115.0,
    }
    frame_count = int(sequence_spec["frame_count"])
    rng = random.Random(int(fixed_seed) + 5003 + (attempt_index * 211))
    start_radius_cm = rng.uniform(320.0, 390.0)
    end_radius_cm = rng.uniform(72.0, 105.0)
    orbit_span_deg = rng.uniform(70.0, 140.0)
    orbit_mid_deg = rng.uniform(-35.0, 35.0)
    orbit_start_deg = orbit_mid_deg - (orbit_span_deg * 0.5)
    orbit_end_deg = orbit_mid_deg + (orbit_span_deg * 0.5)
    base_height_cm = rng.uniform(145.0, 172.0)
    height_span_cm = rng.uniform(8.0, 20.0)
    lateral_sway_cm = rng.uniform(5.0, 12.0)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    poses = []
    metadata = []
    for frame_offset in range(frame_count):
        progress = 0.0 if frame_count <= 1 else float(frame_offset) / float(frame_count - 1)
        eased = 0.5 - 0.5 * math.cos(progress * math.pi)
        azimuth_deg = orbit_start_deg + ((orbit_end_deg - orbit_start_deg) * eased)
        radius_cm = start_radius_cm + ((end_radius_cm - start_radius_cm) * eased)
        radians_azimuth = math.radians(azimuth_deg)
        camera_loc = {
            "x": centroid["x"] + (math.cos(radians_azimuth) * radius_cm) + math.sin((2.0 * math.pi * progress) + phase) * lateral_sway_cm,
            "y": centroid["y"] + (math.sin(radians_azimuth) * radius_cm) + math.cos((2.0 * math.pi * progress) + phase) * (lateral_sway_cm * 0.5),
            "z": base_height_cm + math.sin((2.0 * math.pi * progress * 0.7) + phase) * height_span_cm,
        }
        pitch_deg, yaw_deg, _roll = v1._look_at_rotation(camera_loc, centroid)
        roll_deg = math.sin((2.0 * math.pi * progress * 1.2) + phase) * 1.25
        pose = {
            "x": float(camera_loc["x"]),
            "y": float(camera_loc["y"]),
            "z": float(camera_loc["z"]),
            "yaw": float(v0._normalize_yaw_deg(yaw_deg)),
            "pitch": float(max(-35.0, min(35.0, pitch_deg))),
            "roll": float(max(-5.0, min(5.0, roll_deg))),
        }
        poses.append(pose)
        metadata.append(
            {
                "frame_offset": int(frame_offset),
                "progress": float(progress),
                "eased_progress": float(eased),
                "azimuth_deg": float(azimuth_deg),
                "radius_cm": float(radius_cm),
                "camera_pose_cm_deg": dict(pose),
                "target_loc_cm": dict(centroid),
            }
        )
    return {
        "poses": poses,
        "metadata": metadata,
        "summary": {
            "camera_policy": DEFAULT_STRESS_TEST_NAME,
            "distance_regime": "extreme_close",
            "attempt_index": int(attempt_index),
            "start_radius_cm": float(start_radius_cm),
            "end_radius_cm": float(end_radius_cm),
            "orbit_start_deg": float(orbit_start_deg),
            "orbit_end_deg": float(orbit_end_deg),
            "orbit_span_deg": float(orbit_span_deg),
            "base_height_cm": float(base_height_cm),
            "height_span_cm": float(height_span_cm),
            "lateral_sway_cm": float(lateral_sway_cm),
        },
    }


def _select_validated_stress_trajectory(mini, level_sequence_info, sequence_spec, fixed_seed, reports_root, max_attempts=12):
    attempt_reports = []
    accepted_report = None
    accepted_bundle = None
    for attempt_index in range(int(max_attempts)):
        bundle = _sample_extreme_close_orbit_pushin_trajectory(sequence_spec, fixed_seed, attempt_index)
        validation = v1._validate_camera_trajectory(
            mini=mini,
            level_sequence_info=level_sequence_info,
            frame_start=sequence_spec["frame_start"],
            frame_end=sequence_spec["frame_end"],
            trajectory_bundle=bundle,
            distance_regime=sequence_spec["distance_regime"],
        )
        report = {
            "stress_test_name": DEFAULT_STRESS_TEST_NAME,
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


def _compute_distance_series(accepted_report):
    per_frame = list((accepted_report.get("validation") or {}).get("per_frame") or [])
    values = []
    for row in per_frame:
        val = row.get("min_effective_clearance_cm")
        if val is None:
            continue
        values.append(float(val))
    if not values:
        return {"count": 0, "min_cm": None, "mean_cm": None, "max_cm": None}
    return {
        "count": int(len(values)),
        "min_cm": float(min(values)),
        "mean_cm": float(sum(values) / float(len(values))),
        "max_cm": float(max(values)),
    }


def _render_stress_sequence(canonical, mini, run_id, run_root, sequence_spec, fixed_seed):
    sequence_root = _ensure_dir(Path(run_root) / "raw" / "sequences" / sequence_spec["sequence_id"])
    reports_root = v1._trajectory_reports_root(run_root)
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

    trajectory_bundle, accepted_report, attempt_reports = _select_validated_stress_trajectory(
        mini=mini,
        level_sequence_info=level_sequence_info,
        sequence_spec=sequence_spec,
        fixed_seed=int(fixed_seed),
        reports_root=reports_root,
    )
    if trajectory_bundle is None or accepted_report is None:
        raise RuntimeError(f"No safe stress-test trajectory found after {len(attempt_reports)} attempts.")

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
    distance_stats = _compute_distance_series(accepted_report)
    trajectory_csv_rows = []
    for row in accepted_report["validation"]["per_frame"]:
        pose = row["camera_pose_cm_deg"]
        trajectory_csv_rows.append(
            {
                "frame_index": int(row["frame_index"]),
                "x": pose["x"],
                "y": pose["y"],
                "z": pose["z"],
                "yaw": pose["yaw"],
                "pitch": pose["pitch"],
                "roll": pose["roll"],
                "min_effective_clearance_cm": row.get("min_effective_clearance_cm"),
                "closest_body_asset_id": row.get("closest_body_asset_id"),
                "camera_inside_body": bool(row.get("camera_inside_body")),
            }
        )
    _write_csv(
        sequence_root / "trajectory.csv",
        trajectory_csv_rows,
        ["frame_index", "x", "y", "z", "yaw", "pitch", "roll", "min_effective_clearance_cm", "closest_body_asset_id", "camera_inside_body"],
    )
    _write_json(sequence_root / "trajectory.json", {"poses": trajectory_bundle["poses"], "metadata": trajectory_bundle["metadata"]})
    manifest = {
        "kind": "BEDLAM360-v1-stress-sequence",
        "stress_test_name": DEFAULT_STRESS_TEST_NAME,
        "sequence_id": sequence_spec["sequence_id"],
        "sequence_name": sequence_spec["sequence_name"],
        "unreal_sequence_name": unreal_sequence_name,
        "frame_start": int(sequence_spec["frame_start"]),
        "frame_end": int(sequence_spec["frame_end"]),
        "frame_count": int(sequence_spec["frame_count"]),
        "scene_id": sequence_spec["scene_id"],
        "camera_policy": sequence_spec["camera_policy"],
        "distance_regime": sequence_spec["distance_regime"],
        "seed_bundle": sequence_spec.get("seed_bundle"),
        "known_good_asset_id": sequence_spec["known_good_asset_id"],
        "body_specs": body_specs,
        "motion_triplet_payload": sequence_spec["motion_triplet_payload"],
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
        "trajectory_distance_stats_cm": distance_stats,
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
    manifest_path = sequence_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "sequence_id": sequence_spec["sequence_id"],
        "sequence_root": str(sequence_root),
        "manifest_path": str(manifest_path),
        "trajectory_report_path": manifest["trajectory_report_path"],
        "range_result": range_result,
    }


def render_stress_test(
    output_root=DEFAULT_OUTPUT_ROOT,
    frame_count=DEFAULT_FRAME_COUNT,
    fixed_seed=DEFAULT_FIXED_SEED,
    erp_post_rotation_config=None,
):
    if unreal is None:
        raise RuntimeError("Render stage requires Unreal Python.")
    canonical = _load_canonical_module()
    mini = _load_mini_module()
    run_id = _make_run_id()
    run_root = _ensure_dir(Path(output_root) / run_id)
    sequence_spec = _build_stress_sequence_spec(canonical, int(frame_count), int(fixed_seed))
    rendered = _render_stress_sequence(
        canonical=canonical,
        mini=mini,
        run_id=run_id,
        run_root=run_root,
        sequence_spec=sequence_spec,
        fixed_seed=int(fixed_seed),
    )
    trajectory_report_summary = v1._write_trajectory_summary_reports(run_root, [rendered])
    render_manifest = {
        "kind": "BEDLAM360-v1-stress-render",
        "version": 1,
        **v1.build_pipeline_versions(),
        "run_id": run_id,
        "created_at_utc": _utc_now().isoformat(),
        "dataset_root": str(run_root),
        "stress_test_name": DEFAULT_STRESS_TEST_NAME,
        "fixed_seed": int(fixed_seed),
        "seed_bundle": v1.build_seed_bundle(int(fixed_seed), sequence_id="seq_0000"),
        "target_frames_requested": int(frame_count),
        "target_frames": int(frame_count),
        "frame_count": int(frame_count),
        "sequence_count": 1,
        "body_count": 1,
        "sequence_specs": [sequence_spec],
        "rendered_sequences": [rendered],
        "trajectory_report_summary": trajectory_report_summary,
        "erp_post_rotation_config": dict(erp_post_rotation_config or _stress_post_rotation_config()),
        "projection_mode": DEFAULT_ERP_PROJECTION_MODE,
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
    _log(f"Stress-test render root: {run_root}")
    return run_root


def _simple_line_plot_png(path, values, width=1280, height=360, title="Camera-body distance (cm)"):
    _require_postprocess_deps()
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(canvas, (50, 20), (width - 20, height - 40), (220, 220, 220), 1, cv2.LINE_AA)
    if values:
        arr = np.asarray(values, dtype=np.float64)
        ymin = float(np.min(arr))
        ymax = float(np.max(arr))
        yrange = max(1e-6, ymax - ymin)
        pts = []
        for idx, value in enumerate(arr):
            x = 50 + int(round((idx / max(1, len(arr) - 1)) * (width - 70)))
            y = (height - 40) - int(round(((float(value) - ymin) / yrange) * (height - 70)))
            pts.append((x, y))
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, (20, 90, 220), 2, cv2.LINE_AA)
        for text, y in [(f"max {ymax:.1f}", 38), (f"min {ymin:.1f}", height - 48)]:
            cv2.putText(canvas, text, (width - 190, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (56, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return path


def _load_rgb_image(path):
    _require_postprocess_deps()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def _try_load_rgb_image(path):
    _require_postprocess_deps()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def _read_projection_context(run_root):
    _require_postprocess_deps()
    gt_alignment, _validator = _load_stress_postprocess_modules()
    run_root = Path(run_root)
    raw_root = run_root / "raw"
    manifest = v0._read_json(run_root / "manifest.json")
    frames = v0._read_json(raw_root / "metadata" / "frames.json")
    frame_mapping = v0._read_json(raw_root / "metadata" / "frame_mapping.json")
    projections = np.load(raw_root / "projections2d" / "projections2d_erp.npz")
    vertices_npz = np.load(raw_root / "vertices" / "vertices.npz")
    body_world = np.load(raw_root / "metadata" / "body_world_transforms.npy")
    camera_world = np.load(raw_root / "metadata" / "camera_world_transforms.npy")
    return {
        "run_root": run_root,
        "raw_root": raw_root,
        "manifest": manifest,
        "frames": frames,
        "frame_mapping": frame_mapping,
        "joints2d": projections["joints2d"],
        "joints_available": projections["available"],
        "vertices": vertices_npz["vertices"],
        "vertices_available": vertices_npz["available"],
        "body_world": body_world,
        "camera_world": camera_world,
    }


def _draw_sparse_vertices(image, points_uv, valid_mask, color):
    _require_postprocess_deps()
    image = image.copy()
    for point, is_valid in zip(points_uv, valid_mask):
        if not bool(is_valid):
            continue
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        cv2.circle(image, (x, y), 2, color, -1, cv2.LINE_AA)
    return image


def _build_original_overlay_exports(run_root, vertex_step=100):
    _require_postprocess_deps()
    gt_alignment, validator = _load_stress_postprocess_modules()
    ctx = _read_projection_context(run_root)
    raw_root = ctx["raw_root"]
    overlay_root = _ensure_dir(raw_root / "projections2d" / "original_overlays")
    rgb_dir = _ensure_dir(overlay_root / "rgb")
    joints_dir = _ensure_dir(overlay_root / "overlays_joints")
    vertices_dir = _ensure_dir(overlay_root / "overlays_vertices")
    _, _, _, preview_tools = _load_postprocess_modules()
    projection_mode = str(ctx["manifest"].get("erp_projection_mode", DEFAULT_ERP_PROJECTION_MODE))
    yaw_offset_deg = float(ctx["manifest"].get("erp_yaw_zero_offset_deg_used", gt_alignment.DEFAULT_FALLBACK_ERP_YAW_ZERO_OFFSET_DEG))
    rgb_paths = []
    joint_paths = []
    vertex_paths = []
    for frame_i, frame_meta in enumerate(ctx["frames"]):
        image_path = raw_root / "images" / frame_meta["image_png"]
        image = _load_rgb_image(image_path)
        rgb_out = rgb_dir / Path(frame_meta["image_png"]).name
        cv2.imwrite(str(rgb_out), image)
        joints_image = image.copy()
        vertices_image = image.copy()
        body_map = ctx["frame_mapping"][frame_i]["body_frame_mapping"]
        camera_inv, _camera_world = gt_alignment._camera_inverse_for_erp(  # pylint: disable=protected-access
            ctx["camera_world"][frame_i],
            projection_mode=projection_mode,
        )
        for body_i in range(min(int(ctx["joints2d"].shape[1]), int(len(body_map)))):
            if not bool(ctx["joints_available"][frame_i, body_i].any()):
                continue
            asset_id = body_map[body_i].get("asset_id", f"body_{body_i}")
            color = gt_alignment.BODY_COLORS_BGR[body_i % len(gt_alignment.BODY_COLORS_BGR)]
            points_uv = np.asarray(ctx["joints2d"][frame_i, body_i], dtype=np.float32)
            valid = np.asarray(ctx["joints_available"][frame_i, body_i], dtype=bool)
            joints_image = validator._draw_seam_aware_skeleton(joints_image, points_uv, valid, color)  # pylint: disable=protected-access
            joints_image = gt_alignment._draw_body_joints(  # pylint: disable=protected-access
                joints_image,
                points_uv,
                valid,
                color,
                asset_id,
            )
            if bool(ctx["vertices_available"][frame_i, body_i]):
                verts_local_m = np.asarray(ctx["vertices"][frame_i, body_i][:: max(1, int(vertex_step))], dtype=np.float32)
                verts_local_cm = (gt_alignment.SMPL_TO_UNREAL_LOCAL_CM @ verts_local_m.T).T
                verts_world_cm = gt_alignment._transform_points(verts_local_cm, ctx["body_world"][frame_i, body_i])  # pylint: disable=protected-access
                verts_camera_cm = gt_alignment._transform_points(verts_world_cm, camera_inv)  # pylint: disable=protected-access
                verts_uv, verts_valid = gt_alignment._project_camera_xyz_to_erp_with_yaw_offset(  # pylint: disable=protected-access
                    verts_camera_cm,
                    width=int(image.shape[1]),
                    height=int(image.shape[0]),
                    yaw_offset_deg=yaw_offset_deg,
                )
                vertices_image = _draw_sparse_vertices(vertices_image, verts_uv, verts_valid, color)
        joints_path = joints_dir / f"{Path(frame_meta['image_png']).stem}_joints_overlay.png"
        verts_path = vertices_dir / f"{Path(frame_meta['image_png']).stem}_vertices_overlay.png"
        cv2.imwrite(str(joints_path), joints_image)
        cv2.imwrite(str(verts_path), vertices_image)
        rgb_paths.append(rgb_out)
        joint_paths.append(joints_path)
        vertex_paths.append(verts_path)
    preview_dir = _ensure_dir(overlay_root / "previews")
    preview_rgb_mp4 = preview_dir / "original_rgb.mp4"
    preview_joints_mp4 = preview_dir / "original_rgb_joints.mp4"
    preview_vertices_mp4 = preview_dir / "original_rgb_vertices.mp4"
    preview_tools.export_mp4_preview("stress_rgb", rgb_paths, preview_rgb_mp4, fps=12)
    preview_tools.export_mp4_preview("stress_joints", joint_paths, preview_joints_mp4, fps=12)
    preview_tools.export_mp4_preview("stress_vertices", vertex_paths, preview_vertices_mp4, fps=12)
    return {
        "rgb_dir": str(rgb_dir),
        "joints_dir": str(joints_dir),
        "vertices_dir": str(vertices_dir),
        "preview_rgb_mp4": str(preview_rgb_mp4),
        "preview_joints_mp4": str(preview_joints_mp4),
        "preview_vertices_mp4": str(preview_vertices_mp4),
    }


def _sample_frame_indices(total_count, closest_index):
    candidates = sorted({0, max(0, total_count // 4), max(0, total_count // 2), max(0, (3 * total_count) // 4), max(0, total_count - 1), int(closest_index)})
    return [idx for idx in candidates if 0 <= idx < total_count]


def _resolve_nearest_auditable_frame(run_root, requested_frame_index, body_id=0):
    _require_postprocess_deps()
    ctx = _read_projection_context(run_root)
    requested_frame_index = int(requested_frame_index)
    body_id = int(body_id)
    joints_npz = np.load(Path(run_root) / "raw" / "joints3d" / "joints3d.npz")
    availability = np.asarray(joints_npz["available"][:, body_id], dtype=bool)
    valid_indices = [idx for idx, valid in enumerate(availability) if bool(valid)]
    if not valid_indices:
        return None
    return int(min(valid_indices, key=lambda idx: abs(int(idx) - requested_frame_index)))


def _make_horizontal_sheet(images, labels, output_path, panel_width=420):
    _require_postprocess_deps()
    panels = []
    for label, image in zip(labels, images):
        h, w = image.shape[:2]
        scale = float(panel_width) / float(max(1, w))
        resized = cv2.resize(image, (panel_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        label_band = np.full((34, panel_width, 3), 18, dtype=np.uint8)
        cv2.putText(label_band, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        panels.append(np.vstack([label_band, resized]))
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.full((max_h - panel.shape[0], panel.shape[1], 3), 245, dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    sheet = np.hstack(padded)
    cv2.imwrite(str(output_path), sheet)
    return output_path


def _write_stress_contact_sheets(run_root, closest_frame_index):
    _require_postprocess_deps()
    run_root = Path(run_root)
    raw_root = run_root / "raw"
    sheet_root = _ensure_dir(run_root / "stress_validation")
    alignment_overlay_dir = raw_root / "projections2d" / "erp_alignment" / "overlays"
    original_rgb_dir = raw_root / "projections2d" / "original_overlays" / "rgb"
    original_vertices_dir = raw_root / "projections2d" / "original_overlays" / "overlays_vertices"
    frames = v0._read_json(raw_root / "metadata" / "frames.json")
    sample_indices = _sample_frame_indices(len(frames), int(closest_frame_index))
    row_panels = []
    labels = []
    for frame_index in sample_indices:
        frame_meta = frames[frame_index]
        stem = Path(frame_meta["image_png"]).stem
        rgb = _load_rgb_image(original_rgb_dir / frame_meta["image_png"])
        joints = _try_load_rgb_image(alignment_overlay_dir / f"{stem}_joints_overlay.png")
        vertices = _try_load_rgb_image(original_vertices_dir / f"{stem}_vertices_overlay.png")
        if joints is None:
            joints = rgb.copy()
        if vertices is None:
            vertices = rgb.copy()
        row_panels.append(np.vstack([rgb, joints, vertices]))
        labels.append(f"frame {frame_index}")
    if row_panels:
        _make_horizontal_sheet(row_panels, labels, sheet_root / "sampled_frames_contact_sheet.png", panel_width=340)
    closest_stem = Path(frames[int(closest_frame_index)]["image_png"]).stem
    closest_rgb = _load_rgb_image(original_rgb_dir / frames[int(closest_frame_index)]["image_png"])
    closest_joints = _try_load_rgb_image(alignment_overlay_dir / f"{closest_stem}_joints_overlay.png")
    closest_vertices = _try_load_rgb_image(original_vertices_dir / f"{closest_stem}_vertices_overlay.png")
    closest_images = [
        closest_rgb,
        closest_rgb.copy() if closest_joints is None else closest_joints,
        closest_rgb.copy() if closest_vertices is None else closest_vertices,
    ]
    closest_labels = ["RGB", "Joints overlay", "Sparse vertices"]
    closest_contact_path = _make_horizontal_sheet(
        closest_images,
        closest_labels,
        sheet_root / "closest_frame_contact_sheet.png",
        panel_width=520,
    )
    return {
        "sampled_frames_contact_sheet": str(sheet_root / "sampled_frames_contact_sheet.png"),
        "closest_frame_contact_sheet": str(closest_contact_path),
        "sampled_frame_indices": sample_indices,
    }


def _summarize_trajectory_validation(run_root):
    _require_postprocess_deps()
    accepted_csv = Path(run_root) / "trajectory_reports" / "accepted_trajectories.csv"
    rows = []
    if accepted_csv.exists():
        with accepted_csv.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
    accepted_json = sorted((Path(run_root) / "trajectory_reports" / "accepted").glob("*.json"))
    accepted_report = v0._read_json(accepted_json[0]) if accepted_json else {}
    per_frame = list((accepted_report.get("validation") or {}).get("per_frame") or [])
    distance_values = [float(row["min_effective_clearance_cm"]) for row in per_frame if row.get("min_effective_clearance_cm") is not None]
    plot_root = _ensure_dir(Path(run_root) / "stress_validation")
    plot_path = _simple_line_plot_png(plot_root / "camera_body_distance_plot.png", distance_values)
    _write_csv(
        plot_root / "camera_body_distance_series.csv",
        [
            {
                "frame_index": int(row["frame_index"]),
                "min_effective_clearance_cm": row.get("min_effective_clearance_cm"),
                "closest_body_asset_id": row.get("closest_body_asset_id"),
                "camera_inside_body": bool(row.get("camera_inside_body")),
            }
            for row in per_frame
        ],
        ["frame_index", "min_effective_clearance_cm", "closest_body_asset_id", "camera_inside_body"],
    )
    return {
        "accepted_trajectory_rows": rows,
        "distance_plot_path": str(plot_path),
        "distance_series_csv": str(plot_root / "camera_body_distance_series.csv"),
        "closest_frame_index": None if not per_frame else int(min(per_frame, key=lambda item: float(item.get("min_effective_clearance_cm") if item.get("min_effective_clearance_cm") is not None else 1e9))["frame_index"]),
        "closest_body_asset_id": None if not per_frame else min(per_frame, key=lambda item: float(item.get("min_effective_clearance_cm") if item.get("min_effective_clearance_cm") is not None else 1e9)).get("closest_body_asset_id"),
        "min_distance_cm": None if not distance_values else float(min(distance_values)),
        "mean_distance_cm": None if not distance_values else float(sum(distance_values) / len(distance_values)),
        "max_distance_cm": None if not distance_values else float(max(distance_values)),
    }


def postprocess_stress_test(
    run_root,
    extra_npz_roots=None,
    smplx_model_roots=None,
    erp_projection_mode=DEFAULT_ERP_PROJECTION_MODE,
):
    _require_postprocess_deps()
    _gt_alignment, validator = _load_stress_postprocess_modules()
    run_root = Path(run_root)
    render_manifest = v0._read_json(run_root / "render_manifest.json")
    erp_post_rotation_config = dict(render_manifest.get("erp_post_rotation_config") or _stress_post_rotation_config())
    v1.postprocess_v1_dataset(
        run_root=run_root,
        extra_npz_roots=extra_npz_roots,
        smplx_model_roots=smplx_model_roots,
        erp_projection_mode=erp_projection_mode,
        erp_post_rotation_config=erp_post_rotation_config,
    )
    original_overlay_exports = _build_original_overlay_exports(run_root)
    validator.validate_projection(run_root)
    close_human_qa_root = validator.build_close_human_qa_pack(run_root, max_cases=24, vertex_step=100)
    trajectory_summary = _summarize_trajectory_validation(run_root)
    closest_frame_index = int(trajectory_summary["closest_frame_index"]) if trajectory_summary["closest_frame_index"] is not None else 0
    audit_frame_index = _resolve_nearest_auditable_frame(run_root, closest_frame_index, body_id=0)
    gt_render_audit = None
    gt_render_audit_note = None
    gt_availability_report_path = run_root / "raw" / "metadata" / "gt_availability_report.json"
    gt_availability_report = v0._read_json(gt_availability_report_path) if gt_availability_report_path.exists() else {}
    if audit_frame_index is None:
        gt_render_audit_note = "Skipped GT/render consistency audit because joints3d availability is false for every frame/body in this stress-test run."
    else:
        gt_render_audit = validator.audit_gt_render_consistency(run_root, frame_index=audit_frame_index, body_id=0, vertex_step=100)
    contact_sheets = _write_stress_contact_sheets(run_root, closest_frame_index=closest_frame_index)
    stress_manifest = {
        "kind": "BEDLAM360-v1-stress-postprocess",
        "created_at_utc": _utc_now().isoformat(),
        "run_root": str(run_root),
        "stress_test_name": DEFAULT_STRESS_TEST_NAME,
        "erp_projection_mode": str(erp_projection_mode),
        "erp_post_rotation_config": erp_post_rotation_config,
        "original_overlay_exports": original_overlay_exports,
        "post_rotated_exports": {
            "root": str(run_root / "raw" / "projections2d" / "erp_post_rotation"),
            "report_json": str(run_root / "raw" / "projections2d" / "erp_post_rotation" / "post_rotation_report.json"),
            "schedule_json": str(run_root / "raw" / "projections2d" / "erp_post_rotation" / "post_rotation_schedule.json"),
        },
        "close_human_qa_root": str(close_human_qa_root),
        "trajectory_validation_summary": trajectory_summary,
        "gt_render_audit_frame_index": None if audit_frame_index is None else int(audit_frame_index),
        "gt_availability_report_path": str(gt_availability_report_path),
        "gt_availability_report": gt_availability_report,
        "gt_render_audit_note": gt_render_audit_note,
        "contact_sheets": contact_sheets,
        "gt_render_consistency_audit": gt_render_audit,
        "quality_report_csv": str(run_root / "quality_report.csv"),
        "manifest_json": str(run_root / "manifest.json"),
    }
    stress_manifest_path = run_root / "stress_test_manifest.json"
    _write_json(stress_manifest_path, stress_manifest)
    _log(f"Stress-test postprocess manifest: {stress_manifest_path}")
    return run_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("render", "postprocess"), default="render")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-root", default="")
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--fixed-seed", type=int, default=DEFAULT_FIXED_SEED)
    parser.add_argument("--npz-root", action="append", default=[])
    parser.add_argument("--smplx-model-root", action="append", default=[str(DEFAULT_SMPLX_MODEL_ROOT)])
    parser.add_argument("--erp-post-rotation-yaw-amplitude", type=float, default=45.0)
    parser.add_argument("--erp-post-rotation-pitch-amplitude", type=float, default=12.0)
    parser.add_argument("--erp-post-rotation-roll-amplitude", type=float, default=9.0)
    parser.add_argument("--no-debug-text", action="store_true")
    args = parser.parse_args()

    erp_post_rotation_config = _stress_post_rotation_config(
        yaw_amplitude_deg=float(args.erp_post_rotation_yaw_amplitude),
        pitch_amplitude_deg=float(args.erp_post_rotation_pitch_amplitude),
        roll_amplitude_deg=float(args.erp_post_rotation_roll_amplitude),
        draw_debug_text=not bool(args.no_debug_text),
    )

    if args.stage == "render":
        run_root = render_stress_test(
            output_root=args.output_root,
            frame_count=int(args.frame_count),
            fixed_seed=int(args.fixed_seed),
            erp_post_rotation_config=erp_post_rotation_config,
        )
    else:
        if not args.run_root:
            raise RuntimeError("--run-root is required for --stage postprocess")
        run_root = postprocess_stress_test(
            run_root=args.run_root,
            extra_npz_roots=args.npz_root,
            smplx_model_roots=args.smplx_model_root,
            erp_projection_mode=DEFAULT_ERP_PROJECTION_MODE,
        )
    _log(f"Completed stage={args.stage} run_root={run_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if unreal is not None:
            unreal.log_error(traceback.format_exc())
        raise
