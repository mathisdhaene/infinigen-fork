import importlib
import csv
import json
from pathlib import Path
import sys

import unreal


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bedlam360_mini_validation
import capture_scene_cube
import reconstruct_full_bedlam_scene
import reconstruct_one_bedlam_body

bedlam360_mini_validation = importlib.reload(bedlam360_mini_validation)
capture_scene_cube = importlib.reload(capture_scene_cube)
reconstruct_full_bedlam_scene = importlib.reload(reconstruct_full_bedlam_scene)
reconstruct_one_bedlam_body = importlib.reload(reconstruct_one_bedlam_body)


DEFAULT_ASSET_ID = "it_4052_3XL_2400"
DEFAULT_OUTPUT_ROOT = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_geometrycache_validation")
DEFAULT_ACTOR_LABEL = "SceneCaptureCube"
DEFAULT_NUM_FRAMES = 120
DEFAULT_FPS = 30
DEFAULT_BATCH_ASSET_IDS = [
    "it_4001_XL_2403",
    "it_4001_XL_2404",
    "it_4029_L_2400",
    "it_4029_L_2402",
    "it_4031_XL_2403",
    "it_4049_2XL_2400",
    "it_4052_3XL_2403",
    "it_4052_3XL_2406",
]


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _reset_dir(path):
    path = Path(path)
    if path.exists():
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_body_pose(asset_id):
    return {
        "asset_id": asset_id,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "comment": "debug_geometrycache_playback_validation",
        "comment_map": {},
        "index": "0",
    }


def _component_state(component):
    state = {}
    for key, getter in (
        ("duration_seconds", "get_duration"),
        ("number_of_frames", "get_number_of_frames"),
        ("animation_time_seconds", "get_animation_time"),
        ("is_playing", "is_playing"),
        ("is_looping", "is_looping"),
        ("playback_speed", "get_playback_speed"),
        ("start_time_offset", "get_start_time_offset"),
    ):
        try:
            value = getattr(component, getter)()
        except Exception:
            value = None
        state[key] = value
    for property_name in ("manual_tick", "running", "looping", "elapsed_time"):
        try:
            state[f"property_{property_name}"] = component.get_editor_property(property_name)
        except Exception:
            state[f"property_{property_name}"] = None
    return state


def _movement_summary(report):
    frames = report.get("frames", [])
    actual_times = [frame.get("actual_time_seconds") for frame in frames if frame.get("actual_time_seconds") is not None]
    target_times = [frame.get("time_seconds") for frame in frames if frame.get("time_seconds") is not None]
    pixel_diffs = [frame.get("pixel_diff_prev") for frame in frames if frame.get("pixel_diff_prev") is not None]
    max_actual_time = max(actual_times) if actual_times else None
    max_time_error = None
    if frames:
        errors = []
        for frame in frames:
            if frame.get("actual_time_seconds") is not None and frame.get("time_seconds") is not None:
                errors.append(abs(frame["actual_time_seconds"] - frame["time_seconds"]))
        if errors:
            max_time_error = max(errors)
    mean_pixel_diff = sum(pixel_diffs) / len(pixel_diffs) if pixel_diffs else None
    movement_detected = bool(mean_pixel_diff is not None and mean_pixel_diff > 0.5 and max_actual_time is not None and max_actual_time > 0.10)
    return {
        "asset_id": report.get("asset_id"),
        "duration_seconds": report.get("duration_seconds"),
        "num_frames": report.get("num_frames"),
        "max_actual_time_seconds": max_actual_time,
        "max_target_time_seconds": max(target_times) if target_times else None,
        "max_time_error_seconds": max_time_error,
        "mean_pixel_diff": mean_pixel_diff,
        "movement_detected": movement_detected,
        "tick_driver_last": None if not frames else (frames[-1].get("playback_status") or {}).get("tick_driver"),
        "streaming_mode_note_last": None if not frames else (frames[-1].get("playback_status") or {}).get("streaming_mode_note"),
    }


def _write_summary_csv(summary_rows, csv_path):
    fieldnames = [
        "asset_id",
        "duration_seconds",
        "num_frames",
        "max_actual_time_seconds",
        "max_target_time_seconds",
        "max_time_error_seconds",
        "mean_pixel_diff",
        "movement_detected",
        "tick_driver_last",
        "streaming_mode_note_last",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def validate_geometrycache_actor_playback(
    asset_id=DEFAULT_ASSET_ID,
    output_root=DEFAULT_OUTPUT_ROOT,
    actor_label=DEFAULT_ACTOR_LABEL,
    num_frames=DEFAULT_NUM_FRAMES,
    fps=DEFAULT_FPS,
):
    output_root = _reset_dir(output_root)
    images_dir = _ensure_dir(output_root / "images")
    metadata_dir = _ensure_dir(output_root / "metadata")
    previews_dir = _ensure_dir(output_root / "previews")

    reconstruct_full_bedlam_scene.clear_existing_bedlam_bodies()
    body_pose = _make_body_pose(asset_id)
    resolved_asset = reconstruct_one_bedlam_body.resolve_body_asset(asset_id)
    actor = reconstruct_one_bedlam_body.spawn_body_actor(body_pose, resolved_asset)
    actor.set_actor_label(f"BEDLAM360_validation_{asset_id}")
    gc_component = actor.get_geometry_cache_component() if hasattr(actor, "get_geometry_cache_component") else None
    if gc_component is None:
        raise RuntimeError("Spawned actor does not expose a GeometryCacheComponent.")

    actor_state_before = _component_state(gc_component)
    unreal.log(f"[BEDLAM360][GC_VALIDATE] actor_state_before={json.dumps(actor_state_before, indent=2)}")

    camera_actor = capture_scene_cube.find_scene_capture_cube(actor_label)
    capture_component = capture_scene_cube.get_capture_component(camera_actor)
    texture_target = capture_scene_cube.get_texture_target(capture_component)
    export_lib = unreal.BEDLAM360ExportLibrary

    camera_loc = {"x": 260.0, "y": -180.0, "z": 140.0}
    pitch, yaw, roll = bedlam360_mini_validation._look_at_rotation(camera_loc, {"x": 0.0, "y": 0.0, "z": 120.0})
    capture_scene_cube.set_actor_pose(camera_actor, camera_loc["x"], camera_loc["y"], camera_loc["z"], pitch, yaw, roll)
    capture_component.capture_scene()

    sequence_pngs = []
    frames = []
    for frame_index in range(num_frames):
        sample_time = frame_index / float(max(1, fps))
        target_state = bedlam360_mini_validation._calculate_motion_state_metadata(gc_component, sample_time)
        playback_status = bedlam360_mini_validation._playback_driven_prepare_targets(
            spawned_bodies=[{"actor_label": actor.get_actor_label(), "body_pose": body_pose, "geometry_cache_component": gc_component}],
            target_animation_states=[target_state],
            sequence_name="gc_validation",
            frame_name=f"frame_{frame_index:04d}",
            tick_dt=1.0 / float(max(1, fps)),
            max_wait_seconds=max(10.0, sample_time + 2.0),
        )

        frame_name = f"gc_validation_frame_{frame_index:04d}"
        hdr_path = images_dir / f"{frame_name}_erp.hdr"
        exr_path = images_dir / f"{frame_name}_erp.exr"
        capture_result = bedlam360_mini_validation.stabilized_capture_and_export(
            actor=camera_actor,
            component=capture_component,
            texture_target=texture_target,
            export_lib=export_lib,
            frame_name=frame_name,
            hdr_path=hdr_path,
            exr_path=exr_path,
            faces_dir=None,
            warmup_ticks=3,
            discard_captures=1,
        )

        frame_record = {
            "frame_sample_index": frame_index,
            "animation_time_seconds": sample_time,
            "target_state": target_state,
            "playback_status": playback_status,
            "component_state_after": _component_state(gc_component),
            "hdr_path": str(hdr_path),
            "exr_path": str(exr_path),
            "camera_pose_cm_deg": {
                "x": camera_loc["x"],
                "y": camera_loc["y"],
                "z": camera_loc["z"],
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
            },
            "animation_frame_indices": [target_state.get("sample_frame_index")],
            "hdr_ok": bool(capture_result["hdr_ok"]),
            "exr_ok": bool(capture_result["exr_ok"]),
        }
        pose_json_path = metadata_dir / f"{frame_name}_pose.json"
        pose_json_path.write_text(json.dumps(frame_record, indent=2), encoding="utf-8")
        png_path = images_dir / f"{frame_name}_erp.png"
        preview_status = bedlam360_mini_validation._run_preview_frame(
            image_path=exr_path if capture_result["exr_ok"] else hdr_path,
            output_png_path=png_path,
            metadata_json_path=pose_json_path,
            overlay=True,
        )
        frame_record["png_path"] = str(png_path)
        frame_record["preview_png_status"] = preview_status
        pose_json_path.write_text(json.dumps(frame_record, indent=2), encoding="utf-8")
        frames.append(frame_record)
        if preview_status.get("success"):
            sequence_pngs.append(png_path)

    mp4_path = previews_dir / f"{asset_id}_validation.mp4"
    mp4_status = bedlam360_mini_validation._run_preview_mp4(
        sequence_name=asset_id,
        png_paths=sorted(sequence_pngs, key=lambda path: path.name),
        output_mp4_path=mp4_path,
        fps=fps,
    )
    diagnostics_path = output_root / "temporal_diagnostics.json"
    diagnostics_status = bedlam360_mini_validation._run_temporal_diagnostics(
        sorted(sequence_pngs, key=lambda path: path.name),
        diagnostics_path,
    )
    diagnostics_data = {}
    if diagnostics_status.get("success") and diagnostics_path.is_file():
        diagnostics_data = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    by_png_path = {
        str(Path(frame_record.get("png_path", "")).resolve()): frame_record
        for frame_record in diagnostics_data.get("frames", [])
    }
    for frame_record in frames:
        png_path = frame_record.get("png_path")
        diagnostic_frame = by_png_path.get(str(Path(png_path).resolve())) if png_path else None
        frame_record["pixel_diff_prev"] = None if diagnostic_frame is None else diagnostic_frame.get("mean_absdiff_from_previous")
        frame_record["frame_index"] = frame_record["frame_sample_index"]
        frame_record["time_seconds"] = frame_record["animation_time_seconds"]
        actor_results = (frame_record.get("playback_status") or {}).get("actor_results", [])
        frame_record["actual_time_seconds"] = None if not actor_results else actor_results[0].get("actual_time_seconds")

    report = {
        "asset_id": asset_id,
        "resolved_asset": resolved_asset,
        "asset_path": resolved_asset.get("body_geometry_cache_path") or resolved_asset.get("unreal_asset_path"),
        "asset_class": "GeometryCache",
        "actor_label": actor.get_actor_label(),
        "component_state_before": actor_state_before,
        "duration_seconds": actor_state_before.get("duration_seconds"),
        "num_frames": actor_state_before.get("number_of_frames"),
        "manual_tick": actor_state_before.get("property_manual_tick"),
        "is_playing": actor_state_before.get("is_playing"),
        "frames": frames,
        "preview_mp4_path": str(mp4_path),
        "preview_mp4_status": mp4_status,
        "temporal_diagnostics_path": str(diagnostics_path),
        "temporal_diagnostics_status": diagnostics_status,
        "temporal_diagnostics": diagnostics_data,
    }
    report["summary"] = _movement_summary(report)
    report_path = output_root / "geometrycache_playback_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    unreal.log(f"[BEDLAM360][GC_VALIDATE] Wrote playback validation report: {report_path}")
    return report_path


def validate_geometrycache_actor_playback_batch(
    asset_ids=DEFAULT_BATCH_ASSET_IDS,
    output_root=DEFAULT_OUTPUT_ROOT,
    actor_label=DEFAULT_ACTOR_LABEL,
    num_frames=DEFAULT_NUM_FRAMES,
    fps=DEFAULT_FPS,
):
    output_root = _reset_dir(output_root)
    summary_rows = []
    report_paths = []
    for asset_id in asset_ids:
        asset_root = output_root / asset_id
        report_path = validate_geometrycache_actor_playback(
            asset_id=asset_id,
            output_root=asset_root,
            actor_label=actor_label,
            num_frames=num_frames,
            fps=fps,
        )
        report_paths.append(str(report_path))
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        summary_rows.append(report.get("summary", _movement_summary(report)))

    summary_csv_path = output_root / "geometrycache_playback_summary.csv"
    _write_summary_csv(summary_rows, summary_csv_path)
    batch_manifest = {
        "asset_ids": list(asset_ids),
        "num_frames": num_frames,
        "fps": fps,
        "report_paths": report_paths,
        "summary_csv_path": str(summary_csv_path),
        "summary_rows": summary_rows,
    }
    manifest_path = output_root / "geometrycache_playback_batch_manifest.json"
    manifest_path.write_text(json.dumps(batch_manifest, indent=2), encoding="utf-8")
    unreal.log(f"[BEDLAM360][GC_VALIDATE] Wrote batch summary CSV: {summary_csv_path}")
    unreal.log(f"[BEDLAM360][GC_VALIDATE] Wrote batch manifest: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    validate_geometrycache_actor_playback_batch()
