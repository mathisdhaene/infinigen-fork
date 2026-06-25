import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys

import unreal


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bedlam360_mini_validation
import capture_scene_cube
import debug_geometrycache_levelsequence_validation as ls_validate

bedlam360_mini_validation = importlib.reload(bedlam360_mini_validation)
capture_scene_cube = importlib.reload(capture_scene_cube)
ls_validate = importlib.reload(ls_validate)


DEFAULT_OUTPUT_ROOT = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_dynamic_motion_scan")
DEFAULT_CAPTURE_ACTOR_LABEL = "SceneCaptureCube"
DEFAULT_FPS = 30
DEFAULT_QUICK_NUM_FRAMES = 180
DEFAULT_QUICK_SAMPLE_FRAMES = [0, 60, 120]
DEFAULT_QUICK_MAX_ASSETS = 20
DEFAULT_CAMERA_MODE = "fixed"
DEFAULT_TOPK_FULL = 5


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


def _bounds_motion_delta(a, b):
    if a is None or b is None:
        return 0.0
    origin_a = a["origin_cm"]
    origin_b = b["origin_cm"]
    extent_a = a["extent_cm"]
    extent_b = b["extent_cm"]
    origin_delta = (
        abs(origin_a["x"] - origin_b["x"])
        + abs(origin_a["y"] - origin_b["y"])
        + abs(origin_a["z"] - origin_b["z"])
    )
    extent_delta = (
        abs(extent_a["x"] - extent_b["x"])
        + abs(extent_a["y"] - extent_b["y"])
        + abs(extent_a["z"] - extent_b["z"])
    )
    return float(origin_delta + extent_delta)


def _get_actor_bounds_record(actor):
    if actor is None:
        return None
    try:
        bounds = actor.get_actor_bounds(False)
    except Exception:
        try:
            bounds = actor.get_actor_bounds(only_colliding_components=False)
        except Exception:
            return None
    if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
        return None
    origin, extent = bounds[0], bounds[1]
    return {
        "origin_cm": {"x": origin.x, "y": origin.y, "z": origin.z},
        "extent_cm": {"x": extent.x, "y": extent.y, "z": extent.z},
    }


def _infer_motion_type(mean_center_crop_diff, bounds_motion_score):
    if bounds_motion_score >= 20.0:
        return "large_body_motion"
    if bounds_motion_score >= 8.0:
        return "turning_or_arm_swing"
    if mean_center_crop_diff >= 2.0:
        return "articulation"
    if mean_center_crop_diff >= 0.5:
        return "subtle_articulation"
    return "low_motion"


def _run_contact_sheet(image_paths, output_path, cols=3):
    command = [
        "python3",
        str(bedlam360_mini_validation.PREVIEW_SCRIPT_PATH),
        "contact-sheet",
        str(output_path),
    ] + [str(path) for path in image_paths] + ["--cols", str(int(cols))]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "contact_sheet_failed"
        unreal.log_warning(f"[BEDLAM360][MOTION_SCAN] contact sheet failed: {stderr}")
        return {"attempted": True, "success": False, "reason": stderr}
    return {"attempted": True, "success": True, "output_path": str(output_path)}


def _write_candidates_csv(rows, csv_path):
    fieldnames = [
        "rank",
        "asset_id",
        "motion_score",
        "mean_center_crop_diff",
        "mean_pixel_diff",
        "bounds_motion_score",
        "duration_seconds",
        "num_frames",
        "motion_type",
        "sample_frames",
        "sample_times_seconds",
        "thumbnail_strip_path",
        "report_json_path",
        "full_validation_report_json_path",
        "full_preview_mp4_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_asset_ids(limit):
    selected = ls_validate._select_dynamic_asset_ids(limit=limit)
    if len(selected) >= limit:
        return selected
    available = ls_validate._list_local_animation_asset_ids()
    for asset_id in available:
        if asset_id not in selected:
            selected.append(asset_id)
        if len(selected) >= limit:
            break
    return selected


def _quick_scan_one_asset(asset_id, output_root, fps, num_frames, sample_frames, camera_mode):
    asset_root = _reset_dir(output_root / asset_id)
    images_dir = _ensure_dir(asset_root / "images")
    metadata_dir = _ensure_dir(asset_root / "metadata")

    ls_validate._destroy_existing_debug_actors("BEDLAM360_ls_validate_")
    ls_validate._remove_or_hide_duplicate_bedlam_actors(bound_actor=None)
    sequence_info = ls_validate._create_level_sequence_for_asset(asset_id, num_frames=num_frames, fps=fps)
    level_sequence = sequence_info["level_sequence"]
    binding_id = sequence_info["binding_id"]

    camera_actor = capture_scene_cube.find_scene_capture_cube(DEFAULT_CAPTURE_ACTOR_LABEL)
    capture_component = capture_scene_cube.get_capture_component(camera_actor)
    texture_target = capture_scene_cube.get_texture_target(capture_component)
    export_lib = unreal.BEDLAM360ExportLibrary

    gc_component = None
    duration_seconds = None
    num_cache_frames = None
    first_eval = ls_validate._evaluate_sequence_frame(level_sequence, 0)
    bound_actor = ls_validate._get_bound_gc_actor(binding_id)
    ls_validate._remove_or_hide_duplicate_bedlam_actors(bound_actor=bound_actor)
    gc_component, binding_info = ls_validate._find_bound_gc_component(binding_id)
    if gc_component is not None:
        state = ls_validate._get_gc_component_state(gc_component)
        duration_seconds = state.get("duration_seconds")
        num_cache_frames = state.get("num_frames")

    max_frame_index = num_frames - 1
    if duration_seconds is not None:
        max_frame_index = min(max_frame_index, int(round(duration_seconds * float(fps))))
    if num_cache_frames is not None:
        max_frame_index = min(max_frame_index, max(0, int(num_cache_frames) - 1))
    sample_frames = sorted({min(max_frame_index, int(frame)) for frame in sample_frames if int(frame) <= max_frame_index})
    if 0 not in sample_frames:
        sample_frames.insert(0, 0)

    frame_records = []
    png_paths = []
    previous_bounds = None
    bounds_motion_values = []
    for sample_frame in sample_frames:
        evaluated_frame = ls_validate._evaluate_sequence_frame(level_sequence, sample_frame)
        bound_actor = ls_validate._get_bound_gc_actor(binding_id)
        ls_validate._remove_or_hide_duplicate_bedlam_actors(bound_actor=bound_actor)
        gc_component, binding_info = ls_validate._find_bound_gc_component(binding_id)
        component_state = None if gc_component is None else ls_validate._get_gc_component_state(gc_component)
        actual_time = None if component_state is None else component_state.get("animation_time_seconds")
        bounds_record = _get_actor_bounds_record(bound_actor)
        bounds_delta = _bounds_motion_delta(previous_bounds, bounds_record) if previous_bounds is not None else None
        if bounds_delta is not None:
            bounds_motion_values.append(bounds_delta)
        previous_bounds = bounds_record

        camera_loc, pitch, yaw, roll = ls_validate._camera_pose_for_frame(sample_frame, num_frames, camera_mode)
        capture_scene_cube.set_actor_pose(camera_actor, camera_loc["x"], camera_loc["y"], camera_loc["z"], pitch, yaw, roll)

        frame_name = f"{asset_id}_scan_frame_{sample_frame:04d}"
        exr_path = images_dir / f"{frame_name}_erp.exr"
        component = capture_component
        component.capture_scene()
        bedlam360_mini_validation._invalidate_editor_viewports()
        exr_ok = export_lib.export_render_target_cube_long_lat_exr(texture_target, str(exr_path))
        if not exr_ok:
            raise RuntimeError(f"EXR export failed for {asset_id} frame {sample_frame}")

        frame_record = {
            "frame_index": sample_frame,
            "time_seconds": sample_frame / float(max(1, fps)),
            "evaluated_frame": first_eval if sample_frame == 0 else evaluated_frame,
            "asset_id": asset_id,
            "asset_path": sequence_info["geometry_cache_asset"].get_path_name(),
            "camera_mode": camera_mode,
            "binding_info": binding_info,
            "actual_time_seconds": actual_time,
            "component_state_after": component_state,
            "bounds_record": bounds_record,
            "bounds_delta_from_previous": bounds_delta,
            "animation_frame_indices": [] if actual_time is None else [int(round(actual_time * float(fps)))],
            "exr_path": str(exr_path),
            "exr_ok": bool(exr_ok),
        }
        pose_json_path = metadata_dir / f"{frame_name}_pose.json"
        pose_json_path.write_text(json.dumps(frame_record, indent=2), encoding="utf-8")
        png_path = images_dir / f"{frame_name}_erp.png"
        preview_status = bedlam360_mini_validation._run_preview_frame(
            image_path=exr_path,
            output_png_path=png_path,
            metadata_json_path=pose_json_path,
            overlay=True,
        )
        frame_record["png_path"] = str(png_path)
        frame_record["preview_png_status"] = preview_status
        pose_json_path.write_text(json.dumps(frame_record, indent=2), encoding="utf-8")
        if exr_path.exists():
            exr_path.unlink()
        frame_records.append(frame_record)
        if preview_status.get("success"):
            png_paths.append(png_path)

    sorted_pngs = sorted(png_paths, key=lambda path: path.name)
    diagnostics_path = asset_root / "temporal_diagnostics.json"
    diagnostics_status = bedlam360_mini_validation._run_temporal_diagnostics(sorted_pngs, diagnostics_path)
    diagnostics = {}
    if diagnostics_status.get("success") and diagnostics_path.is_file():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diff_lookup = {
        item.get("png_path"): {
            "pixel_diff_prev": item.get("mean_absdiff_from_previous"),
            "center_crop_pixel_diff_prev": item.get("center_crop_mean_absdiff_from_previous"),
        }
        for item in diagnostics.get("frames", [])
    }
    pixel_diffs = []
    center_crop_diffs = []
    for frame_record in frame_records:
        diff_record = diff_lookup.get(frame_record.get("png_path")) or {}
        frame_record["pixel_diff_prev"] = diff_record.get("pixel_diff_prev")
        frame_record["center_crop_pixel_diff_prev"] = diff_record.get("center_crop_pixel_diff_prev")
        if frame_record["pixel_diff_prev"] is not None:
            pixel_diffs.append(float(frame_record["pixel_diff_prev"]))
        if frame_record["center_crop_pixel_diff_prev"] is not None:
            center_crop_diffs.append(float(frame_record["center_crop_pixel_diff_prev"]))
        pose_json_path = metadata_dir / f"{Path(frame_record['png_path']).stem.replace('_erp', '')}_pose.json"
        pose_json_path.write_text(json.dumps(frame_record, indent=2), encoding="utf-8")

    mean_pixel_diff = 0.0 if not pixel_diffs else float(sum(pixel_diffs) / len(pixel_diffs))
    mean_center_crop_diff = 0.0 if not center_crop_diffs else float(sum(center_crop_diffs) / len(center_crop_diffs))
    bounds_motion_score = 0.0 if not bounds_motion_values else float(sum(bounds_motion_values) / len(bounds_motion_values))
    motion_score = float(mean_center_crop_diff * 2.0 + mean_pixel_diff + bounds_motion_score * 0.01)
    motion_type = _infer_motion_type(mean_center_crop_diff, bounds_motion_score)

    strip_path = asset_root / f"{asset_id}_contact_sheet.png"
    strip_status = _run_contact_sheet(sorted_pngs, strip_path, cols=len(sorted_pngs))

    report = {
        "asset_id": asset_id,
        "asset_path": sequence_info["geometry_cache_asset"].get_path_name(),
        "duration_seconds": duration_seconds,
        "num_frames": num_cache_frames,
        "sample_frames": sample_frames,
        "sample_times_seconds": [frame / float(max(1, fps)) for frame in sample_frames],
        "camera_mode": camera_mode,
        "level_sequence_summary": sequence_info["level_sequence_summary"],
        "mean_pixel_diff": mean_pixel_diff,
        "mean_center_crop_diff": mean_center_crop_diff,
        "bounds_motion_score": bounds_motion_score,
        "motion_score": motion_score,
        "motion_type": motion_type,
        "movement_detected": bool(mean_center_crop_diff > 0.5 or bounds_motion_score > 5.0),
        "contact_sheet_path": str(strip_path) if strip_status.get("success") else None,
        "frames": frame_records,
    }
    report_path = asset_root / "motion_scan_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    except Exception:
        pass
    return report


def quick_scan_dynamic_motion_candidates(
    output_root=DEFAULT_OUTPUT_ROOT,
    camera_mode=DEFAULT_CAMERA_MODE,
    fps=DEFAULT_FPS,
    num_frames=DEFAULT_QUICK_NUM_FRAMES,
    sample_frames=DEFAULT_QUICK_SAMPLE_FRAMES,
    max_assets=DEFAULT_QUICK_MAX_ASSETS,
    topk_full=DEFAULT_TOPK_FULL,
):
    output_root = _reset_dir(output_root)
    quick_dir = _ensure_dir(output_root / "quick_scan")
    asset_reports_dir = _ensure_dir(quick_dir / "assets")

    asset_ids = _select_asset_ids(limit=max_assets)
    reports = []
    for asset_id in asset_ids:
        unreal.log(f"[BEDLAM360][MOTION_SCAN] quick-scan asset {asset_id}")
        try:
            reports.append(
                _quick_scan_one_asset(
                    asset_id=asset_id,
                    output_root=asset_reports_dir,
                    fps=fps,
                    num_frames=num_frames,
                    sample_frames=sample_frames,
                    camera_mode=camera_mode,
                )
            )
        except Exception as exc:
            unreal.log_warning(f"[BEDLAM360][MOTION_SCAN] quick-scan failed for {asset_id}: {exc}")

    reports.sort(key=lambda item: item.get("motion_score", 0.0), reverse=True)
    rows = []
    for rank, report in enumerate(reports, start=1):
        row = {
            "rank": rank,
            "asset_id": report["asset_id"],
            "motion_score": report.get("motion_score"),
            "mean_center_crop_diff": report.get("mean_center_crop_diff"),
            "mean_pixel_diff": report.get("mean_pixel_diff"),
            "bounds_motion_score": report.get("bounds_motion_score"),
            "duration_seconds": report.get("duration_seconds"),
            "num_frames": report.get("num_frames"),
            "motion_type": report.get("motion_type"),
            "sample_frames": ",".join(str(v) for v in report.get("sample_frames", [])),
            "sample_times_seconds": ",".join(f"{v:.3f}" for v in report.get("sample_times_seconds", [])),
            "thumbnail_strip_path": report.get("contact_sheet_path"),
            "report_json_path": str(asset_reports_dir / report["asset_id"] / "motion_scan_report.json"),
            "full_validation_report_json_path": None,
            "full_preview_mp4_path": None,
        }
        rows.append(row)

    top_reports = reports[:topk_full]
    full_dir = _ensure_dir(output_root / "top5_full_validation")
    for report in top_reports:
        asset_id = report["asset_id"]
        unreal.log(
            f"[BEDLAM360][MOTION_SCAN] top asset {asset_id} score={report['motion_score']:.3f} "
            f"duration={report.get('duration_seconds')} motion_type={report.get('motion_type')}"
        )
        full_report = ls_validate.validate_geometrycache_with_levelsequence(
            asset_id=asset_id,
            num_frames=120,
            fps=fps,
            actor_label=DEFAULT_CAPTURE_ACTOR_LABEL,
            output_root=full_dir,
            camera_mode="fixed",
        )
        mp4_status = full_report.get("mp4_status") or {}
        for row in rows:
            if row["asset_id"] != asset_id:
                continue
            row["full_validation_report_json_path"] = str(full_dir / asset_id / "geometrycache_levelsequence_validation.json")
            row["full_preview_mp4_path"] = mp4_status.get("mp4_path")
            break

    csv_path = output_root / "dynamic_motion_candidates.csv"
    _write_candidates_csv(rows, csv_path)
    manifest_path = output_root / "dynamic_motion_candidates_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "camera_mode": camera_mode,
                "fps": fps,
                "quick_num_frames": num_frames,
                "quick_sample_frames": sample_frames,
                "max_assets": max_assets,
                "topk_full": topk_full,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    unreal.log(f"[BEDLAM360][MOTION_SCAN] Wrote quick-scan ranked candidates: {csv_path}")
    return rows


if __name__ == "__main__":
    quick_scan_dynamic_motion_candidates()
