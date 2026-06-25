import argparse
import json
import math
import shlex
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SCENES_ROOT = Path("outputs/indoors/bedlam360_v3_scenes")
DEFAULT_STARTERPACK_WHITELIST = Path(
    "/media/mathis/PANO/bedlam2_render/config/whitelist_animations_starterpack.json"
)
DEFAULT_RENDERER_SCRIPT = Path(
    "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_selected_infinigen_bedlam_erp.py"
)
DEFAULT_BRIDGE_RUNS_ROOT = Path(
    "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_infinigen_bridge/runs"
)
CAMERA_HEIGHT_M = 1.20
DEFAULT_FLOOR_OFFSET_CM = 14.0
DEFAULT_SMOKE_COMMAND_MODE = "print_only"


def _shell_join(parts):
    return " ".join(shlex.quote(str(p)) for p in parts)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_motion_id(value):
    motion_id = str(value or "")
    suffix = "_root_trajectory"
    if motion_id.endswith(suffix):
        return motion_id[: -len(suffix)]
    return motion_id


def _load_allowed_motion_ids(whitelist_path: Path):
    payload = _read_json(whitelist_path)
    allowed = set()
    by_identity = {}
    for identity, motions in sorted(payload.items()):
        ids = [f"{identity}_{motion}" for motion in motions]
        allowed.update(ids)
        by_identity[identity] = ids
    return allowed, by_identity


def _scene_paths(scene_root: Path):
    planning_root = scene_root / "miniscene_selection_v0"
    return {
        "scene_root": scene_root,
        "scene_collision_metadata": scene_root / "scene_collision_metadata.json",
        "human_spawn_poses": scene_root / "human_spawn_poses.json",
        "blender_light_manifest": scene_root / "blender_light_manifest.json",
        "usd_stage": scene_root / "usd_export" / "export_scene.blend" / "export_scene.usdc",
        "usd_light_manifest": scene_root / "usd_export" / "usd_light_manifest.json",
        "starterpack_manifest": planning_root / "bedlam360_infinigen_miniscenes_starterpack_only.json",
        "starterpack_filter_report": planning_root / "starterpack_manifest_filter_report.json",
        "renderable_motion_ids": planning_root / "renderable_motion_ids.json",
    }


def _required_file_checks(paths):
    results = {}
    missing = []
    for key, path in paths.items():
        exists = Path(path).exists()
        results[key] = {"path": str(path), "exists": bool(exists)}
        if not exists and key in {
            "scene_collision_metadata",
            "human_spawn_poses",
            "blender_light_manifest",
            "usd_light_manifest",
            "starterpack_manifest",
        }:
            missing.append(str(path))
    return results, missing


def _manifest_summary(manifest, filter_report, allowed_motion_ids):
    miniscenes = list(manifest.get("miniscenes", []))
    room_counts = Counter()
    scene_type_counts = Counter()
    duplicate_motion_scene_ids = []
    disallowed_motion_ids = Counter()
    scene_rows = []
    two_human_count = 0
    single_human_count = 0

    for index, miniscene in enumerate(miniscenes):
        room = str(miniscene.get("room") or "")
        scene_type = str(miniscene.get("scene_type") or "")
        humans = list(miniscene.get("humans", []))
        motion_ids = [_normalize_motion_id(h.get("motion_id")) for h in humans]
        room_counts[room] += 1
        scene_type_counts[scene_type] += 1
        if scene_type == "single_human":
            single_human_count += 1
        if scene_type == "two_human":
            two_human_count += 1
            if len(motion_ids) != len(set(motion_ids)):
                duplicate_motion_scene_ids.append(str(miniscene.get("miniscene_id")))
        for motion_id in motion_ids:
            if motion_id not in allowed_motion_ids:
                disallowed_motion_ids[motion_id] += 1
        scene_rows.append(
            {
                "index": int(index),
                "miniscene_id": str(miniscene.get("miniscene_id")),
                "room": room,
                "scene_type": scene_type,
                "human_count": len(humans),
                "motion_ids": motion_ids,
            }
        )

    possible_two_human = None
    if filter_report:
        possible_two_human = int(
            filter_report.get("summary", {}).get("valid_two_human_groups_after", 0)
        )
    two_human_expected = possible_two_human is not None and possible_two_human > 0

    errors = []
    warnings = []
    if single_human_count < 1:
        errors.append("manifest_has_no_single_human_scene")
    if two_human_expected and two_human_count < 1:
        errors.append("manifest_has_no_two_human_scene_but_two_human_groups_exist")
    elif two_human_count < 1:
        warnings.append("manifest_has_no_two_human_scene")
    if duplicate_motion_scene_ids:
        errors.append("duplicate_motion_ids_present_in_two_human_scenes")
    if disallowed_motion_ids:
        errors.append("manifest_contains_non_starterpack_motion_ids")

    return {
        "scene_count": len(miniscenes),
        "single_human_count": single_human_count,
        "two_human_count": two_human_count,
        "room_coverage": dict(sorted(room_counts.items())),
        "scene_type_counts": dict(sorted(scene_type_counts.items())),
        "duplicate_motion_scene_ids": duplicate_motion_scene_ids,
        "disallowed_motion_ids": dict(sorted(disallowed_motion_ids.items())),
        "possible_two_human_groups_after_filter": possible_two_human,
        "two_human_expected": two_human_expected,
        "errors": errors,
        "warnings": warnings,
        "scene_rows": scene_rows,
    }


def _select_smoke_test_scenes(manifest):
    miniscenes = list(manifest.get("miniscenes", []))
    first_single = None
    first_two = None
    for index, miniscene in enumerate(miniscenes):
        scene_type = str(miniscene.get("scene_type") or "")
        if first_single is None and scene_type == "single_human":
            first_single = (index, miniscene)
        if first_two is None and scene_type == "two_human":
            first_two = (index, miniscene)
    selected = []
    if first_single is not None:
        selected.append(
            {
                "label": "single_human_smoke_test",
                "index": int(first_single[0]),
                "miniscene_id": str(first_single[1].get("miniscene_id")),
                "room": str(first_single[1].get("room")),
                "scene_type": "single_human",
            }
        )
    if first_two is not None:
        selected.append(
            {
                "label": "two_human_smoke_test",
                "index": int(first_two[0]),
                "miniscene_id": str(first_two[1].get("miniscene_id")),
                "room": str(first_two[1].get("room")),
                "scene_type": "two_human",
            }
        )
    return selected


def _smoke_test_command(
    renderer_prefix,
    renderer_script: Path,
    manifest_path: Path,
    smoke_scene,
):
    return list(renderer_prefix) + [
        str(renderer_script),
        "--manifest",
        str(manifest_path),
        "--miniscene-id",
        str(smoke_scene["miniscene_id"]),
    ]


def _smoke_test_instruction_record(
    scene_root: Path,
    manifest_path: Path,
    smoke_scene,
    renderer_prefix,
    renderer_script: Path,
):
    if not renderer_prefix:
        renderer_prefix = ["py"]
    cmd = _smoke_test_command(
        renderer_prefix=renderer_prefix,
        renderer_script=renderer_script,
        manifest_path=manifest_path,
        smoke_scene=smoke_scene,
    )
    return {
        "scene_root": str(scene_root),
        "manifest_path": str(manifest_path),
        "label": smoke_scene["label"],
        "miniscene_id": smoke_scene["miniscene_id"],
        "room": smoke_scene["room"],
        "scene_type": smoke_scene["scene_type"],
        "command": _shell_join(cmd),
        "renderer_script": str(renderer_script),
    }


def _bridge_report_paths(runs_root: Path):
    return sorted(runs_root.glob("*/bridge_report.json"))


def _find_new_bridge_report(before_paths, after_paths):
    before = {str(p) for p in before_paths}
    new_paths = [p for p in after_paths if str(p) not in before]
    if new_paths:
        return max(new_paths, key=lambda p: p.stat().st_mtime)
    if after_paths:
        return max(after_paths, key=lambda p: p.stat().st_mtime)
    return None


def _nearest_human_xy_distance(spatial_sanity):
    distances = []
    camera = (spatial_sanity or {}).get("camera_position_infinigen_m") or {}
    humans = (spatial_sanity or {}).get("humans") or []
    cx = camera.get("x")
    cy = camera.get("y")
    if cx is None or cy is None:
        return None
    for human in humans:
        position = human.get("position_infinigen_xy_m") or {}
        hx = position.get("x")
        hy = position.get("y")
        if hx is None or hy is None:
            continue
        distances.append(float(math.hypot(float(cx) - float(hx), float(cy) - float(hy))))
    if not distances:
        return None
    return min(distances)


def _camera_sanity_from_bridge_report(bridge_report, floor_offset_cm):
    camera_selection = bridge_report.get("camera_selection") or {}
    spatial = bridge_report.get("room_camera_spatial_sanity") or {}
    strategy = camera_selection.get("selected_camera_xy_strategy") or {}
    room_floor_z = float(camera_selection.get("room_floor_z_m", 0.0))
    camera_position_inf = camera_selection.get("camera_position_infinigen_m") or {}
    camera_position_unreal = camera_selection.get("camera_position_unreal_cm") or {}
    camera_z_m = camera_position_inf.get("z")
    camera_z_cm = camera_position_unreal.get("z")
    expected_z_m = room_floor_z + CAMERA_HEIGHT_M
    expected_z_cm = (expected_z_m * 100.0) + float(floor_offset_cm)
    min_human_distance = _nearest_human_xy_distance(spatial)
    human_clearance = camera_selection.get("camera_human_clearance_m")
    obstacle_clearance = camera_selection.get("camera_obstacle_clearance_m")
    nearest_obstacle = strategy.get("nearest_obstacle") or {}
    nearest_obstacle_distance = nearest_obstacle.get("distance_m")

    checks = {
        "camera_inside_room": bool(spatial.get("camera_inside_selected_room_polygon")),
        "camera_obstacle_free": bool(strategy.get("obstacle_free", False)),
        "camera_not_too_close_to_humans": True,
        "camera_height_matches_infinigen_m": camera_z_m is not None and abs(float(camera_z_m) - expected_z_m) < 1e-4,
        "camera_height_matches_unreal_cm": camera_z_cm is not None and abs(float(camera_z_cm) - expected_z_cm) < 1e-3,
    }
    if human_clearance is not None and min_human_distance is not None:
        checks["camera_not_too_close_to_humans"] = bool(
            float(min_human_distance) >= float(human_clearance)
        )
    errors = []
    if not checks["camera_inside_room"]:
        errors.append("camera_outside_selected_room_polygon")
    if not checks["camera_obstacle_free"]:
        errors.append("camera_inside_or_too_close_to_obstacle")
    if not checks["camera_not_too_close_to_humans"]:
        errors.append("camera_too_close_to_human")
    if not checks["camera_height_matches_infinigen_m"]:
        errors.append("camera_height_mismatch_infinigen")
    if not checks["camera_height_matches_unreal_cm"]:
        errors.append("camera_height_mismatch_unreal")
    return {
        "checks": checks,
        "room_floor_z_m": room_floor_z,
        "expected_camera_z_infinigen_m": expected_z_m,
        "actual_camera_z_infinigen_m": camera_z_m,
        "expected_camera_z_unreal_cm": expected_z_cm,
        "actual_camera_z_unreal_cm": camera_z_cm,
        "min_human_xy_distance_m": min_human_distance,
        "required_human_clearance_m": human_clearance,
        "required_obstacle_clearance_m": obstacle_clearance,
        "nearest_obstacle_distance_m": nearest_obstacle_distance,
        "errors": errors,
    }


def _render_outputs_ok(bridge_report):
    run_root = Path(bridge_report.get("run_root") or "")
    rgb_paths = sorted(run_root.glob("**/*_rgb.png")) if run_root else []
    exr_paths = sorted(run_root.glob("**/*_erp.exr")) if run_root else []
    bridge_report_path = run_root / "bridge_report.json" if run_root else None
    errors = []
    if not rgb_paths:
        errors.append("missing_rgb_png_output")
    if not exr_paths:
        errors.append("missing_erp_exr_output")
    if not bridge_report_path or not bridge_report_path.exists():
        errors.append("missing_bridge_report_json")
    return {
        "rgb_count": len(rgb_paths),
        "exr_count": len(exr_paths),
        "bridge_report_exists": bool(bridge_report_path and bridge_report_path.exists()),
        "sample_rgb": str(rgb_paths[0]) if rgb_paths else None,
        "sample_exr": str(exr_paths[0]) if exr_paths else None,
        "errors": errors,
    }


def _run_smoke_test(
    renderer_prefix,
    renderer_script: Path,
    manifest_path: Path,
    smoke_scene,
    bridge_runs_root: Path,
):
    before = _bridge_report_paths(bridge_runs_root)
    cmd = _smoke_test_command(
        renderer_prefix=renderer_prefix,
        renderer_script=renderer_script,
        manifest_path=manifest_path,
        smoke_scene=smoke_scene,
    )
    subprocess.run(cmd, check=True)
    after = _bridge_report_paths(bridge_runs_root)
    report_path = _find_new_bridge_report(before, after)
    if report_path is None:
        raise RuntimeError(
            f"Smoke test rendered but no bridge_report.json was found for {smoke_scene['miniscene_id']}"
        )
    bridge_report = _read_json(report_path)
    return {
        "command": _shell_join(cmd),
        "bridge_report_path": str(report_path),
        "bridge_report": bridge_report,
    }


def _write_smoke_test_instruction_files(scenes_root: Path, instruction_rows):
    sh_path = scenes_root / "v3_smoke_test_commands.sh"
    txt_path = scenes_root / "v3_smoke_test_unreal_instructions.txt"
    sh_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# BEDLAM360 v3 smoke-test commands",
        "# Run these from the appropriate Unreal Python execution context.",
        "",
    ]
    txt_lines = [
        "BEDLAM360 v3 smoke-test instructions",
        "",
        "Offline validation passed for these scenes, but Unreal render smoke tests are pending.",
        "Load the matching USD scene in Unreal first, then paste the command below into Unreal's Python command entry.",
        "These commands are already formatted for Unreal as `py \"...script.py\" ...`.",
        "",
    ]
    for row in instruction_rows:
        header = (
            f"# {Path(row['scene_root']).name} | {row['label']} | "
            f"{row['scene_type']} | {row['room']} | {row['miniscene_id']}"
        )
        sh_lines.append(header)
        sh_lines.append(row["command"])
        sh_lines.append("")
        txt_lines.extend(
            [
                f"scene_root: {row['scene_root']}",
                f"manifest_path: {row['manifest_path']}",
                f"label: {row['label']}",
                f"scene_type: {row['scene_type']}",
                f"room: {row['room']}",
                f"miniscene_id: {row['miniscene_id']}",
                f"command: {row['command']}",
                "",
            ]
        )
    sh_path.write_text("\n".join(sh_lines) + "\n", encoding="utf-8")
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    return {
        "shell_script": str(sh_path),
        "instructions_txt": str(txt_path),
    }


def _batch_dataset_instruction_rows(scene_results, renderer_script: Path):
    rows = []
    renderer_script = Path(renderer_script).expanduser().resolve()
    for row in scene_results:
        scene_root = Path(row["scene_root"]).expanduser().resolve()
        manifest_path = scene_root / "miniscene_selection_v0" / "bedlam360_infinigen_miniscenes_starterpack_only.json"
        usd_stage_path = scene_root / "usd_export" / "export_scene.blend" / "export_scene.usdc"
        batch_command = (
            f'py "{renderer_script}" '
            f'--scene-root "{scene_root}" '
            f'--manifest "{manifest_path}" '
            f'--batch --batch-balanced-rooms --max-clips 6 '
            f'--frame-start 0 --frame-end 120'
        )
        rows.append(
            {
                "seed": scene_root.name,
                "scene_root": str(scene_root),
                "expected_usd_stage_path": str(usd_stage_path),
                "manifest_path": str(manifest_path),
                "output_report_location": str(
                    Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_infinigen_bridge/batch_miniscene_runs")
                ),
                "command": batch_command,
            }
        )
    return rows


def _write_dataset_batch_instruction_files(scenes_root: Path, dataset_rows):
    sh_path = scenes_root / "v3_dataset_batch_commands.sh"
    txt_path = scenes_root / "v3_dataset_batch_unreal_instructions.txt"
    sh_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# BEDLAM360 v3 small dataset batch commands",
        "# For each seed: load the matching USD stage in Unreal first, then paste the command.",
        "",
    ]
    txt_lines = [
        "BEDLAM360 v3 small dataset batch instructions",
        "",
        "For each seed below:",
        "1. Load the expected USD stage in Unreal.",
        "2. Paste the batch command into Unreal's Python command entry.",
        "3. The renderer will hard-error if the loaded USD scene does not match the scene root.",
        "",
    ]
    for row in dataset_rows:
        sh_lines.append(f"# {row['seed']}")
        sh_lines.append(row["command"])
        sh_lines.append("")
        txt_lines.extend(
            [
                f"seed: {row['seed']}",
                f"scene_root: {row['scene_root']}",
                f"expected_usd_stage_path: {row['expected_usd_stage_path']}",
                f"manifest_path: {row['manifest_path']}",
                f"output_report_location: {row['output_report_location']}",
                f"command: {row['command']}",
                "",
            ]
        )
    sh_path.write_text("\n".join(sh_lines) + "\n", encoding="utf-8")
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    return {
        "shell_script": str(sh_path),
        "instructions_txt": str(txt_path),
    }


def _validate_scene(
    scene_root: Path,
    allowed_motion_ids,
    floor_offset_cm,
    run_smoke_test,
    smoke_command_mode,
    renderer_prefix,
    renderer_script,
    bridge_runs_root,
):
    paths = _scene_paths(scene_root)
    file_checks, missing_files = _required_file_checks(paths)
    result = {
        "scene_root": str(scene_root),
        "seed": scene_root.name,
        "required_files": file_checks,
        "errors": [],
        "warnings": [],
        "smoke_tests": [],
        "offline_validation_passed": False,
        "offline_ready_for_v4": False,
        "render_smoke_ready_for_v4": False,
        "ready_for_v4": False,
        "smoke_test_status": "not_planned",
        "smoke_test_blocking_reason": None,
        "smoke_test_instructions": [],
    }
    if missing_files:
        result["errors"].extend(f"missing_required_file:{p}" for p in missing_files)
        return result

    manifest = _read_json(paths["starterpack_manifest"])
    filter_report = None
    if paths["starterpack_filter_report"].exists():
        filter_report = _read_json(paths["starterpack_filter_report"])
    manifest_summary = _manifest_summary(manifest, filter_report, allowed_motion_ids)
    result["manifest_summary"] = manifest_summary
    result["errors"].extend(manifest_summary["errors"])
    result["warnings"].extend(manifest_summary["warnings"])
    result["offline_validation_passed"] = len(result["errors"]) == 0
    result["offline_ready_for_v4"] = result["offline_validation_passed"]

    smoke_scenes = _select_smoke_test_scenes(manifest)
    result["smoke_test_plan"] = smoke_scenes
    if not run_smoke_test:
        result["smoke_test_status"] = "not_requested"
        result["smoke_test_blocking_reason"] = "smoke_test_not_requested"
        result["warnings"].append("smoke_test_not_run")
    elif smoke_command_mode in {"print_only", "manual_unreal"}:
        result["smoke_test_status"] = "pending_manual_unreal"
        result["smoke_test_blocking_reason"] = "manual_unreal_execution_required"
        result["smoke_test_instructions"] = [
            _smoke_test_instruction_record(
                scene_root=scene_root,
                manifest_path=paths["starterpack_manifest"],
                smoke_scene=smoke_scene,
                renderer_prefix=renderer_prefix,
                renderer_script=renderer_script,
            )
            for smoke_scene in smoke_scenes
        ]
        result["warnings"].append("smoke_tests_pending_manual_unreal")
    elif smoke_command_mode == "unreal_cli":
        result["smoke_test_status"] = "running_unreal_cli"
        for smoke_scene in smoke_scenes:
            try:
                smoke = _run_smoke_test(
                    renderer_prefix,
                    renderer_script,
                    paths["starterpack_manifest"],
                    smoke_scene,
                    bridge_runs_root,
                )
                bridge_report = smoke["bridge_report"]
                camera_sanity = _camera_sanity_from_bridge_report(
                    bridge_report,
                    floor_offset_cm=floor_offset_cm,
                )
                output_sanity = _render_outputs_ok(bridge_report)
                smoke_result = {
                    "label": smoke_scene["label"],
                    "miniscene_id": smoke_scene["miniscene_id"],
                    "room": smoke_scene["room"],
                    "scene_type": smoke_scene["scene_type"],
                    "command": smoke["command"],
                    "bridge_report_path": smoke["bridge_report_path"],
                    "camera_sanity": camera_sanity,
                    "output_sanity": output_sanity,
                    "bridge_report_summary": {
                        "miniscene_id": bridge_report.get("miniscene_id"),
                        "room": bridge_report.get("room"),
                        "selected_miniscene_room": bridge_report.get("selected_miniscene_room"),
                        "run_root": bridge_report.get("run_root"),
                        "erp_output_path": bridge_report.get("erp_output_path"),
                    },
                }
                result["smoke_tests"].append(smoke_result)
                result["errors"].extend(camera_sanity["errors"])
                result["errors"].extend(output_sanity["errors"])
            except Exception as exc:
                result["smoke_tests"].append(
                    {
                        "label": smoke_scene["label"],
                        "miniscene_id": smoke_scene["miniscene_id"],
                        "room": smoke_scene["room"],
                        "scene_type": smoke_scene["scene_type"],
                        "error": str(exc),
                    }
                )
                result["errors"].append(
                    f"smoke_test_failed:{smoke_scene['label']}:{smoke_scene['miniscene_id']}"
                )
    else:
        raise RuntimeError(f"Unsupported smoke_command_mode: {smoke_command_mode}")

    if smoke_command_mode == "unreal_cli" and run_smoke_test:
        result["render_smoke_ready_for_v4"] = not any(
            str(error).startswith("smoke_test_failed:")
            or str(error).startswith("camera_")
            or str(error).startswith("missing_rgb_")
            or str(error).startswith("missing_erp_")
            or str(error).startswith("missing_bridge_report")
            for error in result["errors"]
        )
        result["smoke_test_status"] = (
            "passed" if result["render_smoke_ready_for_v4"] else "failed"
        )
        if not result["render_smoke_ready_for_v4"]:
            result["smoke_test_blocking_reason"] = "one_or_more_smoke_tests_failed"

    result["ready_for_v4"] = bool(
        result["offline_ready_for_v4"] and result["render_smoke_ready_for_v4"]
    )
    return result


def _print_summary(scene_results):
    print("scene | singles | two_human | offline_ready | smoke_status | render_smoke_ready | ready_for_v4 | errors")
    for row in scene_results:
        manifest_summary = row.get("manifest_summary") or {}
        print(
            f"{Path(row['scene_root']).name} | "
            f"{manifest_summary.get('single_human_count', 0)} | "
            f"{manifest_summary.get('two_human_count', 0)} | "
            f"{row.get('offline_ready_for_v4')} | "
            f"{row.get('smoke_test_status')} | "
            f"{row.get('render_smoke_ready_for_v4')} | "
            f"{row.get('ready_for_v4')} | "
            f"{len(row.get('errors', []))}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", type=Path, default=DEFAULT_SCENES_ROOT)
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument(
        "--starterpack-whitelist",
        type=Path,
        default=DEFAULT_STARTERPACK_WHITELIST,
    )
    parser.add_argument("--run-smoke-test", action="store_true", default=False)
    parser.add_argument(
        "--smoke-command-mode",
        choices=("print_only", "unreal_cli", "manual_unreal"),
        default=DEFAULT_SMOKE_COMMAND_MODE,
    )
    parser.add_argument(
        "--renderer-prefix",
        nargs="+",
        default=[],
        help="Command prefix for smoke tests. Leave empty for print/manual modes.",
    )
    parser.add_argument(
        "--renderer-script",
        type=Path,
        default=DEFAULT_RENDERER_SCRIPT,
    )
    parser.add_argument(
        "--bridge-runs-root",
        type=Path,
        default=DEFAULT_BRIDGE_RUNS_ROOT,
    )
    parser.add_argument(
        "--floor-offset-cm",
        type=float,
        default=DEFAULT_FLOOR_OFFSET_CM,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    allowed_motion_ids, by_identity = _load_allowed_motion_ids(args.starterpack_whitelist)
    scene_roots = sorted(
        p for p in args.scenes_root.glob("seed_*") if p.is_dir()
    )
    if args.scene_limit is not None:
        scene_roots = scene_roots[: args.scene_limit]

    results = []
    smoke_instruction_rows = []
    for scene_root in scene_roots:
        scene_result = _validate_scene(
                scene_root=scene_root,
                allowed_motion_ids=allowed_motion_ids,
                floor_offset_cm=args.floor_offset_cm,
                run_smoke_test=args.run_smoke_test,
                smoke_command_mode=args.smoke_command_mode,
                renderer_prefix=args.renderer_prefix,
                renderer_script=args.renderer_script,
                bridge_runs_root=args.bridge_runs_root,
            )
        results.append(scene_result)
        smoke_instruction_rows.extend(scene_result.get("smoke_test_instructions", []))

    all_errors = sum(len(r.get("errors", [])) for r in results)
    offline_ready_count = sum(1 for r in results if r.get("offline_ready_for_v4"))
    render_smoke_ready_count = sum(1 for r in results if r.get("render_smoke_ready_for_v4"))
    ready_count = sum(1 for r in results if r.get("ready_for_v4"))
    room_coverage = defaultdict(int)
    for row in results:
        summary = row.get("manifest_summary") or {}
        for room_name in (summary.get("room_coverage") or {}):
            room_coverage[room_name] += 1
    instruction_files = _write_smoke_test_instruction_files(
        args.scenes_root, smoke_instruction_rows
    )
    report = {
        "phase": "bedlam360_v3_acceptance_validation",
        "scenes_root": str(args.scenes_root),
        "scene_count": len(results),
        "run_smoke_test": bool(args.run_smoke_test),
        "smoke_command_mode": args.smoke_command_mode,
        "starterpack_whitelist": str(args.starterpack_whitelist),
        "starterpack_motion_id_count": len(allowed_motion_ids),
        "starterpack_motion_identities": dict(sorted((k, len(v)) for k, v in by_identity.items())),
        "offline_ready_scene_count": offline_ready_count,
        "render_smoke_ready_scene_count": render_smoke_ready_count,
        "ready_scene_count": ready_count,
        "all_errors_count": all_errors,
        "offline_validation_passed": offline_ready_count == len(results) and len(results) > 0,
        "offline_ready_for_v4": offline_ready_count == len(results) and len(results) > 0,
        "render_smoke_ready_for_v4": render_smoke_ready_count == len(results) and len(results) > 0,
        "ready_for_v4": ready_count == len(results) and all_errors == 0 and len(results) > 0,
        "smoke_test_status": (
            "pending_manual_unreal"
            if args.run_smoke_test and args.smoke_command_mode in {"print_only", "manual_unreal"}
            else ("executed" if args.run_smoke_test and args.smoke_command_mode == "unreal_cli" else "not_requested")
        ),
        "smoke_test_blocking_reason": (
            "manual_unreal_execution_required"
            if args.run_smoke_test and args.smoke_command_mode in {"print_only", "manual_unreal"}
            else None
        ),
        "smoke_test_instruction_files": instruction_files,
        "room_coverage_across_scenes": dict(sorted(room_coverage.items())),
        "scenes": results,
    }
    dataset_instruction_files = _write_dataset_batch_instruction_files(
        args.scenes_root,
        _batch_dataset_instruction_rows(results, args.renderer_script),
    )
    report["dataset_batch_instruction_files"] = dataset_instruction_files

    output_json = args.output_json or (args.scenes_root / "bedlam360_v3_acceptance_report.json")
    _write_json(output_json, report)
    _print_summary(results)
    print(f"Acceptance report: {output_json}")
    print(f"Smoke-test shell commands: {instruction_files['shell_script']}")
    print(f"Smoke-test instructions: {instruction_files['instructions_txt']}")
    print(f"Dataset batch shell commands: {dataset_instruction_files['shell_script']}")
    print(f"Dataset batch instructions: {dataset_instruction_files['instructions_txt']}")
    print(f"offline_ready_for_v4={report['offline_ready_for_v4']}")
    print(f"render_smoke_ready_for_v4={report['render_smoke_ready_for_v4']}")
    print(f"ready_for_v4={report['ready_for_v4']}")


if __name__ == "__main__":
    main()
