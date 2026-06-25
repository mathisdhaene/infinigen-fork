import argparse
import csv
import gc
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import unreal


MINISCENE_MANIFEST_PATH = Path(
    "/media/mathis/PANO/infinigen/outputs/indoors/human_spawn_poc/miniscene_selection_v0/bedlam360_infinigen_miniscenes.json"
)
SCENE_ROOT = None
MINISCENE_INDEX = 0
MINISCENE_ID = None
MINISCENE_ROOM_HINT = None
ALLOW_FALLBACK_TO_NEXT_RENDERABLE = True
ALLOW_FALLBACK_FOR_EXPLICIT_SELECTION = False
PREFER_SAME_ROOM_ON_FALLBACK = True
SEQUENCE_NAME = "infinigen_selected_miniscene_erp_bridge"
CAPTURE_PIPELINE_DIAGNOSTICS = False
CAPTURE_DIAGNOSTIC_FRAME_INDEX = None
CAPTURE_DIAGNOSTIC_REFERENCE_VIEW_WIDTH = 1280
CAPTURE_DIAGNOSTIC_REFERENCE_VIEW_HEIGHT = 720
BATCH_RENDER_MINISCENES = False
BATCH_MAX_MINISCENES = 10
BATCH_ROOM_FILTER = None
BATCH_BALANCED_ROOMS = False
BATCH_MAX_PER_ROOM_FIRST_PASS = 2
MIN_HUMAN_COUNT_PER_RENDER = 2
RENDER_OUTPUT_PROFILE = "full_debug"
ENABLE_ADAPTIVE_PREVIEW_FRAMES = False
ENABLE_FIXED_PREVIEW_FRAMES = False
ENABLE_PER_FRAME_STATS = False
ENABLE_BODY_EVAL_CSV = False
ENABLE_MP4_PREVIEW = False
RGB_TONEMAP_MODE = "sequence_adaptive_rgb"
MAX_PLANNER_RUNTIME_ROOT_ERROR_CM = 50.0
ENABLE_CUBEMAP_FACE_DIAGNOSTICS = False
CUBEMAP_FACE_DIAGNOSTIC_EXPORT_MODES = ("immediate", "flush", "wait_one_tick")
CUBEMAP_FACE_DIAGNOSTIC_WAIT_SECONDS = 1.0 / 30.0
ENABLE_CUBEMAP_FACE_ANOMALY_RETRY = False
CUBEMAP_FACE_ANOMALY_RETRY_MODE = "wait_one_tick"
CUBEMAP_FACE_ANOMALY_RETRY_MAX_ATTEMPTS = 1
EXPORT_FACE_DIAGNOSTICS_ONLY_ON_RETRY = True
REJECT_CLIPS_WITH_ARTIFACTS = False
LIST_RENDERABLE_MINISCENES = False
LIST_RENDERABLE_LIMIT = 50
DIAGNOSE_RENDERABILITY = False
LIST_AVAILABLE_MOTIONS = False
LIGHTING_INTENSITY_SWEEP = False
# LDR FinalColor RGB is much more sensitive than the earlier HDR-debug preview path.
# Sweep around the practical range we are actually using now.
SWEEP_USD_INTENSITY_VALUES = [0, 1, 5, 20, 100]
SWEEP_FILL_INTENSITY_VALUES = [0, 1, 3, 10, 30]
SWEEP_COMBINED_USD_SCALE = 5
SWEEP_FRAME_INDEX = None
USE_MINISCENE_ANCHOR_CAMERA = True
CAMERA_HEIGHT_M = 1.20
CAMERA_OFFSET_X_M = 0.0
CAMERA_OFFSET_Y_M = -1.5
CAMERA_ROOM_MARGIN_M = 0.6
CAMERA_OBSTACLE_CLEARANCE_M = 0.45
CAMERA_HUMAN_CLEARANCE_M = 1.0
CAMERA_MIN_WALL_CLEARANCE_M = 0.6
CAMERA_ENABLE_INTERIOR_LOS_CHECK = True
CAMERA_ENABLE_UNREAL_COLLISION_PROBE = True
CAMERA_COLLISION_PROBE_RAY_LENGTH_CM = 80.0
CAMERA_MIN_COLLISION_FREE_RAY_DISTANCE_CM = 30.0
CAMERA_MIN_GEOMETRY_CLEARANCE_CM = 20.0
DEBUG_MARKER_Z_OFFSET_CM = 25.0
AGGRESSIVE_CLIP_CLEANUP = True
BEDLAM_DEBUG_APPEARANCE_MODE = "full"
EMIT_MEMREPORT = False
EMIT_RHI_MEMORY_DUMP = False
MEMORY_DUMP_CLIP_INDICES = (1, 5)

import render_validated_infinigen_bedlam_erp as base  # noqa: E402

base = importlib.reload(base)

_CLIP_ASSET_AUDIT_STATE = {
    "clip_index": 0,
    "seen_asset_paths": set(),
    "seen_paths_by_category": {},
    "previous_clip_after_cleanup_by_category": {},
}


def log_info(message):
    unreal.log(f"[BEDLAM_INFINIGEN_MINISCENE_ERP] {message}")


def _safe_float(value):
    try:
        if value in (None, "", "N/A", "[Not Supported]"):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value):
    try:
        if value in (None, "", "N/A", "[Not Supported]"):
            return None
        return int(float(value))
    except Exception:
        return None


def _read_proc_status_bytes(pid):
    rss_bytes = None
    vms_bytes = None
    try:
        status_path = Path(f"/proc/{int(pid)}/status")
        if status_path.is_file():
            for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_bytes = int(parts[1]) * 1024
                elif line.startswith("VmSize:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        vms_bytes = int(parts[1]) * 1024
    except Exception:
        pass
    return rss_bytes, vms_bytes


def _nvidia_smi_snapshot(unreal_pid=None):
    snapshot = {
        "gpu_query_ok": False,
        "process_query_ok": False,
        "gpus": [],
        "unreal_process_gpu_memory_mb": None,
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            snapshot["gpu_query_ok"] = True
            for raw_line in completed.stdout.splitlines():
                parts = [part.strip() for part in raw_line.split(",")]
                if len(parts) < 7:
                    continue
                snapshot["gpus"].append(
                    {
                        "gpu_index": _safe_int(parts[0]),
                        "gpu_name": parts[1],
                        "gpu_memory_used_mb": _safe_float(parts[2]),
                        "gpu_memory_total_mb": _safe_float(parts[3]),
                        "gpu_util_percent": _safe_float(parts[4]),
                        "gpu_temperature_c": _safe_float(parts[5]),
                        "gpu_power_watts": _safe_float(parts[6]),
                    }
                )
    except Exception as exc:
        snapshot["gpu_query_error"] = str(exc)

    if unreal_pid is not None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                snapshot["process_query_ok"] = True
                total_mb = 0.0
                for raw_line in completed.stdout.splitlines():
                    parts = [part.strip() for part in raw_line.split(",")]
                    if len(parts) < 4:
                        continue
                    if _safe_int(parts[0]) != int(unreal_pid):
                        continue
                    used_mb = _safe_float(parts[3]) or 0.0
                    total_mb += used_mb
                snapshot["unreal_process_gpu_memory_mb"] = float(total_mb)
        except Exception as exc:
            snapshot["process_query_error"] = str(exc)
    return snapshot


def _issue_memreport_checkpoint(tag, run_root):
    world = None
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()
    except Exception:
        world = None
    commands = ["MemReport", "MemReport -full"]
    issued = []
    for command in commands:
        try:
            unreal.SystemLibrary.execute_console_command(world, command)
            issued.append({"command": command, "issued": True})
        except Exception as exc:
            issued.append({"command": command, "issued": False, "error": str(exc)})
    return {
        "tag": str(tag),
        "issued_commands": issued,
        "expected_saved_dir_hint": str(Path(run_root) / "Saved" / "Profiling" / "MemReports"),
    }


def _issue_rhi_memory_dump(tag, run_root):
    world = None
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()
    except Exception:
        world = None
    command = "rhi.DumpMemory"
    report = {
        "tag": str(tag),
        "command": command,
        "issued": False,
        "expected_saved_dir_hint": str(Path(run_root) / "Saved" / "Logs"),
    }
    try:
        unreal.SystemLibrary.execute_console_command(world, command)
        report["issued"] = True
    except Exception as exc:
        report["error"] = str(exc)
    return report


def _safe_get_path_name(obj):
    try:
        if obj is not None and hasattr(obj, "get_path_name"):
            return str(obj.get_path_name())
    except Exception:
        return None
    return None


def _safe_get_name(obj):
    try:
        if obj is not None and hasattr(obj, "get_name"):
            return str(obj.get_name())
    except Exception:
        return None
    return None


def _safe_actor_label(actor):
    try:
        return str(actor.get_actor_label())
    except Exception:
        return _safe_get_name(actor)


def _safe_actor_components(actor):
    try:
        getter = getattr(actor, "get_components_by_class", None)
        if callable(getter):
            return list(getter(unreal.ActorComponent))
    except Exception:
        pass
    return []


def _unique_paths(paths):
    seen = set()
    ordered = []
    for path in paths or []:
        path = str(path or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _find_runtime_loaded_object_path(asset_path):
    asset_path = str(asset_path or "").strip()
    if not asset_path:
        return None
    find_object = getattr(unreal, "find_object", None)
    if not callable(find_object):
        return None
    candidates = [asset_path]
    if "." not in asset_path and "/" in asset_path:
        leaf = asset_path.rsplit("/", 1)[-1]
        candidates.append(f"{asset_path}.{leaf}")
    for candidate in _unique_paths(candidates):
        try:
            obj = find_object(None, candidate)
        except Exception:
            obj = None
        if obj is not None:
            return _safe_get_path_name(obj) or candidate
    return None


def _asset_reference_entry(asset_type, asset_path):
    resolved_loaded_object_path = _find_runtime_loaded_object_path(asset_path)
    seen_before = str(asset_path) in _CLIP_ASSET_AUDIT_STATE["seen_asset_paths"]
    return {
        "asset_type": str(asset_type),
        "asset_path": str(asset_path),
        "seen_in_previous_clips": bool(seen_before),
        "new_this_clip": not bool(seen_before),
        "runtime_loaded_before_cleanup": bool(resolved_loaded_object_path),
        "runtime_loaded_object_path_before_cleanup": resolved_loaded_object_path,
    }


def _collect_clip_asset_references(result):
    asset_entries = []
    spawned_roles = list((result or {}).get("spawned_roles") or [])
    for role in spawned_roles:
        resolved_asset = dict(role.get("resolved_asset") or {})
        appearance_metadata = dict(role.get("appearance_metadata") or {})
        body_material_meta = dict(appearance_metadata.get("body_material") or {})
        clothing_meta = dict(appearance_metadata.get("clothing") or {})
        hair_meta = dict(appearance_metadata.get("hair") or {})
        shoe_meta = dict(appearance_metadata.get("shoe") or {})
        requested_meta = _dict_or_empty(appearance_metadata.get("requested"))
        clothing_requested_meta = _dict_or_empty(clothing_meta.get("requested"))
        clothing_overlay_meta = _dict_or_empty(appearance_metadata.get("texture_clothing_overlay"))

        for asset_type, asset_path in (
            ("body_geometry_cache", resolved_asset.get("body_geometry_cache_path")),
            ("animation_or_primary_asset", resolved_asset.get("unreal_asset_path")),
            ("body_material", body_material_meta.get("material_path")),
            ("body_texture", body_material_meta.get("resolved_texture_path") or requested_meta.get("texture_body")),
            (
                "clothing_geometry_cache",
                clothing_meta.get("asset_path") or clothing_requested_meta.get("clothing_asset_path"),
            ),
            (
                "clothing_material",
                clothing_meta.get("material_path") or clothing_requested_meta.get("clothing_material_path"),
            ),
            ("clothing_overlay_texture", clothing_overlay_meta.get("material_path")),
            ("hair_groom", hair_meta.get("groom_path")),
            ("hair_binding", hair_meta.get("binding_path")),
            ("hair_material", hair_meta.get("material_path")),
            ("shoe_material", shoe_meta.get("material_path")),
        ):
            if asset_path:
                asset_entries.append(_asset_reference_entry(asset_type, asset_path))
    level_sequence_asset_path = (((result or {}).get("level_sequence") or {}).get("asset_path"))
    if level_sequence_asset_path:
        asset_entries.append(_asset_reference_entry("level_sequence_asset", level_sequence_asset_path))
    unique_entries = []
    seen = set()
    for entry in asset_entries:
        key = (entry["asset_type"], entry["asset_path"])
        if key in seen:
            continue
        seen.add(key)
        unique_entries.append(entry)
    return unique_entries


def _count_asset_types(entries, loaded_key=None):
    counts = {}
    for entry in list(entries or []):
        if loaded_key is not None and not bool(entry.get(loaded_key)):
            continue
        asset_type = str(entry.get("asset_type") or "unknown")
        counts[asset_type] = counts.get(asset_type, 0) + 1
    return dict(sorted(counts.items()))


def _selected_asset_paths(entries, asset_type):
    return sorted(
        str(entry.get("asset_path"))
        for entry in list(entries or [])
        if str(entry.get("asset_type")) == str(asset_type) and entry.get("asset_path")
    )


def _dict_or_empty(value):
    return dict(value) if isinstance(value, dict) else {}


def _body_geometry_cache_path_from_resolved_asset(resolved_asset):
    if isinstance(resolved_asset, dict):
        return resolved_asset.get("body_geometry_cache_path") or resolved_asset.get("unreal_asset_path")
    if resolved_asset is None:
        return None
    try:
        candidate = getattr(resolved_asset, "body_geometry_cache", None)
        return _safe_get_path_name(candidate) or _safe_get_path_name(resolved_asset)
    except Exception:
        return None


def _collect_expected_clip_asset_references(body_specs, appearance_mode):
    body_specs = list(
        base.canonical_validation._apply_bedlam_debug_appearance_mode(body_specs, appearance_mode)
    )
    entries = []
    resolver = base.canonical_validation.mini.reconstruct_one_bedlam_body
    for spec in body_specs:
        spec = dict(spec or {})
        asset_id = spec.get("asset_id")
        texture_body = spec.get("texture_body")
        texture_clothing = spec.get("texture_clothing")
        hair = spec.get("hair")
        haircolor = spec.get("haircolor")
        shoe = spec.get("shoe")
        resolved_asset = resolver.resolve_body_asset(asset_id) if asset_id else None
        hair_paths = resolver._resolve_hair_paths(asset_id, hair, haircolor) if asset_id else {}
        expected_pairs = (
            ("body_geometry_cache", _body_geometry_cache_path_from_resolved_asset(resolved_asset)),
            ("animation_or_primary_asset", None if isinstance(resolved_asset, dict) else _safe_get_path_name(resolved_asset)),
            ("body_material", resolver._resolve_body_material_path(texture_body, shoe=shoe) if texture_body else None),
            ("clothing_geometry_cache", resolver._resolve_clothing_geometry_cache_path(asset_id) if texture_clothing else None),
            ("clothing_material", resolver._resolve_clothing_material_path(texture_clothing) if texture_clothing else None),
            ("hair_groom", (hair_paths or {}).get("groom_path")),
            ("hair_binding", (hair_paths or {}).get("binding_path")),
            ("hair_material", (hair_paths or {}).get("material_path")),
            ("shoe_material", resolver._resolve_body_material_path(texture_body, shoe=shoe) if (texture_body and shoe) else None),
        )
        for asset_type, asset_path in expected_pairs:
            if asset_path:
                entries.append(_asset_reference_entry(asset_type, asset_path))
    unique_entries = []
    seen = set()
    for entry in entries:
        key = (entry["asset_type"], entry["asset_path"])
        if key in seen:
            continue
        seen.add(key)
        unique_entries.append(entry)
    return unique_entries


def _asset_type_to_residency_category(asset_type):
    asset_type = str(asset_type or "")
    mapping = {
        "body_geometry_cache": "body_geometry_cache",
        "clothing_geometry_cache": "clothing_geometry_cache",
        "hair_groom": "hair_groom",
        "hair_binding": "hair_binding",
        "body_material": "body_material",
        "clothing_material": "clothing_material",
        "hair_material": "hair_material",
        "shoe_material": "shoe_material",
        "body_texture": "texture",
        "clothing_overlay_texture": "texture",
        "animation_or_primary_asset": "other_geometry_cache",
    }
    return mapping.get(asset_type, None)


def _empty_residency_groups():
    return {
        "body_geometry_cache": [],
        "clothing_geometry_cache": [],
        "hair_groom": [],
        "hair_binding": [],
        "body_material": [],
        "clothing_material": [],
        "hair_material": [],
        "shoe_material": [],
        "texture": [],
        "other_material": [],
        "other_geometry_cache": [],
    }


def _group_asset_paths_by_category(entries, loaded_key=None):
    groups = {key: set() for key in _empty_residency_groups().keys()}
    for entry in list(entries or []):
        if loaded_key is not None and not bool(entry.get(loaded_key)):
            continue
        category = _asset_type_to_residency_category(entry.get("asset_type"))
        asset_path = str(entry.get("asset_path") or "").strip()
        if not category or not asset_path:
            continue
        groups[category].add(asset_path)
    return {key: sorted(values) for key, values in groups.items()}


def _residency_diff(current_groups, previous_groups):
    diff = {}
    for category in _empty_residency_groups().keys():
        current = set((current_groups or {}).get(category) or [])
        previous = set((previous_groups or {}).get(category) or [])
        diff[category] = sorted(current - previous)
    return diff


def _paths_union(groups):
    combined = set()
    for values in dict(groups or {}).values():
        combined.update(values or [])
    return combined


def _append_asset_residency_csv_rows(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    fieldnames = [
        "clip_index",
        "miniscene_id",
        "appearance_mode",
        "checkpoint",
        "gpu_mb",
        "category",
        "loaded_count",
        "unique_seen_so_far",
        "new_this_clip_count",
        "still_loaded_after_cleanup_count",
    ]
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_unreal_inventory(inventory):
    inventory = dict(inventory or {})
    return {
        "actor_class_counts": dict(inventory.get("actor_class_counts") or {}),
        "matching_component_class_counts": dict(inventory.get("matching_component_class_counts") or {}),
        "geometry_cache_component_count": len(list(inventory.get("geometry_cache_components") or [])),
        "skeletal_mesh_component_count": len(list(inventory.get("skeletal_mesh_components") or [])),
        "groom_component_count": len(list(inventory.get("groom_components") or [])),
        "loaded_geometry_cache_assets": sorted(
            {
                str(item.get("geometry_cache_asset_path"))
                for item in list(inventory.get("geometry_cache_components") or [])
                if item.get("geometry_cache_asset_path")
            }
        ),
        "loaded_skeletal_mesh_assets": sorted(
            {
                str(item.get("skeletal_mesh_asset_path"))
                for item in list(inventory.get("skeletal_mesh_components") or [])
                if item.get("skeletal_mesh_asset_path")
            }
        ),
        "loaded_groom_assets": sorted(
            {
                str(item.get("groom_asset_path"))
                for item in list(inventory.get("groom_components") or [])
                if item.get("groom_asset_path")
            }
        ),
        "render_targets": dict(inventory.get("render_targets") or {}),
        "level_sequence_asset_count": len(list((inventory.get("level_sequences") or {}).get("sequence_assets") or [])),
    }


def _should_emit_heavy_memory_dump(tag, clip_index):
    tag = str(tag)
    if tag not in {"after_clip_before_cleanup", "after_cleanup"}:
        return False
    try:
        return int(clip_index) in set(int(item) for item in MEMORY_DUMP_CLIP_INDICES)
    except Exception:
        return False


def _safe_component_flag(component, prop_name):
    try:
        return component.get_editor_property(prop_name)
    except Exception:
        method_name = f"is_{prop_name}"
        method = getattr(component, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                return None
    return None


def _matches_inventory_interest(*values):
    needles = (
        "bedlam",
        "smpl",
        "human",
        "body",
        "clothing",
        "cloth",
        "hair",
        "geometrycache",
        "skeletalmesh",
        "groom",
        "alembic",
        "cache",
    )
    haystack = " ".join(str(value or "") for value in values).lower()
    return any(needle in haystack for needle in needles)


def _component_asset_path(component, primary_prop_names):
    for prop_name in primary_prop_names:
        try:
            asset = component.get_editor_property(prop_name)
        except Exception:
            asset = None
        asset_path = _safe_get_path_name(asset)
        if asset_path:
            return asset_path
    return None


def _level_sequence_inventory():
    inventory = {
        "registry_scan_root": "/Game/BEDLAM360_Debug",
        "sequence_assets": [],
        "errors": [],
    }
    try:
        registry = unreal.AssetRegistryHelpers.get_asset_registry()
        assets = registry.get_assets_by_path("/Game/BEDLAM360_Debug", recursive=True)
        for asset_data in assets:
            try:
                class_name = str(asset_data.asset_class_path.asset_name)
            except Exception:
                class_name = None
            if class_name not in {"LevelSequence"}:
                continue
            inventory["sequence_assets"].append(
                {
                    "asset_name": str(asset_data.asset_name),
                    "package_name": str(asset_data.package_name),
                    "object_path": f"{asset_data.package_name}.{asset_data.asset_name}",
                    "class_name": class_name,
                }
            )
    except Exception as exc:
        inventory["errors"].append(str(exc))
    return inventory


def _collect_unreal_inventory(tag):
    inventory = {
        "tag": str(tag),
        "actor_class_counts": {},
        "matching_actors": [],
        "matching_component_class_counts": {},
        "geometry_cache_components": [],
        "skeletal_mesh_components": [],
        "groom_components": [],
        "render_targets": {
            "scene_capture_cube_actor_count": 0,
            "scene_capture_cube_actors": [],
            "texture_render_target_cube_count": 0,
            "texture_render_target_2d_count": 0,
            "texture_render_target_cube_paths": [],
            "texture_render_target_2d_paths": [],
        },
        "level_sequences": _level_sequence_inventory(),
        "errors": [],
    }
    try:
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        actors = list(actor_subsystem.get_all_level_actors())
    except Exception as exc:
        inventory["errors"].append(f"get_all_level_actors: {exc}")
        return inventory

    rt_cube_paths = set()
    rt_2d_paths = set()

    for actor in actors:
        class_name = None
        try:
            class_name = str(actor.get_class().get_name())
        except Exception:
            class_name = "UnknownActorClass"
        inventory["actor_class_counts"][class_name] = inventory["actor_class_counts"].get(class_name, 0) + 1
        actor_label = _safe_actor_label(actor)
        actor_name = _safe_get_name(actor)
        actor_path = _safe_get_path_name(actor)

        is_scene_capture_cube = str(class_name) == "SceneCaptureCube"
        if is_scene_capture_cube:
            inventory["render_targets"]["scene_capture_cube_actor_count"] += 1

        components = _safe_actor_components(actor)
        component_entries = []
        include_actor = _matches_inventory_interest(actor_label, actor_name, actor_path, class_name) or is_scene_capture_cube
        for component in components:
            try:
                component_class = str(component.get_class().get_name())
            except Exception:
                component_class = "UnknownComponentClass"
            component_name = _safe_get_name(component)
            component_path = _safe_get_path_name(component)
            if _matches_inventory_interest(component_class, component_name, component_path):
                include_actor = True
            if include_actor:
                inventory["matching_component_class_counts"][component_class] = (
                    inventory["matching_component_class_counts"].get(component_class, 0) + 1
                )
                component_entries.append(
                    {
                        "component_name": component_name,
                        "component_class": component_class,
                        "component_path": component_path,
                    }
                )

            if component_class == "GeometryCacheComponent":
                inventory["geometry_cache_components"].append(
                    {
                        "owner_actor_label": actor_label,
                        "owner_actor_name": actor_name,
                        "component_name": component_name,
                        "component_path": component_path,
                        "geometry_cache_asset_path": _component_asset_path(component, ("geometry_cache",)),
                        "visible": _safe_component_flag(component, "visible"),
                        "active": _safe_component_flag(component, "active"),
                        "registered": _safe_component_flag(component, "registered"),
                    }
                )
            elif component_class == "SkeletalMeshComponent":
                inventory["skeletal_mesh_components"].append(
                    {
                        "owner_actor_label": actor_label,
                        "owner_actor_name": actor_name,
                        "component_name": component_name,
                        "component_path": component_path,
                        "skeletal_mesh_asset_path": _component_asset_path(component, ("skeletal_mesh", "skeletal_mesh_asset")),
                        "visible": _safe_component_flag(component, "visible"),
                        "active": _safe_component_flag(component, "active"),
                        "registered": _safe_component_flag(component, "registered"),
                    }
                )
            elif component_class == "GroomComponent":
                inventory["groom_components"].append(
                    {
                        "owner_actor_label": actor_label,
                        "owner_actor_name": actor_name,
                        "component_name": component_name,
                        "component_path": component_path,
                        "groom_asset_path": _component_asset_path(component, ("groom_asset", "groom")),
                        "binding_asset_path": _component_asset_path(component, ("binding_asset",)),
                        "visible": _safe_component_flag(component, "visible"),
                        "active": _safe_component_flag(component, "active"),
                        "registered": _safe_component_flag(component, "registered"),
                    }
                )

            if component_class in {"SceneCaptureComponentCube", "SceneCaptureComponent2D"}:
                texture_target_path = _component_asset_path(component, ("texture_target",))
                if texture_target_path:
                    if "TextureRenderTargetCube" in str(texture_target_path) or texture_target_path.endswith("_RTCube"):
                        rt_cube_paths.add(texture_target_path)
                    else:
                        try:
                            asset = component.get_editor_property("texture_target")
                            asset_class = str(asset.get_class().get_name()) if asset is not None else ""
                        except Exception:
                            asset_class = ""
                        if asset_class == "TextureRenderTargetCube":
                            rt_cube_paths.add(texture_target_path)
                        elif asset_class == "TextureRenderTarget2D":
                            rt_2d_paths.add(texture_target_path)
                        else:
                            rt_2d_paths.add(texture_target_path)

        if include_actor:
            inventory["matching_actors"].append(
                {
                    "actor_label": actor_label,
                    "actor_name": actor_name,
                    "actor_path": actor_path,
                    "actor_class": class_name,
                    "components": component_entries,
                }
            )
        if is_scene_capture_cube:
            inventory["render_targets"]["scene_capture_cube_actors"].append(
                {
                    "actor_label": actor_label,
                    "actor_name": actor_name,
                    "actor_path": actor_path,
                    "actor_class": class_name,
                }
            )

    inventory["render_targets"]["texture_render_target_cube_paths"] = sorted(rt_cube_paths)
    inventory["render_targets"]["texture_render_target_2d_paths"] = sorted(rt_2d_paths)
    inventory["render_targets"]["texture_render_target_cube_count"] = len(rt_cube_paths)
    inventory["render_targets"]["texture_render_target_2d_count"] = len(rt_2d_paths)
    inventory["actor_class_counts"] = dict(sorted(inventory["actor_class_counts"].items()))
    inventory["matching_component_class_counts"] = dict(sorted(inventory["matching_component_class_counts"].items()))
    return inventory


def _capture_memory_checkpoint(tag, run_root, extra=None):
    run_root = Path(run_root)
    checkpoint_dir = base._ensure_dir(run_root / "memory_checkpoints")
    unreal_pid = int(os.getpid())
    rss_bytes, vms_bytes = _read_proc_status_bytes(unreal_pid)
    gpu_snapshot = _nvidia_smi_snapshot(unreal_pid=unreal_pid)
    clip_index = _safe_int((extra or {}).get("clip_index"))
    emit_heavy_dump = _should_emit_heavy_memory_dump(tag, clip_index)
    checkpoint = {
        "tag": str(tag),
        "timestamp_unix": float(time.time()),
        "unreal_pid": unreal_pid,
        "python_process_rss_bytes": rss_bytes,
        "python_process_vms_bytes": vms_bytes,
        "gpu_snapshot": gpu_snapshot,
        "emit_memreport_enabled": bool(EMIT_MEMREPORT),
        "emit_rhi_memory_dump_enabled": bool(EMIT_RHI_MEMORY_DUMP),
        "heavy_dump_selected_for_checkpoint": bool(emit_heavy_dump),
        "memreport_checkpoint": _issue_memreport_checkpoint(tag, run_root) if (EMIT_MEMREPORT and emit_heavy_dump) else None,
        "rhi_memory_dump": _issue_rhi_memory_dump(tag, run_root) if (EMIT_RHI_MEMORY_DUMP and emit_heavy_dump) else None,
    }
    if extra:
        checkpoint["extra"] = extra
    inventory = _collect_unreal_inventory(tag)
    inventory_path = checkpoint_dir / f"{str(tag)}_unreal_inventory.json"
    base._write_json(inventory_path, inventory)
    checkpoint["unreal_inventory_path"] = str(inventory_path)
    checkpoint["unreal_inventory_summary"] = _summarize_unreal_inventory(inventory)
    checkpoint_path = checkpoint_dir / f"{str(tag)}.json"
    base._write_json(checkpoint_path, checkpoint)
    checkpoint["checkpoint_path"] = str(checkpoint_path)
    log_info("MEMORY_CHECKPOINT " + json.dumps(checkpoint, indent=2))
    return checkpoint


def _clip_cleanup_summary(result, run_root, clip_asset_references=None):
    def _matching_names(target, tokens, callable_only=False, limit=120):
        names = []
        seen = set()
        try:
            candidates = dir(target)
        except Exception:
            candidates = []
        lowered_tokens = tuple(str(token).lower() for token in (tokens or ()))
        for name in candidates:
            name_str = str(name)
            lowered = name_str.lower()
            if lowered_tokens and not any(token in lowered for token in lowered_tokens):
                continue
            if callable_only:
                try:
                    if not callable(getattr(target, name_str, None)):
                        continue
                except Exception:
                    continue
            if name_str in seen:
                continue
            seen.add(name_str)
            names.append(name_str)
            if len(names) >= int(limit):
                break
        return sorted(names)

    def _global_unload_api_diagnostics():
        tokens = ("unload", "release", "garbage", "collect", "flush", "stream", "memory", "cache")
        report = {}
        for label, target in (
            ("EditorAssetLibrary", getattr(unreal, "EditorAssetLibrary", None)),
            ("SystemLibrary", getattr(unreal, "SystemLibrary", None)),
            ("AssetRegistryHelpers", getattr(unreal, "AssetRegistryHelpers", None)),
            ("RenderingLibrary", getattr(unreal, "RenderingLibrary", None)),
        ):
            if target is None:
                report[label] = {"available": False}
                continue
            report[label] = {
                "available": True,
                "matching_methods": _matching_names(target, tokens, callable_only=True),
                "matching_members": _matching_names(target, tokens, callable_only=False),
            }
        try:
            asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
        except Exception:
            asset_registry = None
        report["AssetRegistry"] = {
            "available": bool(asset_registry is not None),
            "matching_methods": _matching_names(asset_registry, tokens, callable_only=True) if asset_registry is not None else [],
            "matching_members": _matching_names(asset_registry, tokens, callable_only=False) if asset_registry is not None else [],
        }
        return report

    def _asset_unload_capability_report(asset_path):
        tokens = ("unload", "release", "stream", "flush", "clear", "close", "destroy", "cache", "groom", "geometry")
        asset_path = str(asset_path or "").strip()
        report = {
            "asset_path": asset_path,
            "resolved_runtime_loaded_object_path": _find_runtime_loaded_object_path(asset_path) if asset_path else None,
            "does_asset_exist": None,
            "editor_asset_library_load_ok": False,
            "unreal_load_asset_ok": False,
            "asset_object_name": None,
            "asset_class_name": None,
            "asset_path_name": None,
            "matching_asset_methods": [],
            "matching_asset_members": [],
            "matching_class_methods": [],
            "matching_class_members": [],
        }
        asset_object = None
        if not asset_path:
            return report
        try:
            report["does_asset_exist"] = bool(unreal.EditorAssetLibrary.does_asset_exist(asset_path))
        except Exception as exc:
            report["does_asset_exist"] = f"error: {exc}"
        for loader_name, loader in (
            ("EditorAssetLibrary.load_asset", getattr(unreal.EditorAssetLibrary, "load_asset", None)),
            ("unreal.load_asset", getattr(unreal, "load_asset", None)),
        ):
            if not callable(loader):
                continue
            try:
                candidate = loader(asset_path)
                if candidate is not None:
                    asset_object = candidate
                    if loader_name == "EditorAssetLibrary.load_asset":
                        report["editor_asset_library_load_ok"] = True
                    if loader_name == "unreal.load_asset":
                        report["unreal_load_asset_ok"] = True
                    break
            except Exception:
                continue
        if asset_object is None:
            return report
        report["asset_object_name"] = _safe_get_name(asset_object)
        report["asset_path_name"] = _safe_get_path_name(asset_object)
        try:
            asset_class = asset_object.get_class()
            report["asset_class_name"] = str(asset_class.get_name()) if asset_class is not None else None
        except Exception:
            asset_class = None
        report["matching_asset_methods"] = _matching_names(asset_object, tokens, callable_only=True)
        report["matching_asset_members"] = _matching_names(asset_object, tokens, callable_only=False)
        if asset_class is not None:
            report["matching_class_methods"] = _matching_names(asset_class, tokens, callable_only=True)
            report["matching_class_members"] = _matching_names(asset_class, tokens, callable_only=False)
        return report

    def _find_actor_by_label(actor_label):
        if not actor_label:
            return None
        try:
            actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
            for actor in actor_subsystem.get_all_level_actors():
                try:
                    if str(actor.get_actor_label()) == str(actor_label):
                        return actor
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def _release_geometry_cache_component(actor):
        report = {
            "actor_label": None,
            "component_found": False,
            "actions": [],
            "errors": [],
        }
        if actor is None:
            report["errors"].append("missing_actor")
            return report
        try:
            report["actor_label"] = str(actor.get_actor_label())
        except Exception:
            report["actor_label"] = str(actor)
        component = None
        try:
            if hasattr(actor, "get_geometry_cache_component"):
                component = actor.get_geometry_cache_component()
        except Exception as exc:
            report["errors"].append(f"get_geometry_cache_component: {exc}")
        if component is None:
            return report
        report["component_found"] = True
        for method_name in ("stop",):
            try:
                method = getattr(component, method_name, None)
                if callable(method):
                    method()
                    report["actions"].append(method_name)
            except Exception as exc:
                report["errors"].append(f"{method_name}: {exc}")
        for prop_name, prop_value in (
            ("manual_tick", False),
            ("running", False),
            ("looping", False),
            ("geometry_cache", None),
        ):
            try:
                component.set_editor_property(prop_name, prop_value)
                report["actions"].append(f"set_{prop_name}")
            except Exception as exc:
                report["errors"].append(f"set_editor_property({prop_name}): {exc}")
        try:
            num_materials = int(component.get_num_materials())
        except Exception:
            num_materials = 0
        for material_index in range(max(0, num_materials)):
            try:
                component.set_material(material_index, None)
                report["actions"].append(f"clear_material_{material_index}")
            except Exception as exc:
                report["errors"].append(f"clear_material_{material_index}: {exc}")
        return report

    def _release_heavy_components(actor):
        report = {
            "actor_label": None,
            "component_reports": [],
            "errors": [],
        }
        if actor is None:
            report["errors"].append("missing_actor")
            return report
        try:
            report["actor_label"] = str(actor.get_actor_label())
        except Exception:
            report["actor_label"] = str(actor)
        try:
            components = list(_safe_actor_components(actor))
        except Exception as exc:
            report["errors"].append(f"get_components: {exc}")
            return report
        for component in components:
            component_report = {
                "component_name": _safe_get_name(component),
                "component_class": None,
                "actions": [],
                "errors": [],
            }
            try:
                component_class = str(component.get_class().get_name())
            except Exception:
                component_class = "UnknownComponentClass"
            component_report["component_class"] = component_class
            try:
                num_materials = int(component.get_num_materials()) if hasattr(component, "get_num_materials") else 0
            except Exception:
                num_materials = 0
            if component_class == "GroomComponent":
                for prop_name, prop_value in (
                    ("groom_asset", None),
                    ("binding_asset", None),
                ):
                    try:
                        component.set_editor_property(prop_name, prop_value)
                        component_report["actions"].append(f"set_{prop_name}")
                    except Exception as exc:
                        component_report["errors"].append(f"set_editor_property({prop_name}): {exc}")
            elif component_class == "SkeletalMeshComponent":
                for prop_name in ("skeletal_mesh", "skeletal_mesh_asset"):
                    try:
                        component.set_editor_property(prop_name, None)
                        component_report["actions"].append(f"set_{prop_name}")
                    except Exception:
                        pass
            for material_index in range(max(0, num_materials)):
                try:
                    component.set_material(material_index, None)
                    component_report["actions"].append(f"clear_material_{material_index}")
                except Exception as exc:
                    component_report["errors"].append(f"clear_material_{material_index}: {exc}")
            for method_name in ("unregister_component", "destroy_component"):
                try:
                    method = getattr(component, method_name, None)
                    if callable(method):
                        method()
                        component_report["actions"].append(method_name)
                except Exception as exc:
                    component_report["errors"].append(f"{method_name}: {exc}")
            if component_report["actions"] or component_report["errors"]:
                report["component_reports"].append(component_report)
        return report

    def _unload_clip_assets(asset_references):
        report = {
            "attempted_paths": [],
            "successful_paths": [],
            "failed_paths": [],
        }
        if not asset_references:
            return report
        unload_asset = getattr(unreal.EditorAssetLibrary, "unload_asset", None)
        if not callable(unload_asset):
            report["failed_paths"].append({"asset_path": None, "error": "EditorAssetLibrary.unload_asset_unavailable"})
            return report
        for entry in list(asset_references or []):
            asset_path = str(entry.get("asset_path") or "").strip()
            if not asset_path:
                continue
            report["attempted_paths"].append(asset_path)
            try:
                unload_asset(asset_path)
                report["successful_paths"].append(asset_path)
            except Exception as exc:
                report["failed_paths"].append({"asset_path": asset_path, "error": str(exc)})
        return report

    cleanup = {
        "attempted": True,
        "run_root": str(run_root),
        "aggressive_cleanup_enabled": bool(AGGRESSIVE_CLIP_CLEANUP),
        "global_unload_api_diagnostics": _global_unload_api_diagnostics(),
        "asset_unload_capability_reports": [],
        "sequence_asset_path": None,
        "level_sequence_closed": False,
        "level_sequence_deleted": False,
        "sequencer_stop_called": False,
        "sequence_bindings_removed": False,
        "sequence_python_references_cleared": False,
        "spawned_actor_labels_targeted": [],
        "spawned_actor_destroy_reports": [],
        "bedlam_actors_cleared": [],
        "clip_asset_unload_report": None,
        "capture_source_init_state_cleared": False,
        "python_gc_collected": None,
        "python_gc_collected_second_pass": None,
        "unreal_collect_garbage_called": False,
        "unreal_collect_garbage_called_second_pass": False,
        "errors": [],
    }
    level_sequence_asset_path = (((result or {}).get("level_sequence") or {}).get("asset_path"))
    cleanup["sequence_asset_path"] = level_sequence_asset_path
    try:
        unreal.LevelSequenceEditorBlueprintLibrary.pause()
        cleanup["sequencer_stop_called"] = True
    except Exception as exc:
        cleanup["errors"].append(f"pause_level_sequence: {exc}")
    try:
        unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
        cleanup["level_sequence_closed"] = True
    except Exception as exc:
        cleanup["errors"].append(f"close_level_sequence: {exc}")
    if bool(AGGRESSIVE_CLIP_CLEANUP):
        try:
            level_sequence_obj = None
            if level_sequence_asset_path and unreal.EditorAssetLibrary.does_asset_exist(level_sequence_asset_path):
                level_sequence_obj = unreal.EditorAssetLibrary.load_asset(level_sequence_asset_path)
            if level_sequence_obj is not None and hasattr(level_sequence_obj, "get_bindings"):
                for binding in list(level_sequence_obj.get_bindings() or []):
                    try:
                        level_sequence_obj.remove_binding(binding)
                    except Exception:
                        pass
                cleanup["sequence_bindings_removed"] = True
        except Exception as exc:
            cleanup["errors"].append(f"remove_level_sequence_bindings: {exc}")
    if level_sequence_asset_path:
        try:
            if unreal.EditorAssetLibrary.does_asset_exist(level_sequence_asset_path):
                cleanup["level_sequence_deleted"] = bool(unreal.EditorAssetLibrary.delete_asset(level_sequence_asset_path))
        except Exception as exc:
            cleanup["errors"].append(f"delete_level_sequence_asset: {exc}")
    try:
        unique_asset_paths = []
        seen_asset_paths = set()
        for entry in list(clip_asset_references or []):
            asset_path = str(entry.get("asset_path") or "").strip()
            if not asset_path or asset_path in seen_asset_paths:
                continue
            seen_asset_paths.add(asset_path)
            unique_asset_paths.append(asset_path)
        cleanup["asset_unload_capability_reports"] = [
            _asset_unload_capability_report(asset_path)
            for asset_path in unique_asset_paths
        ]
    except Exception as exc:
        cleanup["errors"].append(f"asset_unload_capability_reports: {exc}")
    spawned_actor_labels = []
    try:
        for item in list((result or {}).get("spawned_roles") or []):
            actor_label = item.get("actor_label")
            if actor_label:
                spawned_actor_labels.append(str(actor_label))
        for item in list((result or {}).get("appearance_debug_by_body") or []):
            actor_label = item.get("actor_label")
            if actor_label:
                spawned_actor_labels.append(str(actor_label))
    except Exception as exc:
        cleanup["errors"].append(f"collect_spawned_actor_labels: {exc}")
    cleanup["spawned_actor_labels_targeted"] = sorted(set(spawned_actor_labels))
    try:
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for actor_label in cleanup["spawned_actor_labels_targeted"]:
            actor = _find_actor_by_label(actor_label)
            destroy_report = {
                "actor_label": str(actor_label),
                "found": bool(actor is not None),
                "geometry_cache_release": None,
                "destroy_called": False,
                "destroy_result": None,
                "errors": [],
            }
            if actor is not None:
                destroy_report["geometry_cache_release"] = _release_geometry_cache_component(actor)
                destroy_report["heavy_component_release"] = _release_heavy_components(actor)
                try:
                    destroy_result = actor_subsystem.destroy_actor(actor)
                    destroy_report["destroy_called"] = True
                    destroy_report["destroy_result"] = bool(destroy_result)
                except Exception as exc:
                    destroy_report["errors"].append(f"destroy_actor: {exc}")
            cleanup["spawned_actor_destroy_reports"].append(destroy_report)
    except Exception as exc:
        cleanup["errors"].append(f"destroy_spawned_roles: {exc}")
    try:
        removed = base.canonical_validation.mini.reconstruct_full_bedlam_scene.clear_existing_bedlam_bodies()
        cleanup["bedlam_actors_cleared"] = list(removed or [])
    except Exception as exc:
        cleanup["errors"].append(f"clear_existing_bedlam_bodies: {exc}")
    try:
        base.canonical_validation.mini._CAPTURE_SOURCE_INIT_STATE.clear()
        cleanup["capture_source_init_state_cleared"] = True
    except Exception as exc:
        cleanup["errors"].append(f"clear_capture_source_init_state: {exc}")
    try:
        cleanup["python_gc_collected"] = int(gc.collect())
    except Exception as exc:
        cleanup["errors"].append(f"python_gc_collect: {exc}")
    try:
        unreal.SystemLibrary.collect_garbage()
        cleanup["unreal_collect_garbage_called"] = True
    except Exception as exc:
        cleanup["errors"].append(f"unreal_collect_garbage: {exc}")
    try:
        if isinstance(result, dict):
            for key in ("spawned_roles", "appearance_debug_by_body", "spawn_material_diagnostics"):
                if key in result:
                    result[key] = []
            if isinstance(result.get("level_sequence"), dict):
                result["level_sequence"].clear()
            if isinstance(result.get("range_result"), dict):
                range_result = result.get("range_result") or {}
                for key in ("frame_records", "body_evaluations", "per_frame_body_states"):
                    if key in range_result:
                        range_result[key] = []
            cleanup["sequence_python_references_cleared"] = True
    except Exception as exc:
        cleanup["errors"].append(f"clear_python_references: {exc}")
    try:
        cleanup["clip_asset_unload_report"] = _unload_clip_assets(clip_asset_references)
    except Exception as exc:
        cleanup["errors"].append(f"unload_clip_assets: {exc}")
    try:
        cleanup["python_gc_collected_second_pass"] = int(gc.collect())
    except Exception as exc:
        cleanup["errors"].append(f"python_gc_collect_second_pass: {exc}")
    try:
        unreal.SystemLibrary.collect_garbage()
        cleanup["unreal_collect_garbage_called_second_pass"] = True
    except Exception as exc:
        cleanup["errors"].append(f"unreal_collect_garbage_second_pass: {exc}")
    try:
        time.sleep(0.25)
    except Exception:
        pass
    log_info("CLIP_CLEANUP " + json.dumps(cleanup, indent=2))
    return cleanup


def _argv_contains_any(*flags):
    argv = list(sys.argv[1:])
    return any(flag in argv for flag in flags)


def _load_run_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scene-root", type=Path, default=SCENE_ROOT)
    parser.add_argument("--manifest-path", "--manifest", dest="manifest_path", type=Path, default=MINISCENE_MANIFEST_PATH)
    parser.add_argument("--miniscene-index", type=int, default=MINISCENE_INDEX)
    parser.add_argument("--miniscene-id", default=MINISCENE_ID)
    parser.add_argument("--miniscene-room", default=MINISCENE_ROOM_HINT)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument(
        "--rgb-tonemap-mode",
        choices=("ldr_passthrough", "sequence_adaptive_rgb"),
        default=RGB_TONEMAP_MODE,
    )
    parser.add_argument(
        "--render-output-profile",
        choices=("full_debug", "dataset_rgb_only", "dataset_rgb_exr", "dataset_rgb_fast", "dataset_hdr_only", "preview_only"),
        default=RENDER_OUTPUT_PROFILE,
    )
    parser.add_argument("--batch", action="store_true", default=BATCH_RENDER_MINISCENES)
    parser.add_argument("--batch-max-miniscenes", "--max-clips", dest="batch_max_miniscenes", type=int, default=BATCH_MAX_MINISCENES)
    parser.add_argument("--batch-room-filter", default=BATCH_ROOM_FILTER)
    parser.add_argument("--batch-balanced-rooms", action="store_true", default=BATCH_BALANCED_ROOMS)
    parser.add_argument("--list-renderable", action="store_true", default=LIST_RENDERABLE_MINISCENES)
    parser.add_argument("--list-renderable-limit", type=int, default=LIST_RENDERABLE_LIMIT)
    parser.add_argument("--diagnose-renderability", action="store_true", default=DIAGNOSE_RENDERABILITY)
    parser.add_argument("--list-available-motions", action="store_true", default=LIST_AVAILABLE_MOTIONS)
    parser.add_argument("--reject-clips-with-artifacts", action="store_true", default=REJECT_CLIPS_WITH_ARTIFACTS)
    parser.add_argument("--offline-run-root", type=Path, default=None)
    parser.add_argument("--pause-after-spawn-before-render", action="store_true", default=False)
    parser.add_argument("--pause-after-spawn-seconds", type=float, default=0.0)
    parser.add_argument("--render-warmup-frame-count", type=int, default=0)
    parser.add_argument("--discard-warmup-frames", action="store_true", default=False)
    parser.add_argument("--probe-frame-before-render", action="store_true", default=False)
    parser.add_argument("--reject-beige-probe", action="store_true", default=False)
    parser.add_argument(
        "--bedlam-debug-appearance-mode",
        choices=("full", "no_humans", "body_only", "body_no_clothing", "body_no_hair"),
        default=BEDLAM_DEBUG_APPEARANCE_MODE,
    )
    parser.add_argument("--emit-memreport", action="store_true", default=EMIT_MEMREPORT)
    parser.add_argument("--emit-rhi-memory-dump", action="store_true", default=EMIT_RHI_MEMORY_DUMP)
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    args.raw_sys_argv = list(sys.argv)
    args.explicit_manifest_arg = _argv_contains_any("--manifest", "--manifest-path")
    args.explicit_miniscene_id_arg = _argv_contains_any("--miniscene-id")
    args.explicit_miniscene_room_arg = _argv_contains_any("--miniscene-room")
    args.explicit_miniscene_index_arg = _argv_contains_any("--miniscene-index")
    args.explicit_frame_start_arg = _argv_contains_any("--frame-start")
    args.explicit_frame_end_arg = _argv_contains_any("--frame-end")
    return args


def _apply_render_output_profile(config):
    profile = str(getattr(config, "render_output_profile", RENDER_OUTPUT_PROFILE))
    settings = {
        "full_debug": {
            "export_hdr": True,
            "export_erp_exr": True,
            "export_rgb_png": True,
            "keep_rgb_source_exr": True,
            "expect_single_longlat_export_pass": False,
            "capture_source_mode": "ldr_final_color",
            "adaptive_preview_frames": True,
            "fixed_preview_frames": True,
            "per_frame_stats": True,
            "body_eval_csv": True,
            "mp4_preview": True,
        },
        "dataset_rgb_exr": {
            "export_hdr": False,
            "export_erp_exr": True,
            "export_rgb_png": True,
            "keep_rgb_source_exr": True,
            "expect_single_longlat_export_pass": False,
            "capture_source_mode": "ldr_final_color",
            "adaptive_preview_frames": False,
            "fixed_preview_frames": False,
            "per_frame_stats": False,
            "body_eval_csv": False,
            "mp4_preview": False,
        },
        "dataset_rgb_fast": {
            "export_hdr": False,
            "export_erp_exr": False,
            "export_rgb_png": True,
            "keep_rgb_source_exr": False,
            "expect_single_longlat_export_pass": True,
            "capture_source_mode": "ldr_final_color",
            "adaptive_preview_frames": False,
            "fixed_preview_frames": False,
            "per_frame_stats": False,
            "body_eval_csv": False,
            "mp4_preview": True,
        },
        "dataset_hdr_only": {
            "export_hdr": False,
            "export_erp_exr": True,
            "export_rgb_png": False,
            "keep_rgb_source_exr": False,
            "expect_single_longlat_export_pass": True,
            "capture_source_mode": "hdr_scene_color",
            "adaptive_preview_frames": False,
            "fixed_preview_frames": False,
            "per_frame_stats": False,
            "body_eval_csv": False,
            "mp4_preview": False,
        },
        "dataset_rgb_only": {
            "export_hdr": False,
            "export_erp_exr": False,
            "export_rgb_png": True,
            "keep_rgb_source_exr": True,
            "expect_single_longlat_export_pass": False,
            "capture_source_mode": "ldr_final_color",
            "adaptive_preview_frames": False,
            "fixed_preview_frames": False,
            "per_frame_stats": False,
            "body_eval_csv": False,
            "mp4_preview": False,
        },
        "preview_only": {
            "export_hdr": False,
            "export_erp_exr": False,
            "export_rgb_png": True,
            "keep_rgb_source_exr": True,
            "expect_single_longlat_export_pass": False,
            "capture_source_mode": "ldr_final_color",
            "adaptive_preview_frames": False,
            "fixed_preview_frames": False,
            "per_frame_stats": False,
            "body_eval_csv": False,
            "mp4_preview": True,
        },
    }[profile]

    base.canonical_validation.mini.ENABLE_EXPORT_HDR = bool(settings["export_hdr"])
    base.canonical_validation.mini.ENABLE_EXPORT_ERP_EXR = bool(settings["export_erp_exr"])
    base.canonical_validation.mini.ENABLE_EXPORT_RGB_PNG = bool(settings["export_rgb_png"])
    base.canonical_validation.mini.ENABLE_ADAPTIVE_PREVIEW_FRAMES = bool(settings["adaptive_preview_frames"])
    base.canonical_validation.mini.ENABLE_FIXED_PREVIEW_FRAMES = bool(settings["fixed_preview_frames"])
    base.canonical_validation.mini.ENABLE_MP4_PREVIEW = bool(settings["mp4_preview"])
    base.canonical_validation.mini.KEEP_RGB_SOURCE_EXR = bool(settings["keep_rgb_source_exr"])
    base.canonical_validation.mini.EXPECT_SINGLE_LONGLAT_EXPORT_PASS = bool(settings["expect_single_longlat_export_pass"])
    base.canonical_validation.mini.DEFAULT_RGB_TONEMAP_MODE = str(getattr(config, "rgb_tonemap_mode", RGB_TONEMAP_MODE))
    base.canonical_validation.ENABLE_PER_FRAME_STATS = bool(settings["per_frame_stats"])
    base.canonical_validation.ENABLE_BODY_EVAL_CSV = bool(settings["body_eval_csv"])
    base.canonical_validation.ENABLE_MP4_PREVIEW = bool(settings["mp4_preview"])
    base.ERP_CAPTURE_SOURCE_MODE = str(settings["capture_source_mode"])
    if str(settings["capture_source_mode"]) == "ldr_final_color":
        base.canonical_validation.mini.DEFAULT_CAPTURE_SOURCE_MODE = "ldr_final_color"
        if str(base.canonical_validation.mini.DEFAULT_RGB_TONEMAP_MODE) == "sequence_adaptive_rgb":
            base.canonical_validation.mini.DEFAULT_FRAME_PNG_MODE = "sequence_adaptive_rgb"
            base.canonical_validation.mini.DEFAULT_RGB_OUTPUT_CONVENTION = "sequence_adaptive_rgb"
            settings["keep_rgb_source_exr"] = True
            base.canonical_validation.mini.KEEP_RGB_SOURCE_EXR = True
        else:
            base.canonical_validation.mini.DEFAULT_FRAME_PNG_MODE = "ldr_passthrough"
            base.canonical_validation.mini.DEFAULT_RGB_OUTPUT_CONVENTION = "unreal_final_color_ldr_capture"
    else:
        base.canonical_validation.mini.DEFAULT_CAPTURE_SOURCE_MODE = "hdr_scene_color"
        base.canonical_validation.mini.DEFAULT_FRAME_PNG_MODE = "adaptive_preview"
        base.canonical_validation.mini.DEFAULT_RGB_OUTPUT_CONVENTION = "custom_tonemapped_debug_preview"
    explicit_cli_profile = bool(_argv_contains_any("--render-output-profile"))
    profile_source = "default"
    if bool(getattr(config, "batch_inherited_render_output_profile", False)):
        profile_source = "batch_inherited"
    elif explicit_cli_profile:
        profile_source = "cli"
    return {
        "requested_render_output_profile": str(getattr(config, "render_output_profile", RENDER_OUTPUT_PROFILE)),
        "rgb_tonemap_mode": str(getattr(config, "rgb_tonemap_mode", RGB_TONEMAP_MODE)),
        "effective_render_output_profile": profile,
        "profile_source": profile_source,
        "render_output_profile": profile,
        **settings,
    }


def _load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_scene_collision_metadata_json(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"scene_collision_metadata.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _miniscene_brief(miniscene, index=None):
    humans = list(miniscene.get("humans", []))
    return {
        "index": None if index is None else int(index),
        "miniscene_id": miniscene.get("miniscene_id"),
        "room": miniscene.get("room"),
        "scene_type": miniscene.get("scene_type"),
        "human_count": len(humans),
        "motion_ids": [_canonical_motion_id(h.get("motion_id")) for h in humans],
        "humans": [
            {
                "spawn_pose_index": h.get("spawn_pose_index"),
                "motion_id": _canonical_motion_id(h.get("motion_id")),
                "position_xyz_m": h.get("position_xyz_m"),
                "yaw_rad": h.get("yaw_rad"),
                "activity_hint": h.get("activity_hint"),
                "target_object": h.get("target_object"),
                "room": h.get("room"),
            }
            for h in humans
        ],
    }


def _startup_manifest_validation(config):
    scene_root = None
    if getattr(config, "scene_root", None) is not None:
        scene_root = Path(config.scene_root).expanduser().resolve()
    manifest_path = Path(config.manifest_path).expanduser()
    if not manifest_path.is_absolute() and scene_root is not None:
        manifest_path = (scene_root / manifest_path).resolve()
    else:
        manifest_path = manifest_path.resolve()
    log_info("STARTUP_RAW_SYS_ARGV " + json.dumps(config.raw_sys_argv))
    log_info(
        "STARTUP_CONFIG_REQUEST "
        + json.dumps(
            {
                "scene_root": None if scene_root is None else str(scene_root),
                "resolved_manifest_path": str(manifest_path),
                "miniscene_index": config.miniscene_index,
                "miniscene_id": getattr(config, "miniscene_id", None),
                "miniscene_room": getattr(config, "miniscene_room", None),
                "frame_start_override": getattr(config, "frame_start", None),
                "frame_end_override": getattr(config, "frame_end", None),
                "explicit_manifest_arg": bool(getattr(config, "explicit_manifest_arg", False)),
                "explicit_miniscene_id_arg": bool(getattr(config, "explicit_miniscene_id_arg", False)),
                "explicit_miniscene_room_arg": bool(getattr(config, "explicit_miniscene_room_arg", False)),
                "explicit_miniscene_index_arg": bool(getattr(config, "explicit_miniscene_index_arg", False)),
            },
            indent=2,
        )
    )
    if scene_root is not None and not scene_root.exists():
        raise RuntimeError(f"Scene root does not exist: {scene_root}")
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest path does not exist: {manifest_path}")
    manifest = _load_manifest(manifest_path)
    miniscenes = list(manifest.get("miniscenes", []))
    if scene_root is None:
        scene_root = manifest_path.parent.parent
    log_info(
        "STARTUP_MANIFEST_SUMMARY "
        + json.dumps(
            {
                "scene_root": str(scene_root),
                "manifest_path": str(manifest_path),
                "scene_count": len(miniscenes),
            },
            indent=2,
        )
    )
    return scene_root, manifest_path, manifest, miniscenes


def _configure_scene_root_paths(scene_root: Path):
    scene_root = Path(scene_root).expanduser().resolve()
    base.SCENE_METADATA_PATH = scene_root / "scene_collision_metadata.json"
    base.BLENDER_LIGHT_MANIFEST_PATH = scene_root / "blender_light_manifest.json"
    base.USD_STAGE_PATH = scene_root / "usd_export" / "export_scene.blend" / "export_scene.usdc"
    base.USD_LIGHT_MANIFEST_PATH = scene_root / "usd_export" / "usd_light_manifest.json"
    return {
        "scene_root": str(scene_root),
        "scene_metadata_path": str(base.SCENE_METADATA_PATH),
        "blender_light_manifest_path": str(base.BLENDER_LIGHT_MANIFEST_PATH),
        "usd_stage_path": str(base.USD_STAGE_PATH),
        "usd_light_manifest_path": str(base.USD_LIGHT_MANIFEST_PATH),
    }


def _normalize_path_string(value):
    if value in (None, ""):
        return None
    try:
        return str(Path(str(value)).expanduser().resolve())
    except Exception:
        return str(value)


def _validate_loaded_usd_stage_matches_scene_root(expected_stage_path: Path):
    actors = unreal.EditorLevelLibrary.get_all_level_actors()
    stage_actors = base._find_usd_stage_actors(actors)
    expected_stage_path = Path(expected_stage_path).expanduser().resolve()
    rows = []
    for actor in stage_actors:
        root_layer = base._safe_actor_property(actor, "root_layer")
        root_layer_str = base._json_safe_value(root_layer)
        normalized = _normalize_path_string(root_layer_str)
        row = {
            "actor_label": actor.get_actor_label(),
            "actor_name": actor.get_name(),
            "actor_class": actor.get_class().get_name(),
            "root_layer": root_layer_str,
            "normalized_root_layer": normalized,
            "matches_expected_stage": bool(normalized == str(expected_stage_path)),
        }
        rows.append(row)
    report = {
        "expected_stage_path": str(expected_stage_path),
        "stage_actor_count": len(stage_actors),
        "stage_actors": rows,
        "matching_stage_actor_count": sum(1 for row in rows if row["matches_expected_stage"]),
    }
    if not stage_actors:
        raise RuntimeError(
            "No UsdStageActor found in the current level. "
            f"Expected loaded USD scene: {expected_stage_path}"
        )
    if report["matching_stage_actor_count"] < 1:
        raise RuntimeError(
            "Loaded USD scene does not match requested scene root. "
            f"Expected: {expected_stage_path} | Found stage actors: {json.dumps(rows, indent=2)}"
        )
    return report


def _load_miniscene(manifest_path, miniscene_index):
    manifest = _load_manifest(manifest_path)
    miniscenes = manifest.get("miniscenes", [])
    if miniscene_index < 0 or miniscene_index >= len(miniscenes):
        raise RuntimeError(
            f"Requested miniscene index {miniscene_index}, but only {len(miniscenes)} exist"
        )
    return manifest, miniscenes[miniscene_index]


def _resolve_requested_miniscene_index(manifest_path, miniscene_index=None, miniscene_id=None, room_hint=None):
    manifest = _load_manifest(manifest_path)
    miniscenes = manifest.get("miniscenes", [])
    if miniscene_id not in (None, ""):
        target = str(miniscene_id)
        for index, miniscene in enumerate(miniscenes):
            if str(miniscene.get("miniscene_id")) == target:
                return manifest, index
        raise RuntimeError(f"Could not find miniscene_id={target!r} in manifest.")
    if room_hint not in (None, ""):
        room_hint_lower = str(room_hint).lower()
        for index, miniscene in enumerate(miniscenes):
            room_name = str(miniscene.get("room") or "")
            if room_hint_lower in room_name.lower():
                return manifest, index
        raise RuntimeError(f"Could not find room hint {room_hint!r} in manifest.")
    if miniscene_index is None:
        miniscene_index = 0
    if miniscene_index < 0 or miniscene_index >= len(miniscenes):
        raise RuntimeError(
            f"Requested miniscene index {miniscene_index}, but only {len(miniscenes)} exist"
        )
    return manifest, int(miniscene_index)


def _available_geometry_cache_asset_ids():
    return set(base.canonical_validation.mini._available_geometry_cache_asset_ids())


def _geometry_cache_registry_index():
    registry_assets = base.canonical_validation.mini.reconstruct_one_bedlam_body._list_available_registry_body_assets()
    index = {}
    for asset_data in registry_assets:
        asset_id = str(asset_data.asset_name)
        if asset_id.endswith("_0000"):
            continue
        package_name = str(getattr(asset_data, "package_name", ""))
        asset_name = str(getattr(asset_data, "asset_name", ""))
        object_path = str(getattr(asset_data, "object_path", ""))
        if not object_path and package_name and asset_name:
            object_path = f"{package_name}.{asset_name}"
        asset_class_path = ""
        try:
            class_path_obj = getattr(asset_data, "asset_class_path", None)
            if class_path_obj is not None:
                asset_class_path = str(class_path_obj)
        except Exception:
            asset_class_path = ""
        row = {
            "asset_id": asset_id,
            "asset_name": asset_name,
            "package_name": package_name,
            "package_path": str(getattr(asset_data, "package_path", "")),
            "object_path": object_path,
            "asset_class_path": asset_class_path,
        }
        index.setdefault(asset_id, []).append(row)
    return index


def _motion_identity(motion_id):
    parts = str(motion_id).split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return str(motion_id)


def _missing_manifest_motion_ids(miniscene, available_asset_ids):
    missing = []
    for human in miniscene.get("humans", []):
        motion_id = _canonical_motion_id(human["motion_id"])
        if motion_id not in available_asset_ids:
            missing.append(motion_id)
    return missing


def _duplicate_motion_count(miniscene):
    motion_ids = [_canonical_motion_id(human["motion_id"]) for human in miniscene.get("humans", [])]
    return max(0, len(motion_ids) - len(set(motion_ids)))


def _luminance_summary(stats_block):
    images = ((stats_block or {}).get("data") or {}).get("images") or []
    if not images:
        return None
    first = images[0]
    return {
        "mean": first.get("luminance_mean"),
        "p95": first.get("luminance_p95"),
        "max": first.get("luminance_max"),
    }


def _select_renderable_miniscene(manifest_path, requested_index, allow_fallback=True):
    manifest = _load_manifest(manifest_path)
    miniscenes = manifest.get("miniscenes", [])
    if requested_index < 0 or requested_index >= len(miniscenes):
        raise RuntimeError(
            f"Requested miniscene index {requested_index}, but only {len(miniscenes)} exist"
        )
    available_asset_ids = _available_geometry_cache_asset_ids()
    first = miniscenes[requested_index]
    missing = _missing_manifest_motion_ids(first, available_asset_ids)
    if not missing:
        return manifest, first, requested_index, {
            "requested_index": int(requested_index),
            "selected_index": int(requested_index),
            "fallback_used": False,
            "missing_motion_ids": [],
        }
    if not allow_fallback or not ALLOW_FALLBACK_TO_NEXT_RENDERABLE:
        requested_id = first.get("miniscene_id")
        requested_room = first.get("room")
        raise RuntimeError(
            f"Selected miniscene is not renderable in this Unreal registry. "
            f"requested_index={requested_index} miniscene_id={requested_id!r} room={requested_room!r} "
            f"missing_motions={missing}"
        )
    requested_room = str(first.get("room") or "")
    fallback_order = []
    if PREFER_SAME_ROOM_ON_FALLBACK and requested_room:
        same_room = []
        for candidate_index, candidate in enumerate(miniscenes):
            if candidate_index == requested_index:
                continue
            if str(candidate.get("room") or "") != requested_room:
                continue
            same_room.append((abs(candidate_index - requested_index), candidate_index, candidate))
        same_room.sort(key=lambda item: (item[0], item[1]))
        fallback_order.extend((candidate_index, candidate) for _dist, candidate_index, candidate in same_room)

    for candidate_index in range(requested_index + 1, len(miniscenes)):
        candidate = miniscenes[candidate_index]
        if PREFER_SAME_ROOM_ON_FALLBACK and requested_room and str(candidate.get("room") or "") == requested_room:
            continue
        fallback_order.append((candidate_index, candidate))

    for candidate_index in range(0, requested_index):
        candidate = miniscenes[candidate_index]
        if PREFER_SAME_ROOM_ON_FALLBACK and requested_room and str(candidate.get("room") or "") == requested_room:
            continue
        fallback_order.append((candidate_index, candidate))

    seen_indices = set()
    for candidate_index, candidate in fallback_order:
        if candidate_index in seen_indices:
            continue
        seen_indices.add(candidate_index)
        candidate_missing = _missing_manifest_motion_ids(candidate, available_asset_ids)
        if not candidate_missing:
            return manifest, candidate, candidate_index, {
                "requested_index": int(requested_index),
                "selected_index": int(candidate_index),
                "fallback_used": True,
                "missing_motion_ids": missing,
                "requested_room": requested_room,
                "selected_room": str(candidate.get("room") or ""),
                "same_room_fallback": bool(str(candidate.get("room") or "") == requested_room),
            }
    raise RuntimeError(
        f"No renderable miniscene found at or after index {requested_index}. First missing motions: {missing}"
    )


def _iter_renderable_miniscenes(manifest_path, room_filter=None):
    manifest = _load_manifest(manifest_path)
    miniscenes = list(manifest.get("miniscenes", []))
    available_asset_ids = _available_geometry_cache_asset_ids()
    room_filter_lower = None if room_filter in (None, "") else str(room_filter).lower()
    items = []
    for index, miniscene in enumerate(miniscenes):
        room_name = str(miniscene.get("room") or "")
        if room_filter_lower is not None and room_filter_lower not in room_name.lower():
            continue
        missing = _missing_manifest_motion_ids(miniscene, available_asset_ids)
        if missing:
            continue
        items.append(
            {
                "index": int(index),
                "miniscene": miniscene,
                "room": room_name,
                "selection_info": {
                    "requested_index": int(index),
                    "selected_index": int(index),
                    "fallback_used": False,
                    "missing_motion_ids": [],
                },
            }
        )
    return manifest, items


def _list_renderable_miniscenes_report(manifest_path, room_filter=None, limit=50):
    manifest, items = _iter_renderable_miniscenes(manifest_path, room_filter=room_filter)
    available_asset_ids = _available_geometry_cache_asset_ids()
    room_distribution = {}
    for item in items:
        room_distribution[item["room"]] = room_distribution.get(item["room"], 0) + 1

    rows = []
    for item in items[: int(limit)]:
        miniscene = item["miniscene"]
        rows.append(
            {
                "index": int(item["index"]),
                "miniscene_id": miniscene.get("miniscene_id"),
                "room": miniscene.get("room"),
                "scene_type": miniscene.get("scene_type"),
                "human_count": len(miniscene.get("humans", [])),
                "motion_ids": [_canonical_motion_id(h["motion_id"]) for h in miniscene.get("humans", [])],
                "duplicate_motion_count": _duplicate_motion_count(miniscene),
            }
        )

    report = {
        "mode": "list_renderable_miniscenes",
        "manifest_path": str(manifest_path),
        "room_filter": room_filter,
        "available_geometry_cache_asset_ids_count": len(available_asset_ids),
        "renderable_count": len(items),
        "room_distribution": room_distribution,
        "rows": rows,
    }
    log_info("RENDERABLE_MINISCENES " + json.dumps(report, indent=2))
    return report


def _greedy_unlock_set(scene_rows):
    remaining = {
        row["miniscene_id"]: set(row["missing_motion_ids"])
        for row in scene_rows
        if not row["renderable"] and row["missing_motion_ids"]
    }
    unlock_steps = []
    while remaining:
        coverage = {}
        for missing_ids in remaining.values():
            for motion_id in missing_ids:
                coverage[motion_id] = coverage.get(motion_id, 0) + 1
        if not coverage:
            break
        best_motion_id, best_count = sorted(
            coverage.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        unlocked = sorted(
            miniscene_id
            for miniscene_id, missing_ids in remaining.items()
            if best_motion_id in missing_ids
        )
        unlock_steps.append(
            {
                "motion_id": best_motion_id,
                "would_unlock_or_partially_unlock_count": int(best_count),
                "affected_miniscene_ids": unlocked,
            }
        )
        for miniscene_id in list(remaining.keys()):
            remaining[miniscene_id].discard(best_motion_id)
            if not remaining[miniscene_id]:
                del remaining[miniscene_id]
    return unlock_steps


def _diagnose_renderability_report(manifest_path, room_filter=None):
    manifest = _load_manifest(manifest_path)
    miniscenes = list(manifest.get("miniscenes", []))
    room_filter_lower = None if room_filter in (None, "") else str(room_filter).lower()
    registry_index = _geometry_cache_registry_index()
    available_asset_ids = set(registry_index.keys())

    rows = []
    missing_frequency = {}
    for index, miniscene in enumerate(miniscenes):
        room_name = str(miniscene.get("room") or "")
        if room_filter_lower is not None and room_filter_lower not in room_name.lower():
            continue
        motion_rows = []
        missing_motion_ids = []
        for human in list(miniscene.get("humans", [])):
            motion_id = _canonical_motion_id(human["motion_id"])
            asset_rows = registry_index.get(motion_id, [])
            exists = bool(asset_rows)
            if not exists:
                missing_motion_ids.append(motion_id)
                missing_frequency[motion_id] = missing_frequency.get(motion_id, 0) + 1
            motion_rows.append(
                {
                    "required_motion_id": motion_id,
                    "geometry_cache_exists": exists,
                    "resolved_asset_path": asset_rows[0]["object_path"] if exists else None,
                    "resolved_package_name": asset_rows[0]["package_name"] if exists else None,
                    "resolved_asset_class_path": asset_rows[0]["asset_class_path"] if exists else None,
                    "matching_registry_assets": asset_rows,
                    "missing_reason": None if exists else "geometry_cache_not_in_current_unreal_registry",
                }
            )
        row = {
            "index": int(index),
            "miniscene_id": miniscene.get("miniscene_id"),
            "room": room_name,
            "number_of_humans": len(miniscene.get("humans", [])),
            "motion_ids": [_canonical_motion_id(h["motion_id"]) for h in miniscene.get("humans", [])],
            "motions": motion_rows,
            "missing_motion_ids": sorted(set(missing_motion_ids)),
            "renderable": not missing_motion_ids,
        }
        rows.append(row)

    missing_ranked = [
        {"motion_id": motion_id, "missing_count": int(count)}
        for motion_id, count in sorted(missing_frequency.items(), key=lambda item: (-item[1], item[0]))
    ]
    unlock_set = _greedy_unlock_set(rows)
    report = {
        "mode": "diagnose_renderability",
        "manifest_path": str(manifest_path),
        "room_filter": room_filter,
        "available_geometry_cache_asset_ids_count": len(available_asset_ids),
        "total_matching_miniscenes": len(rows),
        "renderable_count": sum(1 for row in rows if row["renderable"]),
        "missing_motion_ids_ranked_by_frequency": missing_ranked,
        "smallest_set_of_missing_motion_assets_that_would_unlock_most_miniscenes": unlock_set,
        "miniscenes": rows,
    }
    log_info("RENDERABILITY_DIAGNOSTIC " + json.dumps(report, indent=2))
    return report


def _list_available_motions_report(manifest_path):
    manifest = _load_manifest(manifest_path)
    registry_index = _geometry_cache_registry_index()
    available_motion_ids = sorted(registry_index.keys())

    available_by_identity = {}
    for motion_id in available_motion_ids:
        identity = _motion_identity(motion_id)
        available_by_identity.setdefault(identity, []).append(
            {
                "motion_id": motion_id,
                "registry_assets": registry_index.get(motion_id, []),
            }
        )

    miniscenes = list(manifest.get("miniscenes", []))
    planner_motion_ids = sorted(
        {
            _canonical_motion_id(human["motion_id"])
            for miniscene in miniscenes
            for human in list(miniscene.get("humans", []))
        }
    )
    planner_identities = {}
    for motion_id in planner_motion_ids:
        identity = _motion_identity(motion_id)
        planner_identities.setdefault(identity, []).append(motion_id)

    missing_identities_compared_to_planner = sorted(
        identity
        for identity in planner_identities
        if identity not in available_by_identity
    )

    partially_missing_identities = []
    for identity, planner_ids in sorted(planner_identities.items()):
        available_ids = {row["motion_id"] for row in available_by_identity.get(identity, [])}
        missing_ids = sorted(motion_id for motion_id in planner_ids if motion_id not in available_ids)
        if missing_ids and available_ids:
            partially_missing_identities.append(
                {
                    "identity": identity,
                    "available_count": len(available_ids),
                    "planner_motion_count": len(planner_ids),
                    "missing_motion_ids": missing_ids,
                }
            )

    report = {
        "mode": "list_available_motions",
        "manifest_path": str(manifest_path),
        "total_discovered_motion_ids": len(available_motion_ids),
        "available_identities": {
            identity: {
                "geometry_cache_count": len(rows),
                "motion_ids": [row["motion_id"] for row in rows],
            }
            for identity, rows in sorted(available_by_identity.items())
        },
        "planner_motion_id_count": len(planner_motion_ids),
        "planner_identities": {
            identity: {
                "planner_motion_count": len(sorted(set(ids))),
                "planner_motion_ids": sorted(set(ids)),
            }
            for identity, ids in sorted(planner_identities.items())
        },
        "missing_identities_compared_to_planner": missing_identities_compared_to_planner,
        "partially_missing_identities_compared_to_planner": partially_missing_identities,
    }
    log_info("AVAILABLE_MOTION_FAMILIES " + json.dumps(report, indent=2))
    return report


def _room_priority_key(room_name):
    room_name = str(room_name or "").lower()
    priorities = [
        "kitchen",
        "living-room",
        "dining-room",
        "bedroom",
        "bathroom",
    ]
    for rank, token in enumerate(priorities):
        if token in room_name:
            return (rank, room_name)
    return (len(priorities), room_name)


def _select_batch_miniscenes(manifest_path, max_count, room_filter=None, balanced_rooms=False):
    source_manifest, items = _iter_renderable_miniscenes(manifest_path, room_filter=room_filter)
    if not items:
        return source_manifest, []
    items = [
        item
        for item in items
        if len(list(item["miniscene"].get("humans", []))) >= int(MIN_HUMAN_COUNT_PER_RENDER)
    ]
    if not items:
        return source_manifest, []

    rooms = {}
    for item in items:
        rooms.setdefault(item["room"], []).append(item)
    for room_items in rooms.values():
        room_items.sort(
            key=lambda item: (
                _duplicate_motion_count(item["miniscene"]),
                len(item["miniscene"].get("humans", [])),
                int(item["index"]),
            )
        )

    ordered_room_names = sorted(rooms.keys(), key=_room_priority_key)
    selected = []
    seen_ids = set()
    per_room_counts = {room_name: 0 for room_name in ordered_room_names}

    def _first_unseen_room_item(room_items, start_index=0):
        for idx in range(max(0, int(start_index)), len(room_items)):
            candidate = room_items[idx]
            candidate_id = candidate["miniscene"].get("miniscene_id")
            if candidate_id not in seen_ids:
                return idx, candidate
        return None, None

    if balanced_rooms:
        preferred_items = [item for item in items if _duplicate_motion_count(item["miniscene"]) == 0]
        fallback_items = [item for item in items if _duplicate_motion_count(item["miniscene"]) > 0]
        candidate_pool = preferred_items if preferred_items else items
        rooms = {}
        for item in candidate_pool:
            rooms.setdefault(item["room"], []).append(item)
        ordered_room_names = sorted(rooms.keys(), key=_room_priority_key)
        # First pass: cover multi-human scenes across rooms before considering anything smaller.
        for preferred_scene_type in ("two_human", "single_human"):
            for room_name in ordered_room_names:
                if len(selected) >= int(max_count):
                    break
                room_items = rooms[room_name]
                candidate = next(
                    (
                        item
                        for item in room_items
                        if item["miniscene"].get("scene_type") == preferred_scene_type
                        and item["miniscene"].get("miniscene_id") not in seen_ids
                    ),
                    None,
                )
                if candidate is None:
                    continue
                candidate_id = candidate["miniscene"].get("miniscene_id")
                selected.append(candidate)
                seen_ids.add(candidate_id)
                per_room_counts[room_name] += 1

        per_room_cap = int(BATCH_MAX_PER_ROOM_FIRST_PASS)
        round_index = 1
        while len(selected) < int(max_count):
            added_this_round = False
            for room_name in ordered_room_names:
                room_items = rooms[room_name]
                if per_room_counts[room_name] >= per_room_cap:
                    continue
                _candidate_index, candidate = _first_unseen_room_item(room_items, start_index=round_index)
                if candidate is not None:
                    candidate_id = candidate["miniscene"].get("miniscene_id")
                    selected.append(candidate)
                    seen_ids.add(candidate_id)
                    per_room_counts[room_name] += 1
                    added_this_round = True
                    if len(selected) >= int(max_count):
                        break
            if not added_this_round:
                break
            round_index += 1

        if len(selected) < int(max_count):
            round_index = 1
            while len(selected) < int(max_count):
                added_this_round = False
                for room_name in ordered_room_names:
                    room_items = rooms[room_name]
                    _candidate_index, candidate = _first_unseen_room_item(room_items, start_index=round_index)
                    if candidate is not None:
                        candidate_id = candidate["miniscene"].get("miniscene_id")
                        selected.append(candidate)
                        seen_ids.add(candidate_id)
                        per_room_counts[room_name] += 1
                        added_this_round = True
                        if len(selected) >= int(max_count):
                            break
                if not added_this_round:
                    break
                round_index += 1

        if len(selected) < int(max_count) and fallback_items:
            fallback_rooms = {}
            for item in fallback_items:
                candidate_id = item["miniscene"].get("miniscene_id")
                if candidate_id in seen_ids:
                    continue
                fallback_rooms.setdefault(item["room"], []).append(item)
            fallback_room_names = sorted(fallback_rooms.keys(), key=_room_priority_key)
            round_index = 0
            while len(selected) < int(max_count):
                added_this_round = False
                for room_name in fallback_room_names:
                    room_items = fallback_rooms[room_name]
                    _candidate_index, candidate = _first_unseen_room_item(room_items, start_index=round_index)
                    if candidate is not None:
                        candidate_id = candidate["miniscene"].get("miniscene_id")
                        selected.append(candidate)
                        seen_ids.add(candidate_id)
                        per_room_counts[room_name] = per_room_counts.get(room_name, 0) + 1
                        added_this_round = True
                        if len(selected) >= int(max_count):
                            break
                if not added_this_round:
                    break
                round_index += 1
    else:
        round_index = 0
        while len(selected) < int(max_count):
            added_this_round = False
            for room_name in ordered_room_names:
                room_items = rooms[room_name]
                if round_index < len(room_items):
                    candidate = room_items[round_index]
                    candidate_id = candidate["miniscene"].get("miniscene_id")
                    if candidate_id not in seen_ids:
                        selected.append(candidate)
                        seen_ids.add(candidate_id)
                        per_room_counts[room_name] += 1
                        added_this_round = True
                        if len(selected) >= int(max_count):
                            break
            if not added_this_round:
                break
            round_index += 1
    return source_manifest, selected


def _human_spawn_location(human_record):
    return base.transform_infinigen_position_to_bedlam(human_record["position_xyz_m"])


def _human_yaw_deg(human_record):
    return base.transform_infinigen_yaw_to_bedlam(human_record.get("yaw_rad"))


def _appearance_fields(body_slot=0):
    per_slot = getattr(base.canonical_validation, "DEFAULT_FULL_APPEARANCE_FIELDS_BY_SLOT", {}) or {}
    slot_fields = dict(per_slot.get(int(body_slot), base.BRIDGE_APPEARANCE_FIELDS))
    return {
        "texture_body": slot_fields.get("texture_body") if base.ENABLE_BODY_TEXTURE else None,
        "texture_clothing": slot_fields.get("texture_clothing") if base.ENABLE_CLOTHING else None,
        "texture_clothing_overlay": slot_fields.get("texture_clothing_overlay") if base.ENABLE_CLOTHING else None,
        "hair": slot_fields.get("hair") if base.ENABLE_HAIR else None,
        "haircolor": slot_fields.get("haircolor") if base.ENABLE_HAIR else None,
        "shoe": slot_fields.get("shoe") if base.ENABLE_SHOES else None,
        "shoe_offset": slot_fields.get("shoe_offset") if base.ENABLE_SHOES else None,
    }


def _motion_phase_offset_frames(human_record, body_slot, human_count):
    explicit = human_record.get("motion_phase_offset_frames")
    if explicit is not None:
        return int(explicit)
    if human_count <= 1:
        return 0
    return 0 if int(body_slot) == 0 else 15 * int(body_slot)


def _canonical_motion_id(motion_id):
    motion_id = str(motion_id)
    suffix = "_root_trajectory"
    if motion_id.endswith(suffix):
        return motion_id[: -len(suffix)]
    return motion_id


def _build_body_spec(human_record, body_slot):
    spawn_location = _human_spawn_location(human_record)
    yaw_deg = _human_yaw_deg(human_record)
    appearance = _appearance_fields(body_slot=body_slot)
    phase_offset_frames = _motion_phase_offset_frames(human_record, body_slot, human_record.get("_human_count", 1))
    return {
        "asset_id": _canonical_motion_id(human_record["motion_id"]),
        "x": float(spawn_location.x),
        "y": float(spawn_location.y),
        "z": float(spawn_location.z),
        "yaw": float(yaw_deg),
        "pitch": 0.0,
        "roll": 0.0,
        "start_frame": 1 + int(phase_offset_frames),
        **appearance,
        "body_slot": int(body_slot),
        "motion_phase_offset_frames": int(phase_offset_frames),
        "appearance_id": str(appearance.get("texture_body") or f"body_slot_{body_slot}"),
    }


def _anchor_location(body_specs):
    if not body_specs:
        return unreal.Vector(0.0, 0.0, 0.0)
    x = sum(spec["x"] for spec in body_specs) / len(body_specs)
    y = sum(spec["y"] for spec in body_specs) / len(body_specs)
    z = sum(spec["z"] for spec in body_specs) / len(body_specs)
    return unreal.Vector(float(x), float(y), float(z))


def _body_spec_report(body_spec, human_record):
    return {
        "motion_id": body_spec["asset_id"],
        "spawn_pose_index": human_record.get("spawn_pose_index"),
        "activity_hint": human_record.get("activity_hint"),
        "target_object": human_record.get("target_object"),
        "location_cm": {"x": body_spec["x"], "y": body_spec["y"], "z": body_spec["z"]},
        "yaw_deg": body_spec["yaw"],
        "body_slot": body_spec["body_slot"],
        "body_id": body_spec.get("appearance_id"),
        "motion_phase_offset_frames": int(body_spec.get("motion_phase_offset_frames", 0)),
        "appearance_id": body_spec.get("appearance_id"),
    }


def _human_room_names(miniscene):
    rooms = []
    for human in miniscene.get("humans", []):
        room = human.get("room")
        if room and room not in rooms:
            rooms.append(room)
    return rooms


def _human_anchor_infinigen_m(miniscene):
    humans = list(miniscene.get("humans", []))
    if not humans:
        return [0.0, 0.0, 0.0]
    xs = [float(human["position_xyz_m"][0]) for human in humans]
    ys = [float(human["position_xyz_m"][1]) for human in humans]
    zs = [float(human["position_xyz_m"][2]) for human in humans]
    return [
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        sum(zs) / len(zs),
    ]


def _room_floor_z_m(scene_metadata, room_name):
    room_record = base._room_record_from_metadata(scene_metadata, room_name)
    if room_record is None:
        return 0.0
    return float(room_record.get("floor_z") or 0.0)


def _clamp(value, lo, hi):
    return max(float(lo), min(float(value), float(hi)))


def _room_bbox_inset_xy(room_record, xy, margin_m):
    bbox = (room_record or {}).get("bbox") or {}
    mn = bbox.get("min_xyz")
    mx = bbox.get("max_xyz")
    if not mn or not mx:
        return [float(xy[0]), float(xy[1])]
    min_x = float(mn[0]) + float(margin_m)
    max_x = float(mx[0]) - float(margin_m)
    min_y = float(mn[1]) + float(margin_m)
    max_y = float(mx[1]) - float(margin_m)
    if min_x > max_x:
        min_x = max_x = 0.5 * (float(mn[0]) + float(mx[0]))
    if min_y > max_y:
        min_y = max_y = 0.5 * (float(mn[1]) + float(mx[1]))
    return [
        _clamp(float(xy[0]), min_x, max_x),
        _clamp(float(xy[1]), min_y, max_y),
    ]


def _orientation_xy(a, b, c):
    return (
        (float(b[1]) - float(a[1])) * (float(c[0]) - float(b[0]))
        - (float(b[0]) - float(a[0])) * (float(c[1]) - float(b[1]))
    )


def _on_segment_xy(a, b, c):
    return (
        min(float(a[0]), float(c[0])) - 1e-9 <= float(b[0]) <= max(float(a[0]), float(c[0])) + 1e-9
        and min(float(a[1]), float(c[1])) - 1e-9 <= float(b[1]) <= max(float(a[1]), float(c[1])) + 1e-9
    )


def _segments_intersect_xy(a1, a2, b1, b2):
    o1 = _orientation_xy(a1, a2, b1)
    o2 = _orientation_xy(a1, a2, b2)
    o3 = _orientation_xy(b1, b2, a1)
    o4 = _orientation_xy(b1, b2, a2)

    if (o1 > 0.0 and o2 < 0.0 or o1 < 0.0 and o2 > 0.0) and (o3 > 0.0 and o4 < 0.0 or o3 < 0.0 and o4 > 0.0):
        return True
    if abs(o1) <= 1e-9 and _on_segment_xy(a1, b1, a2):
        return True
    if abs(o2) <= 1e-9 and _on_segment_xy(a1, b2, a2):
        return True
    if abs(o3) <= 1e-9 and _on_segment_xy(b1, a1, b2):
        return True
    if abs(o4) <= 1e-9 and _on_segment_xy(b1, a2, b2):
        return True
    return False


def _segment_intersects_polygon_xy(start_xy, end_xy, polygon_xy):
    pts = [(float(px), float(py)) for px, py in list(polygon_xy or [])]
    if len(pts) < 3:
        return False
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        if _segments_intersect_xy(start_xy, end_xy, a, b):
            return True
    return False


def _camera_room_line_of_sight_clear(candidate_xy, anchor_xy, centroid_xy, obstacles):
    targets = []
    if anchor_xy is not None:
        targets.append(("anchor", [float(anchor_xy[0]), float(anchor_xy[1])]))
    if centroid_xy is not None:
        targets.append(("room_centroid", [float(centroid_xy[0]), float(centroid_xy[1])]))
    if not targets or not CAMERA_ENABLE_INTERIOR_LOS_CHECK:
        return {"clear": True, "target": None, "reason": None}

    for target_name, target_xy in targets:
        blocked_by = None
        for obstacle in obstacles or []:
            footprint = obstacle.get("footprint_world_xy") or []
            if len(footprint) < 3:
                continue
            if _segment_intersects_polygon_xy(candidate_xy, target_xy, footprint):
                blocked_by = {
                    "object_name": obstacle.get("object_name"),
                    "category_hint": obstacle.get("category_hint"),
                    "target": target_name,
                }
                break
        if blocked_by is None:
            return {"clear": True, "target": target_name, "reason": None}

    return {
        "clear": False,
        "target": None,
        "reason": "line_of_sight_blocked_by_obstacle",
    }


def _camera_candidate_seeds(room_record, desired_xy, anchor_xy, centroid_xy, margin_m):
    seeds = [
        ("desired", desired_xy),
        ("bbox_inset_desired", _room_bbox_inset_xy(room_record, desired_xy, margin_m)),
        ("anchor", anchor_xy),
        ("bbox_inset_anchor", _room_bbox_inset_xy(room_record, anchor_xy, margin_m)),
    ]
    if centroid_xy is not None:
        centroid_xy = [float(centroid_xy[0]), float(centroid_xy[1])]
        seeds.extend(
            [
                ("centroid", centroid_xy),
                (
                    "centroid_plus_offset",
                    _room_bbox_inset_xy(
                        room_record,
                        [
                            float(centroid_xy[0]) + float(CAMERA_OFFSET_X_M),
                            float(centroid_xy[1]) + float(CAMERA_OFFSET_Y_M),
                        ],
                        margin_m,
                    ),
                ),
            ]
        )
    offsets = [
        (0.0, 0.0),
        (0.6, 0.0),
        (-0.6, 0.0),
        (0.0, 0.6),
        (0.0, -0.6),
        (1.2, 0.0),
        (-1.2, 0.0),
        (0.0, 1.2),
        (0.0, -1.2),
        (0.9, 0.9),
        (0.9, -0.9),
        (-0.9, 0.9),
        (-0.9, -0.9),
        (1.5, 0.0),
        (-1.5, 0.0),
        (0.0, 1.5),
        (0.0, -1.5),
    ]
    expanded = []
    for seed_label, seed_xy in seeds:
        sx = float(seed_xy[0])
        sy = float(seed_xy[1])
        for dx, dy in offsets:
            expanded.append(
                (
                    f"{seed_label}_dx{dx:+.1f}_dy{dy:+.1f}",
                    _room_bbox_inset_xy(room_record, [sx + float(dx), sy + float(dy)], margin_m),
                )
            )
    seen = set()
    unique = []
    for label, xy in expanded:
        key = (round(float(xy[0]), 4), round(float(xy[1]), 4))
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, [float(xy[0]), float(xy[1])]))
    return unique


def _is_geometry_actor_for_camera_probe(actor):
    try:
        class_name = actor.get_class().get_name()
    except Exception:
        class_name = actor.__class__.__name__
    class_name = str(class_name or "")
    geometry_tokens = [
        "StaticMesh",
        "InstancedStaticMesh",
        "HierarchicalInstancedStaticMesh",
        "Brush",
        "GeometryCache",
        "SkeletalMesh",
        "Landscape",
        "Usd",
    ]
    non_geometry_tokens = [
        "Light",
        "Camera",
        "SceneCapture",
        "WorldSettings",
        "PlayerStart",
        "AtmosphericFog",
        "Sky",
    ]
    if any(token in class_name for token in non_geometry_tokens):
        return False
    return any(token in class_name for token in geometry_tokens)


def _should_ignore_actor_for_camera_probe(actor):
    try:
        label = str(actor.get_actor_label() or "")
    except Exception:
        label = ""
    ignore_prefixes = [
        "DEBUG_",
        "SceneCaptureCube",
        "INFINIGEN_",
        "CONVERTED_USD_",
        "BEDLAM_",
    ]
    if any(label.startswith(prefix) for prefix in ignore_prefixes):
        return True
    return not _is_geometry_actor_for_camera_probe(actor)


def _distance_point_to_bounds_cm(point_cm, origin_cm, extent_cm):
    px = float(point_cm.x)
    py = float(point_cm.y)
    pz = float(point_cm.z)
    ox = float(origin_cm.x)
    oy = float(origin_cm.y)
    oz = float(origin_cm.z)
    ex = max(0.0, float(extent_cm.x))
    ey = max(0.0, float(extent_cm.y))
    ez = max(0.0, float(extent_cm.z))

    dx = abs(px - ox) - ex
    dy = abs(py - oy) - ey
    dz = abs(pz - oz) - ez
    if dx <= 0.0 and dy <= 0.0 and dz <= 0.0:
        # Inside the AABB: report negative penetration depth so we can reject hard.
        return -min(ex - abs(px - ox), ey - abs(py - oy), ez - abs(pz - oz))
    return math.sqrt(max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2 + max(dz, 0.0) ** 2)


def _probe_camera_candidate_unreal(camera_xyz_m):
    if not CAMERA_ENABLE_UNREAL_COLLISION_PROBE:
        return {
            "supported": False,
            "collision_free": True,
            "inside_geometry": False,
            "wall_clearance_cm": None,
            "nearest_geometry": None,
            "hit_count": 0,
            "reason": "camera_unreal_collision_probe_disabled",
        }

    try:
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        actors = list(actor_subsystem.get_all_level_actors())
    except Exception as exc:
        return {
            "supported": False,
            "collision_free": True,
            "inside_geometry": False,
            "wall_clearance_cm": None,
            "nearest_geometry": None,
            "hit_count": 0,
            "reason": f"camera_unreal_collision_probe_unavailable:{exc}",
        }

    point_cm = base.transform_infinigen_position_to_bedlam(camera_xyz_m)
    nearest_geometry = None
    inside_hits = []
    considered = 0
    for actor in actors:
        if _should_ignore_actor_for_camera_probe(actor):
            continue
        try:
            bounds = actor.get_actor_bounds(False)
        except Exception:
            try:
                bounds = actor.get_actor_bounds(True)
            except Exception:
                continue
        if not isinstance(bounds, tuple) or len(bounds) < 2:
            continue
        origin_cm, extent_cm = bounds[0], bounds[1]
        if max(float(extent_cm.x), float(extent_cm.y), float(extent_cm.z)) > 100000.0:
            continue
        considered += 1
        distance_cm = float(_distance_point_to_bounds_cm(point_cm, origin_cm, extent_cm))
        row = {
            "label": str(actor.get_actor_label()),
            "class_name": str(actor.get_class().get_name()),
            "distance_cm": distance_cm,
            "origin_cm": {"x": float(origin_cm.x), "y": float(origin_cm.y), "z": float(origin_cm.z)},
            "extent_cm": {"x": float(extent_cm.x), "y": float(extent_cm.y), "z": float(extent_cm.z)},
        }
        if nearest_geometry is None or distance_cm < nearest_geometry["distance_cm"]:
            nearest_geometry = row
        if distance_cm <= 0.0:
            inside_hits.append(row)

    wall_clearance_cm = None if nearest_geometry is None else max(0.0, float(nearest_geometry["distance_cm"]))
    collision_free = (not inside_hits) and (
        wall_clearance_cm is None or wall_clearance_cm >= float(CAMERA_MIN_GEOMETRY_CLEARANCE_CM)
    )
    return {
        "supported": True,
        "collision_free": bool(collision_free),
        "inside_geometry": bool(inside_hits),
        "wall_clearance_cm": wall_clearance_cm,
        "nearest_geometry": nearest_geometry,
        "hit_count": int(len(inside_hits)),
        "considered_actor_count": int(considered),
        "reason": None if collision_free else ("inside_geometry" if inside_hits else "geometry_clearance_below_minimum"),
    }


def _select_safe_room_camera_xy(
    room_record,
    anchor_xy,
    desired_xy,
    margin_m,
    obstacles=None,
    obstacle_clearance_m=0.45,
    humans=None,
    human_clearance_m=1.0,
):
    polygon_xy = base._room_polygon_points(room_record)
    centroid_xy = base._room_centroid_xy(room_record)
    desired_xy = [float(desired_xy[0]), float(desired_xy[1])]
    anchor_xy = [float(anchor_xy[0]), float(anchor_xy[1])]
    if centroid_xy is not None:
        centroid_xy = [float(centroid_xy[0]), float(centroid_xy[1])]
    candidates = _camera_candidate_seeds(
        room_record,
        desired_xy=desired_xy,
        anchor_xy=anchor_xy,
        centroid_xy=centroid_xy,
        margin_m=margin_m,
    )

    scored = []
    for label, xy in candidates:
        inside = _point_in_polygon_xy(xy, polygon_xy) if len(polygon_xy) >= 3 else base._point_in_room_bbox_xy(room_record, xy[0], xy[1])
        boundary_distance = _distance_point_to_room_polygon_xy(xy, room_record)
        desired_distance = math.hypot(float(xy[0]) - float(desired_xy[0]), float(xy[1]) - float(desired_xy[1]))
        nearest_obstacle = _nearest_obstacle_for_point_xy(xy, obstacles or [])
        obstacle_distance = float("inf") if nearest_obstacle is None else float(nearest_obstacle["distance_m"])
        obstacle_free = nearest_obstacle is None or (
            (not nearest_obstacle["contains"]) and obstacle_distance >= float(obstacle_clearance_m)
        )
        nearest_human = _nearest_human_distance_for_point_xy(xy, humans or [])
        human_distance = float("inf") if nearest_human is None else float(nearest_human["distance_m"])
        human_safe = nearest_human is None or human_distance >= float(human_clearance_m)
        room_interior_line_of_sight = _camera_room_line_of_sight_clear(
            xy,
            anchor_xy=anchor_xy,
            centroid_xy=centroid_xy,
            obstacles=obstacles or [],
        )
        wall_clearance_m = float(boundary_distance)
        inside_geometry = bool(nearest_obstacle and nearest_obstacle["contains"])
        collision_free = (
            bool(inside)
            and not inside_geometry
            and wall_clearance_m >= float(CAMERA_MIN_WALL_CLEARANCE_M)
            and bool(obstacle_free)
            and bool(room_interior_line_of_sight["clear"])
        )
        rejection_reasons = []
        if not inside:
            rejection_reasons.append("outside_room_polygon")
        if wall_clearance_m < float(CAMERA_MIN_WALL_CLEARANCE_M):
            rejection_reasons.append("wall_clearance_below_minimum")
        if nearest_obstacle and nearest_obstacle["contains"]:
            rejection_reasons.append("inside_obstacle_geometry")
        elif not obstacle_free:
            rejection_reasons.append("too_close_to_obstacle")
        if not human_safe:
            rejection_reasons.append("too_close_to_human")
        if not room_interior_line_of_sight["clear"]:
            rejection_reasons.append(str(room_interior_line_of_sight["reason"]))
        scored.append(
            {
                "label": label,
                "xy": [float(xy[0]), float(xy[1])],
                "inside": bool(inside),
                "boundary_distance_m": float(boundary_distance),
                "wall_clearance_m": wall_clearance_m,
                "desired_distance_m": float(desired_distance),
                "obstacle_free": bool(obstacle_free),
                "nearest_obstacle": nearest_obstacle,
                "human_safe": bool(human_safe),
                "nearest_human": nearest_human,
                "room_interior_line_of_sight": room_interior_line_of_sight,
                "collision_free": bool(collision_free),
                "inside_geometry": bool(inside_geometry),
                "candidate_rejection_reasons": rejection_reasons,
                "candidate_rejection_reason": None if not rejection_reasons else ",".join(rejection_reasons),
            }
        )

    safe = [
        row for row in scored
        if row["inside"]
        and row["boundary_distance_m"] >= float(margin_m)
        and row["wall_clearance_m"] >= float(CAMERA_MIN_WALL_CLEARANCE_M)
        and row["obstacle_free"]
        and row["human_safe"]
        and row["room_interior_line_of_sight"]["clear"]
        and row["collision_free"]
    ]
    if safe:
        best = min(safe, key=lambda row: (row["desired_distance_m"], -row["boundary_distance_m"], row["label"]))
    else:
        best = None
    return best, scored


def _resolve_miniscene_anchor_camera(miniscene, scene_metadata):
    anchor_xyz_m = _human_anchor_infinigen_m(miniscene)
    selected_room = miniscene.get("room")
    room_record = base._room_record_from_metadata(scene_metadata, selected_room)
    floor_z_m = _room_floor_z_m(scene_metadata, selected_room)
    room_obstacles = _camera_relevant_room_obstacles(
        scene_metadata,
        selected_room,
        camera_height_m=float(CAMERA_HEIGHT_M),
    )
    desired_camera_xy_m = [
        float(anchor_xyz_m[0]) + float(CAMERA_OFFSET_X_M),
        float(anchor_xyz_m[1]) + float(CAMERA_OFFSET_Y_M),
    ]
    selected_camera_xy, camera_xy_candidates = _select_safe_room_camera_xy(
        room_record,
        anchor_xy=[float(anchor_xyz_m[0]), float(anchor_xyz_m[1])],
        desired_xy=desired_camera_xy_m,
        margin_m=float(CAMERA_ROOM_MARGIN_M),
        obstacles=room_obstacles,
        obstacle_clearance_m=float(CAMERA_OBSTACLE_CLEARANCE_M),
        humans=list(miniscene.get("humans", [])),
        human_clearance_m=float(CAMERA_HUMAN_CLEARANCE_M),
    )
    for row in camera_xy_candidates:
        camera_xyz_candidate_m = [
            float(row["xy"][0]),
            float(row["xy"][1]),
            float(floor_z_m) + float(CAMERA_HEIGHT_M),
        ]
        unreal_probe = _probe_camera_candidate_unreal(camera_xyz_candidate_m)
        row["unreal_probe"] = unreal_probe
        if unreal_probe.get("wall_clearance_cm") is not None:
            row["wall_clearance_cm_unreal"] = float(unreal_probe["wall_clearance_cm"])
        if not bool(unreal_probe.get("collision_free", True)):
            reasons = list(row.get("candidate_rejection_reasons") or [])
            reasons.append(str(unreal_probe.get("reason") or "unreal_collision_probe_failed"))
            row["candidate_rejection_reasons"] = reasons
            row["candidate_rejection_reason"] = ",".join(reasons)

    def _camera_candidate_rank(row):
        probe = row.get("unreal_probe") or {}
        return (
            bool(row.get("inside")),
            bool(row.get("obstacle_free")),
            bool(row.get("human_safe")),
            bool(row.get("room_interior_line_of_sight", {}).get("clear", True)),
            bool(probe.get("collision_free", True)),
            -int(probe.get("hit_count", 0)),
            float(probe.get("wall_clearance_cm") or -1.0),
            float(row.get("wall_clearance_m") or -1.0),
            -float(row.get("desired_distance_m") or 0.0),
        )

    valid_camera_xy_candidates = [
        row for row in camera_xy_candidates
        if bool(row.get("inside"))
        and float(row.get("wall_clearance_m") or 0.0) >= float(CAMERA_MIN_WALL_CLEARANCE_M)
        and bool(row.get("obstacle_free"))
        and bool(row.get("human_safe"))
        and bool(row.get("room_interior_line_of_sight", {}).get("clear", True))
        and bool((row.get("unreal_probe") or {}).get("collision_free", True))
    ]
    if valid_camera_xy_candidates:
        selected_camera_xy = max(valid_camera_xy_candidates, key=_camera_candidate_rank)
    elif camera_xy_candidates:
        best_rejected_candidate = max(camera_xy_candidates, key=_camera_candidate_rank)
        raise RuntimeError(
            "No valid static camera candidate survived room/obstacle/human/Unreal-geometry checks. "
            f"selected_room={selected_room!r} miniscene_id={miniscene.get('miniscene_id')!r} "
            f"best_candidate={json.dumps(best_rejected_candidate, default=str)}"
        )
    else:
        raise RuntimeError(
            "No camera candidates were generated for the selected room. "
            f"selected_room={selected_room!r} miniscene_id={miniscene.get('miniscene_id')!r}"
        )
    camera_xyz_m = [
        float(selected_camera_xy["xy"][0]),
        float(selected_camera_xy["xy"][1]),
        float(floor_z_m) + float(CAMERA_HEIGHT_M),
    ]
    camera_location_cm = base.transform_infinigen_position_to_bedlam(camera_xyz_m)
    camera_pose = {
        "x": float(camera_location_cm.x),
        "y": float(camera_location_cm.y),
        "z": float(camera_location_cm.z),
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
    }
    return {
        "camera_pose_cm_deg": camera_pose,
        "source": "forced_miniscene_anchor_camera",
        "selected_camera": None,
        "available_cameras": [],
        "existing_level_camera_bypassed": True,
        "selected_miniscene_id": miniscene.get("miniscene_id"),
        "selected_room": selected_room,
        "human_anchor_infinigen_m": {
            "x": float(anchor_xyz_m[0]),
            "y": float(anchor_xyz_m[1]),
            "z": float(anchor_xyz_m[2]),
        },
        "room_floor_z_m": float(floor_z_m),
        "camera_position_infinigen_m": {
            "x": float(camera_xyz_m[0]),
            "y": float(camera_xyz_m[1]),
            "z": float(camera_xyz_m[2]),
        },
        "camera_position_unreal_cm": {
            "x": float(camera_location_cm.x),
            "y": float(camera_location_cm.y),
            "z": float(camera_location_cm.z),
        },
        "camera_offsets_m": {
            "x": float(CAMERA_OFFSET_X_M),
            "y": float(CAMERA_OFFSET_Y_M),
            "height": float(CAMERA_HEIGHT_M),
        },
        "camera_room_margin_m": float(CAMERA_ROOM_MARGIN_M),
        "camera_obstacle_clearance_m": float(CAMERA_OBSTACLE_CLEARANCE_M),
        "camera_human_clearance_m": float(CAMERA_HUMAN_CLEARANCE_M),
        "camera_min_wall_clearance_m": float(CAMERA_MIN_WALL_CLEARANCE_M),
        "camera_min_geometry_clearance_cm": float(CAMERA_MIN_GEOMETRY_CLEARANCE_CM),
        "desired_camera_position_infinigen_xy_m": {
            "x": float(desired_camera_xy_m[0]),
            "y": float(desired_camera_xy_m[1]),
        },
        "selected_camera_xy_strategy": selected_camera_xy,
        "wall_clearance_cm": 100.0 * float(selected_camera_xy.get("wall_clearance_m") or 0.0),
        "collision_free": bool(selected_camera_xy.get("collision_free")),
        "inside_geometry": bool(selected_camera_xy.get("inside_geometry")),
        "candidate_rejection_reason": selected_camera_xy.get("candidate_rejection_reason"),
        "camera_xy_candidates": camera_xy_candidates,
        "room_obstacle_count_considered": len(room_obstacles),
    }


def _point_in_polygon_xy(point_xy, polygon_xy):
    x = float(point_xy[0])
    y = float(point_xy[1])
    pts = [(float(px), float(py)) for px, py in list(polygon_xy or [])]
    if len(pts) < 3:
        return False
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if abs(yj - yi) > 1e-12 else 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _point_segment_distance_xy(point_xy, a_xy, b_xy):
    px, py = float(point_xy[0]), float(point_xy[1])
    ax, ay = float(a_xy[0]), float(a_xy[1])
    bx, by = float(b_xy[0]), float(b_xy[1])
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        dx = px - ax
        dy = py - ay
        return math.sqrt(dx * dx + dy * dy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    qx = ax + t * abx
    qy = ay + t * aby
    dx = px - qx
    dy = py - qy
    return math.sqrt(dx * dx + dy * dy)


def _distance_point_to_polygon_xy(point_xy, polygon_xy):
    pts = [(float(px), float(py)) for px, py in list(polygon_xy or [])]
    if len(pts) < 3:
        return float("inf")
    best = float("inf")
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        best = min(best, _point_segment_distance_xy(point_xy, a, b))
    return best


def _distance_point_to_room_polygon_xy(point_xy, room_record):
    pts = base._room_polygon_points(room_record)
    if len(pts) < 3:
        centroid = base._room_centroid_xy(room_record)
        if centroid is None:
            return float("inf")
        dx = float(point_xy[0]) - float(centroid[0])
        dy = float(point_xy[1]) - float(centroid[1])
        return math.sqrt(dx * dx + dy * dy)
    best = float("inf")
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        best = min(best, _point_segment_distance_xy(point_xy, a, b))
    return best


def _camera_relevant_room_obstacles(scene_metadata, room_name, camera_height_m):
    relevant = []
    for obstacle in scene_metadata.get("obstacles", []):
        if obstacle.get("room") != room_name:
            continue
        footprint = obstacle.get("footprint_world_xy") or []
        if len(footprint) < 3:
            continue
        z_min = obstacle.get("z_min")
        z_max = obstacle.get("z_max")
        if z_max is not None and float(z_max) < 0.4:
            continue
        if z_min is not None and float(z_min) > float(camera_height_m) + 0.4:
            continue
        relevant.append(obstacle)
    return relevant


def _nearest_obstacle_for_point_xy(point_xy, obstacles):
    best = None
    for obstacle in obstacles:
        footprint = obstacle.get("footprint_world_xy") or []
        if len(footprint) < 3:
            continue
        contains = _point_in_polygon_xy(point_xy, footprint)
        distance = _distance_point_to_polygon_xy(point_xy, footprint)
        row = {
            "object_name": obstacle.get("object_name"),
            "category_hint": obstacle.get("category_hint"),
            "contains": bool(contains),
            "distance_m": float(distance),
        }
        if best is None or row["distance_m"] < best["distance_m"]:
            best = row
    return best


def _nearest_human_distance_for_point_xy(point_xy, humans):
    best = None
    for human in humans or []:
        pos = human.get("position_xyz_m") or [0.0, 0.0, 0.0]
        hx = float(pos[0])
        hy = float(pos[1])
        distance = float(math.hypot(float(point_xy[0]) - hx, float(point_xy[1]) - hy))
        row = {
            "spawn_pose_index": human.get("spawn_pose_index"),
            "motion_id": human.get("motion_id"),
            "distance_m": distance,
        }
        if best is None or row["distance_m"] < best["distance_m"]:
            best = row
    return best


def _nearest_room_for_point_xy(scene_metadata, point_xy):
    rooms = list(scene_metadata.get("rooms", []))
    containing = []
    distance_rows = []
    for room in rooms:
        room_name = room.get("name")
        poly = base._room_polygon_points(room)
        contains = _point_in_polygon_xy(point_xy, poly) if len(poly) >= 3 else False
        distance = _distance_point_to_room_polygon_xy(point_xy, room)
        row = {
            "room": room_name,
            "contains": bool(contains),
            "distance_m": float(distance),
        }
        distance_rows.append(row)
        if contains:
            containing.append(row)
    if containing:
        containing.sort(key=lambda row: (row["distance_m"], str(row["room"])))
        return containing[0]
    distance_rows.sort(key=lambda row: (row["distance_m"], str(row["room"])))
    return distance_rows[0] if distance_rows else None


def _destroy_debug_markers():
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    destroyed = 0
    for actor in actor_subsystem.get_all_level_actors():
        try:
            label = actor.get_actor_label()
        except Exception:
            continue
        if str(label).startswith("DEBUG_"):
            actor_subsystem.destroy_actor(actor)
            destroyed += 1
    return destroyed


def _spawn_debug_marker(label, location_cm):
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor_class = getattr(unreal, "TargetPoint", None) or getattr(unreal, "Actor", None)
    if actor_class is None:
        return {"label": label, "spawned": False, "reason": "no_spawnable_debug_actor_class"}
    location = unreal.Vector(
        float(location_cm.x),
        float(location_cm.y),
        float(location_cm.z) + float(DEBUG_MARKER_Z_OFFSET_CM),
    )
    try:
        actor = actor_subsystem.spawn_actor_from_class(
            actor_class,
            location,
            unreal.Rotator(0.0, 0.0, 0.0),
        )
    except Exception as exc:
        return {"label": label, "spawned": False, "reason": str(exc)}
    actor.set_actor_label(label)
    try:
        actor.set_folder_path("InfinigenBridgeDebug")
    except Exception:
        pass
    return {
        "label": label,
        "spawned": True,
        "actor_name": actor.get_name(),
        "class": actor.get_class().get_name(),
        "location_cm": {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        },
    }


def _selected_room_spatial_sanity(miniscene, scene_metadata, camera_selection):
    selected_room_name = miniscene.get("room")
    room_record = base._room_record_from_metadata(scene_metadata, selected_room_name)
    room_polygon_xy = base._room_polygon_points(room_record) if room_record is not None else []
    room_centroid_xy = base._room_centroid_xy(room_record) if room_record is not None else None
    room_floor_z_m = float((room_record or {}).get("floor_z") or 0.0)
    destroyed_existing_debug_markers = _destroy_debug_markers()

    camera_infinigen_m = camera_selection.get("camera_position_infinigen_m")
    if camera_infinigen_m is None:
        camera_pose = camera_selection.get("camera_pose_cm_deg") or {}
        camera_unreal_cm = unreal.Vector(
            float(camera_pose.get("x", 0.0)),
            float(camera_pose.get("y", 0.0)),
            float(camera_pose.get("z", 0.0)),
        )
        camera_infinigen_m = {
            "x": float(camera_unreal_cm.x) / 100.0,
            "y": -float(camera_unreal_cm.y) / 100.0,
            "z": (float(camera_unreal_cm.z) - float(base.SCENE_FLOOR_OFFSET_CM)) / 100.0,
        }

    camera_xy = [float(camera_infinigen_m["x"]), float(camera_infinigen_m["y"])]
    camera_inside_selected_room = _point_in_polygon_xy(camera_xy, room_polygon_xy) if len(room_polygon_xy) >= 3 else False
    nearest_camera_room = _nearest_room_for_point_xy(scene_metadata, camera_xy)

    human_rows = []
    debug_markers = []
    for idx, human in enumerate(list(miniscene.get("humans", []))):
        pos = human.get("position_xyz_m") or [0.0, 0.0, 0.0]
        xy = [float(pos[0]), float(pos[1])]
        inside = _point_in_polygon_xy(xy, room_polygon_xy) if len(room_polygon_xy) >= 3 else False
        nearest_room = _nearest_room_for_point_xy(scene_metadata, xy)
        unreal_cm = base.transform_infinigen_position_to_bedlam(pos)
        human_rows.append(
            {
                "human_index": int(idx),
                "spawn_pose_index": human.get("spawn_pose_index"),
                "room": human.get("room"),
                "position_infinigen_xy_m": {"x": float(xy[0]), "y": float(xy[1])},
                "position_infinigen_xyz_m": {
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                },
                "inside_selected_room_polygon": bool(inside),
                "nearest_room": nearest_room,
                "position_unreal_cm": {
                    "x": float(unreal_cm.x),
                    "y": float(unreal_cm.y),
                    "z": float(unreal_cm.z),
                },
            }
        )
        debug_markers.append(_spawn_debug_marker(f"DEBUG_HUMAN_{idx}", unreal_cm))

    camera_unreal_cm = base.transform_infinigen_position_to_bedlam(
        [float(camera_infinigen_m["x"]), float(camera_infinigen_m["y"]), float(camera_infinigen_m["z"])]
    )
    debug_markers.append(_spawn_debug_marker("DEBUG_ROOM_CAMERA", camera_unreal_cm))

    room_centroid_unreal_cm = None
    if room_centroid_xy is not None:
        room_centroid_unreal_cm = base.transform_infinigen_position_to_bedlam(
            [float(room_centroid_xy[0]), float(room_centroid_xy[1]), float(room_floor_z_m)]
        )
        debug_markers.append(_spawn_debug_marker("DEBUG_SELECTED_ROOM_CENTROID", room_centroid_unreal_cm))

    return {
        "selected_miniscene_id": miniscene.get("miniscene_id"),
        "selected_room": selected_room_name,
        "selected_room_polygon_infinigen_xy": [[float(x), float(y)] for x, y in room_polygon_xy],
        "selected_room_floor_z_m": float(room_floor_z_m),
        "selected_room_centroid_infinigen_xy_m": (
            {"x": float(room_centroid_xy[0]), "y": float(room_centroid_xy[1])}
            if room_centroid_xy is not None
            else None
        ),
        "camera_position_infinigen_m": {
            "x": float(camera_infinigen_m["x"]),
            "y": float(camera_infinigen_m["y"]),
            "z": float(camera_infinigen_m["z"]),
        },
        "camera_inside_selected_room_polygon": bool(camera_inside_selected_room),
        "camera_nearest_room": nearest_camera_room,
        "camera_position_unreal_cm": {
            "x": float(camera_unreal_cm.x),
            "y": float(camera_unreal_cm.y),
            "z": float(camera_unreal_cm.z),
        },
        "humans": human_rows,
        "destroyed_existing_debug_markers": int(destroyed_existing_debug_markers),
        "debug_markers": debug_markers,
    }


def _lit_rooms_from_report(lighting_report):
    rooms = []
    for light in lighting_report.get("lights", []):
        room = light.get("room")
        if room and room not in rooms:
            rooms.append(room)
    selected_room = lighting_report.get("selected_room")
    nearby_practicals = lighting_report.get("nearby_converted_point_lights") or []
    # Converted USD practicals already illuminate the selected room even when no
    # fallback bridge light was added there.
    if selected_room and nearby_practicals and selected_room not in rooms:
        rooms.append(selected_room)
    return rooms


def _copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _manual_warmup_disabled_report():
    return {
        "enabled": False,
        "mode": "manual_only",
        "reason": "automatic_render_pipeline_warmup_removed_use_manual_warmup_script",
    }


def _frame_record_with_preview_metadata(frame_record):
    record = dict(frame_record)
    pose_json_path = record.get("pose_json_path")
    if pose_json_path and Path(pose_json_path).is_file():
        try:
            pose_payload = json.loads(Path(pose_json_path).read_text(encoding="utf-8"))
        except Exception:
            pose_payload = None
        if isinstance(pose_payload, dict):
            if "preview_status" in pose_payload:
                record["preview_status"] = pose_payload["preview_status"]
            if "preview_png_status" in pose_payload:
                record["preview_png_status"] = pose_payload["preview_png_status"]
    return record


def _run_rgb_preview_mp4_postprocess(sequence_name, png_paths, output_mp4_path, fps):
    png_paths = [str(path) for path in png_paths if path and Path(path).is_file()]
    if not png_paths:
        return {
            "attempted": False,
            "success": False,
            "reason": "no_rgb_png_paths",
        }
    preview_script = Path(base.canonical_validation.mini.PREVIEW_SCRIPT_PATH)
    if not preview_script.is_file():
        return {
            "attempted": False,
            "success": False,
            "reason": "missing_preview_script",
        }
    output_mp4_path = Path(output_mp4_path)
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "python3",
        str(preview_script),
        "mp4",
        str(sequence_name),
        str(output_mp4_path),
        *png_paths,
        "--fps",
        str(int(fps)),
    ]
    t0 = time.perf_counter()
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except Exception as exc:
        return {
            "attempted": True,
            "success": False,
            "reason": str(exc),
            "command": command,
            "duration_seconds": float(time.perf_counter() - t0),
        }
    if completed.returncode != 0:
        return {
            "attempted": True,
            "success": False,
            "reason": completed.stderr.strip() or completed.stdout.strip() or "mp4_generation_failed",
            "command": command,
            "duration_seconds": float(time.perf_counter() - t0),
        }
    return {
        "attempted": True,
        "success": True,
        "mp4_path": str(output_mp4_path),
        "command": command,
        "duration_seconds": float(time.perf_counter() - t0),
    }


def _generate_run_preview_videos(run_root, result, sequence_name):
    range_result = dict(result.get("range_result") or {})
    raw_frame_records = list(range_result.get("frame_records") or [])
    frame_records = [_frame_record_with_preview_metadata(row) for row in raw_frame_records]
    fps = int((result.get("level_sequence") or {}).get("display_fps") or base.canonical_validation.DEFAULT_PREVIEW_FPS)

    rgb_png_paths = [str(row.get("png_path")) for row in frame_records if row.get("png_path") and Path(row.get("png_path")).is_file()]
    preview_dir = Path(run_root) / "preview"
    rgb_output = preview_dir / "preview_rgb.mp4"
    rgb_status = _run_rgb_preview_mp4_postprocess(
        sequence_name=f"{sequence_name}_rgb",
        png_paths=rgb_png_paths,
        output_mp4_path=rgb_output,
        fps=fps,
    )

    return {
        "fps": int(fps),
        "frame_count": int(len(rgb_png_paths)),
        "preview_rgb_mp4": str(rgb_output) if rgb_status.get("success") else None,
        "preview_rgb_mp4_status": rgb_status,
        "preview_rgb_mp4_path": str(rgb_output) if rgb_status.get("success") else None,
        "preview_frame_count": int(len(rgb_png_paths)),
        "preview_fps": int(fps),
        "adaptive_preview_available": False,
        "preview_adaptive_mp4": None,
        "preview_adaptive_mp4_status": {"attempted": False, "success": False, "reason": "adaptive_preview_not_required"},
        "profiling": {
            "mp4_generation_seconds": float(rgb_status.get("duration_seconds", 0.0) or 0.0),
            "preview_generation_seconds": float(rgb_status.get("duration_seconds", 0.0) or 0.0),
        },
    }


def _frame_range_resolution_report(config, miniscene, frame_start, frame_end, result):
    level_sequence = dict(result.get("level_sequence") or {})
    render_resolution = dict(result.get("frame_range_resolution") or {})
    range_result = dict(result.get("range_result") or {})
    frame_records = list(range_result.get("frame_records") or [])
    actual_rendered_frame_start = None
    actual_rendered_frame_end = None
    if frame_records:
        try:
            actual_rendered_frame_start = int(Path(frame_records[0]["pose_json_path"]).stem.split("_")[-1])
        except Exception:
            actual_rendered_frame_start = int(range_result.get("frame_start", frame_start))
        try:
            actual_rendered_frame_end = int(Path(frame_records[-1]["pose_json_path"]).stem.split("_")[-1])
        except Exception:
            actual_rendered_frame_end = int(range_result.get("frame_end", frame_end))
    return {
        "cli_frame_start": None if getattr(config, "frame_start", None) is None else int(config.frame_start),
        "cli_frame_end": None if getattr(config, "frame_end", None) is None else int(config.frame_end),
        "manifest_frame_start": int(miniscene.get("render_options", {}).get("frame_start", frame_start)),
        "manifest_frame_end": int(miniscene.get("render_options", {}).get("frame_end", frame_end)),
        "resolved_frame_start": int(frame_start),
        "resolved_frame_end": int(frame_end),
        "motion_limited_resolution": {
            "requested_frame_start": render_resolution.get("requested_frame_start"),
            "requested_frame_end": render_resolution.get("requested_frame_end"),
            "effective_frame_start": render_resolution.get("effective_frame_start"),
            "effective_frame_end": render_resolution.get("effective_frame_end"),
            "clamped": render_resolution.get("clamped"),
        },
        "sequence_playback_range": {
            "playback_start_frame": level_sequence.get("playback_start_frame"),
            "playback_end_frame": level_sequence.get("playback_end_frame"),
            "timeline_frame_count": level_sequence.get("timeline_frame_count"),
            "sequence_frame_count_requested": level_sequence.get("sequence_frame_count_requested"),
            "natural_timeline_frame_count": level_sequence.get("natural_timeline_frame_count"),
            "natural_timeline_duration_seconds": level_sequence.get("natural_timeline_duration_seconds"),
        },
        "actual_rendered_frame_range": {
            "frame_start": int(range_result.get("frame_start", frame_start)),
            "frame_end": int(range_result.get("frame_end", frame_end)),
            "frame_count": int(range_result.get("frame_count", max(0, int(frame_end) - int(frame_start) + 1))),
            "first_recorded_frame_index": actual_rendered_frame_start,
            "last_recorded_frame_index": actual_rendered_frame_end,
        },
    }


def _geometry_cache_frame_range_report(result, frame_start, frame_end):
    level_sequence = dict(result.get("level_sequence") or {})
    render_resolution = dict(result.get("frame_range_resolution") or {})
    effective_frame_end = int(render_resolution.get("effective_frame_end", frame_end))
    rows = []
    for binding in list(level_sequence.get("body_bindings") or []):
        source_duration_seconds = binding.get("source_duration_seconds")
        source_frame_count = binding.get("source_frame_count")
        section_start_frame = int(binding.get("section_start_frame", 0) or 0)
        motion_supported_end_frame = None
        if source_frame_count is not None:
            remaining = max(0, int(source_frame_count) - int(section_start_frame))
            motion_supported_end_frame = None if remaining <= 0 else int(remaining - 1)
        render_ended_after_source = False
        if motion_supported_end_frame is not None:
            render_ended_after_source = int(effective_frame_end) > int(motion_supported_end_frame)
        rows.append(
            {
                "asset_id": binding.get("asset_id"),
                "actor_label": binding.get("actor_label"),
                "body_slot": binding.get("body_slot"),
                "appearance_id": binding.get("appearance_id"),
                "geometry_cache_asset_path": binding.get("geometry_cache_asset_path"),
                "geometry_cache_duration_seconds": source_duration_seconds,
                "geometry_cache_frame_count": source_frame_count,
                "render_frame_start": int(frame_start),
                "render_frame_end": int(effective_frame_end),
                "section_start_frame": binding.get("section_start_frame"),
                "section_end_frame": binding.get("section_end_frame"),
                "natural_section_end_frame": binding.get("natural_section_end_frame"),
                "section_start_offset_seconds": binding.get("section_start_offset_seconds"),
                "requested_start_offset_seconds": binding.get("requested_start_offset_seconds"),
                "motion_supported_end_frame": motion_supported_end_frame,
                "motion_ended_before_render_ended": bool(render_ended_after_source),
            }
        )
    return rows


def _range_luminance_stats(run_root, range_result):
    frame_records = list(range_result.get("frame_records") or [])
    rgb_paths = [record.get("png_path") for record in frame_records if record.get("png_path")]
    exr_paths = [record.get("exr_path") for record in frame_records if record.get("exr_path")]
    stats_dir = Path(run_root) / "metadata"
    rgb_stats = base._run_image_stats(rgb_paths, stats_dir / "rgb_image_stats.json")
    exr_stats = base._run_image_stats(exr_paths, stats_dir / "exr_image_stats.json")
    return {
        "rgb": rgb_stats,
        "exr": exr_stats,
    }


def _runtime_human_verification(result, frame_index_hint=None):
    spawned_roles = list(result.get("spawned_roles") or [])
    range_result = result.get("range_result") or {}
    frame_records = list(range_result.get("frame_records") or [])
    target_record = None
    if frame_records:
        if frame_index_hint is None:
            target_record = frame_records[0]
        else:
            target_record = next(
                (record for record in frame_records if int(record.get("frame_sample_index", -1)) == int(frame_index_hint)),
                frame_records[0],
            )
    pose_payload = None
    if target_record and target_record.get("pose_json_path"):
        try:
            pose_payload = json.loads(Path(target_record["pose_json_path"]).read_text(encoding="utf-8"))
        except Exception:
            pose_payload = None
    body_evals = list((pose_payload or {}).get("body_evaluations") or [])
    eval_by_label = {row.get("actor_label"): row for row in body_evals if row.get("actor_label")}

    records = []
    for item in spawned_roles:
        actor_label = item.get("actor_label")
        body_eval = eval_by_label.get(actor_label, {})
        appearance_metadata = item.get("appearance_metadata") or {}
        requested = appearance_metadata.get("requested") or {}
        records.append(
            {
                "actor_label": actor_label,
                "body_asset_id": item.get("asset_id"),
                "appearance_id": item.get("appearance_id"),
                "motion_id": item.get("asset_id"),
                "requested_start_frame": item.get("requested_start_frame"),
                "motion_phase_offset_frames": item.get("motion_phase_offset_frames"),
                "actual_evaluated_frame": body_eval.get("sample_frame_index"),
                "geometry_cache_component": body_eval.get("geometry_cache_asset_path"),
                "sequence_binding_id": body_eval.get("binding_guid"),
                "sequence_binding_name": body_eval.get("binding_name"),
                "appearance_requested": requested,
            }
        )
    return records


def _actor_name_from_label(actor_label):
    if not actor_label:
        return None
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in actor_subsystem.get_all_level_actors():
        try:
            if actor.get_actor_label() == actor_label:
                return actor.get_name()
        except Exception:
            continue
    return None


def _frame_body_eval_map(range_result, frame_indices):
    frame_records = list((range_result or {}).get("frame_records") or [])
    payloads = {}
    for frame_index in frame_indices:
        record = next(
            (item for item in frame_records if int(item.get("frame_sample_index", -1)) == int(frame_index)),
            None,
        )
        if record is None or not record.get("pose_json_path"):
            payloads[int(frame_index)] = {}
            continue
        try:
            pose_payload = json.loads(Path(record["pose_json_path"]).read_text(encoding="utf-8"))
        except Exception:
            payloads[int(frame_index)] = {}
            continue
        body_evals = list((pose_payload or {}).get("body_evaluations") or [])
        payloads[int(frame_index)] = {
            row.get("actor_label"): row for row in body_evals if row.get("actor_label")
        }
    return payloads


def _planner_runtime_diagnostic_frame_indices(frame_start, frame_end):
    requested = [0, 30, 60, 90, 120]
    selected = []
    for frame_index in requested:
        if int(frame_start) <= int(frame_index) <= int(frame_end):
            selected.append(int(frame_index))
    if not selected:
        selected = [int(frame_start), int(frame_end)]
    return sorted(set(selected))


def _normalize_motion_root_motion_id(motion_id):
    value = str(motion_id or "")
    suffix = "_root_trajectory"
    return value[:-len(suffix)] if value.endswith(suffix) else value


def _rotate_xy_tuple(local_xy, yaw_rad):
    x = float(local_xy[0])
    y = float(local_xy[1])
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return (c * x - s * y, s * x + c * y)


def _motion_root_json_path(scene_root, motion_id):
    scene_root = Path(scene_root)
    return scene_root / "miniscene_selection_v0" / "_audit_motion_roots" / f"{_normalize_motion_root_motion_id(motion_id)}_root_trajectory.json"


def _load_motion_root_frames(scene_root, motion_id):
    path = _motion_root_json_path(scene_root, motion_id)
    if not path.is_file():
        return None, str(path), f"motion_root_json_missing:{path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, str(path), f"motion_root_json_load_failed:{exc}"
    return list(payload.get("frames") or []), str(path), None


def _planner_root_xy_for_frame(scene_root, human_record, frame_index):
    motion_id = human_record.get("motion_id")
    frames, root_json_path, load_error = _load_motion_root_frames(scene_root, motion_id)
    if frames is None:
        return None, root_json_path, load_error
    if not frames:
        return None, root_json_path, "motion_root_json_empty"
    clamped = max(0, min(int(frame_index), len(frames) - 1))
    frame = frames[clamped]
    spawn_xyz = list(human_record.get("position_xyz_m") or [0.0, 0.0, 0.0])
    spawn_yaw = float(human_record.get("yaw_rad", 0.0) or 0.0)
    local_xy = (float(frame.get("root_x_m", 0.0)), float(frame.get("root_y_m", 0.0)))
    rotated_xy = _rotate_xy_tuple(local_xy, spawn_yaw)
    world_xy = (
        float(spawn_xyz[0]) + float(rotated_xy[0]),
        float(spawn_xyz[1]) + float(rotated_xy[1]),
    )
    return {
        "frame_index": int(clamped),
        "planner_predicted_root_xy_m": [float(world_xy[0]), float(world_xy[1])],
        "planner_root_local_xy_m": [float(local_xy[0]), float(local_xy[1])],
    }, root_json_path, None


def _runtime_root_proxy_xy_from_body_eval(body_eval):
    origin = ((body_eval or {}).get("bounds") or {}).get("origin") or {}
    if "x" not in origin or "y" not in origin:
        return None
    return [float(origin["x"]) / 100.0, -float(origin["y"]) / 100.0]


def _human_identity_for_report(spawned_role):
    asset_id = str(spawned_role.get("asset_id") or "")
    parts = asset_id.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else asset_id


def _safe_int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _planner_runtime_trajectory_consistency(scene_root, selected_miniscene, result, selected_room_record, frame_start, frame_end):
    humans = list(selected_miniscene.get("humans") or [])
    spawned_roles = list(result.get("spawned_roles") or [])
    render_range = result.get("range_result") or {}
    frame_indices = _planner_runtime_diagnostic_frame_indices(frame_start, frame_end)
    eval_maps = _frame_body_eval_map(render_range, frame_indices)
    room_polygon_xy = base._room_polygon_points(selected_room_record) if selected_room_record is not None else []
    rows = []
    mismatch_rows = []
    out_of_room_rows = []
    for body_slot, human_record in enumerate(humans):
        actor_label = None
        appearance_id = None
        geometry_cache_asset_path = None
        body_asset_id = None
        for item in spawned_roles:
            item_body_slot = _safe_int_or_none(item.get("body_slot"))
            if item_body_slot is not None and item_body_slot == int(body_slot):
                actor_label = item.get("actor_label")
                appearance_id = item.get("appearance_id")
                body_asset_id = item.get("asset_id")
                break
        if actor_label is None:
            expected_motion_id = _normalize_motion_root_motion_id(human_record.get("motion_id"))
            for item in spawned_roles:
                item_motion_id = _normalize_motion_root_motion_id(item.get("asset_id"))
                if item_motion_id == expected_motion_id:
                    actor_label = item.get("actor_label")
                    appearance_id = item.get("appearance_id")
                    body_asset_id = item.get("asset_id")
                    break
        frame_rows = []
        errors_cm = []
        out_of_room_frames = []
        root_json_path = None
        planner_load_error = None
        for frame_index in frame_indices:
            eval_row = (eval_maps.get(int(frame_index)) or {}).get(actor_label) or {}
            if geometry_cache_asset_path is None and eval_row.get("geometry_cache_asset_path") is not None:
                geometry_cache_asset_path = eval_row.get("geometry_cache_asset_path")
            planner_row, root_json_path, load_error = _planner_root_xy_for_frame(scene_root, human_record, frame_index)
            if planner_load_error is None and load_error is not None:
                planner_load_error = load_error
            runtime_xy = _runtime_root_proxy_xy_from_body_eval(eval_row)
            error_cm = None
            if planner_row is not None and runtime_xy is not None:
                dx = float(planner_row["planner_predicted_root_xy_m"][0]) - float(runtime_xy[0])
                dy = float(planner_row["planner_predicted_root_xy_m"][1]) - float(runtime_xy[1])
                error_cm = float(math.hypot(dx, dy) * 100.0)
                errors_cm.append(error_cm)
            inside_room = None
            wall_clearance_m = None
            if runtime_xy is not None and len(room_polygon_xy) >= 3:
                inside_room = bool(_point_in_polygon_xy(runtime_xy, room_polygon_xy))
                wall_clearance_m = float(_distance_point_to_room_polygon_xy(runtime_xy, selected_room_record))
                if not inside_room:
                    out_of_room_frames.append(int(frame_index))
            frame_rows.append(
                {
                    "frame_index": int(frame_index),
                    "planner_root_frame_index": None if planner_row is None else planner_row.get("frame_index"),
                    "planner_predicted_root_xy_m": None if planner_row is None else planner_row.get("planner_predicted_root_xy_m"),
                    "runtime_root_proxy_xy_m": runtime_xy,
                    "error_cm": error_cm,
                    "runtime_inside_selected_room_polygon": inside_room,
                    "runtime_wall_clearance_m": wall_clearance_m,
                    "sample_frame_index": eval_row.get("sample_frame_index"),
                    "sample_time_seconds": eval_row.get("sample_time_seconds"),
                    "section_start_frame": eval_row.get("section_start_frame"),
                    "section_start_offset": eval_row.get("section_start_offset"),
                }
            )
        max_error_cm = max(errors_cm) if errors_cm else None
        mean_error_cm = (sum(errors_cm) / len(errors_cm)) if errors_cm else None
        frame_of_max_error = None
        if max_error_cm is not None:
            best = max((row for row in frame_rows if row.get("error_cm") is not None), key=lambda row: row["error_cm"])
            frame_of_max_error = int(best["frame_index"])
        runtime_wall_clearances = [row["runtime_wall_clearance_m"] for row in frame_rows if row.get("runtime_wall_clearance_m") is not None]
        runtime_min_wall_clearance_m = min(runtime_wall_clearances) if runtime_wall_clearances else None
        runtime_room_containment_valid = len(out_of_room_frames) == 0
        mismatch = max_error_cm is not None and float(max_error_cm) > float(MAX_PLANNER_RUNTIME_ROOT_ERROR_CM)
        if mismatch:
            mismatch_rows.append(
                {
                    "body_slot": int(body_slot),
                    "motion_id": _normalize_motion_root_motion_id(human_record.get("motion_id")),
                    "max_error_cm": float(max_error_cm),
                    "frame_of_max_error": frame_of_max_error,
                }
            )
        if out_of_room_frames:
            out_of_room_rows.append(
                {
                    "body_slot": int(body_slot),
                    "motion_id": _normalize_motion_root_motion_id(human_record.get("motion_id")),
                    "runtime_out_of_room_frame_indices": out_of_room_frames,
                }
            )
        rows.append(
            {
                "body_slot": int(body_slot),
                "motion_id": _normalize_motion_root_motion_id(human_record.get("motion_id")),
                "identity_id": _human_identity_for_report({"asset_id": body_asset_id}),
                "appearance_id": appearance_id,
                "actor_label": actor_label,
                "geometry_cache_asset_path": geometry_cache_asset_path,
                "motion_root_json_path": root_json_path,
                "motion_root_load_error": planner_load_error,
                "planner_root_xy_by_frame": frame_rows,
                "runtime_root_proxy_xy_by_frame": [
                    {
                        "frame_index": row["frame_index"],
                        "runtime_root_proxy_xy_m": row["runtime_root_proxy_xy_m"],
                    }
                    for row in frame_rows
                ],
                "error_cm_by_frame": [
                    {
                        "frame_index": row["frame_index"],
                        "error_cm": row["error_cm"],
                    }
                    for row in frame_rows
                ],
                "max_error_cm": max_error_cm,
                "mean_error_cm": mean_error_cm,
                "frame_of_max_error": frame_of_max_error,
                "runtime_out_of_room_frame_indices": out_of_room_frames,
                "runtime_min_wall_clearance_m": runtime_min_wall_clearance_m,
                "runtime_room_containment_valid": bool(runtime_room_containment_valid),
            }
        )
    invalid_reason = None
    if out_of_room_rows:
        invalid_reason = "runtime_root_out_of_room"
    elif mismatch_rows:
        invalid_reason = "planner_runtime_trajectory_mismatch"
    return {
        "diagnostic_frame_indices": frame_indices,
        "max_planner_runtime_root_error_cm": float(MAX_PLANNER_RUNTIME_ROOT_ERROR_CM),
        "humans": rows,
        "planner_runtime_trajectory_mismatch": bool(mismatch_rows),
        "mismatch_rows": mismatch_rows,
        "runtime_out_of_room_detected": bool(out_of_room_rows),
        "out_of_room_rows": out_of_room_rows,
        "invalid_reason": invalid_reason,
    }


def _offline_run_trajectory_consistency_audit(run_root):
    run_root = Path(run_root)
    report_path = run_root / "bridge_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"bridge_report.json not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scene_root = Path(report.get("scene_root") or "")
    selected_miniscene = dict(report.get("selected_miniscene") or {})
    scene_metadata = _load_scene_collision_metadata_json(scene_root / "scene_collision_metadata.json")
    selected_room_name = selected_miniscene.get("room")
    room_record = next((room for room in scene_metadata.get("rooms", []) if room.get("name") == selected_room_name), None)
    body_specs = list(report.get("humans_spawned") or [])
    spawned_roles = [
        {
            "actor_label": row.get("actor_label"),
            "body_slot": row.get("body_slot"),
            "asset_id": row.get("body_asset_id") or row.get("motion_id"),
            "appearance_id": row.get("appearance_id"),
        }
        for row in list(report.get("runtime_human_verification") or [])
        if row.get("actor_label")
        and "_clothing" not in str(row.get("actor_label"))
        and str(row.get("body_asset_id") or row.get("motion_id") or "").count("_") >= 3
    ]
    synthetic_result = {
        "spawned_roles": spawned_roles,
        "range_result": {
            "frame_records": [],
        },
    }
    metadata_dir = run_root / f"frames_{int(report['render_options']['frame_start']):04d}_{int(report['render_options']['frame_end']):04d}" / "metadata"
    if metadata_dir.is_dir():
        for pose_path in sorted(metadata_dir.glob("*_pose.json")):
            try:
                frame_index = int(pose_path.stem.split("_")[-2]) if pose_path.stem.endswith("_pose") else int(pose_path.stem.split("_")[-1])
            except Exception:
                continue
            synthetic_result["range_result"]["frame_records"].append(
                {
                    "frame_sample_index": frame_index,
                    "pose_json_path": str(pose_path),
                }
            )
    diagnostic = _planner_runtime_trajectory_consistency(
        scene_root,
        selected_miniscene,
        synthetic_result,
        room_record,
        int(report["render_options"]["frame_start"]),
        int(report["render_options"]["frame_end"]),
    )
    return {
        "run_root": str(run_root),
        "miniscene_id": report.get("miniscene_id"),
        "room": selected_room_name,
        "planner_runtime_trajectory_consistency": diagnostic,
        "clip_invalid": bool(diagnostic.get("invalid_reason")),
        "invalid_reason": diagnostic.get("invalid_reason"),
    }


def _multi_human_runtime_debug(result, body_specs, render_frame_indices=(12, 60, 120)):
    spawned_roles = list(result.get("spawned_roles") or [])
    appearance_debug_by_body = list(result.get("appearance_debug_by_body") or [])
    render_role_by_label = {
        item.get("actor_label"): item.get("render_role", "body")
        for item in appearance_debug_by_body
        if item.get("actor_label")
    }
    range_result = result.get("range_result") or {}
    eval_maps = _frame_body_eval_map(range_result, render_frame_indices)

    humans = []
    ignored_roles = []
    for item in spawned_roles:
        actor_label = item.get("actor_label")
        render_role = str(render_role_by_label.get(actor_label, "body"))
        if render_role != "body":
            ignored_roles.append(
                {
                    "actor_label": actor_label,
                    "actor_name": _actor_name_from_label(actor_label),
                    "render_role": render_role,
                    "motion_id": item.get("asset_id"),
                    "appearance_id": item.get("appearance_id"),
                }
            )
            continue
        frame_samples = {}
        first_binding = None
        first_geometry_cache_asset = None
        first_section_start_offset = None
        for frame_index in render_frame_indices:
            row = (eval_maps.get(int(frame_index)) or {}).get(actor_label) or {}
            if first_binding is None and row.get("binding_guid") is not None:
                first_binding = row.get("binding_guid")
            if first_geometry_cache_asset is None and row.get("geometry_cache_asset_path") is not None:
                first_geometry_cache_asset = row.get("geometry_cache_asset_path")
            if first_section_start_offset is None and row.get("section_start_offset") is not None:
                first_section_start_offset = row.get("section_start_offset")
            frame_samples[f"frame_{int(frame_index)}"] = {
                "render_frame_index": int(frame_index),
                "timeline_time_seconds": row.get("timeline_time_seconds"),
                "actual_sequence_evaluation_time_seconds": row.get("actual_geometrycache_local_time_seconds"),
                "sample_frame_index": row.get("sample_frame_index"),
                "binding_guid": row.get("binding_guid"),
                "geometry_cache_asset_path": row.get("geometry_cache_asset_path"),
            }

        humans.append(
            {
                "actor_label": actor_label,
                "actor_name": _actor_name_from_label(actor_label),
                "render_role": render_role,
                "body_slot": item.get("body_slot"),
                "motion_id": item.get("asset_id"),
                "appearance_id": item.get("appearance_id"),
                "geometry_cache_asset_path": first_geometry_cache_asset,
                "sequence_binding_id": first_binding,
                "requested_start_frame": item.get("requested_start_frame"),
                "requested_phase_offset_frames": item.get("motion_phase_offset_frames"),
                "actual_section_start_offset_seconds": first_section_start_offset,
                **frame_samples,
            }
        )

    summary = {
        "human_count": len(humans),
        "body_actor_count": len(humans),
        "non_body_render_role_count": len(ignored_roles),
        "ignored_roles": ignored_roles,
        "status": "ok" if len(humans) >= 2 else "not_a_multi_human_render",
        "same_geometry_cache_asset": None,
        "same_motion_id": None,
        "same_binding": None,
        "evaluated_time_difference_human0_human1": {},
    }
    if len(humans) >= 2:
        human0 = humans[0]
        human1 = humans[1]
        summary["same_geometry_cache_asset"] = (
            human0.get("geometry_cache_asset_path") == human1.get("geometry_cache_asset_path")
            if human0.get("geometry_cache_asset_path") is not None and human1.get("geometry_cache_asset_path") is not None
            else None
        )
        summary["same_motion_id"] = human0.get("motion_id") == human1.get("motion_id")
        summary["same_binding"] = (
            human0.get("sequence_binding_id") == human1.get("sequence_binding_id")
            if human0.get("sequence_binding_id") is not None and human1.get("sequence_binding_id") is not None
            else None
        )
        for frame_index in render_frame_indices:
            key = f"frame_{int(frame_index)}"
            t0 = (human0.get(key) or {}).get("actual_sequence_evaluation_time_seconds")
            t1 = (human1.get(key) or {}).get("actual_sequence_evaluation_time_seconds")
            summary["evaluated_time_difference_human0_human1"][key] = (
                None if t0 is None or t1 is None else float(t0) - float(t1)
            )

    return {
        "render_frame_indices": [int(v) for v in render_frame_indices],
        "humans": humans,
        "summary": summary,
    }


def _batch_summary_row(report):
    temporary_lighting = report.get("temporary_lighting") or {}
    luminance_stats = report.get("luminance_stats") or {}
    render_options = report.get("render_options") or {}
    return {
        "scene_root": report.get("scene_root"),
        "miniscene_id": report.get("miniscene_id"),
        "room": report.get("room"),
        "number_of_humans": len(report.get("humans_spawned") or []),
        "motion_ids": report.get("motion_ids") or [],
        "frame_range": {
            "frame_start": render_options.get("frame_start"),
            "frame_end": render_options.get("frame_end"),
        },
        "humans_spawned": report.get("humans_spawned") or [],
        "output_paths": {
            "run_root": report.get("run_root"),
            "erp_output_path": report.get("erp_output_path"),
            "manifest_output_path": report.get("manifest_output_path"),
        },
        "lighting_profile": temporary_lighting.get("lighting_profile"),
        "nearby_practical_lights": temporary_lighting.get("nearby_converted_point_lights") or [],
        "converted_point_light_count": len(
            [item for item in (temporary_lighting.get("converted_usd_prim_light_report") or {}).get("records_converted", []) if item.get("type") == "SphereLight"]
        ),
        "converted_point_light_final_intensities": [
            {
                "label": item.get("label"),
                "final_unreal_intensity": item.get("final_unreal_intensity"),
            }
            for item in (temporary_lighting.get("converted_usd_prim_light_report") or {}).get("records_converted", [])
            if item.get("type") == "SphereLight"
        ],
        "fallback_fill_used": bool(temporary_lighting.get("fallback_fill_added")),
        "artifact_frame_count": int(report.get("artifact_frame_count", 0) or 0),
        "artifact_frame_indices": report.get("artifact_frame_indices") or [],
        "valid_frame_count": report.get("valid_frame_count"),
        "invalid_frame_count": report.get("invalid_frame_count"),
        "clip_valid_for_image_benchmark": report.get("clip_valid_for_image_benchmark"),
        "clip_valid_for_video_benchmark": report.get("clip_valid_for_video_benchmark"),
        "planner_runtime_trajectory_mismatch": bool(report.get("planner_runtime_trajectory_mismatch")),
        "runtime_root_out_of_room": bool(report.get("runtime_root_out_of_room")),
        "runtime_invalid_reason": report.get("runtime_invalid_reason"),
        "rgb_luminance_stats": _luminance_summary(luminance_stats.get("rgb")),
        "exr_luminance_stats": _luminance_summary(luminance_stats.get("exr")),
    }


def _room_distribution(items):
    counts = {}
    for item in items:
        room = item.get("room") if isinstance(item, dict) else None
        if room is None and isinstance(item, dict):
            room = item.get("miniscene", {}).get("room")
        room = str(room or "unknown")
        counts[room] = counts.get(room, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: _room_priority_key(kv[0])))


def _lighting_sweep_cases():
    cases = []
    seen = set()
    for usd_scale in SWEEP_USD_INTENSITY_VALUES:
        key = (int(usd_scale), 0)
        if key not in seen:
            seen.add(key)
            cases.append({"phase": "usd_only", "usd_scale": int(usd_scale), "fill_intensity": 0})
    for fill_intensity in SWEEP_FILL_INTENSITY_VALUES:
        key = (0, int(fill_intensity))
        if key not in seen:
            seen.add(key)
            cases.append({"phase": "fill_only", "usd_scale": 0, "fill_intensity": int(fill_intensity)})
    for fill_intensity in SWEEP_FILL_INTENSITY_VALUES:
        key = (int(SWEEP_COMBINED_USD_SCALE), int(fill_intensity))
        if key not in seen:
            seen.add(key)
            cases.append({"phase": "combined", "usd_scale": int(SWEEP_COMBINED_USD_SCALE), "fill_intensity": int(fill_intensity)})
    return cases


def _write_bridge_manifest(run_id, run_root, body_specs, camera_pose, range_result, miniscene, manifest_path):
    script_path = Path(__file__).resolve()
    script_stat = script_path.stat()
    manifest = {
        "run_id": run_id,
        "sequence_name": SEQUENCE_NAME,
        "created_at_utc": base.canonical_validation._utc_now().isoformat(),
        "asset_ids": [spec["asset_id"] for spec in body_specs],
        "body_specs": body_specs,
        "camera_pose_cm_deg": camera_pose,
        "camera_mode": "existing_level_scene_capture_cube",
        "render_mode": "sequencer_scene_capture_cube",
        "range_expansions": [
            {
                "frame_start": int(range_result["frame_start"]),
                "frame_end": int(range_result["frame_end"]),
                "range_tag": range_result["range_tag"],
            }
        ],
        "capture_fps": base.canonical_validation.DEFAULT_CAPTURE_FPS,
        "preview_fps": base.canonical_validation.DEFAULT_PREVIEW_FPS,
        "use_natural_timing": bool(miniscene.get("render_options", {}).get("use_natural_timing", True)),
        "level_sequence": {
            "asset_path": range_result.get("level_sequence_asset_path"),
            "display_fps": base.canonical_validation.DEFAULT_CAPTURE_FPS,
            "warmup_frames": base.canonical_validation.mini.DEFAULT_LEVEL_SEQUENCE_WARMUP_FRAMES,
            "timeline_frame_count": int(range_result["frame_end"]) + 1,
        },
        "selected_miniscene": {
            "manifest_path": str(manifest_path),
            "miniscene_id": miniscene.get("miniscene_id"),
            "room": miniscene.get("room"),
            "scene_type": miniscene.get("scene_type"),
            "humans": miniscene.get("humans", []),
            "validation_summary": miniscene.get("validation_summary", {}),
            "diversity_tags": miniscene.get("diversity_tags", []),
        },
        "infinigen_bridge": {
            "appearance": {
                "enable_body_texture": bool(base.ENABLE_BODY_TEXTURE),
                "enable_clothing": bool(base.ENABLE_CLOTHING),
                "enable_hair": bool(base.ENABLE_HAIR),
                "enable_shoes": bool(base.ENABLE_SHOES),
                "selected_fields": base.BRIDGE_APPEARANCE_FIELDS,
            },
            "t_infinigen_to_bedlam": {
                "position_mapping": "[x, y, z] -> [100x, -100y, 100z + SCENE_FLOOR_OFFSET_CM]",
                "yaw_mapping": "yaw -> -degrees(yaw)",
                "scene_root_offset_cm": base._vector_to_dict(base.SCENE_ROOT_OFFSET_CM),
                "scene_root_yaw_deg": float(base.SCENE_ROOT_YAW_DEG),
                "scene_floor_offset_cm": float(base.SCENE_FLOOR_OFFSET_CM),
            },
        },
        "script_path": str(script_path),
        "script_mtime_utc": base.canonical_validation.datetime.fromtimestamp(
            script_stat.st_mtime, tz=base.canonical_validation.timezone.utc
        ).isoformat(),
        "git_head": base.canonical_validation._safe_git_head(),
        "range_results": [range_result],
    }
    out_path = run_root / "manifest.json"
    base._write_json(out_path, manifest)
    base._write_json(
        base.BRIDGE_LATEST_RUN_JSON,
        {
            "run_id": run_id,
            "run_root": str(run_root),
            "manifest_path": str(out_path),
            "created_at_utc": manifest["created_at_utc"],
            "source": "render_selected_infinigen_bedlam_erp",
            "selected_miniscene": {
                "manifest_path": str(manifest_path),
                "miniscene_id": miniscene.get("miniscene_id"),
                "room": miniscene.get("room"),
            },
        },
    )
    return manifest, out_path


def _render_lighting_intensity_sweep(
    config,
    source_manifest,
    miniscene,
    selected_index,
    selection_info,
    body_specs,
    camera_selection,
    camera_pose,
    selected_room,
    human_rooms,
):
    frame_start = int(miniscene.get("render_options", {}).get("frame_start", 12))
    sweep_frame_index = int(frame_start if SWEEP_FRAME_INDEX is None else SWEEP_FRAME_INDEX)
    sweep_root = base._ensure_dir(
        base.BRIDGE_ROOT / "lighting_sweeps" / base.canonical_validation._utc_now().strftime("%Y%m%dT%H%M%SZ")
    )
    cases = _lighting_sweep_cases()
    original_usd_scale = base.USD_PRIM_LIGHT_INTENSITY_SCALE
    original_fill_override = base.LOW_FILL_POINT_INTENSITY_OVERRIDE
    original_capture_profile = base.CAPTURE_LIGHTING_PROFILE
    case_reports = []
    try:
        for case in cases:
            usd_scale = int(case["usd_scale"])
            fill_intensity = int(case["fill_intensity"])
            stem = f"usd_{usd_scale:05d}_fill_{fill_intensity:04d}_frame_{sweep_frame_index:04d}"
            run_id = f"sweep_{stem}"
            run_root = base._ensure_dir(sweep_root / stem)

            base.USD_PRIM_LIGHT_INTENSITY_SCALE = float(usd_scale)
            base.LOW_FILL_POINT_INTENSITY_OVERRIDE = float(fill_intensity)
            base.CAPTURE_LIGHTING_PROFILE = "erp_capture"

            render_target_report = base._ensure_texture_target_on_scene_capture_cube()
            exposure_report = base._configure_scene_capture_exposure()
            scene_capture_pose_report = base._place_scene_capture_cube(camera_pose)
            lighting_report = base._setup_temp_indoor_lighting(
                _anchor_location(body_specs),
                room_name=selected_room,
                scene_metadata=base.load_scene_metadata(),
                neighboring_room_names=human_rooms,
                lighting_profile=base.CAPTURE_LIGHTING_PROFILE,
                reference_location_cm=camera_pose,
            )

            log_info(
                f"LIGHTING_SWEEP case phase={case['phase']} usd_scale={usd_scale} fill_intensity={fill_intensity} frame={sweep_frame_index}"
            )
            result = base.canonical_validation.render_full_appearance_sequence_to_root(
                run_id=run_id,
                run_root=run_root,
                sequence_name=f"{SEQUENCE_NAME}_{stem}",
                frame_start=sweep_frame_index,
                frame_end=sweep_frame_index,
                body_specs=body_specs,
                camera_pose=camera_pose,
            )
            range_result = dict(result["range_result"])
            frame_record = _frame_record_with_preview_metadata((range_result.get("frame_records") or [])[0])

            exr_dst = sweep_root / f"{stem}.exr"
            adaptive_preview_dst = sweep_root / f"{stem}_adaptive_preview.png"
            fixed_preview_dst = sweep_root / f"{stem}_fixed_preview.png"
            copied_exr = _copy_if_exists(frame_record.get("exr_path"), exr_dst)
            preview_status = frame_record.get("preview_status") or frame_record.get("preview_png_status") or {}
            copied_adaptive = _copy_if_exists(preview_status.get("preview_png_path") or frame_record.get("png_path"), adaptive_preview_dst)
            copied_fixed = _copy_if_exists(preview_status.get("fixed_exposure_preview_png_path"), fixed_preview_dst)

            exr_stats = base._run_image_stats(
                [copied_exr] if copied_exr else [],
                sweep_root / f"{stem}_exr_stats.json",
            )
            case_reports.append(
                {
                    "phase": case["phase"],
                    "usd_prim_light_intensity_scale": usd_scale,
                    "low_fill_point_intensity": fill_intensity,
                    "frame_index": sweep_frame_index,
                    "stem": stem,
                    "run_id": run_id,
                    "run_root": str(run_root),
                    "scene_capture_cube": render_target_report,
                    "scene_capture_exposure": exposure_report,
                    "scene_capture_cube_pose": scene_capture_pose_report,
                    "temporary_lighting": lighting_report,
                    "output_exr_path": copied_exr,
                    "output_adaptive_preview_png_path": copied_adaptive,
                    "output_fixed_preview_png_path": copied_fixed,
                    "preview_status": preview_status,
                    "exr_luminance_stats": (exr_stats or {}).get("data", {}).get("images", [None])[0],
                    "exr_stats_report": exr_stats,
                }
            )
    finally:
        base.USD_PRIM_LIGHT_INTENSITY_SCALE = original_usd_scale
        base.LOW_FILL_POINT_INTENSITY_OVERRIDE = original_fill_override
        base.CAPTURE_LIGHTING_PROFILE = original_capture_profile
        base._setup_temp_indoor_lighting(
            _anchor_location(body_specs),
            room_name=selected_room,
            scene_metadata=base.load_scene_metadata(),
            neighboring_room_names=human_rooms,
            lighting_profile=base.LIGHTING_PROFILE,
            reference_location_cm=camera_pose,
        )

    report = {
        "mode": "lighting_intensity_sweep",
        "manifest_path": str(config.manifest_path),
        "requested_miniscene_index": int(config.miniscene_index),
        "selected_miniscene_index": int(selected_index),
        "selection_info": selection_info,
        "miniscene_id": miniscene.get("miniscene_id"),
        "room": miniscene.get("room"),
        "camera_pose_cm_deg": camera_pose,
        "sweep_frame_index": int(sweep_frame_index),
        "cases": case_reports,
    }
    report_path = sweep_root / "lighting_intensity_sweep_report.json"
    base._write_json(report_path, report)
    log_info(f"LIGHTING_SWEEP_REPORT {json.dumps({'report_path': str(report_path), 'case_count': len(case_reports)})}")
    return report


def _render_single_miniscene_report(
    config,
    source_manifest,
    miniscene,
    selected_index,
    selection_info,
):
    render_profile_config = _apply_render_output_profile(config)
    _CLIP_ASSET_AUDIT_STATE["clip_index"] = int(_CLIP_ASSET_AUDIT_STATE.get("clip_index", 0)) + 1
    requested_render_output_profile = str(getattr(config, "render_output_profile", RENDER_OUTPUT_PROFILE))
    effective_render_output_profile = str(render_profile_config.get("effective_render_output_profile") or render_profile_config.get("render_output_profile"))
    if requested_render_output_profile != effective_render_output_profile:
        message = (
            "Requested render output profile does not match effective render output profile. "
            f"requested={requested_render_output_profile!r} effective={effective_render_output_profile!r}"
        )
        if bool(getattr(config, "batch", False)) and bool(getattr(config, "batch_inherited_render_output_profile", False)):
            raise RuntimeError(message)
        log_info("RENDER_OUTPUT_PROFILE_WARNING " + message)
    log_info(
        "CLIP_RENDER_OUTPUT_PROFILE "
        + json.dumps(
            {
                "requested_render_output_profile": requested_render_output_profile,
                "effective_render_output_profile": effective_render_output_profile,
                "profile_source": render_profile_config.get("profile_source"),
                "miniscene_id": miniscene.get("miniscene_id"),
            },
            indent=2,
        )
    )
    miniscene = json.loads(json.dumps(miniscene))
    scene_metadata = base.load_scene_metadata()
    humans = [dict(human) for human in list(miniscene.get("humans", []))]
    if not humans:
        raise RuntimeError(f"Mini-scene {miniscene.get('miniscene_id')} contains no humans")
    human_count = len(humans)
    for human in humans:
        human["_human_count"] = human_count

    body_specs = [_build_body_spec(human, i) for i, human in enumerate(humans)]
    anchor_location = _anchor_location(body_specs)
    anchor_yaw = float(body_specs[0]["yaw"])

    if USE_MINISCENE_ANCHOR_CAMERA:
        camera_selection = _resolve_miniscene_anchor_camera(miniscene, scene_metadata)
    else:
        camera_selection = base._resolve_capture_camera_pose(
            anchor_location,
            anchor_yaw,
            room_name=miniscene.get("room"),
        )
    camera_pose = camera_selection["camera_pose_cm_deg"]
    render_target_report = base._ensure_texture_target_on_scene_capture_cube()
    exposure_report = base._configure_scene_capture_exposure()
    scene_capture_pose_report = base._place_scene_capture_cube(camera_pose)
    selected_room = miniscene.get("room")
    selected_room_record = next(
        (room for room in scene_metadata.get("rooms", []) if room.get("name") == selected_room),
        None,
    )
    human_rooms = _human_room_names(miniscene)
    spatial_sanity = _selected_room_spatial_sanity(
        miniscene=miniscene,
        scene_metadata=scene_metadata,
        camera_selection=camera_selection,
    )
    viewport_lighting_report = base._setup_temp_indoor_lighting(
        anchor_location,
        room_name=selected_room,
        scene_metadata=scene_metadata,
        neighboring_room_names=human_rooms,
        lighting_profile=base.LIGHTING_PROFILE,
        reference_location_cm=camera_pose,
    )
    capture_lighting_report = base._setup_temp_indoor_lighting(
        anchor_location,
        room_name=selected_room,
        scene_metadata=scene_metadata,
        neighboring_room_names=human_rooms,
        lighting_profile=base.CAPTURE_LIGHTING_PROFILE,
        reference_location_cm=camera_pose,
    )
    lit_rooms = _lit_rooms_from_report(capture_lighting_report)

    frame_start = int(miniscene.get("render_options", {}).get("frame_start", 12))
    frame_end = int(miniscene.get("render_options", {}).get("frame_end", 18))
    if getattr(config, "frame_start", None) is not None:
        frame_start = int(config.frame_start)
    if getattr(config, "frame_end", None) is not None:
        frame_end = int(config.frame_end)
    miniscene.setdefault("render_options", {})
    miniscene["render_options"]["frame_start"] = int(frame_start)
    miniscene["render_options"]["frame_end"] = int(frame_end)
    use_natural_timing = bool(
        miniscene.get("render_options", {}).get("use_natural_timing", True)
    )

    run_id = base.canonical_validation._make_run_id()
    run_root = base._ensure_dir(base.BRIDGE_RUNS_DIR / run_id)
    appearance_warmup_report = _manual_warmup_disabled_report()
    memory_checkpoints = []
    cleanup_summary = None
    appearance_mode = str(getattr(config, "bedlam_debug_appearance_mode", BEDLAM_DEBUG_APPEARANCE_MODE))
    expected_clip_asset_references = _collect_expected_clip_asset_references(body_specs, appearance_mode)
    before_clip_asset_groups = _group_asset_paths_by_category(
        expected_clip_asset_references,
        loaded_key="runtime_loaded_before_cleanup",
    )

    log_info(
        f"Rendering miniscene_id={miniscene.get('miniscene_id')} requested_index={config.miniscene_index} selected_index={selected_index} "
        f"room={miniscene.get('room')} humans={len(humans)} frames={frame_start}..{frame_end}"
    )
    log_info(
        "SELECTED_MINISCENE "
        + json.dumps(
            _miniscene_brief(miniscene, index=selected_index),
            indent=2,
        )
    )
    log_info(
        "FINAL_FRAME_RANGE "
        + json.dumps(
            {
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
            },
            indent=2,
        )
    )
    log_info(
        f"T_infinigen_to_bedlam: scene_root_offset_cm=({base.SCENE_ROOT_OFFSET_CM.x:.2f}, {base.SCENE_ROOT_OFFSET_CM.y:.2f}, {base.SCENE_ROOT_OFFSET_CM.z:.2f}) "
        f"scene_root_yaw_deg={base.SCENE_ROOT_YAW_DEG:.2f} scene_floor_offset_cm={base.SCENE_FLOOR_OFFSET_CM:.2f}"
    )
    log_info(f"Selected miniscene: {json.dumps(miniscene)}")
    log_info(f"Body specs: {json.dumps([_body_spec_report(spec, human) for spec, human in zip(body_specs, humans)])}")
    log_info(f"Available level cameras: {json.dumps(camera_selection['available_cameras'])}")
    log_info(
        f"Selected capture camera source={camera_selection['source']} selected={json.dumps(camera_selection['selected_camera'])}"
    )
    if camera_selection.get("source") == "forced_miniscene_anchor_camera":
        log_info(
            f"Forced miniscene-anchor camera: {json.dumps({'selected_miniscene_id': camera_selection.get('selected_miniscene_id'), 'selected_room': camera_selection.get('selected_room'), 'human_anchor_infinigen_m': camera_selection.get('human_anchor_infinigen_m'), 'camera_position_infinigen_m': camera_selection.get('camera_position_infinigen_m'), 'camera_position_unreal_cm': camera_selection.get('camera_position_unreal_cm'), 'existing_level_camera_bypassed': camera_selection.get('existing_level_camera_bypassed')})}"
        )
    log_info(f"ROOM_CAMERA_SPATIAL_SANITY {json.dumps(spatial_sanity)}")
    log_info(f"Final SceneCaptureCube pose to render from: {json.dumps(camera_pose)}")
    log_info(f"Placed SceneCaptureCube actor: {json.dumps(scene_capture_pose_report)}")
    log_info(f"SceneCaptureCube render target: {json.dumps(render_target_report)}")
    log_info(f"SceneCaptureCube exposure: {json.dumps(exposure_report)}")
    log_info(f"Lighting target rooms: selected_room={selected_room} human_rooms={human_rooms} lit_rooms={lit_rooms}")
    log_info(f"Viewport lighting profile report: {json.dumps(viewport_lighting_report)}")
    if capture_lighting_report.get("enabled"):
        log_info(f"Capture lighting enabled: {json.dumps(capture_lighting_report)}")
    else:
        log_info(f"Capture lighting / environment report: {json.dumps(capture_lighting_report)}")
    log_info(f"Appearance warmup report: {json.dumps(appearance_warmup_report)}")

    if LIGHTING_INTENSITY_SWEEP:
        return _render_lighting_intensity_sweep(
            config=config,
            source_manifest=source_manifest,
            miniscene=miniscene,
            selected_index=selected_index,
            selection_info=selection_info,
            body_specs=body_specs,
            camera_selection=camera_selection,
            camera_pose=camera_pose,
            selected_room=selected_room,
            human_rooms=human_rooms,
        )

    base.canonical_validation.mini.ENABLE_CUBEMAP_FACE_DIAGNOSTICS = bool(ENABLE_CUBEMAP_FACE_DIAGNOSTICS)
    base.canonical_validation.mini.DEFAULT_CUBEMAP_FACE_DIAGNOSTIC_EXPORT_MODES = tuple(
        CUBEMAP_FACE_DIAGNOSTIC_EXPORT_MODES
    )
    base.canonical_validation.mini.DEFAULT_CUBEMAP_FACE_DIAGNOSTIC_WAIT_SECONDS = float(
        CUBEMAP_FACE_DIAGNOSTIC_WAIT_SECONDS
    )
    base.canonical_validation.mini.ENABLE_CUBEMAP_FACE_ANOMALY_RETRY = bool(
        ENABLE_CUBEMAP_FACE_ANOMALY_RETRY
    )
    base.canonical_validation.mini.DEFAULT_CUBEMAP_FACE_ANOMALY_RETRY_MODE = str(
        CUBEMAP_FACE_ANOMALY_RETRY_MODE
    )
    base.canonical_validation.mini.DEFAULT_CUBEMAP_FACE_ANOMALY_RETRY_MAX_ATTEMPTS = int(
        CUBEMAP_FACE_ANOMALY_RETRY_MAX_ATTEMPTS
    )
    base.canonical_validation.mini.EXPORT_FACE_DIAGNOSTICS_ONLY_ON_RETRY = bool(
        EXPORT_FACE_DIAGNOSTICS_ONLY_ON_RETRY
    )

    memory_checkpoints.append(
        _capture_memory_checkpoint(
            "before_clip",
            run_root,
            extra={
                "miniscene_id": miniscene.get("miniscene_id"),
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "clip_index": int(_CLIP_ASSET_AUDIT_STATE.get("clip_index", 0)),
                "asset_residency_groups": before_clip_asset_groups,
            },
        )
    )
    render_t0 = base.canonical_validation.time.perf_counter()
    result = None
    render_sequence_seconds = None
    try:
        result = base.canonical_validation.render_full_appearance_sequence_to_root(
            run_id=run_id,
            run_root=run_root,
            sequence_name=SEQUENCE_NAME,
            frame_start=frame_start,
            frame_end=frame_end,
            body_specs=body_specs,
            camera_pose=camera_pose,
            pause_after_spawn_before_render=bool(getattr(config, "pause_after_spawn_before_render", False)),
            pause_after_spawn_seconds=float(getattr(config, "pause_after_spawn_seconds", 0.0) or 0.0),
            render_warmup_frame_count=int(getattr(config, "render_warmup_frame_count", 0) or 0),
            discard_warmup_frames=bool(getattr(config, "discard_warmup_frames", False)),
            probe_frame_before_render=bool(getattr(config, "probe_frame_before_render", False)),
            reject_beige_probe=bool(getattr(config, "reject_beige_probe", False)),
            bedlam_debug_appearance_mode=str(
                getattr(config, "bedlam_debug_appearance_mode", BEDLAM_DEBUG_APPEARANCE_MODE)
            ),
        )
        render_sequence_seconds = float(base.canonical_validation.time.perf_counter() - render_t0)
    except Exception:
        render_sequence_seconds = float(base.canonical_validation.time.perf_counter() - render_t0)
        memory_checkpoints.append(
            _capture_memory_checkpoint(
                "render_exception_before_cleanup",
                run_root,
                extra={
                    "miniscene_id": miniscene.get("miniscene_id"),
                    "render_sequence_seconds": render_sequence_seconds,
                    "clip_index": int(_CLIP_ASSET_AUDIT_STATE.get("clip_index", 0)),
                },
            )
        )
        try:
            cleanup_summary = _clip_cleanup_summary(
                result,
                run_root,
                clip_asset_references=_collect_clip_asset_references(result),
            )
        except Exception as cleanup_exc:
            log_info(f"CLIP_CLEANUP_EXCEPTION {cleanup_exc}")
        memory_checkpoints.append(
            _capture_memory_checkpoint(
                "after_cleanup_exception_path",
                run_root,
                extra={
                    "miniscene_id": miniscene.get("miniscene_id"),
                    "clip_index": int(_CLIP_ASSET_AUDIT_STATE.get("clip_index", 0)),
                },
            )
        )
        raise
    if result.get("spawn_material_diagnostics") is not None:
        log_info("BEDLAM_SPAWN_MATERIAL_DIAGNOSTICS " + json.dumps(result.get("spawn_material_diagnostics"), indent=2))
    if bool(result.get("paused_before_render")):
        pause_report = {
            "mode": "pause_after_spawn_before_render",
            "run_id": run_id,
            "run_root": str(run_root),
            "scene_root": str(getattr(config, "scene_root", "")),
            "manifest_path": str(config.manifest_path),
            "miniscene_id": miniscene.get("miniscene_id"),
            "room": miniscene.get("room"),
            "selected_miniscene_index": int(selected_index),
            "camera_selection": camera_selection,
            "camera_pose_cm_deg": camera_pose,
            "humans_spawned": [_body_spec_report(spec, human) for spec, human in zip(body_specs, humans)],
            "appearance_debug_by_body": result.get("appearance_debug_by_body"),
            "spawn_material_diagnostics": result.get("spawn_material_diagnostics"),
            "level_sequence": result.get("level_sequence"),
            "frame_range_resolution": result.get("frame_range_resolution"),
            "paused_before_render": True,
        }
        pause_report_path = run_root / "spawn_pause_report.json"
        base._write_json(pause_report_path, pause_report)
        log_info("PAUSE_AFTER_SPAWN_READY " + json.dumps({"pause_report_path": str(pause_report_path), "run_root": str(run_root)}, indent=2))
        return pause_report
    memory_checkpoints.append(
        _capture_memory_checkpoint(
            "after_clip_before_cleanup",
            run_root,
            extra={
                "miniscene_id": miniscene.get("miniscene_id"),
                "render_sequence_seconds": render_sequence_seconds,
                "clip_index": int(_CLIP_ASSET_AUDIT_STATE.get("clip_index", 0)),
            },
        )
    )
    clip_asset_references = _collect_clip_asset_references(result)
    after_clip_asset_groups = _group_asset_paths_by_category(
        clip_asset_references,
        loaded_key="runtime_loaded_before_cleanup",
    )
    range_result = dict(result["range_result"])
    range_result["level_sequence_asset_path"] = result["level_sequence"]["asset_path"]
    frame_range_resolution = _frame_range_resolution_report(
        config=config,
        miniscene=miniscene,
        frame_start=frame_start,
        frame_end=frame_end,
        result=result,
    )
    geometry_cache_frame_ranges = _geometry_cache_frame_range_report(
        result=result,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    restored_viewport_lighting_report = base._setup_temp_indoor_lighting(
        anchor_location,
        room_name=selected_room,
        scene_metadata=base.load_scene_metadata(),
        neighboring_room_names=human_rooms,
        lighting_profile=base.LIGHTING_PROFILE,
        reference_location_cm=camera_pose,
    )
    capture_pipeline_diagnostics = None
    if CAPTURE_PIPELINE_DIAGNOSTICS:
        frame_records = list(range_result.get("frame_records") or [])
        diagnostic_frame_record = None
        if frame_records:
            if CAPTURE_DIAGNOSTIC_FRAME_INDEX is None:
                diagnostic_frame_record = frame_records[-1]
            else:
                diagnostic_frame_record = next(
                    (
                        record
                        for record in frame_records
                        if int(record.get("frame_sample_index", -1)) == int(CAPTURE_DIAGNOSTIC_FRAME_INDEX)
                    ),
                    frame_records[-1],
                )
        if diagnostic_frame_record is not None:
            capture_pipeline_diagnostics = base._collect_capture_pipeline_diagnostic(
                run_root=run_root,
                frame_record=diagnostic_frame_record,
                reference_view_width=CAPTURE_DIAGNOSTIC_REFERENCE_VIEW_WIDTH,
                reference_view_height=CAPTURE_DIAGNOSTIC_REFERENCE_VIEW_HEIGHT,
            )
            log_info(
                f"Capture pipeline diagnostics written under: {capture_pipeline_diagnostics.get('diagnostic_root')}"
            )

    manifest, output_manifest_path = _write_bridge_manifest(
        run_id=run_id,
        run_root=run_root,
        body_specs=body_specs,
        camera_pose=camera_pose,
        range_result=range_result,
        miniscene=miniscene,
        manifest_path=config.manifest_path,
    )

    t0 = base.canonical_validation.time.perf_counter()
    gt_export = base._run_benchmark_export(range_result["range_tag"])
    gt_export_seconds = float(base.canonical_validation.time.perf_counter() - t0)
    t0 = base.canonical_validation.time.perf_counter()
    luminance_stats = _range_luminance_stats(run_root, range_result) if bool(base.canonical_validation.ENABLE_PER_FRAME_STATS) else {}
    report_stats_seconds = float(base.canonical_validation.time.perf_counter() - t0)
    t0 = base.canonical_validation.time.perf_counter()
    runtime_human_verification = _runtime_human_verification(result, frame_index_hint=frame_start)
    runtime_human_verification_seconds = float(base.canonical_validation.time.perf_counter() - t0)
    t0 = base.canonical_validation.time.perf_counter()
    multi_human_runtime_debug = _multi_human_runtime_debug(result, body_specs, render_frame_indices=(12, 60, 120))
    multi_human_runtime_debug_seconds = float(base.canonical_validation.time.perf_counter() - t0)
    t0 = base.canonical_validation.time.perf_counter()
    preview_videos = _generate_run_preview_videos(run_root, result, SEQUENCE_NAME)
    mp4_generation_seconds = float(base.canonical_validation.time.perf_counter() - t0)
    log_info("FRAME_RANGE_RESOLUTION " + json.dumps(frame_range_resolution, indent=2))
    log_info("GEOMETRY_CACHE_FRAME_RANGES " + json.dumps(geometry_cache_frame_ranges, indent=2))

    render_profiling = {
        "render_output_profile": render_profile_config,
        "sequence_render_seconds": render_sequence_seconds,
        "canonical_range_profiling": dict((range_result.get("profiling") or {})),
        "gt_export_seconds": gt_export_seconds,
        "report_stat_collection_seconds": report_stats_seconds,
        "runtime_human_verification_seconds": runtime_human_verification_seconds,
        "multi_human_runtime_debug_seconds": multi_human_runtime_debug_seconds,
        "mp4_generation_seconds": mp4_generation_seconds,
    }
    aggregate_profile = dict(((range_result.get("profiling") or {}).get("aggregate") or {}))
    render_diagnostics = {
        "capture_source_modes_exported": aggregate_profile.get("capture_source_modes_exported"),
        "longlat_export_pass_count": aggregate_profile.get("longlat_export_pass_count"),
        "direct_longlat_png_export_used": aggregate_profile.get("direct_longlat_png_export_used"),
        "rgb_png_export_backend": aggregate_profile.get("rgb_png_export_backend"),
        "intermediate_ldr_exr_written": aggregate_profile.get("intermediate_ldr_exr_written"),
        "python_preview_subprocess_used": aggregate_profile.get("python_preview_subprocess_used"),
        "sequence_adaptive_requires_intermediate_source": aggregate_profile.get("sequence_adaptive_requires_intermediate_source"),
        "capture_source_initialized_once": aggregate_profile.get("capture_source_initialized_once"),
        "per_frame_capture_source_switch_count": aggregate_profile.get("per_frame_capture_source_switch_count"),
        "cubemap_face_diagnostics_enabled": aggregate_profile.get("cubemap_face_diagnostics_enabled"),
        "cubemap_face_diagnostic_flagged_frame_count": aggregate_profile.get("cubemap_face_diagnostic_flagged_frame_count"),
        "cubemap_face_retry_attempted_frame_count": aggregate_profile.get("cubemap_face_retry_attempted_frame_count"),
        "cubemap_face_retry_resolved_frame_count": aggregate_profile.get("cubemap_face_retry_resolved_frame_count"),
        "cubemap_face_retry_unresolved_frame_count": aggregate_profile.get("cubemap_face_retry_unresolved_frame_count"),
        "cubemap_face_retry_face_diagnostics_exported_frame_count": aggregate_profile.get("cubemap_face_retry_face_diagnostics_exported_frame_count"),
    }
    artifact_summary = dict(range_result.get("artifact_summary") or {})
    runtime_scene_root = Path(getattr(config, "scene_root", "")).expanduser().resolve()
    planner_runtime_trajectory_consistency = _planner_runtime_trajectory_consistency(
        scene_root=runtime_scene_root,
        selected_miniscene=miniscene,
        result=result,
        selected_room_record=selected_room_record,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    runtime_invalid_reason = planner_runtime_trajectory_consistency.get("invalid_reason")
    planner_runtime_trajectory_mismatch = bool(
        planner_runtime_trajectory_consistency.get("planner_runtime_trajectory_mismatch")
    )
    runtime_root_out_of_room = bool(planner_runtime_trajectory_consistency.get("runtime_out_of_room_detected"))
    clip_valid_for_image_benchmark = artifact_summary.get("clip_valid_for_image_benchmark")
    clip_valid_for_video_benchmark = artifact_summary.get("clip_valid_for_video_benchmark")
    if runtime_invalid_reason:
        clip_valid_for_image_benchmark = False
        clip_valid_for_video_benchmark = False

    report = {
        "scene_root": str(getattr(config, "scene_root", "")),
        "miniscene_id": miniscene.get("miniscene_id"),
        "manifest_path": str(config.manifest_path),
        "requested_miniscene_index": int(config.miniscene_index),
        "selected_miniscene_index": int(selected_index),
        "selection_info": selection_info,
        "room": miniscene.get("room"),
        "scene_type": miniscene.get("scene_type"),
        "duplicate_motion_ids_present": len(set(spec["asset_id"] for spec in body_specs)) != len(body_specs),
        "humans_spawned": [_body_spec_report(spec, human) for spec, human in zip(body_specs, humans)],
        "motion_ids": [spec["asset_id"] for spec in body_specs],
        "camera_selection": camera_selection,
        "camera_pose_cm_deg": camera_pose,
        "room_camera_spatial_sanity": spatial_sanity,
        "scene_capture_cube": render_target_report,
        "scene_capture_exposure": exposure_report,
        "scene_capture_cube_pose": scene_capture_pose_report,
        "selected_miniscene_room": selected_room,
        "scene_binding": _configure_scene_root_paths(config.scene_root),
        "usd_stage_validation": _validate_loaded_usd_stage_matches_scene_root(
            Path(_configure_scene_root_paths(config.scene_root)["usd_stage_path"])
        ),
        "human_rooms": human_rooms,
        "lit_rooms": lit_rooms,
        "selected_room_lit": bool(selected_room in lit_rooms if selected_room else False),
        "lighting_warning": None if (selected_room in lit_rooms if selected_room else True) else "selected mini-scene room was not lit",
        "temporary_lighting": capture_lighting_report,
        "viewport_lighting_before_capture": viewport_lighting_report,
        "viewport_lighting_restored_after_capture": restored_viewport_lighting_report,
        "appearance_warmup": appearance_warmup_report,
        "capture_pipeline_diagnostics": capture_pipeline_diagnostics,
        "pre_render_warmup_report": (range_result or {}).get("pre_render_warmup_report"),
        "pre_render_probe_report": (range_result or {}).get("pre_render_probe_report"),
        "run_id": run_id,
        "run_root": str(run_root),
        "erp_output_path": str(run_root),
        "manifest_output_path": str(output_manifest_path),
        "selected_miniscene": miniscene,
        "render_options": {
            "frame_start": frame_start,
            "frame_end": frame_end,
            "use_natural_timing": use_natural_timing,
            "render_warmup_frame_count": int(getattr(config, "render_warmup_frame_count", 0) or 0),
            "discard_warmup_frames": bool(getattr(config, "discard_warmup_frames", False)),
            "probe_frame_before_render": bool(getattr(config, "probe_frame_before_render", False)),
            "reject_beige_probe": bool(getattr(config, "reject_beige_probe", False)),
            "bedlam_debug_appearance_mode": str(
                getattr(config, "bedlam_debug_appearance_mode", BEDLAM_DEBUG_APPEARANCE_MODE)
            ),
            "emit_memreport": bool(getattr(config, "emit_memreport", EMIT_MEMREPORT)),
            "emit_rhi_memory_dump": bool(getattr(config, "emit_rhi_memory_dump", EMIT_RHI_MEMORY_DUMP)),
        },
        "requested_render_output_profile": requested_render_output_profile,
        "effective_render_output_profile": effective_render_output_profile,
        "rgb_tonemap_mode": str(getattr(config, "rgb_tonemap_mode", RGB_TONEMAP_MODE)),
        "bedlam_debug_appearance_mode": str(
            getattr(config, "bedlam_debug_appearance_mode", BEDLAM_DEBUG_APPEARANCE_MODE)
        ),
        "profile_source": render_profile_config.get("profile_source"),
        "render_output_profile": render_profile_config,
        "render_diagnostics": render_diagnostics,
        "render_profiling": render_profiling,
        "artifact_frame_count": int(artifact_summary.get("artifact_frame_count", 0) or 0),
        "artifact_frame_indices": list(artifact_summary.get("artifact_frame_indices") or []),
        "valid_frame_count": artifact_summary.get("valid_frame_count"),
        "invalid_frame_count": artifact_summary.get("invalid_frame_count"),
        "clip_valid_for_image_benchmark": clip_valid_for_image_benchmark,
        "clip_valid_for_video_benchmark": clip_valid_for_video_benchmark,
        "planner_runtime_trajectory_consistency": planner_runtime_trajectory_consistency,
        "planner_runtime_trajectory_mismatch": planner_runtime_trajectory_mismatch,
        "runtime_root_out_of_room": runtime_root_out_of_room,
        "runtime_invalid_reason": runtime_invalid_reason,
        "clip_asset_reference_audit": {
            "clip_index": int(_CLIP_ASSET_AUDIT_STATE.get("clip_index", 0)),
            "asset_references": clip_asset_references,
            "spawned_asset_usage": result.get("spawned_asset_usage"),
            "newly_loaded_asset_paths_this_clip": sorted(
                entry["asset_path"] for entry in clip_asset_references if entry.get("new_this_clip")
            ),
            "seen_asset_paths_before_clip_count": int(len(_CLIP_ASSET_AUDIT_STATE.get("seen_asset_paths", set()))),
        },
        "memory_checkpoints": memory_checkpoints,
        "cleanup_after_clip": None,
        "frame_validity_index_json_path": artifact_summary.get("frame_validity_index_json_path"),
        "frame_validity_index_csv_path": artifact_summary.get("frame_validity_index_csv_path"),
        "frame_range_resolution": frame_range_resolution,
        "geometry_cache_frame_ranges": geometry_cache_frame_ranges,
        "preview_rgb_mp4": preview_videos.get("preview_rgb_mp4"),
        "preview_rgb_mp4_path": preview_videos.get("preview_rgb_mp4_path"),
        "preview_frame_count": preview_videos.get("preview_frame_count"),
        "preview_fps": preview_videos.get("preview_fps"),
        "preview_generation_seconds": float(
            ((preview_videos.get("profiling") or {}).get("preview_generation_seconds", 0.0) or 0.0)
        ),
        "preview_rgb_mp4_status": preview_videos.get("preview_rgb_mp4_status"),
        "preview_adaptive_mp4": preview_videos.get("preview_adaptive_mp4"),
        "preview_adaptive_mp4_status": preview_videos.get("preview_adaptive_mp4_status"),
        "adaptive_preview_available": bool(preview_videos.get("adaptive_preview_available")),
        "luminance_stats": luminance_stats,
        "cubemap_face_diagnostics_enabled": bool(ENABLE_CUBEMAP_FACE_DIAGNOSTICS),
        "cubemap_face_diagnostic_export_modes": list(CUBEMAP_FACE_DIAGNOSTIC_EXPORT_MODES),
        "cubemap_face_diagnostic_wait_seconds": float(CUBEMAP_FACE_DIAGNOSTIC_WAIT_SECONDS),
        "cubemap_face_anomaly_retry_enabled": bool(ENABLE_CUBEMAP_FACE_ANOMALY_RETRY),
        "cubemap_face_anomaly_retry_mode": str(CUBEMAP_FACE_ANOMALY_RETRY_MODE),
        "cubemap_face_anomaly_retry_max_attempts": int(CUBEMAP_FACE_ANOMALY_RETRY_MAX_ATTEMPTS),
        "export_face_diagnostics_only_on_retry": bool(base.canonical_validation.mini.EXPORT_FACE_DIAGNOSTICS_ONLY_ON_RETRY),
        "cubemap_face_diagnostics_summary": range_result.get("cubemap_face_diagnostics_summary"),
        "runtime_human_verification": runtime_human_verification,
        "MULTI_HUMAN_RUNTIME_DEBUG": multi_human_runtime_debug,
        "render_result": result,
        "gt_export": gt_export,
        "bridge_root": str(base.BRIDGE_ROOT),
        "bridge_benchmark_root": str(base.BRIDGE_BENCHMARK_ROOT),
    }
    for entry in clip_asset_references:
        _CLIP_ASSET_AUDIT_STATE["seen_asset_paths"].add(str(entry.get("asset_path")))
    cleanup_summary = _clip_cleanup_summary(
        result,
        run_root,
        clip_asset_references=clip_asset_references,
    )
    report["cleanup_after_clip"] = cleanup_summary
    memory_checkpoints.append(
        _capture_memory_checkpoint(
            "after_cleanup",
            run_root,
            extra={
                "miniscene_id": miniscene.get("miniscene_id"),
                "cleanup_removed_actor_count": len(cleanup_summary.get("bedlam_actors_cleared") or []),
                "clip_index": int(_CLIP_ASSET_AUDIT_STATE.get("clip_index", 0)),
            },
        )
    )
    after_cleanup_asset_states = []
    for entry in clip_asset_references:
        asset_path = str(entry.get("asset_path"))
        loaded_object_path = _find_runtime_loaded_object_path(asset_path)
        after_cleanup_asset_states.append(
            {
                "asset_type": entry.get("asset_type"),
                "asset_path": asset_path,
                "runtime_loaded_after_cleanup": bool(loaded_object_path),
                "runtime_loaded_object_path_after_cleanup": loaded_object_path,
            }
        )
    after_cleanup_asset_groups = _group_asset_paths_by_category(
        after_cleanup_asset_states,
        loaded_key="runtime_loaded_after_cleanup",
    )
    checkpoint_group_map = {
        "before_clip": before_clip_asset_groups,
        "after_clip_before_cleanup": after_clip_asset_groups,
        "after_cleanup": after_cleanup_asset_groups,
    }
    for checkpoint in memory_checkpoints:
        tag = str(checkpoint.get("tag") or "")
        if tag not in checkpoint_group_map:
            continue
        checkpoint.setdefault("extra", {})
        checkpoint["extra"]["asset_residency_groups"] = checkpoint_group_map[tag]
        checkpoint_path = checkpoint.get("checkpoint_path")
        if checkpoint_path:
            try:
                base._write_json(Path(checkpoint_path), checkpoint)
            except Exception:
                pass
    previous_clip_after_cleanup_groups = {
        key: sorted(values)
        for key, values in dict(_CLIP_ASSET_AUDIT_STATE.get("previous_clip_after_cleanup_by_category") or {}).items()
    }
    newly_loaded_since_previous_clip = _residency_diff(
        after_clip_asset_groups,
        previous_clip_after_cleanup_groups,
    )
    newly_loaded_during_this_clip = _residency_diff(
        after_clip_asset_groups,
        before_clip_asset_groups,
    )
    disappeared_after_cleanup = _residency_diff(
        after_clip_asset_groups,
        after_cleanup_asset_groups,
    )
    for category, paths in dict(after_clip_asset_groups).items():
        _CLIP_ASSET_AUDIT_STATE.setdefault("seen_paths_by_category", {}).setdefault(category, set()).update(paths or [])
    _CLIP_ASSET_AUDIT_STATE["previous_clip_after_cleanup_by_category"] = {
        key: set(values or []) for key, values in dict(after_cleanup_asset_groups).items()
    }
    before_clip_gpu_mb = (((memory_checkpoints[0].get("gpu_snapshot") or {}).get("unreal_process_gpu_memory_mb")))
    after_clip_gpu_mb = (((memory_checkpoints[1].get("gpu_snapshot") or {}).get("unreal_process_gpu_memory_mb"))) if len(memory_checkpoints) > 1 else None
    after_cleanup_gpu_mb = (((memory_checkpoints[-1].get("gpu_snapshot") or {}).get("unreal_process_gpu_memory_mb"))) if memory_checkpoints else None
    report["clip_asset_reference_audit"]["assets_still_loaded_after_cleanup"] = [
        row["asset_path"] for row in after_cleanup_asset_states if row.get("runtime_loaded_after_cleanup")
    ]
    report["clip_asset_reference_audit"]["asset_cleanup_state"] = after_cleanup_asset_states
    report["clip_asset_reference_audit"]["loaded_asset_counts"] = {
        "before_cleanup_referenced_asset_count": len(clip_asset_references),
        "after_cleanup_runtime_loaded_asset_count": sum(
            1 for row in after_cleanup_asset_states if row.get("runtime_loaded_after_cleanup")
        ),
        "by_asset_type_before_cleanup": _count_asset_types(clip_asset_references),
        "by_asset_type_runtime_loaded_before_cleanup": _count_asset_types(
            clip_asset_references,
            loaded_key="runtime_loaded_before_cleanup",
        ),
        "by_asset_type_runtime_loaded_after_cleanup": _count_asset_types(
            after_cleanup_asset_states,
            loaded_key="runtime_loaded_after_cleanup",
        ),
    }
    report["clip_asset_reference_audit"]["loaded_geometry_cache_assets"] = (
        _selected_asset_paths(clip_asset_references, "body_geometry_cache")
        + _selected_asset_paths(clip_asset_references, "clothing_geometry_cache")
    )
    report["clip_asset_reference_audit"]["loaded_groom_assets"] = _selected_asset_paths(
        clip_asset_references,
        "hair_groom",
    )
    report["clip_asset_reference_audit"]["loaded_skeletal_mesh_assets"] = _selected_asset_paths(
        clip_asset_references,
        "hair_binding",
    )
    report["clip_asset_reference_audit"]["asset_residency_snapshots"] = {
        "before_clip": before_clip_asset_groups,
        "after_clip_before_cleanup": after_clip_asset_groups,
        "after_cleanup": after_cleanup_asset_groups,
    }
    report["clip_asset_reference_audit"]["asset_residency_diffs"] = {
        "newly_loaded_since_previous_clip": newly_loaded_since_previous_clip,
        "newly_loaded_during_this_clip": newly_loaded_during_this_clip,
        "still_loaded_after_cleanup": after_cleanup_asset_groups,
        "disappeared_after_cleanup": disappeared_after_cleanup,
    }
    report["clip_asset_reference_audit"]["asset_residency_by_category"] = {
        category: {
            "count": len(after_cleanup_asset_groups.get(category) or []),
            "total_unique_paths_so_far": len(
                _CLIP_ASSET_AUDIT_STATE.get("seen_paths_by_category", {}).get(category, set())
            ),
            "new_unique_paths_this_clip": sorted(newly_loaded_since_previous_clip.get(category) or []),
            "new_unique_paths_this_clip_count": len(newly_loaded_since_previous_clip.get(category) or []),
            "paths_still_resident_after_cleanup": list(after_cleanup_asset_groups.get(category) or []),
        }
        for category in _empty_residency_groups().keys()
    }
    asset_residency_csv_path = Path(base.BRIDGE_ROOT) / "asset_residency_by_clip.csv"
    csv_rows = []
    checkpoint_groups = {
        "before_clip": (before_clip_asset_groups, before_clip_gpu_mb),
        "after_clip_before_cleanup": (after_clip_asset_groups, after_clip_gpu_mb),
        "after_cleanup": (after_cleanup_asset_groups, after_cleanup_gpu_mb),
    }
    for checkpoint_name, (groups, gpu_mb) in checkpoint_groups.items():
        for category in _empty_residency_groups().keys():
            csv_rows.append(
                {
                    "clip_index": int(_CLIP_ASSET_AUDIT_STATE.get("clip_index", 0)),
                    "miniscene_id": miniscene.get("miniscene_id"),
                    "appearance_mode": appearance_mode,
                    "checkpoint": checkpoint_name,
                    "gpu_mb": gpu_mb,
                    "category": category,
                    "loaded_count": len(groups.get(category) or []),
                    "unique_seen_so_far": len(
                        _CLIP_ASSET_AUDIT_STATE.get("seen_paths_by_category", {}).get(category, set())
                    ),
                    "new_this_clip_count": len(newly_loaded_since_previous_clip.get(category) or []),
                    "still_loaded_after_cleanup_count": len(after_cleanup_asset_groups.get(category) or []),
                }
            )
    _append_asset_residency_csv_rows(asset_residency_csv_path, csv_rows)
    report["clip_asset_reference_audit"]["asset_residency_csv_path"] = str(asset_residency_csv_path)
    report["vram_delta_report"] = {
        "before_clip_unreal_gpu_memory_mb": before_clip_gpu_mb,
        "after_clip_unreal_gpu_memory_mb": after_clip_gpu_mb,
        "after_cleanup_unreal_gpu_memory_mb": after_cleanup_gpu_mb,
        "clip_delta_mb": None if before_clip_gpu_mb is None or after_clip_gpu_mb is None else float(after_clip_gpu_mb - before_clip_gpu_mb),
        "cleanup_delta_mb": None if after_clip_gpu_mb is None or after_cleanup_gpu_mb is None else float(after_cleanup_gpu_mb - after_clip_gpu_mb),
        "total_delta_mb": None if before_clip_gpu_mb is None or after_cleanup_gpu_mb is None else float(after_cleanup_gpu_mb - before_clip_gpu_mb),
    }
    report["memory_checkpoints"] = memory_checkpoints
    report_path = run_root / "bridge_report.json"
    base._write_json(report_path, report)
    if runtime_invalid_reason:
        raise RuntimeError(
            f"Rendered clip failed runtime trajectory validation: reason={runtime_invalid_reason} "
            f"report_path={report_path}"
        )
    reject_clip_with_artifacts = bool(getattr(config, "reject_clips_with_artifacts", REJECT_CLIPS_WITH_ARTIFACTS))
    if int(report.get("valid_frame_count", 0) or 0) <= 0:
        raise RuntimeError(
            f"Rendered clip has no valid benchmark frames after cubemap artifact tagging. "
            f"artifact_frame_count={report.get('artifact_frame_count')} report_path={report_path}"
        )
    if reject_clip_with_artifacts and int(report.get("artifact_frame_count", 0) or 0) > 0:
        raise RuntimeError(
            f"Rendered clip rejected because artifact-tagged frames were detected and "
            f"--reject-clips-with-artifacts is enabled. artifact_frame_indices={report.get('artifact_frame_indices')} "
            f"report_path={report_path}"
        )
    log_info("MULTI_HUMAN_RUNTIME_DEBUG " + json.dumps(multi_human_runtime_debug, indent=2))
    log_info("FINAL_REPORT " + json.dumps(report, indent=2))
    return report


def _render_batch_miniscenes(config):
    if LIGHTING_INTENSITY_SWEEP:
        raise RuntimeError("Disable LIGHTING_INTENSITY_SWEEP before using batch miniscene rendering.")
    if getattr(config, "scene_root", None) in (None, ""):
        raise RuntimeError(
            "Batch miniscene rendering requires --scene-root so per-clip scene metadata/USD paths stay bound "
            "to the requested Infinigen scene."
        )

    source_manifest, selected_items = _select_batch_miniscenes(
        config.manifest_path,
        max_count=config.batch_max_miniscenes,
        room_filter=config.batch_room_filter,
        balanced_rooms=config.batch_balanced_rooms,
    )
    log_info(
        "BATCH_RENDER_OUTPUT_PROFILE "
        + json.dumps(
            {
                "requested_render_output_profile": str(getattr(config, "render_output_profile", RENDER_OUTPUT_PROFILE)),
                "profile_source": "cli" if bool(_argv_contains_any("--render-output-profile")) else "default",
            },
            indent=2,
        )
    )
    timestamp = base.canonical_validation._utc_now().strftime("%Y%m%dT%H%M%SZ")
    batch_root = base._ensure_dir(base.BRIDGE_ROOT / "batch_miniscene_runs" / timestamp)
    reports = []
    failures = []

    log_info(
        f"BATCH_RENDER start manifest={config.manifest_path} selected={len(selected_items)} "
        f"max={config.batch_max_miniscenes} room_filter={config.batch_room_filter}"
    )
    for item in selected_items:
        miniscene = item["miniscene"]
        index = int(item["index"])
        selection_info = dict(item["selection_info"])
        log_info(
            f"BATCH_RENDER miniscene_id={miniscene.get('miniscene_id')} index={index} room={miniscene.get('room')}"
        )
        item_config = argparse.Namespace(
            scene_root=config.scene_root,
            manifest_path=config.manifest_path,
            miniscene_index=index,
            miniscene_id=miniscene.get("miniscene_id"),
            render_output_profile=getattr(config, "render_output_profile", RENDER_OUTPUT_PROFILE),
            rgb_tonemap_mode=str(getattr(config, "rgb_tonemap_mode", RGB_TONEMAP_MODE)),
            reject_clips_with_artifacts=bool(getattr(config, "reject_clips_with_artifacts", REJECT_CLIPS_WITH_ARTIFACTS)),
            bedlam_debug_appearance_mode=str(
                getattr(config, "bedlam_debug_appearance_mode", BEDLAM_DEBUG_APPEARANCE_MODE)
            ),
            emit_memreport=bool(getattr(config, "emit_memreport", EMIT_MEMREPORT)),
            emit_rhi_memory_dump=bool(getattr(config, "emit_rhi_memory_dump", EMIT_RHI_MEMORY_DUMP)),
            batch_inherited_render_output_profile=True,
            batch=config.batch,
            batch_max_miniscenes=config.batch_max_miniscenes,
            batch_room_filter=config.batch_room_filter,
            batch_balanced_rooms=config.batch_balanced_rooms,
            frame_start=getattr(config, "frame_start", None),
            frame_end=getattr(config, "frame_end", None),
        )
        try:
            report = _render_single_miniscene_report(
                config=item_config,
                source_manifest=source_manifest,
                miniscene=miniscene,
                selected_index=index,
                selection_info=selection_info,
            )
            reports.append(report)
        except Exception as exc:
            failure = {
                "miniscene_id": miniscene.get("miniscene_id"),
                "requested_miniscene_index": index,
                "room": miniscene.get("room"),
                "scene_type": miniscene.get("scene_type"),
                "error": str(exc),
            }
            failures.append(failure)
            log_info(f"BATCH_RENDER failure {json.dumps(failure)}")

    summary = {
        "mode": "batch_miniscene_render",
        "scene_root": str(getattr(config, "scene_root", "")),
        "manifest_path": str(config.manifest_path),
        "batch_root": str(batch_root),
        "requested_max_miniscenes": int(config.batch_max_miniscenes),
        "room_filter": config.batch_room_filter,
        "batch_balanced_rooms": bool(config.batch_balanced_rooms),
        "selected_count": len(selected_items),
        "rendered_count": len(reports),
        "failed_count": len(failures),
        "selected_rooms": [item["room"] for item in selected_items],
        "selected_room_distribution": _room_distribution(selected_items),
        "rendered_room_distribution": _room_distribution(reports),
        "reports": [_batch_summary_row(report) for report in reports],
        "failures": failures,
    }
    summary_path = batch_root / "batch_miniscene_render_report.json"
    base._write_json(summary_path, summary)
    log_info(
        "BATCH_FINAL_REPORT "
        + json.dumps(
            {
                "summary_path": str(summary_path),
                "rendered_count": len(reports),
                "failed_count": len(failures),
            },
            indent=2,
        )
    )
    return summary


def render_selected_infinigen_bedlam_erp():
    global EMIT_MEMREPORT
    global EMIT_RHI_MEMORY_DUMP
    config = _load_run_config()
    EMIT_MEMREPORT = bool(getattr(config, "emit_memreport", EMIT_MEMREPORT))
    EMIT_RHI_MEMORY_DUMP = bool(getattr(config, "emit_rhi_memory_dump", EMIT_RHI_MEMORY_DUMP))
    if getattr(config, "offline_run_root", None):
        report = _offline_run_trajectory_consistency_audit(config.offline_run_root)
        log_info("OFFLINE_TRAJECTORY_CONSISTENCY_AUDIT " + json.dumps(report, indent=2))
        return report
    scene_root, resolved_manifest_path, startup_manifest, startup_miniscenes = _startup_manifest_validation(config)
    config.manifest_path = resolved_manifest_path
    config.scene_root = scene_root
    scene_binding_report = _configure_scene_root_paths(scene_root)
    loaded_usd_stage_validation = _validate_loaded_usd_stage_matches_scene_root(
        Path(scene_binding_report["usd_stage_path"])
    )
    log_info("SCENE_ROOT_BINDING " + json.dumps(scene_binding_report, indent=2))
    log_info("USD_STAGE_VALIDATION " + json.dumps(loaded_usd_stage_validation, indent=2))
    if getattr(config, "list_available_motions", False):
        return _list_available_motions_report(config.manifest_path)
    if getattr(config, "diagnose_renderability", False):
        room_filter = getattr(config, "batch_room_filter", None)
        if room_filter in (None, ""):
            room_filter = getattr(config, "miniscene_room", None)
        return _diagnose_renderability_report(
            config.manifest_path,
            room_filter=room_filter,
        )
    if getattr(config, "list_renderable", False):
        room_filter = getattr(config, "batch_room_filter", None)
        if room_filter in (None, ""):
            room_filter = getattr(config, "miniscene_room", None)
        return _list_renderable_miniscenes_report(
            config.manifest_path,
            room_filter=room_filter,
            limit=getattr(config, "list_renderable_limit", LIST_RENDERABLE_LIMIT),
        )
    if config.batch:
        return _render_batch_miniscenes(config)

    explicit_selection = bool(
        getattr(config, "miniscene_id", None) not in (None, "")
        or getattr(config, "miniscene_room", None) not in (None, "")
        or bool(getattr(config, "explicit_manifest_arg", False))
        or bool(getattr(config, "explicit_miniscene_index_arg", False))
    )
    source_manifest, requested_index = _resolve_requested_miniscene_index(
        config.manifest_path,
        miniscene_index=config.miniscene_index,
        miniscene_id=getattr(config, "miniscene_id", None),
        room_hint=getattr(config, "miniscene_room", None),
    )
    requested_miniscene = startup_miniscenes[requested_index]
    log_info(
        "REQUESTED_MINISCENE "
        + json.dumps(
            _miniscene_brief(requested_miniscene, index=requested_index),
            indent=2,
        )
    )
    allow_fallback = (
        False if explicit_selection else bool(ALLOW_FALLBACK_TO_NEXT_RENDERABLE)
    )
    if explicit_selection and not ALLOW_FALLBACK_FOR_EXPLICIT_SELECTION and not allow_fallback:
        log_info("STARTUP_SELECTION_MODE explicit_selection=true fallback_disabled=true")

    source_manifest, miniscene, selected_index, selection_info = _select_renderable_miniscene(
        config.manifest_path,
        requested_index,
        allow_fallback=allow_fallback,
    )
    if explicit_selection and int(selected_index) != int(requested_index):
        raise RuntimeError(
            f"Explicit mini-scene request resolved to requested_index={requested_index} "
            f"but renderer selected_index={selected_index}. Fallback is not allowed."
        )
    return _render_single_miniscene_report(
        config=config,
        source_manifest=source_manifest,
        miniscene=miniscene,
        selected_index=selected_index,
        selection_info=selection_info,
    )


if __name__ == "__main__":
    render_selected_infinigen_bedlam_erp()
