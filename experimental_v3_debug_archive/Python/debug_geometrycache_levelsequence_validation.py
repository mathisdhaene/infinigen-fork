import csv
import importlib
import json
import math
from pathlib import Path
import sys

import unreal


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bedlam360_mini_validation
import capture_scene_cube
import reconstruct_one_bedlam_body

bedlam360_mini_validation = importlib.reload(bedlam360_mini_validation)
capture_scene_cube = importlib.reload(capture_scene_cube)
reconstruct_one_bedlam_body = importlib.reload(reconstruct_one_bedlam_body)


DEFAULT_ASSET_ID = "it_4052_3XL_2403"
DEFAULT_NUM_FRAMES = 120
DEFAULT_FPS = 30
DEFAULT_CAPTURE_ACTOR_LABEL = "SceneCaptureCube"
DEFAULT_SEQUENCE_DIR = "/Game/BEDLAM360_Debug"
DEFAULT_SEQUENCE_NAME = "LS_GeometryCacheValidation"
DEFAULT_OUTPUT_ROOT = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_levelsequence_validation")
DEFAULT_CAMERA_MODE = "fixed"
DEFAULT_ORBIT_RADIUS_CM = 320.0
DEFAULT_ORBIT_HEIGHT_CM = 140.0
DEFAULT_BATCH_LIMIT = 8
DEFAULT_PREFERRED_DYNAMIC_ASSET_IDS = [
    "it_4052_3XL_2406",
    "it_4052_3XL_2408",
    "it_4052_3XL_2410",
    "it_4083_2XL_2408",
    "it_4191_M_2407",
    "it_4202_L_2406",
    "it_4288_XL_2403",
    "it_4420_L_2405",
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


def _destroy_existing_debug_actors(prefix):
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    removed = 0
    for actor in list(unreal.EditorLevelLibrary.get_all_level_actors()):
        try:
            label = actor.get_actor_label()
        except Exception:
            continue
        if not str(label).startswith(prefix):
            continue
        if actor_subsystem.destroy_actor(actor):
            removed += 1
    if removed:
        unreal.log(f"[BEDLAM360][LS_VALIDATE] Destroyed {removed} existing actors with prefix '{prefix}'")


def _gc_asset_path_from_actor(actor):
    try:
        component = actor.get_geometry_cache_component()
    except Exception:
        return None
    if component is None:
        return None
    try:
        gc_asset = component.get_editor_property("geometry_cache")
    except Exception:
        gc_asset = None
    if gc_asset is None:
        return None
    try:
        return gc_asset.get_path_name()
    except Exception:
        return str(gc_asset)


def _actor_hidden_state(actor):
    state = {"hidden_in_game": None, "hidden_in_editor": None}
    try:
        state["hidden_in_game"] = actor.get_actor_hidden_in_game()
    except Exception:
        try:
            state["hidden_in_game"] = actor.is_hidden()
        except Exception:
            state["hidden_in_game"] = None
    try:
        state["hidden_in_editor"] = actor.is_temporarily_hidden_in_editor()
    except Exception:
        try:
            state["hidden_in_editor"] = actor.is_hidden_ed()
        except Exception:
            state["hidden_in_editor"] = None
    return state


def _set_actor_hidden(actor, hidden):
    try:
        actor.set_actor_hidden_in_game(bool(hidden))
    except Exception:
        pass
    try:
        actor.set_is_temporarily_hidden_in_editor(bool(hidden))
    except Exception:
        pass


def _get_bound_gc_actor(binding_id):
    try:
        bound_objects = unreal.LevelSequenceEditorBlueprintLibrary.get_bound_objects(binding_id)
    except Exception:
        return None
    for obj in bound_objects:
        if obj is None:
            continue
        if hasattr(obj, "get_geometry_cache_component"):
            return obj
        if obj.get_class().get_name() == "GeometryCacheComponent":
            try:
                return obj.get_owner()
            except Exception:
                return None
    return None


def _geometrycache_actor_snapshots(bound_actor=None):
    snapshots = []
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        try:
            if actor.get_class().get_name() != "GeometryCacheActor":
                continue
            label = actor.get_actor_label()
            location = actor.get_actor_location()
        except Exception:
            continue
        hidden_state = _actor_hidden_state(actor)
        snapshots.append(
            {
                "actor_label": label,
                "bound_to_sequence": bool(bound_actor is not None and actor == bound_actor),
                "geometry_cache_asset_path": _gc_asset_path_from_actor(actor),
                "hidden_in_game": hidden_state["hidden_in_game"],
                "hidden_in_editor": hidden_state["hidden_in_editor"],
                "location_cm": {"x": location.x, "y": location.y, "z": location.z},
            }
        )
    return snapshots


def _log_geometrycache_actor_snapshots(stage, bound_actor=None):
    snapshots = _geometrycache_actor_snapshots(bound_actor=bound_actor)
    unreal.log(f"[BEDLAM360][LS_VALIDATE][{stage}] geometry_cache_actors={json.dumps(snapshots, indent=2)}")
    return snapshots


def _remove_or_hide_duplicate_bedlam_actors(bound_actor=None):
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    removed = 0
    hidden = 0
    for actor in list(unreal.EditorLevelLibrary.get_all_level_actors()):
        try:
            if actor.get_class().get_name() != "GeometryCacheActor":
                continue
            label = actor.get_actor_label()
        except Exception:
            continue
        if not (str(label).startswith("BEDLAM360_") or str(label).startswith("BEDLAM360_ls_validate_")):
            continue
        if bound_actor is not None and actor == bound_actor:
            _set_actor_hidden(actor, False)
            continue
        if actor_subsystem.destroy_actor(actor):
            removed += 1
            continue
        _set_actor_hidden(actor, True)
        try:
            actor.set_actor_location(unreal.Vector(50000.0, 50000.0, -50000.0), False, False)
        except Exception:
            pass
        hidden += 1
    if removed or hidden:
        unreal.log(
            f"[BEDLAM360][LS_VALIDATE] duplicate_cleanup removed={removed} hidden={hidden} "
            f"bound_actor={(None if bound_actor is None else bound_actor.get_actor_label())}"
        )


def _sequence_asset_path(asset_id):
    return f"{DEFAULT_SEQUENCE_DIR}/{DEFAULT_SEQUENCE_NAME}_{asset_id}"


def _list_local_animation_asset_ids():
    assets = []
    for path in sorted(reconstruct_one_bedlam_body.LOCAL_ANIMATIONS_ROOT.glob("*/*.uasset")):
        stem = path.stem
        if stem.endswith("_Anim") or stem.endswith("_Skeleton") or stem.endswith("_PhysicsAsset"):
            continue
        assets.append(stem)
    return assets


def _select_dynamic_asset_ids(limit=DEFAULT_BATCH_LIMIT):
    available = set(_list_local_animation_asset_ids())
    selected = []
    for asset_id in DEFAULT_PREFERRED_DYNAMIC_ASSET_IDS:
        if asset_id in available and asset_id not in selected:
            selected.append(asset_id)
        if len(selected) >= limit:
            return selected

    by_identity = {}
    for asset_id in sorted(available):
        identity = "_".join(asset_id.split("_")[:-1])
        by_identity.setdefault(identity, []).append(asset_id)

    identities = sorted(by_identity.keys(), key=lambda key: (-len(by_identity[key]), key))
    for identity in identities:
        candidates = sorted(by_identity[identity], reverse=True)
        for asset_id in candidates[:2]:
            if asset_id not in selected:
                selected.append(asset_id)
            if len(selected) >= limit:
                return selected
    return selected


def _make_binding_id(binding_proxy):
    binding_id = unreal.MovieSceneObjectBindingID()
    try:
        binding_id.set_editor_property("guid", binding_proxy.get_id())
    except Exception:
        binding_id.set_editor_property("Guid", binding_proxy.get_id())
    return binding_id


def _get_gc_component_state(component):
    state = {}
    for key, getter_name in (
        ("duration_seconds", "get_duration"),
        ("num_frames", "get_number_of_frames"),
        ("animation_time_seconds", "get_animation_time"),
        ("is_playing", "is_playing"),
        ("is_looping", "is_looping"),
        ("playback_speed", "get_playback_speed"),
    ):
        try:
            state[key] = getattr(component, getter_name)()
        except Exception:
            state[key] = None
    for property_name in ("manual_tick", "running", "looping", "elapsed_time"):
        try:
            state[f"property_{property_name}"] = component.get_editor_property(property_name)
        except Exception:
            state[f"property_{property_name}"] = None
    return state


def _find_bound_gc_component(binding_id):
    try:
        bound_objects = unreal.LevelSequenceEditorBlueprintLibrary.get_bound_objects(binding_id)
    except Exception as exc:
        return None, {"resolved": False, "reason": str(exc), "bound_object_count": 0}

    info = {"resolved": False, "reason": None, "bound_object_count": len(bound_objects), "bound_object_classes": []}
    for obj in bound_objects:
        if obj is None:
            continue
        info["bound_object_classes"].append(obj.get_class().get_name())
        if hasattr(obj, "get_geometry_cache_component"):
            component = obj.get_geometry_cache_component()
            if component is not None:
                info["resolved"] = True
                info["reason"] = "actor_geometry_cache_component"
                info["bound_actor_label"] = obj.get_actor_label()
                return component, info
        if obj.get_class().get_name() == "GeometryCacheComponent":
            info["resolved"] = True
            info["reason"] = "direct_component"
            try:
                info["bound_actor_label"] = obj.get_owner().get_actor_label()
            except Exception:
                info["bound_actor_label"] = None
            return obj, info
    info["reason"] = "no_geometry_cache_component_found"
    return None, info


def _set_transform_defaults(binding, x=0.0, y=0.0, z=0.0, pitch=0.0, yaw=0.0, roll=0.0):
    transform_track = binding.add_track(unreal.MovieScene3DTransformTrack)
    transform_section = transform_track.add_section()
    transform_section.set_start_frame_bounded(False)
    transform_section.set_end_frame_bounded(False)
    channels = transform_section.get_all_channels()
    channels[0].set_default(x)
    channels[1].set_default(y)
    channels[2].set_default(z)
    channels[3].set_default(roll)
    channels[4].set_default(pitch)
    channels[5].set_default(yaw)


def _camera_pose_for_frame(frame_index, num_frames, camera_mode):
    target_loc = {"x": 0.0, "y": 0.0, "z": 120.0}
    if camera_mode == "orbit":
        angle = (frame_index / float(max(1, num_frames))) * 360.0
        radians = math.radians(angle)
        camera_loc = {
            "x": target_loc["x"] + DEFAULT_ORBIT_RADIUS_CM * math.cos(radians),
            "y": target_loc["y"] + DEFAULT_ORBIT_RADIUS_CM * math.sin(radians),
            "z": DEFAULT_ORBIT_HEIGHT_CM,
        }
    else:
        camera_loc = {"x": 260.0, "y": -180.0, "z": 140.0}
    pitch, yaw, roll = bedlam360_mini_validation._look_at_rotation(camera_loc, target_loc)
    return camera_loc, pitch, yaw, roll


def _write_batch_gallery(sequences, output_html_path):
    cards = []
    for sequence in sequences:
        thumb = sequence.get("thumbnail_png_path") or ""
        mp4 = sequence.get("preview_mp4_path") or ""
        summary = (
            f"{sequence.get('asset_id', '')} | mode={sequence.get('camera_mode', '')} | "
            f"frames={sequence.get('frame_count', '')} | motion={sequence.get('movement_detected', '')}"
        )
        if thumb:
            media = f'<a href="{mp4}"><img src="{thumb}" alt="{summary}"></a>'
        elif mp4:
            media = f'<a href="{mp4}">{sequence.get("asset_id", "")}</a>'
        else:
            media = sequence.get("asset_id", "")
        cards.append(
            f"""
            <div class="card">
              <div class="media">{media}</div>
              <div class="summary">{summary}</div>
            </div>
            """
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>BEDLAM360 LevelSequence Validation</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#111; color:#eee; margin:24px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:20px; }}
    .card {{ background:#1b1b1b; padding:12px; border-radius:10px; }}
    .card img {{ width:100%; height:auto; display:block; border-radius:6px; }}
    a {{ color:#9ad; text-decoration:none; }}
    .summary {{ margin-top:10px; font-size:14px; line-height:1.4; }}
  </style>
</head>
<body>
  <h1>BEDLAM360 LevelSequence Validation</h1>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    Path(output_html_path).write_text(html, encoding="utf-8")


def _write_batch_summary_csv(rows, csv_path):
    fieldnames = [
        "asset_id",
        "camera_mode",
        "duration_seconds",
        "num_frames",
        "max_actual_time_seconds",
        "max_target_time_seconds",
        "max_time_error_seconds",
        "mean_pixel_diff",
        "mean_center_crop_pixel_diff",
        "movement_detected",
        "section_assignment_success",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _create_level_sequence_for_asset(asset_id, num_frames, fps):
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    sequence_asset_path = _sequence_asset_path(asset_id)
    asset_dir, asset_name = sequence_asset_path.rsplit("/", 1)
    unreal.EditorAssetLibrary.make_directory(asset_dir)
    if unreal.EditorAssetLibrary.does_asset_exist(sequence_asset_path):
        unreal.EditorAssetLibrary.delete_asset(sequence_asset_path)

    level_sequence = asset_tools.create_asset(
        asset_name=asset_name,
        package_path=asset_dir,
        asset_class=unreal.LevelSequence,
        factory=unreal.LevelSequenceFactoryNew(),
    )
    if level_sequence is None:
        raise RuntimeError(f"Could not create LevelSequence asset at {sequence_asset_path}")

    frame_rate = unreal.FrameRate(fps, 1)
    level_sequence.set_display_rate(frame_rate)
    try:
        level_sequence.set_tick_resolution(frame_rate)
    except Exception:
        try:
            level_sequence.get_movie_scene().set_tick_resolution_directly(frame_rate)
        except Exception:
            pass
    level_sequence.set_playback_start(0)
    level_sequence.set_playback_end(num_frames)

    resolved_asset = reconstruct_one_bedlam_body.resolve_body_asset(asset_id)
    geometry_cache_path = resolved_asset.get("body_geometry_cache_path") or resolved_asset["unreal_asset_path"]
    geometry_cache_asset = unreal.EditorAssetLibrary.load_asset(geometry_cache_path)
    if geometry_cache_asset is None:
        raise RuntimeError(f"Could not load GeometryCache asset: {geometry_cache_path}")

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    temp_actor = actor_subsystem.spawn_actor_from_class(unreal.GeometryCacheActor, unreal.Vector(0, 0, 0))
    temp_actor.set_actor_label(f"BEDLAM360_ls_validate_{asset_id}")
    gc_component = temp_actor.get_geometry_cache_component()
    gc_component.set_editor_property("geometry_cache", geometry_cache_asset)
    reconstruct_one_bedlam_body.configure_geometry_cache_component(gc_component, manual_tick=True, looping=False)

    binding = level_sequence.add_spawnable_from_instance(temp_actor)
    actor_subsystem.destroy_actor(temp_actor)

    geometry_track = binding.add_track(unreal.MovieSceneGeometryCacheTrack)
    geometry_section = geometry_track.add_section()
    geometry_section.set_range(0, num_frames)

    section_assignment = {"attempted": True, "success": False, "reason": None}
    try:
        params = geometry_section.get_editor_property("params")
        params.geometry_cache_asset = geometry_cache_asset
        params.play_rate = 1.0
        geometry_section.set_editor_property("params", params)
        section_assignment["success"] = True
    except Exception as exc:
        section_assignment["reason"] = str(exc)

    _set_transform_defaults(binding, x=0.0, y=0.0, z=0.0, pitch=0.0, yaw=0.0, roll=0.0)

    unreal.EditorAssetLibrary.save_loaded_asset(level_sequence)
    return {
        "level_sequence": level_sequence,
        "binding": binding,
        "binding_id": _make_binding_id(binding),
        "resolved_asset": resolved_asset,
        "geometry_cache_asset": geometry_cache_asset,
        "section_assignment": section_assignment,
        "level_sequence_summary": _summarize_level_sequence(level_sequence),
        "sequence_asset_path": sequence_asset_path,
    }


def _evaluate_sequence_frame(level_sequence, frame_index):
    if not unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(level_sequence):
        raise RuntimeError(f"Could not open LevelSequence: {level_sequence.get_path_name()}")
    unreal.LevelSequenceEditorBlueprintLibrary.pause()
    unreal.LevelSequenceEditorBlueprintLibrary.set_current_time(int(frame_index))
    unreal.LevelSequenceEditorBlueprintLibrary.refresh_current_level_sequence()
    bedlam360_mini_validation._invalidate_editor_viewports()
    return unreal.LevelSequenceEditorBlueprintLibrary.get_current_time()


def _write_summary_csv(summary_row, csv_path):
    fieldnames = [
        "asset_id",
        "camera_mode",
        "duration_seconds",
        "num_frames",
        "max_actual_time_seconds",
        "max_target_time_seconds",
        "max_time_error_seconds",
        "mean_pixel_diff",
        "mean_center_crop_pixel_diff",
        "movement_detected",
        "section_assignment_success",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)


def _safe_getattr_call(obj, method_name):
    try:
        method = getattr(obj, method_name)
    except Exception:
        return None
    try:
        return method()
    except Exception:
        return None


def _safe_section_range(section):
    result = {
        "start_frame": None,
        "end_frame": None,
        "has_start_frame": None,
        "has_end_frame": None,
    }
    for key, method_name in (
        ("start_frame", "get_start_frame"),
        ("end_frame", "get_end_frame"),
        ("has_start_frame", "has_start_frame"),
        ("has_end_frame", "has_end_frame"),
    ):
        result[key] = _safe_getattr_call(section, method_name)
    return result


def _safe_geometry_cache_asset_path_from_section(section):
    try:
        params = section.get_editor_property("params")
    except Exception:
        return None
    try:
        asset = params.geometry_cache_asset
    except Exception:
        asset = None
    if asset is None:
        return None
    try:
        return asset.get_path_name()
    except Exception:
        return str(asset)


def _summarize_level_sequence(level_sequence):
    summary = {
        "sequence_asset_path": level_sequence.get_path_name(),
        "display_rate": None,
        "spawnable_count": 0,
        "possessable_count": 0,
        "bindings": [],
    }
    try:
        display_rate = level_sequence.get_display_rate()
        summary["display_rate"] = {"numerator": display_rate.numerator, "denominator": display_rate.denominator}
    except Exception:
        summary["display_rate"] = None

    bindings = []
    try:
        bindings.extend(list(level_sequence.get_spawnables()))
        summary["spawnable_count"] = len(bindings)
    except Exception:
        pass
    try:
        possessables = list(level_sequence.get_possessables())
        summary["possessable_count"] = len(possessables)
        bindings.extend(possessables)
    except Exception:
        pass

    for binding in bindings:
        binding_record = {
            "name": None,
            "binding_type": "unknown",
            "track_count": 0,
            "tracks": [],
        }
        try:
            binding_record["name"] = binding.get_name()
        except Exception:
            binding_record["name"] = None
        binding_record["binding_type"] = "spawnable" if binding in bindings[: summary["spawnable_count"]] else "possessable"
        try:
            tracks = list(binding.get_tracks())
        except Exception:
            tracks = []
        binding_record["track_count"] = len(tracks)
        for track in tracks:
            track_record = {
                "track_class": track.get_class().get_name(),
                "section_count": 0,
                "sections": [],
            }
            try:
                sections = list(track.get_sections())
            except Exception:
                sections = []
            track_record["section_count"] = len(sections)
            for section in sections:
                section_record = {
                    "section_class": section.get_class().get_name(),
                    "range": _safe_section_range(section),
                    "geometry_cache_asset_path": _safe_geometry_cache_asset_path_from_section(section),
                }
                track_record["sections"].append(section_record)
            binding_record["tracks"].append(track_record)
        summary["bindings"].append(binding_record)
    return summary


def _compute_png_pair_diff(path_a, path_b):
    return None


def _frame_pair_diff_from_records(frame_records, a, b):
    frames_by_index = {frame.get("frame_index"): frame for frame in frame_records}
    if a not in frames_by_index or b not in frames_by_index:
        return None
    if a == b:
        return 0.0
    running = []
    start = min(a, b) + 1
    end = max(a, b)
    for frame_index in range(start, end + 1):
        frame = frames_by_index.get(frame_index)
        if frame is None:
            continue
        value = frame.get("center_crop_pixel_diff_prev")
        if value is None:
            value = frame.get("pixel_diff_prev")
        if value is not None:
            running.append(float(value))
    if not running:
        return None
    return float(sum(running) / len(running))


def validate_geometrycache_with_levelsequence(
    asset_id=DEFAULT_ASSET_ID,
    num_frames=DEFAULT_NUM_FRAMES,
    fps=DEFAULT_FPS,
    actor_label=DEFAULT_CAPTURE_ACTOR_LABEL,
    output_root=DEFAULT_OUTPUT_ROOT,
    camera_mode=DEFAULT_CAMERA_MODE,
):
    output_root = _reset_dir(output_root / asset_id)
    images_dir = _ensure_dir(output_root / "images")
    metadata_dir = _ensure_dir(output_root / "metadata")
    previews_dir = _ensure_dir(output_root / "previews")

    _destroy_existing_debug_actors("BEDLAM360_ls_validate_")
    _remove_or_hide_duplicate_bedlam_actors(bound_actor=None)
    initial_actor_snapshot = _geometrycache_actor_snapshots(bound_actor=None)

    sequence_info = _create_level_sequence_for_asset(asset_id, num_frames=num_frames, fps=fps)
    level_sequence = sequence_info["level_sequence"]
    binding_id = sequence_info["binding_id"]

    camera_actor = capture_scene_cube.find_scene_capture_cube(actor_label)
    capture_component = capture_scene_cube.get_capture_component(camera_actor)
    texture_target = capture_scene_cube.get_texture_target(capture_component)
    export_lib = unreal.BEDLAM360ExportLibrary

    frame_records = []
    png_paths = []
    for frame_index in range(num_frames):
        evaluated_frame = _evaluate_sequence_frame(level_sequence, frame_index)
        bound_actor = _get_bound_gc_actor(binding_id)
        _remove_or_hide_duplicate_bedlam_actors(bound_actor=bound_actor)
        gc_component, binding_info = _find_bound_gc_component(binding_id)
        component_state = None if gc_component is None else _get_gc_component_state(gc_component)
        actual_time = None if component_state is None else component_state.get("animation_time_seconds")
        duration = None if component_state is None else component_state.get("duration_seconds")
        sample_frame_index = None
        if actual_time is not None:
            sample_frame_index = int(round(actual_time * float(fps)))
        camera_loc, pitch, yaw, roll = _camera_pose_for_frame(frame_index, num_frames, camera_mode)
        capture_scene_cube.set_actor_pose(camera_actor, camera_loc["x"], camera_loc["y"], camera_loc["z"], pitch, yaw, roll)

        frame_name = f"{asset_id}_ls_frame_{frame_index:04d}"
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
            "frame_index": frame_index,
            "time_seconds": frame_index / float(max(1, fps)),
            "evaluated_frame": evaluated_frame,
            "asset_id": asset_id,
            "asset_path": sequence_info["geometry_cache_asset"].get_path_name(),
            "camera_mode": camera_mode,
            "camera_pose_cm_deg": {
                "x": camera_loc["x"],
                "y": camera_loc["y"],
                "z": camera_loc["z"],
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
            },
            "binding_info": binding_info,
            "bound_actor_label": None if bound_actor is None else bound_actor.get_actor_label(),
            "component_state_after": component_state,
            "actual_time_seconds": actual_time,
            "manual_tick": None if component_state is None else component_state.get("property_manual_tick"),
            "is_playing": None if component_state is None else component_state.get("is_playing"),
            "duration_seconds": duration,
            "animation_frame_indices": [] if sample_frame_index is None else [sample_frame_index],
            "hdr_path": str(hdr_path),
            "exr_path": str(exr_path),
            "hdr_ok": bool(capture_result["hdr_ok"]),
            "exr_ok": bool(capture_result["exr_ok"]),
        }
        if frame_index in (0, 100):
            frame_record["geometry_cache_actor_snapshot"] = _log_geometrycache_actor_snapshots(
                stage=f"frame_{frame_index:04d}",
                bound_actor=bound_actor,
            )
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
        frame_records.append(frame_record)
        if preview_status.get("success"):
            png_paths.append(png_path)

        unreal.log(
            f"[BEDLAM360][LS_VALIDATE][frame_{frame_index:04d}] "
            f"target_time={frame_record['time_seconds']:.6f} actual_time={actual_time} "
            f"manual_tick={frame_record['manual_tick']} is_playing={frame_record['is_playing']} "
            f"bound_actor={frame_record['bound_actor_label']}"
        )

    sorted_pngs = sorted(png_paths, key=lambda path: path.name)
    mp4_path = previews_dir / f"{asset_id}_levelsequence_validation.mp4"
    mp4_status = bedlam360_mini_validation._run_preview_mp4(
        sequence_name=f"{asset_id}_levelsequence_validation",
        png_paths=sorted_pngs,
        output_mp4_path=mp4_path,
        fps=fps,
    )
    diagnostics_path = output_root / "temporal_diagnostics.json"
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
    for frame_record in frame_records:
        diff_record = diff_lookup.get(frame_record.get("png_path")) or {}
        frame_record["pixel_diff_prev"] = diff_record.get("pixel_diff_prev")
        frame_record["center_crop_pixel_diff_prev"] = diff_record.get("center_crop_pixel_diff_prev")
        pose_json_path = metadata_dir / f"{Path(frame_record['png_path']).stem.replace('_erp', '')}_pose.json"
        pose_json_path.write_text(json.dumps(frame_record, indent=2), encoding="utf-8")

    duration = None
    num_cache_frames = None
    actual_times = []
    errors = []
    pixel_diffs = []
    center_crop_pixel_diffs = []
    for frame_record in frame_records:
        state = frame_record.get("component_state_after") or {}
        if duration is None:
            duration = state.get("duration_seconds")
        if num_cache_frames is None:
            num_cache_frames = state.get("num_frames")
        if frame_record.get("actual_time_seconds") is not None:
            actual_times.append(frame_record["actual_time_seconds"])
            errors.append(abs(frame_record["actual_time_seconds"] - frame_record["time_seconds"]))
        if frame_record.get("pixel_diff_prev") is not None:
            pixel_diffs.append(frame_record["pixel_diff_prev"])
        if frame_record.get("center_crop_pixel_diff_prev") is not None:
            center_crop_pixel_diffs.append(frame_record["center_crop_pixel_diff_prev"])

    mean_pixel_diff = None if not pixel_diffs else sum(pixel_diffs) / len(pixel_diffs)
    mean_center_crop_pixel_diff = (
        None if not center_crop_pixel_diffs else sum(center_crop_pixel_diffs) / len(center_crop_pixel_diffs)
    )
    summary_row = {
        "asset_id": asset_id,
        "camera_mode": camera_mode,
        "duration_seconds": duration,
        "num_frames": num_cache_frames,
        "max_actual_time_seconds": None if not actual_times else max(actual_times),
        "max_target_time_seconds": frame_records[-1]["time_seconds"] if frame_records else None,
        "max_time_error_seconds": None if not errors else max(errors),
        "mean_pixel_diff": mean_pixel_diff,
        "mean_center_crop_pixel_diff": mean_center_crop_pixel_diff,
        "movement_detected": bool(
            actual_times
            and max(actual_times) > 0.10
            and (
                (mean_center_crop_pixel_diff is not None and mean_center_crop_pixel_diff > 0.5)
                or (mean_pixel_diff is not None and mean_pixel_diff > 0.5)
            )
        ),
        "section_assignment_success": sequence_info["section_assignment"]["success"],
    }
    summary_csv_path = output_root / "geometrycache_levelsequence_summary.csv"
    _write_summary_csv(summary_row, summary_csv_path)

    frame_diff_pairs = {}
    if frame_records:
        for a, b in ((0, 20), (0, 100)):
            frame_diff_pairs[f"{a}_vs_{b}"] = _frame_pair_diff_from_records(frame_records, a, b)

    report = {
        "asset_id": asset_id,
        "asset_path": sequence_info["geometry_cache_asset"].get_path_name(),
        "sequence_asset_path": sequence_info["sequence_asset_path"],
        "resolved_asset": sequence_info["resolved_asset"],
        "section_assignment": sequence_info["section_assignment"],
        "level_sequence_summary": sequence_info["level_sequence_summary"],
        "initial_geometry_cache_actor_snapshot": initial_actor_snapshot,
        "duration_seconds": duration,
        "num_frames": num_cache_frames,
        "fps": fps,
        "frame_count": num_frames,
        "camera_mode": camera_mode,
        "movement_detected": summary_row["movement_detected"],
        "final_geometry_cache_actor_snapshot": _geometrycache_actor_snapshots(
            bound_actor=_get_bound_gc_actor(binding_id)
        ),
        "summary_csv_path": str(summary_csv_path),
        "mp4_status": mp4_status,
        "temporal_diagnostics_status": diagnostics_status,
        "mean_center_crop_pixel_diff": mean_center_crop_pixel_diff,
        "frame_diff_pairs": frame_diff_pairs,
        "frames": frame_records,
    }
    report_path = output_root / "geometrycache_levelsequence_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    unreal.log(f"[BEDLAM360][LS_VALIDATE] Wrote level-sequence validation report: {report_path}")

    try:
        unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    except Exception:
        pass
    return report


def validate_geometrycache_with_levelsequence_batch(
    asset_ids=None,
    num_frames=DEFAULT_NUM_FRAMES,
    fps=DEFAULT_FPS,
    actor_label=DEFAULT_CAPTURE_ACTOR_LABEL,
    output_root=DEFAULT_OUTPUT_ROOT,
    camera_mode=DEFAULT_CAMERA_MODE,
):
    asset_ids = list(asset_ids or _select_dynamic_asset_ids())
    batch_root = _reset_dir(output_root)
    batch_rows = []
    manifest_sequences = []
    for asset_id in asset_ids:
        unreal.log(f"[BEDLAM360][LS_VALIDATE] Starting validation for asset {asset_id} camera_mode={camera_mode}")
        report = validate_geometrycache_with_levelsequence(
            asset_id=asset_id,
            num_frames=num_frames,
            fps=fps,
            actor_label=actor_label,
            output_root=batch_root,
            camera_mode=camera_mode,
        )
        summary_path = Path(report["summary_csv_path"])
        with open(summary_path, "r", encoding="utf-8", newline="") as fp:
            row = next(csv.DictReader(fp))
        batch_rows.append(row)
        preview_mp4_path = None
        mp4_status = report.get("mp4_status") or {}
        if mp4_status.get("success"):
            preview_mp4_path = mp4_status.get("mp4_path")
        thumb = None
        if report.get("frames"):
            for frame in report["frames"]:
                png_path = frame.get("png_path")
                if png_path:
                    thumb = png_path
                    break
        manifest_sequences.append(
            {
                "asset_id": asset_id,
                "camera_mode": camera_mode,
                "frame_count": num_frames,
                "movement_detected": report.get("movement_detected"),
                "thumbnail_png_path": thumb,
                "preview_mp4_path": preview_mp4_path,
            }
        )

    batch_summary_csv_path = batch_root / "geometrycache_levelsequence_batch_summary.csv"
    _write_batch_summary_csv(batch_rows, batch_summary_csv_path)
    manifest = {
        "camera_mode": camera_mode,
        "fps": fps,
        "frame_count": num_frames,
        "asset_ids": asset_ids,
        "sequences": manifest_sequences,
        "batch_summary_csv_path": str(batch_summary_csv_path),
    }
    manifest_path = batch_root / "geometrycache_levelsequence_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    gallery_path = batch_root / "gallery.html"
    _write_batch_gallery(manifest_sequences, gallery_path)
    unreal.log(f"[BEDLAM360][LS_VALIDATE] Wrote batch manifest: {manifest_path}")
    return manifest


def run_levelsequence_regression_ab_validation(
    asset_id=DEFAULT_ASSET_ID,
    num_frames=120,
    fps=30,
    actor_label=DEFAULT_CAPTURE_ACTOR_LABEL,
    output_root=DEFAULT_OUTPUT_ROOT,
):
    ab_root = _reset_dir(Path(output_root) / "ab_regression")
    legacy_root = ab_root / "legacy"
    batch_root = ab_root / "batch"

    legacy_report = validate_geometrycache_with_levelsequence(
        asset_id=asset_id,
        num_frames=num_frames,
        fps=fps,
        actor_label=actor_label,
        output_root=legacy_root,
        camera_mode="fixed",
    )
    validate_geometrycache_with_levelsequence_batch(
        asset_ids=[asset_id],
        num_frames=num_frames,
        fps=fps,
        actor_label=actor_label,
        output_root=batch_root,
        camera_mode="fixed",
    )
    batch_report_path = batch_root / asset_id / "geometrycache_levelsequence_validation.json"
    batch_report = json.loads(batch_report_path.read_text(encoding="utf-8"))

    comparison = {
        "asset_id": asset_id,
        "num_frames": num_frames,
        "fps": fps,
        "legacy_report_path": str(legacy_root / asset_id / "geometrycache_levelsequence_validation.json"),
        "batch_report_path": str(batch_report_path),
        "visible_actors": {
            "legacy_initial": legacy_report.get("initial_geometry_cache_actor_snapshot"),
            "legacy_final": legacy_report.get("final_geometry_cache_actor_snapshot"),
            "batch_initial": batch_report.get("initial_geometry_cache_actor_snapshot"),
            "batch_final": batch_report.get("final_geometry_cache_actor_snapshot"),
        },
        "level_sequence_bindings": {
            "legacy": legacy_report.get("level_sequence_summary"),
            "batch": batch_report.get("level_sequence_summary"),
        },
        "sequencer_evaluation": {
            "legacy_frame0": legacy_report.get("frames", [{}])[0] if legacy_report.get("frames") else None,
            "legacy_frame20": legacy_report.get("frames", [{}])[20] if len(legacy_report.get("frames", [])) > 20 else None,
            "legacy_frame100": legacy_report.get("frames", [{}])[100] if len(legacy_report.get("frames", [])) > 100 else None,
            "batch_frame0": batch_report.get("frames", [{}])[0] if batch_report.get("frames") else None,
            "batch_frame20": batch_report.get("frames", [{}])[20] if len(batch_report.get("frames", [])) > 20 else None,
            "batch_frame100": batch_report.get("frames", [{}])[100] if len(batch_report.get("frames", [])) > 100 else None,
        },
        "output_frame_diffs": {
            "legacy": legacy_report.get("frame_diff_pairs"),
            "batch": batch_report.get("frame_diff_pairs"),
        },
    }
    comparison_path = ab_root / "geometrycache_levelsequence_ab_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    unreal.log(f"[BEDLAM360][LS_VALIDATE] Wrote A/B comparison report: {comparison_path}")
    return comparison


if __name__ == "__main__":
    run_levelsequence_regression_ab_validation()
