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
import reconstruct_full_bedlam_scene
import reconstruct_one_bedlam_body

bedlam360_mini_validation = importlib.reload(bedlam360_mini_validation)
capture_scene_cube = importlib.reload(capture_scene_cube)
reconstruct_full_bedlam_scene = importlib.reload(reconstruct_full_bedlam_scene)
reconstruct_one_bedlam_body = importlib.reload(reconstruct_one_bedlam_body)


DEFAULT_POSE_JSON = Path(
    "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_mini/metadata/mini_seq_0003/mini_seq_0003_frame_0005_pose.json"
)
DEFAULT_EXPORT_ROOT = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_debug")
DEFAULT_ACTOR_LABEL = "SceneCaptureCube"
FLOAT_TOLERANCE = 1e-3


def _load_json(json_path):
    json_path = Path(json_path)
    if not json_path.is_file():
        raise RuntimeError(f"JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _infer_sequence_name(frame_data):
    sequence_name = frame_data.get("sequence_name")
    if sequence_name:
        return sequence_name
    frame_name = frame_data.get("frame_name", "")
    if "_frame_" in frame_name:
        return frame_name.split("_frame_", 1)[0]
    raise RuntimeError("Could not infer sequence name from pose JSON.")


def _load_sequence_json_for_frame(pose_json_path, frame_data):
    pose_json_path = Path(pose_json_path)
    sequence_name = _infer_sequence_name(frame_data)
    sequence_json_path = pose_json_path.parent / f"{sequence_name}_sequence.json"
    if not sequence_json_path.is_file():
        raise RuntimeError(f"Sequence JSON not found beside pose JSON: {sequence_json_path}")
    sequence_data = _load_json(sequence_json_path)
    return sequence_json_path, sequence_data


def _load_manifest_for_sequence(sequence_json_path):
    metadata_root = sequence_json_path.parent.parent
    manifest_path = metadata_root.parent / "manifest.json"
    if manifest_path.is_file():
        return manifest_path, _load_json(manifest_path)
    return None, None


def _find_frame_record(sequence_data, frame_name):
    for frame in sequence_data.get("frames", []):
        if frame.get("frame_name") == frame_name:
            return frame
    raise RuntimeError(f"Frame '{frame_name}' not found in sequence JSON.")


def _set_actor_label(actor, label):
    if label:
        actor.set_actor_label(label)


def _spawn_bodies_from_sequence_metadata(sequence_data):
    reconstruct_full_bedlam_scene.clear_existing_bedlam_bodies()

    spawned = []
    for body_entry in sequence_data.get("bodies", []):
        body_pose = dict(body_entry["body_pose"])
        resolved_asset = dict(body_entry["resolved_asset"])
        actor = reconstruct_one_bedlam_body.spawn_body_actor(body_pose, resolved_asset)
        actor_label = body_entry.get("actor_label") or f"{reconstruct_full_bedlam_scene.ACTOR_LABEL_PREFIX}{body_pose['asset_id']}"
        _set_actor_label(actor, actor_label)

        geometry_cache_component = None
        if hasattr(actor, "get_geometry_cache_component"):
            geometry_cache_component = actor.get_geometry_cache_component()

        spawned.append(
            {
                "body_pose": body_pose,
                "resolved_asset": resolved_asset,
                "actor_label": actor_label,
                "actor": actor,
                "geometry_cache_component": geometry_cache_component,
            }
        )
    return spawned


def _apply_hdri_state(source_frame_data):
    hdri_status = source_frame_data.get("hdri_status") or {}
    hdri_name = source_frame_data.get("hdri_name")
    if hdri_status.get("applied"):
        return bedlam360_mini_validation._apply_hdri_if_possible(hdri_name)
    return {
        "requested_hdri": hdri_name,
        "applied": False,
        "reason": hdri_status.get("reason", "not_requested_in_source"),
        "asset_path": hdri_status.get("asset_path"),
    }


def _texture_target_info(texture_target):
    info = {
        "name": texture_target.get_name(),
        "path": texture_target.get_path_name(),
    }
    for key in ("size_x", "size_y"):
        try:
            info[key] = int(texture_target.get_editor_property(key))
        except Exception:
            info[key] = None
    return info


def _get_component_property(component, property_names):
    for property_name in property_names:
        try:
            value = component.get_editor_property(property_name)
            return property_name, value
        except Exception:
            continue
    return None, None


def _actor_pose(actor):
    location = actor.get_actor_location()
    rotation = actor.get_actor_rotation()
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
        "roll": float(rotation.roll),
    }


def _vector_to_dict(vector):
    return {"x": float(vector.x), "y": float(vector.y), "z": float(vector.z)}


def _actor_bounds_diagnostics(actor, camera_pose):
    origin, extent = actor.get_actor_bounds(False)
    origin_dict = _vector_to_dict(origin)
    extent_dict = _vector_to_dict(extent)
    bounds_min = {
        "x": origin_dict["x"] - extent_dict["x"],
        "y": origin_dict["y"] - extent_dict["y"],
        "z": origin_dict["z"] - extent_dict["z"],
    }
    bounds_max = {
        "x": origin_dict["x"] + extent_dict["x"],
        "y": origin_dict["y"] + extent_dict["y"],
        "z": origin_dict["z"] + extent_dict["z"],
    }

    dx = origin_dict["x"] - camera_pose["x"]
    dy = origin_dict["y"] - camera_pose["y"]
    dz = origin_dict["z"] - camera_pose["z"]
    center_distance_cm = math.sqrt(dx * dx + dy * dy + dz * dz)
    bounds_radius_cm = math.sqrt(
        extent_dict["x"] * extent_dict["x"] + extent_dict["y"] * extent_dict["y"] + extent_dict["z"] * extent_dict["z"]
    )
    surface_clearance_cm = center_distance_cm - bounds_radius_cm

    inside_bounds = (
        bounds_min["x"] <= camera_pose["x"] <= bounds_max["x"]
        and bounds_min["y"] <= camera_pose["y"] <= bounds_max["y"]
        and bounds_min["z"] <= camera_pose["z"] <= bounds_max["z"]
    )

    return {
        "origin_cm": origin_dict,
        "extent_cm": extent_dict,
        "bounds_min_cm": bounds_min,
        "bounds_max_cm": bounds_max,
        "center_distance_cm": center_distance_cm,
        "center_distance_m": center_distance_cm / 100.0,
        "bounds_radius_cm": bounds_radius_cm,
        "surface_clearance_cm": surface_clearance_cm,
        "surface_clearance_m": surface_clearance_cm / 100.0,
        "camera_inside_bounds_aabb": inside_bounds,
        "vector_from_camera_cm": {"x": dx, "y": dy, "z": dz},
    }


def _cube_face_for_vector(vector):
    x = vector["x"]
    y = vector["y"]
    z = vector["z"]
    ax = abs(x)
    ay = abs(y)
    az = abs(z)
    if ax >= ay and ax >= az:
        return "px" if x >= 0.0 else "nx"
    if ay >= ax and ay >= az:
        return "py" if y >= 0.0 else "ny"
    return "pz" if z >= 0.0 else "nz"


def _bounds_corners(bounds_min, bounds_max):
    corners = []
    for x in (bounds_min["x"], bounds_max["x"]):
        for y in (bounds_min["y"], bounds_max["y"]):
            for z in (bounds_min["z"], bounds_max["z"]):
                corners.append({"x": x, "y": y, "z": z})
    return corners


def _cube_face_visibility_estimate(camera_pose, bounds_diag):
    center_face = _cube_face_for_vector(bounds_diag["vector_from_camera_cm"])
    touched_faces = set()
    for corner in _bounds_corners(bounds_diag["bounds_min_cm"], bounds_diag["bounds_max_cm"]):
        corner_vector = {
            "x": corner["x"] - camera_pose["x"],
            "y": corner["y"] - camera_pose["y"],
            "z": corner["z"] - camera_pose["z"],
        }
        touched_faces.add(_cube_face_for_vector(corner_vector))

    nx_visible = "nx" in touched_faces or center_face == "nx"
    return {
        "center_face": center_face,
        "touched_faces": sorted(touched_faces),
        "nx_face_visible_estimate": nx_visible,
    }


def _near_clip_diagnostics(component):
    property_name, property_value = _get_component_property(
        component,
        (
            "near_clip_plane",
            "custom_near_clipping_plane",
            "override_custom_near_clipping_plane",
        ),
    )

    global_near_clip = None
    try:
        global_near_clip = float(unreal.SystemLibrary.get_console_variable_float_value("NearClipPlane"))
    except Exception:
        global_near_clip = None

    return {
        "component_property_name": property_name,
        "component_property_value": property_value,
        "global_console_value": global_near_clip,
    }


def _compare_float(source_value, debug_value, tolerance=FLOAT_TOLERANCE):
    if source_value is None and debug_value is None:
        return True
    if source_value is None or debug_value is None:
        return False
    return abs(float(source_value) - float(debug_value)) <= tolerance


def _compare_pose_dict(source_pose, debug_pose, prefix, diffs):
    for key in ("x", "y", "z", "pitch", "yaw", "roll"):
        if not _compare_float(source_pose.get(key), debug_pose.get(key)):
            diffs.append(
                {
                    "field": f"{prefix}.{key}",
                    "source": source_pose.get(key),
                    "debug": debug_pose.get(key),
                }
            )


def _compare_value(field_name, source_value, debug_value, diffs):
    if source_value != debug_value:
        diffs.append({"field": field_name, "source": source_value, "debug": debug_value})


def _replay_animation_history(sequence_data, spawned_bodies, target_frame_name, smooth_time_sampling):
    previous_times = {}
    matched_frame = None
    actual_states = None

    for frame in sequence_data.get("frames", []):
        frame_states = frame.get("animation_states", [])
        if len(frame_states) != len(spawned_bodies):
            raise RuntimeError(
                f"Animation-state count mismatch for frame {frame.get('frame_name')}: "
                f"{len(frame_states)} states vs {len(spawned_bodies)} bodies"
            )

        current_actual_states = []
        for spawned_body, source_state in zip(spawned_bodies, frame_states):
            sample_time = source_state.get("sample_time_seconds")
            previous_sample_time = previous_times.get(spawned_body["actor_label"])
            actual_state = bedlam360_mini_validation._sample_motion_state_with_options(
                spawned_body,
                sample_time,
                previous_sample_time_seconds=previous_sample_time,
                smooth_time_sampling=smooth_time_sampling,
            )
            previous_times[spawned_body["actor_label"]] = actual_state.get("sample_time_seconds")
            current_actual_states.append(actual_state)

        bedlam360_mini_validation._invalidate_editor_viewports()
        if frame.get("frame_name") == target_frame_name:
            matched_frame = frame
            actual_states = current_actual_states
            break

    if matched_frame is None or actual_states is None:
        raise RuntimeError(f"Could not replay animation history up to frame '{target_frame_name}'.")
    return matched_frame, actual_states


def _build_body_state(spawned_bodies):
    body_state = []
    for spawned_body in spawned_bodies:
        actor = spawned_body["actor"]
        actor_pose = _actor_pose(actor)
        body_state.append(
            {
                "asset_id": spawned_body["body_pose"]["asset_id"],
                "actor_label": spawned_body["actor_label"],
                "resolved_asset_id": spawned_body["resolved_asset"].get("resolved_asset_id"),
                "transform_cm_deg": actor_pose,
                "source_body_pose": spawned_body["body_pose"],
            }
        )
    return body_state


def _build_frame_diagnostics(actor, component, spawned_bodies):
    camera_pose = _actor_pose(actor)
    near_clip = _near_clip_diagnostics(component)
    actor_reports = []
    nx_visible = []
    nx_missing_candidates = []

    for spawned_body in spawned_bodies:
        body_diag = _actor_bounds_diagnostics(spawned_body["actor"], camera_pose)
        face_diag = _cube_face_visibility_estimate(camera_pose, body_diag)
        report = {
            "asset_id": spawned_body["body_pose"]["asset_id"],
            "actor_label": spawned_body["actor_label"],
            "actor_transform_cm_deg": _actor_pose(spawned_body["actor"]),
            "bounds": body_diag,
            "cube_face_estimate": face_diag,
        }
        actor_reports.append(report)
        if face_diag["nx_face_visible_estimate"]:
            nx_visible.append(spawned_body["body_pose"]["asset_id"])
            if body_diag["camera_inside_bounds_aabb"] or body_diag["surface_clearance_cm"] <= 0.0:
                nx_missing_candidates.append(
                    {
                        "asset_id": spawned_body["body_pose"]["asset_id"],
                        "reason": "camera_intersects_bounds",
                    }
                )
            elif near_clip["global_console_value"] is not None and body_diag["surface_clearance_cm"] < float(near_clip["global_console_value"]):
                nx_missing_candidates.append(
                    {
                        "asset_id": spawned_body["body_pose"]["asset_id"],
                        "reason": "surface_clearance_below_near_clip",
                    }
                )

    return {
        "camera_position_cm": {
            "x": camera_pose["x"],
            "y": camera_pose["y"],
            "z": camera_pose["z"],
        },
        "camera_rotation_deg": {
            "pitch": camera_pose["pitch"],
            "yaw": camera_pose["yaw"],
            "roll": camera_pose["roll"],
        },
        "near_clip": near_clip,
        "actors": actor_reports,
        "nx_face": {
            "visible_actor_asset_ids_estimate": nx_visible,
            "missing_actor_candidates_estimate": nx_missing_candidates,
        },
    }


def _build_comparison_report(
    source_frame_data,
    source_sequence_data,
    source_manifest,
    debug_frame_data,
    debug_sequence_data,
):
    diffs = []

    _compare_pose_dict(source_frame_data.get("camera_pose_cm_deg", {}), debug_frame_data.get("camera_pose_cm_deg", {}), "camera_pose_cm_deg", diffs)
    _compare_value("body_asset_ids", source_frame_data.get("body_asset_ids"), debug_frame_data.get("body_asset_ids"), diffs)
    _compare_value("distance_band", source_frame_data.get("distance_band"), debug_frame_data.get("distance_band"), diffs)
    _compare_value("camera_mode", source_frame_data.get("camera_mode"), debug_frame_data.get("camera_mode"), diffs)
    _compare_value("hdri_name", source_frame_data.get("hdri_name"), debug_frame_data.get("hdri_name"), diffs)
    _compare_value(
        "hdri_status.applied",
        (source_frame_data.get("hdri_status") or {}).get("applied"),
        (debug_frame_data.get("hdri_status") or {}).get("applied"),
        diffs,
    )
    _compare_value(
        "hdri_status.reason",
        (source_frame_data.get("hdri_status") or {}).get("reason"),
        (debug_frame_data.get("hdri_status") or {}).get("reason"),
        diffs,
    )
    _compare_value(
        "texture_target.name",
        source_manifest.get("texture_render_target_cube") if source_manifest else None,
        (debug_frame_data.get("texture_target") or {}).get("name"),
        diffs,
    )
    _compare_value(
        "capture_stabilized",
        source_frame_data.get("capture_stabilized"),
        debug_frame_data.get("capture_stabilized"),
        diffs,
    )
    _compare_value("warmup_ticks", source_frame_data.get("warmup_ticks"), debug_frame_data.get("warmup_ticks"), diffs)
    _compare_value("discard_captures", source_frame_data.get("discard_captures"), debug_frame_data.get("discard_captures"), diffs)
    _compare_value(
        "preload_sequence_used",
        source_frame_data.get("preload_sequence_used"),
        debug_frame_data.get("preload_sequence_used"),
        diffs,
    )
    _compare_value(
        "smooth_time_sampling",
        source_frame_data.get("smooth_time_sampling"),
        debug_frame_data.get("smooth_time_sampling"),
        diffs,
    )
    _compare_value(
        "geometry_cache_actor_count",
        source_frame_data.get("geometry_cache_actor_count"),
        debug_frame_data.get("geometry_cache_actor_count"),
        diffs,
    )

    source_animation_states = source_frame_data.get("animation_states", [])
    debug_animation_states = debug_frame_data.get("animation_states", [])
    if len(source_animation_states) != len(debug_animation_states):
        diffs.append(
            {
                "field": "animation_states.length",
                "source": len(source_animation_states),
                "debug": len(debug_animation_states),
            }
        )
    for index, (source_state, debug_state) in enumerate(zip(source_animation_states, debug_animation_states)):
        for key in ("sample_time_seconds", "sample_frame_index", "duration_seconds", "number_of_frames"):
            source_value = source_state.get(key)
            debug_value = debug_state.get(key)
            field_name = f"animation_states[{index}].{key}"
            if isinstance(source_value, (float, int)) or isinstance(debug_value, (float, int)):
                if not _compare_float(source_value, debug_value):
                    diffs.append({"field": field_name, "source": source_value, "debug": debug_value})
            elif source_value != debug_value:
                diffs.append({"field": field_name, "source": source_value, "debug": debug_value})

    source_bodies = source_sequence_data.get("bodies", [])
    debug_bodies = debug_sequence_data.get("bodies", [])
    if len(source_bodies) != len(debug_bodies):
        diffs.append({"field": "bodies.length", "source": len(source_bodies), "debug": len(debug_bodies)})
    for source_body, debug_body in zip(source_bodies, debug_bodies):
        source_pose = source_body.get("body_pose", {})
        debug_pose = debug_body.get("body_pose", {})
        source_asset = source_pose.get("asset_id")
        debug_asset = debug_pose.get("asset_id")
        if source_asset != debug_asset:
            diffs.append({"field": "bodies.asset_id", "source": source_asset, "debug": debug_asset})
        _compare_pose_dict(source_pose, debug_pose, f"bodies[{source_asset or 'unknown'}].body_pose", diffs)
        _compare_value(
            f"bodies[{source_asset or 'unknown'}].resolved_asset_id",
            (source_body.get("resolved_asset") or {}).get("resolved_asset_id"),
            (debug_body.get("resolved_asset") or {}).get("resolved_asset_id"),
            diffs,
        )

    return {
        "match": len(diffs) == 0,
        "differences": diffs,
        "source": {
            "frame": source_frame_data,
            "sequence": source_sequence_data,
            "manifest": source_manifest,
        },
        "debug": {
            "frame": debug_frame_data,
            "sequence": debug_sequence_data,
        },
    }


def _log_comparison_report(comparison_report):
    differences = comparison_report.get("differences", [])
    if not differences:
        unreal.log("[BEDLAM360] Debug reconstruction metadata matches source metadata within tolerance.")
        return

    unreal.log_warning(f"[BEDLAM360] Debug reconstruction found {len(differences)} metadata differences:")
    for diff in differences:
        unreal.log_warning(
            f"[BEDLAM360]   {diff['field']}: source={diff.get('source')} debug={diff.get('debug')}"
        )


def debug_export_frame(
    pose_json_path=DEFAULT_POSE_JSON,
    export_root=DEFAULT_EXPORT_ROOT,
    actor_label=DEFAULT_ACTOR_LABEL,
):
    frame_data = _load_json(pose_json_path)
    sequence_name = _infer_sequence_name(frame_data)
    frame_name = frame_data["frame_name"]

    sequence_json_path, sequence_data = _load_sequence_json_for_frame(pose_json_path, frame_data)
    manifest_path, manifest_data = _load_manifest_for_sequence(sequence_json_path)
    source_frame_data = _find_frame_record(sequence_data, frame_name)

    spawned_bodies = _spawn_bodies_from_sequence_metadata(sequence_data)
    smooth_time_sampling = bool(source_frame_data.get("smooth_time_sampling", False))
    replayed_frame_data, actual_animation_states = _replay_animation_history(
        sequence_data,
        spawned_bodies,
        frame_name,
        smooth_time_sampling=smooth_time_sampling,
    )

    actor = capture_scene_cube.find_scene_capture_cube(actor_label)
    component = capture_scene_cube.get_capture_component(actor)
    texture_target = capture_scene_cube.get_texture_target(component)
    export_lib = unreal.BEDLAM360ExportLibrary

    camera = source_frame_data["camera_pose_cm_deg"]
    capture_scene_cube.set_actor_pose(
        actor,
        float(camera["x"]),
        float(camera["y"]),
        float(camera["z"]),
        float(camera["pitch"]),
        float(camera["yaw"]),
        float(camera["roll"]),
    )

    hdri_result = _apply_hdri_state(source_frame_data)

    export_dir = Path(export_root) / sequence_name / frame_name
    export_dir.mkdir(parents=True, exist_ok=True)
    hdr_path = export_dir / f"{frame_name}_erp.hdr"
    exr_path = export_dir / f"{frame_name}_erp.exr"
    faces_dir = export_dir / "faces"

    warmup_ticks = int(source_frame_data.get("warmup_ticks", bedlam360_mini_validation.DEFAULT_WARMUP_TICKS))
    discard_captures = int(source_frame_data.get("discard_captures", bedlam360_mini_validation.DEFAULT_DISCARD_CAPTURES))
    capture_result = bedlam360_mini_validation.stabilized_capture_and_export(
        actor=actor,
        component=component,
        texture_target=texture_target,
        export_lib=export_lib,
        frame_name=frame_name,
        hdr_path=hdr_path,
        exr_path=exr_path,
        faces_dir=faces_dir,
        warmup_ticks=warmup_ticks,
        discard_captures=discard_captures,
    )

    texture_info = _texture_target_info(texture_target)
    body_state = _build_body_state(spawned_bodies)
    geometry_cache_actor_count = sum(1 for body in spawned_bodies if body["geometry_cache_component"] is not None)
    frame_diagnostics = _build_frame_diagnostics(actor, component, spawned_bodies)

    debug_frame_data = {
        "sequence_name": sequence_name,
        "frame_name": frame_name,
        "frame_sample_index": source_frame_data.get("frame_sample_index"),
        "distance_band": source_frame_data.get("distance_band"),
        "camera_mode": source_frame_data.get("camera_mode"),
        "hdri_name": source_frame_data.get("hdri_name"),
        "hdri_status": hdri_result,
        "camera_pose_cm_deg": _actor_pose(actor),
        "body_asset_ids": [body["asset_id"] for body in body_state],
        "animation_states": actual_animation_states,
        "capture_stabilized": True,
        "warmup_ticks": warmup_ticks,
        "discard_captures": discard_captures,
        "preload_sequence_used": bool(source_frame_data.get("preload_sequence_used", False)),
        "smooth_time_sampling": smooth_time_sampling,
        "geometry_cache_actor_count": geometry_cache_actor_count,
        "texture_target": texture_info,
        "hdr_path": str(hdr_path),
        "exr_path": str(exr_path),
        "faces_dir": str(faces_dir),
        "hdr_ok": capture_result["hdr_ok"],
        "exr_ok": capture_result["exr_ok"],
        "faces_ok": capture_result["faces_ok"],
        "source_pose_json": str(pose_json_path),
        "source_sequence_json": str(sequence_json_path),
        "source_manifest_json": None if manifest_path is None else str(manifest_path),
        "frame_diagnostics": frame_diagnostics,
    }

    debug_sequence_data = {
        "sequence_name": sequence_name,
        "hdri_name": source_frame_data.get("hdri_name"),
        "hdri_status": hdri_result,
        "capture_stabilization": {
            "capture_stabilized": True,
            "warmup_ticks": warmup_ticks,
            "discard_captures": discard_captures,
            "preload_sequence_used": bool(source_frame_data.get("preload_sequence_used", False)),
            "smooth_time_sampling": smooth_time_sampling,
        },
        "bodies": [
            {
                "asset_id": body["asset_id"],
                "actor_label": body["actor_label"],
                "body_pose": body["source_body_pose"],
                "resolved_asset": next(
                    spawned_body["resolved_asset"]
                    for spawned_body in spawned_bodies
                    if spawned_body["actor_label"] == body["actor_label"]
                ),
                "actual_actor_pose_cm_deg": body["transform_cm_deg"],
            }
            for body in body_state
        ],
        "frames": [debug_frame_data],
        "source_frame_reference": replayed_frame_data.get("pose_json_path"),
    }

    comparison_report = _build_comparison_report(
        source_frame_data=source_frame_data,
        source_sequence_data={"sequence_name": sequence_data.get("sequence_name"), "bodies": sequence_data.get("bodies", [])},
        source_manifest=manifest_data,
        debug_frame_data=debug_frame_data,
        debug_sequence_data={"sequence_name": debug_sequence_data.get("sequence_name"), "bodies": debug_sequence_data.get("bodies", [])},
    )

    frame_metadata_path = export_dir / "debug_frame_metadata.json"
    sequence_metadata_path = export_dir / "debug_sequence_metadata.json"
    comparison_report_path = export_dir / "comparison_report.json"
    diagnostics_path = export_dir / "frame_diagnostics.json"

    with open(frame_metadata_path, "w", encoding="utf-8") as fp:
        json.dump(debug_frame_data, fp, indent=2)
    with open(sequence_metadata_path, "w", encoding="utf-8") as fp:
        json.dump(debug_sequence_data, fp, indent=2)
    with open(comparison_report_path, "w", encoding="utf-8") as fp:
        json.dump(comparison_report, fp, indent=2)
    with open(diagnostics_path, "w", encoding="utf-8") as fp:
        json.dump(frame_diagnostics, fp, indent=2)

    _log_comparison_report(comparison_report)
    unreal.log(f"[BEDLAM360] Camera position: {frame_diagnostics['camera_position_cm']}")
    unreal.log(f"[BEDLAM360] Camera rotation: {frame_diagnostics['camera_rotation_deg']}")
    unreal.log(f"[BEDLAM360] Near clip diagnostics: {frame_diagnostics['near_clip']}")
    for actor_report in frame_diagnostics["actors"]:
        unreal.log(
            f"[BEDLAM360] Actor {actor_report['asset_id']} bounds distance_m="
            f"{actor_report['bounds']['center_distance_m']:.3f} surface_clearance_m="
            f"{actor_report['bounds']['surface_clearance_m']:.3f} inside_bounds="
            f"{actor_report['bounds']['camera_inside_bounds_aabb']} nx_visible="
            f"{actor_report['cube_face_estimate']['nx_face_visible_estimate']}"
        )
    unreal.log(f"[BEDLAM360] nx face visibility estimate: {frame_diagnostics['nx_face']}")
    unreal.log(
        f"[BEDLAM360] Debug frame export complete: frame={frame_name} hdr_ok={capture_result['hdr_ok']} "
        f"exr_ok={capture_result['exr_ok']} faces_ok={capture_result['faces_ok']}"
    )
    unreal.log(f"[BEDLAM360] Debug output folder: {export_dir}")
    return comparison_report_path


if __name__ == "__main__":
    debug_export_frame()
