import importlib
import json
import math
import sys
from pathlib import Path

import unreal


SCENE_METADATA_PATH = Path(
    "/media/mathis/PANO/infinigen/outputs/indoors/human_spawn_poc/scene_collision_metadata.json"
)
SPAWN_POSE_INDEX = 0
BEDLAM_ASSET_ID = "it_4001_XL_2400"
SAMPLE_TIME_SECONDS = 0.5
DEBUG_FOLDER = "ValidatedBedlamMotion"
SEQUENCE_NAME = "infinigen_validated_motion"
FRAME_NAME = "spawn_pose_0"

# T_infinigen_to_bedlam
SCENE_ROOT_OFFSET_CM = unreal.Vector(0.0, 0.0, 0.0)
SCENE_ROOT_YAW_DEG = 0.0

BEDLAM_PYTHON_DIR = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Python")
if str(BEDLAM_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(BEDLAM_PYTHON_DIR))

import reconstruct_one_bedlam_body  # noqa: E402
import bedlam360_mini_validation  # noqa: E402

reconstruct_one_bedlam_body = importlib.reload(reconstruct_one_bedlam_body)
bedlam360_mini_validation = importlib.reload(bedlam360_mini_validation)


def log_info(message):
    unreal.log(f"[BEDLAM_INFINIGEN_BRIDGE] {message}")


def log_warning(message):
    unreal.log_warning(f"[BEDLAM_INFINIGEN_BRIDGE] {message}")


def load_spawn_pose():
    if not SCENE_METADATA_PATH.exists():
        raise FileNotFoundError(f"Scene metadata not found: {SCENE_METADATA_PATH}")
    with SCENE_METADATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    poses = data.get("spawn_poses", [])
    if not poses:
        raise RuntimeError(f"No spawn poses found in {SCENE_METADATA_PATH}")
    if SPAWN_POSE_INDEX >= len(poses):
        raise RuntimeError(
            f"Requested spawn pose index {SPAWN_POSE_INDEX}, but only {len(poses)} poses exist"
        )
    return poses[SPAWN_POSE_INDEX]


def _rotate_xy(vector, yaw_degrees):
    yaw_radians = math.radians(float(yaw_degrees))
    c = math.cos(yaw_radians)
    s = math.sin(yaw_radians)
    return unreal.Vector(
        vector.x * c - vector.y * s,
        vector.x * s + vector.y * c,
        vector.z,
    )


def transform_infinigen_position_to_bedlam(position_xyz):
    x, y, z = position_xyz
    mapped = unreal.Vector(100.0 * x, -100.0 * y, 100.0 * z)
    rotated = _rotate_xy(mapped, SCENE_ROOT_YAW_DEG)
    return unreal.Vector(
        rotated.x + SCENE_ROOT_OFFSET_CM.x,
        rotated.y + SCENE_ROOT_OFFSET_CM.y,
        rotated.z + SCENE_ROOT_OFFSET_CM.z,
    )


def transform_infinigen_yaw_to_bedlam(yaw_radians):
    semantic_yaw_deg = -math.degrees(yaw_radians or 0.0)
    return semantic_yaw_deg + SCENE_ROOT_YAW_DEG


def destroy_existing_bridge_actors(actor_subsystem):
    for actor in actor_subsystem.get_all_level_actors():
        label = actor.get_actor_label()
        if label.startswith("VALIDATED_BEDLAM_"):
            actor_subsystem.destroy_actor(actor)


def _vector_to_dict(v):
    if v is None:
        return None
    return {"x": float(v.x), "y": float(v.y), "z": float(v.z)}


def _bounds_to_dict(bounds):
    if bounds is None:
        return None
    return {
        "center": _vector_to_dict(bounds.get("center")),
        "extent": _vector_to_dict(bounds.get("extent")),
        "size": _vector_to_dict(bounds.get("size")),
        "min": _vector_to_dict(bounds.get("min")),
        "max": _vector_to_dict(bounds.get("max")),
    }


def _actor_bounds(actor, only_colliding=False):
    origin, extent = actor.get_actor_bounds(only_colliding)
    return {
        "center": origin,
        "extent": extent,
        "size": unreal.Vector(extent.x * 2.0, extent.y * 2.0, extent.z * 2.0),
        "min": unreal.Vector(origin.x - extent.x, origin.y - extent.y, origin.z - extent.z),
        "max": unreal.Vector(origin.x + extent.x, origin.y + extent.y, origin.z + extent.z),
    }


def log_actor_transform(actor, label):
    if actor is None:
        log_warning(f"{label} transform unavailable: actor is None")
        return None
    location = actor.get_actor_location()
    rotation = actor.get_actor_rotation()
    scale = actor.get_actor_scale3d()
    payload = {
        "location_cm": _vector_to_dict(location),
        "rotation_deg": {
            "pitch": float(rotation.pitch),
            "yaw": float(rotation.yaw),
            "roll": float(rotation.roll),
        },
        "scale": _vector_to_dict(scale),
    }
    log_info(f"{label} transform: {json.dumps(payload)}")
    return payload


def log_actor_bounds(actor, label):
    if actor is None:
        log_warning(f"{label} bounds unavailable: actor is None")
        return None
    bounds = _actor_bounds(actor, False)
    payload = _bounds_to_dict(bounds)
    log_info(f"{label} bounds: {json.dumps(payload)}")
    return payload


def build_body_pose(spawn_location, final_bedlam_yaw_deg):
    return {
        "asset_id": BEDLAM_ASSET_ID,
        "x": float(spawn_location.x),
        "y": float(spawn_location.y),
        "z": float(spawn_location.z),
        "yaw": float(final_bedlam_yaw_deg),
        "pitch": 0.0,
        "roll": 0.0,
        "start_frame": 1,
        "texture_body": None,
        "texture_clothing": None,
        "texture_clothing_overlay": None,
        "hair": None,
        "haircolor": None,
        "shoe": None,
        "shoe_offset": None,
    }


def spawn_and_play_validated_motion():
    spawn_pose = load_spawn_pose()
    spawn_location = transform_infinigen_position_to_bedlam(
        spawn_pose["position_xyz"]
    )
    final_bedlam_yaw_deg = transform_infinigen_yaw_to_bedlam(spawn_pose.get("yaw"))

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    destroy_existing_bridge_actors(actor_subsystem)

    log_info(
        f"T_infinigen_to_bedlam: scene_root_offset_cm=({SCENE_ROOT_OFFSET_CM.x:.2f}, {SCENE_ROOT_OFFSET_CM.y:.2f}, {SCENE_ROOT_OFFSET_CM.z:.2f}) "
        f"scene_root_yaw_deg={SCENE_ROOT_YAW_DEG:.2f}"
    )
    log_info(
        f"Using validated spawn pose index {SPAWN_POSE_INDEX}: final_bedlam_location=({spawn_location.x:.2f}, {spawn_location.y:.2f}, {spawn_location.z:.2f}) "
        f"final_bedlam_yaw={final_bedlam_yaw_deg:.2f}"
    )
    log_info(
        f"Spawn metadata: room={spawn_pose.get('room')} target={spawn_pose.get('target_object')} activity={spawn_pose.get('activity_hint')}"
    )

    body_pose = build_body_pose(spawn_location, final_bedlam_yaw_deg)
    resolved_asset = reconstruct_one_bedlam_body.resolve_body_asset(BEDLAM_ASSET_ID)
    actor, appearance_metadata = reconstruct_one_bedlam_body.spawn_body_actor(
        body_pose, resolved_asset, return_metadata=True
    )
    actor.set_actor_label("VALIDATED_BEDLAM_BODY_0")
    actor.set_folder_path(DEBUG_FOLDER)

    gc_component = (
        actor.get_geometry_cache_component()
        if hasattr(actor, "get_geometry_cache_component")
        else None
    )
    if gc_component is None:
        raise RuntimeError("Spawned BEDLAM actor has no GeometryCacheComponent")

    spawned_body = {
        "actor_label": actor.get_actor_label(),
        "body_pose": body_pose,
        "resolved_asset": resolved_asset,
        "appearance_metadata": appearance_metadata,
        "actor": actor,
        "geometry_cache_component": gc_component,
    }

    target_state = bedlam360_mini_validation._calculate_motion_state_metadata(
        gc_component, SAMPLE_TIME_SECONDS
    )
    target_frame_index = int(target_state.get("sample_frame_index") or 0)

    sequence_runtime = bedlam360_mini_validation._create_level_sequence_for_sequence_bodies(
        sequence_name=SEQUENCE_NAME,
        spawned_bodies=[spawned_body],
        sequence_frame_count=max(1, target_frame_index + 1),
        use_natural_timing=True,
    )
    bedlam360_mini_validation._log_frame_debug(
        "before_sequence_evaluate",
        SEQUENCE_NAME,
        FRAME_NAME,
        {
            "target_sequence_frame_index": target_frame_index,
            "requested_sample_time_seconds": target_state.get("sample_time_seconds"),
            "requested_sample_frame_index": target_frame_index,
            "display_fps": sequence_runtime["display_fps"],
            "warmup_frames": sequence_runtime["warmup_frames"],
            "timeline_frame_count": sequence_runtime["timeline_frame_count"],
            "natural_timeline_frame_count": sequence_runtime["natural_timeline_frame_count"],
        },
    )
    sequence_eval_time = bedlam360_mini_validation._evaluate_level_sequence_frame(
        sequence_runtime["level_sequence"],
        target_frame_index,
        previous_frame_index=None,
        warmup_frames=sequence_runtime["warmup_frames"],
    )
    live_bodies = bedlam360_mini_validation._sync_sequence_bound_bodies(
        sequence_runtime["bodies"]
    )
    if not live_bodies:
        raise RuntimeError("Sequencer evaluation did not return any live BEDLAM bodies")
    live_body = live_bodies[0]
    live_actor = live_body["actor"]
    live_component = live_body["geometry_cache_component"]
    actual_time = None
    if live_component is not None and hasattr(live_component, "get_animation_time"):
        try:
            actual_time = float(live_component.get_animation_time())
        except Exception:
            actual_time = None
    if actual_time is None:
        actual_time = float(target_state.get("sample_time_seconds") or 0.0)
    actual_state = bedlam360_mini_validation._calculate_motion_state_metadata(
        live_component, actual_time
    )
    actual_frame_index = actual_state.get("sample_frame_index")
    playback_status = {
        "mode": "sequencer_levelsequence",
        "sequence_asset_path": sequence_runtime["sequence_asset_path"],
        "display_fps": sequence_runtime["display_fps"],
        "warmup_frames": sequence_runtime["warmup_frames"],
        "requested_time_seconds": float(target_state.get("sample_time_seconds") or 0.0),
        "requested_frame_index": target_frame_index,
        "actual_time_seconds": actual_time,
        "actual_frame_index": actual_frame_index,
        "time_error_seconds": None
        if target_state.get("sample_time_seconds") is None
        else float(actual_time - float(target_state["sample_time_seconds"])),
        "sequence_eval_time": str(sequence_eval_time),
        "timeline_frame_count": sequence_runtime["timeline_frame_count"],
        "natural_timeline_frame_count": sequence_runtime["natural_timeline_frame_count"],
        "streaming_mode_note": "sequencer_manual_tick_path",
    }
    bedlam360_mini_validation._log_frame_debug(
        "after_sequence_evaluate",
        SEQUENCE_NAME,
        FRAME_NAME,
        playback_status,
    )

    transform_payload = log_actor_transform(live_actor, "VALIDATED_BEDLAM_BODY_0")
    bounds_payload = log_actor_bounds(live_actor, "VALIDATED_BEDLAM_BODY_0")

    report = {
        "spawn_pose": spawn_pose,
        "body_pose": body_pose,
        "resolved_asset": resolved_asset,
        "appearance_metadata": appearance_metadata,
        "target_animation_state": target_state,
        "actual_animation_state": actual_state,
        "playback_status": playback_status,
        "final_actor_transform": transform_payload,
        "final_actor_bounds": bounds_payload,
        "sequence_runtime": {
            "sequence_asset_path": sequence_runtime["sequence_asset_path"],
            "display_fps": sequence_runtime["display_fps"],
            "warmup_frames": sequence_runtime["warmup_frames"],
            "timeline_frame_count": sequence_runtime["timeline_frame_count"],
            "natural_timeline_frame_count": sequence_runtime["natural_timeline_frame_count"],
        },
    }
    log_info("FINAL_REPORT " + json.dumps(report, indent=2))


if __name__ == "__main__":
    spawn_and_play_validated_motion()
