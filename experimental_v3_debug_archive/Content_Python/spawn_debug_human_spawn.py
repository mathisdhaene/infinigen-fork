import importlib
import json
import math
import sys
from pathlib import Path

import unreal


JSON_PATH = Path(
    "/media/mathis/PANO/infinigen/outputs/indoors/human_spawn_poc_usd/human_spawn_poses.json"
)
DEBUG_FOLDER = "DebugHumanSpawns"
SPAWN_POSE_INDEX = 0
BEDLAM_ASSET_ID = "it_4482_S_2400"
SAMPLE_TIME_SECONDS = 0.5
SEARCH_RADII_CM = (300.0, 600.0)
SEARCH_RADIUS_CM = max(SEARCH_RADII_CM)
NEARBY_LIMIT = 40
SPAWN_DEBUG_RULERS = True

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

RULER_MESH_PATH = "/Game/StarterContent/Shapes/Shape_Cube.Shape_Cube"
RULER_HUMAN_MATERIAL_PATH = "/Game/StarterContent/Materials/M_Metal_Gold.M_Metal_Gold"
RULER_COUNTER_MATERIAL_PATH = "/Game/StarterContent/Materials/M_Metal_Copper.M_Metal_Copper"
RULER_CEILING_MATERIAL_PATH = "/Game/StarterContent/Materials/M_Metal_Steel.M_Metal_Steel"

KEYWORDS = {
    "sink": ("sink",),
    "counter": ("counter", "kitchen", "workspace"),
    "cabinet": ("cabinet",),
}
HORIZONTAL_SURFACE_MIN_SPAN_CM = 40.0
HORIZONTAL_SURFACE_MAX_THICKNESS_CM = 60.0
HORIZONTAL_SURFACE_MIN_AREA_CM2 = 2500.0
WALL_MIN_HEIGHT_CM = 180.0
WALL_MAX_THICKNESS_CM = 60.0


def log_info(message):
    unreal.log(f"[BEDLAM_SCALE_DIAGNOSTIC] {message}")


def log_warning(message):
    unreal.log_warning(f"[BEDLAM_SCALE_DIAGNOSTIC] {message}")


def load_spawn_pose():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Spawn pose JSON not found: {JSON_PATH}")
    with JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    poses = data.get("spawn_poses", [])
    if not poses:
        raise RuntimeError(f"No spawn poses found in {JSON_PATH}")
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


def make_body_rotation(final_bedlam_yaw_deg):
    return unreal.Rotator(0.0, final_bedlam_yaw_deg, 0.0)


def destroy_existing_debug_actors(actor_subsystem):
    for actor in actor_subsystem.get_all_level_actors():
        label = actor.get_actor_label()
        if label.startswith("DEBUG_BEDLAM_") or label.startswith("DEBUG_RULER_"):
            actor_subsystem.destroy_actor(actor)


def _distance_xy(a, b):
    dx = float(a.x) - float(b.x)
    dy = float(a.y) - float(b.y)
    return math.sqrt(dx * dx + dy * dy)


def _actor_bounds(actor, only_colliding=False):
    origin, extent = actor.get_actor_bounds(only_colliding)
    return {
        "center": origin,
        "extent": extent,
        "size": unreal.Vector(extent.x * 2.0, extent.y * 2.0, extent.z * 2.0),
        "min": unreal.Vector(origin.x - extent.x, origin.y - extent.y, origin.z - extent.z),
        "max": unreal.Vector(origin.x + extent.x, origin.y + extent.y, origin.z + extent.z),
    }


def _vector_to_dict(v):
    if v is None:
        return None
    return {"x": float(v.x), "y": float(v.y), "z": float(v.z)}


def _value_coord(value, axis):
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get(axis)
        return None if raw is None else float(raw)
    raw = getattr(value, axis, None)
    return None if raw is None else float(raw)


def _component_bounds(component):
    origin = None
    extent = None
    try:
        origin, extent = unreal.KismetSystemLibrary.get_component_bounds(component)
    except Exception:
        try:
            bounds = getattr(component, "bounds", None)
            if bounds is not None:
                origin = getattr(bounds, "origin", None)
                extent = getattr(bounds, "box_extent", None)
        except Exception:
            origin = None
            extent = None
    if origin is None or extent is None:
        return None
    if float(extent.x) <= 0.0 and float(extent.y) <= 0.0 and float(extent.z) <= 0.0:
        return None
    return {
        "center": origin,
        "extent": extent,
        "size": unreal.Vector(extent.x * 2.0, extent.y * 2.0, extent.z * 2.0),
        "min": unreal.Vector(origin.x - extent.x, origin.y - extent.y, origin.z - extent.z),
        "max": unreal.Vector(origin.x + extent.x, origin.y + extent.y, origin.z + extent.z),
    }


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


def _matches_keywords(actor, keywords):
    text = f"{actor.get_actor_label()} {actor.get_name()}".lower()
    return any(keyword in text for keyword in keywords)


def _class_name(actor):
    try:
        return actor.get_class().get_name()
    except Exception:
        return type(actor).__name__


def log_actor_transform(actor, label):
    if actor is None:
        log_warning(f"{label} transform unavailable: actor is None")
        return None
    try:
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
    except Exception as exc:
        log_warning(f"{label} transform unavailable: {exc}")
        return None


def log_bounds_analysis(actor, label):
    if actor is None:
        log_warning(f"{label} bounds unavailable: actor is None")
        return None
    try:
        bounds = _actor_bounds(actor, False)
        size = bounds["size"]
        axes = {"x": size.x, "y": size.y, "z": size.z}
        longest_axis = max(axes, key=axes.get)
        orientation_guess = "vertical" if longest_axis == "z" else "horizontal"
        payload = _bounds_to_dict(bounds)
        payload["longest_axis"] = longest_axis
        payload["orientation_guess"] = orientation_guess
        log_info(f"{label} bounds: {json.dumps(payload)}")
        return payload
    except Exception as exc:
        log_warning(f"{label} bounds unavailable: {exc}")
        return None


def _find_stage_actors(actor_subsystem):
    out = []
    for actor in actor_subsystem.get_all_level_actors():
        try:
            cls = _class_name(actor).lower()
            label = actor.get_actor_label().lower()
            if "usdstageactor" not in cls and "usdstage" not in cls and "usd" not in label:
                continue
            bounds = _actor_bounds(actor, False)
            out.append(
                {
                    "label": actor.get_actor_label(),
                    "class": _class_name(actor),
                    "location_cm": _vector_to_dict(actor.get_actor_location()),
                    "rotation_deg": {
                        "pitch": float(actor.get_actor_rotation().pitch),
                        "yaw": float(actor.get_actor_rotation().yaw),
                        "roll": float(actor.get_actor_rotation().roll),
                    },
                    "scale": _vector_to_dict(actor.get_actor_scale3d()),
                    "bounds": _bounds_to_dict(bounds),
                }
            )
        except Exception as exc:
            log_warning(f"Failed to inspect potential USD stage actor '{actor}': {exc}")
    return out


def _find_stage_actor_objects(actor_subsystem):
    actors = []
    for actor in actor_subsystem.get_all_level_actors():
        try:
            cls = _class_name(actor).lower()
            label = actor.get_actor_label().lower()
            if "usdstageactor" in cls or "usdstage" in cls or "usd" in label:
                actors.append(actor)
        except Exception:
            continue
    return actors


def _component_owner_name(component):
    try:
        owner = component.get_owner()
        if owner is None:
            return None
        return owner.get_actor_label()
    except Exception:
        return None


def _component_name(component):
    try:
        return component.get_name()
    except Exception:
        return str(component)


def _find_stage_components(stage_actors, spawn_location):
    components = []
    for actor in stage_actors:
        try:
            actor_components = actor.get_components_by_class(unreal.ActorComponent)
        except Exception:
            actor_components = []
        for component in actor_components:
            try:
                bounds = _component_bounds(component)
                if bounds is None:
                    continue
                dist_xy = _distance_xy(bounds["center"], spawn_location)
                if dist_xy > SEARCH_RADIUS_CM:
                    continue
                cls = _class_name(component)
                components.append(
                    {
                        "component": component,
                        "component_name": _component_name(component),
                        "owner_label": _component_owner_name(component),
                        "class": cls,
                        "distance_xy_cm": dist_xy,
                        "bounds": bounds,
                    }
                )
            except Exception:
                continue
    components.sort(key=lambda item: item["distance_xy_cm"])
    return components[:NEARBY_LIMIT]


def _component_report(item):
    if item is None:
        return None
    return {
        "component_name": item["component_name"],
        "owner_label": item["owner_label"],
        "class": item["class"],
        "distance_xy_cm": float(item["distance_xy_cm"]),
        "bounds": _bounds_to_dict(item["bounds"]),
    }


def _try_query_usd_stage_prim_bounds(stage_actor, spawn_location):
    report = {
        "supported": False,
        "warning": None,
        "samples": None,
    }
    try:
        if not hasattr(unreal, "UsdStageActor"):
            report["warning"] = "Unreal USD Python API symbols not available"
            return report
        report["supported"] = True
        samples = []
        stage_accessors = [
            "get_usd_stage",
            "get_stage",
        ]
        stage = None
        for accessor in stage_accessors:
            if hasattr(stage_actor, accessor):
                try:
                    stage = getattr(stage_actor, accessor)()
                    if stage is not None:
                        break
                except Exception:
                    continue
        if stage is None:
            report["warning"] = "Could not access loaded USD stage from Python"
            return report
        for accessor_name in ("traverse", "Traverse", "GetPseudoRoot"):
            if hasattr(stage, accessor_name):
                report["warning"] = f"USD stage handle available via {accessor_name}, but prim-bounds query not yet implemented in script"
                break
        report["samples"] = samples or None
        return report
    except Exception as exc:
        report["warning"] = f"USD prim-bounds query failed: {exc}"
        return report


def _find_nearby_scene_actors(actor_subsystem, spawn_location):
    nearby = []
    for actor in actor_subsystem.get_all_level_actors():
        try:
            label = actor.get_actor_label()
            if label.startswith("DEBUG_BEDLAM_") or label.startswith("DEBUG_RULER_"):
                continue
            cls = _class_name(actor)
            if cls in {"WorldSettings", "DefaultPhysicsVolume", "Brush"}:
                continue
            bounds = _actor_bounds(actor, False)
            size = bounds["size"]
            if size.x <= 0.0 and size.y <= 0.0 and size.z <= 0.0:
                continue
            dist_xy = _distance_xy(bounds["center"], spawn_location)
            if dist_xy > SEARCH_RADIUS_CM:
                continue
            nearby.append(
                {
                    "actor": actor,
                    "label": label,
                    "class": cls,
                    "bounds": bounds,
                    "distance_xy_cm": dist_xy,
                }
            )
        except Exception:
            continue
    nearby.sort(key=lambda item: item["distance_xy_cm"])
    return nearby[:NEARBY_LIMIT]


def _pick_actor_by_keyword(nearby, key):
    keywords = KEYWORDS[key]
    for item in nearby:
        if _matches_keywords(item["actor"], keywords):
            return item
    return None


def _surface_descriptor(item):
    bounds = item["bounds"]
    size = bounds["size"]
    min_v = bounds["min"]
    max_v = bounds["max"]
    width = float(size.x)
    depth = float(size.y)
    height = float(size.z)
    footprint_area = width * depth
    min_span = min(width, depth)
    max_span = max(width, depth)
    top_z = float(max_v.z)
    bottom_z = float(min_v.z)
    broad_horizontal = (
        max_span >= HORIZONTAL_SURFACE_MIN_SPAN_CM
        and min_span >= HORIZONTAL_SURFACE_MIN_SPAN_CM * 0.5
        and height <= HORIZONTAL_SURFACE_MAX_THICKNESS_CM
        and footprint_area >= HORIZONTAL_SURFACE_MIN_AREA_CM2
    )
    wall_like = (
        height >= WALL_MIN_HEIGHT_CM
        and min_span <= WALL_MAX_THICKNESS_CM
        and max_span >= 80.0
    )
    return {
        "label": item["label"],
        "class": item["class"],
        "distance_xy_cm": float(item["distance_xy_cm"]),
        "bounds": bounds,
        "width_cm": width,
        "depth_cm": depth,
        "height_cm": height,
        "footprint_area_cm2": footprint_area,
        "top_z_cm": top_z,
        "bottom_z_cm": bottom_z,
        "broad_horizontal": broad_horizontal,
        "wall_like": wall_like,
    }


def _estimate_scene_geometry(nearby, spawn_location, human_bounds):
    descriptors = [_surface_descriptor(item) for item in nearby]
    human_min_z = None
    human_max_z = None
    if human_bounds:
        min_v = human_bounds.get("min")
        max_v = human_bounds.get("max")
        human_min_z = _value_coord(min_v, "z")
        human_max_z = _value_coord(max_v, "z")
    spawn_z = float(spawn_location.z)
    foot_z = human_min_z if human_min_z is not None else spawn_z

    floor_candidates = []
    ceiling_candidates = []
    counter_candidates = []
    wall_candidates = []

    for desc in descriptors:
        if desc["wall_like"]:
            wall_candidates.append(desc)
        if not desc["broad_horizontal"]:
            continue

        top_delta = abs(desc["top_z_cm"] - foot_z)
        bottom_delta = abs(desc["bottom_z_cm"] - foot_z)
        closest_surface_z = desc["top_z_cm"] if top_delta <= bottom_delta else desc["bottom_z_cm"]
        if abs(closest_surface_z - foot_z) <= 30.0:
            candidate = dict(desc)
            candidate["surface_z_cm"] = closest_surface_z
            floor_candidates.append(candidate)

        if human_max_z is not None and desc["bottom_z_cm"] > human_max_z + 30.0:
            candidate = dict(desc)
            candidate["surface_z_cm"] = desc["bottom_z_cm"]
            ceiling_candidates.append(candidate)

    floor_candidates.sort(
        key=lambda c: (
            abs(c["surface_z_cm"] - foot_z),
            -c["footprint_area_cm2"],
            c["distance_xy_cm"],
        )
    )
    floor = floor_candidates[0] if floor_candidates else None
    floor_z = None if floor is None else float(floor["surface_z_cm"])

    if floor_z is not None:
        for desc in descriptors:
            if not desc["broad_horizontal"]:
                continue
            top_z = desc["top_z_cm"]
            rel_height = top_z - floor_z
            if 70.0 <= rel_height <= 120.0:
                candidate = dict(desc)
                candidate["surface_z_cm"] = top_z
                candidate["relative_height_cm"] = rel_height
                counter_candidates.append(candidate)

    ceiling_candidates.sort(
        key=lambda c: (
            c["surface_z_cm"],
            c["distance_xy_cm"],
            -c["footprint_area_cm2"],
        )
    )
    ceiling = ceiling_candidates[0] if ceiling_candidates else None
    ceiling_z = None if ceiling is None else float(ceiling["surface_z_cm"])
    room_height = None if floor_z is None or ceiling_z is None else float(ceiling_z - floor_z)

    counter_candidates.sort(
        key=lambda c: (
            c["distance_xy_cm"],
            abs(c.get("relative_height_cm", 0.0) - 90.0),
            -c["footprint_area_cm2"],
        )
    )
    counter = counter_candidates[0] if counter_candidates else None

    wall_candidates.sort(key=lambda c: (c["distance_xy_cm"], -c["height_cm"]))

    return {
        "floor_candidate": floor,
        "ceiling_candidate": ceiling,
        "counter_candidate": counter,
        "wall_candidates": wall_candidates[:10],
        "descriptors": descriptors,
        "floor_z_cm": floor_z,
        "ceiling_z_cm": ceiling_z,
        "room_height_cm": room_height,
    }


def _rank_nearby_items(nearby, radius_cm):
    ranked = []
    for item in nearby:
        if item["distance_xy_cm"] > radius_cm:
            continue
        bounds = item["bounds"]
        ranked.append(
            {
                "label": item["label"],
                "class": item["class"],
                "distance_xy_cm": float(item["distance_xy_cm"]),
                "bounds": _bounds_to_dict(bounds),
                "dimensions_cm": {
                    "width": float(bounds["size"].x),
                    "depth": float(bounds["size"].y),
                    "height": float(bounds["size"].z),
                },
            }
        )
    return ranked[:NEARBY_LIMIT]


def _candidate_report(candidate):
    if candidate is None:
        return None
    return {
        "label": candidate["label"],
        "class": candidate["class"],
        "distance_xy_cm": float(candidate["distance_xy_cm"]),
        "surface_z_cm": float(candidate.get("surface_z_cm")) if candidate.get("surface_z_cm") is not None else None,
        "relative_height_cm": float(candidate.get("relative_height_cm")) if candidate.get("relative_height_cm") is not None else None,
        "width_cm": float(candidate["width_cm"]),
        "depth_cm": float(candidate["depth_cm"]),
        "height_cm": float(candidate["height_cm"]),
        "footprint_area_cm2": float(candidate["footprint_area_cm2"]),
        "bounds": _bounds_to_dict(candidate["bounds"]),
    }


def _plausibility_summary(human_height_cm, room_height_cm, counter_candidate):
    human_ok = human_height_cm is not None and 160.0 <= float(human_height_cm) <= 190.0
    counter_height = None
    counter_ok = None
    counter_kind = None
    if counter_candidate is not None and counter_candidate.get("relative_height_cm") is not None:
        counter_height = float(counter_candidate["relative_height_cm"])
        if 85.0 <= counter_height <= 95.0:
            counter_ok = True
            counter_kind = "counter_like"
        elif 70.0 <= counter_height <= 80.0:
            counter_ok = True
            counter_kind = "table_like"
        else:
            counter_ok = False
            counter_kind = "unusual_horizontal_surface"
    ceiling_ok = None if room_height_cm is None else 240.0 <= float(room_height_cm) <= 270.0
    notes = []
    if human_ok is False:
        notes.append("human height outside 160-190 cm reference range")
    if ceiling_ok is False:
        notes.append("room height outside 240-270 cm reference range")
    if counter_ok is False:
        notes.append("nearest horizontal work surface outside 70-120 cm reference range")
    if room_height_cm is None:
        notes.append("room height could not be estimated from nearby geometry")
    if counter_candidate is None:
        notes.append("no counter/table-like horizontal surface detected nearby")
    plausible = not any(
        value is False for value in (human_ok, counter_ok, ceiling_ok) if value is not None
    )
    return {
        "human_height_plausible": human_ok,
        "counter_surface_height_cm": counter_height,
        "counter_surface_kind": counter_kind,
        "counter_surface_plausible": counter_ok,
        "room_height_plausible": ceiling_ok,
        "overall_plausible": plausible,
        "notes": notes,
    }


def _spawn_ruler(actor_subsystem, label, location, height_cm, material_path):
    try:
        actor = actor_subsystem.spawn_actor_from_class(
            unreal.StaticMeshActor,
            location,
            unreal.Rotator(0.0, 0.0, 0.0),
        )
        if actor is None:
            return None
        actor.set_actor_label(label)
        actor.set_folder_path(DEBUG_FOLDER)
        actor.set_actor_scale3d(unreal.Vector(0.03, 0.03, float(height_cm) / 100.0))
        mesh = unreal.load_asset(RULER_MESH_PATH)
        material = unreal.load_asset(material_path)
        smc = actor.get_component_by_class(unreal.StaticMeshComponent)
        if smc is not None and mesh is not None:
            smc.set_editor_property("static_mesh", mesh)
            if material is not None:
                smc.set_material(0, material)
        return actor
    except Exception as exc:
        log_warning(f"Failed to spawn ruler {label}: {exc}")
        return None


def _print_measurement_report(
    spawn_location,
    human_actor,
    nearby,
    stage_actors,
    stage_components,
    usd_stage_query,
    playback_status,
):
    human_bounds = log_bounds_analysis(human_actor, "DEBUG_BEDLAM_BODY_0") or {}
    size = (((human_bounds or {}).get("size")) or {})
    min_v = (((human_bounds or {}).get("min")) or {})
    max_v = (((human_bounds or {}).get("max")) or {})
    geometry_source = nearby
    geometry_mode = "actor_bounds"
    if stage_components:
        geometry_source = [
            {
                "actor": item["component"],
                "label": item["component_name"],
                "class": item["class"],
                "bounds": item["bounds"],
                "distance_xy_cm": item["distance_xy_cm"],
            }
            for item in stage_components
        ]
        geometry_mode = "usd_component_bounds"

    geometry = _estimate_scene_geometry(geometry_source, spawn_location, human_bounds)
    floor_z = geometry["floor_z_cm"]
    ceiling_z = geometry["ceiling_z_cm"]
    room_height = geometry["room_height_cm"]

    sink = _pick_actor_by_keyword(nearby, "sink")
    counter = _pick_actor_by_keyword(nearby, "counter")
    cabinet = _pick_actor_by_keyword(nearby, "cabinet")
    geometric_counter = geometry["counter_candidate"]
    wall_candidates = geometry["wall_candidates"]

    human_height_cm = size.get("z")
    human_width_cm = size.get("x")
    human_depth_cm = size.get("y")
    plausibility = _plausibility_summary(human_height_cm, room_height, geometric_counter)

    def item_report(item):
        if item is None:
            return None
        bounds = item["bounds"]
        return {
            "label": item["label"],
            "class": item["class"],
            "distance_xy_cm": float(item["distance_xy_cm"]),
            "bounds": _bounds_to_dict(bounds),
            "height_cm": float(bounds["size"].z),
            "depth_cm": float(bounds["size"].y),
            "width_cm": float(bounds["size"].x),
        }

    actual_time = None
    time_error = None
    if playback_status and playback_status.get("actor_results"):
        first = playback_status["actor_results"][0]
        actual_time = first.get("actual_time_seconds")
        time_error = first.get("time_error_seconds")
        if time_error is not None and abs(float(time_error)) > 1e-3:
            log_warning(
                f"Playback timing mismatch: requested={SAMPLE_TIME_SECONDS:.3f}s actual={float(actual_time):.6f}s error={float(time_error):.6f}s"
            )

    report = {
        "bedlam_actor_transform": log_actor_transform(human_actor, "DEBUG_BEDLAM_BODY_0"),
        "bedlam_actor_bounds": human_bounds,
        "human_height_cm": human_height_cm,
        "human_width_cm": human_width_cm,
        "human_depth_cm": human_depth_cm,
        "human_min_z_cm": min_v.get("z"),
        "human_max_z_cm": max_v.get("z"),
        "spawn_point": {
            "location_cm": _vector_to_dict(spawn_location),
            "floor_z_cm": floor_z,
            "ceiling_z_cm": ceiling_z,
            "approx_room_height_cm": room_height,
            "foot_above_floor_cm": None if floor_z is None or min_v.get("z") is None else float(min_v.get("z") - floor_z),
        },
        "playback": {
            "requested_time_seconds": SAMPLE_TIME_SECONDS,
            "actual_time_seconds": actual_time,
            "time_error_seconds": time_error,
            "raw": playback_status,
        },
        "usd_stage_actors": stage_actors or None,
        "usd_stage_component_bounds": [_component_report(item) for item in stage_components] or None,
        "usd_stage_prim_query": usd_stage_query,
        "nearby_objects": {
            "sink": item_report(sink),
            "counter": item_report(counter),
            "cabinet": item_report(cabinet),
        },
        "geometry_estimates": {
            "floor_candidate": _candidate_report(geometry["floor_candidate"]),
            "ceiling_candidate": _candidate_report(geometry["ceiling_candidate"]),
            "nearest_counter_or_table_surface": _candidate_report(geometric_counter),
            "nearby_wall_candidates": [_candidate_report(item) for item in wall_candidates] or None,
        },
        "geometry_mode": geometry_mode,
        "nearby_ranked_300cm": _rank_nearby_items(geometry_source, SEARCH_RADII_CM[0]) or None,
        "nearby_ranked_600cm": _rank_nearby_items(geometry_source, SEARCH_RADII_CM[1]) or None,
        "nearest_object_dimensions_cm": None
        if not geometry_source
        else {
            "label": geometry_source[0]["label"],
            "class": geometry_source[0]["class"],
            "width": float(geometry_source[0]["bounds"]["size"].x),
            "depth": float(geometry_source[0]["bounds"]["size"].y),
            "height": float(geometry_source[0]["bounds"]["size"].z),
        },
        "plausibility_summary": plausibility,
        "fallback_recommendation": None
        if geometry_source
        else (
            "USD stage components did not expose usable bounds. Fallback: import the USD into level as normal actors and remeasure StaticMeshActor bounds, or measure room/object dimensions directly in Blender from scene.blend."
        ),
    }
    log_info("FULL_REPORT " + json.dumps(report, indent=2))
    return report, sink, counter, geometric_counter, floor_z, ceiling_z


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


def spawn_and_drive_bedlam_body():
    pose = load_spawn_pose()
    spawn_location = transform_infinigen_position_to_bedlam(pose["position_xyz"])
    final_bedlam_yaw_deg = transform_infinigen_yaw_to_bedlam(pose.get("yaw"))

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    destroy_existing_debug_actors(actor_subsystem)

    log_info(
        f"T_infinigen_to_bedlam: scene_root_offset_cm=({SCENE_ROOT_OFFSET_CM.x:.2f}, {SCENE_ROOT_OFFSET_CM.y:.2f}, {SCENE_ROOT_OFFSET_CM.z:.2f}) "
        f"scene_root_yaw_deg={SCENE_ROOT_YAW_DEG:.2f}"
    )
    log_info(
        f"Using spawn pose index {SPAWN_POSE_INDEX}: final_bedlam_location=({spawn_location.x:.2f}, {spawn_location.y:.2f}, {spawn_location.z:.2f}) "
        f"final_bedlam_yaw={final_bedlam_yaw_deg:.2f} sample_time_seconds={SAMPLE_TIME_SECONDS:.3f}"
    )
    log_info(
        f"Spawn metadata: room={pose.get('room')} target={pose.get('target_object')} activity={pose.get('activity_hint')}"
    )

    body_pose = build_body_pose(spawn_location, final_bedlam_yaw_deg)
    resolved_asset = reconstruct_one_bedlam_body.resolve_body_asset(BEDLAM_ASSET_ID)
    actor, appearance_metadata = reconstruct_one_bedlam_body.spawn_body_actor(
        body_pose, resolved_asset, return_metadata=True
    )
    actor.set_actor_label("DEBUG_BEDLAM_BODY_0")
    actor.set_folder_path(DEBUG_FOLDER)

    gc_component = actor.get_geometry_cache_component() if hasattr(actor, "get_geometry_cache_component") else None
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
    playback_status = bedlam360_mini_validation._playback_driven_prepare_targets(
        spawned_bodies=[spawned_body],
        target_animation_states=[target_state],
        sequence_name="infinigen_debug",
        frame_name="infinigen_pose_0",
    )

    nearby = _find_nearby_scene_actors(actor_subsystem, spawn_location)
    stage_actors = _find_stage_actors(actor_subsystem)
    stage_actor_objects = _find_stage_actor_objects(actor_subsystem)
    stage_components = _find_stage_components(stage_actor_objects, spawn_location)
    usd_stage_query = None
    if stage_actor_objects:
        usd_stage_query = _try_query_usd_stage_prim_bounds(stage_actor_objects[0], spawn_location)

    log_info(f"Resolved asset: {json.dumps(resolved_asset, indent=2)}")
    log_info(f"Appearance metadata: {json.dumps(appearance_metadata, indent=2)}")
    log_info(f"Target animation state: {json.dumps(target_state, indent=2)}")
    log_info(f"Playback status: {json.dumps(playback_status, indent=2)}")

    _report, sink, counter, geometric_counter, floor_z, ceiling_z = _print_measurement_report(
        spawn_location,
        actor,
        nearby,
        stage_actors,
        stage_components,
        usd_stage_query,
        playback_status,
    )

    if SPAWN_DEBUG_RULERS:
        _spawn_ruler(
            actor_subsystem,
            "DEBUG_RULER_HUMAN_180CM",
            unreal.Vector(spawn_location.x + 40.0, spawn_location.y, spawn_location.z + 90.0),
            180.0,
            RULER_HUMAN_MATERIAL_PATH,
        )
        counter_for_ruler = counter if counter is not None else geometric_counter
        if counter_for_ruler is not None:
            counter_center = counter_for_ruler["bounds"]["center"]
            _spawn_ruler(
                actor_subsystem,
                "DEBUG_RULER_COUNTER_90CM",
                unreal.Vector(counter_center.x + 20.0, counter_center.y, (floor_z if floor_z is not None else spawn_location.z) + 45.0),
                90.0,
                RULER_COUNTER_MATERIAL_PATH,
            )
        if ceiling_z is not None and floor_z is not None:
            _spawn_ruler(
                actor_subsystem,
                "DEBUG_RULER_CEILING_250CM",
                unreal.Vector(spawn_location.x - 40.0, spawn_location.y, floor_z + 125.0),
                250.0,
                RULER_CEILING_MATERIAL_PATH,
            )


if __name__ == "__main__":
    spawn_and_drive_bedlam_body()
