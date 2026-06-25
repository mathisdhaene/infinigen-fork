import json
import logging
import math
import hashlib
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon
from mathutils import Vector

from infinigen.core import tags as t

logger = logging.getLogger(__name__)


TARGET_TAGS = {
    t.Semantics.Table: ("standing_near_table", "standing"),
    t.Semantics.KitchenCounter: ("standing_near_counter", "standing"),
    t.Semantics.Sink: ("standing_near_sink", "standing"),
    t.Semantics.Desk: ("standing_near_desk", "standing"),
    t.Semantics.Chair: ("sitting_near_chair", "sitting"),
    t.Semantics.LoungeSeating: ("sitting_near_sofa", "sitting"),
    t.Semantics.Seating: ("sitting_near_seat", "sitting"),
}

OBSTACLE_CATEGORY_TAGS = (
    (t.Semantics.KitchenCounter, "counter"),
    (t.Semantics.Sink, "sink"),
    (t.Semantics.Table, "table"),
    (t.Semantics.Desk, "desk"),
    (t.Semantics.Chair, "chair"),
    (t.Semantics.LoungeSeating, "sofa"),
    (t.Semantics.Seating, "seat"),
    (t.Semantics.Storage, "cabinet"),
    (t.Semantics.KitchenAppliance, "appliance"),
    (t.Semantics.Bed, "bed"),
    (t.Semantics.Door, "door"),
)

FREE_SPACE_WALL_MARGIN_M = 0.5
FREE_SPACE_OBSTACLE_CLEARANCE_M = 0.4
FREE_SPACE_GRID_SPACING_M = 1.5
FREE_SPACE_RANDOM_SAMPLES = 8
FREE_SPACE_MAX_POSES_PER_ROOM = 8
FREE_SPACE_DUPLICATE_RADIUS_M = 0.75


def _safe_float(value):
    if value is None:
        return None
    return float(value)


def _safe_name(obj):
    return getattr(obj, "name", None)


def _serialize_tags(tags):
    return sorted(str(tag) for tag in tags)


def _world_location(obj):
    return np.asarray(obj.matrix_world.translation, dtype=float)


def _world_rotation_euler(obj):
    return np.asarray(obj.matrix_world.to_euler(), dtype=float)


def _world_bounds_corners(obj):
    return np.asarray(
        [obj.matrix_world @ Vector(corner) for corner in obj.bound_box], dtype=float
    )


def _world_bounds_minmax(obj):
    corners = _world_bounds_corners(obj)
    return corners.min(axis=0), corners.max(axis=0)


def _xy_polygon_from_bounds(bounds_min, bounds_max):
    return [
        [float(bounds_min[0]), float(bounds_min[1])],
        [float(bounds_max[0]), float(bounds_min[1])],
        [float(bounds_max[0]), float(bounds_max[1])],
        [float(bounds_min[0]), float(bounds_max[1])],
    ]


def _serialize_xy_polygon(polygon):
    if polygon is None:
        return None
    try:
        geom_type = polygon.geom_type
    except Exception:
        return None

    def coords_to_list(coords):
        return [[float(x), float(y)] for x, y in coords]

    if geom_type == "Polygon":
        return {
            "type": "Polygon",
            "exterior": coords_to_list(polygon.exterior.coords),
            "interiors": [coords_to_list(ring.coords) for ring in polygon.interiors],
        }
    if geom_type == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "polygons": [
                {
                    "exterior": coords_to_list(poly.exterior.coords),
                    "interiors": [
                        coords_to_list(ring.coords) for ring in poly.interiors
                    ],
                }
                for poly in polygon.geoms
            ],
        }
    return {"type": geom_type, "wkt": polygon.wkt}


def _bbox_record(bounds_min, bounds_max):
    return {
        "min_xyz": [float(v) for v in bounds_min],
        "max_xyz": [float(v) for v in bounds_max],
    }


def _category_hint(tags):
    for tag, hint in OBSTACLE_CATEGORY_TAGS:
        if tag in tags:
            return hint
    if t.Semantics.Furniture in tags:
        return "furniture"
    if t.Semantics.Object in tags:
        return "object"
    return None


def _category_priority(category_hint):
    order = {
        "counter": 0,
        "sink": 1,
        "table": 2,
        "desk": 3,
        "sofa": 4,
        "seat": 5,
        "chair": 6,
        "cabinet": 7,
        "appliance": 8,
        "bed": 9,
        "door": 10,
        "furniture": 11,
        "object": 12,
    }
    return order.get(category_hint, 99)


def _yaw_towards(source_xy, target_xy):
    delta = np.asarray(target_xy, dtype=float) - np.asarray(source_xy, dtype=float)
    if np.linalg.norm(delta) < 1e-8:
        return None
    return float(math.atan2(delta[1], delta[0]))


def _deterministic_unit_interval(key):
    digest = hashlib.md5(str(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _deterministic_yaw(key):
    return float((2.0 * math.pi * _deterministic_unit_interval(key)) - math.pi)


def _dominant_xy_axis(bounds):
    extents = np.ptp(bounds[:, :2], axis=0)
    if extents[0] >= extents[1]:
        return np.array([1.0, 0.0]), float(extents[0]), float(extents[1])
    return np.array([0.0, 1.0]), float(extents[1]), float(extents[0])


def _room_records(state):
    rooms = []
    for obj_state in state.objs.values():
        tags = getattr(obj_state, "tags", set())
        if t.Semantics.Room not in tags:
            continue
        room_obj = obj_state.obj
        if room_obj is None:
            continue
        room_poly = getattr(obj_state, "polygon", None)
        room_bounds = None
        room_floor_z = None
        try:
            room_min, room_max = _world_bounds_minmax(room_obj)
            room_bounds = np.stack([room_min, room_max], axis=0)
            room_floor_z = float(room_min[2])
        except Exception:
            logger.debug(
                "Failed to read bounds for room %s",
                _safe_name(room_obj),
                exc_info=True,
            )
        rooms.append(
            {
                "name": _safe_name(room_obj),
                "tags": tags,
                "polygon": room_poly,
                "floor_z": room_floor_z,
                "bounds": room_bounds,
            }
        )
    return rooms


def _infer_room_name(point_xy, rooms):
    point = Point(float(point_xy[0]), float(point_xy[1]))

    for room in rooms:
        poly = room["polygon"]
        if poly is None:
            continue
        try:
            if poly.covers(point):
                return room["name"], room["floor_z"]
        except Exception:
            logger.debug(
                "Room polygon lookup failed for %s",
                room["name"],
                exc_info=True,
            )

    best_room = None
    best_dist = None
    for room in rooms:
        poly = room["polygon"]
        if poly is None:
            continue
        try:
            dist = float(poly.distance(point))
        except Exception:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_room = room

    if best_room is not None:
        return best_room["name"], best_room["floor_z"]
    return None, None


def _pick_spawn_position(
    bounds, target_location, target_rotation=None, room_center_xy=None
):
    center_xy = np.mean(bounds[:, :2], axis=0)
    major_axis, major_extent, minor_extent = _dominant_xy_axis(bounds)
    offset_distance = max(0.45, 0.5 * minor_extent + 0.35)
    facing_axis = major_axis
    if target_rotation is not None and len(target_rotation) >= 3:
        yaw = float(target_rotation[2])
        facing_axis = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
        norm = np.linalg.norm(facing_axis)
        if norm > 1e-8:
            facing_axis = facing_axis / norm
        else:
            facing_axis = major_axis
    side_axis = np.array([-facing_axis[1], facing_axis[0]], dtype=float)

    candidates = [
        center_xy + facing_axis * offset_distance,
        center_xy - facing_axis * offset_distance,
        center_xy + side_axis * offset_distance,
        center_xy - side_axis * offset_distance,
    ]

    if room_center_xy is None:
        room_center_xy = np.asarray(target_location[:2], dtype=float)

    best_xy = min(
        candidates,
        key=lambda xy: float(
            np.linalg.norm(np.asarray(xy) - np.asarray(room_center_xy))
        ),
    )
    return np.asarray(best_xy, dtype=float), major_extent, minor_extent


def _activity_and_pose(tags):
    for tag, value in TARGET_TAGS.items():
        if tag in tags:
            return value
    return None, None


def _polygon_contains_xy(polygon, xy):
    point = Point(float(xy[0]), float(xy[1]))
    try:
        return polygon.covers(point)
    except Exception:
        return False


def _point_clearance_to_obstacles(point_xy, obstacle_polygons):
    point = Point(float(point_xy[0]), float(point_xy[1]))
    best = None
    best_obstacle = None
    for obstacle in obstacle_polygons:
        dist = float(obstacle["polygon"].distance(point))
        if best is None or dist < best:
            best = dist
            best_obstacle = obstacle
    return best, best_obstacle


def _point_is_valid_free_space(
    room_polygon,
    wall_margin_m,
    obstacle_polygons,
    obstacle_clearance_m,
    point_xy,
):
    point = Point(float(point_xy[0]), float(point_xy[1]))
    inner_poly = room_polygon.buffer(-float(wall_margin_m))
    if inner_poly.is_empty or not inner_poly.covers(point):
        return False, None
    clearance_to_obstacles, _ = _point_clearance_to_obstacles(point_xy, obstacle_polygons)
    if clearance_to_obstacles is not None and clearance_to_obstacles < float(
        obstacle_clearance_m
    ):
        return False, clearance_to_obstacles
    wall_clearance = float(room_polygon.boundary.distance(point))
    obstacle_clearance = (
        float(clearance_to_obstacles)
        if clearance_to_obstacles is not None
        else wall_clearance
    )
    return True, min(wall_clearance, obstacle_clearance)


def _append_if_distinct(points, candidate_xy, min_distance_m):
    candidate_xy = np.asarray(candidate_xy, dtype=float)
    for existing in points:
        if float(np.linalg.norm(candidate_xy - np.asarray(existing, dtype=float))) < float(
            min_distance_m
        ):
            return False
    points.append(candidate_xy.tolist())
    return True


def _room_obstacle_polygons(obstacle_records, room_name):
    polygons = []
    for obstacle in obstacle_records:
        if obstacle.get("room") != room_name:
            continue
        footprint = obstacle.get("footprint_world_xy")
        if not footprint:
            continue
        try:
            poly = Polygon(footprint)
        except Exception:
            continue
        if poly.is_empty:
            continue
        polygons.append(
            {
                "object_name": obstacle.get("object_name"),
                "category_hint": obstacle.get("category_hint"),
                "polygon": poly,
                "area": float(poly.area),
            }
        )
    polygons.sort(
        key=lambda record: (_category_priority(record["category_hint"]), -record["area"])
    )
    return polygons


def _candidate_yaw_for_free_space(room, point_xy, obstacle_polygons, candidate_index):
    centroid = room["polygon"].centroid
    room_center_xy = np.array([float(centroid.x), float(centroid.y)], dtype=float)
    _, nearest_obstacle = _point_clearance_to_obstacles(point_xy, obstacle_polygons)
    if nearest_obstacle is not None:
        obstacle_center = nearest_obstacle["polygon"].centroid
        yaw = _yaw_towards(point_xy, (obstacle_center.x, obstacle_center.y))
        if yaw is not None:
            return yaw
    yaw = _yaw_towards(point_xy, room_center_xy)
    if yaw is not None:
        return yaw
    return _deterministic_yaw(f"{room['name']}::{candidate_index}")


def _generate_room_free_space_candidates(room, obstacle_records):
    polygon = room["polygon"]
    if polygon is None:
        return []
    try:
        if polygon.is_empty:
            return []
    except Exception:
        return []

    obstacle_polygons = _room_obstacle_polygons(obstacle_records, room["name"])
    candidates_xy = []

    centroid = polygon.centroid
    _append_if_distinct(
        candidates_xy,
        [float(centroid.x), float(centroid.y)],
        FREE_SPACE_DUPLICATE_RADIUS_M,
    )

    minx, miny, maxx, maxy = polygon.bounds
    x_values = np.arange(minx, maxx + 1e-6, FREE_SPACE_GRID_SPACING_M)
    y_values = np.arange(miny, maxy + 1e-6, FREE_SPACE_GRID_SPACING_M)
    for x in x_values:
        for y in y_values:
            _append_if_distinct(
                candidates_xy,
                [float(x), float(y)],
                FREE_SPACE_DUPLICATE_RADIUS_M,
            )

    rng_seed = int.from_bytes(
        hashlib.md5(room["name"].encode("utf-8")).digest()[:8], "big"
    )
    rng = np.random.default_rng(rng_seed)
    for _ in range(FREE_SPACE_RANDOM_SAMPLES):
        sample_xy = [float(rng.uniform(minx, maxx)), float(rng.uniform(miny, maxy))]
        _append_if_distinct(
            candidates_xy,
            sample_xy,
            FREE_SPACE_DUPLICATE_RADIUS_M,
        )

    if polygon.boundary.length > 1e-6:
        inner_poly = polygon.buffer(-FREE_SPACE_WALL_MARGIN_M)
        if not inner_poly.is_empty:
            for fraction in (0.1, 0.35, 0.6, 0.85):
                boundary_pt = polygon.boundary.interpolate(fraction, normalized=True)
                look_in = np.array([centroid.x - boundary_pt.x, centroid.y - boundary_pt.y])
                norm = float(np.linalg.norm(look_in))
                if norm > 1e-8:
                    inward_xy = np.array([boundary_pt.x, boundary_pt.y]) + (
                        look_in / norm
                    ) * max(FREE_SPACE_WALL_MARGIN_M + 0.35, 0.9)
                    _append_if_distinct(
                        candidates_xy,
                        inward_xy.tolist(),
                        FREE_SPACE_DUPLICATE_RADIUS_M,
                    )

    valid_candidates = []
    for candidate_index, candidate_xy in enumerate(candidates_xy):
        if len(valid_candidates) >= FREE_SPACE_MAX_POSES_PER_ROOM:
            break
        is_valid, clearance_score = _point_is_valid_free_space(
            polygon,
            FREE_SPACE_WALL_MARGIN_M,
            obstacle_polygons,
            FREE_SPACE_OBSTACLE_CLEARANCE_M,
            candidate_xy,
        )
        if not is_valid:
            continue
        yaw = _candidate_yaw_for_free_space(
            room,
            candidate_xy,
            obstacle_polygons,
            candidate_index,
        )
        valid_candidates.append(
            {
                "position_xy": [float(candidate_xy[0]), float(candidate_xy[1])],
                "yaw": _safe_float(yaw),
                "clearance_score": _safe_float(clearance_score),
            }
        )
    return valid_candidates


def _export_scene_collision_metadata(output_folder, rooms, obstacle_records, spawn_poses):
    output_path = Path(output_folder) / "scene_collision_metadata.json"
    payload = {
        "coordinate_frame": {
            "source": "Infinigen/Blender world",
            "units": "meters",
            "axes": "Blender convention, Z up",
            "note": "Later conversion to BEDLAM/Unreal should be handled by T_infinigen_to_bedlam.",
        },
        "rooms": [
            {
                "name": room["name"],
                "semantic_tags": _serialize_tags(room["tags"]),
                "floor_polygon_world_xy": _serialize_xy_polygon(room["polygon"]),
                "floor_z": _safe_float(room["floor_z"]),
                "ceiling_z": None
                if room["bounds"] is None
                else _safe_float(room["bounds"][1][2]),
                "bbox": None
                if room["bounds"] is None
                else _bbox_record(room["bounds"][0], room["bounds"][1]),
            }
            for room in rooms
        ],
        "obstacles": obstacle_records,
        "spawn_poses": spawn_poses,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote scene collision metadata to %s", output_path)
    return output_path


def sample_human_spawn_poses(state, solver, output_folder):
    del solver

    spawn_output_path = Path(output_folder) / "human_spawn_poses.json"
    rooms = _room_records(state)
    spawn_poses = []
    obstacle_records = []
    semantic_spawn_count = 0
    free_space_spawn_count = 0
    room_bounds_summary = {
        room["name"]: room["bounds"].tolist() if room["bounds"] is not None else None
        for room in rooms
    }
    all_room_bounds = [room["bounds"] for room in rooms if room["bounds"] is not None]
    house_bounds = None
    if all_room_bounds:
        room_points = np.concatenate(all_room_bounds, axis=0)
        house_bounds = np.stack(
            [room_points.min(axis=0), room_points.max(axis=0)], axis=0
        )
        logger.info("House world bounds min/max: %s", house_bounds.tolist())
    logger.info("Room world bounds min/max: %s", room_bounds_summary)

    for obj_state in state.objs.values():
        tags = getattr(obj_state, "tags", set())
        if t.Semantics.Room in tags:
            continue

        obj = getattr(obj_state, "obj", None)
        if obj is None:
            continue

        try:
            bounds = _world_bounds_corners(obj)
        except Exception:
            logger.warning(
                "Skipping %s because bounds could not be computed", _safe_name(obj)
            )
            continue

        location = _world_location(obj)
        rotation_arr = _world_rotation_euler(obj)
        bounds_center = np.mean(bounds, axis=0)
        bounds_min = bounds.min(axis=0)
        bounds_max = bounds.max(axis=0)

        room_name, room_floor_z = _infer_room_name(location[:2], rooms)
        room_center_xy = None
        if room_name is not None:
            for room in rooms:
                if room["name"] == room_name and room["polygon"] is not None:
                    centroid = room["polygon"].centroid
                    room_center_xy = np.array([centroid.x, centroid.y], dtype=float)
                    break

        obstacle_records.append(
            {
                "object_name": _safe_name(obj),
                "semantic_tags": _serialize_tags(tags),
                "room": room_name,
                "bbox": _bbox_record(bounds_min, bounds_max),
                "footprint_world_xy": _xy_polygon_from_bounds(bounds_min, bounds_max),
                "z_min": _safe_float(bounds_min[2]),
                "z_max": _safe_float(bounds_max[2]),
                "category_hint": _category_hint(tags),
            }
        )

        activity_hint, pose_type = _activity_and_pose(tags)
        if activity_hint is None:
            continue

        spawn_xy, _, _ = _pick_spawn_position(
            bounds,
            location,
            target_rotation=rotation_arr,
            room_center_xy=room_center_xy,
        )
        yaw = _yaw_towards(spawn_xy, location[:2])

        floor_z = room_floor_z
        if floor_z is None:
            floor_z = float(np.min(bounds[:, 2]))

        spawn_position = np.array(
            [
                _safe_float(spawn_xy[0]),
                _safe_float(spawn_xy[1]),
                _safe_float(floor_z),
            ],
            dtype=float,
        )

        logger.info(
            "Spawn target=%s room=%s world_location=%s bounds_center=%s spawn_position=%s yaw=%s",
            _safe_name(obj),
            room_name,
            location.tolist(),
            bounds_center.tolist(),
            spawn_position.tolist(),
            _safe_float(yaw),
        )
        if room_name is not None:
            logger.info(
                "Room %s bounds min/max: %s",
                room_name,
                next(
                    (
                        room["bounds"].tolist()
                        for room in rooms
                        if room["name"] == room_name and room["bounds"] is not None
                    ),
                    None,
                ),
            )

        spawn_poses.append(
            {
                "position_xyz": spawn_position.tolist(),
                "yaw": _safe_float(yaw),
                "room": room_name,
                "target_object": _safe_name(obj),
                "activity_hint": activity_hint,
                "pose_type": pose_type,
                "source": "heuristic_v0",
            }
        )
        semantic_spawn_count += 1

    free_space_by_room = {}
    for room in rooms:
        if room["polygon"] is None:
            continue
        free_space_candidates = _generate_room_free_space_candidates(room, obstacle_records)
        free_space_by_room[room["name"]] = free_space_candidates
        for candidate in free_space_candidates:
            floor_z = room["floor_z"]
            if floor_z is None:
                floor_z = 0.0
            spawn_poses.append(
                {
                    "position_xyz": [
                        float(candidate["position_xy"][0]),
                        float(candidate["position_xy"][1]),
                        float(floor_z),
                    ],
                    "yaw": candidate["yaw"],
                    "room": room["name"],
                    "target_object": None,
                    "activity_hint": "standing_free_space",
                    "pose_type": "standing",
                    "source": "room_free_space_v0",
                    "clearance_score": candidate["clearance_score"],
                }
            )
            free_space_spawn_count += 1

    with spawn_output_path.open("w", encoding="utf-8") as f:
        json.dump({"spawn_poses": spawn_poses}, f, indent=2)

    total_by_room = {}
    for pose in spawn_poses:
        room_name = pose.get("room")
        total_by_room[room_name] = total_by_room.get(room_name, 0) + 1

    logger.info("Rooms discovered: %d", len(rooms))
    logger.info("Semantic/object-adjacent spawn poses: %d", semantic_spawn_count)
    logger.info("Free-space spawn poses: %d", free_space_spawn_count)
    logger.info("Total spawn poses per room: %s", total_by_room)
    logger.info(
        "Free-space spawn poses per room: %s",
        {
            room_name: len(candidates)
            for room_name, candidates in sorted(free_space_by_room.items())
        },
    )
    logger.info("Wrote %d human spawn poses to %s", len(spawn_poses), spawn_output_path)
    _export_scene_collision_metadata(output_folder, rooms, obstacle_records, spawn_poses)
    return spawn_output_path
