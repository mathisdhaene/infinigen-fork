import argparse
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon, mapping, shape


CAPSULE_RADIUS_M = 0.35
CAPSULE_HEIGHT_M = 1.80
DEFAULT_BEDLAM_NPZ_ROOT = Path(
    "/media/mathis/PANO/BEDLAM2/animations/training/b2_motions_npz_training/motions_npz_training"
)
DEFAULT_BATCH_SUMMARY_NAME = "batch_collision_results.json"
DEFAULT_NPZ_TRANS_GROUND_AXES = "xz"
ROOT_EXPORT_SCHEMA_VERSION = 2


def _load_metadata(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _room_polygon(room_record):
    poly = room_record.get("floor_polygon_world_xy")
    if not poly:
        return None
    poly_type = poly.get("type")
    if poly_type == "Polygon":
        return Polygon(poly["exterior"], holes=poly.get("interiors") or None)
    if poly_type == "MultiPolygon":
        return shape(poly)
    raise ValueError(f"Unsupported room polygon type: {poly_type}")


def _obstacle_polygon(obstacle_record):
    footprint = obstacle_record.get("footprint_world_xy")
    if not footprint:
        return None
    return Polygon(footprint)


def _point_xy(position_xyz):
    return (float(position_xyz[0]), float(position_xyz[1]))


def _forward_xy(yaw_radians):
    yaw = float(yaw_radians or 0.0)
    return (math.cos(yaw), math.sin(yaw))


def _rotate_xy(point_xy, yaw_radians):
    c = math.cos(float(yaw_radians))
    s = math.sin(float(yaw_radians))
    x, y = float(point_xy[0]), float(point_xy[1])
    return (c * x - s * y, s * x + c * y)


def _offset_point(point_xy, direction_xy, distance_m):
    return (
        float(point_xy[0]) + float(direction_xy[0]) * float(distance_m),
        float(point_xy[1]) + float(direction_xy[1]) * float(distance_m),
    )


def _swept_capsule_polygon(points_xy, radius_m):
    if len(points_xy) == 1:
        return Point(points_xy[0]).buffer(radius_m)
    return LineString(points_xy).buffer(radius_m, cap_style=1, join_style=1)


def _find_room(metadata, room_name):
    for room in metadata.get("rooms", []):
        if room.get("name") == room_name:
            return room
    return None


def _find_room_obstacles(metadata, room_name):
    return [
        obstacle
        for obstacle in metadata.get("obstacles", [])
        if obstacle.get("room") == room_name
    ]


def _nearest_obstacle(
    obstacles,
    origin_xy,
    excluded_names=None,
    z_band_min=None,
    z_band_max=None,
):
    excluded_names = set(excluded_names or [])
    best = None
    best_dist = None
    for obstacle in obstacles:
        if obstacle.get("object_name") in excluded_names:
            continue
        if z_band_min is not None and z_band_max is not None:
            z_min = obstacle.get("z_min")
            z_max = obstacle.get("z_max")
            if z_min is not None and z_max is not None:
                z_min = float(z_min)
                z_max = float(z_max)
                if z_max < float(z_band_min) or z_min > float(z_band_max):
                    continue
        poly = _obstacle_polygon(obstacle)
        if poly is None or poly.is_empty:
            continue
        center = poly.centroid
        dist = center.distance(Point(origin_xy))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = obstacle
    return best


def _trajectory_stationary(spawn_xy, _forward_xy_vec, _room_poly, _obstacles, _target_name):
    return [spawn_xy]


def _trajectory_small_motion(spawn_xy, forward_xy_vec, _room_poly, _obstacles, _target_name):
    return [
        spawn_xy,
        _offset_point(spawn_xy, forward_xy_vec, 0.6),
        _offset_point(spawn_xy, forward_xy_vec, -0.2),
    ]


def _trajectory_exit_room(spawn_xy, forward_xy_vec, room_poly, _obstacles, _target_name):
    step = max(4.0, math.sqrt(room_poly.area))
    return [spawn_xy, _offset_point(spawn_xy, forward_xy_vec, step)]


def _trajectory_cross_obstacle(
    spawn_xy,
    forward_xy_vec,
    room_poly,
    obstacles,
    target_name,
    spawn_z,
    capsule_height_m,
):
    obstacle = _nearest_obstacle(
        obstacles,
        spawn_xy,
        excluded_names={target_name},
        z_band_min=spawn_z,
        z_band_max=spawn_z + capsule_height_m,
    )
    if obstacle is None:
        return [
            spawn_xy,
            _offset_point(spawn_xy, forward_xy_vec, max(2.0, math.sqrt(room_poly.area) / 2.0)),
        ]
    poly = _obstacle_polygon(obstacle)
    center = poly.centroid
    direction = (center.x - spawn_xy[0], center.y - spawn_xy[1])
    norm = math.hypot(direction[0], direction[1])
    if norm < 1e-8:
        direction = forward_xy_vec
        norm = math.hypot(direction[0], direction[1]) or 1.0
    unit = (direction[0] / norm, direction[1] / norm)
    end = _offset_point((center.x, center.y), unit, 0.75)
    return [spawn_xy, (center.x, center.y), end]


def _axis_angle_to_rotation_matrix(axis_angle):
    vec = np.asarray(axis_angle, dtype=float).reshape(3)
    theta = float(np.linalg.norm(vec))
    if theta < 1e-10:
        return np.eye(3, dtype=float)
    axis = vec / theta
    x, y, z = axis
    c = math.cos(theta)
    s = math.sin(theta)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=float,
    )


def _yaw_from_axis_angle(axis_angle):
    rot = _axis_angle_to_rotation_matrix(axis_angle)
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def _extract_global_orient(poses_array):
    poses_array = np.asarray(poses_array, dtype=float)
    if poses_array.ndim != 2 or poses_array.shape[1] < 3:
        return np.zeros((poses_array.shape[0], 3), dtype=float)
    return poses_array[:, :3]


def _extract_trans_ground_components(trans_xyz, ground_axes):
    ground_axes = str(ground_axes or DEFAULT_NPZ_TRANS_GROUND_AXES).lower()
    if ground_axes == "xy":
        return (
            np.asarray(trans_xyz[:, 0], dtype=float),
            np.asarray(trans_xyz[:, 1], dtype=float),
            np.asarray(trans_xyz[:, 2], dtype=float),
            "z",
        )
    if ground_axes == "xz":
        return (
            np.asarray(trans_xyz[:, 0], dtype=float),
            np.asarray(trans_xyz[:, 2], dtype=float),
            np.asarray(trans_xyz[:, 1], dtype=float),
            "y",
        )
    raise RuntimeError(f"Unsupported npz trans ground axes: {ground_axes}")


def export_motion_root_from_npz(
    asset_id,
    output_path,
    npz_root=DEFAULT_BEDLAM_NPZ_ROOT,
    npz_trans_ground_axes=DEFAULT_NPZ_TRANS_GROUND_AXES,
):
    npz_path = Path(npz_root) / f"{asset_id}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"BEDLAM motion NPZ not found for asset_id={asset_id}: {npz_path}")

    payload = np.load(npz_path)
    trans = np.asarray(payload["trans"], dtype=float)
    poses = np.asarray(payload["poses"], dtype=float)
    mocap_frame_rate = float(payload.get("mocap_frame_rate", 30.0))
    if trans.ndim != 2 or trans.shape[1] < 3:
        raise RuntimeError(f"Unexpected trans shape in {npz_path}: {trans.shape}")

    global_orient = _extract_global_orient(poses)
    root_yaws = np.asarray([_yaw_from_axis_angle(frame) for frame in global_orient], dtype=float)

    ground_x, ground_y, vertical, vertical_axis = _extract_trans_ground_components(
        trans[:, :3],
        npz_trans_ground_axes,
    )
    local_ground_x = np.asarray(ground_x, dtype=float) - float(ground_x[0])
    local_ground_y = np.asarray(ground_y, dtype=float) - float(ground_y[0])
    local_vertical = np.asarray(vertical, dtype=float) - float(vertical[0])
    initial_yaw = float(root_yaws[0]) if root_yaws.size else 0.0
    c = math.cos(-initial_yaw)
    s = math.sin(-initial_yaw)
    rotated_x = c * local_ground_x - s * local_ground_y
    rotated_y = s * local_ground_x + c * local_ground_y
    normalized_yaw = root_yaws - initial_yaw

    frames = []
    for frame_idx in range(len(local_ground_x)):
        frames.append(
            {
                "frame_index": int(frame_idx),
                "time_sec": float(frame_idx / mocap_frame_rate),
                "root_x_m": float(rotated_x[frame_idx]),
                "root_y_m": float(rotated_y[frame_idx]),
                "root_z_m": float(local_vertical[frame_idx]),
                "root_yaw_rad": float(normalized_yaw[frame_idx]),
            }
        )

    exported = {
        "source": {
            "type": "BEDLAM/AMASS npz",
            "asset_id": asset_id,
            "npz_path": str(npz_path),
        },
        "root_export_schema_version": int(ROOT_EXPORT_SCHEMA_VERSION),
        "normalization": {
            "first_frame_translation_subtracted": True,
            "first_frame_yaw_neutralized": True,
            "initial_root_yaw_rad": initial_yaw,
        },
        "trans_ground_axes": str(npz_trans_ground_axes),
        "vertical_axis": str(vertical_axis),
        "frame_count": int(len(local_ground_x)),
        "mocap_frame_rate": mocap_frame_rate,
        "frames": frames,
    }
    _write_json(output_path, exported)
    return Path(output_path), exported


def _motion_root_metrics(exported_payload):
    frames = exported_payload.get("frames", [])
    frame_count = int(exported_payload.get("frame_count", len(frames)))
    fps = float(exported_payload.get("mocap_frame_rate", 30.0))
    duration = float(frame_count / fps) if fps > 0.0 else None
    displacement = 0.0
    if len(frames) >= 2:
        first = frames[0]
        last = frames[-1]
        dx = float(last["root_x_m"]) - float(first["root_x_m"])
        dy = float(last["root_y_m"]) - float(first["root_y_m"])
        dz = float(last.get("root_z_m", 0.0)) - float(first.get("root_z_m", 0.0))
        displacement = float(math.sqrt(dx * dx + dy * dy + dz * dz))
    return {
        "frame_count": frame_count,
        "duration": duration,
        "root_displacement_m": displacement,
    }


def _available_motion_asset_ids(npz_root):
    npz_root = Path(npz_root)
    return sorted(path.stem for path in npz_root.glob("*.npz"))


def export_motion_roots_dir(
    output_dir,
    motion_asset_ids=None,
    max_motions=None,
    npz_root=DEFAULT_BEDLAM_NPZ_ROOT,
    npz_trans_ground_axes=DEFAULT_NPZ_TRANS_GROUND_AXES,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    asset_ids = list(motion_asset_ids or [])
    if not asset_ids:
        asset_ids = _available_motion_asset_ids(npz_root)
        if max_motions is not None:
            asset_ids = asset_ids[: int(max_motions)]
    elif max_motions is not None:
        asset_ids = asset_ids[: int(max_motions)]

    index_rows = []
    for asset_id in asset_ids:
        output_json = output_dir / f"{asset_id}_root_trajectory.json"
        exported_path, exported_payload = export_motion_root_from_npz(
            asset_id,
            output_json,
            npz_root=npz_root,
            npz_trans_ground_axes=npz_trans_ground_axes,
        )
        metrics = _motion_root_metrics(exported_payload)
        index_rows.append(
            {
                "asset_id": asset_id,
                "npz_path": exported_payload["source"]["npz_path"],
                "frame_count": metrics["frame_count"],
                "duration": metrics["duration"],
                "root_displacement_m": metrics["root_displacement_m"],
                "output_json": str(exported_path),
            }
        )

    index_payload = {
        "npz_root": str(npz_root),
        "npz_trans_ground_axes": str(npz_trans_ground_axes),
        "motion_count": len(index_rows),
        "motions": index_rows,
    }
    index_path = output_dir / "motion_root_index.json"
    _write_json(index_path, index_payload)
    return index_path, index_payload


def _load_motion_root_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise RuntimeError(f"No frames found in motion root trajectory: {path}")
    return payload


def _motion_id_from_payload(payload, fallback_path=None):
    source = payload.get("source", {})
    motion_id = source.get("asset_id")
    if motion_id:
        return str(motion_id)
    if fallback_path is not None:
        return Path(fallback_path).stem
    return "unknown_motion"


def _collect_motion_root_paths(motion_root_args=None, motion_root_dir=None):
    paths = []
    for path in motion_root_args or []:
        p = Path(path)
        if p not in paths:
            paths.append(p)
    if motion_root_dir is not None:
        root_dir = Path(motion_root_dir)
        for p in sorted(root_dir.glob("*.json")):
            if p.name == "motion_root_index.json":
                continue
            if not p.stem.endswith("_root_trajectory") and "motion_root" not in p.stem:
                continue
            if p not in paths:
                paths.append(p)
    return paths


def _transform_motion_root_to_spawn(frames, spawn_xy, spawn_yaw_rad):
    points_xy = []
    transformed_frames = []
    for frame in frames:
        local_xy = (float(frame["root_x_m"]), float(frame["root_y_m"]))
        rotated = _rotate_xy(local_xy, spawn_yaw_rad)
        world_xy = (float(spawn_xy[0]) + rotated[0], float(spawn_xy[1]) + rotated[1])
        points_xy.append(world_xy)
        transformed_frames.append(
            {
                "frame_index": int(frame["frame_index"]),
                "time_sec": float(frame["time_sec"]),
                "world_x_m": float(world_xy[0]),
                "world_y_m": float(world_xy[1]),
                "world_z_m": float(frame.get("root_z_m", 0.0)),
                "world_yaw_rad": float(spawn_yaw_rad + float(frame.get("root_yaw_rad", 0.0))),
            }
        )
    return points_xy, transformed_frames


def _validate_trajectory(
    name,
    points_xy,
    room_poly,
    eroded_room_poly,
    obstacles,
    radius_m,
    height_band_m,
    base_z_m,
    target_name=None,
    allow_target_object=False,
):
    swept = _swept_capsule_polygon(points_xy, radius_m)
    reasons = []
    band_z_min = float(base_z_m)
    band_z_max = float(base_z_m) + float(height_band_m)

    if eroded_room_poly.is_empty:
        reasons.append("room erosion by capsule radius produced empty polygon")
    elif not eroded_room_poly.covers(swept):
        reasons.append("swept capsule exits room polygon eroded by radius")

    collisions = []
    for obstacle in obstacles:
        if allow_target_object and target_name and obstacle.get("object_name") == target_name:
            continue
        poly = _obstacle_polygon(obstacle)
        if poly is None or poly.is_empty:
            continue
        z_min = obstacle.get("z_min")
        z_max = obstacle.get("z_max")
        if z_min is not None and z_max is not None:
            z_min = float(z_min)
            z_max = float(z_max)
            if z_max < band_z_min or z_min > band_z_max:
                continue
        if swept.intersects(poly):
            collisions.append(
                {
                    "object_name": obstacle.get("object_name"),
                    "category_hint": obstacle.get("category_hint"),
                }
            )

    if collisions:
        reasons.append(
            "swept capsule intersects obstacles: "
            + ", ".join(
                f"{c['object_name']} ({c['category_hint']})" for c in collisions[:8]
            )
        )

    return {
        "name": name,
        "points_xy": [[float(x), float(y)] for x, y in points_xy],
        "swept_polygon": swept,
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "collisions": collisions,
        "allow_target_object": allow_target_object,
        "z_band_m": [band_z_min, band_z_max],
    }


def _geojson_feature(geometry, properties):
    return {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": properties,
    }


def _trajectory_geometry(points_xy):
    if len(points_xy) == 1:
        return Point(points_xy[0])
    return LineString(points_xy)


def _save_debug_geojson(output_path, room_poly, eroded_room_poly, obstacles, spawn_xy, result):
    features = [
        _geojson_feature(room_poly, {"kind": "room_polygon"}),
        _geojson_feature(eroded_room_poly, {"kind": "room_polygon_eroded"}),
        _geojson_feature(Point(spawn_xy), {"kind": "spawn_point"}),
        _geojson_feature(_trajectory_geometry(result["points_xy"]), {"kind": "trajectory"}),
        _geojson_feature(result["swept_polygon"], {"kind": "swept_capsule", "pass": result["pass"]}),
    ]
    for obstacle in obstacles:
        poly = _obstacle_polygon(obstacle)
        if poly is None or poly.is_empty:
            continue
        features.append(
            _geojson_feature(
                poly,
                {
                    "kind": "obstacle",
                    "object_name": obstacle.get("object_name"),
                    "category_hint": obstacle.get("category_hint"),
                },
            )
        )
    payload = {"type": "FeatureCollection", "features": features}
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_debug_png(output_path, room_poly, eroded_room_poly, obstacles, spawn_xy, result):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(8, 8))

    def draw_polygon(poly, color, label, alpha=0.2, linewidth=2):
        if poly.is_empty:
            return
        if poly.geom_type == "Polygon":
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=alpha, color=color, label=label)
            ax.plot(xs, ys, color=color, linewidth=linewidth)
        else:
            for geom in getattr(poly, "geoms", []):
                draw_polygon(geom, color, label, alpha=alpha, linewidth=linewidth)

    draw_polygon(room_poly, "lightgray", "room", alpha=0.25)
    draw_polygon(eroded_room_poly, "lightblue", "eroded_room", alpha=0.2)
    for obstacle in obstacles:
        poly = _obstacle_polygon(obstacle)
        if poly is None or poly.is_empty:
            continue
        draw_polygon(poly, "salmon", obstacle.get("category_hint") or "obstacle", alpha=0.35, linewidth=1)

    swept = result["swept_polygon"]
    draw_polygon(swept, "green" if result["pass"] else "red", "swept_capsule", alpha=0.25)
    traj = _trajectory_geometry(result["points_xy"])
    if traj.geom_type == "Point":
        ax.scatter([traj.x], [traj.y], color="black", s=20, label="trajectory")
    else:
        xs, ys = traj.xy
        ax.plot(xs, ys, color="black", linewidth=2, linestyle="--", label="trajectory")
    ax.scatter([spawn_xy[0]], [spawn_xy[1]], color="blue", s=30, label="spawn")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{result['name']} - {'PASS' if result['pass'] else 'FAIL'}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _run_synthetic_tests(metadata, spawn_index, output_dir):
    spawn_pose = metadata["spawn_poses"][spawn_index]
    room_name = spawn_pose.get("room")
    room_record = _find_room(metadata, room_name)
    if room_record is None:
        raise RuntimeError(f"Could not find room '{room_name}' for spawn pose {spawn_index}")

    room_poly = _room_polygon(room_record)
    if room_poly is None or room_poly.is_empty:
        raise RuntimeError(f"Room '{room_name}' has no usable floor polygon")

    spawn_xy = _point_xy(spawn_pose["position_xyz"])
    target_name = spawn_pose.get("target_object")
    spawn_z = float(spawn_pose["position_xyz"][2])
    obstacles = _find_room_obstacles(metadata, room_name)
    eroded_room_poly = room_poly.buffer(-CAPSULE_RADIUS_M)
    forward_xy_vec = _forward_xy(spawn_pose.get("yaw"))

    synthetic_trajectories = [
        (
            "stationary_at_spawn",
            _trajectory_stationary(spawn_xy, forward_xy_vec, room_poly, obstacles, target_name),
            True,
        ),
        (
            "small_forward_backward_motion",
            _trajectory_small_motion(spawn_xy, forward_xy_vec, room_poly, obstacles, target_name),
            True,
        ),
        (
            "long_motion_exiting_room",
            _trajectory_exit_room(spawn_xy, forward_xy_vec, room_poly, obstacles, target_name),
            True,
        ),
        (
            "motion_crossing_obstacle",
            _trajectory_cross_obstacle(
                spawn_xy,
                forward_xy_vec,
                room_poly,
                obstacles,
                target_name,
                spawn_z,
                CAPSULE_HEIGHT_M,
            ),
            False,
        ),
    ]

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, points_xy, allow_target in synthetic_trajectories:
        result = _validate_trajectory(
            name=name,
            points_xy=points_xy,
            room_poly=room_poly,
            eroded_room_poly=eroded_room_poly,
            obstacles=obstacles,
            radius_m=CAPSULE_RADIUS_M,
            height_band_m=CAPSULE_HEIGHT_M,
            base_z_m=spawn_z,
            target_name=target_name,
            allow_target_object=allow_target,
        )
        results.append(result)
        geojson_path = output_dir / f"{name}.geojson"
        _save_debug_geojson(
            geojson_path,
            room_poly=room_poly,
            eroded_room_poly=eroded_room_poly,
            obstacles=obstacles,
            spawn_xy=spawn_xy,
            result=result,
        )
        _save_debug_png(
            output_dir / f"{name}.png",
            room_poly=room_poly,
            eroded_room_poly=eroded_room_poly,
            obstacles=obstacles,
            spawn_xy=spawn_xy,
            result=result,
        )
    return {
        "spawn_pose": spawn_pose,
        "room_name": room_name,
        "capsule_radius_m": CAPSULE_RADIUS_M,
        "capsule_height_m": CAPSULE_HEIGHT_M,
        "obstacle_count": len(obstacles),
        "results": results,
    }


def _run_motion_root_test(metadata, spawn_index, output_dir, motion_root_payload):
    spawn_pose = metadata["spawn_poses"][spawn_index]
    room_name = spawn_pose.get("room")
    room_record = _find_room(metadata, room_name)
    if room_record is None:
        raise RuntimeError(f"Could not find room '{room_name}' for spawn pose {spawn_index}")

    room_poly = _room_polygon(room_record)
    if room_poly is None or room_poly.is_empty:
        raise RuntimeError(f"Room '{room_name}' has no usable floor polygon")

    spawn_xy = _point_xy(spawn_pose["position_xyz"])
    spawn_z = float(spawn_pose["position_xyz"][2])
    spawn_yaw = float(spawn_pose.get("yaw") or 0.0)
    target_name = spawn_pose.get("target_object")
    obstacles = _find_room_obstacles(metadata, room_name)
    eroded_room_poly = room_poly.buffer(-CAPSULE_RADIUS_M)

    result = _evaluate_motion_root_pair(
        metadata=metadata,
        spawn_index=spawn_index,
        motion_root_payload=motion_root_payload,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_debug_geojson(
        output_dir / "bedlam_motion_root.geojson",
        room_poly=result["room_polygon"],
        eroded_room_poly=result["eroded_room_polygon"],
        obstacles=result["room_obstacles"],
        spawn_xy=result["spawn_xy"],
        result=result["trajectory_result"],
    )
    _save_debug_png(
        output_dir / "bedlam_motion_root.png",
        room_poly=result["room_polygon"],
        eroded_room_poly=result["eroded_room_polygon"],
        obstacles=result["room_obstacles"],
        spawn_xy=result["spawn_xy"],
        result=result["trajectory_result"],
    )
    _write_json(
        output_dir / "bedlam_motion_root_world.json",
        {"frames": result["trajectory_result"]["transformed_frames"]},
    )

    return {
        "spawn_pose": result["spawn_pose"],
        "room_name": result["room_name"],
        "capsule_radius_m": CAPSULE_RADIUS_M,
        "capsule_height_m": CAPSULE_HEIGHT_M,
        "obstacle_count": len(result["room_obstacles"]),
        "result": result["trajectory_result"],
    }


def _sanitize_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _maybe_write_pair_debug(output_dir, evaluation, write_debug):
    if write_debug == "none":
        return None
    trajectory_result = evaluation["trajectory_result"]
    if write_debug == "failed_only" and trajectory_result["pass"]:
        return None

    motion_id = _motion_id_from_payload(trajectory_result.get("source_motion", {}))
    spawn_index = evaluation["spawn_index"]
    pair_dir = output_dir / f"spawn_{spawn_index:03d}" / _sanitize_name(motion_id)
    pair_dir.mkdir(parents=True, exist_ok=True)
    _save_debug_geojson(
        pair_dir / "bedlam_motion_root.geojson",
        room_poly=evaluation["room_polygon"],
        eroded_room_poly=evaluation["eroded_room_polygon"],
        obstacles=evaluation["room_obstacles"],
        spawn_xy=evaluation["spawn_xy"],
        result=trajectory_result,
    )
    _save_debug_png(
        pair_dir / "bedlam_motion_root.png",
        room_poly=evaluation["room_polygon"],
        eroded_room_poly=evaluation["eroded_room_polygon"],
        obstacles=evaluation["room_obstacles"],
        spawn_xy=evaluation["spawn_xy"],
        result=trajectory_result,
    )
    _write_json(
        pair_dir / "bedlam_motion_root_world.json",
        {"frames": trajectory_result["transformed_frames"]},
    )
    return pair_dir


def _run_batch_motion_root_tests(
    metadata,
    motion_root_paths,
    output_dir,
    write_debug="failed_only",
):
    motion_payloads = []
    for motion_root_path in motion_root_paths:
        payload = _load_motion_root_json(motion_root_path)
        motion_payloads.append((Path(motion_root_path), payload))

    pair_rows = []
    valid_pairs = []
    invalid_pairs = []
    for motion_root_path, payload in motion_payloads:
        motion_id = _motion_id_from_payload(payload, fallback_path=motion_root_path)
        for spawn_index, spawn_pose in enumerate(metadata.get("spawn_poses", [])):
            evaluation = _evaluate_motion_root_pair(metadata, spawn_index, payload)
            evaluation["spawn_index"] = spawn_index
            pair_debug_dir = _maybe_write_pair_debug(output_dir, evaluation, write_debug)
            trajectory_result = evaluation["trajectory_result"]
            row = {
                "motion_id": motion_id,
                "motion_root_path": str(motion_root_path),
                "spawn_index": int(spawn_index),
                "room": spawn_pose.get("room"),
                "activity_hint": spawn_pose.get("activity_hint"),
                "pass": bool(trajectory_result["pass"]),
                "reasons": list(trajectory_result["reasons"]),
                "debug_dir": None if pair_debug_dir is None else str(pair_debug_dir),
            }
            pair_rows.append(row)
            if row["pass"]:
                valid_pairs.append(row)
            else:
                invalid_pairs.append(row)

    summary = {
        "scene_path": None,
        "number_of_spawn_poses": len(metadata.get("spawn_poses", [])),
        "number_of_motions": len(motion_payloads),
        "capsule_radius_m": CAPSULE_RADIUS_M,
        "capsule_height_m": CAPSULE_HEIGHT_M,
        "valid_pair_count": len(valid_pairs),
        "invalid_pair_count": len(invalid_pairs),
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "pairs": pair_rows,
    }
    return summary


def _print_batch_table(summary):
    print(
        "motion_id | spawn_index | room | activity_hint | PASS/FAIL | reason"
    )
    for row in summary["pairs"]:
        verdict = "PASS" if row["pass"] else "FAIL"
        reason = "; ".join(row["reasons"]) if row["reasons"] else "ok"
        print(
            f"{row['motion_id']} | {row['spawn_index']} | {row['room']} | "
            f"{row['activity_hint']} | {verdict} | {reason}"
        )


def _evaluate_motion_root_pair(metadata, spawn_index, motion_root_payload):
    spawn_pose = metadata["spawn_poses"][spawn_index]
    room_name = spawn_pose.get("room")
    room_record = _find_room(metadata, room_name)
    if room_record is None:
        raise RuntimeError(f"Could not find room '{room_name}' for spawn pose {spawn_index}")

    room_poly = _room_polygon(room_record)
    if room_poly is None or room_poly.is_empty:
        raise RuntimeError(f"Room '{room_name}' has no usable floor polygon")

    spawn_xy = _point_xy(spawn_pose["position_xyz"])
    spawn_z = float(spawn_pose["position_xyz"][2])
    spawn_yaw = float(spawn_pose.get("yaw") or 0.0)
    target_name = spawn_pose.get("target_object")
    obstacles = _find_room_obstacles(metadata, room_name)
    eroded_room_poly = room_poly.buffer(-CAPSULE_RADIUS_M)

    points_xy, transformed_frames = _transform_motion_root_to_spawn(
        motion_root_payload["frames"], spawn_xy, spawn_yaw
    )
    trajectory_result = _validate_trajectory(
        name="bedlam_motion_root",
        points_xy=points_xy,
        room_poly=room_poly,
        eroded_room_poly=eroded_room_poly,
        obstacles=obstacles,
        radius_m=CAPSULE_RADIUS_M,
        height_band_m=CAPSULE_HEIGHT_M,
        base_z_m=spawn_z,
        target_name=target_name,
        allow_target_object=True,
    )
    trajectory_result["transformed_frames"] = transformed_frames
    trajectory_result["source_motion"] = {
        "asset_id": motion_root_payload.get("source", {}).get("asset_id"),
        "npz_path": motion_root_payload.get("source", {}).get("npz_path"),
        "frame_count": motion_root_payload.get("frame_count"),
        "mocap_frame_rate": motion_root_payload.get("mocap_frame_rate"),
    }
    return {
        "spawn_pose": spawn_pose,
        "room_name": room_name,
        "room_polygon": room_poly,
        "eroded_room_polygon": eroded_room_poly,
        "room_obstacles": obstacles,
        "spawn_xy": spawn_xy,
        "trajectory_result": trajectory_result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs/indoors/human_spawn_poc/scene_collision_metadata.json"),
    )
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/indoors/human_spawn_poc/collision_debug"),
    )
    parser.add_argument("--motion-root", type=Path, action="append", default=None)
    parser.add_argument("--motion-root-dir", type=Path, default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument(
        "--write-debug",
        choices=("none", "failed_only", "all"),
        default="failed_only",
    )
    parser.add_argument("--export-motion-root", type=Path, default=None)
    parser.add_argument("--export-motion-roots-dir", type=Path, default=None)
    parser.add_argument("--motion-asset-id", type=str, default=None)
    parser.add_argument("--max-motions", type=int, default=None)
    parser.add_argument("--npz-root", type=Path, default=DEFAULT_BEDLAM_NPZ_ROOT)
    parser.add_argument(
        "--npz-trans-ground-axes",
        choices=("xy", "xz"),
        default=DEFAULT_NPZ_TRANS_GROUND_AXES,
    )
    args = parser.parse_args()

    if args.export_motion_roots_dir is not None:
        index_path, index_payload = export_motion_roots_dir(
            args.export_motion_roots_dir,
            motion_asset_ids=args.motion_asset_id,
            max_motions=args.max_motions,
            npz_root=args.npz_root,
            npz_trans_ground_axes=args.npz_trans_ground_axes,
        )
        print(f"Exported motion root trajectories: {args.export_motion_roots_dir}")
        print(f"Motion count: {index_payload['motion_count']}")
        print(f"Index file: {index_path}")
        return

    if args.export_motion_root is not None:
        if not args.motion_asset_id:
            raise RuntimeError("--motion-asset-id is required with --export-motion-root")
        exported_path, exported_payload = export_motion_root_from_npz(
            args.motion_asset_id,
            args.export_motion_root,
            npz_root=args.npz_root,
            npz_trans_ground_axes=args.npz_trans_ground_axes,
        )
        print(f"Exported motion root trajectory: {exported_path}")
        print(f"Asset id: {exported_payload['source']['asset_id']}")
        print(f"Frame count: {exported_payload['frame_count']}")
        print(f"Mocap FPS: {exported_payload['mocap_frame_rate']}")
        return

    metadata = _load_metadata(args.metadata)
    motion_root_paths = _collect_motion_root_paths(args.motion_root, args.motion_root_dir)
    if args.batch:
        if not motion_root_paths:
            raise RuntimeError("--batch requires --motion-root and/or --motion-root-dir")
        summary = _run_batch_motion_root_tests(
            metadata,
            motion_root_paths,
            args.output_dir,
            write_debug=args.write_debug,
        )
        summary["scene_path"] = str(args.metadata)
        summary_path = args.output_dir / DEFAULT_BATCH_SUMMARY_NAME
        _write_json(summary_path, summary)
        print(f"Scene metadata: {args.metadata}")
        print(f"Spawn poses: {summary['number_of_spawn_poses']}")
        print(f"Motions: {summary['number_of_motions']}")
        print(f"Valid pairs: {summary['valid_pair_count']}")
        print(f"Invalid pairs: {summary['invalid_pair_count']}")
        print(f"Summary JSON: {summary_path}")
        print("")
        _print_batch_table(summary)
        return

    if motion_root_paths:
        motion_root_payload = _load_motion_root_json(motion_root_paths[0])
        report = _run_motion_root_test(metadata, args.spawn_index, args.output_dir, motion_root_payload)
        result = report["result"]
        print(f"Scene metadata: {args.metadata}")
        print(f"Motion root: {motion_root_paths[0]}")
        print(f"Spawn pose room: {report['room_name']}")
        print(f"Capsule radius: {report['capsule_radius_m']:.2f} m")
        print(f"Capsule height band: {report['capsule_height_m']:.2f} m")
        print(f"Room obstacle count: {report['obstacle_count']}")
        print("")
        verdict = "PASS" if result["pass"] else "FAIL"
        print(f"[{verdict}] {result['name']}")
        if result["reasons"]:
            for reason in result["reasons"]:
                print(f"  - {reason}")
        else:
            print("  - no room-boundary or obstacle collisions detected")
        print(f"  - source_motion={result['source_motion']}")
        print(f"  - debug: {args.output_dir / 'bedlam_motion_root.geojson'}")
        return

    report = _run_synthetic_tests(metadata, args.spawn_index, args.output_dir)

    print(f"Scene metadata: {args.metadata}")
    print(f"Spawn pose room: {report['room_name']}")
    print(f"Capsule radius: {report['capsule_radius_m']:.2f} m")
    print(f"Capsule height band: {report['capsule_height_m']:.2f} m")
    print(f"Room obstacle count: {report['obstacle_count']}")
    print("")
    for result in report["results"]:
        verdict = "PASS" if result["pass"] else "FAIL"
        print(f"[{verdict}] {result['name']}")
        if result["reasons"]:
            for reason in result["reasons"]:
                print(f"  - {reason}")
        else:
            print("  - no room-boundary or obstacle collisions detected")
        print(f"  - allow_target_object={result['allow_target_object']}")
        print(f"  - debug: {args.output_dir / (result['name'] + '.geojson')}")
        print("")


if __name__ == "__main__":
    main()
