import argparse
import copy
import json
import math
import random
import time
from collections import Counter, defaultdict
from itertools import combinations, permutations
from pathlib import Path
from statistics import median

from shapely.geometry import Point

from infinigen_examples.validate_scene_collision_metadata import (
    CAPSULE_HEIGHT_M,
    CAPSULE_RADIUS_M,
    DEFAULT_BEDLAM_NPZ_ROOT,
    DEFAULT_NPZ_TRANS_GROUND_AXES,
    ROOT_EXPORT_SCHEMA_VERSION,
    _collect_motion_root_paths,
    _evaluate_motion_root_pair,
    _find_room,
    _find_room_obstacles,
    _load_metadata,
    _load_motion_root_json,
    _motion_id_from_payload,
    _obstacle_polygon,
    _room_polygon,
    _transform_motion_root_to_spawn,
    _write_json,
    export_motion_root_from_npz,
)

MOTION_SET_MODE = "starterpack_only"
STARTERPACK_WHITELIST_PATH = Path(
    "/media/mathis/PANO/bedlam2_render/config/whitelist_animations_starterpack.json"
)
MIN_HUMAN_WALL_CLEARANCE_M = 0.8
MIN_HUMAN_OBSTACLE_CLEARANCE_M = 0.25

PATH_LENGTH_BINS_M = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf"))
DEFAULT_CAMERA_DISTANCE_BINS_M = (0.0, 1.5, 3.0, 999.0)
DEFAULT_AZIMUTH_BIN_COUNT = 8
DEFAULT_AZIMUTH_BIN_COUNT_FINE = 12
DEFAULT_SCALE_BIN_THRESHOLDS_M = (0.75, 1.5, 3.0, 5.0)


def _expected_vertical_axis_for_ground_axes(npz_trans_ground_axes):
    return "y" if str(npz_trans_ground_axes) == "xz" else "z"


def _root_cache_validation(
    motion_id,
    cache_path,
    npz_path,
    requested_axes,
):
    if not Path(cache_path).exists():
        return False, "missing_cache", {}
    try:
        payload = _load_motion_root_json(cache_path)
    except Exception as exc:
        return False, f"unreadable_cache:{exc}", {}
    source = payload.get("source") or {}
    cached_schema = payload.get("root_export_schema_version")
    cached_axes = payload.get("trans_ground_axes")
    cached_vertical_axis = payload.get("vertical_axis")
    cached_motion_id = _normalize_motion_id(
        source.get("asset_id") or _motion_id_from_payload(payload, fallback_path=cache_path)
    )
    cached_npz_path = str(source.get("npz_path") or "")
    requested_npz_path = str(npz_path)
    metadata = {
        "cached_schema_version": cached_schema,
        "cached_trans_ground_axes": cached_axes,
        "cached_vertical_axis": cached_vertical_axis,
        "cached_motion_id": cached_motion_id,
        "cached_npz_path": cached_npz_path,
        "requested_trans_ground_axes": str(requested_axes),
        "requested_npz_path": requested_npz_path,
    }
    if cached_schema is None:
        return False, "missing_schema_version", metadata
    if int(cached_schema) < int(ROOT_EXPORT_SCHEMA_VERSION):
        return False, "stale_schema_version", metadata
    if not cached_axes:
        return False, "missing_trans_ground_axes", metadata
    if str(cached_axes) != str(requested_axes):
        return False, "trans_ground_axes_mismatch", metadata
    expected_vertical_axis = _expected_vertical_axis_for_ground_axes(requested_axes)
    if not cached_vertical_axis:
        return False, "missing_vertical_axis", metadata
    if str(cached_vertical_axis) != expected_vertical_axis:
        return False, "vertical_axis_mismatch", metadata
    if cached_motion_id != _normalize_motion_id(motion_id):
        return False, "motion_id_mismatch", metadata
    if cached_npz_path != requested_npz_path:
        return False, "npz_path_mismatch", metadata
    return True, None, metadata


def _clearance_rejection_summary(
    min_wall_clearance_m,
    min_obstacle_clearance_m,
    wall_frame_clearances,
    obstacle_frame_clearances,
    min_required_wall_clearance_m,
    min_required_obstacle_clearance_m,
):
    rejected_by_wall_clearance = (
        min_wall_clearance_m is not None
        and float(min_wall_clearance_m) < float(min_required_wall_clearance_m)
    )
    rejected_by_obstacle_clearance = (
        min_obstacle_clearance_m is not None
        and float(min_obstacle_clearance_m) < float(min_required_obstacle_clearance_m)
    )
    rejection_frame_indices = sorted(
        {
            *(
                frame_index
                for frame_index, clearance in wall_frame_clearances
                if clearance is not None and clearance < float(min_required_wall_clearance_m)
            ),
            *(
                frame_index
                for frame_index, clearance in obstacle_frame_clearances
                if clearance is not None and clearance < float(min_required_obstacle_clearance_m)
            ),
        }
    )
    return {
        "rejected_by_wall_clearance": bool(rejected_by_wall_clearance),
        "rejected_by_obstacle_clearance": bool(rejected_by_obstacle_clearance),
        "min_required_wall_clearance_m": float(min_required_wall_clearance_m),
        "min_required_obstacle_clearance_m": float(min_required_obstacle_clearance_m),
        "clearance_rejection_frame_indices": rejection_frame_indices,
    }


def _room_summary(room_record):
    if room_record is None:
        return None
    return {
        "name": room_record.get("name"),
        "semantic_tags": list(room_record.get("semantic_tags", [])),
        "floor_z": room_record.get("floor_z"),
        "ceiling_z": room_record.get("ceiling_z"),
        "bbox": room_record.get("bbox"),
        "floor_polygon_world_xy": room_record.get("floor_polygon_world_xy"),
    }


def _spawn_pose_summary(spawn_index, spawn_pose):
    return {
        "spawn_index": int(spawn_index),
        "room": spawn_pose.get("room"),
        "position_xyz": list(spawn_pose.get("position_xyz", [])),
        "yaw": spawn_pose.get("yaw"),
        "target_object": spawn_pose.get("target_object"),
        "activity_hint": spawn_pose.get("activity_hint"),
        "pose_type": spawn_pose.get("pose_type"),
        "source": spawn_pose.get("source"),
    }


def _motion_summary(motion_root_path, payload):
    source = payload.get("source", {})
    return {
        "motion_id": _motion_id_from_payload(payload, fallback_path=motion_root_path),
        "motion_root_path": str(motion_root_path),
        "npz_path": source.get("npz_path"),
        "frame_count": payload.get("frame_count"),
        "mocap_frame_rate": payload.get("mocap_frame_rate"),
    }


def _load_allowed_motion_ids(motion_set_mode, starterpack_whitelist_path):
    motion_set_mode = str(motion_set_mode or "all")
    if motion_set_mode == "all":
        return None, {
            "motion_set_mode": motion_set_mode,
            "source_path": None,
            "motion_id_count": None,
            "motion_ids": None,
            "motion_ids_by_identity": None,
        }
    if motion_set_mode != "starterpack_only":
        raise RuntimeError(f"Unsupported motion set mode: {motion_set_mode}")
    whitelist_path = Path(starterpack_whitelist_path)
    payload = json.loads(whitelist_path.read_text(encoding="utf-8"))
    allowed = set()
    by_identity = {}
    for identity, motions in sorted(payload.items()):
        ids = [f"{identity}_{motion}" for motion in motions]
        by_identity[identity] = ids
        allowed.update(ids)
    return allowed, {
        "motion_set_mode": motion_set_mode,
        "source_path": str(whitelist_path),
        "motion_id_count": len(allowed),
        "motion_ids": sorted(allowed),
        "motion_ids_by_identity": by_identity,
    }


def _normalize_motion_id(value):
    motion_id = str(value or "")
    suffix = "_root_trajectory"
    if motion_id.endswith(suffix):
        return motion_id[: -len(suffix)]
    return motion_id


def _motion_identity(value):
    motion_id = _normalize_motion_id(value)
    parts = motion_id.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else motion_id


def _filter_motion_root_paths_by_allowed_ids(motion_root_paths, allowed_motion_ids):
    if allowed_motion_ids is None:
        return list(motion_root_paths), {"kept": len(motion_root_paths), "dropped": 0, "dropped_motion_ids": []}
    kept = []
    dropped_ids = []
    for motion_root_path in motion_root_paths:
        payload = _load_motion_root_json(motion_root_path)
        motion_id = _motion_id_from_payload(payload, fallback_path=motion_root_path)
        if motion_id in allowed_motion_ids:
            kept.append(Path(motion_root_path))
        else:
            dropped_ids.append(motion_id)
    return kept, {
        "kept": len(kept),
        "dropped": len(dropped_ids),
        "dropped_motion_ids": sorted(dropped_ids),
    }


def _motion_ids_by_identity(motion_ids):
    grouped = defaultdict(list)
    for motion_id in sorted(_normalize_motion_id(motion_id) for motion_id in (motion_ids or [])):
        grouped[_motion_identity(motion_id)].append(motion_id)
    return dict(grouped)


def _select_motion_ids_for_testing(
    motion_ids,
    max_motion_roots_tested=None,
    motion_root_selection_seed=0,
    prefer_identity_diversity=False,
):
    normalized = sorted({_normalize_motion_id(motion_id) for motion_id in (motion_ids or [])})
    if max_motion_roots_tested is None or int(max_motion_roots_tested) <= 0:
        return list(normalized)
    max_motion_roots_tested = int(max_motion_roots_tested)
    if len(normalized) <= max_motion_roots_tested:
        return list(normalized)

    rng = random.Random(int(motion_root_selection_seed))
    if not prefer_identity_diversity:
        shuffled = list(normalized)
        rng.shuffle(shuffled)
        return sorted(shuffled[:max_motion_roots_tested])

    grouped = _motion_ids_by_identity(normalized)
    identity_keys = list(grouped.keys())
    rng.shuffle(identity_keys)
    for identity in identity_keys:
        rng.shuffle(grouped[identity])

    selected = []
    seen = set()
    while len(selected) < max_motion_roots_tested:
        progress = False
        for identity in identity_keys:
            motions = grouped[identity]
            while motions and motions[0] in seen:
                motions.pop(0)
            if not motions:
                continue
            motion_id = motions.pop(0)
            if motion_id in seen:
                continue
            selected.append(motion_id)
            seen.add(motion_id)
            progress = True
            if len(selected) >= max_motion_roots_tested:
                break
        if not progress:
            break
    return sorted(selected)


def _build_auto_motion_root_paths(
    motion_ids,
    output_dir,
    npz_root=DEFAULT_BEDLAM_NPZ_ROOT,
    npz_trans_ground_axes=DEFAULT_NPZ_TRANS_GROUND_AXES,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_paths = []
    missing_npz_motion_ids = []
    exported_motion_ids = []
    root_cache_diagnostics = []
    for motion_id in motion_ids:
        npz_path = Path(npz_root) / f"{motion_id}.npz"
        if not npz_path.is_file():
            missing_npz_motion_ids.append(motion_id)
            continue
        output_json = output_dir / f"{motion_id}_root_trajectory.json"
        root_cache_hit, reexport_reason, cache_metadata = _root_cache_validation(
            motion_id,
            output_json,
            npz_path,
            npz_trans_ground_axes,
        )
        if not root_cache_hit:
            export_motion_root_from_npz(
                motion_id,
                output_json,
                npz_root=npz_root,
                npz_trans_ground_axes=npz_trans_ground_axes,
            )
        root_cache_diagnostics.append(
            {
                "motion_id": motion_id,
                "root_cache_hit": bool(root_cache_hit),
                "root_cache_reexported": not bool(root_cache_hit),
                "root_cache_reexport_reason": reexport_reason,
                **cache_metadata,
                "output_json": str(output_json),
            }
        )
        kept_paths.append(output_json)
        exported_motion_ids.append(motion_id)
    return kept_paths, {
        "requested_motion_ids": len(list(motion_ids)),
        "exported_motion_ids": sorted(exported_motion_ids),
        "exported_count": len(exported_motion_ids),
        "missing_npz_motion_ids": sorted(missing_npz_motion_ids),
        "missing_npz_count": len(missing_npz_motion_ids),
        "npz_root": str(npz_root),
        "npz_trans_ground_axes": str(npz_trans_ground_axes),
        "output_dir": str(output_dir),
        "root_export_schema_version": int(ROOT_EXPORT_SCHEMA_VERSION),
        "root_cache_diagnostics": root_cache_diagnostics,
    }


def _resolve_motion_root_paths(
    explicit_motion_root_paths,
    allowed_motion_ids,
    output_dir,
    motion_set_mode,
    max_motion_roots_tested=None,
    motion_root_selection_seed=0,
    prefer_identity_diversity=False,
    npz_root=DEFAULT_BEDLAM_NPZ_ROOT,
    npz_trans_ground_axes=DEFAULT_NPZ_TRANS_GROUND_AXES,
):
    explicit_motion_root_paths = list(explicit_motion_root_paths or [])
    report = {
        "motion_set_mode": motion_set_mode,
        "explicit_motion_root_count": len(explicit_motion_root_paths),
        "max_motion_roots_tested": None if max_motion_roots_tested is None else int(max_motion_roots_tested),
        "motion_root_selection_seed": int(motion_root_selection_seed),
        "prefer_identity_diversity": bool(prefer_identity_diversity),
        "selection_source": None,
        "selected_motion_ids": [],
        "auto_export": None,
    }

    if explicit_motion_root_paths:
        motion_rows = []
        for motion_root_path in explicit_motion_root_paths:
            payload = _load_motion_root_json(motion_root_path)
            motion_rows.append(
                {
                    "motion_id": _normalize_motion_id(
                        _motion_id_from_payload(payload, fallback_path=motion_root_path)
                    ),
                    "path": Path(motion_root_path),
                }
            )
        available_ids = [row["motion_id"] for row in motion_rows]
        selected_ids = _select_motion_ids_for_testing(
            available_ids,
            max_motion_roots_tested=max_motion_roots_tested,
            motion_root_selection_seed=motion_root_selection_seed,
            prefer_identity_diversity=prefer_identity_diversity,
        )
        selected_set = set(selected_ids)
        selected_paths = [row["path"] for row in motion_rows if row["motion_id"] in selected_set]
        report["selection_source"] = "explicit_motion_roots"
        report["selected_motion_ids"] = selected_ids
        report["selected_count"] = len(selected_paths)
        return selected_paths, report

    if motion_set_mode == "starterpack_only" and allowed_motion_ids:
        selected_ids = _select_motion_ids_for_testing(
            sorted(allowed_motion_ids),
            max_motion_roots_tested=max_motion_roots_tested,
            motion_root_selection_seed=motion_root_selection_seed,
            prefer_identity_diversity=prefer_identity_diversity,
        )
        auto_root_dir = Path(output_dir) / "_auto_motion_roots"
        selected_paths, auto_report = _build_auto_motion_root_paths(
            selected_ids,
            auto_root_dir,
            npz_root=npz_root,
            npz_trans_ground_axes=npz_trans_ground_axes,
        )
        report["selection_source"] = "starterpack_auto_export"
        report["selected_motion_ids"] = selected_ids
        report["selected_count"] = len(selected_paths)
        report["auto_export"] = auto_report
        report["npz_trans_ground_axes"] = str(npz_trans_ground_axes)
        return selected_paths, report

    report["selection_source"] = "none"
    report["selected_count"] = 0
    return [], report


def _filter_valid_by_room_by_allowed_motion_ids(valid_by_room, allowed_motion_ids):
    if allowed_motion_ids is None:
        return dict(valid_by_room)
    return {
        room_name: [
            row for row in rows
            if _normalize_motion_id(row.get("motion_id")) in allowed_motion_ids
        ]
        for room_name, rows in valid_by_room.items()
    }


def _filter_valid_groups_by_room_by_allowed_motion_ids(valid_groups_by_room, allowed_motion_ids):
    if allowed_motion_ids is None:
        return dict(valid_groups_by_room)
    return {
        room_name: [
            group for group in groups
            if all(
                _normalize_motion_id(human.get("motion_id")) in allowed_motion_ids
                for human in group.get("humans", [])
            )
        ]
        for room_name, groups in valid_groups_by_room.items()
    }


def _motion_distribution_from_single_pairs(valid_by_room):
    counts = defaultdict(int)
    by_identity = defaultdict(int)
    for rows in valid_by_room.values():
        for row in rows:
            motion_id = _normalize_motion_id(row.get("motion_id"))
            counts[motion_id] += 1
            by_identity["_".join(motion_id.split("_")[:3])] += 1
    return {
        "motion_ids": dict(sorted(counts.items())),
        "identities": dict(sorted(by_identity.items())),
    }


def _parse_spawn_yaw_sweep_deg(value):
    text = str(value or "0").strip()
    if not text:
        return [0.0]
    parsed = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(float(token))
    if not parsed:
        return [0.0]
    unique = sorted({float(item) for item in parsed})
    if 0.0 not in unique:
        unique.insert(0, 0.0)
    return unique


def _point_clearance_to_obstacles_m(room_obstacles, point_xy):
    point = Point(point_xy)
    best = None
    for obstacle in room_obstacles:
        poly = _obstacle_polygon(obstacle)
        if poly is None or poly.is_empty:
            continue
        distance = float(poly.distance(point))
        best = distance if best is None else min(best, distance)
    return best


def _point_clearance_to_room_boundary_m(room_poly, point_xy):
    point = Point(point_xy)
    if room_poly is None or room_poly.is_empty or (not room_poly.covers(point)):
        return None
    return float(room_poly.boundary.distance(point))


def _room_centroid_xy(room_poly):
    centroid = room_poly.centroid
    return (float(centroid.x), float(centroid.y))


def _motion_path_metrics_from_payload(payload):
    frames = list(payload.get("frames", []))
    if len(frames) < 2:
        return {
            "path_length_m": 0.0,
            "max_displacement_from_start_m": 0.0,
        }
    path_length = 0.0
    max_displacement = 0.0
    start_x = float(frames[0].get("root_x_m", 0.0))
    start_y = float(frames[0].get("root_y_m", 0.0))
    prev_x = start_x
    prev_y = start_y
    for frame in frames[1:]:
        x = float(frame.get("root_x_m", 0.0))
        y = float(frame.get("root_y_m", 0.0))
        path_length += float(math.hypot(x - prev_x, y - prev_y))
        max_displacement = max(max_displacement, float(math.hypot(x - start_x, y - start_y)))
        prev_x = x
        prev_y = y
    return {
        "path_length_m": float(path_length),
        "max_displacement_from_start_m": float(max_displacement),
    }


def _path_length_bin_label(path_length_m):
    for lo, hi in zip(PATH_LENGTH_BINS_M[:-1], PATH_LENGTH_BINS_M[1:]):
        if path_length_m < hi:
            hi_label = "inf" if math.isinf(hi) else f"{hi:g}"
            return f"[{lo:g},{hi_label})"
    return "[8,inf)"


def _distribution_summary(values):
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": float(min(values)),
        "median": float(median(values)),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
    }


def _parse_float_list(value):
    return [float(token.strip()) for token in str(value or "").split(",") if token.strip()]


def _camera_distance_bin_label(distance_m, bin_edges):
    edges = list(bin_edges)
    if len(edges) < 2:
        return "all"
    for left, right in zip(edges[:-1], edges[1:]):
        if float(left) <= float(distance_m) < float(right):
            right_label = "inf" if right >= 999.0 else f"{right:g}"
            return f"[{left:g},{right_label})"
    return f"[{edges[-2]:g},{edges[-1]:g})"


def _azimuth_bin_label(azimuth_deg, bin_count=DEFAULT_AZIMUTH_BIN_COUNT):
    bin_count = max(1, int(bin_count))
    azimuth_deg = float(azimuth_deg) % 360.0
    bin_width = 360.0 / float(bin_count)
    bin_index = int(azimuth_deg // bin_width) % bin_count
    start = bin_index * bin_width
    end = start + bin_width
    return f"[{start:.0f},{end:.0f})"


def _scale_bin_label(distance_m, thresholds_m=DEFAULT_SCALE_BIN_THRESHOLDS_M):
    distance_m = float(distance_m)
    t0, t1, t2, t3 = [float(v) for v in thresholds_m]
    if distance_m < t0:
        return "extreme_close"
    if distance_m < t1:
        return "close"
    if distance_m < t2:
        return "medium"
    if distance_m < t3:
        return "far"
    return "tiny"


def _seam_bin_label(seam_distance_deg, near_seam_threshold_deg=20.0):
    seam_distance_deg = float(seam_distance_deg)
    near = float(near_seam_threshold_deg)
    if seam_distance_deg <= near:
        return "near_seam"
    if seam_distance_deg <= 2.0 * near:
        return "mid_seam"
    return "far_from_seam"


def _camera_proxy_for_room(metadata, room_name):
    room_record = _find_room(metadata, room_name)
    if room_record is None:
        return None
    room_poly = _room_polygon(room_record)
    if room_poly is None or room_poly.is_empty:
        return None
    point = room_poly.representative_point()
    return {
        "proxy_type": "room_representative_point",
        "room": room_name,
        "position_xy_m": [float(point.x), float(point.y)],
        "floor_z_m": float(room_record.get("floor_z") or 0.0),
    }


def _human_spatial_camera_features(
    human_position_xyz_m,
    camera_proxy,
    distance_bins_m,
    near_seam_threshold_deg=20.0,
    azimuth_bin_count=DEFAULT_AZIMUTH_BIN_COUNT,
    scale_bin_thresholds_m=DEFAULT_SCALE_BIN_THRESHOLDS_M,
):
    camera_xy = camera_proxy["position_xy_m"]
    hx = float(human_position_xyz_m[0])
    hy = float(human_position_xyz_m[1])
    dx = hx - float(camera_xy[0])
    dy = hy - float(camera_xy[1])
    distance_m = float(math.hypot(dx, dy))
    azimuth_deg = float((math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0)
    erp_u = azimuth_deg / 360.0
    seam_distance_deg = min(azimuth_deg, 360.0 - azimuth_deg)
    distance_bin = _camera_distance_bin_label(distance_m, distance_bins_m)
    azimuth_bin = _azimuth_bin_label(azimuth_deg, azimuth_bin_count)
    azimuth_bin_8 = _azimuth_bin_label(azimuth_deg, DEFAULT_AZIMUTH_BIN_COUNT)
    azimuth_bin_12 = _azimuth_bin_label(azimuth_deg, DEFAULT_AZIMUTH_BIN_COUNT_FINE)
    estimated_bbox_scale = float(CAPSULE_HEIGHT_M / max(distance_m, 1e-6))
    scale_bin = _scale_bin_label(distance_m, scale_bin_thresholds_m)
    seam_crossing_risk = bool(
        seam_distance_deg <= max(10.0, float(near_seam_threshold_deg) * 0.5)
        and distance_m < 3.0
    )
    return {
        "distance_to_camera_m": distance_m,
        "azimuth_relative_to_camera_deg": azimuth_deg,
        "longitude_deg": azimuth_deg,
        "erp_u_center_normalized": erp_u,
        "erp_longitude_deg": azimuth_deg,
        "seam_distance_deg": seam_distance_deg,
        "near_seam_bool": bool(seam_distance_deg <= float(near_seam_threshold_deg)),
        "seam_crossing_risk_bool": seam_crossing_risk,
        "seam_bin": _seam_bin_label(seam_distance_deg, near_seam_threshold_deg),
        "distance_bin": distance_bin,
        "azimuth_bin": azimuth_bin,
        "azimuth_bin_8": azimuth_bin_8,
        "azimuth_bin_12": azimuth_bin_12,
        "apparent_scale_proxy": float(1.0 / max(distance_m, 1e-6)),
        "estimated_bbox_scale": estimated_bbox_scale,
        "scale_bin": scale_bin,
        "bbox_height_px_est": None,
        "bbox_area_ratio_est": None,
        "truncation_risk_bool": bool(seam_crossing_risk and distance_m < 1.5),
        "close_camera_bool": bool(distance_m < 1.5),
        "extreme_close_bool": bool(distance_m < 1.0),
        "far_camera_bool": bool(distance_m > 3.0),
    }


def _multi_human_occlusion_risk_score(human_features):
    best = 0.0
    for left, right in combinations(human_features, 2):
        azimuth_gap = abs(float(left["azimuth_relative_to_camera_deg"]) - float(right["azimuth_relative_to_camera_deg"]))
        azimuth_gap = min(azimuth_gap, 360.0 - azimuth_gap)
        distance_gap = abs(float(left["distance_to_camera_m"]) - float(right["distance_to_camera_m"]))
        if azimuth_gap < 20.0:
            best = max(best, float((20.0 - azimuth_gap) / 20.0) * float(min(distance_gap, 3.0) / 3.0))
    return float(best)


def _scene_spatial_camera_summary(
    metadata,
    room_name,
    humans,
    distance_bins_m,
    near_seam_threshold_deg=20.0,
    azimuth_bin_count=DEFAULT_AZIMUTH_BIN_COUNT,
    scale_bin_thresholds_m=DEFAULT_SCALE_BIN_THRESHOLDS_M,
):
    camera_proxy = _camera_proxy_for_room(metadata, room_name)
    if camera_proxy is None:
        return {
            "camera_proxy_available": False,
            "camera_proxy": None,
            "per_human": [],
        }
    per_human = []
    for human_index, human in enumerate(humans):
        position_xyz = human.get("position_xyz_m") or human.get("position_xyz") or []
        if len(position_xyz) < 2:
            continue
        features = _human_spatial_camera_features(
            position_xyz,
            camera_proxy,
            distance_bins_m=distance_bins_m,
            near_seam_threshold_deg=near_seam_threshold_deg,
            azimuth_bin_count=azimuth_bin_count,
            scale_bin_thresholds_m=scale_bin_thresholds_m,
        )
        per_human.append(
            {
                "human_index": int(human_index),
                "motion_id": human.get("motion_id"),
                "position_xyz_m": list(position_xyz),
                **features,
            }
        )
    distances = [float(row["distance_to_camera_m"]) for row in per_human]
    distance_bins = sorted({row["distance_bin"] for row in per_human})
    azimuth_bins = sorted({row["azimuth_bin"] for row in per_human})
    scale_bins = sorted({row["scale_bin"] for row in per_human})
    has_close = any(bool(row["close_camera_bool"]) for row in per_human)
    has_far = any(bool(row["far_camera_bool"]) for row in per_human)
    has_near_seam = any(bool(row["near_seam_bool"]) for row in per_human)
    seam_crossing_risk_count = sum(1 for row in per_human if row["seam_crossing_risk_bool"])
    has_multi_depth = (max(distances) - min(distances) >= 1.0) if distances else False
    occlusion_risk = _multi_human_occlusion_risk_score(per_human)
    azimuth_values = [float(row["azimuth_relative_to_camera_deg"]) for row in per_human]
    min_pairwise_azimuth_sep = None
    if len(azimuth_values) >= 2:
        min_pairwise_azimuth_sep = min(
            min(abs(a - b), 360.0 - abs(a - b))
            for a, b in combinations(azimuth_values, 2)
        )
    depth_range_m = None if not distances else float(max(distances) - min(distances))
    azimuth_coverage_score = float(len(azimuth_bins))
    distance_coverage_score = float(len(distance_bins))
    seam_coverage_score = float(sum(1 for row in per_human if row["near_seam_bool"])) + float(seam_crossing_risk_count)
    scale_coverage_score = float(len(scale_bins))
    spatial_diversity_score = (
        2.0 * len(distance_bins)
        + 1.0 * len(azimuth_bins)
        + 1.0 * len(scale_bins)
        + (1.0 if has_close else 0.0)
        + (1.0 if has_far else 0.0)
        + (1.0 if has_multi_depth else 0.0)
    )
    erp_difficulty_score = (
        sum(1.0 for row in per_human if row["near_seam_bool"])
        + sum(1.0 for row in per_human if row["seam_crossing_risk_bool"])
        + sum(1.0 for row in per_human if row["extreme_close_bool"])
        + (1.0 if has_multi_depth else 0.0)
        + float(occlusion_risk)
    )
    return {
        "camera_proxy_available": True,
        "camera_proxy": camera_proxy,
        "per_human": per_human,
        "min_distance_to_camera_m": None if not distances else float(min(distances)),
        "max_distance_to_camera_m": None if not distances else float(max(distances)),
        "depth_range_m": depth_range_m,
        "distance_bin_coverage": distance_bins,
        "azimuth_bin_coverage": azimuth_bins,
        "scale_bin_coverage": scale_bins,
        "has_close_human": bool(has_close),
        "has_far_human": bool(has_far),
        "has_near_seam_human": bool(has_near_seam),
        "near_seam_human_count": int(sum(1 for row in per_human if row["near_seam_bool"])),
        "seam_crossing_risk_human_count": int(seam_crossing_risk_count),
        "has_multi_depth_humans": bool(has_multi_depth),
        "min_pairwise_azimuth_separation_deg": min_pairwise_azimuth_sep,
        "azimuth_coverage_score": azimuth_coverage_score,
        "distance_coverage_score": distance_coverage_score,
        "seam_coverage_score": seam_coverage_score,
        "scale_coverage_score": scale_coverage_score,
        "multi_human_occlusion_risk_score": float(occlusion_risk),
        "spatial_diversity_score": float(spatial_diversity_score),
        "erp_difficulty_score": float(erp_difficulty_score),
    }


def _camera_proxy_signature(camera_proxy):
    if not camera_proxy:
        return None
    position_xyz = camera_proxy.get("position_xyz_m") or [0.0, 0.0, 0.0]
    yaw_deg = float(camera_proxy.get("yaw_deg") or 0.0)
    return {
        "position_xyz_m": [float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2])],
        "yaw_deg": yaw_deg,
        "key": (
            round(float(position_xyz[0]), 3),
            round(float(position_xyz[1]), 3),
            round(float(position_xyz[2]), 3),
            round(yaw_deg, 3),
        ),
    }


def _safe_sorted_profile(values):
    return sorted(value for value in values if value is not None)


def _scene_spatial_dedup_signature(scene):
    spatial = scene.get("spatial_camera") or {}
    per_human = spatial.get("per_human") or []
    humans = scene.get("humans") or []
    by_index = {int(row.get("human_index", -1)): row for row in per_human}
    human_rows = []
    for human_index, human in enumerate(humans):
        spatial_row = by_index.get(int(human_index), {})
        position_xyz = human.get("position_xyz_m") or [0.0, 0.0, 0.0]
        yaw_rad = human.get("yaw_rad")
        yaw_deg = math.degrees(float(yaw_rad)) if yaw_rad is not None else None
        motion_id = _normalize_motion_id(human.get("motion_id"))
        identity_id = _motion_identity(motion_id)
        human_rows.append(
            {
                "human_index": int(human_index),
                "motion_id": motion_id,
                "identity_id": identity_id,
                "spawn_xy_m": [float(position_xyz[0]), float(position_xyz[1])],
                "spawn_xyz_m": [float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2])],
                "spawn_yaw_deg": None if yaw_deg is None else float(yaw_deg),
                "distance_bin": spatial_row.get("distance_bin"),
                "azimuth_bin": spatial_row.get("azimuth_bin"),
                "seam_bin": spatial_row.get("seam_bin"),
                "scale_bin": spatial_row.get("scale_bin"),
                "distance_to_camera_m": spatial_row.get("distance_to_camera_m"),
                "azimuth_relative_to_camera_deg": spatial_row.get("azimuth_relative_to_camera_deg"),
                "seam_distance_deg": spatial_row.get("seam_distance_deg"),
            }
        )
    camera_signature = _camera_proxy_signature(spatial.get("camera_proxy"))
    return {
        "room": scene.get("room"),
        "human_count": int(scene.get("human_count", len(humans))),
        "camera_proxy": camera_signature,
        "motion_ids": _safe_sorted_profile(row["motion_id"] for row in human_rows),
        "identity_ids": _safe_sorted_profile(row["identity_id"] for row in human_rows),
        "distance_bins": _safe_sorted_profile(row["distance_bin"] for row in human_rows),
        "azimuth_bins": _safe_sorted_profile(row["azimuth_bin"] for row in human_rows),
        "seam_bins": _safe_sorted_profile(row["seam_bin"] for row in human_rows),
        "scale_bins": _safe_sorted_profile(row["scale_bin"] for row in human_rows),
        "humans": human_rows,
    }


def _jaccard_similarity(left_values, right_values):
    left_set = {value for value in left_values if value is not None}
    right_set = {value for value in right_values if value is not None}
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return float(len(left_set & right_set) / len(union))


def _profile_exact_match(left_values, right_values):
    return list(left_values) == list(right_values)


def _human_exact_match(left_human, right_human):
    yaw_left = left_human.get("spawn_yaw_deg")
    yaw_right = right_human.get("spawn_yaw_deg")
    return (
        left_human.get("motion_id") == right_human.get("motion_id")
        and left_human.get("identity_id") == right_human.get("identity_id")
        and round(float(left_human["spawn_xy_m"][0]), 3) == round(float(right_human["spawn_xy_m"][0]), 3)
        and round(float(left_human["spawn_xy_m"][1]), 3) == round(float(right_human["spawn_xy_m"][1]), 3)
        and (
            (yaw_left is None and yaw_right is None)
            or (
                yaw_left is not None
                and yaw_right is not None
                and round(float(yaw_left), 3) == round(float(yaw_right), 3)
            )
        )
    )


def _xy_distance_m(left_xy, right_xy):
    dx = float(left_xy[0]) - float(right_xy[0])
    dy = float(left_xy[1]) - float(right_xy[1])
    return float(math.sqrt(dx * dx + dy * dy))


def _yaw_difference_deg(left_yaw, right_yaw):
    if left_yaw is None or right_yaw is None:
        return None
    diff = abs(float(left_yaw) - float(right_yaw)) % 360.0
    return float(min(diff, 360.0 - diff))


def _best_human_matching(left_humans, right_humans):
    if len(left_humans) != len(right_humans):
        return None
    indices = list(range(len(right_humans)))
    best = None
    for perm in permutations(indices):
        pair_rows = []
        total_xy = 0.0
        total_yaw = 0.0
        yaw_count = 0
        exact_same_count = 0
        for left_index, right_index in enumerate(perm):
            left_human = left_humans[left_index]
            right_human = right_humans[right_index]
            xy_distance_m = _xy_distance_m(left_human["spawn_xy_m"], right_human["spawn_xy_m"])
            yaw_diff_deg = _yaw_difference_deg(left_human.get("spawn_yaw_deg"), right_human.get("spawn_yaw_deg"))
            if yaw_diff_deg is not None:
                total_yaw += yaw_diff_deg
                yaw_count += 1
            if _human_exact_match(left_human, right_human):
                exact_same_count += 1
            total_xy += xy_distance_m
            pair_rows.append(
                {
                    "left_human_index": int(left_human["human_index"]),
                    "right_human_index": int(right_human["human_index"]),
                    "left_motion_id": left_human.get("motion_id"),
                    "right_motion_id": right_human.get("motion_id"),
                    "xy_distance_m": float(xy_distance_m),
                    "yaw_difference_deg": yaw_diff_deg,
                    "exact_same_human_motion_spawn_yaw": _human_exact_match(left_human, right_human),
                }
            )
        mean_xy = float(total_xy / len(left_humans)) if left_humans else 0.0
        mean_yaw = None if yaw_count == 0 else float(total_yaw / yaw_count)
        score = (mean_xy, 0 if exact_same_count > 0 else 1, float("inf") if mean_yaw is None else mean_yaw)
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "mean_matched_xy_distance_m": mean_xy,
                "mean_matched_yaw_difference_deg": mean_yaw,
                "exact_same_human_count": int(exact_same_count),
                "pair_rows": pair_rows,
            }
    return best


def _scene_quality_tuple(scene):
    spatial = scene.get("spatial_camera") or {}
    motion_ids = {_normalize_motion_id(human.get("motion_id")) for human in scene.get("humans", [])}
    identity_ids = {_motion_identity(human.get("motion_id")) for human in scene.get("humans", [])}
    return (
        float(spatial.get("erp_difficulty_score") or 0.0),
        float(spatial.get("depth_range_m") or 0.0),
        int(spatial.get("seam_crossing_risk_human_count") or 0),
        len(motion_ids),
        len(identity_ids),
        float(spatial.get("spatial_diversity_score") or 0.0),
        scene.get("miniscene_id") or "",
    )


def _scene_is_spatial_duplicate(left_scene, right_scene, xy_threshold_m):
    left_sig = _scene_spatial_dedup_signature(left_scene)
    right_sig = _scene_spatial_dedup_signature(right_scene)
    same_room = left_sig.get("room") == right_sig.get("room")
    same_human_count = int(left_sig.get("human_count", 0)) == int(right_sig.get("human_count", 0))
    left_camera = (left_sig.get("camera_proxy") or {}).get("key")
    right_camera = (right_sig.get("camera_proxy") or {}).get("key")
    same_camera_proxy = left_camera is not None and left_camera == right_camera
    matching = _best_human_matching(left_sig.get("humans", []), right_sig.get("humans", []))
    mean_xy = None if matching is None else matching["mean_matched_xy_distance_m"]
    same_profiles = (
        _profile_exact_match(left_sig.get("distance_bins", []), right_sig.get("distance_bins", []))
        and _profile_exact_match(left_sig.get("azimuth_bins", []), right_sig.get("azimuth_bins", []))
        and _profile_exact_match(left_sig.get("seam_bins", []), right_sig.get("seam_bins", []))
    )
    exact_same_human = bool(matching and matching.get("exact_same_human_count", 0) > 0)
    is_duplicate = bool(
        same_room
        and same_human_count
        and same_camera_proxy
        and matching is not None
        and mean_xy is not None
        and float(mean_xy) < float(xy_threshold_m)
        and (exact_same_human or same_profiles)
    )
    return {
        "is_duplicate": bool(is_duplicate),
        "same_room": bool(same_room),
        "same_human_count": bool(same_human_count),
        "same_camera_proxy": bool(same_camera_proxy),
        "mean_matched_xy_distance_m": mean_xy,
        "mean_matched_yaw_difference_deg": None if matching is None else matching.get("mean_matched_yaw_difference_deg"),
        "exact_same_human_count": 0 if matching is None else int(matching.get("exact_same_human_count", 0)),
        "pair_rows": [] if matching is None else matching.get("pair_rows", []),
        "motion_jaccard": _jaccard_similarity(left_sig.get("motion_ids", []), right_sig.get("motion_ids", [])),
        "identity_jaccard": _jaccard_similarity(left_sig.get("identity_ids", []), right_sig.get("identity_ids", [])),
        "distance_profile_identical": _profile_exact_match(left_sig.get("distance_bins", []), right_sig.get("distance_bins", [])),
        "azimuth_profile_identical": _profile_exact_match(left_sig.get("azimuth_bins", []), right_sig.get("azimuth_bins", [])),
        "seam_profile_identical": _profile_exact_match(left_sig.get("seam_bins", []), right_sig.get("seam_bins", [])),
        "scale_profile_identical": _profile_exact_match(left_sig.get("scale_bins", []), right_sig.get("scale_bins", [])),
        "left_signature": left_sig,
        "right_signature": right_sig,
    }


def _build_group_scene_candidate(
    metadata,
    room_name,
    group,
    spawn_lookup,
    ordinal,
    frame_start,
    frame_end,
    distance_bins_m,
    near_seam_threshold_deg,
    azimuth_bin_count,
    scale_bin_thresholds_m=DEFAULT_SCALE_BIN_THRESHOLDS_M,
):
    scene = _build_group_miniscene(
        room_name,
        ordinal,
        group,
        spawn_lookup,
        frame_start,
        frame_end,
    )
    scene["spatial_camera"] = _scene_spatial_camera_summary(
        metadata,
        room_name,
        scene.get("humans", []),
        distance_bins_m=distance_bins_m,
        near_seam_threshold_deg=near_seam_threshold_deg,
        azimuth_bin_count=azimuth_bin_count,
        scale_bin_thresholds_m=scale_bin_thresholds_m,
    )
    return scene


def _post_select_spatial_dedup(
    selected_scenes,
    candidate_scenes,
    enable_spatial_dedup=False,
    spatial_dedup_xy_threshold_m=0.4,
    spatial_dedup_report_only=False,
):
    report = {
        "enabled": bool(enable_spatial_dedup),
        "report_only": bool(spatial_dedup_report_only),
        "xy_threshold_m": float(spatial_dedup_xy_threshold_m),
        "scene_count_before": int(len(selected_scenes)),
        "scene_count_after_dedup": int(len(selected_scenes)),
        "scene_count_after_backfill": int(len(selected_scenes)),
        "duplicate_pair_count": 0,
        "duplicate_pairs": [],
        "removed_miniscenes": [],
        "kept_miniscenes": [],
        "backfill_added_miniscenes": [],
    }
    if not enable_spatial_dedup:
        return list(selected_scenes), report

    active_scenes = list(selected_scenes)
    removed_ids = set()
    duplicate_pairs = []
    for left_index in range(len(selected_scenes)):
        left_scene = selected_scenes[left_index]
        if left_scene.get("miniscene_id") in removed_ids:
            continue
        for right_index in range(left_index + 1, len(selected_scenes)):
            right_scene = selected_scenes[right_index]
            if right_scene.get("miniscene_id") in removed_ids:
                continue
            similarity = _scene_is_spatial_duplicate(
                left_scene,
                right_scene,
                spatial_dedup_xy_threshold_m,
            )
            if not similarity.get("is_duplicate"):
                continue
            left_quality = _scene_quality_tuple(left_scene)
            right_quality = _scene_quality_tuple(right_scene)
            if right_quality > left_quality:
                kept_scene = right_scene
                removed_scene = left_scene
            else:
                kept_scene = left_scene
                removed_scene = right_scene
            removed_ids.add(removed_scene.get("miniscene_id"))
            duplicate_pairs.append(
                {
                    "left_miniscene_id": left_scene.get("miniscene_id"),
                    "right_miniscene_id": right_scene.get("miniscene_id"),
                    "kept_miniscene_id": kept_scene.get("miniscene_id"),
                    "removed_miniscene_id": removed_scene.get("miniscene_id"),
                    "removal_reason": "near_duplicate_spatial_signature",
                    "kept_quality_tuple": list(left_quality if kept_scene is left_scene else right_quality),
                    "removed_quality_tuple": list(right_quality if removed_scene is right_scene else left_quality),
                    **similarity,
                }
            )
    deduped_scenes = [
        scene
        for scene in selected_scenes
        if scene.get("miniscene_id") not in removed_ids
    ]
    report["duplicate_pair_count"] = int(len(duplicate_pairs))
    report["duplicate_pairs"] = duplicate_pairs
    report["removed_miniscenes"] = sorted(removed_ids)
    report["kept_miniscenes"] = sorted(scene.get("miniscene_id") for scene in deduped_scenes)
    report["scene_count_after_dedup"] = int(len(deduped_scenes))
    if spatial_dedup_report_only:
        return list(selected_scenes), report

    target_count = len(selected_scenes)
    selected_ids = {scene.get("miniscene_id") for scene in deduped_scenes}
    for candidate in sorted(candidate_scenes, key=_scene_quality_tuple, reverse=True):
        if len(deduped_scenes) >= target_count:
            break
        candidate_id = candidate.get("miniscene_id")
        if candidate_id in selected_ids:
            continue
        duplicate = False
        for existing in deduped_scenes:
            similarity = _scene_is_spatial_duplicate(
                candidate,
                existing,
                spatial_dedup_xy_threshold_m,
            )
            if similarity.get("is_duplicate"):
                duplicate = True
                break
        if duplicate:
            continue
        deduped_scenes.append(candidate)
        selected_ids.add(candidate_id)
        report["backfill_added_miniscenes"].append(candidate_id)
    report["scene_count_after_backfill"] = int(len(deduped_scenes))
    return deduped_scenes, report


def _spatial_camera_audit_report(
    metadata,
    valid_single_by_room,
    final_manifest,
    distance_bins_m,
    near_seam_threshold_deg=20.0,
    azimuth_bin_count=DEFAULT_AZIMUTH_BIN_COUNT,
    scale_bin_thresholds_m=DEFAULT_SCALE_BIN_THRESHOLDS_M,
):
    valid_single_rows = []
    distance_bin_counts = Counter()
    azimuth_bin_counts = Counter()
    azimuth_bin_8_counts = Counter()
    azimuth_bin_12_counts = Counter()
    seam_bucket_counts = Counter()
    scale_bin_counts = Counter()
    close_count = 0
    far_count = 0
    near_seam_human_count = 0
    seam_crossing_risk_human_count = 0
    all_humans = []
    per_room_spatial_coverage = defaultdict(lambda: {
        "distance_bins": set(),
        "azimuth_bins": set(),
        "scale_bins": set(),
        "scene_count": 0,
    })
    for room_name, rows in valid_single_by_room.items():
        for row in rows:
            spawn_pose = row.get("spawn_pose") or {}
            summary = _scene_spatial_camera_summary(
                metadata,
                room_name,
                [
                    {
                        "motion_id": row.get("motion_id"),
                        "position_xyz_m": spawn_pose.get("position_xyz"),
                    }
                ],
                distance_bins_m=distance_bins_m,
                near_seam_threshold_deg=near_seam_threshold_deg,
                azimuth_bin_count=azimuth_bin_count,
                scale_bin_thresholds_m=scale_bin_thresholds_m,
            )
            valid_single_rows.append(
                {
                    "room": room_name,
                    "spawn_index": row.get("spawn_index"),
                    "motion_id": row.get("motion_id"),
                    "spatial_camera": summary,
                }
            )
    final_scene_rows = []
    for scene in final_manifest.get("miniscenes", []):
        summary = _scene_spatial_camera_summary(
            metadata,
            scene.get("room"),
            scene.get("humans", []),
            distance_bins_m=distance_bins_m,
            near_seam_threshold_deg=near_seam_threshold_deg,
            azimuth_bin_count=azimuth_bin_count,
            scale_bin_thresholds_m=scale_bin_thresholds_m,
        )
        final_scene_rows.append(
            {
                "miniscene_id": scene.get("miniscene_id"),
                "room": scene.get("room"),
                "human_count": int(scene.get("human_count", len(scene.get("humans", [])))),
                "spatial_camera": summary,
            }
        )
        for human in summary.get("per_human", []):
            distance_bin_counts[human["distance_bin"]] += 1
            azimuth_bin_counts[human["azimuth_bin"]] += 1
            azimuth_bin_8_counts[human["azimuth_bin_8"]] += 1
            azimuth_bin_12_counts[human["azimuth_bin_12"]] += 1
            seam_bucket_counts[human["seam_bin"]] += 1
            scale_bin_counts[human["scale_bin"]] += 1
            close_count += int(bool(human["close_camera_bool"]))
            far_count += int(bool(human["far_camera_bool"]))
            near_seam_human_count += int(bool(human["near_seam_bool"]))
            seam_crossing_risk_human_count += int(bool(human["seam_crossing_risk_bool"]))
            per_room_spatial_coverage[scene.get("room")]["distance_bins"].add(human["distance_bin"])
            per_room_spatial_coverage[scene.get("room")]["azimuth_bins"].add(human["azimuth_bin"])
            per_room_spatial_coverage[scene.get("room")]["scale_bins"].add(human["scale_bin"])
            all_humans.append(
                {
                    "miniscene_id": scene.get("miniscene_id"),
                    "room": scene.get("room"),
                    **human,
                }
            )
        per_room_spatial_coverage[scene.get("room")]["scene_count"] += 1
    all_humans_sorted_by_distance = sorted(all_humans, key=lambda row: float(row["distance_to_camera_m"]))
    all_humans_sorted_by_seam = sorted(all_humans, key=lambda row: float(row["seam_distance_deg"]))
    return {
        "camera_proxy_policy": "room_representative_point",
        "distance_bins_m": list(distance_bins_m),
        "near_seam_threshold_deg": float(near_seam_threshold_deg),
        "azimuth_bin_count": int(azimuth_bin_count),
        "scale_bin_thresholds_m": list(scale_bin_thresholds_m),
        "valid_single_candidate_rows": valid_single_rows,
        "final_miniscene_rows": final_scene_rows,
        "summary_histograms": {
            "distance_bins": dict(sorted(distance_bin_counts.items())),
            "azimuth_bins": dict(sorted(azimuth_bin_counts.items())),
            "azimuth_bins_8": dict(sorted(azimuth_bin_8_counts.items())),
            "azimuth_bins_12": dict(sorted(azimuth_bin_12_counts.items())),
            "seam_bins": dict(sorted(seam_bucket_counts.items())),
            "scale_proxy_bins": dict(sorted(scale_bin_counts.items())),
            "close_human_count": int(close_count),
            "far_human_count": int(far_count),
            "near_seam_human_count": int(near_seam_human_count),
            "seam_crossing_risk_human_count": int(seam_crossing_risk_human_count),
            "scenes_with_near_seam_human": int(
                sum(1 for row in final_scene_rows if row["spatial_camera"].get("has_near_seam_human"))
            ),
            "scenes_with_seam_crossing_risk": int(
                sum(1 for row in final_scene_rows if row["spatial_camera"].get("seam_crossing_risk_human_count", 0) > 0)
            ),
            "per_room_spatial_coverage": {
                room: {
                    "scene_count": int(payload["scene_count"]),
                    "distance_bins": sorted(payload["distance_bins"]),
                    "azimuth_bins": sorted(payload["azimuth_bins"]),
                    "scale_bins": sorted(payload["scale_bins"]),
                }
                for room, payload in sorted(per_room_spatial_coverage.items())
            },
            "distance_histogram_0p5m": dict(
                sorted(
                    Counter(
                        f"[{math.floor(float(row['distance_to_camera_m'])/0.5)*0.5:.1f},{math.floor(float(row['distance_to_camera_m'])/0.5)*0.5+0.5:.1f})"
                        for row in all_humans
                    ).items(),
                    key=lambda kv: float(kv[0].split(',')[0][1:]),
                )
            ),
            "top_10_closest_humans": all_humans_sorted_by_distance[:10],
            "top_10_farthest_humans": list(reversed(all_humans_sorted_by_distance[-10:])),
            "top_10_closest_to_seam_humans": all_humans_sorted_by_seam[:10],
        },
    }


def _build_augmented_metadata(
    metadata,
    spawn_yaw_sweep_deg,
    extra_free_space_spawn_samples=0,
    free_space_sampling_seed=0,
    min_spawn_wall_clearance_m=MIN_HUMAN_WALL_CLEARANCE_M,
    min_spawn_obstacle_clearance_m=MIN_HUMAN_OBSTACLE_CLEARANCE_M,
):
    augmented = copy.deepcopy(metadata)
    original_spawn_poses = list(augmented.get("spawn_poses", []))
    room_template_spawns = defaultdict(list)
    for index, spawn_pose in enumerate(original_spawn_poses):
        room_template_spawns[str(spawn_pose.get("room"))].append((index, spawn_pose))

    extra_spawn_poses = []
    rejected_extra_samples = Counter()
    extra_sample_rows = []
    rng = random.Random(int(free_space_sampling_seed))
    extra_free_space_spawn_samples = max(0, int(extra_free_space_spawn_samples))
    if extra_free_space_spawn_samples > 0:
        for room_record in augmented.get("rooms", []):
            room_name = str(room_record.get("name"))
            room_poly = _room_polygon(room_record)
            if room_poly is None or room_poly.is_empty:
                continue
            room_obstacles = _find_room_obstacles(augmented, room_name)
            min_x, min_y, max_x, max_y = room_poly.bounds
            attempts = 0
            accepted = 0
            max_attempts = max(100, int(extra_free_space_spawn_samples) * 40)
            templates = room_template_spawns.get(room_name, [])
            while accepted < extra_free_space_spawn_samples and attempts < max_attempts:
                attempts += 1
                point_xy = (
                    float(rng.uniform(min_x, max_x)),
                    float(rng.uniform(min_y, max_y)),
                )
                point = Point(point_xy)
                if not room_poly.covers(point):
                    rejected_extra_samples["outside_room_polygon"] += 1
                    continue
                wall_clearance = _point_clearance_to_room_boundary_m(room_poly, point_xy)
                if wall_clearance is None or wall_clearance < float(min_spawn_wall_clearance_m):
                    rejected_extra_samples["wall_clearance"] += 1
                    continue
                obstacle_clearance = _point_clearance_to_obstacles_m(room_obstacles, point_xy)
                if (
                    obstacle_clearance is not None
                    and obstacle_clearance < float(min_spawn_obstacle_clearance_m)
                ):
                    rejected_extra_samples["obstacle_clearance"] += 1
                    continue
                if templates:
                    template_index, template_spawn = templates[accepted % len(templates)]
                    spawn_yaw = float(template_spawn.get("yaw") or 0.0)
                    pose_type = template_spawn.get("pose_type")
                    activity_hint = template_spawn.get("activity_hint")
                    target_object = template_spawn.get("target_object")
                else:
                    template_index = None
                    template_spawn = {}
                    spawn_yaw = 0.0
                    pose_type = "free_space_augmented"
                    activity_hint = None
                    target_object = None
                position_z = float(room_record.get("floor_z") or 0.0)
                row = {
                    "room": room_name,
                    "position_xyz": [float(point_xy[0]), float(point_xy[1]), position_z],
                    "yaw": spawn_yaw,
                    "target_object": target_object,
                    "activity_hint": activity_hint,
                    "pose_type": pose_type,
                    "source": "free_space_augmented",
                    "augmentation": {
                        "type": "free_space_spawn",
                        "template_spawn_index": template_index,
                        "wall_clearance_m": wall_clearance,
                        "obstacle_clearance_m": obstacle_clearance,
                    },
                }
                extra_spawn_poses.append(row)
                extra_sample_rows.append(
                    {
                        "room": room_name,
                        "position_xy": [float(point_xy[0]), float(point_xy[1])],
                        "yaw": float(spawn_yaw),
                        "wall_clearance_m": wall_clearance,
                        "obstacle_clearance_m": obstacle_clearance,
                        "template_spawn_index": template_index,
                    }
                )
                accepted += 1
            rejected_extra_samples[f"{room_name}__attempts"] += attempts
            rejected_extra_samples[f"{room_name}__accepted"] += accepted

    base_spawn_poses = list(original_spawn_poses) + list(extra_spawn_poses)
    yaw_offsets_deg = _parse_spawn_yaw_sweep_deg(spawn_yaw_sweep_deg)
    augmented_spawn_poses = []
    yaw_augmented_rows = []
    for base_index, spawn_pose in enumerate(base_spawn_poses):
        base_yaw = float(spawn_pose.get("yaw") or 0.0)
        for yaw_offset_deg in yaw_offsets_deg:
            if abs(float(yaw_offset_deg)) < 1e-9:
                new_spawn = copy.deepcopy(spawn_pose)
            else:
                new_spawn = copy.deepcopy(spawn_pose)
                new_spawn["yaw"] = float(base_yaw + math.radians(float(yaw_offset_deg)))
                augmentation = dict(new_spawn.get("augmentation") or {})
                augmentation.update(
                    {
                        "type": "yaw_sweep",
                        "base_spawn_index": int(base_index),
                        "yaw_offset_deg": float(yaw_offset_deg),
                    }
                )
                new_spawn["augmentation"] = augmentation
            augmented_spawn_poses.append(new_spawn)
            yaw_augmented_rows.append(
                {
                    "base_spawn_index": int(base_index),
                    "room": new_spawn.get("room"),
                    "base_yaw_rad": base_yaw,
                    "yaw_offset_deg": float(yaw_offset_deg),
                    "final_yaw_rad": float(new_spawn.get("yaw") or 0.0),
                    "source": new_spawn.get("source"),
                }
            )
    augmented["spawn_poses"] = augmented_spawn_poses
    report = {
        "original_spawn_pose_count": len(original_spawn_poses),
        "extra_free_space_spawn_samples_requested": int(extra_free_space_spawn_samples),
        "extra_free_space_spawn_samples_generated": len(extra_spawn_poses),
        "base_spawn_pose_count_before_yaw_sweep": len(base_spawn_poses),
        "yaw_offsets_deg": [float(item) for item in yaw_offsets_deg],
        "augmented_spawn_pose_count": len(augmented_spawn_poses),
        "free_space_sampling_seed": int(free_space_sampling_seed),
        "min_spawn_wall_clearance_m": float(min_spawn_wall_clearance_m),
        "min_spawn_obstacle_clearance_m": float(min_spawn_obstacle_clearance_m),
        "extra_sample_rejection_counts": dict(sorted(rejected_extra_samples.items())),
        "extra_sample_rows": extra_sample_rows,
        "yaw_augmented_rows": yaw_augmented_rows[:200],
    }
    return augmented, report


def _ordered_spawn_indices_for_motion(
    metadata,
    motion_id,
    placement_search_seed=0,
):
    grouped = defaultdict(list)
    for spawn_index, spawn_pose in enumerate(metadata.get("spawn_poses", [])):
        grouped[str(spawn_pose.get("room"))].append(int(spawn_index))
    room_names = sorted(grouped.keys())
    ordered = []
    rng = random.Random(f"{int(placement_search_seed)}::{motion_id}")
    for room_name in room_names:
        room_indices = list(grouped[room_name])
        rng.shuffle(room_indices)
        grouped[room_name] = room_indices
    max_len = max((len(indices) for indices in grouped.values()), default=0)
    for i in range(max_len):
        for room_name in room_names:
            room_indices = grouped[room_name]
            if i < len(room_indices):
                ordered.append(room_indices[i])
    return ordered


def _categorize_rejection_reason(reason):
    text = str(reason or "").lower()
    if "no usable floor polygon" in text or "could not find room" in text:
        return "missing_metadata"
    if "room erosion" in text or "exits room polygon" in text:
        return "room_feasibility"
    if "intersects obstacles" in text:
        return "collision"
    return "other"


def _candidate_generation_diagnostics(
    motion_set_report,
    resolved_motion_root_report,
    motion_payloads,
    all_results,
    valid_by_room,
    valid_groups_by_room,
    scene_collision_filter_report,
    npz_root,
):
    allowed_motion_ids = set(motion_set_report.get("motion_ids") or [])
    selected_motion_ids = [_normalize_motion_id(motion_id) for motion_id in resolved_motion_root_report.get("selected_motion_ids", [])]
    selected_motion_id_set = set(selected_motion_ids)
    valid_single_motion_counts = Counter()
    valid_multi_motion_counts = Counter()
    results_by_motion = defaultdict(list)
    reasons_by_motion = defaultdict(Counter)
    category_counts_by_motion = defaultdict(Counter)
    valid_rooms_by_motion = defaultdict(set)
    tested_spawn_count_by_motion = Counter()
    scene_collision_rejection_by_motion = Counter()
    path_metrics_by_motion = {}
    for motion_root_path, payload in motion_payloads:
        motion_id = _normalize_motion_id(_motion_id_from_payload(payload, fallback_path=motion_root_path))
        path_metrics_by_motion[motion_id] = _motion_path_metrics_from_payload(payload)

    for rows in valid_by_room.values():
        for row in rows:
            valid_single_motion_counts[_normalize_motion_id(row.get("motion_id"))] += 1

    for groups in valid_groups_by_room.values():
        for group in groups:
            for human in group.get("humans", []):
                valid_multi_motion_counts[_normalize_motion_id(human.get("motion_id"))] += 1

    for row in all_results:
        motion_id = _normalize_motion_id(row.get("motion_id"))
        results_by_motion[motion_id].append(row)
        tested_spawn_count_by_motion[motion_id] += 1
        if bool((row.get("human_scene_collision_audit") or {}).get("human_scene_collision_detected", False)):
            scene_collision_rejection_by_motion[motion_id] += 1
        for reason in row.get("reasons", []):
            reasons_by_motion[motion_id][str(reason)] += 1
            category_counts_by_motion[motion_id][_categorize_rejection_reason(reason)] += 1
        if row.get("pass"):
            valid_rooms_by_motion[motion_id].add(row.get("room"))

    all_motion_ids = sorted(allowed_motion_ids | selected_motion_id_set | set(results_by_motion.keys()))
    diagnostics_rows = []
    rejection_reason_counts = Counter()
    bottleneck_counts = Counter()
    accepted_path_lengths = []
    rejected_path_lengths = []
    accepted_path_bins = Counter()
    rejected_path_bins = Counter()
    for motion_id in all_motion_ids:
        identity = _motion_identity(motion_id)
        npz_path = Path(npz_root) / f"{motion_id}.npz"
        tested_rows = results_by_motion.get(motion_id, [])
        tested = bool(tested_rows)
        valid_single_pairs = int(valid_single_motion_counts.get(motion_id, 0))
        valid_multi_groups = int(valid_multi_motion_counts.get(motion_id, 0))
        category_counts = category_counts_by_motion.get(motion_id, Counter())
        reason_counts = reasons_by_motion.get(motion_id, Counter())
        has_valid = valid_single_pairs > 0
        rejected_by_collision = bool(tested and not has_valid and category_counts.get("collision", 0) > 0)
        rejected_by_room_feasibility = bool(
            tested and not has_valid and category_counts.get("room_feasibility", 0) > 0
        )
        rejected_by_missing_metadata = bool(
            (not npz_path.is_file()) or (tested and not has_valid and category_counts.get("missing_metadata", 0) > 0)
        )
        path_metrics = path_metrics_by_motion.get(motion_id) or {
            "path_length_m": None,
            "max_displacement_from_start_m": None,
        }
        path_length_m = path_metrics.get("path_length_m")
        if tested and not has_valid:
            if rejected_by_collision:
                bottleneck_counts["collision_rejection"] += 1
            elif rejected_by_room_feasibility:
                bottleneck_counts["room_feasibility_rejection"] += 1
            elif rejected_by_missing_metadata:
                bottleneck_counts["missing_metadata"] += 1
            else:
                bottleneck_counts["other_rejection"] += 1
        elif not tested:
            bottleneck_counts["not_tested"] += 1
        if path_length_m is not None:
            if has_valid:
                accepted_path_lengths.append(path_length_m)
                accepted_path_bins[_path_length_bin_label(path_length_m)] += 1
            elif tested:
                rejected_path_lengths.append(path_length_m)
                rejected_path_bins[_path_length_bin_label(path_length_m)] += 1
        for reason, count in reason_counts.items():
            rejection_reason_counts[reason] += int(count)
        diagnostics_rows.append(
            {
                "motion_id": motion_id,
                "identity": identity,
                "root_trajectory_available": bool(npz_path.is_file()),
                "geometry_cache_renderable": motion_id in allowed_motion_ids if allowed_motion_ids else None,
                "tested_in_collision_planner": tested,
                "tested_spawn_pair_count": int(tested_spawn_count_by_motion.get(motion_id, 0)),
                "valid_single_pair_count": valid_single_pairs,
                "valid_multi_human_group_memberships": valid_multi_groups,
                "valid_room_count": len(valid_rooms_by_motion.get(motion_id, set())),
                "rejected_by_scene_collision": int(scene_collision_rejection_by_motion.get(motion_id, 0)),
                "rejected_by_collision": rejected_by_collision,
                "rejected_by_room_feasibility": rejected_by_room_feasibility,
                "rejected_by_missing_metadata": rejected_by_missing_metadata,
                "path_length_m": path_length_m,
                "max_displacement_from_start_m": path_metrics.get("max_displacement_from_start_m"),
                "rejection_reason_counts": dict(sorted(reason_counts.items())),
            }
        )

    return {
        "motion_set_mode": motion_set_report.get("motion_set_mode"),
        "starterpack_motion_id_count": len(allowed_motion_ids) if allowed_motion_ids else None,
        "resolved_motion_root_report": resolved_motion_root_report,
        "total_motion_ids_considered": len(all_motion_ids),
        "tested_motion_id_count": sum(1 for row in diagnostics_rows if row["tested_in_collision_planner"]),
        "valid_single_motion_id_count": sum(1 for row in diagnostics_rows if row["valid_single_pair_count"] > 0),
        "valid_multi_motion_id_count": sum(
            1 for row in diagnostics_rows if row["valid_multi_human_group_memberships"] > 0
        ),
        "unique_identities_in_valid_single_pairs": len(
            {_motion_identity(motion_id) for motion_id, count in valid_single_motion_counts.items() if count > 0}
        ),
        "unique_identities_in_valid_multi_groups": len(
            {_motion_identity(motion_id) for motion_id, count in valid_multi_motion_counts.items() if count > 0}
        ),
        "aggregate_rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "aggregate_bottleneck_counts": dict(sorted(bottleneck_counts.items())),
        "accepted_motion_path_length_distribution": _distribution_summary(accepted_path_lengths),
        "rejected_motion_path_length_distribution": _distribution_summary(rejected_path_lengths),
        "accepted_motion_path_length_bins": dict(sorted(accepted_path_bins.items())),
        "rejected_motion_path_length_bins": dict(sorted(rejected_path_bins.items())),
        "scene_collision_filter_report": scene_collision_filter_report,
        "motions": diagnostics_rows,
    }


def _starterpack_manifest_filter_report(
    metadata,
    motion_set_report,
    prefilter_manifest,
    valid_single_before,
    valid_single_after,
    valid_two_before,
    valid_two_after,
    manifest,
    scene_collision_filter_report=None,
):
    room_names = sorted(room.get("name") for room in metadata.get("rooms", []))
    per_room = {}
    for room_name in room_names:
        single_before = len(valid_single_before.get(room_name, []))
        single_after = len(valid_single_after.get(room_name, []))
        two_before = len(valid_two_before.get(room_name, []))
        two_after = len(valid_two_after.get(room_name, []))
        per_room[room_name] = {
            "valid_single_pairs_before": single_before,
            "valid_single_pairs_after": single_after,
            "dropped_single_pairs": single_before - single_after,
            "valid_two_human_groups_before": two_before,
            "valid_two_human_groups_after": two_after,
            "dropped_two_human_groups": two_before - two_after,
            "selected_miniscenes_total": int(
                (manifest.get("selected_per_room", {}).get(room_name, {}) or {}).get("total", 0)
            ),
        }

    all_motion_ids_before = sorted(
        {
            _normalize_motion_id(row.get("motion_id"))
            for rows in valid_single_before.values()
            for row in rows
        }
    )
    all_motion_ids_after = sorted(
        {
            _normalize_motion_id(row.get("motion_id"))
            for rows in valid_single_after.values()
            for row in rows
        }
    )
    dropped_motion_ids = sorted(set(all_motion_ids_before) - set(all_motion_ids_after))
    available_motion_distribution = _motion_distribution_from_single_pairs(valid_single_after)
    prefilter_scene_ids = sorted(
        scene.get("miniscene_id") for scene in (prefilter_manifest.get("miniscenes", []) or [])
    )
    kept_scene_ids = sorted(scene.get("miniscene_id") for scene in manifest.get("miniscenes", []))
    dropped_scene_ids = sorted(set(prefilter_scene_ids) - set(kept_scene_ids))

    return {
        "phase": "starterpack_only_manifest_generation",
        "motion_set": motion_set_report,
        "kept_scene_count": len(kept_scene_ids),
        "kept_scene_ids": kept_scene_ids,
        "dropped_scene_count": len(dropped_scene_ids),
        "dropped_scene_ids": dropped_scene_ids,
        "rooms_covered": sorted(
            room_name
            for room_name, row in (manifest.get("selected_per_room", {}) or {}).items()
            if int(row.get("total", 0)) > 0
        ),
        "rooms_without_renderable_miniscenes": list(
            manifest.get("rooms_without_renderable_miniscenes", [])
        ),
        "missing_motion_ids": dropped_motion_ids,
        "available_motion_distribution": available_motion_distribution,
        "scene_collision_filter_report": scene_collision_filter_report,
        "per_room": per_room,
        "summary": {
            "selected_single_human_count": int(manifest.get("selected_single_human_count", 0)),
            "selected_two_human_count": int(manifest.get("selected_two_human_count", 0)),
            "total_selected_miniscenes": int(manifest.get("total_selected_miniscenes", 0)),
            "valid_single_pairs_before": int(sum(len(v) for v in valid_single_before.values())),
            "valid_single_pairs_after": int(sum(len(v) for v in valid_single_after.values())),
            "valid_two_human_groups_before": int(sum(len(v) for v in valid_two_before.values())),
            "valid_two_human_groups_after": int(sum(len(v) for v in valid_two_after.values())),
            "rejected_by_scene_collision": int(
                (scene_collision_filter_report or {}).get("rejected_by_scene_collision", 0)
            ),
        },
    }


def _trajectory_stats(trajectory_result):
    points_xy = trajectory_result.get("points_xy", [])
    return {
        "point_count": len(points_xy),
        "frame_count": len(trajectory_result.get("transformed_frames", [])),
        "z_band_m": list(trajectory_result.get("z_band_m", [])),
    }


def _single_pair_row(evaluation, spawn_index, motion_root_path):
    spawn_pose = evaluation["spawn_pose"]
    trajectory_result = evaluation["trajectory_result"]
    source_motion = trajectory_result.get("source_motion", {})
    return {
        "room": evaluation["room_name"],
        "spawn_index": int(spawn_index),
        "pair_key": f"{evaluation['room_name']}::{int(spawn_index)}::{str(motion_root_path)}",
        "spawn_pose": _spawn_pose_summary(spawn_index, spawn_pose),
        "motion_id": _motion_id_from_payload(source_motion, fallback_path=motion_root_path),
        "motion_root_path": str(motion_root_path),
        "source_motion": source_motion,
        "pass": bool(trajectory_result["pass"]),
        "reasons": list(trajectory_result["reasons"]),
        "collision_objects": list(trajectory_result.get("collisions", [])),
        "trajectory_stats": _trajectory_stats(trajectory_result),
    }


def _group_spawn_poses_by_room(metadata):
    grouped = defaultdict(list)
    for spawn_index, spawn_pose in enumerate(metadata.get("spawn_poses", [])):
        grouped[spawn_pose.get("room")].append(_spawn_pose_summary(spawn_index, spawn_pose))
    return dict(grouped)


def _build_single_pair_cache(
    metadata,
    motion_root_paths,
    max_placement_attempts_per_motion=0,
    max_valid_placements_per_motion=0,
    max_valid_placements_per_room=0,
    placement_search_seed=0,
):
    motion_payloads = []
    for motion_root_path in motion_root_paths:
        payload = _load_motion_root_json(motion_root_path)
        motion_payloads.append((Path(motion_root_path), payload))

    all_results = []
    valid_by_room = defaultdict(list)
    geometry_cache = {}

    for motion_root_path, payload in motion_payloads:
        motion_id = _motion_id_from_payload(payload, fallback_path=motion_root_path)
        spawn_indices = _ordered_spawn_indices_for_motion(
            metadata,
            motion_id,
            placement_search_seed=placement_search_seed,
        )
        attempts_for_motion = 0
        valid_for_motion = 0
        valid_for_room = Counter()
        for spawn_index in spawn_indices:
            if (
                int(max_placement_attempts_per_motion) > 0
                and attempts_for_motion >= int(max_placement_attempts_per_motion)
            ):
                break
            spawn_pose = metadata["spawn_poses"][spawn_index]
            room_name = str(spawn_pose.get("room"))
            if (
                int(max_valid_placements_per_room) > 0
                and valid_for_room[room_name] >= int(max_valid_placements_per_room)
            ):
                continue
            if (
                int(max_valid_placements_per_motion) > 0
                and valid_for_motion >= int(max_valid_placements_per_motion)
            ):
                break
            attempts_for_motion += 1
            evaluation = _evaluate_motion_root_pair(metadata, spawn_index, payload)
            row = _single_pair_row(evaluation, spawn_index, motion_root_path)
            all_results.append(row)
            if row["pass"]:
                valid_by_room[row["room"]].append(row)
                valid_for_motion += 1
                valid_for_room[row["room"]] += 1
                geometry_cache[row["pair_key"]] = {
                    "swept_polygon": evaluation["trajectory_result"]["swept_polygon"],
                    "transformed_frames": list(
                        evaluation["trajectory_result"].get("transformed_frames", [])
                    ),
                    "row": row,
                }

    return motion_payloads, all_results, dict(valid_by_room), geometry_cache


def _two_human_group_row(
    room_name,
    left_row,
    right_row,
    interhuman_mode,
    overlap_area=None,
    min_distance_m=None,
    min_distance_frame_index=None,
):
    left_motion = left_row["motion_id"]
    right_motion = right_row["motion_id"]
    left_spawn = left_row["spawn_index"]
    right_spawn = right_row["spawn_index"]
    group_id = (
        f"{room_name}__spawn{left_spawn:03d}_{left_motion}__"
        f"spawn{right_spawn:03d}_{right_motion}"
    )
    return {
        "group_id": group_id,
        "room": room_name,
        "humans": [
            {
                "spawn_index": left_spawn,
                "motion_id": left_motion,
                "spawn_pose": left_row["spawn_pose"],
                "motion_root_path": left_row["motion_root_path"],
            },
            {
                "spawn_index": right_spawn,
                "motion_id": right_motion,
                "spawn_pose": right_row["spawn_pose"],
                "motion_root_path": right_row["motion_root_path"],
            },
        ],
        "interhuman_mode": interhuman_mode,
        "interhuman_overlap_area_m2": None
        if overlap_area is None
        else float(overlap_area),
        "min_interhuman_distance_m": None
        if min_distance_m is None
        else float(min_distance_m),
        "min_interhuman_distance_frame_index": None
        if min_distance_frame_index is None
        else int(min_distance_frame_index),
        "pass": True,
        "reasons": [],
    }


def _failed_two_human_row(
    room_name,
    left_row,
    right_row,
    interhuman_mode,
    required_threshold_m,
    min_distance_m=None,
    min_distance_frame_index=None,
    overlap_area=None,
):
    left_motion = left_row["motion_id"]
    right_motion = right_row["motion_id"]
    left_spawn = left_row["spawn_index"]
    right_spawn = right_row["spawn_index"]
    distance_gap_m = None
    if min_distance_m is not None:
        distance_gap_m = float(min_distance_m) - float(required_threshold_m)
    return {
        "room": room_name,
        "interhuman_mode": interhuman_mode,
        "human_a": {
            "spawn_index": left_spawn,
            "motion_id": left_motion,
            "motion_root_path": left_row["motion_root_path"],
            "single_valid": bool(left_row["pass"]),
        },
        "human_b": {
            "spawn_index": right_spawn,
            "motion_id": right_motion,
            "motion_root_path": right_row["motion_root_path"],
            "single_valid": bool(right_row["pass"]),
        },
        "min_interhuman_distance_m": None
        if min_distance_m is None
        else float(min_distance_m),
        "min_interhuman_distance_frame_index": None
        if min_distance_frame_index is None
        else int(min_distance_frame_index),
        "required_threshold_m": float(required_threshold_m),
        "distance_gap_m": distance_gap_m,
        "interhuman_overlap_area_m2": None
        if overlap_area is None
        else float(overlap_area),
    }


def _pair_key_for_human_entry(room_name, human_entry):
    return (
        f"{room_name}::{int(human_entry['spawn_index'])}::"
        f"{str(human_entry['motion_root_path'])}"
    )


def _group_row_from_members(
    room_name,
    member_rows,
    pair_metrics,
    interhuman_mode,
):
    sorted_members = sorted(
        member_rows,
        key=lambda row: (
            int(row["spawn_index"]),
            str(row["motion_id"]),
            str(row["motion_root_path"]),
        ),
    )
    group_id = "__".join(
        [
            room_name,
            f"{len(sorted_members)}h",
            *[
                f"spawn{int(row['spawn_index']):03d}_{str(row['motion_id'])}"
                for row in sorted_members
            ],
        ]
    )
    min_pair_metric = None
    if pair_metrics:
        min_pair_metric = min(
            pair_metrics,
            key=lambda row: float(row.get("min_interhuman_distance_m") or float("inf")),
        )
    return {
        "group_id": group_id,
        "room": room_name,
        "human_count": len(sorted_members),
        "humans": [
            {
                "spawn_index": int(row["spawn_index"]),
                "motion_id": row["motion_id"],
                "spawn_pose": row["spawn_pose"],
                "motion_root_path": row["motion_root_path"],
            }
            for row in sorted_members
        ],
        "interhuman_mode": interhuman_mode,
        "interhuman_overlap_area_m2": None,
        "min_interhuman_distance_m": None
        if min_pair_metric is None
        else float(min_pair_metric.get("min_interhuman_distance_m")),
        "min_interhuman_distance_frame_index": None
        if min_pair_metric is None
        else int(min_pair_metric.get("min_interhuman_distance_frame_index")),
        "pairwise_validation_pairs": [
            {
                "left_spawn_index": int(metric["humans"][0]["spawn_index"]),
                "left_motion_id": metric["humans"][0]["motion_id"],
                "right_spawn_index": int(metric["humans"][1]["spawn_index"]),
                "right_motion_id": metric["humans"][1]["motion_id"],
                "min_interhuman_distance_m": metric.get("min_interhuman_distance_m"),
                "min_interhuman_distance_frame_index": metric.get(
                    "min_interhuman_distance_frame_index"
                ),
            }
            for metric in pair_metrics
        ],
        "pass": True,
        "reasons": [],
    }


def _normalized_frame_sample(frames, normalized_index, sample_count):
    if not frames:
        return None, None
    if len(frames) == 1 or sample_count <= 1:
        return 0, frames[0]
    t = float(normalized_index) / float(sample_count - 1)
    frame_index = int(round(t * (len(frames) - 1)))
    frame_index = max(0, min(frame_index, len(frames) - 1))
    return frame_index, frames[frame_index]


def _synchronized_interhuman_metrics(left_frames, right_frames):
    if not left_frames or not right_frames:
        return None
    sample_count = max(len(left_frames), len(right_frames))
    best = None
    for normalized_index in range(sample_count):
        left_frame_index, left_frame = _normalized_frame_sample(
            left_frames, normalized_index, sample_count
        )
        right_frame_index, right_frame = _normalized_frame_sample(
            right_frames, normalized_index, sample_count
        )
        dx = float(left_frame["world_x_m"]) - float(right_frame["world_x_m"])
        dy = float(left_frame["world_y_m"]) - float(right_frame["world_y_m"])
        dist = float(math.hypot(dx, dy))
        row = {
            "distance_m": dist,
            "normalized_frame_index": int(normalized_index),
            "left_frame_index": int(left_frame_index),
            "right_frame_index": int(right_frame_index),
        }
        if best is None or row["distance_m"] < best["distance_m"]:
            best = row
    return best


def _build_multi_human_groups(
    valid_by_room,
    geometry_cache,
    interhuman_mode="synchronized",
    overlap_eps=1e-6,
    interhuman_margin_m=0.05,
    near_miss_limit=50,
    max_human_count=4,
    allow_duplicate_motion_in_group=False,
):
    valid_two_human_groups_by_room = {}
    valid_groups_by_room = {}
    failed_near_misses_by_room = {}
    room_stats = {}
    max_human_count = max(2, int(max_human_count))
    for room_name, valid_rows in valid_by_room.items():
        room_groups = []
        pair_groups = []
        pair_lookup = {}
        room_failed = []
        min_distances = []
        attempted_pair_count = 0
        required_threshold_m = (2.0 * CAPSULE_RADIUS_M) + float(interhuman_margin_m)
        for left_row, right_row in combinations(valid_rows, 2):
            if left_row["spawn_index"] == right_row["spawn_index"]:
                continue
            attempted_pair_count += 1
            left_key = left_row["pair_key"]
            right_key = right_row["pair_key"]
            left_geom = geometry_cache.get(left_key)
            right_geom = geometry_cache.get(right_key)
            if left_geom is None or right_geom is None:
                continue
            if interhuman_mode == "swept":
                overlap_area = left_geom["swept_polygon"].intersection(
                    right_geom["swept_polygon"]
                ).area
                if overlap_area > overlap_eps:
                    room_failed.append(
                        _failed_two_human_row(
                            room_name,
                            left_row,
                            right_row,
                            interhuman_mode=interhuman_mode,
                            required_threshold_m=required_threshold_m,
                            overlap_area=overlap_area,
                        )
                    )
                    continue
                pair_group = _two_human_group_row(
                    room_name,
                    left_row,
                    right_row,
                    interhuman_mode=interhuman_mode,
                    overlap_area=overlap_area,
                )
            else:
                metrics = _synchronized_interhuman_metrics(
                    left_geom.get("transformed_frames", []),
                    right_geom.get("transformed_frames", []),
                )
                if metrics is None:
                    continue
                min_distances.append(metrics["distance_m"])
                if metrics["distance_m"] < required_threshold_m:
                    room_failed.append(
                        _failed_two_human_row(
                            room_name,
                            left_row,
                            right_row,
                            interhuman_mode=interhuman_mode,
                            required_threshold_m=required_threshold_m,
                            min_distance_m=metrics["distance_m"],
                            min_distance_frame_index=metrics["normalized_frame_index"],
                        )
                    )
                    continue
                pair_group = _two_human_group_row(
                    room_name,
                    left_row,
                    right_row,
                    interhuman_mode=interhuman_mode,
                    min_distance_m=metrics["distance_m"],
                    min_distance_frame_index=metrics["normalized_frame_index"],
                )
            pair_groups.append(pair_group)
            room_groups.append(pair_group)
            pair_lookup[frozenset((left_key, right_key))] = pair_group

        valid_two_human_groups_by_room[room_name] = list(pair_groups)

        adjacency = defaultdict(set)
        for group in pair_groups:
            left_key = _pair_key_for_human_entry(room_name, group["humans"][0])
            right_key = _pair_key_for_human_entry(room_name, group["humans"][1])
            adjacency[left_key].add(right_key)
            adjacency[right_key].add(left_key)

        row_by_key = {row["pair_key"]: row for row in valid_rows}
        node_keys = sorted(row_by_key.keys())
        for group_size in range(3, max_human_count + 1):
            for combo_keys in combinations(node_keys, group_size):
                combo_rows = [row_by_key[key] for key in combo_keys]
                spawn_indices = [int(row["spawn_index"]) for row in combo_rows]
                if len(set(spawn_indices)) != len(spawn_indices):
                    continue
                motion_ids = [_normalize_motion_id(row["motion_id"]) for row in combo_rows]
                if (not allow_duplicate_motion_in_group) and len(set(motion_ids)) != len(motion_ids):
                    continue
                pair_metrics = []
                is_clique = True
                for left_key, right_key in combinations(combo_keys, 2):
                    if right_key not in adjacency.get(left_key, set()):
                        is_clique = False
                        break
                    pair_metric = pair_lookup.get(frozenset((left_key, right_key)))
                    if pair_metric is None:
                        is_clique = False
                        break
                    pair_metrics.append(pair_metric)
                if not is_clique:
                    continue
                room_groups.append(
                    _group_row_from_members(
                        room_name,
                        combo_rows,
                        pair_metrics,
                        interhuman_mode,
                    )
                )

        valid_groups_by_room[room_name] = room_groups
        room_failed.sort(
            key=lambda row: (
                -float("-inf")
                if row["distance_gap_m"] is None
                else row["distance_gap_m"]
            ),
            reverse=True,
        )
        failed_near_misses_by_room[room_name] = room_failed[: int(near_miss_limit)]
        room_stats[room_name] = {
            "valid_single_pairs": len(valid_rows),
            "attempted_two_human_pairs": attempted_pair_count,
            "valid_two_human_groups": len(pair_groups),
            "valid_multi_human_groups": len(room_groups),
            "valid_group_count_by_human_count": {
                str(human_count): sum(
                    1
                    for group in room_groups
                    if int(group.get("human_count", 2)) == human_count
                )
                for human_count in range(2, max_human_count + 1)
            },
            "required_threshold_m": float(required_threshold_m),
            "max_min_distance_m": None
            if not min_distances
            else float(max(min_distances)),
            "median_min_distance_m": None
            if not min_distances
            else float(median(min_distances)),
        }
    return (
        valid_two_human_groups_by_room,
        valid_groups_by_room,
        failed_near_misses_by_room,
        room_stats,
    )


def _expand_pair_groups_to_multi_groups(
    valid_by_room,
    valid_two_by_room,
    max_human_count=4,
    allow_duplicate_motion_in_group=False,
):
    max_human_count = max(2, int(max_human_count))
    valid_groups_by_room = {}
    room_stats = {}
    for room_name, valid_rows in valid_by_room.items():
        pair_groups = list(valid_two_by_room.get(room_name, []))
        room_groups = list(pair_groups)
        pair_lookup = {}
        adjacency = defaultdict(set)
        row_by_key = {row["pair_key"]: row for row in valid_rows}
        for group in pair_groups:
            if len(group.get("humans", [])) != 2:
                continue
            left_key = _pair_key_for_human_entry(room_name, group["humans"][0])
            right_key = _pair_key_for_human_entry(room_name, group["humans"][1])
            pair_lookup[frozenset((left_key, right_key))] = group
            adjacency[left_key].add(right_key)
            adjacency[right_key].add(left_key)
        node_keys = sorted(row_by_key.keys())
        for group_size in range(3, max_human_count + 1):
            for combo_keys in combinations(node_keys, group_size):
                combo_rows = [row_by_key[key] for key in combo_keys]
                spawn_indices = [int(row["spawn_index"]) for row in combo_rows]
                if len(set(spawn_indices)) != len(spawn_indices):
                    continue
                motion_ids = [_normalize_motion_id(row["motion_id"]) for row in combo_rows]
                if (not allow_duplicate_motion_in_group) and len(set(motion_ids)) != len(motion_ids):
                    continue
                pair_metrics = []
                is_clique = True
                for left_key, right_key in combinations(combo_keys, 2):
                    if right_key not in adjacency.get(left_key, set()):
                        is_clique = False
                        break
                    pair_metric = pair_lookup.get(frozenset((left_key, right_key)))
                    if pair_metric is None:
                        is_clique = False
                        break
                    pair_metrics.append(pair_metric)
                if not is_clique:
                    continue
                room_groups.append(
                    _group_row_from_members(
                        room_name,
                        combo_rows,
                        pair_metrics,
                        pair_groups[0].get("interhuman_mode") if pair_groups else "synchronized",
                    )
                )
        valid_groups_by_room[room_name] = room_groups
        room_stats[room_name] = {
            "valid_single_pairs": len(valid_rows),
            "valid_two_human_groups": len(pair_groups),
            "valid_multi_human_groups": len(room_groups),
            "valid_group_count_by_human_count": {
                str(human_count): sum(
                    1
                    for group in room_groups
                    if int(group.get("human_count", 2)) == human_count
                )
                for human_count in range(2, max_human_count + 1)
            },
        }
    return valid_groups_by_room, room_stats


def _valid_single_pairs_payload(metadata_path, metadata, valid_by_room):
    rooms_payload = []
    grouped_spawns = _group_spawn_poses_by_room(metadata)
    room_names = sorted(room.get("name") for room in metadata.get("rooms", []))
    for room_name in room_names:
        room_record = _find_room(metadata, room_name)
        rooms_payload.append(
            {
                "room": _room_summary(room_record),
                "spawn_poses": grouped_spawns.get(room_name, []),
                "valid_pairs": valid_by_room.get(room_name, []),
            }
        )
    return {
        "scene_path": str(metadata_path),
        "capsule_radius_m": CAPSULE_RADIUS_M,
        "capsule_height_m": CAPSULE_HEIGHT_M,
        "room_count": len(metadata.get("rooms", [])),
        "spawn_pose_count": len(metadata.get("spawn_poses", [])),
        "rooms": rooms_payload,
    }


def _all_single_pair_results_payload(metadata_path, metadata, motion_payloads, all_results):
    return {
        "scene_path": str(metadata_path),
        "capsule_radius_m": CAPSULE_RADIUS_M,
        "capsule_height_m": CAPSULE_HEIGHT_M,
        "room_count": len(metadata.get("rooms", [])),
        "spawn_pose_count": len(metadata.get("spawn_poses", [])),
        "motion_count": len(motion_payloads),
        "motions": [_motion_summary(path, payload) for path, payload in motion_payloads],
        "results": all_results,
    }


def _valid_two_human_groups_payload(
    metadata_path,
    metadata,
    valid_groups_by_room,
    interhuman_mode,
    interhuman_margin_m,
):
    grouped_spawns = _group_spawn_poses_by_room(metadata)
    rooms_payload = []
    room_names = sorted(room.get("name") for room in metadata.get("rooms", []))
    for room_name in room_names:
        room_record = _find_room(metadata, room_name)
        rooms_payload.append(
            {
                "room": _room_summary(room_record),
                "spawn_poses": grouped_spawns.get(room_name, []),
                "valid_two_human_groups": valid_groups_by_room.get(room_name, []),
            }
        )
    return {
        "scene_path": str(metadata_path),
        "capsule_radius_m": CAPSULE_RADIUS_M,
        "capsule_height_m": CAPSULE_HEIGHT_M,
        "interhuman_mode": interhuman_mode,
        "interhuman_margin_m": float(interhuman_margin_m),
        "room_count": len(metadata.get("rooms", [])),
        "spawn_pose_count": len(metadata.get("spawn_poses", [])),
        "rooms": rooms_payload,
    }


def _valid_multi_human_groups_payload(
    metadata_path,
    metadata,
    valid_groups_by_room,
    interhuman_mode,
    interhuman_margin_m,
):
    grouped_spawns = _group_spawn_poses_by_room(metadata)
    rooms_payload = []
    room_names = sorted(room.get("name") for room in metadata.get("rooms", []))
    for room_name in room_names:
        room_record = _find_room(metadata, room_name)
        rooms_payload.append(
            {
                "room": _room_summary(room_record),
                "spawn_poses": grouped_spawns.get(room_name, []),
                "valid_multi_human_groups": valid_groups_by_room.get(room_name, []),
            }
        )
    return {
        "scene_path": str(metadata_path),
        "capsule_radius_m": CAPSULE_RADIUS_M,
        "capsule_height_m": CAPSULE_HEIGHT_M,
        "interhuman_mode": interhuman_mode,
        "interhuman_margin_m": float(interhuman_margin_m),
        "room_count": len(metadata.get("rooms", [])),
        "spawn_pose_count": len(metadata.get("spawn_poses", [])),
        "rooms": rooms_payload,
    }


def _failed_two_human_near_misses_payload(
    metadata_path,
    metadata,
    failed_near_misses_by_room,
    room_stats,
    interhuman_mode,
    interhuman_margin_m,
):
    grouped_spawns = _group_spawn_poses_by_room(metadata)
    rooms_payload = []
    room_names = sorted(room.get("name") for room in metadata.get("rooms", []))
    for room_name in room_names:
        room_record = _find_room(metadata, room_name)
        rooms_payload.append(
            {
                "room": _room_summary(room_record),
                "spawn_poses": grouped_spawns.get(room_name, []),
                "stats": room_stats.get(room_name, {}),
                "near_misses": failed_near_misses_by_room.get(room_name, []),
            }
        )
    return {
        "scene_path": str(metadata_path),
        "capsule_radius_m": CAPSULE_RADIUS_M,
        "capsule_height_m": CAPSULE_HEIGHT_M,
        "interhuman_mode": interhuman_mode,
        "interhuman_margin_m": float(interhuman_margin_m),
        "room_count": len(metadata.get("rooms", [])),
        "spawn_pose_count": len(metadata.get("spawn_poses", [])),
        "rooms": rooms_payload,
    }


def _print_summary(
    metadata,
    motion_payloads,
    valid_by_room,
    valid_groups_by_room,
    failed_near_misses_by_room,
    room_stats,
):
    print(f"Rooms: {len(metadata.get('rooms', []))}")
    print(f"Spawn poses: {len(metadata.get('spawn_poses', []))}")
    print(f"Motions: {len(motion_payloads)}")
    print("")
    print("room | spawn_poses | valid_single_pairs | valid_multi_human_groups")
    grouped_spawns = _group_spawn_poses_by_room(metadata)
    room_names = sorted(room.get("name") for room in metadata.get("rooms", []))
    for room_name in room_names:
        spawn_count = len(grouped_spawns.get(room_name, []))
        single_count = len(valid_by_room.get(room_name, []))
        multi_count = len(valid_groups_by_room.get(room_name, []))
        print(f"{room_name} | {spawn_count} | {single_count} | {multi_count}")
    print("")
    print("Near-miss diagnostics")
    for room_name in room_names:
        stats = room_stats.get(room_name, {})
        print(
            f"{room_name}: valid_singles={stats.get('valid_single_pairs', 0)} "
            f"attempted_two_human_pairs={stats.get('attempted_two_human_pairs', 0)} "
            f"valid_group_count_by_human_count={stats.get('valid_group_count_by_human_count', {})} "
            f"max_min_distance={stats.get('max_min_distance_m')} "
            f"median_min_distance={stats.get('median_min_distance_m')} "
            f"required_threshold={stats.get('required_threshold_m')}"
        )
        near_misses = failed_near_misses_by_room.get(room_name, [])[:10]
        for row in near_misses:
            print(
                f"  A=({row['human_a']['spawn_index']},{row['human_a']['motion_id']}) "
                f"B=({row['human_b']['spawn_index']},{row['human_b']['motion_id']}) "
                f"min_d={row['min_interhuman_distance_m']} "
                f"frame={row['min_interhuman_distance_frame_index']} "
                f"gap={row['distance_gap_m']}"
            )


def _load_valid_pairs_by_room(path, key_name):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for room_entry in payload.get("rooms", []):
        room = room_entry.get("room", {}) or {}
        room_name = room.get("name")
        if room_name is None:
            continue
        result[room_name] = list(room_entry.get(key_name, []))
    return payload, result


def _spawn_pose_lookup(metadata):
    return {
        int(index): spawn_pose
        for index, spawn_pose in enumerate(metadata.get("spawn_poses", []))
    }


def _human_entry_from_spawn_pose(spawn_pose_index, motion_id, metadata_spawn_pose):
    return {
        "spawn_pose_index": int(spawn_pose_index),
        "motion_id": motion_id,
        "position_xyz_m": list(metadata_spawn_pose.get("position_xyz", [])),
        "yaw_rad": metadata_spawn_pose.get("yaw"),
        "activity_hint": metadata_spawn_pose.get("activity_hint"),
        "target_object": metadata_spawn_pose.get("target_object"),
        "room": metadata_spawn_pose.get("room"),
        "pose_type": metadata_spawn_pose.get("pose_type"),
        "source": metadata_spawn_pose.get("source"),
    }


def _point_to_boundary_clearance_m(room_poly, point_xy, radius_m):
    point = Point(point_xy)
    if room_poly is None or room_poly.is_empty:
        return None
    boundary_distance = float(room_poly.boundary.distance(point))
    if room_poly.covers(point):
        return boundary_distance - float(radius_m)
    return -boundary_distance - float(radius_m)


def _motion_payload_cache_from_manifest(
    manifest,
    output_dir,
    npz_root=DEFAULT_BEDLAM_NPZ_ROOT,
    npz_trans_ground_axes=DEFAULT_NPZ_TRANS_GROUND_AXES,
):
    motion_ids = sorted(
        {
            _normalize_motion_id(human.get("motion_id"))
            for miniscene in manifest.get("miniscenes", [])
            for human in miniscene.get("humans", [])
            if human.get("motion_id") is not None
        }
    )
    motion_root_dir = Path(output_dir) / "_audit_motion_roots"
    motion_root_dir.mkdir(parents=True, exist_ok=True)
    payload_by_motion_id = {}
    missing_motion_ids = []
    source_paths = {}
    root_cache_diagnostics = []
    for motion_id in motion_ids:
        motion_root_path = motion_root_dir / f"{motion_id}_root_trajectory.json"
        npz_path = Path(npz_root) / f"{motion_id}.npz"
        if not npz_path.is_file():
            missing_motion_ids.append(motion_id)
            continue
        root_cache_hit, reexport_reason, cache_metadata = _root_cache_validation(
            motion_id,
            motion_root_path,
            npz_path,
            npz_trans_ground_axes,
        )
        if not root_cache_hit:
            export_motion_root_from_npz(
                motion_id,
                motion_root_path,
                npz_root=npz_root,
                npz_trans_ground_axes=npz_trans_ground_axes,
            )
        root_cache_diagnostics.append(
            {
                "motion_id": motion_id,
                "root_cache_hit": bool(root_cache_hit),
                "root_cache_reexported": not bool(root_cache_hit),
                "root_cache_reexport_reason": reexport_reason,
                **cache_metadata,
                "output_json": str(motion_root_path),
            }
        )
        payload = _load_motion_root_json(motion_root_path)
        normalized_motion_id = _normalize_motion_id(
            _motion_id_from_payload(payload, fallback_path=motion_root_path)
        )
        payload_by_motion_id[normalized_motion_id] = payload
        source_paths[normalized_motion_id] = str(motion_root_path)
    return payload_by_motion_id, {
        "motion_root_dir": str(motion_root_dir),
        "resolved_motion_id_count": len(payload_by_motion_id),
        "resolved_motion_ids": sorted(payload_by_motion_id.keys()),
        "missing_motion_ids": sorted(missing_motion_ids),
        "source_paths": source_paths,
        "npz_root": str(npz_root),
        "npz_trans_ground_axes": str(npz_trans_ground_axes),
        "root_export_schema_version": int(ROOT_EXPORT_SCHEMA_VERSION),
        "root_cache_diagnostics": root_cache_diagnostics,
    }


def _audit_human_scene_collision(
    metadata,
    spawn_lookup,
    human_entry,
    motion_root_payload,
    radius_m,
    sample_stride=1,
    capsule_height_m=CAPSULE_HEIGHT_M,
    min_required_wall_clearance_m=MIN_HUMAN_WALL_CLEARANCE_M,
    min_required_obstacle_clearance_m=MIN_HUMAN_OBSTACLE_CLEARANCE_M,
):
    spawn_pose_index = int(human_entry["spawn_pose_index"])
    spawn_pose = spawn_lookup[spawn_pose_index]
    room_name = spawn_pose.get("room")
    room_record = _find_room(metadata, room_name)
    room_poly = _room_polygon(room_record) if room_record is not None else None
    if room_record is None or room_poly is None or room_poly.is_empty:
        return {
            "human_index": None,
            "spawn_pose_index": spawn_pose_index,
            "motion_id": _normalize_motion_id(human_entry.get("motion_id")),
            "room": room_name,
            "audit_success": False,
            "invalid_reason": "missing_room_polygon",
        }

    obstacles = _find_room_obstacles(metadata, room_name)
    spawn_xy = (
        float(spawn_pose["position_xyz"][0]),
        float(spawn_pose["position_xyz"][1]),
    )
    spawn_yaw = float(spawn_pose.get("yaw") or 0.0)
    points_xy, transformed_frames = _transform_motion_root_to_spawn(
        motion_root_payload["frames"],
        spawn_xy,
        spawn_yaw,
    )
    eroded_room_poly = room_poly.buffer(-float(radius_m))
    sample_stride = max(1, int(sample_stride))
    collision_frame_indices = []
    out_of_room_frame_indices = []
    collision_object_labels = set()
    collision_object_types = set()
    min_clearance_to_obstacle_m = None
    min_clearance_to_wall_m = None
    per_frame_rows = []
    wall_frame_clearances = []
    obstacle_frame_clearances = []

    for frame in transformed_frames[::sample_stride]:
        frame_index = int(frame["frame_index"])
        point_xy = (float(frame["world_x_m"]), float(frame["world_y_m"]))
        point = Point(point_xy)
        footprint = point.buffer(float(radius_m))
        frame_z = float(spawn_pose["position_xyz"][2]) + float(frame.get("world_z_m", 0.0))
        wall_clearance_m = _point_to_boundary_clearance_m(room_poly, point_xy, radius_m)
        if wall_clearance_m is not None:
            min_clearance_to_wall_m = (
                wall_clearance_m
                if min_clearance_to_wall_m is None
                else min(min_clearance_to_wall_m, wall_clearance_m)
            )
        wall_frame_clearances.append((frame_index, wall_clearance_m))
        inside_room = (not eroded_room_poly.is_empty) and bool(eroded_room_poly.covers(footprint))
        if not inside_room:
            out_of_room_frame_indices.append(frame_index)

        frame_collision_labels = []
        frame_collision_types = []
        frame_min_obstacle_clearance_m = None
        for obstacle in obstacles:
            poly = _obstacle_polygon(obstacle)
            if poly is None or poly.is_empty:
                continue
            z_min = obstacle.get("z_min")
            z_max = obstacle.get("z_max")
            if z_min is not None and z_max is not None:
                z_min = float(z_min)
                z_max = float(z_max)
                if z_max < frame_z or z_min > (frame_z + float(capsule_height_m)):
                    continue
            clearance_m = float(poly.distance(point) - float(radius_m))
            frame_min_obstacle_clearance_m = (
                clearance_m
                if frame_min_obstacle_clearance_m is None
                else min(frame_min_obstacle_clearance_m, clearance_m)
            )
            min_clearance_to_obstacle_m = (
                clearance_m
                if min_clearance_to_obstacle_m is None
                else min(min_clearance_to_obstacle_m, clearance_m)
            )
            obstacle_frame_clearances.append((frame_index, clearance_m))
            if footprint.intersects(poly):
                label = str(obstacle.get("object_name"))
                category = str(obstacle.get("category_hint"))
                frame_collision_labels.append(label)
                frame_collision_types.append(category)
                collision_object_labels.add(label)
                collision_object_types.add(category)
        if frame_collision_labels:
            collision_frame_indices.append(frame_index)
        per_frame_rows.append(
            {
                "frame_index": frame_index,
                "inside_room": bool(inside_room),
                "wall_clearance_m": wall_clearance_m,
                "obstacle_clearance_m": frame_min_obstacle_clearance_m,
                "collision_object_labels": frame_collision_labels,
                "collision_object_types": frame_collision_types,
            }
        )

    clearance_summary = _clearance_rejection_summary(
        min_clearance_to_wall_m,
        min_clearance_to_obstacle_m,
        wall_frame_clearances,
        obstacle_frame_clearances,
        min_required_wall_clearance_m=min_required_wall_clearance_m,
        min_required_obstacle_clearance_m=min_required_obstacle_clearance_m,
    )
    human_scene_collision_detected = bool(
        collision_frame_indices
        or out_of_room_frame_indices
        or clearance_summary["rejected_by_wall_clearance"]
        or clearance_summary["rejected_by_obstacle_clearance"]
    )
    return {
        "human_index": None,
        "spawn_pose_index": spawn_pose_index,
        "motion_id": _normalize_motion_id(human_entry.get("motion_id")),
        "room": room_name,
        "audit_success": True,
        "human_scene_collision_detected": human_scene_collision_detected,
        "collision_frame_indices": sorted(set(collision_frame_indices)),
        "collision_human_indices": [],
        "collision_object_labels": sorted(collision_object_labels),
        "collision_object_types": sorted(collision_object_types),
        "out_of_room_frame_indices": sorted(set(out_of_room_frame_indices)),
        "min_clearance_to_obstacle_m": min_clearance_to_obstacle_m,
        "min_clearance_to_wall_m": min_clearance_to_wall_m,
        "sample_stride": sample_stride,
        "sampled_frame_count": len(per_frame_rows),
        "per_frame_rows": per_frame_rows,
        **clearance_summary,
    }


def _audit_single_pair_row(
    metadata,
    row,
    geometry_entry,
    radius_m,
    sample_stride=1,
    capsule_height_m=CAPSULE_HEIGHT_M,
    min_required_wall_clearance_m=MIN_HUMAN_WALL_CLEARANCE_M,
    min_required_obstacle_clearance_m=MIN_HUMAN_OBSTACLE_CLEARANCE_M,
):
    spawn_pose = (row.get("spawn_pose") or {})
    room_name = row.get("room")
    room_record = _find_room(metadata, room_name)
    room_poly = _room_polygon(room_record) if room_record is not None else None
    if room_record is None or room_poly is None or room_poly.is_empty:
        return {
            "audit_success": False,
            "invalid_reason": "missing_room_polygon",
            "human_scene_collision_detected": True,
        }
    obstacles = _find_room_obstacles(metadata, room_name)
    eroded_room_poly = room_poly.buffer(-float(radius_m))
    transformed_frames = list((geometry_entry or {}).get("transformed_frames", []))
    sample_stride = max(1, int(sample_stride))
    base_z = float((spawn_pose.get("position_xyz") or [0.0, 0.0, 0.0])[2])
    collision_frame_indices = []
    out_of_room_frame_indices = []
    collision_object_labels = set()
    collision_object_types = set()
    min_clearance_to_obstacle_m = None
    min_clearance_to_wall_m = None
    wall_frame_clearances = []
    obstacle_frame_clearances = []

    for frame in transformed_frames[::sample_stride]:
        frame_index = int(frame["frame_index"])
        point_xy = (float(frame["world_x_m"]), float(frame["world_y_m"]))
        point = Point(point_xy)
        footprint = point.buffer(float(radius_m))
        wall_clearance_m = _point_to_boundary_clearance_m(room_poly, point_xy, radius_m)
        if wall_clearance_m is not None:
            min_clearance_to_wall_m = (
                wall_clearance_m
                if min_clearance_to_wall_m is None
                else min(min_clearance_to_wall_m, wall_clearance_m)
            )
        wall_frame_clearances.append((frame_index, wall_clearance_m))
        if eroded_room_poly.is_empty or (not eroded_room_poly.covers(footprint)):
            out_of_room_frame_indices.append(frame_index)
        frame_z = base_z + float(frame.get("world_z_m", 0.0))
        frame_had_collision = False
        for obstacle in obstacles:
            poly = _obstacle_polygon(obstacle)
            if poly is None or poly.is_empty:
                continue
            z_min = obstacle.get("z_min")
            z_max = obstacle.get("z_max")
            if z_min is not None and z_max is not None:
                z_min = float(z_min)
                z_max = float(z_max)
                if z_max < frame_z or z_min > (frame_z + float(capsule_height_m)):
                    continue
            clearance_m = float(poly.distance(point) - float(radius_m))
            min_clearance_to_obstacle_m = (
                clearance_m
                if min_clearance_to_obstacle_m is None
                else min(min_clearance_to_obstacle_m, clearance_m)
            )
            obstacle_frame_clearances.append((frame_index, clearance_m))
            if footprint.intersects(poly):
                frame_had_collision = True
                collision_object_labels.add(str(obstacle.get("object_name")))
                collision_object_types.add(str(obstacle.get("category_hint")))
        if frame_had_collision:
            collision_frame_indices.append(frame_index)

    clearance_summary = _clearance_rejection_summary(
        min_clearance_to_wall_m,
        min_clearance_to_obstacle_m,
        wall_frame_clearances,
        obstacle_frame_clearances,
        min_required_wall_clearance_m=min_required_wall_clearance_m,
        min_required_obstacle_clearance_m=min_required_obstacle_clearance_m,
    )
    return {
        "audit_success": True,
        "human_scene_collision_detected": bool(
            collision_frame_indices
            or out_of_room_frame_indices
            or clearance_summary["rejected_by_wall_clearance"]
            or clearance_summary["rejected_by_obstacle_clearance"]
        ),
        "collision_frame_indices": sorted(set(collision_frame_indices)),
        "collision_object_labels": sorted(collision_object_labels),
        "collision_object_types": sorted(collision_object_types),
        "out_of_room_frame_indices": sorted(set(out_of_room_frame_indices)),
        "min_clearance_to_obstacle_m": min_clearance_to_obstacle_m,
        "min_clearance_to_wall_m": min_clearance_to_wall_m,
        "human_collision_radius_m": float(radius_m),
        "human_collision_sample_stride": int(sample_stride),
        "frame_count": len(transformed_frames),
        **clearance_summary,
    }


def _filter_valid_pairs_by_human_scene_collision(
    metadata,
    all_results,
    valid_by_room,
    geometry_cache,
    radius_m,
    sample_stride=1,
    reject_human_scene_collisions=True,
    min_required_wall_clearance_m=MIN_HUMAN_WALL_CLEARANCE_M,
    min_required_obstacle_clearance_m=MIN_HUMAN_OBSTACLE_CLEARANCE_M,
):
    audited_rows = []
    filtered_valid_by_room = defaultdict(list)
    filtered_geometry_cache = {}
    rejected_by_room = Counter()
    rejected_object_types = Counter()
    kept_by_room = Counter()
    rejected_by_wall_clearance = Counter()
    rejected_by_obstacle_clearance = Counter()

    for row in all_results:
        audited_row = dict(row)
        geometry_entry = geometry_cache.get(row["pair_key"])
        if geometry_entry is None:
            audit = {
                "audit_success": False,
                "invalid_reason": "missing_geometry_cache_entry",
                "human_scene_collision_detected": True,
            }
        else:
            audit = _audit_single_pair_row(
                metadata,
                row,
                geometry_entry,
                radius_m=radius_m,
                sample_stride=sample_stride,
                min_required_wall_clearance_m=min_required_wall_clearance_m,
                min_required_obstacle_clearance_m=min_required_obstacle_clearance_m,
            )
        audited_row["human_scene_collision_audit"] = audit
        audited_row["human_scene_collision_detected"] = bool(
            audit.get("human_scene_collision_detected", False)
        )
        audited_rows.append(audited_row)
        if not bool(row.get("pass")):
            continue
        if bool(audit.get("human_scene_collision_detected")) and bool(reject_human_scene_collisions):
            rejected_by_room[row["room"]] += 1
            rejected_object_types.update(audit.get("collision_object_types", []))
            if bool(audit.get("rejected_by_wall_clearance")):
                rejected_by_wall_clearance[row["room"]] += 1
            if bool(audit.get("rejected_by_obstacle_clearance")):
                rejected_by_obstacle_clearance[row["room"]] += 1
            continue
        filtered_valid_by_room[row["room"]].append(audited_row)
        kept_by_room[row["room"]] += 1
        if geometry_entry is not None:
            filtered_geometry_cache[row["pair_key"]] = geometry_entry

    report = {
        "placement_attempt_count": len(all_results),
        "valid_single_pairs_before_scene_collision_filter": int(
            sum(len(rows) for rows in valid_by_room.values())
        ),
        "valid_single_pairs_after_scene_collision_filter": int(
            sum(len(rows) for rows in filtered_valid_by_room.values())
        ),
        "rejected_by_scene_collision": int(sum(rejected_by_room.values())),
        "rejected_by_wall_clearance": int(sum(rejected_by_wall_clearance.values())),
        "rejected_by_obstacle_clearance": int(sum(rejected_by_obstacle_clearance.values())),
        "reject_human_scene_collisions": bool(reject_human_scene_collisions),
        "human_collision_radius_m": float(radius_m),
        "human_collision_sample_stride": int(sample_stride),
        "min_required_wall_clearance_m": float(min_required_wall_clearance_m),
        "min_required_obstacle_clearance_m": float(min_required_obstacle_clearance_m),
        "valid_candidates_per_room": dict(sorted(kept_by_room.items())),
        "rejected_candidates_per_room": dict(sorted(rejected_by_room.items())),
        "rejected_by_wall_clearance_per_room": dict(sorted(rejected_by_wall_clearance.items())),
        "rejected_by_obstacle_clearance_per_room": dict(sorted(rejected_by_obstacle_clearance.items())),
        "rejected_obstacle_type_counts": dict(sorted(rejected_object_types.items())),
        "audited_results": audited_rows,
    }
    return dict(filtered_valid_by_room), filtered_geometry_cache, report


def _audit_manifest_human_scene_collisions(
    metadata,
    manifest,
    output_dir,
    radius_m,
    sample_stride=1,
    reject_human_scene_collisions=True,
    npz_root=DEFAULT_BEDLAM_NPZ_ROOT,
    npz_trans_ground_axes=DEFAULT_NPZ_TRANS_GROUND_AXES,
    min_required_wall_clearance_m=MIN_HUMAN_WALL_CLEARANCE_M,
    min_required_obstacle_clearance_m=MIN_HUMAN_OBSTACLE_CLEARANCE_M,
):
    output_dir = Path(output_dir)
    spawn_lookup = _spawn_pose_lookup(metadata)
    payload_by_motion_id, motion_payload_report = _motion_payload_cache_from_manifest(
        manifest,
        output_dir=output_dir,
        npz_root=npz_root,
        npz_trans_ground_axes=npz_trans_ground_axes,
    )
    audited_miniscenes = []
    scene_rows = []
    room_collision_counts = Counter()
    obstacle_type_counts = Counter()
    human_count_collision_counts = Counter()
    wall_clearance_rejection_count = 0
    obstacle_clearance_rejection_count = 0

    for miniscene in manifest.get("miniscenes", []):
        scene = dict(miniscene)
        human_audits = []
        all_collision_frames = set()
        collision_human_indices = set()
        collision_object_labels = set()
        collision_object_types = set()
        out_of_room_frames = set()
        min_clearance_to_obstacle_m = None
        min_clearance_to_wall_m = None
        audit_errors = []
        for human_index, human in enumerate(scene.get("humans", [])):
            motion_id = _normalize_motion_id(human.get("motion_id"))
            payload = payload_by_motion_id.get(motion_id)
            if payload is None:
                human_audit = {
                    "human_index": int(human_index),
                    "spawn_pose_index": int(human["spawn_pose_index"]),
                    "motion_id": motion_id,
                    "room": human.get("room"),
                    "audit_success": False,
                    "invalid_reason": "missing_motion_root_payload",
                }
                audit_errors.append(human_audit["invalid_reason"])
            else:
                human_audit = _audit_human_scene_collision(
                    metadata,
                    spawn_lookup,
                    human,
                    payload,
                    radius_m=radius_m,
                    sample_stride=sample_stride,
                    min_required_wall_clearance_m=min_required_wall_clearance_m,
                    min_required_obstacle_clearance_m=min_required_obstacle_clearance_m,
                )
                human_audit["human_index"] = int(human_index)
            human_audits.append(human_audit)
            if not human_audit.get("audit_success", False):
                continue
            if bool(human_audit.get("human_scene_collision_detected")):
                collision_human_indices.add(int(human_index))
                all_collision_frames.update(human_audit.get("collision_frame_indices", []))
                all_collision_frames.update(human_audit.get("out_of_room_frame_indices", []))
                all_collision_frames.update(human_audit.get("clearance_rejection_frame_indices", []))
                out_of_room_frames.update(human_audit.get("out_of_room_frame_indices", []))
                collision_object_labels.update(human_audit.get("collision_object_labels", []))
                collision_object_types.update(human_audit.get("collision_object_types", []))
            if bool(human_audit.get("rejected_by_wall_clearance")):
                wall_clearance_rejection_count += 1
            if bool(human_audit.get("rejected_by_obstacle_clearance")):
                obstacle_clearance_rejection_count += 1
            obstacle_clearance = human_audit.get("min_clearance_to_obstacle_m")
            if obstacle_clearance is not None:
                min_clearance_to_obstacle_m = (
                    obstacle_clearance
                    if min_clearance_to_obstacle_m is None
                    else min(min_clearance_to_obstacle_m, obstacle_clearance)
                )
            wall_clearance = human_audit.get("min_clearance_to_wall_m")
            if wall_clearance is not None:
                min_clearance_to_wall_m = (
                    wall_clearance
                    if min_clearance_to_wall_m is None
                    else min(min_clearance_to_wall_m, wall_clearance)
                )

        scene_collision_detected = bool(collision_human_indices) or bool(audit_errors)
        validation_summary = dict(scene.get("validation_summary") or {})
        validation_summary.update(
            {
                "scene_collision_pass": not bool(scene_collision_detected),
                "human_scene_collision_detected": bool(scene_collision_detected),
                "collision_frame_indices": sorted(all_collision_frames),
                "collision_human_indices": sorted(collision_human_indices),
                "collision_object_labels": sorted(collision_object_labels),
                "collision_object_types": sorted(collision_object_types),
                "out_of_room_frame_indices": sorted(out_of_room_frames),
                "min_clearance_to_obstacle_m": min_clearance_to_obstacle_m,
                "min_clearance_to_wall_m": min_clearance_to_wall_m,
                "human_collision_radius_m": float(radius_m),
                "human_collision_sample_stride": int(sample_stride),
                "rejected_by_wall_clearance": any(
                    bool(human_audit.get("rejected_by_wall_clearance")) for human_audit in human_audits
                ),
                "rejected_by_obstacle_clearance": any(
                    bool(human_audit.get("rejected_by_obstacle_clearance")) for human_audit in human_audits
                ),
                "min_required_wall_clearance_m": float(min_required_wall_clearance_m),
                "min_required_obstacle_clearance_m": float(min_required_obstacle_clearance_m),
                "clearance_rejection_frame_indices": sorted(
                    {
                        frame_index
                        for human_audit in human_audits
                        for frame_index in human_audit.get("clearance_rejection_frame_indices", [])
                    }
                ),
                "human_scene_collision_audit_errors": audit_errors,
            }
        )
        scene["validation_summary"] = validation_summary
        scene["human_scene_collision_detected"] = bool(scene_collision_detected)
        scene["human_scene_collision_audit"] = {
            "per_human": human_audits,
            "collision_frame_indices": sorted(all_collision_frames),
            "collision_human_indices": sorted(collision_human_indices),
            "collision_object_labels": sorted(collision_object_labels),
            "collision_object_types": sorted(collision_object_types),
            "out_of_room_frame_indices": sorted(out_of_room_frames),
            "min_clearance_to_obstacle_m": min_clearance_to_obstacle_m,
            "min_clearance_to_wall_m": min_clearance_to_wall_m,
            "human_collision_radius_m": float(radius_m),
            "human_collision_sample_stride": int(sample_stride),
            "rejected_by_wall_clearance": any(
                bool(human_audit.get("rejected_by_wall_clearance")) for human_audit in human_audits
            ),
            "rejected_by_obstacle_clearance": any(
                bool(human_audit.get("rejected_by_obstacle_clearance")) for human_audit in human_audits
            ),
            "min_required_wall_clearance_m": float(min_required_wall_clearance_m),
            "min_required_obstacle_clearance_m": float(min_required_obstacle_clearance_m),
            "clearance_rejection_frame_indices": sorted(
                {
                    frame_index
                    for human_audit in human_audits
                    for frame_index in human_audit.get("clearance_rejection_frame_indices", [])
                }
            ),
            "audit_errors": audit_errors,
        }
        scene["clip_valid_for_image_benchmark"] = not bool(scene_collision_detected)
        scene["invalid_reason"] = "human_scene_collision" if scene_collision_detected else None

        if scene_collision_detected:
            room_collision_counts[str(scene.get("room"))] += 1
            human_count_collision_counts[int(scene.get("human_count", len(scene.get("humans", []))))] += 1
            obstacle_type_counts.update(sorted(collision_object_types))

        scene_rows.append(
            {
                "miniscene_id": scene.get("miniscene_id"),
                "room": scene.get("room"),
                "human_count": int(scene.get("human_count", len(scene.get("humans", [])))),
                "human_scene_collision_detected": bool(scene_collision_detected),
                "collision_frame_indices": sorted(all_collision_frames),
                "collision_human_indices": sorted(collision_human_indices),
                "collision_object_labels": sorted(collision_object_labels),
                "collision_object_types": sorted(collision_object_types),
                "out_of_room_frame_indices": sorted(out_of_room_frames),
                "min_clearance_to_obstacle_m": min_clearance_to_obstacle_m,
                "min_clearance_to_wall_m": min_clearance_to_wall_m,
                "rejected_by_wall_clearance": any(
                    bool(human_audit.get("rejected_by_wall_clearance")) for human_audit in human_audits
                ),
                "rejected_by_obstacle_clearance": any(
                    bool(human_audit.get("rejected_by_obstacle_clearance")) for human_audit in human_audits
                ),
                "clearance_rejection_frame_indices": sorted(
                    {
                        frame_index
                        for human_audit in human_audits
                        for frame_index in human_audit.get("clearance_rejection_frame_indices", [])
                    }
                ),
                "audit_errors": audit_errors,
            }
        )
        if (not reject_human_scene_collisions) or (not scene_collision_detected):
            audited_miniscenes.append(scene)

    audited_manifest = dict(manifest)
    audited_manifest["miniscenes"] = audited_miniscenes
    audited_manifest["total_selected_miniscenes"] = len(audited_miniscenes)
    audited_manifest["selected_single_human_count"] = sum(
        1 for scene in audited_miniscenes if int(scene.get("human_count", len(scene.get("humans", [])) or 1)) == 1
    )
    audited_manifest["selected_multi_human_count"] = sum(
        1 for scene in audited_miniscenes if int(scene.get("human_count", len(scene.get("humans", [])) or 0)) >= 2
    )
    selected_human_count_distribution = Counter(
        int(scene.get("human_count", len(scene.get("humans", [])) or 0))
        for scene in audited_miniscenes
        if int(scene.get("human_count", len(scene.get("humans", [])) or 0)) >= 2
    )
    audited_manifest["selected_human_count_distribution"] = dict(sorted(selected_human_count_distribution.items()))
    audited_manifest["selected_two_human_count"] = selected_human_count_distribution.get(2, 0)
    audited_manifest["selected_three_human_count"] = selected_human_count_distribution.get(3, 0)
    audited_manifest["selected_four_human_count"] = selected_human_count_distribution.get(4, 0)
    motion_usage_counts = Counter()
    identity_usage_counts = Counter()
    duplicate_motion_count = 0
    duplicate_identity_count = 0
    for scene in audited_miniscenes:
        if int(scene.get("human_count", len(scene.get("humans", [])) or 0)) >= 2:
            motion_ids = [_normalize_motion_id(human.get("motion_id")) for human in scene.get("humans", [])]
            identity_ids = [_motion_identity(human.get("motion_id")) for human in scene.get("humans", [])]
            if len(set(motion_ids)) != len(motion_ids):
                duplicate_motion_count += 1
            if len(set(identity_ids)) != len(identity_ids):
                duplicate_identity_count += 1
        for human in scene.get("humans", []):
            motion_usage_counts[_normalize_motion_id(human.get("motion_id"))] += 1
            identity_usage_counts[_motion_identity(human.get("motion_id"))] += 1
    audited_manifest["selected_motion_usage_counts"] = dict(sorted(motion_usage_counts.items()))
    audited_manifest["selected_identity_usage_counts"] = dict(sorted(identity_usage_counts.items()))
    audited_manifest["selected_multi_human_duplicate_motion_count"] = int(duplicate_motion_count)
    audited_manifest["selected_multi_human_duplicate_identity_count"] = int(duplicate_identity_count)
    selected_per_room = dict(audited_manifest.get("selected_per_room") or {})
    for room_name, row in selected_per_room.items():
        matching = [scene for scene in audited_miniscenes if scene.get("room") == room_name]
        row = dict(row)
        row["post_collision_filter_total"] = len(matching)
        row["post_collision_filter_human_count_distribution"] = dict(
            sorted(Counter(int(scene.get("human_count", len(scene.get("humans", [])))) for scene in matching).items())
        )
        selected_per_room[room_name] = row
    audited_manifest["selected_per_room"] = selected_per_room
    audited_manifest["rooms_without_renderable_miniscenes"] = sorted(
        room_name
        for room_name, row in selected_per_room.items()
        if int(row.get("post_collision_filter_total", 0)) == 0
    )
    audited_manifest["post_collision_filter_total_selected_miniscenes"] = len(audited_miniscenes)
    audited_manifest["human_scene_collision_reject_enabled"] = bool(reject_human_scene_collisions)
    audited_manifest["human_scene_collision_radius_m"] = float(radius_m)
    audited_manifest["human_scene_collision_sample_stride"] = int(sample_stride)
    audited_manifest["min_required_wall_clearance_m"] = float(min_required_wall_clearance_m)
    audited_manifest["min_required_obstacle_clearance_m"] = float(min_required_obstacle_clearance_m)
    audited_manifest["human_scene_collision_rejected_count"] = int(
        len(manifest.get("miniscenes", [])) - len(audited_miniscenes)
    )

    report = {
        "scene_path": manifest.get("scene_path"),
        "motion_payload_report": motion_payload_report,
        "human_collision_radius_m": float(radius_m),
        "human_collision_sample_stride": int(sample_stride),
        "min_required_wall_clearance_m": float(min_required_wall_clearance_m),
        "min_required_obstacle_clearance_m": float(min_required_obstacle_clearance_m),
        "reject_human_scene_collisions": bool(reject_human_scene_collisions),
        "total_miniscenes": len(manifest.get("miniscenes", [])),
        "kept_miniscenes": len(audited_miniscenes),
        "rejected_miniscenes": int(len(manifest.get("miniscenes", [])) - len(audited_miniscenes)),
        "colliding_miniscene_count": int(sum(1 for row in scene_rows if row["human_scene_collision_detected"])),
        "rejected_by_wall_clearance_count": int(wall_clearance_rejection_count),
        "rejected_by_obstacle_clearance_count": int(obstacle_clearance_rejection_count),
        "room_collision_counts": dict(sorted(room_collision_counts.items())),
        "obstacle_type_collision_counts": dict(sorted(obstacle_type_counts.items())),
        "human_count_collision_counts": dict(sorted(human_count_collision_counts.items())),
        "scene_rows": scene_rows,
    }
    return audited_manifest, report


def _audit_scenes_root_human_scene_collisions(
    scenes_root,
    radius_m,
    sample_stride=1,
    reject_human_scene_collisions=True,
    write_audited_manifests=False,
    npz_root=DEFAULT_BEDLAM_NPZ_ROOT,
    npz_trans_ground_axes=DEFAULT_NPZ_TRANS_GROUND_AXES,
    min_required_wall_clearance_m=MIN_HUMAN_WALL_CLEARANCE_M,
    min_required_obstacle_clearance_m=MIN_HUMAN_OBSTACLE_CLEARANCE_M,
):
    scenes_root = Path(scenes_root)
    scene_reports = []
    room_collision_counts = Counter()
    obstacle_type_collision_counts = Counter()
    human_count_collision_counts = Counter()
    for scene_root in sorted(path for path in scenes_root.iterdir() if path.is_dir()):
        metadata_path = scene_root / "scene_collision_metadata.json"
        manifest_path = scene_root / "miniscene_selection_v0" / "bedlam360_infinigen_miniscenes_starterpack_only.json"
        if not metadata_path.exists() or not manifest_path.exists():
            continue
        metadata = _load_metadata(metadata_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audited_manifest, report = _audit_manifest_human_scene_collisions(
            metadata,
            manifest,
            output_dir=manifest_path.parent,
            radius_m=radius_m,
            sample_stride=sample_stride,
            reject_human_scene_collisions=reject_human_scene_collisions,
            npz_root=npz_root,
            npz_trans_ground_axes=npz_trans_ground_axes,
            min_required_wall_clearance_m=min_required_wall_clearance_m,
            min_required_obstacle_clearance_m=min_required_obstacle_clearance_m,
        )
        report["scene_root"] = str(scene_root)
        report["metadata_path"] = str(metadata_path)
        report["manifest_path"] = str(manifest_path)
        report["scene_name"] = scene_root.name
        scene_reports.append(report)
        room_collision_counts.update(report.get("room_collision_counts", {}))
        obstacle_type_collision_counts.update(report.get("obstacle_type_collision_counts", {}))
        human_count_collision_counts.update(report.get("human_count_collision_counts", {}))
        _write_json(manifest_path.parent / "human_scene_collision_audit_report.json", report)
        if write_audited_manifests:
            _write_json(manifest_path, audited_manifest)
    aggregate = {
        "scenes_root": str(scenes_root),
        "human_collision_radius_m": float(radius_m),
        "human_collision_sample_stride": int(sample_stride),
        "min_required_wall_clearance_m": float(min_required_wall_clearance_m),
        "min_required_obstacle_clearance_m": float(min_required_obstacle_clearance_m),
        "reject_human_scene_collisions": bool(reject_human_scene_collisions),
        "write_audited_manifests": bool(write_audited_manifests),
        "scene_count": len(scene_reports),
        "colliding_miniscene_count": int(sum(report.get("colliding_miniscene_count", 0) for report in scene_reports)),
        "room_collision_counts": dict(sorted(room_collision_counts.items())),
        "obstacle_type_collision_counts": dict(sorted(obstacle_type_collision_counts.items())),
        "human_count_collision_counts": dict(sorted(human_count_collision_counts.items())),
        "scene_reports": scene_reports,
    }
    return aggregate


def _single_candidate_score(candidate, selected, spawn_lookup):
    row = candidate["row"]
    spawn_pose = spawn_lookup[row["spawn_index"]]
    used_spawns = {scene["humans"][0]["spawn_pose_index"] for scene in selected}
    used_motions = {scene["humans"][0]["motion_id"] for scene in selected}
    used_activities = {scene["humans"][0]["activity_hint"] for scene in selected}
    used_sources = {scene["humans"][0]["source"] for scene in selected}
    score = 0.0
    if row["spawn_index"] not in used_spawns:
        score += 100.0
    if row["motion_id"] not in used_motions:
        score += 30.0
    if spawn_pose.get("activity_hint") not in used_activities:
        score += 10.0
    if spawn_pose.get("source") not in used_sources:
        score += 5.0
    score += 0.01 * float(row.get("trajectory_stats", {}).get("frame_count", 0))
    return score


def _candidate_tiebreak_value(candidate_key, selection_seed):
    rng = random.Random(f"{int(selection_seed)}::{candidate_key}")
    return rng.random()


def _group_candidate_score(candidate, selected, spawn_lookup, selection_seed=0):
    group = candidate["group"]
    humans = group["humans"]
    human_count = int(group.get("human_count", len(humans)))
    used_spawns = Counter(
        human["spawn_pose_index"]
        for scene in selected
        for human in scene["humans"]
    )
    used_motions = Counter(
        _normalize_motion_id(human["motion_id"])
        for scene in selected
        for human in scene["humans"]
    )
    used_identities = Counter(
        _motion_identity(human["motion_id"])
        for scene in selected
        for human in scene["humans"]
    )
    used_activities = Counter(
        human["activity_hint"]
        for scene in selected
        for human in scene["humans"]
        if human.get("activity_hint") is not None
    )
    room_human_count_counts = Counter(
        (scene["room"], len(scene["humans"]))
        for scene in selected
    )

    score = 0.0
    group_motion_ids = [_normalize_motion_id(human["motion_id"]) for human in humans]
    group_identity_ids = [_motion_identity(human["motion_id"]) for human in humans]
    group_spawn_indices = [
        int(human.get("spawn_pose_index", human.get("spawn_index")))
        for human in humans
    ]
    group_activity_hints = []
    for human in humans:
        spawn_pose_index = int(
            human.get("spawn_pose_index", human.get("spawn_index"))
        )
        motion_id = _normalize_motion_id(human["motion_id"])
        identity_id = _motion_identity(human["motion_id"])
        metadata_spawn_pose = spawn_lookup[spawn_pose_index]
        activity_hint = metadata_spawn_pose.get("activity_hint")
        group_activity_hints.append(activity_hint)
        score += 100.0 if used_spawns[spawn_pose_index] == 0 else -20.0 * used_spawns[spawn_pose_index]
        score += 80.0 if used_motions[motion_id] == 0 else -25.0 * used_motions[motion_id]
        score += 45.0 if used_identities[identity_id] == 0 else -12.0 * used_identities[identity_id]
        if activity_hint is not None:
            score += 10.0 if used_activities[activity_hint] == 0 else -2.0 * used_activities[activity_hint]

    if len(set(group_motion_ids)) == len(group_motion_ids):
        score += 140.0
    else:
        score -= 160.0
    if len(set(group_identity_ids)) == len(group_identity_ids):
        score += 80.0
    else:
        score -= 60.0
    if len(set(group_spawn_indices)) == len(group_spawn_indices):
        score += 50.0

    activity_values = [value for value in group_activity_hints if value is not None]
    if len(set(activity_values)) == len(activity_values) and activity_values:
        score += 20.0

    score += 35.0 * max(0, human_count - 1)
    score -= 40.0 * room_human_count_counts[(group["room"], human_count)]
    score += float(group.get("min_interhuman_distance_m") or 0.0)
    score += _candidate_tiebreak_value(group["group_id"], selection_seed)
    return score


def _group_candidate_score_with_spatial(
    metadata,
    candidate,
    selected,
    spawn_lookup,
    selection_seed=0,
    distance_bins_m=DEFAULT_CAMERA_DISTANCE_BINS_M,
    near_seam_threshold_deg=20.0,
    azimuth_bin_count=DEFAULT_AZIMUTH_BIN_COUNT,
    spatial_diversity_weight=50.0,
    motion_diversity_weight=1.0,
    distance_diversity_weight=4.0,
    azimuth_diversity_weight=2.0,
    seam_diversity_weight=2.0,
    scale_diversity_weight=2.0,
    multi_depth_weight=2.0,
    spatial_selection_mode="legacy",
):
    base_score = _group_candidate_score(candidate, selected, spawn_lookup, selection_seed)
    group = candidate["group"]
    summary = _scene_spatial_camera_summary(
        metadata,
        group["room"],
        [
            {
                "motion_id": human["motion_id"],
                "position_xyz_m": (human.get("spawn_pose") or {}).get("position_xyz"),
            }
            for human in group.get("humans", [])
        ],
        distance_bins_m=distance_bins_m,
        near_seam_threshold_deg=near_seam_threshold_deg,
        azimuth_bin_count=azimuth_bin_count,
    )
    if str(spatial_selection_mode) != "erp":
        spatial_bonus = (
            4.0 * len(set(summary.get("distance_bin_coverage", [])) - {
                bin_name
                for scene in selected
                for bin_name in (scene.get("spatial_camera") or {}).get("distance_bin_coverage", [])
            })
            + 2.0 * len(set(summary.get("azimuth_bin_coverage", [])) - {
                bin_name
                for scene in selected
                for bin_name in (scene.get("spatial_camera") or {}).get("azimuth_bin_coverage", [])
            })
            + 2.0 * float(summary.get("has_near_seam_human", False))
            + 2.0 * float(summary.get("has_close_human", False))
            + 1.5 * float(summary.get("has_far_human", False))
            + 2.0 * float(summary.get("has_multi_depth_humans", False))
            + 1.5 * float(summary.get("multi_human_occlusion_risk_score", 0.0))
        )
        return (float(base_score) * float(motion_diversity_weight)) + (
            float(spatial_bonus + summary.get("erp_difficulty_score", 0.0)) * float(spatial_diversity_weight)
        )
    used_distance_bins = {
        bin_name
        for scene in selected
        for bin_name in (scene.get("spatial_camera") or {}).get("distance_bin_coverage", [])
    }
    used_azimuth_bins = {
        bin_name
        for scene in selected
        for bin_name in (scene.get("spatial_camera") or {}).get("azimuth_bin_coverage", [])
    }
    used_scale_bins = {
        bin_name
        for scene in selected
        for bin_name in (scene.get("spatial_camera") or {}).get("scale_bin_coverage", [])
    }
    novel_distance_bins = len(set(summary.get("distance_bin_coverage", [])) - used_distance_bins)
    novel_azimuth_bins = len(set(summary.get("azimuth_bin_coverage", [])) - used_azimuth_bins)
    novel_scale_bins = len(set(summary.get("scale_bin_coverage", [])) - used_scale_bins)
    spatial_bonus = (
        float(distance_diversity_weight) * novel_distance_bins
        + float(azimuth_diversity_weight) * novel_azimuth_bins
        + float(seam_diversity_weight) * float(summary.get("near_seam_human_count", 0))
        + float(seam_diversity_weight) * float(summary.get("seam_crossing_risk_human_count", 0))
        + float(scale_diversity_weight) * novel_scale_bins
        + float(multi_depth_weight) * float(summary.get("has_multi_depth_humans", False))
        + 1.5 * float(summary.get("multi_human_occlusion_risk_score", 0.0))
        + 0.5 * float(summary.get("has_close_human", False))
        + 0.5 * float(summary.get("has_far_human", False))
    )
    return (float(base_score) * float(motion_diversity_weight)) + (
        float(spatial_bonus + summary.get("erp_difficulty_score", 0.0)) * float(spatial_diversity_weight)
    )


def _build_single_miniscene(room_name, ordinal, row, spawn_lookup, frame_start, frame_end):
    metadata_spawn_pose = spawn_lookup[row["spawn_index"]]
    human = _human_entry_from_spawn_pose(
        row["spawn_index"], row["motion_id"], metadata_spawn_pose
    )
    return {
        "miniscene_id": f"{room_name}__single__{ordinal:03d}",
        "room": room_name,
        "scene_type": "single_human",
        "humans": [human],
        "validation_summary": {
            "scene_collision_pass": True,
            "interhuman_mode": None,
            "single_pair_reasons": list(row.get("reasons", [])),
        },
        "diversity_tags": [
            "single_human",
            human["activity_hint"],
            human["source"],
        ],
        "render_options": {
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "use_natural_timing": True,
        },
    }


def _build_group_miniscene(
    room_name, ordinal, group, spawn_lookup, frame_start, frame_end
):
    humans = []
    activity_hints = []
    sources = []
    for group_human in group["humans"]:
        metadata_spawn_pose = spawn_lookup[group_human["spawn_index"]]
        human = _human_entry_from_spawn_pose(
            group_human["spawn_index"], group_human["motion_id"], metadata_spawn_pose
        )
        humans.append(human)
        activity_hints.append(human["activity_hint"])
        sources.append(human["source"])
    human_count = len(humans)
    duplicate_motion_ids_present = len({human["motion_id"] for human in humans}) != len(humans)
    duplicate_identity_ids_present = len({_motion_identity(human["motion_id"]) for human in humans}) != len(humans)
    return {
        "miniscene_id": f"{room_name}__{human_count}_human__{ordinal:03d}",
        "room": room_name,
        "scene_type": "multi_human" if human_count > 2 else "two_human",
        "human_count": human_count,
        "duplicate_motion_ids_present": bool(duplicate_motion_ids_present),
        "duplicate_identity_ids_present": bool(duplicate_identity_ids_present),
        "humans": humans,
        "validation_summary": {
            "scene_collision_pass": True,
            "interhuman_mode": group.get("interhuman_mode"),
            "min_interhuman_distance_m": group.get("min_interhuman_distance_m"),
            "min_interhuman_distance_frame_index": group.get(
                "min_interhuman_distance_frame_index"
            ),
            "interhuman_overlap_area_m2": group.get("interhuman_overlap_area_m2"),
        },
        "diversity_tags": [
            f"{human_count}_human",
            *sorted(set(activity_hints)),
            *sorted(set(sources)),
        ],
        "render_options": {
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "use_natural_timing": True,
        },
    }


def _select_diverse_single_miniscenes(
    room_name, rows, spawn_lookup, max_count, frame_start, frame_end
):
    candidates = [{"row": row} for row in rows]
    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < int(max_count):
        best = max(
            remaining,
            key=lambda candidate: (
                _single_candidate_score(candidate, selected, spawn_lookup),
                -candidate["row"]["spawn_index"],
                candidate["row"]["motion_id"],
            ),
        )
        ordinal = len(selected)
        selected.append(
            _build_single_miniscene(
                room_name,
                ordinal,
                best["row"],
                spawn_lookup,
                frame_start,
                frame_end,
            )
        )
        remaining.remove(best)
    return selected


def _select_diverse_group_miniscenes(
    metadata,
    room_name,
    groups,
    spawn_lookup,
    max_count,
    frame_start,
    frame_end,
    selection_seed=0,
    allow_duplicate_motion_in_group=False,
    enable_spatial_camera_selection=False,
    distance_bins_m=DEFAULT_CAMERA_DISTANCE_BINS_M,
    near_seam_threshold_deg=20.0,
    azimuth_bin_count=DEFAULT_AZIMUTH_BIN_COUNT,
    spatial_diversity_weight=50.0,
    motion_diversity_weight=1.0,
    distance_diversity_weight=4.0,
    azimuth_diversity_weight=2.0,
    seam_diversity_weight=2.0,
    scale_diversity_weight=2.0,
    multi_depth_weight=2.0,
    spatial_selection_mode="legacy",
    return_candidate_pool=False,
):
    candidates = []
    for group in groups:
        motion_ids = [human["motion_id"] for human in group.get("humans", [])]
        has_duplicate_motion_ids = len(set(motion_ids)) != len(motion_ids)
        if has_duplicate_motion_ids and not allow_duplicate_motion_in_group:
            continue
        candidates.append({"group": group})
    candidate_pool = [
        _build_group_scene_candidate(
            metadata,
            room_name,
            candidate["group"],
            spawn_lookup,
            ordinal=index,
            frame_start=frame_start,
            frame_end=frame_end,
            distance_bins_m=distance_bins_m,
            near_seam_threshold_deg=near_seam_threshold_deg,
            azimuth_bin_count=azimuth_bin_count,
        )
        for index, candidate in enumerate(candidates)
    ] if return_candidate_pool else None
    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < int(max_count):
        if enable_spatial_camera_selection:
            best = max(
                remaining,
                key=lambda candidate: (
                    _group_candidate_score_with_spatial(
                        metadata,
                        candidate,
                        selected,
                        spawn_lookup,
                        selection_seed=selection_seed,
                        distance_bins_m=distance_bins_m,
                        near_seam_threshold_deg=near_seam_threshold_deg,
                        azimuth_bin_count=azimuth_bin_count,
                        spatial_diversity_weight=spatial_diversity_weight,
                        motion_diversity_weight=motion_diversity_weight,
                        distance_diversity_weight=distance_diversity_weight,
                        azimuth_diversity_weight=azimuth_diversity_weight,
                        seam_diversity_weight=seam_diversity_weight,
                        scale_diversity_weight=scale_diversity_weight,
                        multi_depth_weight=multi_depth_weight,
                        spatial_selection_mode=spatial_selection_mode,
                    ),
                    int(candidate["group"].get("human_count", len(candidate["group"].get("humans", [])))),
                    float(candidate["group"].get("min_interhuman_distance_m") or 0.0),
                    _candidate_tiebreak_value(candidate["group"]["group_id"], selection_seed),
                    candidate["group"]["group_id"],
                ),
            )
        else:
            best = max(
                remaining,
                key=lambda candidate: (
                    _group_candidate_score(candidate, selected, spawn_lookup, selection_seed),
                    int(candidate["group"].get("human_count", len(candidate["group"].get("humans", [])))),
                    float(candidate["group"].get("min_interhuman_distance_m") or 0.0),
                    _candidate_tiebreak_value(candidate["group"]["group_id"], selection_seed),
                    candidate["group"]["group_id"],
                ),
            )
        ordinal = len(selected)
        scene = _build_group_miniscene(
            room_name,
            ordinal,
            best["group"],
            spawn_lookup,
            frame_start,
            frame_end,
        )
        scene["spatial_camera"] = _scene_spatial_camera_summary(
            metadata,
            room_name,
            scene.get("humans", []),
            distance_bins_m=distance_bins_m,
            near_seam_threshold_deg=near_seam_threshold_deg,
            azimuth_bin_count=azimuth_bin_count,
        )
        selected.append(scene)
        remaining.remove(best)
    if return_candidate_pool:
        return selected, candidate_pool
    return selected


def _build_miniscene_manifest(
    metadata,
    scene_path,
    valid_single_by_room,
    valid_two_by_room,
    max_single_per_room,
    max_two_human_per_room,
    frame_start,
    frame_end,
    min_human_count=2,
    max_human_count=4,
    selection_seed=0,
    allow_duplicate_motion_in_group=False,
    motion_set_report=None,
    enable_spatial_camera_selection=False,
    distance_bins_m=DEFAULT_CAMERA_DISTANCE_BINS_M,
    near_seam_threshold_deg=20.0,
    azimuth_bin_count=DEFAULT_AZIMUTH_BIN_COUNT,
    spatial_diversity_weight=50.0,
    motion_diversity_weight=1.0,
    distance_diversity_weight=4.0,
    azimuth_diversity_weight=2.0,
    seam_diversity_weight=2.0,
    scale_diversity_weight=2.0,
    multi_depth_weight=2.0,
    spatial_selection_mode="legacy",
    enable_spatial_dedup=False,
    spatial_dedup_xy_threshold_m=0.4,
    spatial_dedup_report_only=False,
):
    spawn_lookup = _spawn_pose_lookup(metadata)
    room_names = sorted(room.get("name") for room in metadata.get("rooms", []))
    miniscenes = []
    selected_per_room = {}
    single_count = 0
    multi_count = 0
    selected_human_count_distribution = Counter()
    multi_candidate_pool = []
    min_human_count = int(min_human_count)
    max_human_count = max(int(max_human_count), min_human_count)
    for room_name in room_names:
        room_singles = []
        if min_human_count <= 1 and int(max_single_per_room) > 0:
            room_singles = _select_diverse_single_miniscenes(
                room_name,
                valid_single_by_room.get(room_name, []),
                spawn_lookup,
                max_single_per_room,
                frame_start,
                frame_end,
            )
        room_groups = [
            group
            for group in valid_two_by_room.get(room_name, [])
            if min_human_count <= int(group.get("human_count", len(group.get("humans", [])))) <= max_human_count
        ]
        room_multis_result = _select_diverse_group_miniscenes(
            metadata,
            room_name,
            room_groups,
            spawn_lookup,
            max_two_human_per_room,
            frame_start,
            frame_end,
            selection_seed=selection_seed,
            allow_duplicate_motion_in_group=allow_duplicate_motion_in_group,
            enable_spatial_camera_selection=enable_spatial_camera_selection,
            distance_bins_m=distance_bins_m,
            near_seam_threshold_deg=near_seam_threshold_deg,
            azimuth_bin_count=azimuth_bin_count,
            spatial_diversity_weight=spatial_diversity_weight,
            motion_diversity_weight=motion_diversity_weight,
            distance_diversity_weight=distance_diversity_weight,
            azimuth_diversity_weight=azimuth_diversity_weight,
            seam_diversity_weight=seam_diversity_weight,
            scale_diversity_weight=scale_diversity_weight,
            multi_depth_weight=multi_depth_weight,
            spatial_selection_mode=spatial_selection_mode,
            return_candidate_pool=bool(enable_spatial_dedup),
        )
        if enable_spatial_dedup:
            room_multis, room_candidate_pool = room_multis_result
            multi_candidate_pool.extend(room_candidate_pool)
        else:
            room_multis = room_multis_result
        selected_per_room[room_name] = {
            "single_human": len(room_singles),
            "multi_human": len(room_multis),
            "human_count_distribution": dict(
                sorted(
                    Counter(int(scene.get("human_count", len(scene.get("humans", [])))) for scene in room_multis).items()
                )
            ),
            "total": len(room_singles) + len(room_multis),
        }
        miniscenes.extend(room_singles)
        miniscenes.extend(room_multis)
        single_count += len(room_singles)
        multi_count += len(room_multis)
        for scene in room_multis:
            selected_human_count_distribution[int(scene.get("human_count", len(scene.get("humans", []))))] += 1
    spatial_dedup_report = None
    if enable_spatial_dedup:
        deduped_multi_scenes, spatial_dedup_report = _post_select_spatial_dedup(
            [scene for scene in miniscenes if int(scene.get("human_count", len(scene.get("humans", [])))) >= 2],
            multi_candidate_pool,
            enable_spatial_dedup=enable_spatial_dedup,
            spatial_dedup_xy_threshold_m=spatial_dedup_xy_threshold_m,
            spatial_dedup_report_only=spatial_dedup_report_only,
        )
        kept_multi_ids = {scene.get("miniscene_id") for scene in deduped_multi_scenes}
        miniscenes = [
            scene
            for scene in miniscenes
            if int(scene.get("human_count", len(scene.get("humans", [])))) < 2
            or scene.get("miniscene_id") in kept_multi_ids
        ]
        new_multi_ids = {
            scene.get("miniscene_id")
            for scene in deduped_multi_scenes
            if scene.get("miniscene_id") not in {existing.get("miniscene_id") for existing in miniscenes}
        }
        miniscenes.extend(
            scene for scene in deduped_multi_scenes
            if scene.get("miniscene_id") in new_multi_ids
        )
        selected_per_room = {}
        single_count = 0
        multi_count = 0
        selected_human_count_distribution = Counter()
        room_scene_rows = defaultdict(list)
        for scene in miniscenes:
            room_scene_rows[scene.get("room")].append(scene)
        for room_name in room_names:
            room_scenes = room_scene_rows.get(room_name, [])
            room_singles = [
                scene for scene in room_scenes
                if int(scene.get("human_count", len(scene.get("humans", [])))) <= 1
            ]
            room_multis = [
                scene for scene in room_scenes
                if int(scene.get("human_count", len(scene.get("humans", [])))) >= 2
            ]
            selected_per_room[room_name] = {
                "single_human": len(room_singles),
                "multi_human": len(room_multis),
                "human_count_distribution": dict(
                    sorted(
                        Counter(int(scene.get("human_count", len(scene.get("humans", [])))) for scene in room_multis).items()
                    )
                ),
                "total": len(room_singles) + len(room_multis),
            }
            single_count += len(room_singles)
            multi_count += len(room_multis)
            for scene in room_multis:
                selected_human_count_distribution[int(scene.get("human_count", len(scene.get("humans", []))))] += 1
    duplicate_multi_motion_count = sum(
        1
        for scene in miniscenes
        if int(scene.get("human_count", len(scene.get("humans", [])))) >= 2
        and bool(scene.get("duplicate_motion_ids_present"))
    )
    duplicate_multi_identity_count = sum(
        1
        for scene in miniscenes
        if int(scene.get("human_count", len(scene.get("humans", [])))) >= 2
        and bool(scene.get("duplicate_identity_ids_present"))
    )
    selected_motion_usage_counts = Counter()
    selected_identity_usage_counts = Counter()
    for scene in miniscenes:
        for human in scene.get("humans", []):
            selected_motion_usage_counts[_normalize_motion_id(human.get("motion_id"))] += 1
            selected_identity_usage_counts[_motion_identity(human.get("motion_id"))] += 1
    rooms_without_renderable_miniscenes = sorted(
        room_name
        for room_name, row in selected_per_room.items()
        if int(row["total"]) == 0
    )
    return {
        "scene_path": str(scene_path),
        "motion_set": motion_set_report,
        "enable_spatial_camera_selection": bool(enable_spatial_camera_selection),
        "spatial_selection_mode": str(spatial_selection_mode),
        "enable_spatial_dedup": bool(enable_spatial_dedup),
        "spatial_dedup_xy_threshold_m": float(spatial_dedup_xy_threshold_m),
        "spatial_dedup_report_only": bool(spatial_dedup_report_only),
        "max_single_per_room": int(max_single_per_room),
        "max_two_human_per_room": int(max_two_human_per_room),
        "min_human_count": min_human_count,
        "max_human_count": max_human_count,
        "selection_seed": int(selection_seed),
        "allow_duplicate_motion_in_group": bool(allow_duplicate_motion_in_group),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "use_natural_timing": True,
        "selected_per_room": selected_per_room,
        "rooms_without_renderable_miniscenes": rooms_without_renderable_miniscenes,
        "selected_single_human_count": single_count,
        "selected_multi_human_count": multi_count,
        "selected_two_human_count": selected_human_count_distribution.get(2, 0),
        "selected_three_human_count": selected_human_count_distribution.get(3, 0),
        "selected_four_human_count": selected_human_count_distribution.get(4, 0),
        "selected_human_count_distribution": dict(sorted(selected_human_count_distribution.items())),
        "selected_multi_human_duplicate_motion_count": duplicate_multi_motion_count,
        "selected_multi_human_duplicate_identity_count": duplicate_multi_identity_count,
        "selected_motion_usage_counts": dict(sorted(selected_motion_usage_counts.items())),
        "selected_identity_usage_counts": dict(sorted(selected_identity_usage_counts.items())),
        "total_selected_miniscenes": len(miniscenes),
        "spatial_dedup_report": spatial_dedup_report,
        "miniscenes": miniscenes,
    }


def _print_manifest_summary(manifest):
    print("")
    print("Mini-scene manifest summary")
    print(f"Total selected mini-scenes: {manifest['total_selected_miniscenes']}")
    print(f"Selected single-human count: {manifest['selected_single_human_count']}")
    print(f"Selected multi-human count: {manifest.get('selected_multi_human_count', 0)}")
    print(f"Selected two-human count: {manifest.get('selected_two_human_count', 0)}")
    print(f"Selected three-human count: {manifest.get('selected_three_human_count', 0)}")
    print(f"Selected four-human count: {manifest.get('selected_four_human_count', 0)}")
    print(
        "Selected multi-human scenes with duplicate motion IDs: "
        f"{manifest.get('selected_multi_human_duplicate_motion_count', 0)}"
    )
    print(
        "Selected multi-human scenes with duplicate identity IDs: "
        f"{manifest.get('selected_multi_human_duplicate_identity_count', 0)}"
    )
    motion_set = manifest.get("motion_set") or {}
    print(
        f"Motion set mode: {motion_set.get('motion_set_mode')} "
        f"motion_id_count={motion_set.get('motion_id_count')}"
    )
    print("room | selected_single_human | selected_multi_human | total")
    for room_name in sorted(manifest.get("selected_per_room", {})):
        row = manifest["selected_per_room"][room_name]
        print(
            f"{room_name} | {row['single_human']} | {row['multi_human']} | {row['total']}"
        )
    rooms_without = manifest.get("rooms_without_renderable_miniscenes", [])
    if rooms_without:
        print("Rooms without renderable mini-scenes:")
        for room_name in rooms_without:
            print(f"  {room_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
    )
    parser.add_argument("--motion-root", type=Path, action="append", default=None)
    parser.add_argument("--motion-root-dir", type=Path, default=None)
    parser.add_argument("--npz-root", type=Path, default=DEFAULT_BEDLAM_NPZ_ROOT)
    parser.add_argument(
        "--npz-trans-ground-axes",
        choices=("xy", "xz"),
        default=DEFAULT_NPZ_TRANS_GROUND_AXES,
    )
    parser.add_argument("--valid-single-json", type=Path, default=None)
    parser.add_argument("--valid-two-human-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--interhuman-mode",
        choices=("swept", "synchronized"),
        default="synchronized",
    )
    parser.add_argument("--interhuman-margin-m", type=float, default=0.05)
    parser.add_argument("--near-miss-limit", type=int, default=50)
    parser.add_argument("--max-single-per-room", type=int, default=0)
    parser.add_argument("--max-two-human-per-room", type=int, default=5)
    parser.add_argument("--max-motion-roots-tested", type=int, default=None)
    parser.add_argument("--motion-root-selection-seed", type=int, default=0)
    parser.add_argument("--placement-search-seed", type=int, default=0)
    parser.add_argument(
        "--prefer-identity-diversity",
        action="store_true",
        default=False,
    )
    parser.add_argument("--min-human-count", type=int, default=2)
    parser.add_argument("--max-human-count", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--frame-start", type=int, default=12)
    parser.add_argument("--frame-end", type=int, default=18)
    parser.add_argument("--spawn-yaw-sweep-deg", type=str, default="0")
    parser.add_argument("--extra-free-space-spawn-samples", type=int, default=0)
    parser.add_argument("--free-space-sampling-seed", type=int, default=0)
    parser.add_argument("--min-spawn-wall-clearance-m", type=float, default=MIN_HUMAN_WALL_CLEARANCE_M)
    parser.add_argument(
        "--min-spawn-obstacle-clearance-m",
        type=float,
        default=MIN_HUMAN_OBSTACLE_CLEARANCE_M,
    )
    parser.add_argument("--max-placement-attempts-per-motion", type=int, default=0)
    parser.add_argument("--max-valid-placements-per-motion", type=int, default=0)
    parser.add_argument("--max-valid-placements-per-room", type=int, default=0)
    parser.add_argument("--enable-spatial-camera-audit", action="store_true", default=False)
    parser.add_argument("--enable-spatial-camera-selection", action="store_true", default=False)
    parser.add_argument("--camera-distance-bins", type=str, default="0,1.5,3.0,999")
    parser.add_argument("--near-seam-threshold-deg", type=float, default=20.0)
    parser.add_argument("--spatial-diversity-weight", type=float, default=50.0)
    parser.add_argument("--distance-diversity-weight", type=float, default=4.0)
    parser.add_argument("--azimuth-diversity-weight", type=float, default=2.0)
    parser.add_argument("--seam-diversity-weight", type=float, default=2.0)
    parser.add_argument("--scale-diversity-weight", type=float, default=2.0)
    parser.add_argument("--multi-depth-weight", type=float, default=2.0)
    parser.add_argument(
        "--spatial-selection-mode",
        choices=("legacy", "erp"),
        default="legacy",
    )
    parser.add_argument("--motion-diversity-weight", type=float, default=1.0)
    parser.add_argument("--enable-spatial-dedup", action="store_true", default=False)
    parser.add_argument("--spatial-dedup-xy-threshold-m", type=float, default=0.4)
    parser.add_argument("--spatial-dedup-report-only", action="store_true", default=False)
    parser.add_argument(
        "--motion-set-mode",
        choices=("all", "starterpack_only"),
        default=MOTION_SET_MODE,
    )
    parser.add_argument(
        "--starterpack-whitelist",
        type=Path,
        default=STARTERPACK_WHITELIST_PATH,
    )
    parser.add_argument(
        "--allow-duplicate-motion-in-group",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--reject-human-scene-collisions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--human-collision-radius-m", type=float, default=CAPSULE_RADIUS_M)
    parser.add_argument("--human-collision-sample-stride", type=int, default=1)
    parser.add_argument(
        "--min-human-wall-clearance-m",
        type=float,
        default=MIN_HUMAN_WALL_CLEARANCE_M,
    )
    parser.add_argument(
        "--min-human-obstacle-clearance-m",
        type=float,
        default=MIN_HUMAN_OBSTACLE_CLEARANCE_M,
    )
    parser.add_argument("--audit-scenes-root", type=Path, default=None)
    parser.add_argument("--write-audited-manifests", action="store_true", default=False)
    args = parser.parse_args()

    if args.audit_scenes_root is not None:
        aggregate = _audit_scenes_root_human_scene_collisions(
            args.audit_scenes_root,
            radius_m=args.human_collision_radius_m,
            sample_stride=args.human_collision_sample_stride,
            reject_human_scene_collisions=args.reject_human_scene_collisions,
            write_audited_manifests=args.write_audited_manifests,
            npz_root=args.npz_root,
            npz_trans_ground_axes=args.npz_trans_ground_axes,
            min_required_wall_clearance_m=args.min_human_wall_clearance_m,
            min_required_obstacle_clearance_m=args.min_human_obstacle_clearance_m,
        )
        aggregate_path = Path(args.audit_scenes_root) / "v3_human_scene_collision_audit_report.json"
        _write_json(aggregate_path, aggregate)
        print("scene | colliding_miniscenes | kept | rejected | room_collision_counts")
        for report in aggregate.get("scene_reports", []):
            print(
                f"{report.get('scene_name')} | {report.get('colliding_miniscene_count', 0)} | "
                f"{report.get('kept_miniscenes', 0)} | {report.get('rejected_miniscenes', 0)} | "
                f"{report.get('room_collision_counts', {})}"
            )
        print(f"Audit report: {aggregate_path}")
        return

    if args.metadata is None:
        raise RuntimeError("--metadata is required unless --audit-scenes-root is used")
    if args.output_dir is None:
        raise RuntimeError("--output-dir is required unless --audit-scenes-root is used")

    metadata = _load_metadata(args.metadata)
    planning_start_time = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid_single_path = args.output_dir / "valid_single_pairs_by_room.json"
    all_single_path = args.output_dir / "all_single_pair_results.json"
    valid_two_human_path = args.output_dir / "valid_two_human_groups_by_room.json"
    valid_multi_human_path = args.output_dir / "valid_multi_human_groups_by_room.json"
    failed_two_human_path = args.output_dir / "failed_two_human_near_misses.json"
    motion_set_ids_path = args.output_dir / "renderable_motion_ids.json"
    starterpack_report_path = args.output_dir / "starterpack_manifest_filter_report.json"
    candidate_generation_diagnostics_path = args.output_dir / "candidate_generation_diagnostics.json"
    human_scene_collision_audit_report_path = args.output_dir / "human_scene_collision_audit_report.json"
    spawn_augmentation_report_path = args.output_dir / "spawn_augmentation_report.json"
    spatial_camera_audit_report_path = args.output_dir / "spatial_camera_audit_report.json"
    spatial_dedup_report_path = args.output_dir / "spatial_dedup_report.json"
    if args.motion_set_mode == "starterpack_only":
        manifest_path = args.output_dir / "bedlam360_infinigen_miniscenes_starterpack_only.json"
    else:
        manifest_path = args.output_dir / "bedlam360_infinigen_miniscenes.json"

    metadata, spawn_augmentation_report = _build_augmented_metadata(
        metadata,
        spawn_yaw_sweep_deg=args.spawn_yaw_sweep_deg,
        extra_free_space_spawn_samples=args.extra_free_space_spawn_samples,
        free_space_sampling_seed=args.free_space_sampling_seed,
        min_spawn_wall_clearance_m=args.min_spawn_wall_clearance_m,
        min_spawn_obstacle_clearance_m=args.min_spawn_obstacle_clearance_m,
    )
    _write_json(spawn_augmentation_report_path, spawn_augmentation_report)

    allowed_motion_ids, motion_set_report = _load_allowed_motion_ids(
        args.motion_set_mode,
        args.starterpack_whitelist,
    )
    camera_distance_bins_m = _parse_float_list(args.camera_distance_bins)
    _write_json(motion_set_ids_path, motion_set_report)

    if args.valid_single_json is not None or args.valid_two_human_json is not None:
        if args.valid_single_json is None or args.valid_two_human_json is None:
            raise RuntimeError(
                "--valid-single-json and --valid-two-human-json must be provided together"
            )
        _, valid_by_room = _load_valid_pairs_by_room(args.valid_single_json, "valid_pairs")
        _, valid_two_by_room = _load_valid_pairs_by_room(
            args.valid_two_human_json, "valid_two_human_groups"
        )
        valid_groups_by_room, room_stats = _expand_pair_groups_to_multi_groups(
            valid_by_room,
            valid_two_by_room,
            max_human_count=args.max_human_count,
            allow_duplicate_motion_in_group=args.allow_duplicate_motion_in_group,
        )
        prefilter_manifest = _build_miniscene_manifest(
            metadata,
            args.metadata,
            valid_by_room,
            valid_groups_by_room,
            args.max_single_per_room,
            args.max_two_human_per_room,
            args.frame_start,
            args.frame_end,
            min_human_count=args.min_human_count,
            max_human_count=args.max_human_count,
            selection_seed=args.selection_seed,
            allow_duplicate_motion_in_group=args.allow_duplicate_motion_in_group,
            motion_set_report={
                "motion_set_mode": "all",
                "source_path": None,
                "motion_id_count": None,
                "motion_ids": None,
                "motion_ids_by_identity": None,
            },
        )
        valid_by_room_before = dict(valid_by_room)
        valid_groups_by_room_before = dict(valid_groups_by_room)
        valid_by_room = _filter_valid_by_room_by_allowed_motion_ids(
            valid_by_room,
            allowed_motion_ids,
        )
        valid_two_by_room = _filter_valid_groups_by_room_by_allowed_motion_ids(
            valid_two_by_room,
            allowed_motion_ids,
        )
        valid_groups_by_room = _filter_valid_groups_by_room_by_allowed_motion_ids(
            valid_groups_by_room,
            allowed_motion_ids,
        )
        manifest = _build_miniscene_manifest(
            metadata,
            args.metadata,
            valid_by_room,
            valid_groups_by_room,
            args.max_single_per_room,
            args.max_two_human_per_room,
            args.frame_start,
            args.frame_end,
            min_human_count=args.min_human_count,
            max_human_count=args.max_human_count,
            selection_seed=args.selection_seed,
            allow_duplicate_motion_in_group=args.allow_duplicate_motion_in_group,
            motion_set_report=motion_set_report,
            enable_spatial_camera_selection=args.enable_spatial_camera_selection,
            distance_bins_m=camera_distance_bins_m,
            near_seam_threshold_deg=args.near_seam_threshold_deg,
            spatial_diversity_weight=args.spatial_diversity_weight,
            distance_diversity_weight=args.distance_diversity_weight,
            azimuth_diversity_weight=args.azimuth_diversity_weight,
            seam_diversity_weight=args.seam_diversity_weight,
            scale_diversity_weight=args.scale_diversity_weight,
            multi_depth_weight=args.multi_depth_weight,
            spatial_selection_mode=args.spatial_selection_mode,
            motion_diversity_weight=args.motion_diversity_weight,
            enable_spatial_dedup=args.enable_spatial_dedup,
            spatial_dedup_xy_threshold_m=args.spatial_dedup_xy_threshold_m,
            spatial_dedup_report_only=args.spatial_dedup_report_only,
        )
        manifest, human_scene_collision_audit_report = _audit_manifest_human_scene_collisions(
            metadata,
            manifest,
            output_dir=args.output_dir,
            radius_m=args.human_collision_radius_m,
            sample_stride=args.human_collision_sample_stride,
            reject_human_scene_collisions=args.reject_human_scene_collisions,
            npz_root=args.npz_root,
            npz_trans_ground_axes=args.npz_trans_ground_axes,
            min_required_wall_clearance_m=args.min_human_wall_clearance_m,
            min_required_obstacle_clearance_m=args.min_human_obstacle_clearance_m,
        )
        _write_json(manifest_path, manifest)
        _write_json(
            human_scene_collision_audit_report_path,
            human_scene_collision_audit_report,
        )
        _write_json(
            valid_two_human_path,
            _valid_two_human_groups_payload(
                args.metadata,
                metadata,
                valid_two_by_room,
                args.interhuman_mode,
                args.interhuman_margin_m,
            ),
        )
        _write_json(
            valid_multi_human_path,
            _valid_multi_human_groups_payload(
                args.metadata,
                metadata,
                valid_groups_by_room,
                args.interhuman_mode,
                args.interhuman_margin_m,
            ),
        )
        _write_json(
            starterpack_report_path,
            _starterpack_manifest_filter_report(
                metadata,
                motion_set_report,
                prefilter_manifest,
                valid_by_room_before,
                valid_by_room,
                valid_groups_by_room_before,
                valid_groups_by_room,
                manifest,
                scene_collision_filter_report=None,
            ),
        )
        if args.enable_spatial_camera_audit or args.enable_spatial_camera_selection:
            _write_json(
                spatial_camera_audit_report_path,
                _spatial_camera_audit_report(
                    metadata,
                    valid_by_room,
                    manifest,
                    distance_bins_m=camera_distance_bins_m,
                    near_seam_threshold_deg=args.near_seam_threshold_deg,
                ),
            )
        if args.enable_spatial_dedup:
            _write_json(
                spatial_dedup_report_path,
                manifest.get("spatial_dedup_report") or {},
            )
        _print_manifest_summary(manifest)
        print("")
        print(f"renderable_motion_ids.json: {motion_set_ids_path}")
        print(f"starterpack_manifest_filter_report.json: {starterpack_report_path}")
        if args.enable_spatial_camera_audit or args.enable_spatial_camera_selection:
            print(f"spatial_camera_audit_report.json: {spatial_camera_audit_report_path}")
        if args.enable_spatial_dedup:
            print(f"spatial_dedup_report.json: {spatial_dedup_report_path}")
        print(f"human_scene_collision_audit_report.json: {human_scene_collision_audit_report_path}")
        print(f"valid_two_human_groups_by_room.json: {valid_two_human_path}")
        print(f"valid_multi_human_groups_by_room.json: {valid_multi_human_path}")
        print(f"{manifest_path.name}: {manifest_path}")
        return

    explicit_motion_root_paths = _collect_motion_root_paths(args.motion_root, args.motion_root_dir)
    motion_root_paths, resolved_motion_root_report = _resolve_motion_root_paths(
        explicit_motion_root_paths,
        allowed_motion_ids,
        args.output_dir,
        args.motion_set_mode,
        max_motion_roots_tested=args.max_motion_roots_tested,
        motion_root_selection_seed=args.motion_root_selection_seed,
        prefer_identity_diversity=args.prefer_identity_diversity,
        npz_root=args.npz_root,
        npz_trans_ground_axes=args.npz_trans_ground_axes,
    )
    if not motion_root_paths:
        raise RuntimeError(
            "--motion-root-dir or at least one --motion-root is required unless "
            "--valid-single-json/--valid-two-human-json are provided. "
            "In starterpack_only mode you can also omit them and use auto-export from NPZ."
        )
    motion_root_paths, motion_root_filter_report = _filter_motion_root_paths_by_allowed_ids(
        motion_root_paths,
        allowed_motion_ids,
    )
    motion_set_report["motion_root_filter"] = motion_root_filter_report
    motion_set_report["resolved_motion_root_report"] = resolved_motion_root_report

    motion_payloads, all_results, valid_by_room, geometry_cache = _build_single_pair_cache(
        metadata,
        motion_root_paths,
        max_placement_attempts_per_motion=args.max_placement_attempts_per_motion,
        max_valid_placements_per_motion=args.max_valid_placements_per_motion,
        max_valid_placements_per_room=args.max_valid_placements_per_room,
        placement_search_seed=args.placement_search_seed,
    )
    (
        valid_by_room,
        geometry_cache,
        scene_collision_filter_report,
    ) = _filter_valid_pairs_by_human_scene_collision(
        metadata,
        all_results,
        valid_by_room,
        geometry_cache,
        radius_m=args.human_collision_radius_m,
        sample_stride=args.human_collision_sample_stride,
        reject_human_scene_collisions=args.reject_human_scene_collisions,
        min_required_wall_clearance_m=args.min_human_wall_clearance_m,
        min_required_obstacle_clearance_m=args.min_human_obstacle_clearance_m,
    )
    audited_all_results = list(scene_collision_filter_report.get("audited_results", all_results))
    (
        valid_two_by_room,
        valid_groups_by_room,
        failed_near_misses_by_room,
        room_stats,
    ) = _build_multi_human_groups(
        valid_by_room,
        geometry_cache,
        interhuman_mode=args.interhuman_mode,
        interhuman_margin_m=args.interhuman_margin_m,
        near_miss_limit=args.near_miss_limit,
        max_human_count=args.max_human_count,
        allow_duplicate_motion_in_group=args.allow_duplicate_motion_in_group,
    )
    prefilter_valid_by_room = dict(valid_by_room)
    prefilter_valid_groups_by_room = dict(valid_groups_by_room)
    prefilter_manifest = _build_miniscene_manifest(
        metadata,
        args.metadata,
        valid_by_room,
        valid_groups_by_room,
        args.max_single_per_room,
        args.max_two_human_per_room,
        args.frame_start,
        args.frame_end,
        min_human_count=args.min_human_count,
        max_human_count=args.max_human_count,
        selection_seed=args.selection_seed,
        allow_duplicate_motion_in_group=args.allow_duplicate_motion_in_group,
        motion_set_report={
            "motion_set_mode": "all",
            "source_path": None,
            "motion_id_count": None,
            "motion_ids": None,
            "motion_ids_by_identity": None,
        },
        enable_spatial_camera_selection=False,
    )
    valid_by_room = _filter_valid_by_room_by_allowed_motion_ids(
        valid_by_room,
        allowed_motion_ids,
    )
    valid_groups_by_room = _filter_valid_groups_by_room_by_allowed_motion_ids(
        valid_groups_by_room,
        allowed_motion_ids,
    )

    _write_json(
        valid_single_path,
        _valid_single_pairs_payload(args.metadata, metadata, valid_by_room),
    )
    _write_json(
        all_single_path,
        _all_single_pair_results_payload(args.metadata, metadata, motion_payloads, audited_all_results),
    )
    _write_json(
        valid_two_human_path,
        _valid_two_human_groups_payload(
            args.metadata,
            metadata,
            valid_two_by_room,
            args.interhuman_mode,
            args.interhuman_margin_m,
        ),
    )
    _write_json(
        valid_multi_human_path,
        _valid_multi_human_groups_payload(
            args.metadata,
            metadata,
            valid_groups_by_room,
            args.interhuman_mode,
            args.interhuman_margin_m,
        ),
    )
    _write_json(
        failed_two_human_path,
        _failed_two_human_near_misses_payload(
            args.metadata,
            metadata,
            failed_near_misses_by_room,
            room_stats,
            args.interhuman_mode,
            args.interhuman_margin_m,
        ),
    )
    _write_json(
        candidate_generation_diagnostics_path,
        {
            **_candidate_generation_diagnostics(
            motion_set_report,
            resolved_motion_root_report,
            motion_payloads,
            audited_all_results,
            valid_by_room,
            valid_groups_by_room,
            scene_collision_filter_report,
            args.npz_root,
            ),
            "spawn_augmentation_report": spawn_augmentation_report,
            "planner_search_config": {
                "spawn_yaw_sweep_deg": _parse_spawn_yaw_sweep_deg(args.spawn_yaw_sweep_deg),
                "extra_free_space_spawn_samples": int(args.extra_free_space_spawn_samples),
                "free_space_sampling_seed": int(args.free_space_sampling_seed),
                "min_spawn_wall_clearance_m": float(args.min_spawn_wall_clearance_m),
                "min_spawn_obstacle_clearance_m": float(args.min_spawn_obstacle_clearance_m),
                "max_placement_attempts_per_motion": int(args.max_placement_attempts_per_motion),
                "max_valid_placements_per_motion": int(args.max_valid_placements_per_motion),
                "max_valid_placements_per_room": int(args.max_valid_placements_per_room),
                "placement_search_seed": int(args.placement_search_seed),
            },
            "planning_runtime_seconds": float(time.perf_counter() - planning_start_time),
        },
    )
    manifest = _build_miniscene_manifest(
        metadata,
        args.metadata,
        valid_by_room,
        valid_groups_by_room,
        args.max_single_per_room,
        args.max_two_human_per_room,
        args.frame_start,
        args.frame_end,
        min_human_count=args.min_human_count,
        max_human_count=args.max_human_count,
        selection_seed=args.selection_seed,
        allow_duplicate_motion_in_group=args.allow_duplicate_motion_in_group,
        motion_set_report=motion_set_report,
        enable_spatial_camera_selection=args.enable_spatial_camera_selection,
        distance_bins_m=camera_distance_bins_m,
        near_seam_threshold_deg=args.near_seam_threshold_deg,
        spatial_diversity_weight=args.spatial_diversity_weight,
        distance_diversity_weight=args.distance_diversity_weight,
        azimuth_diversity_weight=args.azimuth_diversity_weight,
        seam_diversity_weight=args.seam_diversity_weight,
        scale_diversity_weight=args.scale_diversity_weight,
        multi_depth_weight=args.multi_depth_weight,
        spatial_selection_mode=args.spatial_selection_mode,
        motion_diversity_weight=args.motion_diversity_weight,
        enable_spatial_dedup=args.enable_spatial_dedup,
        spatial_dedup_xy_threshold_m=args.spatial_dedup_xy_threshold_m,
        spatial_dedup_report_only=args.spatial_dedup_report_only,
    )
    manifest, human_scene_collision_audit_report = _audit_manifest_human_scene_collisions(
        metadata,
        manifest,
        output_dir=args.output_dir,
        radius_m=args.human_collision_radius_m,
        sample_stride=args.human_collision_sample_stride,
        reject_human_scene_collisions=False,
        npz_root=args.npz_root,
        npz_trans_ground_axes=args.npz_trans_ground_axes,
        min_required_wall_clearance_m=args.min_human_wall_clearance_m,
        min_required_obstacle_clearance_m=args.min_human_obstacle_clearance_m,
    )
    _write_json(manifest_path, manifest)
    _write_json(
        human_scene_collision_audit_report_path,
        human_scene_collision_audit_report,
    )
    _write_json(
        starterpack_report_path,
        _starterpack_manifest_filter_report(
            metadata,
            motion_set_report,
            prefilter_manifest,
            prefilter_valid_by_room,
            valid_by_room,
            prefilter_valid_groups_by_room,
            valid_groups_by_room,
            manifest,
            scene_collision_filter_report=scene_collision_filter_report,
        ),
    )
    if args.enable_spatial_camera_audit or args.enable_spatial_camera_selection:
        _write_json(
            spatial_camera_audit_report_path,
            _spatial_camera_audit_report(
                metadata,
                valid_by_room,
                manifest,
                distance_bins_m=camera_distance_bins_m,
                near_seam_threshold_deg=args.near_seam_threshold_deg,
            ),
        )
    if args.enable_spatial_dedup:
        _write_json(
            spatial_dedup_report_path,
            manifest.get("spatial_dedup_report") or {},
        )

    _print_summary(
        metadata,
        motion_payloads,
        valid_by_room,
        valid_groups_by_room,
        failed_near_misses_by_room,
        room_stats,
    )
    print("")
    print(f"renderable_motion_ids.json: {motion_set_ids_path}")
    print(f"starterpack_manifest_filter_report.json: {starterpack_report_path}")
    print(f"valid_single_pairs_by_room.json: {valid_single_path}")
    print(f"all_single_pair_results.json: {all_single_path}")
    print(f"valid_two_human_groups_by_room.json: {valid_two_human_path}")
    print(f"valid_multi_human_groups_by_room.json: {valid_multi_human_path}")
    print(f"failed_two_human_near_misses.json: {failed_two_human_path}")
    print(f"candidate_generation_diagnostics.json: {candidate_generation_diagnostics_path}")
    print(f"spawn_augmentation_report.json: {spawn_augmentation_report_path}")
    if args.enable_spatial_camera_audit or args.enable_spatial_camera_selection:
        print(f"spatial_camera_audit_report.json: {spatial_camera_audit_report_path}")
    if args.enable_spatial_dedup:
        print(f"spatial_dedup_report.json: {spatial_dedup_report_path}")
    print(f"human_scene_collision_audit_report.json: {human_scene_collision_audit_report_path}")
    print(f"{manifest_path.name}: {manifest_path}")
    _print_manifest_summary(manifest)


if __name__ == "__main__":
    main()
