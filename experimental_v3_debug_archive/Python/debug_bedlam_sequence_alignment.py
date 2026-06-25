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

import reconstruct_one_bedlam_body

reconstruct_one_bedlam_body = importlib.reload(reconstruct_one_bedlam_body)


DEFAULT_SEQ_CSV = Path("/media/mathis/PANO/BEDLAM2/gt_test_meta/20240606_1_500_stadium_closeup/be_seq.csv")
DEFAULT_CAMERA_CSV = Path("/media/mathis/PANO/BEDLAM2/gt_test/20240606_1_500_stadium_closeup/ground_truth/meta_exr_csv/seq_000433_camera.csv")
DEFAULT_SEQUENCE_NAME = "seq_000433"
DEFAULT_REPORT_PATH = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/debug_seq_000433_alignment.json")
DEFAULT_SAMPLED_FRAMES = [0, 30, 60, 90, 120]


def _load_camera_rows(camera_csv_path):
    camera_csv_path = Path(camera_csv_path)
    if not camera_csv_path.is_file():
        raise RuntimeError(f"Camera CSV not found: {camera_csv_path}")

    with open(camera_csv_path, "r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _find_sequence_group_and_bodies(seq_rows, sequence_name):
    for index, row in enumerate(seq_rows):
        if row["Type"] != "Group":
            continue
        if row["CommentMap"].get("sequence_name") != sequence_name:
            continue

        body_rows = []
        for next_index in range(index + 1, len(seq_rows)):
            next_row = seq_rows[next_index]
            if next_row["Type"] == "Group":
                break
            if next_row["Type"] == "Body":
                body_rows.append(next_row)
        return row, body_rows

    return None, None


def _camera_pose_from_row(row):
    return {
        "name": row["name"],
        "x": float(row["x"]),
        "y": float(row["y"]),
        "z": float(row["z"]),
        "yaw": float(row["yaw"]),
        "pitch": float(row["pitch"]),
        "roll": float(row["roll"]),
    }


def _camera_stats(camera_rows):
    stats = {}
    for key in ("x", "y", "z", "yaw", "pitch", "roll"):
        values = [float(row[key]) for row in camera_rows]
        stats[key] = {"min": min(values), "max": max(values)}
    return stats


def _distance_meters(camera_pose, body_pose):
    dx = camera_pose["x"] - body_pose["x"]
    dy = camera_pose["y"] - body_pose["y"]
    dz = camera_pose["z"] - body_pose["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz) / 100.0


def _sampled_frame_distances(camera_rows, body_poses, sampled_frames):
    results = []
    for frame_index in sampled_frames:
        if frame_index < 0 or frame_index >= len(camera_rows):
            continue
        camera_pose = _camera_pose_from_row(camera_rows[frame_index])
        distances = []
        for body_pose in body_poses:
            distances.append(
                {
                    "asset_id": body_pose["asset_id"],
                    "distance_m": _distance_meters(camera_pose, body_pose),
                }
            )
        distances.sort(key=lambda item: item["distance_m"])
        results.append(
            {
                "frame_index": frame_index,
                "camera_pose": camera_pose,
                "nearest_body": distances[0] if distances else None,
                "all_distances_m": distances,
            }
        )
    return results


def debug_bedlam_sequence_alignment(
    seq_csv_path=DEFAULT_SEQ_CSV,
    camera_csv_path=DEFAULT_CAMERA_CSV,
    sequence_name=DEFAULT_SEQUENCE_NAME,
    report_path=DEFAULT_REPORT_PATH,
    sampled_frames=None,
):
    if sampled_frames is None:
        sampled_frames = list(DEFAULT_SAMPLED_FRAMES)

    seq_rows = reconstruct_one_bedlam_body._load_be_seq_rows(seq_csv_path)
    group_row, body_rows = _find_sequence_group_and_bodies(seq_rows, sequence_name)

    if group_row is None:
        unreal.log_error(f"[BEDLAM360] Sequence group '{sequence_name}' not found in {seq_csv_path}")
        report = {
            "sequence_name": sequence_name,
            "seq_csv_path": str(seq_csv_path),
            "camera_csv_path": str(camera_csv_path),
            "group_found": False,
            "error": f"Sequence group '{sequence_name}' not found in be_seq.csv",
        }
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
        return report_path

    body_poses = [reconstruct_one_bedlam_body._body_pose_from_row(row) for row in body_rows]
    camera_rows = _load_camera_rows(camera_csv_path)
    camera_frame0 = _camera_pose_from_row(camera_rows[0])
    camera_minmax = _camera_stats(camera_rows)

    body_distances = []
    for body_pose in body_poses:
        body_distances.append(
            {
                "asset_id": body_pose["asset_id"],
                "distance_m": _distance_meters(camera_frame0, body_pose),
                "position_cm": {
                    "x": body_pose["x"],
                    "y": body_pose["y"],
                    "z": body_pose["z"],
                },
                "rotation_deg": {
                    "yaw": body_pose["yaw"],
                    "pitch": body_pose["pitch"],
                    "roll": body_pose["roll"],
                },
                "start_frame": body_pose["comment_map"].get("start_frame"),
                "index": body_pose["index"],
            }
        )
    body_distances.sort(key=lambda item: item["distance_m"])

    sampled_frame_info = _sampled_frame_distances(camera_rows, body_poses, sampled_frames)

    unreal.log(f"[BEDLAM360] Sequence {sequence_name} body_count={len(body_poses)}")
    for body_pose in body_poses:
        unreal.log(
            f"[BEDLAM360] Body asset_id={body_pose['asset_id']} "
            f"pos=({body_pose['x']:.2f}, {body_pose['y']:.2f}, {body_pose['z']:.2f}) "
            f"rot=({body_pose['yaw']:.2f}, {body_pose['pitch']:.2f}, {body_pose['roll']:.2f}) "
            f"start_frame={body_pose['comment_map'].get('start_frame')}"
        )

    unreal.log(
        f"[BEDLAM360] Camera frame0 pos=({camera_frame0['x']:.2f}, {camera_frame0['y']:.2f}, {camera_frame0['z']:.2f}) "
        f"rot=({camera_frame0['yaw']:.2f}, {camera_frame0['pitch']:.2f}, {camera_frame0['roll']:.2f})"
    )
    unreal.log(f"[BEDLAM360] Camera min/max={camera_minmax}")
    if body_distances:
        unreal.log(
            f"[BEDLAM360] Nearest body to frame0 camera: "
            f"{body_distances[0]['asset_id']} at {body_distances[0]['distance_m']:.3f} m"
        )

    report = {
        "sequence_name": sequence_name,
        "seq_csv_path": str(seq_csv_path),
        "camera_csv_path": str(camera_csv_path),
        "group_found": True,
        "group_row": {
            "index": group_row.get("Index"),
            "comment": group_row.get("Comment"),
            "comment_map": group_row.get("CommentMap"),
        },
        "body_count": len(body_poses),
        "bodies": [
            {
                "asset_id": body_pose["asset_id"],
                "index": body_pose["index"],
                "position_cm": {
                    "x": body_pose["x"],
                    "y": body_pose["y"],
                    "z": body_pose["z"],
                },
                "rotation_deg": {
                    "yaw": body_pose["yaw"],
                    "pitch": body_pose["pitch"],
                    "roll": body_pose["roll"],
                },
                "start_frame": body_pose["comment_map"].get("start_frame"),
                "comment_map": body_pose["comment_map"],
            }
            for body_pose in body_poses
        ],
        "camera_frame0": camera_frame0,
        "camera_position_minmax": {
            "x": camera_minmax["x"],
            "y": camera_minmax["y"],
            "z": camera_minmax["z"],
        },
        "camera_rotation_minmax": {
            "yaw": camera_minmax["yaw"],
            "pitch": camera_minmax["pitch"],
            "roll": camera_minmax["roll"],
        },
        "frame0_body_distances_m": body_distances,
        "nearest_body_frame0": body_distances[0] if body_distances else None,
        "sampled_frame_distances": sampled_frame_info,
    }

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    unreal.log(f"[BEDLAM360] Wrote alignment report: {report_path}")
    return report_path


if __name__ == "__main__":
    debug_bedlam_sequence_alignment()
