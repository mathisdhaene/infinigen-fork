import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


STARTERPACK_WHITELIST_PATH = Path(
    "/media/mathis/PANO/bedlam2_render/config/whitelist_animations_starterpack.json"
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def _starterpack_summary(path: Path):
    payload = _read_json(path)
    motion_ids = []
    by_identity = {}
    for identity, motions in sorted(payload.items()):
        ids = [f"{identity}_{motion}" for motion in motions]
        by_identity[identity] = ids
        motion_ids.extend(ids)
    return {
        "source_path": str(path),
        "total_motion_ids_available": len(motion_ids),
        "total_identities_available": len(by_identity),
        "motions_per_identity": {k: len(v) for k, v in sorted(by_identity.items())},
        "motion_ids": sorted(motion_ids),
    }


def _renderable_summary(path: Path | None):
    if path is None or not path.exists():
        return {
            "source_path": None,
            "available": False,
            "note": "No renderable_motion_ids.json found.",
        }
    payload = _read_json(path)
    by_identity = payload.get("motion_ids_by_identity") or {}
    motion_ids = payload.get("motion_ids") or []
    note = None
    if payload.get("motion_set_mode") == "starterpack_only" and len(motion_ids) == 150:
        note = (
            "This renderable list mirrors the starterpack whitelist and may not reflect a "
            "reduced Unreal-side registry subset."
        )
    return {
        "source_path": str(path),
        "available": True,
        "motion_set_mode": payload.get("motion_set_mode"),
        "total_renderable_motion_ids": len(motion_ids),
        "total_renderable_identities": len(by_identity),
        "motions_per_identity": {k: len(v) for k, v in sorted(by_identity.items())},
        "motion_ids": sorted(motion_ids),
        "note": note,
    }


def _candidate_summary(valid_single_path: Path, valid_multi_path: Path | None):
    valid_single = _read_json(valid_single_path)
    room_valid_pairs = {}
    motion_counts = Counter()
    identity_counts = Counter()
    fail_reasons = Counter()
    total_candidates = 0
    for room_entry in valid_single.get("rooms", []):
        room_name = ((room_entry.get("room") or {}).get("name"))
        valid_pairs = list(room_entry.get("valid_pairs", []))
        room_valid_pairs[room_name] = len(valid_pairs)
        total_candidates += len(valid_pairs)
        for row in valid_pairs:
            motion_id = _normalize_motion_id(row.get("motion_id"))
            motion_counts[motion_id] += 1
            identity_counts[_motion_identity(motion_id)] += 1
    all_single_results_path = valid_single_path.parent / "all_single_pair_results.json"
    if all_single_results_path.exists():
        all_results = _read_json(all_single_results_path)
        for row in all_results.get("results", []):
            if not row.get("pass", False):
                for reason in row.get("reasons", []):
                    fail_reasons[str(reason)] += 1

    multi_summary = None
    if valid_multi_path and valid_multi_path.exists():
        payload = _read_json(valid_multi_path)
        group_count_by_human_count = Counter()
        total_multi = 0
        for room_entry in payload.get("rooms", []):
            for group in room_entry.get("valid_multi_human_groups", []):
                human_count = int(group.get("human_count", len(group.get("humans", []))))
                group_count_by_human_count[human_count] += 1
                total_multi += 1
        multi_summary = {
            "total_multi_human_candidates": total_multi,
            "group_count_by_human_count": dict(sorted(group_count_by_human_count.items())),
        }

    return {
        "candidate_count": total_candidates,
        "candidate_count_by_room": room_valid_pairs,
        "unique_motion_ids": len(motion_counts),
        "unique_identities": len(identity_counts),
        "motion_usage_counts": dict(sorted(motion_counts.items())),
        "identity_usage_counts": dict(sorted(identity_counts.items())),
        "rejection_reasons": dict(sorted(fail_reasons.items())),
        "multi_human_candidates": multi_summary,
    }


def _candidate_generation_diagnostics_summary(path: Path | None):
    if path is None or not path.exists():
        return {
            "available": False,
            "source_path": None,
        }
    payload = _read_json(path)
    return {
        "available": True,
        "source_path": str(path),
        "tested_motion_id_count": payload.get("tested_motion_id_count"),
        "valid_single_motion_id_count": payload.get("valid_single_motion_id_count"),
        "valid_multi_motion_id_count": payload.get("valid_multi_motion_id_count"),
        "unique_identities_in_valid_single_pairs": payload.get("unique_identities_in_valid_single_pairs"),
        "unique_identities_in_valid_multi_groups": payload.get("unique_identities_in_valid_multi_groups"),
        "aggregate_rejection_reason_counts": payload.get("aggregate_rejection_reason_counts"),
        "aggregate_bottleneck_counts": payload.get("aggregate_bottleneck_counts"),
        "resolved_motion_root_report": payload.get("resolved_motion_root_report"),
    }


def _selected_manifest_summary(manifest_path: Path):
    manifest = _read_json(manifest_path)
    motion_counts = Counter()
    identity_counts = Counter()
    room_human_count_counts = Counter()
    duplicate_motion_scene_count = 0
    duplicate_identity_scene_count = 0
    for scene in manifest.get("miniscenes", []):
        humans = list(scene.get("humans", []))
        human_count = len(humans)
        room_human_count_counts[(scene.get("room"), human_count)] += 1
        scene_motion_ids = [_normalize_motion_id(h.get("motion_id")) for h in humans]
        scene_identity_ids = [_motion_identity(h.get("motion_id")) for h in humans]
        if len(scene_motion_ids) != len(set(scene_motion_ids)):
            duplicate_motion_scene_count += 1
        if len(scene_identity_ids) != len(set(scene_identity_ids)):
            duplicate_identity_scene_count += 1
        for motion_id in scene_motion_ids:
            motion_counts[motion_id] += 1
        for identity_id in scene_identity_ids:
            identity_counts[identity_id] += 1
    return {
        "selected_miniscene_count": int(manifest.get("total_selected_miniscenes", 0)),
        "selected_human_count_distribution": dict(
            sorted((int(k), int(v)) for k, v in (manifest.get("selected_human_count_distribution") or {}).items())
        ),
        "selected_motion_usage_counts": dict(sorted(motion_counts.items())),
        "selected_identity_usage_counts": dict(sorted(identity_counts.items())),
        "duplicate_motion_ids_within_scene_count": duplicate_motion_scene_count,
        "duplicate_identity_ids_within_scene_count": duplicate_identity_scene_count,
        "room_human_count_distribution": {
            f"{room}::{human_count}": count
            for (room, human_count), count in sorted(room_human_count_counts.items())
        },
        "rooms_covered": sorted(
            {
                scene.get("room")
                for scene in manifest.get("miniscenes", [])
                if scene.get("room") is not None
            }
        ),
    }


def _bottleneck_explanation(starterpack, renderable, candidates, selected):
    notes = []
    if not renderable.get("available"):
        notes.append("renderable_motion_list_missing")
    elif renderable.get("total_renderable_motion_ids", 0) < starterpack.get("total_motion_ids_available", 0):
        notes.append("unreal_renderability_limitation")
    if candidates.get("unique_motion_ids", 0) <= 8:
        notes.append("collision_or_room_feasibility_limitation")
    human_dist = selected.get("selected_human_count_distribution", {})
    if set(human_dist.keys()) <= {2}:
        notes.append("planner_selection_bias_exactly_two_humans")
    if selected.get("duplicate_motion_ids_within_scene_count", 0) > 0:
        notes.append("planner_selection_bias_duplicate_motions_within_scene")
    if not notes:
        notes.append("starterpack_limit_not_primary_current_bottleneck_is_selection_bias_or_room_feasibility")
    return notes


def audit_scene(scene_root: Path, starterpack_summary):
    planning_root = scene_root / "miniscene_selection_v0"
    manifest_path = planning_root / "bedlam360_infinigen_miniscenes_starterpack_only.json"
    valid_single_path = planning_root / "valid_single_pairs_by_room.json"
    valid_multi_path = planning_root / "valid_multi_human_groups_by_room.json"
    renderable_path = planning_root / "renderable_motion_ids.json"
    candidate_generation_diagnostics_path = planning_root / "candidate_generation_diagnostics.json"
    renderable = _renderable_summary(renderable_path)
    candidates = _candidate_summary(valid_single_path, valid_multi_path if valid_multi_path.exists() else None)
    candidate_generation = _candidate_generation_diagnostics_summary(candidate_generation_diagnostics_path)
    selected = _selected_manifest_summary(manifest_path)
    return {
        "scene_root": str(scene_root),
        "starterpack_whitelist": starterpack_summary,
        "unreal_renderable_motions": renderable,
        "candidate_generation_diagnostics": candidate_generation,
        "planner_candidates_before_selection": candidates,
        "final_selected_manifest": selected,
        "bottleneck_explanation": _bottleneck_explanation(
            starterpack_summary, renderable, candidates, selected
        ),
    }


def _print_summary(report):
    print("scene | selected | human_counts | unique_motions | unique_identities | bottleneck")
    for row in report.get("scenes", []):
        selected = row["final_selected_manifest"]
        candidate = row["planner_candidates_before_selection"]
        human_counts = selected.get("selected_human_count_distribution", {})
        print(
            f"{Path(row['scene_root']).name} | {selected.get('selected_miniscene_count', 0)} | "
            f"{human_counts} | {candidate.get('unique_motion_ids', 0)} | "
            f"{candidate.get('unique_identities', 0)} | "
            f"{','.join(row.get('bottleneck_explanation', []))}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenes-root",
        type=Path,
        default=Path("/media/mathis/PANO/infinigen/outputs/indoors/bedlam360_v3_scenes"),
    )
    parser.add_argument(
        "--starterpack-whitelist",
        type=Path,
        default=STARTERPACK_WHITELIST_PATH,
    )
    args = parser.parse_args()

    starterpack_summary = _starterpack_summary(args.starterpack_whitelist)
    scene_roots = sorted(path for path in args.scenes_root.glob("seed_*") if path.is_dir())
    scenes = [audit_scene(scene_root, starterpack_summary) for scene_root in scene_roots]
    report = {
        "scenes_root": str(args.scenes_root),
        "starterpack_whitelist": starterpack_summary,
        "scene_count": len(scenes),
        "scenes": scenes,
    }
    output_path = args.scenes_root / "v3_diversity_audit_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(report)
    print(f"v3_diversity_audit_report.json: {output_path}")


if __name__ == "__main__":
    main()
