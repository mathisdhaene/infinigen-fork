import argparse
import importlib
import json
import sys
from pathlib import Path

import unreal

import render_selected_infinigen_bedlam_erp as selected  # noqa: E402

selected = importlib.reload(selected)


def log_info(message):
    unreal.log(f"[BEDLAM_INFINIGEN_WARMUP] {message}")


def _load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, default=None)
    parser.add_argument("--manifest", "--manifest-path", dest="manifest_path", type=Path, required=True)
    parser.add_argument("--miniscene-index", type=int, default=0)
    parser.add_argument("--miniscene-id", default=None)
    parser.add_argument("--miniscene-room", default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_known_args(sys.argv[1:])[0]


def warmup_selected_infinigen_bedlam_appearance():
    args = _load_args()

    config = argparse.Namespace(
        raw_sys_argv=list(sys.argv),
        scene_root=args.scene_root,
        manifest_path=args.manifest_path,
        miniscene_index=args.miniscene_index,
        miniscene_id=args.miniscene_id,
        miniscene_room=args.miniscene_room,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        explicit_manifest_arg=True,
        explicit_miniscene_id_arg=bool(args.miniscene_id not in (None, "")),
        explicit_miniscene_room_arg=bool(args.miniscene_room not in (None, "")),
        explicit_miniscene_index_arg=True,
        explicit_frame_start_arg=bool(args.frame_start is not None),
        explicit_frame_end_arg=bool(args.frame_end is not None),
    )

    scene_root, resolved_manifest_path, _startup_manifest, startup_miniscenes = selected._startup_manifest_validation(config)
    config.manifest_path = resolved_manifest_path
    config.scene_root = scene_root
    scene_binding_report = selected._configure_scene_root_paths(scene_root)
    usd_stage_validation = selected._validate_loaded_usd_stage_matches_scene_root(selected.base.USD_STAGE_PATH)

    _source_manifest, requested_index = selected._resolve_requested_miniscene_index(
        config.manifest_path,
        miniscene_index=config.miniscene_index,
        miniscene_id=config.miniscene_id,
        room_hint=config.miniscene_room,
    )
    requested_miniscene = startup_miniscenes[requested_index]
    _source_manifest, miniscene, selected_index, selection_info = selected._select_renderable_miniscene(
        config.manifest_path,
        requested_index,
        allow_fallback=False,
    )

    humans = [dict(human) for human in list(miniscene.get("humans", []))]
    if not humans:
        raise RuntimeError(f"Mini-scene {miniscene.get('miniscene_id')} contains no humans")
    human_count = len(humans)
    for human in humans:
        human["_human_count"] = human_count
    body_specs = [selected._build_body_spec(human, i) for i, human in enumerate(humans)]

    scene_metadata = selected.base.load_scene_metadata()
    if selected.USE_MINISCENE_ANCHOR_CAMERA:
        camera_selection = selected._resolve_miniscene_anchor_camera(miniscene, scene_metadata)
    else:
        anchor_location = selected._anchor_location(body_specs)
        anchor_yaw = float(body_specs[0]["yaw"])
        camera_selection = selected.base._resolve_capture_camera_pose(
            anchor_location,
            anchor_yaw,
            room_name=miniscene.get("room"),
        )
    camera_pose = camera_selection["camera_pose_cm_deg"]

    frame_start = int(miniscene.get("render_options", {}).get("frame_start", 12))
    frame_end = int(miniscene.get("render_options", {}).get("frame_end", 18))
    if args.frame_start is not None:
        frame_start = int(args.frame_start)
    if args.frame_end is not None:
        frame_end = int(args.frame_end)

    run_root = selected.base._ensure_dir(selected.base.BRIDGE_ROOT / "manual_warmup")
    if args.force:
        try:
            setattr(unreal, selected._appearance_warmup_session_key(), False)
        except Exception:
            pass

    original_enable = bool(selected.ENABLE_APPEARANCE_WARMUP)
    original_once = bool(selected.APPEARANCE_WARMUP_ONCE_PER_UNREAL_SESSION)
    try:
        selected.ENABLE_APPEARANCE_WARMUP = True
        selected.APPEARANCE_WARMUP_ONCE_PER_UNREAL_SESSION = False
        warmup_report = selected._run_appearance_warmup(
            run_root=run_root,
            body_specs=body_specs,
            camera_pose=camera_pose,
            frame_start=frame_start,
            frame_end=frame_end,
        )
    finally:
        selected.ENABLE_APPEARANCE_WARMUP = original_enable
        selected.APPEARANCE_WARMUP_ONCE_PER_UNREAL_SESSION = original_once

    final_report = {
        "scene_binding": scene_binding_report,
        "usd_stage_validation": usd_stage_validation,
        "requested_miniscene": selected._miniscene_brief(requested_miniscene, index=requested_index),
        "selected_miniscene": selected._miniscene_brief(miniscene, index=selected_index),
        "selection_info": selection_info,
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "camera_pose_cm_deg": camera_pose,
        "warmup_report": warmup_report,
    }
    log_info("FINAL_REPORT " + json.dumps(final_report, indent=2))
    return final_report


if __name__ == "__main__":
    warmup_selected_infinigen_bedlam_appearance()
