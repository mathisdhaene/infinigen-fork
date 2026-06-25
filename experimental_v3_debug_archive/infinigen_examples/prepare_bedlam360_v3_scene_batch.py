import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_OUTPUT_ROOT = Path("outputs/indoors/bedlam360_v3_scenes")
DEFAULT_MOTION_ROOT_DIR = Path("outputs/indoors/human_spawn_poc/motion_roots_10")
DEFAULT_STARTERPACK_WHITELIST = Path(
    "/media/mathis/PANO/bedlam2_render/config/whitelist_animations_starterpack.json"
)
DEFAULT_UNREAL_RENDERER = Path(
    "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_selected_infinigen_bedlam_erp.py"
)
DEFAULT_BLENDER_PYTHON = Path("blender/4.2/python/bin/python3.11")
DEFAULT_BLENDER_BIN = Path("blender/blender")
DEFAULT_SCENE_CONFIGS = ["base", "fast_solve.gin"]
DEFAULT_SCENE_OVERRIDES = ["compose_indoors.terrain_enabled=False"]


@dataclass(frozen=True)
class ScenePaths:
    seed: int
    root: Path
    usd_root: Path
    planning_root: Path
    scene_blend: Path
    scene_collision_metadata: Path
    human_spawn_poses: Path
    solve_state: Path
    blender_light_manifest: Path
    usd_stage: Path
    usd_light_manifest: Path
    starterpack_manifest: Path
    starterpack_filter_report: Path
    render_recipe_json: Path


def _shell_join(parts: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _scene_paths(output_root: Path, seed: int) -> ScenePaths:
    root = output_root / f"seed_{seed}"
    usd_root = root / "usd_export"
    planning_root = root / "miniscene_selection_v0"
    return ScenePaths(
        seed=seed,
        root=root,
        usd_root=usd_root,
        planning_root=planning_root,
        scene_blend=root / "scene.blend",
        scene_collision_metadata=root / "scene_collision_metadata.json",
        human_spawn_poses=root / "human_spawn_poses.json",
        solve_state=root / "solve_state.json",
        blender_light_manifest=root / "blender_light_manifest.json",
        usd_stage=usd_root / "export_scene.blend" / "export_scene.usdc",
        usd_light_manifest=usd_root / "usd_light_manifest.json",
        starterpack_manifest=planning_root / "bedlam360_infinigen_miniscenes_starterpack_only.json",
        starterpack_filter_report=planning_root / "starterpack_manifest_filter_report.json",
        render_recipe_json=planning_root / "starterpack_render_recipe.json",
    )


def _generate_scene_command(blender_python: Path, scene: ScenePaths, configs, overrides):
    cmd = [
        blender_python,
        "infinigen_examples/generate_indoors.py",
        "--seed",
        str(scene.seed),
        "--task",
        "coarse",
        "--output_folder",
        scene.root,
        "-g",
        *configs,
    ]
    if overrides:
        cmd.extend(["-p", *overrides])
    return cmd


def _export_usd_command(blender_python: Path, scene: ScenePaths, export_format: str):
    return [
        blender_python,
        "-m",
        "infinigen.tools.export",
        "--input_folder",
        scene.root,
        "--output_folder",
        scene.usd_root,
        "-f",
        export_format,
        "-r",
        "1024",
        "--omniverse",
    ]


def _export_blender_light_manifest_command(blender_bin: Path, scene: ScenePaths):
    return [
        blender_bin,
        "-b",
        scene.scene_blend,
        "--python",
        "infinigen_examples/export_blender_light_manifest.py",
        "--",
        "--output-path",
        scene.blender_light_manifest,
    ]


def _export_usd_light_manifest_command(blender_python: Path, scene: ScenePaths):
    return [
        blender_python,
        "infinigen_examples/export_usd_light_manifest.py",
        "--stage-path",
        scene.usd_stage,
        "--output-path",
        scene.usd_light_manifest,
    ]


def _planner_command(
    blender_python: Path,
    scene: ScenePaths,
    motion_root_dir: Path,
    starterpack_whitelist: Path,
    max_single_per_room: int,
    max_two_human_per_room: int,
    min_human_count: int,
    frame_start: int,
    frame_end: int,
):
    return [
        blender_python,
        "infinigen_examples/plan_bedlam360_miniscenes.py",
        "--metadata",
        scene.scene_collision_metadata,
        "--motion-root-dir",
        motion_root_dir,
        "--output-dir",
        scene.planning_root,
        "--motion-set-mode",
        "starterpack_only",
        "--starterpack-whitelist",
        starterpack_whitelist,
        "--max-single-per-room",
        str(max_single_per_room),
        "--max-two-human-per-room",
        str(max_two_human_per_room),
        "--min-human-count",
        str(min_human_count),
        "--frame-start",
        str(frame_start),
        "--frame-end",
        str(frame_end),
    ]


def _copy_scene_sidecars(scene: ScenePaths):
    scene.usd_root.mkdir(parents=True, exist_ok=True)
    for src in (
        scene.scene_collision_metadata,
        scene.human_spawn_poses,
        scene.solve_state,
        scene.blender_light_manifest,
    ):
        if src.exists():
            dst = scene.usd_root / src.name
            dst.write_bytes(src.read_bytes())


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_command(cmd, dry_run: bool):
    print(_shell_join(cmd))
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], check=True)


def _scene_recipe(
    scene: ScenePaths,
    blender_python: Path,
    blender_bin: Path,
    motion_root_dir: Path,
    starterpack_whitelist: Path,
    unreal_renderer: Path,
    configs,
    overrides,
    export_format: str,
    max_single_per_room: int,
    max_two_human_per_room: int,
    min_human_count: int,
    frame_start: int,
    frame_end: int,
):
    generate_cmd = _generate_scene_command(blender_python, scene, configs, overrides)
    export_cmd = _export_usd_command(blender_python, scene, export_format)
    blender_manifest_cmd = _export_blender_light_manifest_command(blender_bin, scene)
    usd_manifest_cmd = _export_usd_light_manifest_command(blender_python, scene)
    planner_cmd = _planner_command(
        blender_python,
        scene,
        motion_root_dir,
        starterpack_whitelist,
        max_single_per_room,
        max_two_human_per_room,
        min_human_count,
        frame_start,
        frame_end,
    )
    sample_render_cmd = [
        "py",
        unreal_renderer,
        "--manifest",
        scene.starterpack_manifest,
        "--miniscene-index",
        "0",
    ]
    return {
        "seed": scene.seed,
        "scene_root": str(scene.root),
        "artifacts": {
            "scene_blend": str(scene.scene_blend),
            "scene_collision_metadata": str(scene.scene_collision_metadata),
            "human_spawn_poses": str(scene.human_spawn_poses),
            "solve_state": str(scene.solve_state),
            "blender_light_manifest": str(scene.blender_light_manifest),
            "usd_stage": str(scene.usd_stage),
            "usd_light_manifest": str(scene.usd_light_manifest),
            "starterpack_manifest": str(scene.starterpack_manifest),
            "starterpack_filter_report": str(scene.starterpack_filter_report),
        },
        "commands": {
            "generate_scene": _shell_join(generate_cmd),
            "export_usd": _shell_join(export_cmd),
            "export_blender_light_manifest": _shell_join(blender_manifest_cmd),
            "export_usd_light_manifest": _shell_join(usd_manifest_cmd),
            "plan_starterpack_miniscenes": _shell_join(planner_cmd),
            "render_first_miniscene_example": _shell_join(sample_render_cmd),
        },
    }


def _write_shell_recipe(output_root: Path, recipes):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# BEDLAM360 v3 batch prep recipe",
        "# Generated by infinigen_examples/prepare_bedlam360_v3_scene_batch.py",
        "",
    ]
    for recipe in recipes:
        lines.append(f"# seed {recipe['seed']}")
        commands = recipe["commands"]
        lines.append(commands["generate_scene"])
        lines.append(commands["export_usd"])
        lines.append(commands["export_blender_light_manifest"])
        lines.append(commands["export_usd_light_manifest"])
        lines.append(commands["plan_starterpack_miniscenes"])
        lines.append("")
    script_path = output_root / "run_bedlam360_v3_batch_prep.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare BEDLAM360 v3 Infinigen scenes and starterpack-only mini-scene manifests."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--motion-root-dir", type=Path, default=DEFAULT_MOTION_ROOT_DIR)
    parser.add_argument(
        "--starterpack-whitelist",
        type=Path,
        default=DEFAULT_STARTERPACK_WHITELIST,
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--blender-python", type=Path, default=DEFAULT_BLENDER_PYTHON)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--unreal-renderer", type=Path, default=DEFAULT_UNREAL_RENDERER)
    parser.add_argument("--scene-config", action="append", default=None)
    parser.add_argument("--scene-override", action="append", default=None)
    parser.add_argument("--export-format", default="usdc")
    parser.add_argument("--max-single-per-room", type=int, default=0)
    parser.add_argument("--max-two-human-per-room", type=int, default=5)
    parser.add_argument("--min-human-count", type=int, default=2)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=180)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("generate", "export", "blender_light_manifest", "usd_light_manifest", "plan"),
        default=["generate", "export", "blender_light_manifest", "usd_light_manifest", "plan"],
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--write-shell-recipe", action="store_true", default=True)
    parser.add_argument("--no-write-shell-recipe", dest="write_shell_recipe", action="store_false")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else list(range(args.seed_start, args.seed_start + args.num_scenes))
    configs = list(args.scene_config or DEFAULT_SCENE_CONFIGS)
    overrides = list(args.scene_override or DEFAULT_SCENE_OVERRIDES)

    args.output_root.mkdir(parents=True, exist_ok=True)
    batch_report = {
        "phase": "bedlam360_v3_batch_prep",
        "starterpack_only": True,
        "seeds": list(seeds),
        "output_root": str(args.output_root),
        "motion_root_dir": str(args.motion_root_dir),
        "starterpack_whitelist": str(args.starterpack_whitelist),
        "scene_configs": configs,
        "scene_overrides": overrides,
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "min_human_count": int(args.min_human_count),
        "recipes": [],
    }

    for seed in seeds:
        scene = _scene_paths(args.output_root, int(seed))
        recipe = _scene_recipe(
            scene,
            args.blender_python,
            args.blender_bin,
            args.motion_root_dir,
            args.starterpack_whitelist,
            args.unreal_renderer,
            configs,
            overrides,
            args.export_format,
            args.max_single_per_room,
            args.max_two_human_per_room,
            args.min_human_count,
            args.frame_start,
            args.frame_end,
        )
        batch_report["recipes"].append(recipe)

        if "generate" in args.stages:
            _run_command(
                _generate_scene_command(args.blender_python, scene, configs, overrides),
                args.dry_run,
            )
        if "blender_light_manifest" in args.stages:
            _run_command(
                _export_blender_light_manifest_command(args.blender_bin, scene),
                args.dry_run,
            )
        if "export" in args.stages:
            _run_command(
                _export_usd_command(args.blender_python, scene, args.export_format),
                args.dry_run,
            )
        if "usd_light_manifest" in args.stages:
            _run_command(
                _export_usd_light_manifest_command(args.blender_python, scene),
                args.dry_run,
            )
        if not args.dry_run:
            _copy_scene_sidecars(scene)
        if "plan" in args.stages:
            _run_command(
                _planner_command(
                    args.blender_python,
                    scene,
                    args.motion_root_dir,
                    args.starterpack_whitelist,
                    args.max_single_per_room,
                    args.max_two_human_per_room,
                    args.min_human_count,
                    args.frame_start,
                    args.frame_end,
                ),
                args.dry_run,
            )

    if args.write_shell_recipe:
        batch_report["shell_recipe"] = str(_write_shell_recipe(args.output_root, batch_report["recipes"]))
    _write_json(args.output_root / "bedlam360_v3_batch_prep_report.json", batch_report)

    print(f"Prepared {len(seeds)} scene recipes under {args.output_root}")
    print(f"Batch report: {args.output_root / 'bedlam360_v3_batch_prep_report.json'}")
    if batch_report.get("shell_recipe"):
        print(f"Shell recipe: {batch_report['shell_recipe']}")


if __name__ == "__main__":
    main()
