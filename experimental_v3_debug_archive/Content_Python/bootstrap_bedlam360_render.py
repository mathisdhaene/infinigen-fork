"""
Minimal Unreal Editor bootstrap launcher for BEDLAM360 rendering.

Example Unreal Python Console usage:

import runpy
import sys

sys.argv = [
    "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/bootstrap_bedlam360_render.py",
    "--renderer-script", "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_selected_infinigen_bedlam_erp.py",
    "--bootstrap-wait-ticks-after-asset-registry", "30",
    "--bootstrap-wait-ticks-after-usd-load", "120",
    "--",
    "--scene-root", "/media/mathis/PANO/infinigen/outputs/indoors/bedlam360_v3_scenes/seed_101",
    "--manifest", "/media/mathis/PANO/infinigen/outputs/indoors/bedlam360_v3_scenes/seed_101/miniscene_selection_v0/bedlam360_infinigen_miniscenes_starterpack_only.json",
    "--batch",
    "--batch-balanced-rooms",
    "--max-clips", "1",
    "--frame-start", "0",
    "--frame-end", "120",
    "--render-output-profile", "dataset_rgb_fast",
    "--rgb-tonemap-mode", "ldr_passthrough",
]

runpy.run_path(
    "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/bootstrap_bedlam360_render.py",
    run_name="__main__",
)
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import traceback
from pathlib import Path

import unreal


DEFAULT_RENDERER_SCRIPT = (
    "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/"
    "render_selected_infinigen_bedlam_erp.py"
)


def _split_bootstrap_and_renderer_argv(argv):
    argv = list(argv or [])
    if "--" not in argv:
        return argv, []
    marker_index = argv.index("--")
    return argv[:marker_index], argv[marker_index + 1 :]


def _log(prefix, message, payload=None):
    line = f"[{prefix}] {message}"
    if payload is not None:
        try:
            line += " " + json.dumps(payload, indent=2, sort_keys=True)
        except Exception:
            line += f" {payload}"
    unreal.log_warning(line)


def _fail(prefix, message, payload=None):
    _log(prefix, "BOOTSTRAP_ERROR", {"message": message, "payload": payload})
    raise RuntimeError(f"{message} | payload={payload}")


def _normalize_renderer_args(renderer_args_json, passthrough_args):
    renderer_args = []
    if renderer_args_json not in (None, ""):
        decoded = json.loads(renderer_args_json)
        if not isinstance(decoded, list):
            raise RuntimeError("--renderer-args-json must decode to a JSON list.")
        renderer_args.extend(str(item) for item in decoded)
    renderer_args.extend(str(item) for item in list(passthrough_args or []))
    return renderer_args


def _extract_flag_value(args, *flags):
    args = list(args or [])
    for flag in flags:
        if flag in args:
            index = args.index(flag)
            if index + 1 >= len(args):
                raise RuntimeError(f"Missing value after {flag}")
            return args[index + 1]
    return None


def _resolve_scene_root_from_renderer_args(renderer_args):
    scene_root = _extract_flag_value(renderer_args, "--scene-root")
    if scene_root in (None, ""):
        raise RuntimeError("Renderer args must include --scene-root for bootstrap USD validation.")
    return Path(scene_root).expanduser().resolve()


def _normalize_editor_map_path(map_path):
    text = str(map_path or "").strip()
    if not text:
        return ""
    if "." not in text:
        leaf = text.rsplit("/", 1)[-1]
        text = f"{text}.{leaf}"
    return text


def _editor_map_paths_match(current_map_path, requested_map_path):
    current = _normalize_editor_map_path(current_map_path)
    requested = _normalize_editor_map_path(requested_map_path)
    return bool(current and requested and current == requested)


def _get_current_map_info():
    info = {
        "world_name": None,
        "world_path": None,
        "persistent_level_name": None,
        "persistent_level_path": None,
    }
    world = None
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()
    except Exception:
        world = None
    if world is None:
        try:
            subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
            if subsystem is not None and hasattr(subsystem, "get_editor_world"):
                world = subsystem.get_editor_world()
        except Exception:
            world = None
    if world is None:
        return info
    try:
        info["world_name"] = world.get_name()
    except Exception:
        pass
    try:
        info["world_path"] = world.get_path_name()
    except Exception:
        pass
    try:
        level = world.get_current_level()
    except Exception:
        level = None
    if level is not None:
        try:
            info["persistent_level_name"] = level.get_name()
        except Exception:
            pass
        try:
            info["persistent_level_path"] = level.get_path_name()
        except Exception:
            pass
    return info


def _open_editor_map(map_path):
    requested_map = str(map_path or "")
    attempts = []
    if not requested_map:
        return {"success": False, "requested_map": requested_map, "attempts": attempts}
    api_calls = []
    if hasattr(unreal, "EditorLoadingAndSavingUtils") and hasattr(unreal.EditorLoadingAndSavingUtils, "load_map"):
        api_calls.append(("EditorLoadingAndSavingUtils.load_map", lambda: unreal.EditorLoadingAndSavingUtils.load_map(requested_map)))
    try:
        subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    except Exception:
        subsystem = None
    if subsystem is not None and hasattr(subsystem, "load_level"):
        api_calls.append(("LevelEditorSubsystem.load_level", lambda: subsystem.load_level(requested_map)))
    if hasattr(unreal.EditorLevelLibrary, "load_level"):
        api_calls.append(("EditorLevelLibrary.load_level", lambda: unreal.EditorLevelLibrary.load_level(requested_map)))
    for api_name, callback in api_calls:
        attempt = {
            "api": api_name,
            "requested_map": requested_map,
            "current_map_before_open": _get_current_map_info(),
            "success": False,
        }
        try:
            attempt["return_value"] = str(callback())
            attempt["success"] = True
        except Exception as exc:
            attempt["error"] = str(exc)
        attempt["current_map_immediately_after_open"] = _get_current_map_info()
        attempt["map_match_immediately_after_open"] = _editor_map_paths_match(
            (attempt.get("current_map_immediately_after_open") or {}).get("world_path"),
            requested_map,
        )
        attempts.append(attempt)
        if attempt["success"]:
            return {
                "success": True,
                "requested_map": requested_map,
                "attempts": attempts,
                "final_current_map": attempt.get("current_map_immediately_after_open"),
            }
    return {
        "success": False,
        "requested_map": requested_map,
        "attempts": attempts,
        "final_current_map": _get_current_map_info(),
    }


def _load_renderer_helpers(renderer_script_path):
    original_argv = list(sys.argv)
    try:
        sys.argv = [str(renderer_script_path)]
        return runpy.run_path(str(renderer_script_path), run_name="bedlam360_bootstrap_helpers")
    finally:
        sys.argv = original_argv


class BootstrapController:
    def __init__(self, config, renderer_script_path, renderer_args, renderer_helpers):
        self.config = config
        self.renderer_script_path = Path(renderer_script_path).expanduser().resolve()
        self.renderer_args = list(renderer_args or [])
        self.renderer_helpers = renderer_helpers
        self.prefix = str(config.bootstrap_log_prefix)
        self.scene_root = _resolve_scene_root_from_renderer_args(self.renderer_args)
        self.preferred_editor_map = str(getattr(config, "preferred_editor_map", "") or "")
        self.configure_scene_root_paths = renderer_helpers.get("_configure_scene_root_paths")
        self.validate_loaded_usd_stage_matches_scene_root = renderer_helpers.get(
            "_validate_loaded_usd_stage_matches_scene_root"
        )
        self.base = renderer_helpers.get("base")
        if not callable(self.configure_scene_root_paths):
            raise RuntimeError("Renderer helper _configure_scene_root_paths is unavailable.")
        if not callable(self.validate_loaded_usd_stage_matches_scene_root):
            raise RuntimeError("Renderer helper _validate_loaded_usd_stage_matches_scene_root is unavailable.")
        if self.base is None:
            raise RuntimeError("Renderer helper module did not expose base.")
        self.tick_handle = None
        self.total_ticks = 0
        self.phase = "wait_asset_registry"
        self.phase_ticks = 0
        self.asset_registry_ready_seen = False
        self.wait_for_completion_attempted = False
        self.scene_binding_report = None
        self.expected_stage_path = None
        self.usd_bind_attempts = []
        self.usd_validation_report = None
        self.renderer_launched = False
        self.preferred_map_open_attempted = False

    def start(self):
        register = getattr(unreal, "register_slate_pre_tick_callback", None)
        if not callable(register):
            raise RuntimeError("unreal.register_slate_pre_tick_callback is unavailable in this Unreal session.")
        _log(
            self.prefix,
            "BOOTSTRAP_START",
            {
                "renderer_script": str(self.renderer_script_path),
                "renderer_args": self.renderer_args,
                "scene_root": str(self.scene_root),
                "preferred_editor_map": self.preferred_editor_map,
                "bootstrap_wait_ticks_after_asset_registry": int(
                    self.config.bootstrap_wait_ticks_after_asset_registry
                ),
                "bootstrap_wait_ticks_after_usd_load": int(self.config.bootstrap_wait_ticks_after_usd_load),
                "bootstrap_max_ticks": int(self.config.bootstrap_max_ticks),
            },
        )
        self.tick_handle = register(self.on_tick)

    def stop(self):
        if self.tick_handle is None:
            return
        unregister = getattr(unreal, "unregister_slate_pre_tick_callback", None)
        if callable(unregister):
            try:
                unregister(self.tick_handle)
            except Exception:
                pass
        self.tick_handle = None

    def on_tick(self, delta_time):
        del delta_time
        self.total_ticks += 1
        self.phase_ticks += 1
        try:
            if self.total_ticks > int(self.config.bootstrap_max_ticks):
                _fail(
                    self.prefix,
                    "BOOTSTRAP exceeded bootstrap_max_ticks before launching renderer.",
                    {
                        "phase": self.phase,
                        "total_ticks": int(self.total_ticks),
                        "max_ticks": int(self.config.bootstrap_max_ticks),
                    },
                )
            if self.phase == "wait_asset_registry":
                self._tick_wait_asset_registry()
                return
            if self.phase == "wait_post_asset_registry":
                self._tick_wait_post_asset_registry()
                return
            if self.phase == "usd_load_start":
                self._tick_usd_load_start()
                return
            if self.phase == "wait_post_usd":
                self._tick_wait_post_usd()
                return
            if self.phase == "launch_renderer":
                self._launch_renderer()
                return
        except Exception as exc:
            self.stop()
            _log(
                self.prefix,
                "BOOTSTRAP_ERROR",
                {
                    "phase": self.phase,
                    "total_ticks": int(self.total_ticks),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    def _advance(self, phase):
        self.phase = str(phase)
        self.phase_ticks = 0

    def _tick_wait_asset_registry(self):
        asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
        if asset_registry is None:
            _fail(self.prefix, "AssetRegistryHelpers.get_asset_registry() returned None.")
        is_loading_assets = True
        try:
            is_loading_assets = bool(asset_registry.is_loading_assets())
        except Exception as exc:
            _fail(self.prefix, "Asset registry readiness check failed.", {"error": str(exc)})
        if is_loading_assets:
            return
        if not self.wait_for_completion_attempted and hasattr(asset_registry, "wait_for_completion"):
            self.wait_for_completion_attempted = True
            try:
                asset_registry.wait_for_completion()
            except Exception as exc:
                _log(self.prefix, "BOOTSTRAP_ASSET_REGISTRY_WAIT_FOR_COMPLETION_SKIPPED", {"error": str(exc)})
        self.asset_registry_ready_seen = True
        _log(
            self.prefix,
            "BOOTSTRAP_ASSET_REGISTRY_READY",
            {
                "total_ticks": int(self.total_ticks),
                "wait_ticks_after_asset_registry": int(self.config.bootstrap_wait_ticks_after_asset_registry),
            },
        )
        self._advance("wait_post_asset_registry")

    def _tick_wait_post_asset_registry(self):
        target = int(self.config.bootstrap_wait_ticks_after_asset_registry)
        if self.phase_ticks < target:
            return
        self._advance("usd_load_start")

    def _tick_usd_load_start(self):
        current_map_info = _get_current_map_info()
        if self.preferred_editor_map and not _editor_map_paths_match(
            current_map_info.get("world_path"),
            self.preferred_editor_map,
        ):
            if not self.preferred_map_open_attempted:
                self.preferred_map_open_attempted = True
                open_report = _open_editor_map(self.preferred_editor_map)
                _log(
                    self.prefix,
                    "BOOTSTRAP_PREFERRED_EDITOR_MAP_OPEN",
                    {
                        "requested_map": self.preferred_editor_map,
                        "open_report": open_report,
                    },
                )
                return
            _fail(
                self.prefix,
                "Preferred editor map did not open before bootstrap USD validation.",
                {
                    "requested_map": self.preferred_editor_map,
                    "current_map": current_map_info,
                },
            )
        self.scene_binding_report = self.configure_scene_root_paths(self.scene_root)
        self.expected_stage_path = Path(self.scene_binding_report["usd_stage_path"]).expanduser().resolve()
        _log(
            self.prefix,
            "BOOTSTRAP_USD_LOAD_START",
            {
                "scene_root": str(self.scene_root),
                "preferred_editor_map": self.preferred_editor_map,
                "current_map": current_map_info,
                "scene_binding_report": self.scene_binding_report,
            },
        )
        actors = unreal.EditorLevelLibrary.get_all_level_actors()
        stage_actors = list(self.base._find_usd_stage_actors(actors) or [])
        if not stage_actors:
            _fail(
                self.prefix,
                "No UsdStageActor found in the current level during bootstrap.",
                {
                    "scene_root": str(self.scene_root),
                    "expected_stage_path": str(self.expected_stage_path),
                },
            )
        try:
            self.usd_validation_report = self.validate_loaded_usd_stage_matches_scene_root(self.expected_stage_path)
            _log(
                self.prefix,
                "BOOTSTRAP_USD_LOAD_DONE",
                {
                    "validation_report": self.usd_validation_report,
                    "bind_attempts": self.usd_bind_attempts,
                },
            )
            self._advance("wait_post_usd")
            return
        except Exception as initial_exc:
            self.usd_bind_attempts.append({"phase": "initial_validation_failed", "error": str(initial_exc)})
        for actor in stage_actors:
            actor_report = {
                "actor_label": None,
                "set_root_layer_success": False,
                "fallback_file_path_success": False,
                "errors": [],
            }
            try:
                actor_report["actor_label"] = str(actor.get_actor_label())
            except Exception:
                actor_report["actor_label"] = str(actor)
            try:
                set_root_layer = getattr(actor, "set_root_layer", None)
                if callable(set_root_layer):
                    set_root_layer(str(self.expected_stage_path))
                    actor_report["set_root_layer_success"] = True
            except Exception as exc:
                actor_report["errors"].append(f"set_root_layer: {exc}")
            if not actor_report["set_root_layer_success"]:
                try:
                    file_path = unreal.FilePath()
                    file_path.file_path = str(self.expected_stage_path)
                    actor.set_editor_property("root_layer", file_path)
                    actor_report["fallback_file_path_success"] = True
                except Exception as exc:
                    actor_report["errors"].append(f"set_editor_property(root_layer): {exc}")
            self.usd_bind_attempts.append(actor_report)
        try:
            self.usd_validation_report = self.validate_loaded_usd_stage_matches_scene_root(self.expected_stage_path)
        except Exception as exc:
            _fail(
                self.prefix,
                "USD validation failed after bootstrap binding attempt.",
                {
                    "scene_root": str(self.scene_root),
                    "expected_stage_path": str(self.expected_stage_path),
                    "bind_attempts": self.usd_bind_attempts,
                    "error": str(exc),
                },
            )
        _log(
            self.prefix,
            "BOOTSTRAP_USD_LOAD_DONE",
            {
                "validation_report": self.usd_validation_report,
                "bind_attempts": self.usd_bind_attempts,
            },
        )
        self._advance("wait_post_usd")

    def _tick_wait_post_usd(self):
        target = int(self.config.bootstrap_wait_ticks_after_usd_load)
        _log(
            self.prefix,
            "BOOTSTRAP_WAITING_POST_USD_TICKS",
            {
                "current": int(self.phase_ticks),
                "target": target,
                "total_ticks": int(self.total_ticks),
            },
        )
        if self.phase_ticks < target:
            return
        self._advance("launch_renderer")

    def _launch_renderer(self):
        if self.renderer_launched:
            return
        self.renderer_launched = True
        _log(
            self.prefix,
            "BOOTSTRAP_RENDERER_LAUNCH",
            {
                "renderer_script": str(self.renderer_script_path),
                "renderer_args": self.renderer_args,
                "scene_root": str(self.scene_root),
                "validation_report": self.usd_validation_report,
            },
        )
        self.stop()
        original_argv = list(sys.argv)
        try:
            sys.argv = [str(self.renderer_script_path)] + list(self.renderer_args)
            runpy.run_path(str(self.renderer_script_path), run_name="__main__")
        finally:
            sys.argv = original_argv
        _log(
            self.prefix,
            "BOOTSTRAP_DONE",
            {
                "renderer_script": str(self.renderer_script_path),
                "scene_root": str(self.scene_root),
            },
        )


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Bootstrap BEDLAM360 renderer via Unreal editor ticks.")
    parser.add_argument("--renderer-script", default=DEFAULT_RENDERER_SCRIPT)
    parser.add_argument("--bootstrap-helper-renderer-script", default=None)
    parser.add_argument("--preferred-editor-map", default=None)
    parser.add_argument("--renderer-args-json", default=None)
    parser.add_argument("--bootstrap-wait-ticks-after-asset-registry", type=int, default=30)
    parser.add_argument("--bootstrap-wait-ticks-after-usd-load", type=int, default=120)
    parser.add_argument("--bootstrap-max-ticks", type=int, default=3000)
    parser.add_argument("--bootstrap-log-prefix", default="BEDLAM360_BOOTSTRAP")
    return parser


def main():
    bootstrap_argv, passthrough_argv = _split_bootstrap_and_renderer_argv(sys.argv[1:])
    parser = _build_arg_parser()
    config = parser.parse_args(bootstrap_argv)
    renderer_script_path = Path(config.renderer_script).expanduser().resolve()
    if not renderer_script_path.exists():
        raise RuntimeError(f"Renderer script does not exist: {renderer_script_path}")
    helper_renderer_script_path = (
        Path(config.bootstrap_helper_renderer_script).expanduser().resolve()
        if config.bootstrap_helper_renderer_script not in (None, "")
        else renderer_script_path
    )
    if not helper_renderer_script_path.exists():
        raise RuntimeError(f"Bootstrap helper renderer script does not exist: {helper_renderer_script_path}")
    renderer_args = _normalize_renderer_args(config.renderer_args_json, passthrough_argv)
    if not renderer_args:
        raise RuntimeError(
            "No renderer arguments supplied. Provide --renderer-args-json or pass renderer args after --"
        )
    renderer_helpers = _load_renderer_helpers(helper_renderer_script_path)
    controller = BootstrapController(
        config=config,
        renderer_script_path=renderer_script_path,
        renderer_args=renderer_args,
        renderer_helpers=renderer_helpers,
    )
    controller.start()


if __name__ == "__main__":
    main()
