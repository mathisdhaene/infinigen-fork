import argparse
import csv
import json
import os
import shlex
import shutil
import threading
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


DEFAULT_UNREAL_EDITOR = Path(
    "/media/mathis/PANO/Unreal/UnrealEngine-5.3.2/Engine/Binaries/Linux/UnrealEditor"
)
DEFAULT_UPROJECT = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/BEDLAM360.uproject")
DEFAULT_RENDERER_SCRIPT = Path(
    "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_selected_infinigen_bedlam_erp.py"
)
DEFAULT_BOOTSTRAP_RENDER_LAUNCHER_SCRIPT = Path(
    "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/bootstrap_bedlam360_render.py"
)
DEFAULT_RUNS_ROOT = Path(
    "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_infinigen_bridge/runs"
)
DEFAULT_BATCH_REPORT_ROOT = Path(
    "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_infinigen_bridge/batch_miniscene_runs"
)
DEFAULT_RENDER_OUTPUT_PROFILE = "dataset_rgb_fast"
DEFAULT_RGB_TONEMAP_MODE = "ldr_passthrough"
DEFAULT_CHUNK_SIZE = 2
DEFAULT_MAX_RETRIES_PER_CLIP = 3
DEFAULT_PREVIEW_FPS = 24
DEFAULT_WARMUP_FRAME_START = 0
DEFAULT_WARMUP_FRAME_END = 30
DEFAULT_EDITOR_MAP = "/Game/BEDLAM360_Test.BEDLAM360_Test"
DEFAULT_PRELOADED_WORLD_MIN_ACTOR_COUNT = 20
DEFAULT_PRELOADED_WORLD_MIN_COMPONENT_COUNT = 20
DEFAULT_PRELOADED_WORLD_MIN_MATERIAL_SLOT_COUNT = 20
DEFAULT_USD_STAGE_POST_BIND_WAIT_SECONDS = 10.0
DEFAULT_POST_WARMUP_DELAY_SECONDS = 5.0
DEFAULT_UNREAL_STARTUP_WAIT_SECONDS = 20.0
DEFAULT_USD_MATERIAL_READINESS_TIMEOUT_SECONDS = 120.0
DEFAULT_USD_MATERIAL_READINESS_POLL_SECONDS = 5.0
DEFAULT_USD_MATERIAL_MAX_FALLBACK_RATIO = 0.05
DEFAULT_USD_MATERIAL_STABLE_POLLS = 2
DEFAULT_PROBE_LAUNCH_COUNT = 20
DEFAULT_OPEN_MAP_ONLY_SLEEP_SECONDS = 600.0
DEFAULT_RESOURCE_MONITOR_INTERVAL_SECONDS = 5.0
DEFAULT_RESOURCE_MONITOR_LOG_NAME = "resource_monitor.csv"
LAUNCHER_MARKERS = [
    "RESILIENT_LAUNCHER_PYTHON_ENTERED",
    "RESILIENT_LAUNCHER_IMPORTS_DONE",
    "BOOTSTRAP_START",
    "BOOTSTRAP_ASSET_REGISTRY_READY",
    "BOOTSTRAP_USD_LOAD_START",
    "BOOTSTRAP_USD_LOAD_DONE",
    "BOOTSTRAP_RENDERER_LAUNCH",
    "BOOTSTRAP_DONE",
    "USD_STAGE_BOOTSTRAP_START",
    "USD_STAGE_BOOTSTRAP_FINAL",
    "USD_MATERIAL_READINESS_START",
    "USD_MATERIAL_READINESS_FINAL",
    "SESSION_WARMUP_START",
    "CLIP_RENDER_START",
    "CLIP_RENDER_COMPLETE",
    "CHUNK_COMPLETE",
]
PROBE_MARKERS = [
    "PYTHON_PROBE_START",
    "PYTHON_PROBE_IMPORT_OK",
    "PYTHON_PROBE_END",
]


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def _normalize_path(path_value):
    return str(Path(path_value).expanduser().resolve())


def _load_manifest(manifest_path: Path):
    payload = _read_json(manifest_path)
    miniscenes = list(payload.get("miniscenes", []))
    return payload, miniscenes


def _clip_ids_in_order(miniscenes):
    return [str(scene.get("miniscene_id")) for scene in miniscenes if scene.get("miniscene_id")]


def _chunk_rows(rows, chunk_size):
    chunk_size = max(int(chunk_size), 1)
    return [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]


def _build_warmup_manifest(official_manifest_payload, warmup_clip_id: str):
    miniscenes = list(official_manifest_payload.get("miniscenes", []))
    selected_scene = None
    for scene in miniscenes:
        if str(scene.get("miniscene_id")) == str(warmup_clip_id):
            selected_scene = scene
            break
    if selected_scene is None:
        raise RuntimeError(f"Warm-up clip id not found in manifest: {warmup_clip_id}")
    warmup_manifest = dict(official_manifest_payload)
    warmup_manifest["miniscenes"] = [selected_scene]
    warmup_manifest["total_selected_miniscenes"] = 1
    warmup_manifest["selected_single_human_count"] = 0
    human_count = int(selected_scene.get("human_count", len(selected_scene.get("humans", []))))
    warmup_manifest["selected_multi_human_count"] = 1 if human_count >= 2 else 0
    warmup_manifest["selected_two_human_count"] = 1 if human_count == 2 else 0
    warmup_manifest["selected_three_human_count"] = 1 if human_count == 3 else 0
    warmup_manifest["selected_four_human_count"] = 1 if human_count == 4 else 0
    warmup_manifest["selected_human_count_distribution"] = {str(human_count): 1}
    selected_per_room = {}
    for room_name, room_row in (official_manifest_payload.get("selected_per_room") or {}).items():
        if room_name == selected_scene.get("room"):
            selected_per_room[room_name] = {
                "single_human": 0,
                "multi_human": 1 if human_count >= 2 else 0,
                "human_count_distribution": {str(human_count): 1} if human_count >= 2 else {},
                "total": 1,
            }
        else:
            selected_per_room[room_name] = {
                "single_human": 0,
                "multi_human": 0,
                "human_count_distribution": {},
                "total": 0,
            }
    warmup_manifest["selected_per_room"] = selected_per_room
    warmup_manifest["rooms_without_renderable_miniscenes"] = sorted(
        room_name for room_name, row in selected_per_room.items() if int(row.get("total", 0)) == 0
    )
    warmup_manifest["warmup_only_manifest"] = True
    warmup_manifest["warmup_clip_id"] = str(warmup_clip_id)
    warmup_manifest["warmup_is_not_dataset_completion"] = True
    warmup_manifest["warmup_source_manifest_path"] = official_manifest_payload.get("selected_miniscene", {}).get("manifest_path")
    return warmup_manifest


def _iter_run_manifests(runs_root: Path):
    if not runs_root.exists():
        return
    for run_dir in sorted((row for row in runs_root.iterdir() if row.is_dir()), key=lambda p: p.stat().st_mtime):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            yield run_dir, manifest_path


def _count_rgb_frames(images_dir: Path):
    if not images_dir.exists():
        return 0
    return len(sorted(images_dir.glob("*_rgb.png")))


def _build_ffmpeg_concat_file(frame_paths, concat_path: Path):
    concat_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for frame_path in frame_paths:
        escaped = str(frame_path).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_preview_rgb_mp4(run_dir: Path, manifest_payload, force=False):
    range_results = list(manifest_payload.get("range_results") or [])
    if not range_results:
        return {
            "attempted": False,
            "success": False,
            "reason": "missing_range_results",
        }
    range_row = range_results[0]
    images_dir = Path(range_row.get("images_dir") or "")
    if not images_dir.exists():
        return {
            "attempted": False,
            "success": False,
            "reason": "missing_images_dir",
            "images_dir": str(images_dir),
        }
    frame_paths = sorted(images_dir.glob("*_rgb.png"))
    if not frame_paths:
        return {
            "attempted": False,
            "success": False,
            "reason": "missing_rgb_frames",
            "images_dir": str(images_dir),
        }
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return {
            "attempted": False,
            "success": False,
            "reason": "ffmpeg_not_found",
        }
    preview_dir = run_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    output_path = preview_dir / "preview_rgb.mp4"
    if output_path.exists() and not force:
        return {
            "attempted": False,
            "success": True,
            "reason": "already_exists",
            "preview_rgb_mp4_path": str(output_path),
            "preview_frame_count": len(frame_paths),
        }
    fps = int(manifest_payload.get("preview_fps") or DEFAULT_PREVIEW_FPS)
    with tempfile.TemporaryDirectory(prefix="bedlam360_preview_concat_") as tmp_dir:
        concat_path = Path(tmp_dir) / "rgb_frames.txt"
        _build_ffmpeg_concat_file(frame_paths, concat_path)
        command = [
            ffmpeg_bin,
            "-y",
            "-r",
            str(fps),
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        started = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True)
        duration = float(time.perf_counter() - started)
    return {
        "attempted": True,
        "success": bool(result.returncode == 0 and output_path.exists()),
        "returncode": int(result.returncode),
        "command": _shell_join(command),
        "preview_rgb_mp4_path": str(output_path),
        "preview_frame_count": len(frame_paths),
        "preview_fps": fps,
        "duration_seconds": duration,
        "stderr_tail": "\n".join((result.stderr or "").splitlines()[-20:]),
        "stdout_tail": "\n".join((result.stdout or "").splitlines()[-20:]),
    }


def _run_matches_target(manifest_payload, manifest_path: Path, miniscene_id: str):
    if bool(manifest_payload.get("warmup_only_manifest")):
        return False
    selected = manifest_payload.get("selected_miniscene") or {}
    run_manifest_path = selected.get("manifest_path")
    run_miniscene_id = selected.get("miniscene_id")
    if not run_manifest_path or not run_miniscene_id:
        return False
    return (
        _normalize_path(run_manifest_path) == _normalize_path(manifest_path)
        and str(run_miniscene_id) == str(miniscene_id)
    )


def _classify_run(manifest_payload, run_dir: Path, requested_frame_start: int, requested_frame_end: int):
    range_results = list(manifest_payload.get("range_results") or [])
    if not range_results:
        return {
            "state": "partial",
            "reason": "missing_range_results",
            "run_dir": str(run_dir),
        }
    range_row = range_results[0]
    actual_start = range_row.get("frame_start")
    actual_end = range_row.get("frame_end")
    images_dir = Path(range_row.get("images_dir") or "")
    rgb_frame_count = _count_rgb_frames(images_dir)
    requested_frame_count = int(requested_frame_end) - int(requested_frame_start) + 1
    actual_frame_count = None
    if actual_start is not None and actual_end is not None:
        actual_frame_count = int(actual_end) - int(actual_start) + 1
    preview_rgb_mp4_path = manifest_payload.get("preview_rgb_mp4_path") or manifest_payload.get("preview_rgb_mp4")
    preview_rgb_exists = bool(preview_rgb_mp4_path and Path(preview_rgb_mp4_path).exists())
    frame_range_ok = (
        actual_start is not None
        and actual_end is not None
        and int(actual_start) <= int(requested_frame_start)
        and int(actual_end) >= int(requested_frame_end)
    )
    rgb_count_ok = rgb_frame_count >= requested_frame_count
    if frame_range_ok and rgb_count_ok and preview_rgb_exists:
        state = "completed"
    elif frame_range_ok and rgb_count_ok:
        state = "missing_preview_only"
    elif rgb_frame_count > 0:
        state = "partial"
    else:
        state = "failed"
    return {
        "state": state,
        "run_dir": str(run_dir),
        "actual_frame_start": actual_start,
        "actual_frame_end": actual_end,
        "actual_frame_count": actual_frame_count,
        "requested_frame_start": int(requested_frame_start),
        "requested_frame_end": int(requested_frame_end),
        "requested_frame_count": int(requested_frame_count),
        "images_dir": str(images_dir),
        "rgb_frame_count": int(rgb_frame_count),
        "preview_rgb_mp4_path": preview_rgb_mp4_path,
        "preview_rgb_exists": bool(preview_rgb_exists),
        "range_tag": range_row.get("range_tag"),
        "artifact_summary": range_row.get("artifact_summary"),
    }


def _scan_clip_statuses(runs_root: Path, manifest_path: Path, clip_ids, requested_frame_start: int, requested_frame_end: int):
    best_by_clip = {}
    run_rows_by_clip = defaultdict(list)
    ranking = {"completed": 4, "missing_preview_only": 3, "partial": 2, "failed": 1}
    clip_id_set = set(clip_ids)
    for run_dir, run_manifest_path in _iter_run_manifests(runs_root):
        try:
            payload = _read_json(run_manifest_path)
        except Exception:
            continue
        selected = payload.get("selected_miniscene") or {}
        run_clip_id = selected.get("miniscene_id")
        if run_clip_id not in clip_id_set:
            continue
        if not _run_matches_target(payload, manifest_path, run_clip_id):
            continue
        row = _classify_run(payload, run_dir, requested_frame_start, requested_frame_end)
        row["manifest_json_path"] = str(run_manifest_path)
        row["run_id"] = payload.get("run_id")
        row["miniscene_id"] = run_clip_id
        run_rows_by_clip[run_clip_id].append(row)
        current = best_by_clip.get(run_clip_id)
        if current is None:
            best_by_clip[run_clip_id] = row
            continue
        current_rank = ranking.get(current["state"], 0)
        row_rank = ranking.get(row["state"], 0)
        if row_rank > current_rank:
            best_by_clip[run_clip_id] = row
        elif row_rank == current_rank:
            current_mtime = Path(current["manifest_json_path"]).stat().st_mtime
            row_mtime = run_manifest_path.stat().st_mtime
            if row_mtime > current_mtime:
                best_by_clip[run_clip_id] = row
    statuses = {}
    for clip_id in clip_ids:
        best_row = best_by_clip.get(
            clip_id,
            {
                "state": "remaining",
                "miniscene_id": clip_id,
            },
        )
        statuses[clip_id] = dict(best_row)
        statuses[clip_id]["all_matching_runs"] = [dict(row) for row in run_rows_by_clip.get(clip_id, [])]
    return statuses


def _tail_lines(path: Path, line_count=40):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return lines[-int(line_count) :]


RESOURCE_MONITOR_FIELDNAMES = [
    "timestamp_unix",
    "elapsed_seconds",
    "chunk_id",
    "unreal_pid",
    "unreal_alive",
    "cpu_percent",
    "rss_bytes",
    "vms_bytes",
    "system_ram_used_bytes",
    "system_ram_available_bytes",
    "system_swap_used_bytes",
    "gpu_index",
    "gpu_name",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_util_percent",
    "gpu_temperature_c",
    "gpu_power_watts",
    "unreal_gpu_memory_used_mb",
    "monitor_error",
]


def _safe_float(value):
    try:
        if value in (None, "", "N/A", "[Not Supported]"):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value):
    try:
        if value in (None, "", "N/A", "[Not Supported]"):
            return None
        return int(float(value))
    except Exception:
        return None


def _read_proc_status_memory_bytes(pid: int):
    rss_bytes = None
    vms_bytes = None
    status_path = Path(f"/proc/{int(pid)}/status")
    if not status_path.exists():
        return rss_bytes, vms_bytes
    try:
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    rss_bytes = int(parts[1]) * 1024
            elif line.startswith("VmSize:"):
                parts = line.split()
                if len(parts) >= 2:
                    vms_bytes = int(parts[1]) * 1024
    except Exception:
        pass
    return rss_bytes, vms_bytes


def _sample_process_and_system_memory(pid: int, ps_process=None):
    row = {
        "cpu_percent": None,
        "rss_bytes": None,
        "vms_bytes": None,
        "system_ram_used_bytes": None,
        "system_ram_available_bytes": None,
        "system_swap_used_bytes": None,
        "monitor_error": None,
    }
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None
    if psutil is not None:
        try:
            proc = ps_process if ps_process is not None else psutil.Process(int(pid))
            mem = proc.memory_info()
            row["rss_bytes"] = int(mem.rss)
            row["vms_bytes"] = int(mem.vms)
            try:
                row["cpu_percent"] = float(proc.cpu_percent(interval=None))
            except Exception:
                row["cpu_percent"] = None
            vm = psutil.virtual_memory()
            row["system_ram_used_bytes"] = int(vm.used)
            row["system_ram_available_bytes"] = int(vm.available)
            sm = psutil.swap_memory()
            row["system_swap_used_bytes"] = int(sm.used)
            return row, proc
        except Exception as exc:
            row["monitor_error"] = f"psutil:{exc}"
    rss_bytes, vms_bytes = _read_proc_status_memory_bytes(pid)
    row["rss_bytes"] = rss_bytes
    row["vms_bytes"] = vms_bytes
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").splitlines()
        values = {}
        for line in meminfo:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            if parts:
                values[key] = int(parts[0]) * 1024
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if total is not None and available is not None:
            row["system_ram_available_bytes"] = int(available)
            row["system_ram_used_bytes"] = int(total - available)
        swap_total = values.get("SwapTotal")
        swap_free = values.get("SwapFree")
        if swap_total is not None and swap_free is not None:
            row["system_swap_used_bytes"] = int(swap_total - swap_free)
    except Exception as exc:
        if not row.get("monitor_error"):
            row["monitor_error"] = f"proc_meminfo:{exc}"
    return row, ps_process


def _run_nvidia_smi_query(query: str):
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _run_nvidia_smi_compute_apps():
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _sample_gpu_rows(unreal_pid: int):
    gpu_rows = []
    monitor_error = None
    app_rows = []
    try:
        lines = _run_nvidia_smi_query("index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw")
        for line in lines:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                continue
            gpu_rows.append({
                "gpu_index": _safe_int(parts[0]),
                "gpu_name": parts[1],
                "gpu_memory_used_mb": _safe_float(parts[2]),
                "gpu_memory_total_mb": _safe_float(parts[3]),
                "gpu_util_percent": _safe_float(parts[4]),
                "gpu_temperature_c": _safe_float(parts[5]),
                "gpu_power_watts": _safe_float(parts[6]),
                "unreal_gpu_memory_used_mb": None,
            })
    except Exception as exc:
        monitor_error = f"nvidia_smi_gpu:{exc}"
    try:
        lines = _run_nvidia_smi_compute_apps()
        for line in lines:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            pid = _safe_int(parts[0])
            if pid == int(unreal_pid):
                app_rows.append({
                    "pid": pid,
                    "process_name": parts[1],
                    "gpu_uuid": parts[2],
                    "used_memory_mb": _safe_float(parts[3]),
                })
    except Exception as exc:
        monitor_error = monitor_error or f"nvidia_smi_compute:{exc}"
    unreal_gpu_used = None
    if app_rows:
        unreal_gpu_used = float(sum(float(row.get("used_memory_mb") or 0.0) for row in app_rows))
    for row in gpu_rows:
        row["unreal_gpu_memory_used_mb"] = unreal_gpu_used
    return gpu_rows or [{
        "gpu_index": None,
        "gpu_name": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_util_percent": None,
        "gpu_temperature_c": None,
        "gpu_power_watts": None,
        "unreal_gpu_memory_used_mb": unreal_gpu_used,
    }], monitor_error


def _resource_monitor_loop(process, csv_path: Path, chunk_id: str, interval_seconds: float, stop_event):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ps_process = None
    try:
        import psutil  # type: ignore
        ps_process = psutil.Process(int(process.pid))
        try:
            ps_process.cpu_percent(interval=None)
        except Exception:
            pass
    except Exception:
        ps_process = None
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESOURCE_MONITOR_FIELDNAMES)
        writer.writeheader()
        while not stop_event.is_set():
            now = time.time()
            unreal_alive = process.poll() is None
            base_row = {
                "timestamp_unix": now,
                "elapsed_seconds": float(now - started),
                "chunk_id": str(chunk_id),
                "unreal_pid": int(process.pid),
                "unreal_alive": bool(unreal_alive),
            }
            mem_row, ps_process = _sample_process_and_system_memory(process.pid, ps_process=ps_process)
            gpu_rows, gpu_error = _sample_gpu_rows(process.pid)
            if not gpu_rows:
                gpu_rows = [{}]
            for gpu_row in gpu_rows:
                row = dict(base_row)
                row.update(mem_row)
                row.update(gpu_row)
                if gpu_error and not row.get("monitor_error"):
                    row["monitor_error"] = gpu_error
                writer.writerow(row)
            handle.flush()
            if not unreal_alive:
                break
            stop_event.wait(max(float(interval_seconds), 0.1))


def _summarize_resource_csv(csv_path: Path):
    summary = {
        "resource_monitor_csv_path": str(csv_path),
        "sample_count": 0,
        "max_unreal_rss_bytes": None,
        "max_system_ram_used_bytes": None,
        "max_system_swap_used_bytes": None,
        "max_gpu_memory_used_mb": None,
        "max_unreal_gpu_memory_used_mb": None,
        "start_timestamp_unix": None,
        "end_timestamp_unix": None,
        "elapsed_seconds_end": None,
        "monitor_errors": [],
    }
    if not csv_path.exists():
        summary["missing"] = True
        return summary
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        summary["read_error"] = str(exc)
        return summary
    if not rows:
        return summary
    summary["sample_count"] = len(rows)
    summary["start_timestamp_unix"] = _safe_float(rows[0].get("timestamp_unix"))
    summary["end_timestamp_unix"] = _safe_float(rows[-1].get("timestamp_unix"))
    summary["elapsed_seconds_end"] = _safe_float(rows[-1].get("elapsed_seconds"))
    for field in ("rss_bytes", "system_ram_used_bytes", "system_swap_used_bytes", "gpu_memory_used_mb", "unreal_gpu_memory_used_mb"):
        values = [_safe_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            key = {
                "rss_bytes": "max_unreal_rss_bytes",
                "system_ram_used_bytes": "max_system_ram_used_bytes",
                "system_swap_used_bytes": "max_system_swap_used_bytes",
                "gpu_memory_used_mb": "max_gpu_memory_used_mb",
                "unreal_gpu_memory_used_mb": "max_unreal_gpu_memory_used_mb",
            }[field]
            summary[key] = max(values)
    errors = sorted({str(row.get("monitor_error")) for row in rows if row.get("monitor_error") not in (None, "", "None")})
    summary["monitor_errors"] = errors
    return summary


def _resource_summary_rows(resource_logs_dir: Path):
    rows = []
    for csv_path in sorted(resource_logs_dir.glob("*_resource_monitor.csv")):
        summary = _summarize_resource_csv(csv_path)
        row = {
            "chunk_id": csv_path.stem.replace("_resource_monitor", ""),
            "resource_monitor_csv_path": str(csv_path),
            "sample_count": summary.get("sample_count"),
            "start_timestamp_unix": summary.get("start_timestamp_unix"),
            "end_timestamp_unix": summary.get("end_timestamp_unix"),
            "elapsed_seconds_end": summary.get("elapsed_seconds_end"),
            "max_unreal_rss_bytes": summary.get("max_unreal_rss_bytes"),
            "max_system_ram_used_bytes": summary.get("max_system_ram_used_bytes"),
            "max_system_swap_used_bytes": summary.get("max_system_swap_used_bytes"),
            "max_gpu_memory_used_mb": summary.get("max_gpu_memory_used_mb"),
            "max_unreal_gpu_memory_used_mb": summary.get("max_unreal_gpu_memory_used_mb"),
            "monitor_errors": "|".join(summary.get("monitor_errors") or []),
        }
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                samples = list(csv.DictReader(handle))
        except Exception:
            samples = []
        for metric in ("rss_bytes", "system_ram_used_bytes", "system_swap_used_bytes", "gpu_memory_used_mb", "unreal_gpu_memory_used_mb"):
            vals = [_safe_float(sample.get(metric)) for sample in samples]
            vals = [v for v in vals if v is not None]
            row[f"{metric}_mean"] = (sum(vals) / len(vals)) if vals else None
            row[f"{metric}_start"] = vals[0] if vals else None
            row[f"{metric}_end"] = vals[-1] if vals else None
            row[f"{metric}_delta"] = ((vals[-1] - vals[0]) if len(vals) >= 2 else 0.0) if vals else None
        rows.append(row)
    return rows


def _write_resource_summary(resource_logs_dir: Path):
    rows = _resource_summary_rows(resource_logs_dir)
    summary_json_path = resource_logs_dir / "resource_summary.json"
    summary_csv_path = resource_logs_dir / "resource_summary.csv"
    payload = {
        "resource_logs_dir": str(resource_logs_dir),
        "chunk_count": len(rows),
        "rows": rows,
        "generated_at_unix": time.time(),
    }
    _write_json(summary_json_path, payload)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["chunk_id", "resource_monitor_csv_path"]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return {
        "resource_logs_dir": str(resource_logs_dir),
        "resource_summary_json_path": str(summary_json_path),
        "resource_summary_csv_path": str(summary_csv_path),
        "chunk_count": len(rows),
    }


def _parse_launcher_stage_diagnostics(log_path: Path):
    found_markers = []
    line_numbers = {}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return {
            "markers_found": [],
            "last_seen_marker": None,
            "failure_stage": "unknown_no_run_output",
            "log_tail": [f"<log_read_failed: {exc}>"],
        }
    for line_index, line in enumerate(lines, start=1):
        for marker in LAUNCHER_MARKERS:
            if marker in line and marker not in line_numbers:
                line_numbers[marker] = line_index
                found_markers.append(marker)
    last_seen_marker = found_markers[-1] if found_markers else None
    found_set = set(found_markers)
    if "RESILIENT_LAUNCHER_PYTHON_ENTERED" not in found_set:
        failure_stage = "startup_before_python"
    elif "RESILIENT_LAUNCHER_IMPORTS_DONE" not in found_set:
        failure_stage = "python_early_startup_failure"
    elif "BOOTSTRAP_START" in found_set and "BOOTSTRAP_RENDERER_LAUNCH" not in found_set:
        failure_stage = "bootstrap_failure"
    elif "USD_STAGE_BOOTSTRAP_START" in found_set and "USD_STAGE_BOOTSTRAP_FINAL" not in found_set:
        failure_stage = "usd_bootstrap_failure"
    elif "USD_MATERIAL_READINESS_START" in found_set and "USD_MATERIAL_READINESS_FINAL" not in found_set:
        failure_stage = "usd_material_readiness_failure"
    elif "USD_STAGE_BOOTSTRAP_FINAL" in found_set and "SESSION_WARMUP_START" not in found_set:
        failure_stage = "warmup_failure"
    elif "CLIP_RENDER_START" in found_set and "CHUNK_COMPLETE" not in found_set:
        failure_stage = "render_failure"
    else:
        failure_stage = "unknown_no_run_output"
    failure_reason = None
    joined_tail = "\n".join(lines[-200:])
    if "USD_MATERIAL_READINESS_TIMEOUT" in joined_tail:
        failure_reason = "USD_MATERIAL_READINESS_TIMEOUT"
    return {
        "markers_found": found_markers,
        "marker_line_numbers": line_numbers,
        "last_seen_marker": last_seen_marker,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "log_tail": lines[-60:],
    }


def _parse_probe_diagnostics(log_path: Path):
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return {
            "markers_found": [],
            "failure_stage": "startup_before_python",
            "log_tail": [f"<log_read_failed: {exc}>"],
        }
    found_markers = []
    for line in lines:
        for marker in PROBE_MARKERS:
            if marker in line and marker not in found_markers:
                found_markers.append(marker)
    found_set = set(found_markers)
    if "PYTHON_PROBE_START" not in found_set:
        failure_stage = "startup_before_python"
    elif "PYTHON_PROBE_IMPORT_OK" not in found_set:
        failure_stage = "python_early_startup_failure"
    elif "PYTHON_PROBE_END" not in found_set:
        failure_stage = "python_probe_incomplete"
    else:
        failure_stage = "probe_success"
    return {
        "markers_found": found_markers,
        "last_seen_marker": found_markers[-1] if found_markers else None,
        "failure_stage": failure_stage,
        "log_tail": lines[-60:],
    }


def _latest_attempt_diagnostics_for_clip(launch_history, clip_id):
    for launch_row in reversed(launch_history):
        if clip_id in set(launch_row.get("chunk_clip_ids") or []):
            return {
                "failure_stage": launch_row.get("failure_stage"),
                "failure_reason": launch_row.get("failure_reason"),
                "last_seen_marker": launch_row.get("last_seen_marker"),
                "unreal_exit_code": launch_row.get("returncode"),
                "log_tail": launch_row.get("log_tail"),
                "attempt_index": launch_row.get("attempt_index"),
                "chunk_index": launch_row.get("chunk_index"),
                "launcher_markers_found": launch_row.get("launcher_markers_found"),
            }
    return {}


def _apply_attempt_failure_overrides(statuses, attempt_counts, launch_history):
    adjusted = {}
    for clip_id, row in statuses.items():
        row = dict(row)
        attempted = int(attempt_counts.get(clip_id, 0))
        if row.get("state") == "remaining" and attempted > 0:
            row["state"] = "failed"
            row["failure_reason"] = "attempted_but_no_matching_run_output_detected"
            row["attempted_render_sessions"] = attempted
            row.update(_latest_attempt_diagnostics_for_clip(launch_history, clip_id))
        adjusted[clip_id] = row
    return adjusted


def _build_unreal_chunk_launcher(
    launcher_path: Path,
    renderer_script: Path,
    scene_root: Path,
    manifest_path: Path,
    warmup_manifest_path: Path,
    runs_root: Path,
    clip_ids,
    frame_start: int,
    frame_end: int,
    render_output_profile: str,
    rgb_tonemap_mode: str,
    preferred_editor_map: str,
    spawn_usd_stage_if_missing: bool,
    enable_session_warmup: bool,
    warmup_frame_start: int,
    warmup_frame_end: int,
    usd_stage_post_bind_wait_seconds: float,
    post_warmup_delay_seconds: float,
    unreal_startup_wait_seconds: float,
    render_warmup_frame_count: int,
    discard_warmup_frames: bool,
    probe_frame_before_render: bool,
    reject_beige_probe: bool,
    usd_material_readiness_timeout_seconds: float,
    usd_material_readiness_poll_seconds: float,
    allow_usd_material_readiness_timeout: bool,
    preloaded_world_min_actor_count: int,
    preloaded_world_min_component_count: int,
    preloaded_world_min_material_slot_count: int,
    pause_after_spawn_before_render: bool,
    open_map_only: bool,
    open_map_only_sleep_seconds: float,
    open_map_only_no_quit: bool,
):
    payload = {
        "renderer_script": str(renderer_script),
        "scene_root": str(scene_root),
        "manifest_path": str(manifest_path),
        "warmup_manifest_path": None if warmup_manifest_path is None else str(warmup_manifest_path),
        "runs_root": str(runs_root),
        "preferred_editor_map": str(preferred_editor_map) if preferred_editor_map else None,
        "spawn_usd_stage_if_missing": bool(spawn_usd_stage_if_missing),
        "clip_ids": list(clip_ids),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "render_output_profile": str(render_output_profile),
        "rgb_tonemap_mode": str(rgb_tonemap_mode),
        "enable_session_warmup": bool(enable_session_warmup),
        "warmup_frame_start": int(warmup_frame_start),
        "warmup_frame_end": int(warmup_frame_end),
        "usd_stage_post_bind_wait_seconds": float(usd_stage_post_bind_wait_seconds),
        "post_warmup_delay_seconds": float(post_warmup_delay_seconds),
        "unreal_startup_wait_seconds": float(unreal_startup_wait_seconds),
        "render_warmup_frame_count": int(render_warmup_frame_count),
        "discard_warmup_frames": bool(discard_warmup_frames),
        "probe_frame_before_render": bool(probe_frame_before_render),
        "reject_beige_probe": bool(reject_beige_probe),
        "usd_material_readiness_timeout_seconds": float(usd_material_readiness_timeout_seconds),
        "usd_material_readiness_poll_seconds": float(usd_material_readiness_poll_seconds),
        "allow_usd_material_readiness_timeout": bool(allow_usd_material_readiness_timeout),
        "preloaded_world_min_actor_count": int(preloaded_world_min_actor_count),
        "preloaded_world_min_component_count": int(preloaded_world_min_component_count),
        "preloaded_world_min_material_slot_count": int(preloaded_world_min_material_slot_count),
        "pause_after_spawn_before_render": bool(pause_after_spawn_before_render),
        "open_map_only": bool(open_map_only),
        "open_map_only_sleep_seconds": float(open_map_only_sleep_seconds),
        "open_map_only_no_quit": bool(open_map_only_no_quit),
    }
    launcher_template = """print("[BEDLAM360_V3_RESILIENT_CHUNK] RESILIENT_LAUNCHER_PYTHON_ENTERED")
import json
import os
import runpy
import shutil
import sys
import time
import traceback
from pathlib import Path

PAYLOAD = __PAYLOAD__

def _log(message, payload=None):
    prefix = "[BEDLAM360_V3_RESILIENT_CHUNK]"
    if payload is None:
        print(prefix, message)
    else:
        print(prefix, message, json.dumps(payload, indent=2))

_log("RESILIENT_LAUNCHER_IMPORTS_DONE", {"payload_keys": sorted(PAYLOAD.keys())})
startup_wait_seconds = float(PAYLOAD.get("unreal_startup_wait_seconds", 0.0) or 0.0)
if startup_wait_seconds > 0.0:
    _log("UNREAL_STARTUP_WAIT_START", {"sleep_seconds": startup_wait_seconds})
    time.sleep(startup_wait_seconds)
    _log("UNREAL_STARTUP_WAIT_DONE", {"slept_seconds": startup_wait_seconds})

def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return str(value)
    except Exception:
        return repr(value)

def _list_run_dirs(root_path):
    root = Path(root_path)
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.stat().st_mtime)

def _new_run_dirs_after(before_paths, root_path):
    before = {str(Path(path)) for path in before_paths}
    return [path for path in _list_run_dirs(root_path) if str(path) not in before]

def _get_current_map_info(unreal):
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

def _get_all_level_actors(unreal):
    try:
        return list(unreal.EditorLevelLibrary.get_all_level_actors())
    except Exception:
        return []

def _find_usd_stage_actors(unreal):
    actors = _get_all_level_actors(unreal)
    stage_actors = []
    for actor in actors:
        actor_class = ""
        label = ""
        name = ""
        try:
            if actor and actor.get_class():
                actor_class = str(actor.get_class().get_name())
        except Exception:
            actor_class = ""
        try:
            label = str(actor.get_actor_label())
        except Exception:
            label = ""
        try:
            name = str(actor.get_name())
        except Exception:
            name = ""
        if "UsdStageActor" in actor_class or "UsdStageActor" in label or "UsdStageActor" in name:
            stage_actors.append(actor)
    return stage_actors

def _safe_actor_root_layer(actor):
    try:
        return actor.get_editor_property("root_layer")
    except Exception:
        try:
            return getattr(actor, "root_layer")
        except Exception:
            return None

def _open_editor_map(unreal, map_path):
    attempts = []
    requested_map = str(map_path or "")
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
            "requested_map": requested_map,
            "current_map_before_open": _get_current_map_info(unreal),
            "api": api_name,
            "success": False,
        }
        try:
            attempt["return_value"] = _json_safe(callback())
            attempt["success"] = True
        except Exception as exc:
            attempt["error"] = str(exc)
        attempt["current_map_immediately_after_open"] = _get_current_map_info(unreal)
        attempt["viewport_invalidation_immediate"] = _invalidate_editor_viewports(unreal)
        attempt["post_open_wait"] = _wait_after_usd_bind(2.0)
        attempt["viewport_invalidation_after_wait"] = _invalidate_editor_viewports(unreal)
        attempt["current_map_after_short_wait"] = _get_current_map_info(unreal)
        attempt["map_match_after_short_wait"] = _editor_map_paths_match(
            (attempt.get("current_map_after_short_wait") or {}).get("world_path"),
            requested_map,
        )
        attempts.append(attempt)
        if attempt["success"] and attempt["map_match_after_short_wait"]:
            return {
                "success": True,
                "requested_map": requested_map,
                "attempts": attempts,
                "final_current_map": attempt.get("current_map_after_short_wait"),
            }
    return {
        "success": False,
        "requested_map": requested_map,
        "attempts": attempts,
        "final_current_map": _get_current_map_info(unreal),
    }

def _spawn_usd_stage_actor(unreal):
    if not hasattr(unreal, "UsdStageActor"):
        return None, "UsdStageActor class not exposed in Unreal Python"
    try:
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    except Exception as exc:
        return None, f"Could not get EditorActorSubsystem: {exc}"
    if actor_subsystem is None:
        return None, "EditorActorSubsystem unavailable"
    try:
        actor = actor_subsystem.spawn_actor_from_class(
            unreal.UsdStageActor,
            unreal.Vector(0.0, 0.0, 0.0),
            unreal.Rotator(0.0, 0.0, 0.0),
        )
    except Exception as exc:
        return None, f"spawn_actor_from_class failed: {exc}"
    try:
        actor.set_actor_label("UsdStageActor_AutoSpawned")
    except Exception:
        pass
    return actor, None

def _bind_usd_stage_actor(actor, expected_stage_path):
    try:
        import unreal
    except Exception as exc:
        return {
            "file_path_attempts": [{"api": "import_unreal", "success": False, "error": str(exc)}],
            "attempts": [{"mode": "import_unreal", "success": False, "error": str(exc)}],
            "post_edit_error": None,
            "final_root_layer": _json_safe(_safe_actor_root_layer(actor)),
        }
    attempts = []
    file_path_attempts = []
    file_path_value = None
    try:
        if hasattr(unreal, "FilePath"):
            file_path_value = unreal.FilePath()
            try:
                file_path_value.set_editor_property("file_path", str(expected_stage_path))
                file_path_attempts.append({"api": "FilePath.set_editor_property", "success": True})
            except Exception as exc:
                file_path_attempts.append({"api": "FilePath.set_editor_property", "success": False, "error": str(exc)})
                try:
                    file_path_value.file_path = str(expected_stage_path)
                    file_path_attempts.append({"api": "FilePath.file_path_assign", "success": True})
                except Exception as exc2:
                    file_path_attempts.append({"api": "FilePath.file_path_assign", "success": False, "error": str(exc2)})
                    file_path_value = None
    except Exception as exc:
        file_path_attempts.append({"api": "FilePath.construct", "success": False, "error": str(exc)})
        file_path_value = None
    for mode in (
        "set_root_layer_str",
        "set_editor_property_filepath",
        "attribute_set_filepath",
        "set_editor_property_str",
        "attribute_set_str",
        "set_root_layer_filepath",
    ):
        try:
            if mode == "set_root_layer_str":
                if not hasattr(actor, "set_root_layer"):
                    raise RuntimeError("UsdStageActor has no set_root_layer accessor")
                actor.set_root_layer(str(expected_stage_path))
            elif mode == "set_root_layer_filepath":
                if file_path_value is None:
                    raise RuntimeError("unreal.FilePath unavailable or failed to initialize")
                if not hasattr(actor, "set_root_layer"):
                    raise RuntimeError("UsdStageActor has no set_root_layer accessor")
                actor.set_root_layer(file_path_value)
            elif mode == "set_editor_property_filepath":
                if file_path_value is None:
                    raise RuntimeError("unreal.FilePath unavailable or failed to initialize")
                actor.set_editor_property("root_layer", file_path_value)
            elif mode == "attribute_set_filepath":
                if file_path_value is None:
                    raise RuntimeError("unreal.FilePath unavailable or failed to initialize")
                actor.root_layer = file_path_value
            elif mode == "set_editor_property_str":
                actor.set_editor_property("root_layer", expected_stage_path)
            else:
                actor.root_layer = expected_stage_path
            attempts.append({"mode": mode, "success": True})
            break
        except Exception as exc:
            attempts.append({"mode": mode, "success": False, "error": str(exc)})
    post_edit_error = None
    try:
        if hasattr(actor, "post_edit_change"):
            actor.post_edit_change()
    except Exception as exc:
        post_edit_error = str(exc)
    return {
        "file_path_attempts": file_path_attempts,
        "attempts": attempts,
        "post_edit_error": post_edit_error,
        "final_root_layer": _json_safe(_safe_actor_root_layer(actor)),
    }

def _safe_generated_assets_report(actor):
    report = {
        "generated_assets_accessible": False,
        "generated_assets_count": None,
        "generated_assets_sample": [],
        "generated_component_accessible": False,
        "generated_component_sample": None,
        "errors": [],
    }
    if hasattr(actor, "get_generated_assets"):
        try:
            assets = actor.get_generated_assets()
            report["generated_assets_accessible"] = True
            report["generated_assets_count"] = int(len(assets or []))
            for asset in list(assets or [])[:20]:
                report["generated_assets_sample"].append({
                    "name": _json_safe(asset.get_name() if hasattr(asset, "get_name") else None),
                    "path": _json_safe(asset.get_path_name() if hasattr(asset, "get_path_name") else None),
                    "class": _json_safe(asset.get_class().get_name() if hasattr(asset, "get_class") and asset.get_class() else None),
                })
        except Exception as exc:
            report["errors"].append({"api": "get_generated_assets", "error": str(exc)})
    if hasattr(actor, "get_generated_component"):
        try:
            component = actor.get_generated_component()
            report["generated_component_accessible"] = component is not None
            if component is not None:
                report["generated_component_sample"] = {
                    "name": _json_safe(component.get_name() if hasattr(component, "get_name") else None),
                    "path": _json_safe(component.get_path_name() if hasattr(component, "get_path_name") else None),
                    "class": _json_safe(component.get_class().get_name() if hasattr(component, "get_class") and component.get_class() else None),
                }
        except Exception as exc:
            report["errors"].append({"api": "get_generated_component", "error": str(exc)})
    return report

def _safe_enum_members(enum_type):
    members = []
    if enum_type is None:
        return members
    try:
        for name in dir(enum_type):
            if name.startswith("_"):
                continue
            try:
                value = getattr(enum_type, name)
            except Exception:
                continue
            text = _json_safe(value)
            if str(enum_type.__name__) in str(text) or isinstance(value, int):
                members.append({
                    "name": str(name),
                    "value": _json_safe(value),
                })
    except Exception:
        return members
    return members[:100]

def _safe_world_probe_summary(unreal):
    probe = _collect_usd_material_readiness_probe(unreal)
    return {
        "actor_count": int(probe.get("actor_count", 0)),
        "component_count": int(probe.get("component_count", 0)),
        "total_material_slots": int(probe.get("total_material_slots", 0)),
        "usd_stage_actor_count": int(probe.get("usd_stage_actor_count", 0)),
        "first_10_actors": list((probe.get("first_50_actors") or [])[:10]),
        "first_10_components": list((probe.get("first_50_components") or [])[:10]),
    }

def _purposes_to_load_diagnostic(actor, wait_seconds):
    try:
        import unreal
    except Exception as exc:
        return {
            "success": False,
            "error": f"import_unreal_failed: {exc}",
        }
    report = {
        "success": True,
        "current_value": None,
        "enum_members": [],
        "attempts": [],
    }
    current_value = None
    try:
        current_value = getattr(actor, "purposes_to_load")
        report["current_value"] = _json_safe(current_value)
    except Exception as exc:
        report["current_value_error"] = str(exc)
    enum_type = type(current_value) if current_value is not None else None
    report["enum_type"] = None if enum_type is None else getattr(enum_type, "__name__", str(enum_type))
    report["enum_members"] = _safe_enum_members(enum_type)

    candidate_specs = []
    if current_value is not None:
        candidate_specs.append(("current_value", current_value))
    seen_names = {str(_json_safe(current_value))}
    candidate_names = [
        "PROXY",
        "RENDER",
        "PROXY_AND_RENDER",
        "DEFAULT",
        "ALL",
    ]
    for name in candidate_names:
        value = None
        resolved = False
        if enum_type is not None and hasattr(enum_type, name):
            try:
                value = getattr(enum_type, name)
                resolved = True
            except Exception:
                resolved = False
        if resolved:
            key = str(_json_safe(value))
            if key not in seen_names:
                candidate_specs.append((name, value))
                seen_names.add(key)
    if enum_type is None:
        for name, raw_value in (
            ("proxy_numeric_1", 1),
            ("render_numeric_2", 2),
            ("proxy_render_numeric_3", 3),
            ("default_numeric_0", 0),
        ):
            if str(raw_value) not in seen_names:
                candidate_specs.append((name, raw_value))
                seen_names.add(str(raw_value))

    for candidate_name, candidate_value in candidate_specs:
        row = {
            "candidate_name": str(candidate_name),
            "candidate_value": _json_safe(candidate_value),
            "call_success": False,
            "call_error": None,
            "wait_report": None,
            "compilation_report": None,
            "world_probe_summary": None,
            "outliner_like_world_present": False,
            "usd_stage_actor_inspection_after": None,
        }
        if not hasattr(actor, "set_purposes_to_load"):
            row["call_error"] = "set_purposes_to_load_unavailable"
            report["attempts"].append(row)
            continue
        try:
            actor.set_purposes_to_load(candidate_value)
            row["call_success"] = True
        except Exception as exc:
            row["call_error"] = str(exc)
            report["attempts"].append(row)
            continue
        row["wait_report"] = _wait_after_usd_bind(wait_seconds)
        row["compilation_report"] = _finish_asset_compilation_barrier(unreal)
        _invalidate_editor_viewports(unreal)
        row["world_probe_summary"] = _safe_world_probe_summary(unreal)
        world_probe = _collect_usd_material_readiness_probe(unreal)
        row["outliner_like_world_present"] = bool(int(world_probe.get("actor_count", 0)) >= 20 or int(world_probe.get("component_count", 0)) >= 20)
        row["usd_stage_actor_inspection_after"] = _inspect_usd_stage_actor(actor)
        report["attempts"].append(row)
    return report

def _reload_usd_stage_actor(actor):
    attempts = []
    for accessor in ("reload_stage", "load_stage", "open_stage", "refresh"):
        if not hasattr(actor, accessor):
            continue
        try:
            getattr(actor, accessor)()
            attempts.append({"accessor": accessor, "success": True})
        except Exception as exc:
            attempts.append({"accessor": accessor, "success": False, "error": str(exc)})
    return attempts

def _post_load_population_diagnostics(actor, expected_stage_path, wait_seconds):
    try:
        import unreal
    except Exception as exc:
        return {
            "success": False,
            "error": f"import_unreal_failed: {exc}",
            "attempts": [],
        }
    attempts = []
    before_probe = _collect_usd_material_readiness_probe(unreal)
    before_signature = before_probe.get("signature") or {}
    candidate_calls = (
        "reload_stage",
        "load_stage",
        "open_stage",
        "refresh",
        "refresh_stage",
        "rebuild",
        "update",
        "force_reload",
        "set_isolated_root_layer",
    )
    for accessor in candidate_calls:
        row = {
            "method": accessor,
            "exists": bool(hasattr(actor, accessor)),
            "call_success": False,
            "call_error": None,
            "wait_report": None,
            "compilation_report": None,
            "probe_after": None,
            "counts_changed": False,
        }
        if not row["exists"]:
            attempts.append(row)
            continue
        try:
            fn = getattr(actor, accessor)
            if accessor == "set_isolated_root_layer":
                fn(str(expected_stage_path))
            else:
                fn()
            row["call_success"] = True
        except Exception as exc:
            row["call_error"] = str(exc)
            attempts.append(row)
            continue
        row["wait_report"] = _wait_after_usd_bind(wait_seconds)
        row["compilation_report"] = _finish_asset_compilation_barrier(unreal)
        _invalidate_editor_viewports(unreal)
        probe_after = _collect_usd_material_readiness_probe(unreal)
        row["probe_after"] = probe_after
        after_signature = probe_after.get("signature") or {}
        row["counts_changed"] = bool(
            int(after_signature.get("actor_count", -1)) != int(before_signature.get("actor_count", -1))
            or int(after_signature.get("total_components", -1)) != int(before_signature.get("total_components", -1))
            or int(after_signature.get("total_material_slots", -1)) != int(before_signature.get("total_material_slots", -1))
        )
        attempts.append(row)
        before_probe = probe_after
        before_signature = after_signature
    return {
        "success": True,
        "wait_seconds_per_attempt": float(wait_seconds),
        "attempts": attempts,
        "final_probe": before_probe,
    }

def _validate_usd_stage_access(actor):
    report = {
        "loaded_stage_accessible": False,
        "stage_access_method": None,
        "stage_access_error": None,
    }
    for accessor in ("get_usd_stage", "get_stage"):
        if not hasattr(actor, accessor):
            continue
        try:
            stage = getattr(actor, accessor)()
            if stage is not None:
                report["loaded_stage_accessible"] = True
                report["stage_access_method"] = accessor
                return report
        except Exception as exc:
            report["stage_access_error"] = str(exc)
    return report

def _matches_usd_keyword(name):
    if name is None:
        return False
    text = str(name).lower()
    keywords = (
        "stage",
        "load",
        "open",
        "reload",
        "refresh",
        "root",
        "layer",
        "prim",
        "asset",
        "generated",
        "import",
        "translate",
        "populate",
    )
    return any(keyword in text for keyword in keywords)

def _safe_list_actor_components(actor):
    rows = []
    try:
        components = actor.get_components_by_class(object)
    except Exception:
        try:
            components = actor.get_components_by_class(actor.get_root_component().__class__)
        except Exception:
            components = []
    if not components:
        try:
            root_component = actor.get_root_component()
            if root_component is not None:
                components = [root_component]
        except Exception:
            components = []
    seen = set()
    for component in components or []:
        try:
            path = component.get_path_name()
        except Exception:
            path = str(id(component))
        if path in seen:
            continue
        seen.add(path)
        row = {
            "component_name": None,
            "component_class": None,
            "path": None,
        }
        try:
            row["component_name"] = component.get_name()
        except Exception:
            pass
        try:
            row["component_class"] = component.get_class().get_name()
        except Exception:
            pass
        row["path"] = path
        rows.append(row)
    return rows

def _safe_list_child_actors(actor):
    rows = []
    child_actors = []
    for accessor in ("get_attached_actors",):
        if not hasattr(actor, accessor):
            continue
        try:
            child_actors = getattr(actor, accessor)() or []
            break
        except Exception:
            child_actors = []
    for child in child_actors or []:
        row = {
            "actor_label": None,
            "actor_name": None,
            "actor_class": None,
            "path": None,
        }
        try:
            row["actor_label"] = child.get_actor_label()
        except Exception:
            pass
        try:
            row["actor_name"] = child.get_name()
        except Exception:
            pass
        try:
            row["actor_class"] = child.get_class().get_name()
        except Exception:
            pass
        try:
            row["path"] = child.get_path_name()
        except Exception:
            pass
        rows.append(row)
    return rows

def _safe_list_editor_properties(actor):
    rows = []
    names = []
    try:
        if hasattr(actor, "get_editor_property_names"):
            names = list(actor.get_editor_property_names() or [])
    except Exception:
        names = []
    for name in names:
        try:
            value = actor.get_editor_property(name)
            rows.append({
                "name": str(name),
                "value": _json_safe(value),
                "read_success": True,
            })
        except Exception as exc:
            rows.append({
                "name": str(name),
                "value": None,
                "read_success": False,
                "error": str(exc),
            })
    filtered = [row for row in rows if _matches_usd_keyword(row.get("name"))]
    return {
        "filtered_properties": filtered[:200],
        "all_property_names": [str(name) for name in names[:400]],
        "total_property_name_count": int(len(names)),
    }

def _safe_list_candidate_methods(actor):
    methods = []
    attribute_names = []
    try:
        attribute_names = sorted(set(dir(actor)))
    except Exception:
        attribute_names = []
    for name in attribute_names:
        if not _matches_usd_keyword(name):
            continue
        row = {
            "name": str(name),
            "callable": False,
        }
        try:
            value = getattr(actor, name)
            row["callable"] = bool(callable(value))
            row["value_type"] = type(value).__name__
            if not row["callable"]:
                row["value"] = _json_safe(value)
        except Exception as exc:
            row["access_error"] = str(exc)
        methods.append(row)
    return {
        "candidate_attributes": methods[:400],
        "total_attribute_count": int(len(attribute_names)),
    }

def _inspect_usd_stage_actor(actor):
    report = {
        "actor_label": None,
        "actor_name": None,
        "actor_class": None,
        "path": None,
        "root_layer": _json_safe(_safe_actor_root_layer(actor)),
        "candidate_methods_and_attributes": _safe_list_candidate_methods(actor),
        "editor_properties": _safe_list_editor_properties(actor),
        "direct_components": _safe_list_actor_components(actor),
        "child_actors": _safe_list_child_actors(actor),
        "root_component": None,
    }
    try:
        report["actor_label"] = actor.get_actor_label()
    except Exception:
        pass
    try:
        report["actor_name"] = actor.get_name()
    except Exception:
        pass
    try:
        report["actor_class"] = actor.get_class().get_name()
    except Exception:
        pass
    try:
        report["path"] = actor.get_path_name()
    except Exception:
        pass
    try:
        root_component = actor.get_root_component()
    except Exception:
        root_component = None
    if root_component is not None:
        report["root_component"] = {
            "component_name": _json_safe(getattr(root_component, "get_name", lambda: None)()),
            "component_class": _json_safe(root_component.get_class().get_name() if root_component.get_class() else None),
            "path": _json_safe(root_component.get_path_name() if hasattr(root_component, "get_path_name") else None),
        }
    return report

def _normalize_path_for_compare(value):
    try:
        return str(value).replace(chr(92), "/").strip()
    except Exception:
        return str(value)

def _normalize_editor_map_path(value):
    normalized = _normalize_path_for_compare(value)
    if not normalized:
        return normalized
    if "." in normalized:
        package_part, asset_part = normalized.rsplit(".", 1)
        package_leaf = package_part.rsplit("/", 1)[-1]
        if asset_part == package_leaf:
            normalized = package_part
    return normalized

def _editor_map_paths_match(current_value, requested_value):
    current_normalized = _normalize_editor_map_path(current_value)
    requested_normalized = _normalize_editor_map_path(requested_value)
    return bool(current_normalized and requested_normalized and current_normalized == requested_normalized)

def _root_layer_matches_expected(final_root_layer, expected_stage_path):
    final_value = _normalize_path_for_compare(final_root_layer)
    expected_value = _normalize_path_for_compare(expected_stage_path)
    return bool(expected_value and expected_value in final_value)

def _wait_after_usd_bind(wait_seconds):
    if wait_seconds is None:
        return {"wait_requested_seconds": None, "wait_applied_seconds": 0.0}
    wait_seconds = max(float(wait_seconds), 0.0)
    if wait_seconds <= 0.0:
        return {"wait_requested_seconds": wait_seconds, "wait_applied_seconds": 0.0}
    try:
        import time
        time.sleep(wait_seconds)
        return {"wait_requested_seconds": wait_seconds, "wait_applied_seconds": wait_seconds, "method": "time.sleep"}
    except Exception as exc:
        return {"wait_requested_seconds": wait_seconds, "wait_applied_seconds": 0.0, "method": "sleep_failed", "error": str(exc)}

def _wait_after_warmup(wait_seconds):
    if wait_seconds is None:
        return {"wait_requested_seconds": None, "wait_applied_seconds": 0.0}
    wait_seconds = max(float(wait_seconds), 0.0)
    if wait_seconds <= 0.0:
        return {"wait_requested_seconds": wait_seconds, "wait_applied_seconds": 0.0}
    try:
        import time
        time.sleep(wait_seconds)
        return {"wait_requested_seconds": wait_seconds, "wait_applied_seconds": wait_seconds, "method": "time.sleep"}
    except Exception as exc:
        return {"wait_requested_seconds": wait_seconds, "wait_applied_seconds": 0.0, "method": "sleep_failed", "error": str(exc)}

def _invalidate_editor_viewports(unreal):
    attempts = []
    try:
        if hasattr(unreal.EditorLevelLibrary, "editor_invalidate_viewports"):
            unreal.EditorLevelLibrary.editor_invalidate_viewports()
            attempts.append({"api": "EditorLevelLibrary.editor_invalidate_viewports", "success": True})
    except Exception as exc:
        attempts.append({"api": "EditorLevelLibrary.editor_invalidate_viewports", "success": False, "error": str(exc)})
    try:
        subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        if subsystem is not None and hasattr(subsystem, "redraw_all_viewports"):
            subsystem.redraw_all_viewports()
            attempts.append({"api": "LevelEditorSubsystem.redraw_all_viewports", "success": True})
    except Exception as exc:
        attempts.append({"api": "LevelEditorSubsystem.redraw_all_viewports", "success": False, "error": str(exc)})
    return attempts

def _finish_asset_compilation_barrier(unreal):
    report = {
        "attempts": [],
        "finish_all_compilation_called": False,
    }
    acm_class = getattr(unreal, "AssetCompilingManager", None)
    if acm_class is None:
        report["attempts"].append({"api": "AssetCompilingManager", "success": False, "error": "class_unavailable"})
        return report
    manager = None
    try:
        if hasattr(acm_class, "get"):
            manager = acm_class.get()
            report["attempts"].append({"api": "AssetCompilingManager.get", "success": True})
    except Exception as exc:
        report["attempts"].append({"api": "AssetCompilingManager.get", "success": False, "error": str(exc)})
    try:
        target = manager if manager is not None else acm_class
        if target is not None and hasattr(target, "finish_all_compilation"):
            target.finish_all_compilation()
            report["attempts"].append({"api": "finish_all_compilation", "success": True})
            report["finish_all_compilation_called"] = True
        else:
            report["attempts"].append({"api": "finish_all_compilation", "success": False, "error": "method_unavailable"})
    except Exception as exc:
        report["attempts"].append({"api": "finish_all_compilation", "success": False, "error": str(exc)})
    return report

def _component_visible(component):
    try:
        if hasattr(component, "is_visible"):
            return bool(component.is_visible())
    except Exception:
        pass
    try:
        hidden = component.get_editor_property("hidden_in_game")
        return not bool(hidden)
    except Exception:
        return True

def _material_descriptor(material):
    if material is None:
        return {"name": None, "path": None, "is_fallback": True, "reason": "material_none"}
    material_name = None
    material_path = None
    try:
        material_name = str(material.get_name())
    except Exception:
        material_name = None
    try:
        material_path = str(material.get_path_name())
    except Exception:
        material_path = None
    text = f"{material_name or ''} {material_path or ''}".lower()
    fallback_tokens = (
        "worldgrid",
        "defaultmaterial",
        "default_material",
        "defaultwhitegrid",
        "default_white_grid",
        "checker",
        "checkboard",
        "checkerboard",
    )
    matched = [token for token in fallback_tokens if token in text]
    return {
        "name": material_name,
        "path": material_path,
        "is_fallback": bool(matched),
        "reason": None if not matched else ",".join(matched),
    }

def _iter_render_components(unreal):
    actors = _get_all_level_actors(unreal)
    classes = []
    for class_name in ("StaticMeshComponent", "SkeletalMeshComponent", "GeometryCacheComponent"):
        cls = getattr(unreal, class_name, None)
        if cls is not None:
            classes.append(cls)
    seen = set()
    for actor in actors:
        for cls in classes:
            try:
                components = actor.get_components_by_class(cls)
            except Exception:
                components = []
            for component in components or []:
                try:
                    key = component.get_path_name()
                except Exception:
                    key = str(id(component))
                if key in seen:
                    continue
                seen.add(key)
                yield actor, component

def _histogram_increment(counter, key):
    key = "<unknown>" if key in (None, "") else str(key)
    counter[key] = int(counter.get(key, 0)) + 1

def _sorted_histogram(counter):
    return dict(sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0]))))

def _collect_usd_material_readiness_probe(unreal):
    actor_rows = []
    actor_class_histogram = {}
    all_actors = _get_all_level_actors(unreal)
    usd_stage_actor_count = 0
    translated_actor_count = 0
    translated_actor_samples = []
    for actor in all_actors:
        actor_label = None
        actor_name = None
        actor_class = None
        try:
            actor_label = actor.get_actor_label()
        except Exception:
            pass
        try:
            actor_name = actor.get_name()
        except Exception:
            pass
        try:
            actor_class = actor.get_class().get_name()
        except Exception:
            pass
        _histogram_increment(actor_class_histogram, actor_class)
        if len(actor_rows) < 50:
            actor_rows.append({
                "actor_label": actor_label,
                "actor_name": actor_name,
                "actor_class": actor_class,
            })
        actor_class_text = "" if actor_class is None else str(actor_class)
        if "UsdStageActor" in actor_class_text:
            usd_stage_actor_count += 1
        else:
            translated_actor_count += 1
            if len(translated_actor_samples) < 50:
                translated_actor_samples.append({
                    "actor_label": actor_label,
                    "actor_name": actor_name,
                    "actor_class": actor_class,
                })

    total_components = 0
    visible_components = 0
    total_material_slots = 0
    fallback_material_slots = 0
    none_material_slots = 0
    component_rows = []
    component_class_histogram = {}
    sample_rows = []
    fallback_samples = []
    material_name_rows = []
    for actor, component in _iter_render_components(unreal):
        total_components += 1
        component_class = None
        component_name = None
        owner_label = None
        owner_name = None
        attach_parent_actor_label = None
        attach_parent_actor_name = None
        try:
            component_class = component.get_class().get_name()
        except Exception:
            pass
        try:
            component_name = component.get_name()
        except Exception:
            pass
        try:
            owner_label = actor.get_actor_label()
        except Exception:
            pass
        try:
            owner_name = actor.get_name()
        except Exception:
            pass
        try:
            attach_parent_actor = actor.get_attach_parent_actor()
        except Exception:
            attach_parent_actor = None
        if attach_parent_actor is not None:
            try:
                attach_parent_actor_label = attach_parent_actor.get_actor_label()
            except Exception:
                pass
            try:
                attach_parent_actor_name = attach_parent_actor.get_name()
            except Exception:
                pass
        _histogram_increment(component_class_histogram, component_class)
        if len(component_rows) < 50:
            component_rows.append({
                "actor_label": owner_label,
                "actor_name": owner_name,
                "component_name": component_name,
                "component_class": component_class,
                "attach_parent_actor_label": attach_parent_actor_label,
                "attach_parent_actor_name": attach_parent_actor_name,
            })
        if not _component_visible(component):
            continue
        visible_components += 1
        try:
            slot_count = int(component.get_num_materials())
        except Exception:
            slot_count = 0
        component_row = {
            "actor_label": owner_label,
            "actor_name": owner_name,
            "component_name": component_name,
            "component_class": component_class,
            "attach_parent_actor_label": attach_parent_actor_label,
            "attach_parent_actor_name": attach_parent_actor_name,
            "slot_count": int(slot_count),
            "materials": [],
        }
        for slot_index in range(max(0, slot_count)):
            total_material_slots += 1
            try:
                material = component.get_material(slot_index)
            except Exception:
                material = None
            desc = _material_descriptor(material)
            desc["slot_index"] = int(slot_index)
            component_row["materials"].append(desc)
            if len(material_name_rows) < 50:
                material_name_rows.append({
                    "actor_label": owner_label,
                    "component_name": component_name,
                    "slot_index": int(slot_index),
                    "material_name": desc.get("name"),
                    "material_path": desc.get("path"),
                    "is_fallback": bool(desc.get("is_fallback")),
                })
            if desc["name"] is None and desc["path"] is None:
                none_material_slots += 1
            if desc["is_fallback"]:
                fallback_material_slots += 1
                if len(fallback_samples) < 50:
                    fallback_samples.append({
                        "actor_label": component_row["actor_label"],
                        "component_name": component_row["component_name"],
                        "component_class": component_row["component_class"],
                        "slot_index": int(slot_index),
                        "material_name": desc["name"],
                        "material_path": desc["path"],
                        "fallback_reason": desc["reason"],
                    })
        if len(sample_rows) < 50:
            sample_rows.append(component_row)
    fallback_ratio = 0.0 if total_material_slots <= 0 else float(fallback_material_slots) / float(total_material_slots)
    signature = {
        "actor_count": int(len(all_actors)),
        "visible_components": int(visible_components),
        "total_components": int(total_components),
        "total_material_slots": int(total_material_slots),
        "fallback_material_slots": int(fallback_material_slots),
        "none_material_slots": int(none_material_slots),
        "fallback_ratio_rounded": round(float(fallback_ratio), 4),
        "usd_stage_actor_count": int(usd_stage_actor_count),
        "translated_actor_count": int(translated_actor_count),
    }
    return {
        "actor_count": int(len(all_actors)),
        "actor_class_histogram": _sorted_histogram(actor_class_histogram),
        "first_50_actors": actor_rows,
        "total_components": int(total_components),
        "visible_components": int(visible_components),
        "component_count": int(total_components),
        "component_class_histogram": _sorted_histogram(component_class_histogram),
        "total_material_slots": int(total_material_slots),
        "fallback_material_slots": int(fallback_material_slots),
        "none_material_slots": int(none_material_slots),
        "fallback_ratio": float(fallback_ratio),
        "first_50_components": component_rows,
        "first_50_materials": material_name_rows,
        "usd_stage_actor_count": int(usd_stage_actor_count),
        "translated_usd_actors_exist": bool(translated_actor_count > 0),
        "translated_actor_count": int(translated_actor_count),
        "translated_actor_samples": translated_actor_samples,
        "only_usd_stage_and_helper_components": bool(translated_actor_count <= 0 and total_components <= 5),
        "scene_actors_attached_under_other_hierarchy": bool(
            any(row.get("attach_parent_actor_name") for row in component_rows)
        ),
        "signature": signature,
        "sample_components": sample_rows,
        "fallback_samples": fallback_samples,
    }

def _collect_bedlam_geometry_cache_material_probe(unreal):
    rows = []
    any_worldgrid = False
    for actor, component in _iter_render_components(unreal):
        component_class = None
        try:
            component_class = component.get_class().get_name()
        except Exception:
            component_class = None
        if str(component_class) != "GeometryCacheComponent":
            continue
        actor_label = None
        actor_name = None
        actor_class = None
        try:
            actor_label = actor.get_actor_label()
        except Exception:
            pass
        try:
            actor_name = actor.get_name()
        except Exception:
            pass
        try:
            actor_class = actor.get_class().get_name()
        except Exception:
            pass
        materials = []
        slot_count = 0
        try:
            slot_count = int(component.get_num_materials())
        except Exception:
            slot_count = 0
        for slot_index in range(max(0, slot_count)):
            try:
                material = component.get_material(slot_index)
            except Exception:
                material = None
            desc = _material_descriptor(material)
            desc["slot_index"] = int(slot_index)
            materials.append(desc)
            any_worldgrid = any_worldgrid or bool(
                str(desc.get("name") or "").lower() == "worldgridmaterial"
                or "worldgrid" in str(desc.get("path") or "").lower()
            )
        rows.append({
            "actor_label": actor_label,
            "actor_name": actor_name,
            "actor_class": actor_class,
            "component_name": _json_safe(component.get_name() if hasattr(component, "get_name") else None),
            "component_path": _json_safe(component.get_path_name() if hasattr(component, "get_path_name") else None),
            "materials": materials,
            "worldgrid_material_present": any(bool(m.get("is_fallback")) for m in materials),
        })
    return {
        "geometry_cache_actor_count": int(len(rows)),
        "any_worldgrid_material_present": bool(any_worldgrid),
        "rows": rows[:50],
    }

def _collect_pre_render_state(unreal, tag):
    stage_actors = _find_usd_stage_actors(unreal)
    stage_actor = stage_actors[0] if stage_actors else None
    return {
        "tag": str(tag),
        "current_map": _get_current_map_info(unreal),
        "usd_stage_actor_count": int(len(stage_actors)),
        "usd_stage_actor_root_layer": _json_safe(_safe_actor_root_layer(stage_actor)) if stage_actor is not None else None,
        "usd_stage_actor_child_actors": _safe_list_child_actors(stage_actor) if stage_actor is not None else [],
        "bedlam_geometry_cache_material_probe": _collect_bedlam_geometry_cache_material_probe(unreal),
    }

def _wait_for_usd_material_readiness(expected_stage_path, timeout_seconds, poll_seconds, allow_timeout):
    try:
        import unreal
    except Exception as exc:
        raise RuntimeError(f"USD_MATERIAL_READINESS_IMPORT_FAILED: {exc}")
    timeout_seconds = max(float(timeout_seconds or 0.0), 0.0)
    poll_seconds = max(float(poll_seconds or 0.0), 0.0)
    start_time = time.perf_counter()
    polls = []
    stable_polls = 0
    last_signature = None
    fallback_ratio_threshold = float(__USD_MATERIAL_MAX_FALLBACK_RATIO__)
    required_stable_polls = int(__USD_MATERIAL_STABLE_POLLS__)
    _log("USD_MATERIAL_READINESS_START", {
        "expected_stage_path": expected_stage_path,
        "timeout_seconds": timeout_seconds,
        "poll_seconds": poll_seconds,
        "fallback_ratio_threshold": fallback_ratio_threshold,
        "required_stable_polls": required_stable_polls,
        "allow_timeout": bool(allow_timeout),
    })
    while True:
        compilation_report = _finish_asset_compilation_barrier(unreal)
        invalidate_report = _invalidate_editor_viewports(unreal)
        probe = _collect_usd_material_readiness_probe(unreal)
        signature = probe.get("signature")
        if signature == last_signature:
            stable_polls += 1
        else:
            stable_polls = 1
            last_signature = signature
        elapsed = float(time.perf_counter() - start_time)
        poll_row = {
            "elapsed_seconds": elapsed,
            "stable_polls": int(stable_polls),
            "compilation_report": compilation_report,
            "invalidate_report": invalidate_report,
            "probe": probe,
        }
        polls.append(poll_row)
        _log("USD_MATERIAL_READINESS_POLL", poll_row)
        fallback_ratio = float(probe.get("fallback_ratio", 1.0))
        material_slots = int(probe.get("total_material_slots", 0))
        if material_slots > 0 and fallback_ratio <= fallback_ratio_threshold and stable_polls >= required_stable_polls:
            final_report = {
                "success": True,
                "reason": "fallback_ratio_below_threshold_and_stable",
                "elapsed_seconds": elapsed,
                "poll_count": len(polls),
                "final_probe": probe,
                "polls": polls,
            }
            _log("USD_MATERIAL_READINESS_FINAL", final_report)
            return final_report
        if elapsed >= timeout_seconds:
            final_report = {
                "success": False,
                "reason": "USD_MATERIAL_READINESS_TIMEOUT",
                "elapsed_seconds": elapsed,
                "poll_count": len(polls),
                "final_probe": probe,
                "polls": polls,
            }
            _log("USD_MATERIAL_READINESS_FINAL", final_report)
            if bool(allow_timeout):
                _log("USD_MATERIAL_READINESS_TIMEOUT_ALLOWED", final_report)
                return final_report
            raise RuntimeError(
                "USD_MATERIAL_READINESS_TIMEOUT: "
                f"fallback_ratio={fallback_ratio:.4f} total_material_slots={material_slots} "
                f"elapsed_seconds={elapsed:.1f}"
            )
        if poll_seconds > 0.0:
            time.sleep(poll_seconds)

def _ensure_preloaded_world_populated(
    unreal,
    preferred_editor_map,
    expected_stage_path,
    min_actor_count,
    min_component_count,
    min_material_slot_count,
):
    probe = _collect_usd_material_readiness_probe(unreal)
    current_map = _get_current_map_info(unreal)
    actor_count = int(probe.get("actor_count", 0))
    component_count = int(probe.get("component_count", 0))
    material_slot_count = int(probe.get("total_material_slots", 0))
    actor_histogram = dict(probe.get("actor_class_histogram") or {})
    stage_actors = _find_usd_stage_actors(unreal)
    usd_stage_actor = stage_actors[0] if stage_actors else None
    usd_stage_root_layer = None
    usd_stage_root_layer_valid = False
    usd_stage_root_layer_matches_expected = False
    usd_stage_child_actors = []
    usd_world_child_exists = False
    validation = None
    if usd_stage_actor is not None:
        usd_stage_root_layer = _json_safe(_safe_actor_root_layer(usd_stage_actor))
        usd_stage_root_layer_valid = bool(usd_stage_root_layer)
        usd_stage_root_layer_matches_expected = bool(
            _root_layer_matches_expected(usd_stage_root_layer, expected_stage_path)
        )
        usd_stage_child_actors = _safe_list_child_actors(usd_stage_actor)
        usd_world_child_exists = any(
            str((row.get("actor_label") or "")).startswith("World")
            or str((row.get("actor_name") or "")).startswith("World")
            for row in usd_stage_child_actors
        )
        validation = _validate_usd_stage_access(usd_stage_actor)
    map_matches_preferred = False
    current_map_candidates = [
        str(current_map.get("world_path") or ""),
        str(current_map.get("persistent_level_path") or ""),
        str(current_map.get("world_name") or ""),
        str(current_map.get("persistent_level_name") or ""),
    ]
    preferred_map_text = str(preferred_editor_map or "")
    preferred_map_leaf = preferred_map_text.split("/")[-1] if preferred_map_text else ""
    for candidate in current_map_candidates:
        if not candidate:
            continue
        if _editor_map_paths_match(candidate, preferred_map_text):
            map_matches_preferred = True
            break
        if preferred_map_text and preferred_map_text in candidate:
            map_matches_preferred = True
            break
        if preferred_map_leaf and preferred_map_leaf in candidate:
            map_matches_preferred = True
            break
    scene_capture_cube_count = int(actor_histogram.get("SceneCaptureCube", 0))
    geometry_cache_actor_count = int(actor_histogram.get("GeometryCacheActor", 0))
    static_mesh_actor_count = int(actor_histogram.get("StaticMeshActor", 0))
    baked_geometry_present = bool(geometry_cache_actor_count >= 1 or static_mesh_actor_count >= 1)
    lighting_sky_actor_count = int(sum(
        int(actor_histogram.get(name, 0))
        for name in (
            "DirectionalLight",
            "SkyLight",
            "SkyAtmosphere",
            "VolumetricCloud",
            "ExponentialHeightFog",
            "HDRIBackdrop_C",
            "HDRIBackdrop",
        )
    ))
    populated = bool(
        actor_count >= int(min_actor_count)
        or component_count >= int(min_component_count)
        or material_slot_count >= int(min_material_slot_count)
    )
    usd_backed_stage_ready = bool(
        usd_stage_actor is not None
        and usd_stage_root_layer_matches_expected
        and usd_world_child_exists
    )
    baked_map_ready = bool(
        map_matches_preferred
        and scene_capture_cube_count >= 1
        and baked_geometry_present
        and lighting_sky_actor_count >= 1
    )
    report = {
        "preferred_editor_map": preferred_editor_map,
        "current_map": current_map,
        "expected_stage_path": expected_stage_path,
        "populated": populated,
        "usd_backed_stage_ready": usd_backed_stage_ready,
        "baked_map_ready": baked_map_ready,
        "thresholds": {
            "min_actor_count": int(min_actor_count),
            "min_component_count": int(min_component_count),
            "min_material_slot_count": int(min_material_slot_count),
        },
        "usd_stage_actor_count": int(len(stage_actors)),
        "usd_stage_root_layer": usd_stage_root_layer,
        "usd_stage_root_layer_valid": bool(usd_stage_root_layer_valid),
        "usd_stage_root_layer_matches_expected": bool(usd_stage_root_layer_matches_expected),
        "usd_stage_child_actors": usd_stage_child_actors,
        "usd_world_child_exists": bool(usd_world_child_exists),
        "usd_stage_validation": validation,
        "map_matches_preferred": bool(map_matches_preferred),
        "scene_capture_cube_count": int(scene_capture_cube_count),
        "geometry_cache_actor_count": int(geometry_cache_actor_count),
        "static_mesh_actor_count": int(static_mesh_actor_count),
        "lighting_sky_actor_count": int(lighting_sky_actor_count),
        "probe": probe,
    }
    _log("PRELOADED_WORLD_POPULATION_CHECK", report)
    if usd_backed_stage_ready:
        _log("PRELOADED_EDITOR_MAP_USD_ACCEPTED", {
            "map_path": preferred_editor_map,
            "usd_stage_actor_name": None if usd_stage_actor is None else _json_safe(
                usd_stage_actor.get_name() if hasattr(usd_stage_actor, "get_name") else None
            ),
            "root_layer": usd_stage_root_layer,
            "expected_scene_root": str(Path(expected_stage_path).parent.parent.parent.parent) if expected_stage_path else None,
            "expected_stage_path": expected_stage_path,
            "world_prim_found": bool(usd_world_child_exists),
            "usd_stage_validation": validation,
            "probe_summary": {
                "actor_count": actor_count,
                "component_count": component_count,
                "material_slots": material_slot_count,
            },
        })
        return report
    if baked_map_ready:
        _log("PRELOADED_EDITOR_MAP_BAKED_ACCEPTED", {
            "map_path": preferred_editor_map,
            "current_map": current_map,
            "scene_capture_cube_count": int(scene_capture_cube_count),
            "geometry_cache_actor_count": int(geometry_cache_actor_count),
            "static_mesh_actor_count": int(static_mesh_actor_count),
            "lighting_sky_actor_count": int(lighting_sky_actor_count),
            "probe_summary": {
                "actor_count": actor_count,
                "component_count": component_count,
                "material_slots": material_slot_count,
            },
        })
        return report
    if not populated:
        _log("PRELOADED_EDITOR_MAP_REJECTED", {
            "map_path": preferred_editor_map,
            "current_map": current_map,
            "reason": "neither_usd_backed_nor_baked_acceptance_criteria_met",
            "scene_capture_cube_count": int(scene_capture_cube_count),
            "geometry_cache_actor_count": int(geometry_cache_actor_count),
            "static_mesh_actor_count": int(static_mesh_actor_count),
            "lighting_sky_actor_count": int(lighting_sky_actor_count),
            "usd_stage_actor_count": int(len(stage_actors)),
            "usd_stage_root_layer_matches_expected": bool(usd_stage_root_layer_matches_expected),
            "usd_world_child_exists": bool(usd_world_child_exists),
            "probe_summary": {
                "actor_count": actor_count,
                "component_count": component_count,
                "material_slots": material_slot_count,
            },
        })
        raise RuntimeError(
            "PRELOADED_EDITOR_MAP_WORLD_NOT_POPULATED: "
            f"actor_count={actor_count} component_count={component_count} material_slots={material_slot_count} "
            f"map={preferred_editor_map}"
        )
    return report

def _ensure_usd_stage(expected_stage_path, preferred_editor_map, post_bind_wait_seconds, spawn_usd_stage_if_missing, preloaded_world_min_actor_count, preloaded_world_min_component_count, preloaded_world_min_material_slot_count):
    try:
        import unreal
    except Exception as exc:
        _log("USD_STAGE_BOOTSTRAP_IMPORT_FAILED", {"error": str(exc)})
        raise RuntimeError(f"Could not import unreal module: {exc}")
    before_map = _get_current_map_info(unreal)
    stage_actors_before = _find_usd_stage_actors(unreal)
    _log("USD_STAGE_BOOTSTRAP_START", {
        "current_map": before_map,
        "usd_stage_actor_count_before_rebind": len(stage_actors_before),
        "preferred_editor_map": preferred_editor_map,
        "expected_stage_path": expected_stage_path,
    })
    opened_map_result = None
    if preferred_editor_map and not _editor_map_paths_match((before_map or {}).get("world_path"), preferred_editor_map):
        open_attempts = []
        for retry_index in range(3):
            opened_map_result = _open_editor_map(unreal, preferred_editor_map)
            open_attempts.append(opened_map_result)
            current_map_after_open = _get_current_map_info(unreal)
            _log("USD_STAGE_BOOTSTRAP_MAP_OPEN", {
                "requested_map": preferred_editor_map,
                "attempt_index": int(retry_index),
                "open_result": opened_map_result,
                "current_map_after_open": current_map_after_open,
            })
            if _editor_map_paths_match((current_map_after_open or {}).get("world_path"), preferred_editor_map):
                break
            _wait_after_usd_bind(2.0)
            _invalidate_editor_viewports(unreal)
        current_map_after_retries = _get_current_map_info(unreal)
        if not _editor_map_paths_match((current_map_after_retries or {}).get("world_path"), preferred_editor_map):
            raise RuntimeError(
                "PREFERRED_EDITOR_MAP_OPEN_FAILED: "
                + json.dumps({
                    "requested_map": preferred_editor_map,
                    "current_map": current_map_after_retries,
                    "attempts": open_attempts,
                }, indent=2, default=str)
            )
    elif preferred_editor_map:
        _log("USD_STAGE_BOOTSTRAP_MAP_ALREADY_OPEN", {
            "requested_map": preferred_editor_map,
            "current_map": before_map,
        })
    stage_actors = _find_usd_stage_actors(unreal)
    if stage_actors:
        _log("USD_STAGE_PRELOADED_ACTOR_INSPECTION", {
            "current_map": _get_current_map_info(unreal),
            "usd_stage_actor_count": len(stage_actors),
            "usd_stage_actor_inspection": _inspect_usd_stage_actor(stage_actors[0]),
            "world_probe_before_population_check": _collect_usd_material_readiness_probe(unreal),
            "purposes_to_load_diagnostic": _purposes_to_load_diagnostic(
                stage_actors[0],
                min(max(float(post_bind_wait_seconds or 0.0), 10.0), 20.0),
            ),
        })
    if preferred_editor_map:
        _ensure_preloaded_world_populated(
            unreal,
            preferred_editor_map,
            expected_stage_path,
            preloaded_world_min_actor_count,
            preloaded_world_min_component_count,
            preloaded_world_min_material_slot_count,
        )
    spawned_new_actor = False
    spawn_error = None
    actor = stage_actors[0] if stage_actors else None
    if actor is None and bool(spawn_usd_stage_if_missing):
        actor, spawn_error = _spawn_usd_stage_actor(unreal)
        spawned_new_actor = actor is not None
        _log("USD_STAGE_BOOTSTRAP_SPAWN", {
            "spawned_new_usd_stage_actor": bool(spawned_new_actor),
            "spawn_error": spawn_error,
            "current_map": _get_current_map_info(unreal),
        })
    if actor is None:
        raise RuntimeError(
            "No UsdStageActor found"
            + (
                " and spawning is disabled in this mode. "
                if not bool(spawn_usd_stage_if_missing)
                else " or spawned in the current level. "
            )
            + f"Expected loaded USD scene: {expected_stage_path}"
        )
    bind_report = _bind_usd_stage_actor(actor, expected_stage_path)
    wait_report = _wait_after_usd_bind(post_bind_wait_seconds)
    reload_attempts = _reload_usd_stage_actor(actor)
    validation = _validate_usd_stage_access(actor)
    actor_inspection = _inspect_usd_stage_actor(actor)
    generated_assets_report = _safe_generated_assets_report(actor)
    world_probe_after_bind = _collect_usd_material_readiness_probe(unreal)
    population_attempts = _post_load_population_diagnostics(
        actor,
        expected_stage_path,
        min(max(float(post_bind_wait_seconds or 0.0), 5.0), 10.0),
    )
    root_layer_matches_expected = _root_layer_matches_expected(
        bind_report.get("final_root_layer"),
        expected_stage_path,
    )
    final_report = {
        "current_map": _get_current_map_info(unreal),
        "usd_stage_actor_count_before_rebind": len(stage_actors_before),
        "usd_stage_actor_count_final": len(_find_usd_stage_actors(unreal)),
        "spawned_new_usd_stage_actor": bool(spawned_new_actor),
        "spawn_usd_stage_if_missing": bool(spawn_usd_stage_if_missing),
        "actor_label": actor.get_actor_label() if hasattr(actor, "get_actor_label") else None,
        "actor_name": actor.get_name() if hasattr(actor, "get_name") else None,
        "final_root_layer": bind_report.get("final_root_layer"),
        "root_layer_matches_expected": bool(root_layer_matches_expected),
        "file_path_attempts": bind_report.get("file_path_attempts"),
        "bind_attempts": bind_report.get("attempts"),
        "post_bind_wait": wait_report,
        "reload_attempts": reload_attempts,
        "post_edit_error": bind_report.get("post_edit_error"),
        "validation_before_warmup": validation,
        "generated_assets_report": generated_assets_report,
        "world_probe_after_bind": world_probe_after_bind,
        "post_load_population_attempts": population_attempts,
        "usd_stage_actor_inspection": actor_inspection,
    }
    _log("USD_STAGE_BOOTSTRAP_FINAL", final_report)
    if not root_layer_matches_expected:
        raise RuntimeError(
            "UsdStageActor exists but final root_layer does not match expected USD stage path. "
            f"Expected loaded USD scene: {expected_stage_path}"
        )
    if not validation.get("loaded_stage_accessible"):
        _log("USD_STAGE_ACCESS_WARNING", {
            "warning": "USD stage accessor unavailable after bootstrap; continuing to warm-up because actor exists and root_layer matches expected path.",
            "expected_stage_path": expected_stage_path,
            "final_root_layer": bind_report.get("final_root_layer"),
            "validation_before_warmup": validation,
        })
    return final_report

def _quit_editor():
    try:
        import unreal
    except Exception:
        return
    for accessor in ("quit_editor",):
        fn = getattr(unreal.SystemLibrary, accessor, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass
    try:
        unreal.SystemLibrary.execute_console_command(None, "QUIT_EDITOR")
    except Exception:
        pass

try:
    import unreal
    expected_stage_path = str(PAYLOAD["scene_root"]) + "/usd_export/export_scene.blend/export_scene.usdc"
    __USD_MATERIAL_MAX_FALLBACK_RATIO__ = PAYLOAD.get("usd_material_max_fallback_ratio", 0.05)
    __USD_MATERIAL_STABLE_POLLS__ = PAYLOAD.get("usd_material_stable_polls", 2)
    _ensure_usd_stage(
        expected_stage_path,
        PAYLOAD.get("preferred_editor_map"),
        PAYLOAD.get("usd_stage_post_bind_wait_seconds"),
        PAYLOAD.get("spawn_usd_stage_if_missing"),
        PAYLOAD.get("preloaded_world_min_actor_count", 20),
        PAYLOAD.get("preloaded_world_min_component_count", 20),
        PAYLOAD.get("preloaded_world_min_material_slot_count", 20),
    )
    _wait_for_usd_material_readiness(
        expected_stage_path,
        PAYLOAD.get("usd_material_readiness_timeout_seconds"),
        PAYLOAD.get("usd_material_readiness_poll_seconds"),
        PAYLOAD.get("allow_usd_material_readiness_timeout"),
    )
    if PAYLOAD.get("open_map_only"):
        open_map_report = {
            "current_map": _get_current_map_info(unreal),
            "preferred_editor_map": PAYLOAD.get("preferred_editor_map"),
            "expected_stage_path": expected_stage_path,
            "sleep_seconds": float(PAYLOAD.get("open_map_only_sleep_seconds", 600.0) or 0.0),
            "no_quit": bool(PAYLOAD.get("open_map_only_no_quit")),
            "pre_render_state": _collect_pre_render_state(unreal, "open_map_only"),
        }
        _log("OPEN_MAP_ONLY_READY", open_map_report)
        sleep_seconds = float(PAYLOAD.get("open_map_only_sleep_seconds", 600.0) or 0.0)
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)
        _log("OPEN_MAP_ONLY_DONE", {
            "slept_seconds": sleep_seconds,
            "no_quit": bool(PAYLOAD.get("open_map_only_no_quit")),
        })
        if bool(PAYLOAD.get("open_map_only_no_quit")):
            raise SystemExit(0)
        _quit_editor()
        raise SystemExit(0)
    renderer_script = PAYLOAD["renderer_script"]
    if PAYLOAD.get("enable_session_warmup") and PAYLOAD.get("clip_ids"):
        warmup_clip_id = list(PAYLOAD.get("clip_ids"))[0]
        warmup_before_runs = [str(path) for path in _list_run_dirs(PAYLOAD["runs_root"])]
        warmup_argv = [
            renderer_script,
            "--scene-root", PAYLOAD["scene_root"],
            "--manifest", PAYLOAD["manifest_path"],
            "--miniscene-id", str(warmup_clip_id),
            "--frame-start", str(PAYLOAD["frame_start"]),
            "--frame-end", str(PAYLOAD["frame_end"]),
            "--render-output-profile", PAYLOAD["render_output_profile"],
            "--rgb-tonemap-mode", PAYLOAD["rgb_tonemap_mode"],
        ]
        _log("SESSION_WARMUP_POLICY", {
            "same_unreal_process_as_real_render": True,
            "warmup_manifest_path": PAYLOAD["manifest_path"],
            "warmup_miniscene_id": warmup_clip_id,
            "warmup_frame_start": PAYLOAD.get("frame_start"),
            "warmup_frame_end": PAYLOAD.get("frame_end"),
            "warmup_matches_real_clip_command": True,
            "warmup_not_counted_as_dataset_completion": True,
            "internal_capture_warmup_disabled_for_session_warmup": True,
            "reason": "first BEDLAM render in fresh Unreal session is appearance warm-up; rerendering the same official clip in the same session is expected to be visually correct",
        })
        _log("PRE_WARMUP_RENDER_STATE", _collect_pre_render_state(unreal, "pre_warmup"))
        _log("SESSION_WARMUP_START", {'argv': warmup_argv})
        sys.argv = warmup_argv
        runpy.run_path(renderer_script, run_name="__main__")
        warmup_new_runs = [str(path) for path in _new_run_dirs_after(warmup_before_runs, PAYLOAD["runs_root"])]
        warmup_deleted_runs = []
        warmup_delete_errors = []
        for run_dir in warmup_new_runs:
            try:
                shutil.rmtree(run_dir)
                warmup_deleted_runs.append(run_dir)
            except Exception as exc:
                warmup_delete_errors.append({"run_dir": run_dir, "error": str(exc)})
        _log("SESSION_WARMUP_DONE", {
            'warmup_miniscene_id': warmup_clip_id,
            'warmup_new_run_dirs': warmup_new_runs,
            'warmup_deleted_run_dirs': warmup_deleted_runs,
            'warmup_delete_errors': warmup_delete_errors,
        })
        _log("SESSION_WARMUP_POST_DELAY_START", {'delay_seconds': PAYLOAD.get("post_warmup_delay_seconds", 0.0)})
        warmup_wait_report = _wait_after_warmup(PAYLOAD.get("post_warmup_delay_seconds"))
        _log("SESSION_WARMUP_POST_DELAY_DONE", warmup_wait_report)
    for miniscene_id in PAYLOAD["clip_ids"]:
        argv = [
            renderer_script,
            "--scene-root", PAYLOAD["scene_root"],
            "--manifest", PAYLOAD["manifest_path"],
            "--miniscene-id", miniscene_id,
            "--frame-start", str(PAYLOAD["frame_start"]),
            "--frame-end", str(PAYLOAD["frame_end"]),
            "--render-output-profile", PAYLOAD["render_output_profile"],
            "--rgb-tonemap-mode", PAYLOAD["rgb_tonemap_mode"],
        ]
        session_warmup_enabled = bool(PAYLOAD.get("enable_session_warmup"))
        if not session_warmup_enabled:
            argv.extend([
                "--render-warmup-frame-count", str(PAYLOAD.get("render_warmup_frame_count", 0)),
            ])
            if bool(PAYLOAD.get("discard_warmup_frames")):
                argv.append("--discard-warmup-frames")
        if bool(PAYLOAD.get("probe_frame_before_render")):
            argv.append("--probe-frame-before-render")
        if bool(PAYLOAD.get("reject_beige_probe")):
            argv.append("--reject-beige-probe")
        if PAYLOAD.get("pause_after_spawn_before_render"):
            argv.append("--pause-after-spawn-before-render")
        _log("PRE_CLIP_RENDER_STATE", _collect_pre_render_state(unreal, f"pre_clip:{miniscene_id}"))
        _log("CLIP_RENDER_START", {'miniscene_id': miniscene_id, 'argv': argv})
        sys.argv = argv
        runpy.run_path(renderer_script, run_name="__main__")
        if PAYLOAD.get("pause_after_spawn_before_render"):
            _log("CLIP_RENDER_PAUSED_BEFORE_RENDER", {'miniscene_id': miniscene_id, 'editor_left_open_for_manual_inspection': True})
            raise SystemExit(0)
        _log("CLIP_RENDER_COMPLETE", {'miniscene_id': miniscene_id})
    _log("CHUNK_COMPLETE", PAYLOAD)
except Exception as exc:
    _log("CHUNK_EXCEPTION", {
        'error': str(exc),
        'traceback': traceback.format_exc(),
        'payload': PAYLOAD,
    })
    _quit_editor()
    raise SystemExit(17)
    if not bool(PAYLOAD.get("pause_after_spawn_before_render")):
        _quit_editor()
"""
    launcher_code = launcher_template.replace("__PAYLOAD__", repr(payload))
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_path.write_text(launcher_code, encoding="utf-8")


def _build_bootstrap_render_entrypoint(
    launcher_path: Path,
    bootstrap_script: Path,
    renderer_entry_script: Path,
    bootstrap_helper_renderer_script: Path,
    scene_root: Path,
    manifest_path: Path,
    preferred_editor_map: str | None,
    bootstrap_wait_ticks_after_asset_registry: int,
    bootstrap_wait_ticks_after_usd_load: int,
    bootstrap_max_ticks: int,
    bootstrap_log_prefix: str,
):
    argv = [
        str(bootstrap_script),
        "--renderer-script", str(renderer_entry_script),
        "--bootstrap-helper-renderer-script", str(bootstrap_helper_renderer_script),
        "--bootstrap-wait-ticks-after-asset-registry", str(int(bootstrap_wait_ticks_after_asset_registry)),
        "--bootstrap-wait-ticks-after-usd-load", str(int(bootstrap_wait_ticks_after_usd_load)),
        "--bootstrap-max-ticks", str(int(bootstrap_max_ticks)),
        "--bootstrap-log-prefix", str(bootstrap_log_prefix),
    ]
    if preferred_editor_map not in (None, ""):
        argv.extend(["--preferred-editor-map", str(preferred_editor_map)])
    argv.extend([
        "--",
        "--scene-root", str(scene_root),
        "--manifest", str(manifest_path),
    ])
    launcher_code = f"""print("[BEDLAM360_V3_RESILIENT_CHUNK] RESILIENT_LAUNCHER_PYTHON_ENTERED")
import runpy
import sys

print("[BEDLAM360_V3_RESILIENT_CHUNK] RESILIENT_LAUNCHER_IMPORTS_DONE")
sys.argv = {repr(argv)}
runpy.run_path({repr(str(bootstrap_script))}, run_name="__main__")
"""
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_path.write_text(launcher_code, encoding="utf-8")


def _build_unreal_probe_launcher(launcher_path: Path):
    launcher_code = """print("PYTHON_PROBE_START")
import unreal
print("PYTHON_PROBE_IMPORT_OK")
print("PYTHON_PROBE_END")
"""
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_path.write_text(launcher_code, encoding="utf-8")


def _launch_unreal_chunk(
    unreal_editor: Path,
    uproject: Path,
    launcher_path: Path,
    log_path: Path,
    extra_unreal_args,
    use_exec_cmds_py: bool = False,
    disable_texture_streaming_on_launch: bool = False,
    resource_monitor_csv_path: Path = None,
    resource_monitor_interval_seconds: float = DEFAULT_RESOURCE_MONITOR_INTERVAL_SECONDS,
    disable_resource_monitor: bool = False,
    dry_run=False,
):
    command = [
        str(unreal_editor),
        str(uproject),
    ]
    if bool(use_exec_cmds_py):
        command.append(f"-ExecCmds=py {launcher_path}")
    else:
        command.append(f"-ExecutePythonScript={launcher_path}")
    command.extend([
        "-NoSplash",
        "-Unattended",
        "-StdOut",
        "-FullStdOutLogOutput",
    ])
    if bool(disable_texture_streaming_on_launch):
        command.append("-NoTextureStreaming")
    if extra_unreal_args:
        command.extend(str(arg) for arg in extra_unreal_args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return {
            "attempted": False,
            "command": _shell_join(command),
            "log_path": str(log_path),
            "returncode": None,
            "resource_monitor_csv_path": None if resource_monitor_csv_path is None else str(resource_monitor_csv_path),
        }
    started = time.perf_counter()
    stop_event = threading.Event()
    monitor_thread = None
    resource_summary = None
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# COMMAND\n")
        handle.write(_shell_join(command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
        if not disable_resource_monitor and resource_monitor_csv_path is not None:
            monitor_thread = threading.Thread(
                target=_resource_monitor_loop,
                args=(process, resource_monitor_csv_path, launcher_path.stem, float(resource_monitor_interval_seconds), stop_event),
                daemon=True,
            )
            monitor_thread.start()
        returncode = process.wait()
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=max(float(resource_monitor_interval_seconds) * 2.0, 1.0))
    duration = float(time.perf_counter() - started)
    if not disable_resource_monitor and resource_monitor_csv_path is not None:
        resource_summary = _summarize_resource_csv(resource_monitor_csv_path)
    return {
        "attempted": True,
        "command": _shell_join(command),
        "log_path": str(log_path),
        "returncode": int(returncode),
        "duration_seconds": duration,
        "resource_monitor_csv_path": None if resource_monitor_csv_path is None else str(resource_monitor_csv_path),
        "resource_monitor_interval_seconds": float(resource_monitor_interval_seconds),
        "resource_monitor_disabled": bool(disable_resource_monitor),
        "resource_summary": resource_summary,
    }


def _compose_resume_report(
    scene_root: Path,
    manifest_path: Path,
    clip_ids,
    statuses,
    launch_history,
    frame_start: int,
    frame_end: int,
    chunk_size: int,
    max_retries_per_clip: int,
    resource_logs_dir: Path = None,
):
    completed = []
    failed = []
    partial = []
    missing_preview_only = []
    remaining = []
    for clip_id in clip_ids:
        state = statuses.get(clip_id, {}).get("state")
        if state == "completed":
            completed.append(clip_id)
        elif state == "failed":
            failed.append(clip_id)
        elif state == "partial":
            partial.append(clip_id)
        elif state == "missing_preview_only":
            missing_preview_only.append(clip_id)
        else:
            remaining.append(clip_id)
    return {
        "scene_root": str(scene_root),
        "manifest_path": str(manifest_path),
        "requested_frame_start": int(frame_start),
        "requested_frame_end": int(frame_end),
        "chunk_size": int(chunk_size),
        "max_retries_per_clip": int(max_retries_per_clip),
        "total_clip_count": int(len(clip_ids)),
        "completed_clips": completed,
        "failed_clips": failed,
        "partial_clips": partial,
        "missing_preview_only_clips": missing_preview_only,
        "remaining_clips": remaining,
        "clip_statuses": statuses,
        "launch_history": launch_history,
        "resource_logs": (
            _write_resource_summary(resource_logs_dir)
            if resource_logs_dir is not None
            else None
        ),
        "generated_at_unix": time.time(),
    }


def _compose_probe_report(
    scene_root: Path,
    manifest_path: Path,
    launch_history,
):
    attempted = len(launch_history)
    start_count = 0
    end_count = 0
    before_python_count = 0
    exit_codes = []
    for row in launch_history:
        markers = set(row.get("probe_markers_found") or [])
        if "PYTHON_PROBE_START" in markers:
            start_count += 1
        if "PYTHON_PROBE_END" in markers:
            end_count += 1
        if row.get("probe_failure_stage") == "startup_before_python":
            before_python_count += 1
        exit_codes.append(row.get("returncode"))
    return {
        "mode": "probe_only",
        "scene_root": str(scene_root),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "launches_attempted": attempted,
        "launches_reaching_python_probe_start": start_count,
        "launches_reaching_python_probe_end": end_count,
        "launches_failing_before_python": before_python_count,
        "unreal_exit_codes": exit_codes,
        "launch_history": launch_history,
        "generated_at_unix": time.time(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=False, default=None)
    parser.add_argument("--unreal-editor", type=Path, default=DEFAULT_UNREAL_EDITOR)
    parser.add_argument("--uproject", type=Path, default=DEFAULT_UPROJECT)
    parser.add_argument("--renderer-script", type=Path, default=DEFAULT_RENDERER_SCRIPT)
    parser.add_argument("--bootstrap-render-launcher-script", type=Path, default=DEFAULT_BOOTSTRAP_RENDER_LAUNCHER_SCRIPT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--batch-report-root", type=Path, default=DEFAULT_BATCH_REPORT_ROOT)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=120)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-retries-per-clip", type=int, default=DEFAULT_MAX_RETRIES_PER_CLIP)
    parser.add_argument("--render-output-profile", type=str, default=DEFAULT_RENDER_OUTPUT_PROFILE)
    parser.add_argument("--rgb-tonemap-mode", type=str, default=DEFAULT_RGB_TONEMAP_MODE)
    parser.add_argument("--preferred-editor-map", type=str, default=DEFAULT_EDITOR_MAP)
    parser.add_argument("--spawn-usd-stage-if-missing", action="store_true", default=False)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--generate-missing-previews-only", action="store_true", default=False)
    parser.add_argument("--force-preview-regeneration", action="store_true", default=False)
    parser.add_argument("--unreal-extra-arg", action="append", default=None)
    parser.add_argument("--disable-session-warmup", action="store_true", default=False)
    parser.add_argument("--warmup-frame-start", type=int, default=DEFAULT_WARMUP_FRAME_START)
    parser.add_argument("--warmup-frame-end", type=int, default=DEFAULT_WARMUP_FRAME_END)
    parser.add_argument(
        "--usd-stage-post-bind-wait-seconds",
        type=float,
        default=DEFAULT_USD_STAGE_POST_BIND_WAIT_SECONDS,
    )
    parser.add_argument(
        "--post-warmup-delay-seconds",
        type=float,
        default=DEFAULT_POST_WARMUP_DELAY_SECONDS,
    )
    parser.add_argument(
        "--unreal-startup-wait-seconds",
        type=float,
        default=DEFAULT_UNREAL_STARTUP_WAIT_SECONDS,
    )
    parser.add_argument("--disable-texture-streaming-on-launch", action="store_true", default=False)
    parser.add_argument("--render-warmup-frame-count", type=int, default=32)
    parser.add_argument("--discard-warmup-frames", action="store_true", default=False)
    parser.add_argument("--probe-frame-before-render", action="store_true", default=False)
    parser.add_argument("--reject-beige-probe", action="store_true", default=False)
    parser.add_argument(
        "--usd-material-readiness-timeout-seconds",
        type=float,
        default=DEFAULT_USD_MATERIAL_READINESS_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--usd-material-readiness-poll-seconds",
        type=float,
        default=DEFAULT_USD_MATERIAL_READINESS_POLL_SECONDS,
    )
    parser.add_argument(
        "--allow-usd-material-readiness-timeout",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--preloaded-world-min-actor-count",
        type=int,
        default=DEFAULT_PRELOADED_WORLD_MIN_ACTOR_COUNT,
    )
    parser.add_argument(
        "--preloaded-world-min-component-count",
        type=int,
        default=DEFAULT_PRELOADED_WORLD_MIN_COMPONENT_COUNT,
    )
    parser.add_argument(
        "--preloaded-world-min-material-slot-count",
        type=int,
        default=DEFAULT_PRELOADED_WORLD_MIN_MATERIAL_SLOT_COUNT,
    )
    parser.add_argument("--pause-after-spawn-before-render", action="store_true", default=False)
    parser.add_argument("--open-map-only", action="store_true", default=False)
    parser.add_argument("--open-map-only-sleep-seconds", type=float, default=DEFAULT_OPEN_MAP_ONLY_SLEEP_SECONDS)
    parser.add_argument("--open-map-only-no-quit", action="store_true", default=False)
    parser.add_argument("--resource-monitor-interval-seconds", type=float, default=DEFAULT_RESOURCE_MONITOR_INTERVAL_SECONDS)
    parser.add_argument("--disable-resource-monitor", action="store_true", default=False)
    parser.add_argument("--resource-monitor-log-name", type=str, default=DEFAULT_RESOURCE_MONITOR_LOG_NAME)
    parser.add_argument("--summarize-resource-logs", action="store_true", default=False)
    parser.add_argument("--probe-only", action="store_true", default=False)
    parser.add_argument("--probe-launch-count", type=int, default=DEFAULT_PROBE_LAUNCH_COUNT)
    parser.add_argument("--use-bootstrap-render-launcher", action="store_true", default=False)
    parser.add_argument("--bootstrap-wait-ticks-after-asset-registry", type=int, default=30)
    parser.add_argument("--bootstrap-wait-ticks-after-usd-load", type=int, default=120)
    parser.add_argument("--bootstrap-max-ticks", type=int, default=3000)
    parser.add_argument("--bootstrap-log-prefix", type=str, default="BEDLAM360_BOOTSTRAP")
    args = parser.parse_args()

    scene_root = args.scene_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve() if args.manifest is not None else None
    unreal_editor = args.unreal_editor.expanduser().resolve()
    uproject = args.uproject.expanduser().resolve()
    renderer_script = args.renderer_script.expanduser().resolve()
    bootstrap_render_launcher_script = args.bootstrap_render_launcher_script.expanduser().resolve()
    if args.use_bootstrap_render_launcher and not bootstrap_render_launcher_script.exists():
        raise SystemExit(
            f"--use-bootstrap-render-launcher requested, but bootstrap script does not exist: "
            f"{bootstrap_render_launcher_script}"
        )
    runs_root = args.runs_root.expanduser().resolve()
    resume_dir = (
        args.resume_dir.expanduser().resolve()
        if args.resume_dir is not None
        else (scene_root / "miniscene_selection_v0" / "resilient_render_runner")
    )
    resume_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = resume_dir / "logs"
    launcher_dir = resume_dir / "launchers"
    warmup_manifest_dir = resume_dir / "warmup_manifests"
    resource_logs_dir = resume_dir / "resource_logs"
    report_path = resume_dir / "v3_resilient_render_resume_report.json"

    if args.summarize_resource_logs:
        summary = _write_resource_summary(resource_logs_dir)
        print(json.dumps(summary, indent=2))
        return

    if manifest_path is None:
        raise SystemExit("--manifest is required unless --summarize-resource-logs is used")

    if args.probe_only:
        probe_report_path = resume_dir / "v3_resilient_render_probe_report.json"
        probe_launcher_dir = resume_dir / "probe_launchers"
        probe_logs_dir = resume_dir / "probe_logs"
        probe_history = []
        for probe_index in range(max(int(args.probe_launch_count), 1)):
            probe_label = f"probe_{probe_index:03d}"
            launcher_path = probe_launcher_dir / f"{probe_label}.py"
            log_path = probe_logs_dir / f"{probe_label}.log"
            _build_unreal_probe_launcher(launcher_path)
            launch_row = _launch_unreal_chunk(
                unreal_editor=unreal_editor,
                uproject=uproject,
                launcher_path=launcher_path,
                log_path=log_path,
                extra_unreal_args=args.unreal_extra_arg or [],
                dry_run=args.dry_run,
            )
            probe_diag = _parse_probe_diagnostics(log_path)
            launch_row["probe_index"] = probe_index
            launch_row["probe_markers_found"] = probe_diag.get("markers_found")
            launch_row["last_seen_marker"] = probe_diag.get("last_seen_marker")
            launch_row["probe_failure_stage"] = probe_diag.get("failure_stage")
            launch_row["log_tail"] = probe_diag.get("log_tail")
            probe_history.append(launch_row)
            probe_report = _compose_probe_report(scene_root, manifest_path, probe_history)
            _write_json(probe_report_path, probe_report)
            if args.dry_run:
                break
        final_probe_report = _compose_probe_report(scene_root, manifest_path, probe_history)
        _write_json(probe_report_path, final_probe_report)
        print(json.dumps(final_probe_report, indent=2))
        return

    manifest_payload, miniscenes = _load_manifest(manifest_path)
    clip_ids = _clip_ids_in_order(miniscenes)
    if args.start_index:
        clip_ids = clip_ids[int(args.start_index) :]
    if args.limit is not None:
        clip_ids = clip_ids[: int(args.limit)]

    attempt_counts = defaultdict(int)
    launch_history = []

    while True:
        statuses = _apply_attempt_failure_overrides(_scan_clip_statuses(
            runs_root,
            manifest_path,
            clip_ids,
            args.frame_start,
            args.frame_end,
        ), attempt_counts, launch_history)
        preview_generated_rows = []
        for clip_id, row in statuses.items():
            if row.get("state") != "missing_preview_only":
                continue
            result = _generate_preview_rgb_mp4(
                Path(row["run_dir"]),
                _read_json(Path(row["manifest_json_path"])),
                force=args.force_preview_regeneration,
            )
            preview_generated_rows.append({"miniscene_id": clip_id, **result})
        if preview_generated_rows:
            statuses = _apply_attempt_failure_overrides(_scan_clip_statuses(
                runs_root,
                manifest_path,
                clip_ids,
                args.frame_start,
                args.frame_end,
            ), attempt_counts, launch_history)
        report = _compose_resume_report(
            scene_root,
            manifest_path,
            clip_ids,
            statuses,
            launch_history,
            args.frame_start,
            args.frame_end,
            args.chunk_size,
            args.max_retries_per_clip,
            resource_logs_dir=resource_logs_dir,
        )
        report["preview_generation_results"] = preview_generated_rows
        _write_json(report_path, report)
        if args.generate_missing_previews_only:
            break

        remaining = [
            clip_id
            for clip_id in clip_ids
            if statuses.get(clip_id, {}).get("state") not in {"completed"}
            and attempt_counts[clip_id] < int(args.max_retries_per_clip)
        ]
        if not remaining:
            break

        chunk = remaining[: max(int(args.chunk_size), 1)]
        if not args.dry_run:
            for clip_id in chunk:
                attempt_counts[clip_id] += 1
        chunk_label = f"chunk_{len(launch_history):03d}"
        launcher_path = launcher_dir / f"{chunk_label}.py"
        log_path = logs_dir / f"{chunk_label}.log"
        resource_monitor_csv_path = resource_logs_dir / f"{chunk_label}_resource_monitor.csv"
        warmup_manifest_path = None
        if not args.disable_session_warmup:
            warmup_manifest_path = warmup_manifest_dir / f"{chunk_label}.warmup_manifest.json"
            _write_json(
                warmup_manifest_path,
                _build_warmup_manifest(manifest_payload, chunk[0]),
            )
        direct_renderer_launcher_path = launcher_path
        unreal_entrypoint_path = launcher_path
        if args.use_bootstrap_render_launcher:
            direct_renderer_launcher_path = launcher_dir / f"{chunk_label}.direct_renderer.py"
            unreal_entrypoint_path = launcher_dir / f"{chunk_label}.bootstrap_entrypoint.py"
        _build_unreal_chunk_launcher(
            launcher_path=direct_renderer_launcher_path,
            renderer_script=renderer_script,
            scene_root=scene_root,
            manifest_path=manifest_path,
            warmup_manifest_path=warmup_manifest_path,
            runs_root=runs_root,
            clip_ids=chunk,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            render_output_profile=args.render_output_profile,
            rgb_tonemap_mode=args.rgb_tonemap_mode,
            preferred_editor_map=args.preferred_editor_map,
            spawn_usd_stage_if_missing=args.spawn_usd_stage_if_missing,
            enable_session_warmup=not args.disable_session_warmup,
            warmup_frame_start=args.warmup_frame_start,
            warmup_frame_end=args.warmup_frame_end,
            usd_stage_post_bind_wait_seconds=args.usd_stage_post_bind_wait_seconds,
            post_warmup_delay_seconds=args.post_warmup_delay_seconds,
            unreal_startup_wait_seconds=args.unreal_startup_wait_seconds,
            render_warmup_frame_count=args.render_warmup_frame_count,
            discard_warmup_frames=args.discard_warmup_frames,
            probe_frame_before_render=args.probe_frame_before_render,
            reject_beige_probe=args.reject_beige_probe,
            usd_material_readiness_timeout_seconds=args.usd_material_readiness_timeout_seconds,
            usd_material_readiness_poll_seconds=args.usd_material_readiness_poll_seconds,
            allow_usd_material_readiness_timeout=args.allow_usd_material_readiness_timeout,
            preloaded_world_min_actor_count=args.preloaded_world_min_actor_count,
            preloaded_world_min_component_count=args.preloaded_world_min_component_count,
            preloaded_world_min_material_slot_count=args.preloaded_world_min_material_slot_count,
            pause_after_spawn_before_render=args.pause_after_spawn_before_render,
            open_map_only=args.open_map_only,
            open_map_only_sleep_seconds=args.open_map_only_sleep_seconds,
            open_map_only_no_quit=args.open_map_only_no_quit,
        )
        if args.use_bootstrap_render_launcher:
            _build_bootstrap_render_entrypoint(
                launcher_path=unreal_entrypoint_path,
                bootstrap_script=bootstrap_render_launcher_script,
                renderer_entry_script=direct_renderer_launcher_path,
                bootstrap_helper_renderer_script=renderer_script,
                scene_root=scene_root,
                manifest_path=manifest_path,
                preferred_editor_map=args.preferred_editor_map,
                bootstrap_wait_ticks_after_asset_registry=args.bootstrap_wait_ticks_after_asset_registry,
                bootstrap_wait_ticks_after_usd_load=args.bootstrap_wait_ticks_after_usd_load,
                bootstrap_max_ticks=args.bootstrap_max_ticks,
                bootstrap_log_prefix=args.bootstrap_log_prefix,
            )
        launch_row = _launch_unreal_chunk(
            unreal_editor=unreal_editor,
            uproject=uproject,
            launcher_path=unreal_entrypoint_path,
            log_path=log_path,
            extra_unreal_args=args.unreal_extra_arg or [],
            use_exec_cmds_py=bool(args.use_bootstrap_render_launcher),
            disable_texture_streaming_on_launch=args.disable_texture_streaming_on_launch,
            resource_monitor_csv_path=resource_monitor_csv_path,
            resource_monitor_interval_seconds=args.resource_monitor_interval_seconds,
            disable_resource_monitor=args.disable_resource_monitor,
            dry_run=args.dry_run,
        )
        launcher_diag = _parse_launcher_stage_diagnostics(log_path)
        launch_row["chunk_clip_ids"] = list(chunk)
        launch_row["attempt_counts"] = {clip_id: int(attempt_counts.get(clip_id, 0)) for clip_id in chunk}
        launch_row["session_warmup_enabled"] = bool(not args.disable_session_warmup)
        launch_row["warmup_manifest_path"] = None if warmup_manifest_path is None else str(warmup_manifest_path)
        launch_row["warmup_frame_start"] = int(args.warmup_frame_start)
        launch_row["warmup_frame_end"] = int(args.warmup_frame_end)
        launch_row["usd_stage_post_bind_wait_seconds"] = float(args.usd_stage_post_bind_wait_seconds)
        launch_row["unreal_startup_wait_seconds"] = float(args.unreal_startup_wait_seconds)
        launch_row["usd_material_readiness_timeout_seconds"] = float(args.usd_material_readiness_timeout_seconds)
        launch_row["usd_material_readiness_poll_seconds"] = float(args.usd_material_readiness_poll_seconds)
        launch_row["disable_texture_streaming_on_launch"] = bool(args.disable_texture_streaming_on_launch)
        launch_row["render_warmup_frame_count"] = int(args.render_warmup_frame_count)
        launch_row["discard_warmup_frames"] = bool(args.discard_warmup_frames)
        launch_row["probe_frame_before_render"] = bool(args.probe_frame_before_render)
        launch_row["reject_beige_probe"] = bool(args.reject_beige_probe)
        launch_row["preferred_editor_map"] = str(args.preferred_editor_map) if args.preferred_editor_map else None
        launch_row["spawn_usd_stage_if_missing"] = bool(args.spawn_usd_stage_if_missing)
        launch_row["preloaded_world_min_actor_count"] = int(args.preloaded_world_min_actor_count)
        launch_row["preloaded_world_min_component_count"] = int(args.preloaded_world_min_component_count)
        launch_row["preloaded_world_min_material_slot_count"] = int(args.preloaded_world_min_material_slot_count)
        launch_row["allow_usd_material_readiness_timeout"] = bool(args.allow_usd_material_readiness_timeout)
        launch_row["pause_after_spawn_before_render"] = bool(args.pause_after_spawn_before_render)
        launch_row["open_map_only"] = bool(args.open_map_only)
        launch_row["open_map_only_sleep_seconds"] = float(args.open_map_only_sleep_seconds)
        launch_row["open_map_only_no_quit"] = bool(args.open_map_only_no_quit)
        launch_row["use_bootstrap_render_launcher"] = bool(args.use_bootstrap_render_launcher)
        launch_row["bootstrap_render_launcher_script"] = str(bootstrap_render_launcher_script)
        launch_row["bootstrap_wait_ticks_after_asset_registry"] = int(args.bootstrap_wait_ticks_after_asset_registry)
        launch_row["bootstrap_wait_ticks_after_usd_load"] = int(args.bootstrap_wait_ticks_after_usd_load)
        launch_row["bootstrap_max_ticks"] = int(args.bootstrap_max_ticks)
        launch_row["bootstrap_log_prefix"] = str(args.bootstrap_log_prefix)
        launch_row["unreal_entrypoint_path"] = str(unreal_entrypoint_path)
        launch_row["direct_renderer_launcher_path"] = str(direct_renderer_launcher_path)
        resource_summary = dict(launch_row.get("resource_summary") or {})
        launch_row["max_unreal_rss_bytes"] = resource_summary.get("max_unreal_rss_bytes")
        launch_row["max_system_ram_used_bytes"] = resource_summary.get("max_system_ram_used_bytes")
        launch_row["max_system_swap_used_bytes"] = resource_summary.get("max_system_swap_used_bytes")
        launch_row["max_gpu_memory_used_mb"] = resource_summary.get("max_gpu_memory_used_mb")
        launch_row["max_unreal_gpu_memory_used_mb"] = resource_summary.get("max_unreal_gpu_memory_used_mb")
        launch_row["attempt_index"] = len(launch_history)
        launch_row["chunk_index"] = len(launch_history)
        launch_row["launcher_markers_found"] = launcher_diag.get("markers_found")
        launch_row["last_seen_marker"] = launcher_diag.get("last_seen_marker")
        launch_row["failure_stage"] = launcher_diag.get("failure_stage")
        launch_row["failure_reason"] = launcher_diag.get("failure_reason")
        launch_row["log_tail"] = launcher_diag.get("log_tail")
        launch_history.append(launch_row)
        _write_json(
            report_path,
            _compose_resume_report(
                scene_root,
                manifest_path,
                clip_ids,
                _apply_attempt_failure_overrides(_scan_clip_statuses(
                    runs_root,
                    manifest_path,
                    clip_ids,
                    args.frame_start,
                    args.frame_end,
                ), attempt_counts, launch_history),
                launch_history,
                args.frame_start,
                args.frame_end,
                args.chunk_size,
                args.max_retries_per_clip,
                resource_logs_dir=resource_logs_dir,
            ),
        )
        if args.dry_run:
            break

    final_statuses = _apply_attempt_failure_overrides(_scan_clip_statuses(
        runs_root,
        manifest_path,
        clip_ids,
        args.frame_start,
        args.frame_end,
    ), attempt_counts, launch_history)
    final_preview_rows = []
    for clip_id, row in final_statuses.items():
        if row.get("state") != "missing_preview_only":
            continue
        result = _generate_preview_rgb_mp4(
            Path(row["run_dir"]),
            _read_json(Path(row["manifest_json_path"])),
            force=args.force_preview_regeneration,
        )
        final_preview_rows.append({"miniscene_id": clip_id, **result})
    final_statuses = _apply_attempt_failure_overrides(_scan_clip_statuses(
        runs_root,
        manifest_path,
        clip_ids,
        args.frame_start,
        args.frame_end,
    ), attempt_counts, launch_history)
    final_report = _compose_resume_report(
        scene_root,
        manifest_path,
        clip_ids,
        final_statuses,
        launch_history,
        args.frame_start,
        args.frame_end,
        args.chunk_size,
        args.max_retries_per_clip,
        resource_logs_dir=resource_logs_dir,
    )
    final_report["preview_generation_results"] = final_preview_rows
    _write_json(report_path, final_report)
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
