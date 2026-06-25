import argparse
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--analyze-root")
    return parser


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _json_safe(value):
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    return str(value)


try:
    import unreal  # type: ignore
except Exception:  # pragma: no cover
    unreal = None


DEFAULT_OUTPUT_ROOT = Path("/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_erp_camera_rotation_effect")
DEFAULT_CAMERA_POSE = {
    "x": 0.0,
    "y": 0.0,
    "z": 140.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
}
DEFAULT_MARKER_RADIUS_CM = 260.0
DEFAULT_MARKER_PREFIX = "BEDLAM360_ROTTEST_"
ROTATION_VARIANTS = [
    {"name": "yaw_000", "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "group": "yaw"},
    {"name": "yaw_045", "yaw": 45.0, "pitch": 0.0, "roll": 0.0, "group": "yaw"},
    {"name": "yaw_090", "yaw": 90.0, "pitch": 0.0, "roll": 0.0, "group": "yaw"},
    {"name": "pitch_000", "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "group": "pitch"},
    {"name": "pitch_010", "yaw": 0.0, "pitch": 10.0, "roll": 0.0, "group": "pitch"},
    {"name": "pitch_020", "yaw": 0.0, "pitch": 20.0, "roll": 0.0, "group": "pitch"},
    {"name": "roll_000", "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "group": "roll"},
    {"name": "roll_010", "yaw": 0.0, "pitch": 0.0, "roll": 10.0, "group": "roll"},
    {"name": "roll_020", "yaw": 0.0, "pitch": 0.0, "roll": 20.0, "group": "roll"},
]


if unreal is not None:
    import bedlam360_mini_validation as mini  # type: ignore
    import bedlam360_erp_convention_calibration as calib  # type: ignore

    mini = importlib.reload(mini)
    calib = importlib.reload(calib)


def _run_postprocess(analyze_root):
    import cv2
    import numpy as np

    analyze_root = Path(analyze_root)
    images_dir = analyze_root / "images"
    metadata_dir = analyze_root / "metadata"
    previews_dir = analyze_root / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((analyze_root / "manifest.json").read_text(encoding="utf-8"))
    variants = manifest["variants"]

    def load_gray(path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image: {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, gray

    def file_sha1(path):
        import hashlib

        digest = hashlib.sha1()
        with open(path, "rb") as fp:
            while True:
                chunk = fp.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def image_metrics(base_gray, target_gray):
        diff = cv2.absdiff(base_gray, target_gray)
        mean_abs = float(np.mean(diff))
        max_abs = int(np.max(diff))
        nonzero_fraction = float(np.count_nonzero(diff) / diff.size)
        shift, response = cv2.phaseCorrelate(base_gray.astype(np.float32), target_gray.astype(np.float32))
        return {
            "mean_abs_pixel_diff": mean_abs,
            "max_abs_pixel_diff": max_abs,
            "nonzero_fraction": nonzero_fraction,
            "estimated_shift_xy_px": [float(shift[0]), float(shift[1])],
            "phase_correlation_response": float(response),
        }

    def make_contact_sheet(items, output_path, title):
        images = []
        for item in items:
            image = cv2.imread(str(item["png_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to load image: {item['png_path']}")
            overlay = image.copy()
            cv2.putText(
                overlay,
                item["name"],
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"y={item['camera_pose_cm_deg']['yaw']:.1f} p={item['camera_pose_cm_deg']['pitch']:.1f} r={item['camera_pose_cm_deg']['roll']:.1f}",
                (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            images.append(overlay)
        h = max(image.shape[0] for image in images)
        w = max(image.shape[1] for image in images)
        cols = 3
        rows = int(math.ceil(len(images) / cols))
        margin = 8
        canvas = np.zeros((rows * h + (rows - 1) * margin + 48, cols * w + (cols - 1) * margin, 3), dtype=np.uint8)
        cv2.putText(canvas, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        for idx, image in enumerate(images):
            row = idx // cols
            col = idx % cols
            y0 = 48 + row * (h + margin)
            x0 = col * (w + margin)
            canvas[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image
        cv2.imwrite(str(output_path), canvas)
        return output_path

    group_reports = []
    for group_name in ("yaw", "pitch", "roll"):
        group_variants = [item for item in variants if item["group"] == group_name]
        base_variant = group_variants[0]
        base_image, base_gray = load_gray(base_variant["png_path"])
        del base_image
        rows = []
        for item in group_variants:
            target_image, target_gray = load_gray(item["png_path"])
            del target_image
            metrics = image_metrics(base_gray, target_gray)
            rows.append(
                {
                    "name": item["name"],
                    "group": group_name,
                    "camera_pose_cm_deg": item["camera_pose_cm_deg"],
                    "png_path": item["png_path"],
                    "png_sha1": file_sha1(item["png_path"]),
                    "metrics_vs_group_base": metrics,
                    "cube_face_hashes": item.get("cube_face_hashes", {}),
                }
            )
        contact_sheet_path = previews_dir / f"{group_name}_contact_sheet.png"
        make_contact_sheet(group_variants, contact_sheet_path, f"{group_name.upper()} rotation effect")
        group_reports.append(
            {
                "group": group_name,
                "base_variant": base_variant["name"],
                "contact_sheet_path": str(contact_sheet_path),
                "rows": rows,
            }
        )

    summary = {
        "kind": "bedlam360_erp_camera_rotation_effect",
        "analyze_root": str(analyze_root),
        "camera_position_cm": manifest["camera_position_cm"],
        "summary": [
            "Fixed camera position with varying yaw/pitch/roll.",
            "Metrics are computed against each group's zero-rotation baseline using ERP PNG outputs.",
            "Estimated shifts come from phase correlation on grayscale ERP images.",
            "Cube-face hashes are included to check whether SceneCaptureCube rotation affects face exports as well as long-lat outputs.",
        ],
        "group_reports": group_reports,
    }
    summary_path = previews_dir / "rotation_effect_report.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    print(f"[BEDLAM360] Wrote ERP rotation-effect report: {summary_path}")
    for group in group_reports:
        print(f"[BEDLAM360] {group['group']} contact sheet: {group['contact_sheet_path']}")
    return summary_path


def _run_unreal_render(output_root=DEFAULT_OUTPUT_ROOT, marker_radius_cm=DEFAULT_MARKER_RADIUS_CM):
    output_root = Path(output_root)
    images_dir = output_root / "images"
    metadata_dir = output_root / "metadata"
    previews_dir = output_root / "previews"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    calib._clear_existing_markers(prefix=DEFAULT_MARKER_PREFIX)  # pylint: disable=protected-access
    mini.reconstruct_full_bedlam_scene.clear_existing_bedlam_bodies()
    markers = [calib._spawn_text_marker(spec, DEFAULT_CAMERA_POSE, marker_radius_cm) for spec in calib.MARKER_SPECS]  # pylint: disable=protected-access

    actor = mini.capture_scene_cube.find_scene_capture_cube(mini.DEFAULT_ACTOR_LABEL)
    component = mini.capture_scene_cube.get_capture_component(actor)
    texture_target = mini.capture_scene_cube.get_texture_target(component)
    export_lib = mini.unreal.BEDLAM360ExportLibrary

    variant_records = []
    for variant in ROTATION_VARIANTS:
        frame_name = f"erp_rotation_{variant['name']}"
        hdr_path = images_dir / f"{frame_name}.hdr"
        exr_path = images_dir / f"{frame_name}.exr"
        png_path = images_dir / f"{frame_name}.png"
        faces_dir = images_dir / f"{frame_name}_faces"
        camera_pose = {
            "x": float(DEFAULT_CAMERA_POSE["x"]),
            "y": float(DEFAULT_CAMERA_POSE["y"]),
            "z": float(DEFAULT_CAMERA_POSE["z"]),
            "yaw": float(variant["yaw"]),
            "pitch": float(variant["pitch"]),
            "roll": float(variant["roll"]),
        }
        mini.capture_scene_cube.set_actor_pose(
            actor,
            camera_pose["x"],
            camera_pose["y"],
            camera_pose["z"],
            camera_pose["pitch"],
            camera_pose["yaw"],
            camera_pose["roll"],
        )
        capture_result = mini.stabilized_capture_and_export(
            actor=actor,
            component=component,
            texture_target=texture_target,
            export_lib=export_lib,
            frame_name=frame_name,
            hdr_path=hdr_path,
            exr_path=exr_path,
            faces_dir=faces_dir,
            warmup_ticks=mini.DEFAULT_WARMUP_TICKS,
            discard_captures=mini.DEFAULT_DISCARD_CAPTURES,
        )
        preview_status = mini._run_preview_frame(  # pylint: disable=protected-access
            image_path=exr_path if capture_result["exr_ok"] else hdr_path,
            output_png_path=png_path,
            metadata_json_path=None,
            overlay=False,
        )
        cube_face_hashes = {}
        if faces_dir.exists():
            import hashlib

            for face_path in sorted(faces_dir.glob("*")):
                digest = hashlib.sha1(face_path.read_bytes()).hexdigest()
                cube_face_hashes[face_path.name] = digest
        record = {
            "name": variant["name"],
            "group": variant["group"],
            "camera_pose_cm_deg": camera_pose,
            "hdr_path": str(hdr_path),
            "exr_path": str(exr_path),
            "png_path": str(png_path),
            "faces_dir": str(faces_dir),
            "capture_result": capture_result,
            "preview_status": preview_status,
            "cube_face_hashes": cube_face_hashes,
        }
        (metadata_dir / f"{variant['name']}.json").write_text(json.dumps(_json_safe(record), indent=2), encoding="utf-8")
        variant_records.append(record)

    manifest = {
        "kind": "bedlam360_erp_camera_rotation_effect",
        "camera_position_cm": {
            "x": float(DEFAULT_CAMERA_POSE["x"]),
            "y": float(DEFAULT_CAMERA_POSE["y"]),
            "z": float(DEFAULT_CAMERA_POSE["z"]),
        },
        "marker_radius_cm": float(marker_radius_cm),
        "markers": markers,
        "variants": variant_records,
    }
    (output_root / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    command = [
        "python3",
        str(Path(__file__).resolve()),
        "--postprocess",
        "--analyze-root",
        str(output_root),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    post_status = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    _write_json(output_root / "postprocess_status.json", post_status)
    if completed.stdout:
        unreal.log(completed.stdout)
    if completed.stderr:
        unreal.log_warning(completed.stderr)
    return output_root


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.postprocess:
        if not args.analyze_root:
            raise RuntimeError("--analyze-root is required with --postprocess")
        _run_postprocess(args.analyze_root)
        return
    if unreal is None:
        raise RuntimeError("Render mode requires Unreal Python. Use --postprocess for offline analysis only.")
    _run_unreal_render()


if __name__ == "__main__":
    main()
