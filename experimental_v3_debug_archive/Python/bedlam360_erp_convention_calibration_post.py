import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _detect_marker_centroids(image_bgr, markers, color_distance_threshold=70.0):
    detections = {}
    image_f = image_bgr.astype(np.float32)
    for marker in markers:
        target = np.array(marker["bgr"], dtype=np.float32).reshape(1, 1, 3)
        dist = np.linalg.norm(image_f - target, axis=2)
        mask = (dist <= float(color_distance_threshold)).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best = None
        best_area = 0
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area <= best_area:
                continue
            best_area = area
            best = centroids[label_idx]
        detections[marker["name"]] = {
            "detected_pixel_xy": None if best is None else [float(best[0]), float(best[1])],
            "detected_area_px": int(best_area),
        }
    return detections


def _draw_predicted_vs_actual_overlay(image_bgr, markers, predicted, detected):
    image = image_bgr.copy()
    for marker in markers:
        name = marker["name"]
        color = tuple(int(v) for v in marker["bgr"])
        pred = predicted[name]["predicted_pixel_xy"]
        det = detected[name]["detected_pixel_xy"]
        label = marker["label"]
        if pred is not None:
            px, py = int(round(pred[0])), int(round(pred[1]))
            cv2.drawMarker(image, (px, py), color, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
            cv2.putText(image, f"P {label}", (px + 8, max(18, py - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if det is not None:
            dx, dy = int(round(det[0])), int(round(det[1]))
            cv2.circle(image, (dx, dy), 8, color, thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(image, f"A {label}", (dx + 8, dy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if pred is not None and det is not None:
            cv2.line(image, (int(round(pred[0])), int(round(pred[1]))), (int(round(det[0])), int(round(det[1]))), color, 1, cv2.LINE_AA)
    return image


def run_post(metadata_json_path, png_path, report_json_path, overlay_png_path):
    metadata = _read_json(metadata_json_path)
    image = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load PNG: {png_path}")

    markers = metadata["markers"]
    predicted = metadata["predicted_markers"]
    detected = _detect_marker_centroids(image, markers)
    overlay = _draw_predicted_vs_actual_overlay(image, markers, predicted, detected)
    cv2.imwrite(str(overlay_png_path), overlay)

    marker_report = []
    for marker in markers:
        name = marker["name"]
        pred = predicted[name]["predicted_pixel_xy"]
        det = detected[name]["detected_pixel_xy"]
        delta = None
        if pred is not None and det is not None:
            delta = [float(det[0] - pred[0]), float(det[1] - pred[1])]
        marker_report.append(
            {
                "name": name,
                "label": marker["label"],
                "world_location_cm": marker["world_location_cm"],
                "predicted_pixel_xy": pred,
                "predicted_longitude_deg": predicted[name]["predicted_longitude_deg"],
                "predicted_latitude_deg": predicted[name]["predicted_latitude_deg"],
                "actual_pixel_xy": det,
                "actual_detected_area_px": detected[name]["detected_area_px"],
                "delta_pixel_xy": delta,
            }
        )

    report = {
        "camera_pose_cm_deg": metadata["camera_pose_cm_deg"],
        "marker_radius_cm": metadata["marker_radius_cm"],
        "capture_result": metadata["capture_result"],
        "preview_status": metadata["preview_status"],
        "renderer_projection_assumption": metadata["renderer_projection_assumption"],
        "markers": marker_report,
        "files": {
            "png": str(png_path),
            "overlay_png": str(overlay_png_path),
            "metadata_json": str(metadata_json_path),
        },
    }
    _write_json(report_json_path, report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_json_path")
    parser.add_argument("png_path")
    parser.add_argument("report_json_path")
    parser.add_argument("overlay_png_path")
    args = parser.parse_args()
    run_post(
        metadata_json_path=Path(args.metadata_json_path),
        png_path=Path(args.png_path),
        report_json_path=Path(args.report_json_path),
        overlay_png_path=Path(args.overlay_png_path),
    )


if __name__ == "__main__":
    main()
