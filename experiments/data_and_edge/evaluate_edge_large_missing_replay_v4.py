from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path.cwd()  # Set the working directory to a local ROADY data workspace.
SOURCE = ROOT / "ai/edge_review_v2_v3/out"
DATASET = ROOT / "ai/datasets/edge_large_missing_replay_v4"
MODELS = {
    "v3": ROOT / "ai/runs/edge_yolo26n_balanced_replay_v3-2/weights/best.pt",
    "v4": ROOT / "ai/runs/edge_yolo26n_large_missing_replay_v4/weights/best.pt",
}
THRESHOLDS = (0.05, 0.10, 0.15, 0.20)


def xywh_to_xyxy(box: list[float]) -> list[float]:
    xc, yc, w, h = box
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(first)
    bx1, by1, bx2, by2 = xywh_to_xyxy(second)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def read_manual(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) == 5 and int(values[0]) == 1:
            rows.append([float(value) for value in values[1:]])
    return rows


def predictions(result) -> list[tuple[float, list[float]]]:
    height, width = result.orig_shape
    rows = []
    if result.boxes is None:
        return rows
    for cls_value, conf_value, xyxy in zip(
        result.boxes.cls.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.xyxy.cpu().tolist(),
    ):
        if int(cls_value) != 1:
            continue
        x1, y1, x2, y2 = xyxy
        rows.append(
            (
                float(conf_value),
                [
                    (x1 + x2) / (2 * width),
                    (y1 + y2) / (2 * height),
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ],
            )
        )
    return rows


def main() -> None:
    images = sorted((SOURCE / "images").glob("*.jpg"))
    negative_images = sorted((DATASET / "images/train").glob("negative_*.jpg"))
    report = {}
    for label, weight in MODELS.items():
        model = YOLO(str(weight))
        image_predictions = {}
        for image in images + negative_images:
            result = model.predict(
                source=str(image), imgsz=768, conf=0.01, iou=0.7, device=0, verbose=False
            )[0]
            image_predictions[image.stem] = predictions(result)
        model_report = {}
        for threshold in THRESHOLDS:
            total = matched = hit_images = 0
            for image in images:
                manual = read_manual(SOURCE / "labels" / f"{image.stem}.txt")
                predicted = [box for confidence, box in image_predictions[image.stem] if confidence >= threshold]
                image_matched = sum(
                    any(iou(target, box) >= 0.5 for box in predicted) for target in manual
                )
                total += len(manual)
                matched += image_matched
                hit_images += int(image_matched > 0)
            negative_fp_images = sum(
                any(confidence >= threshold for confidence, _ in image_predictions[image.stem])
                for image in negative_images
            )
            model_report[str(threshold)] = {
                "manual_missed_boxes": total,
                "recovered_boxes_iou50": matched,
                "recovery_recall": matched / total if total else None,
                "hard_example_images": len(images),
                "recovered_event_images": hit_images,
                "event_image_recall": hit_images / len(images) if images else None,
                "reviewed_negative_images": len(negative_images),
                "negative_false_positive_images": negative_fp_images,
            }
        report[label] = model_report
    path = ROOT / "ai/runs/edge_v3_v4_large_missing_recovery_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
