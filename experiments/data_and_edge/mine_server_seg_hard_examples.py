from __future__ import annotations

import argparse
import csv
import heapq
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


SEVERITY_NAMES = ("normal", "mild", "moderate", "severe")
FAILURE_CATEGORIES = (
    "false_negative",
    "false_positive",
    "low_damage_dice",
    "high_ratio_error",
    "severe_underestimated",
    "small_damage",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def mask_from_polygons(polygons: list[list[tuple[float, float]]], size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        points = np.asarray(
            [
                [
                    min(size - 1, max(0, round(x * (size - 1)))),
                    min(size - 1, max(0, round(y * (size - 1)))),
                ]
                for x, y in polygon
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def ground_truth_masks(label: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    polygons: dict[int, list[list[tuple[float, float]]]] = {0: [], 1: []}
    for line in label.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7 or len(values) % 2 == 0:
            continue
        class_id = int(float(values[0]))
        if class_id not in polygons:
            continue
        coords = [float(value) for value in values[1:]]
        polygons[class_id].append(list(zip(coords[0::2], coords[1::2])))
    return mask_from_polygons(polygons[0], size), mask_from_polygons(polygons[1], size)


def prediction_masks(result, size: int, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    output = {
        0: np.zeros((size, size), dtype=bool),
        1: np.zeros((size, size), dtype=bool),
    }
    if result.boxes is None or result.masks is None:
        return output[0], output[1]
    height, width = result.orig_shape
    for class_value, score, polygon in zip(
        result.boxes.cls.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.masks.xy,
    ):
        class_id = int(class_value)
        if class_id not in output or float(score) < confidence or len(polygon) < 3:
            continue
        normalized = [[(float(x) / width, float(y) / height) for x, y in polygon]]
        output[class_id] |= mask_from_polygons(normalized, size)
    return output[0], output[1]


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * float((prediction & target).sum()) / denominator


def damage_ratio(block: np.ndarray, damage: np.ndarray) -> float:
    denominator = int(block.sum())
    return 0.0 if denominator == 0 else float((block & damage).sum()) / denominator


def severity(ratio: float, normal_cutoff: float, mild_limit: float, moderate_limit: float) -> int:
    if ratio < normal_cutoff:
        return 0
    if ratio < mild_limit:
        return 1
    if ratio < moderate_limit:
        return 2
    return 3


def category_scores(row: dict[str, object]) -> dict[str, float]:
    gt_positive = bool(row["gt_positive"])
    pred_positive = bool(row["pred_positive"])
    gt_ratio = float(row["gt_ratio"])
    pred_ratio = float(row["pred_ratio"])
    damage_dice = float(row["damage_dice"])
    gt_severity = int(row["gt_severity_id"])
    pred_severity = int(row["pred_severity_id"])
    scores: dict[str, float] = {}

    if gt_positive and not pred_positive:
        scores["false_negative"] = 1.0 + gt_ratio
    if not gt_positive and pred_positive:
        scores["false_positive"] = pred_ratio
    if gt_positive:
        scores["low_damage_dice"] = 1.0 - damage_dice
        scores["high_ratio_error"] = abs(pred_ratio - gt_ratio)
        if gt_ratio < 0.05:
            scores["small_damage"] = (1.0 - damage_dice) + (0.5 if not pred_positive else 0.0)
    if gt_severity == 3 and pred_severity < 3:
        scores["severe_underestimated"] = float(3 - pred_severity) + abs(pred_ratio - gt_ratio)
    return scores


def push_top(
    heaps: dict[str, list[tuple[float, str]]],
    category: str,
    score: float,
    stem: str,
    quota: int,
) -> None:
    item = (score, stem)
    heap = heaps[category]
    if len(heap) < quota:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def select_examples(
    rows: list[dict[str, object]],
    per_category: int,
    total_limit: int,
) -> list[dict[str, object]]:
    heaps = {category: [] for category in FAILURE_CATEGORIES}
    by_stem = {str(row["stem"]): row for row in rows}
    for row in rows:
        for category, score in category_scores(row).items():
            push_top(heaps, category, score, str(row["stem"]), per_category)

    selected: dict[str, dict[str, object]] = {}
    for category in FAILURE_CATEGORIES:
        for score, stem in sorted(heaps[category], reverse=True):
            row = selected.setdefault(stem, dict(by_stem[stem]))
            categories = set(str(row.get("selection_categories", "")).split("|")) - {""}
            categories.add(category)
            row["selection_categories"] = "|".join(sorted(categories))
            row["selection_score"] = max(float(row.get("selection_score", 0.0)), score)

    ordered = sorted(
        selected.values(),
        key=lambda row: (
            len(str(row["selection_categories"]).split("|")),
            float(row["selection_score"]),
        ),
        reverse=True,
    )
    return ordered[:total_limit]


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    resized = cv2.resize(
        mask.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    image[resized] = (
        image[resized].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha
    ).astype(np.uint8)


def render_review(
    image_path: Path,
    output_path: Path,
    gt_block: np.ndarray,
    gt_damage: np.ndarray,
    pred_block: np.ndarray,
    pred_damage: np.ndarray,
    row: dict[str, object],
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    canvas = image.copy()
    overlay_mask(canvas, gt_block, (255, 80, 0), 0.10)
    overlay_mask(canvas, pred_block, (255, 255, 0), 0.10)
    overlay_mask(canvas, gt_damage, (0, 210, 0), 0.55)
    overlay_mask(canvas, pred_damage, (0, 0, 255), 0.55)
    text = (
        f"GT green / PRED red | dice={float(row['damage_dice']):.3f} "
        f"| ratio={100*float(row['gt_ratio']):.1f}%->{100*float(row['pred_ratio']):.1f}%"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def write_csv(path: Path, rows: list[dict[str, object]], extra_columns: list[str] | None = None) -> None:
    columns = list(rows[0].keys()) if rows else []
    for column in extra_columns or []:
        if column not in columns:
            columns.append(column)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_metrics(path: Path) -> list[dict[str, object]]:
    integer_columns = {
        "gt_positive",
        "pred_positive",
        "gt_severity_id",
        "pred_severity_id",
    }
    float_columns = {
        "block_dice",
        "damage_dice",
        "gt_ratio",
        "pred_ratio",
        "ratio_error_pp",
        "ratio_abs_error_pp",
    }
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, object] = dict(raw)
            for column in integer_columns:
                row[column] = int(str(row[column]))
            for column in float_columns:
                row[column] = float(str(row[column]))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine hard examples from TRAIN only for ROADY server segmentation."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path.home() / "roady/datasets/server_roi_seg_v3",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--mask-size", type=int, default=512)
    parser.add_argument("--normal-cutoff", type=float, default=0.005)
    parser.add_argument("--mild-limit", type=float, default=0.05)
    parser.add_argument("--moderate-limit", type=float, default=0.15)
    parser.add_argument("--per-category", type=int, default=120)
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse output/all_train_metrics.csv and only redo selection/rendering.",
    )
    parser.add_argument("--output", type=Path, default=Path.home() / "roady/runs/hard_examples_v4_train")
    args = parser.parse_args()

    image_dir = args.dataset / "images" / "train"
    label_dir = args.dataset / "labels" / "train"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"TRAIN images/labels not found under {args.dataset}")
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    metrics_path = args.output / "all_train_metrics.csv"
    if args.resume:
        if not metrics_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing {metrics_path}")
        rows = read_metrics(metrics_path)
        print(f"resumed_metrics={len(rows)}", flush=True)
    else:
        rows: list[dict[str, object]] = []
        results = model.predict(
            source=str(image_dir),
            stream=True,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.confidence,
            device=args.device,
            verbose=False,
        )
        for index, result in enumerate(results, 1):
            image_path = Path(result.path)
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            gt_block, gt_damage = ground_truth_masks(label_path, args.mask_size)
            gt_damage &= gt_block
            pred_block, pred_damage = prediction_masks(result, args.mask_size, args.confidence)
            pred_damage &= pred_block
            gt_ratio = damage_ratio(gt_block, gt_damage)
            pred_ratio = damage_ratio(pred_block, pred_damage)
            gt_severity = severity(gt_ratio, args.normal_cutoff, args.mild_limit, args.moderate_limit)
            pred_severity = severity(pred_ratio, args.normal_cutoff, args.mild_limit, args.moderate_limit)
            row: dict[str, object] = {
                "stem": image_path.stem,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "gt_positive": int(gt_damage.any()),
                "pred_positive": int(pred_damage.any()),
                "block_dice": round(dice(pred_block, gt_block), 6),
                "damage_dice": round(dice(pred_damage, gt_damage), 6),
                "gt_ratio": round(gt_ratio, 8),
                "pred_ratio": round(pred_ratio, 8),
                "ratio_error_pp": round(100.0 * (pred_ratio - gt_ratio), 6),
                "ratio_abs_error_pp": round(100.0 * abs(pred_ratio - gt_ratio), 6),
                "gt_severity_id": gt_severity,
                "gt_severity": SEVERITY_NAMES[gt_severity],
                "pred_severity_id": pred_severity,
                "pred_severity": SEVERITY_NAMES[pred_severity],
            }
            rows.append(row)
            if index % 250 == 0:
                print(f"processed={index}", flush=True)

        if not rows:
            raise RuntimeError("No TRAIN samples were evaluated.")
        write_csv(metrics_path, rows)
    selected = select_examples(rows, args.per_category, args.total)

    review_images = args.output / "review_images"
    source_images = args.output / "source_images"
    source_labels = args.output / "source_labels"
    for directory in (review_images, source_images, source_labels):
        directory.mkdir(parents=True, exist_ok=True)

    selected_by_stem = {str(row["stem"]): row for row in selected}
    selected_paths = [str(row["image_path"]) for row in selected]
    rendered = 0
    for start in range(0, len(selected_paths), args.batch):
        chunk = selected_paths[start : start + args.batch]
        chunk_results = model.predict(
            source=chunk,
            stream=False,
            imgsz=args.imgsz,
            batch=len(chunk),
            conf=args.confidence,
            device=args.device,
            verbose=False,
        )
        for result in chunk_results:
            image_path = Path(result.path)
            label_path = label_dir / f"{image_path.stem}.txt"
            row = selected_by_stem[image_path.stem]
            gt_block, gt_damage = ground_truth_masks(label_path, args.mask_size)
            gt_damage &= gt_block
            pred_block, pred_damage = prediction_masks(result, args.mask_size, args.confidence)
            pred_damage &= pred_block
            shutil.copy2(image_path, source_images / image_path.name)
            shutil.copy2(label_path, source_labels / label_path.name)
            render_review(
                image_path,
                review_images / f"{image_path.stem}.jpg",
                gt_block,
                gt_damage,
                pred_block,
                pred_damage,
                row,
            )
            rendered += 1
            if rendered % 100 == 0:
                print(f"rendered={rendered}/{len(selected)}", flush=True)

    manual_columns = [
        "damage_type",
        "block_color",
        "surface_state",
        "lighting",
        "distance",
        "contamination",
        "failure_reason",
        "label_status",
        "reviewer",
        "review_note",
    ]
    for row in selected:
        for column in manual_columns:
            row[column] = ""
    write_csv(args.output / "selected_manifest.csv", selected, manual_columns)

    summary = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "split": "train",
        "confidence": args.confidence,
        "evaluated_images": len(rows),
        "selected_images": len(selected),
        "selection_category_counts": dict(
            Counter(
                category
                for row in selected
                for category in str(row["selection_categories"]).split("|")
                if category
            )
        ),
        "manual_review_columns": manual_columns,
        "legend": {
            "gt_damage": "green",
            "pred_damage": "red",
            "overlap": "yellow",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
