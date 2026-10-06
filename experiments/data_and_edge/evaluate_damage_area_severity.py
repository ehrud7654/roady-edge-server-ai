from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = ("normal", "mild", "moderate", "severe")


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item) for item in value.split(",")})
    if not thresholds or any(item <= 0 or item >= 1 for item in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated values between 0 and 1")
    return thresholds


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
        class_id = int(values[0])
        if class_id not in polygons:
            continue
        coords = [float(value) for value in values[1:]]
        polygons[class_id].append(list(zip(coords[0::2], coords[1::2])))
    return mask_from_polygons(polygons[0], size), mask_from_polygons(polygons[1], size)


def prediction_polygons(result, size: int) -> dict[int, list[tuple[float, np.ndarray]]]:
    predictions: dict[int, list[tuple[float, np.ndarray]]] = {0: [], 1: []}
    if result.boxes is None or result.masks is None:
        return predictions
    height, width = result.orig_shape
    for class_value, confidence, polygon in zip(
        result.boxes.cls.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.masks.xy,
    ):
        class_id = int(class_value)
        if class_id not in predictions or len(polygon) < 3:
            continue
        normalized = [[(float(x) / width, float(y) / height) for x, y in polygon]]
        predictions[class_id].append((float(confidence), mask_from_polygons(normalized, size)))
    return predictions


def union_at(items: list[tuple[float, np.ndarray]], threshold: float, size: int) -> np.ndarray:
    output = np.zeros((size, size), dtype=bool)
    for confidence, mask in items:
        if confidence >= threshold:
            output |= mask
    return output


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


def macro_f1(confusion: np.ndarray) -> float:
    scores = []
    for class_id in range(len(CLASS_NAMES)):
        true_positive = float(confusion[class_id, class_id])
        false_positive = float(confusion[:, class_id].sum() - true_positive)
        false_negative = float(confusion[class_id, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def summarize(state: dict) -> dict[str, float | int | list]:
    count = state["count"]
    positive_count = state["positive_count"]
    confusion = state["confusion"]
    image_tp = state["image_tp"]
    image_fp = state["image_fp"]
    image_fn = state["image_fn"]
    precision = safe_div(image_tp, image_tp + image_fp)
    recall = safe_div(image_tp, image_tp + image_fn)
    f2 = safe_div(5 * precision * recall, 4 * precision + recall)
    severe_total = int(confusion[3, :].sum())
    return {
        "images": count,
        "positive_images": positive_count,
        "block_dice_mean": safe_div(state["block_dice_sum"], count),
        "damage_dice_all_mean": safe_div(state["damage_dice_sum"], count),
        "damage_dice_positive_mean": safe_div(state["damage_dice_positive_sum"], positive_count),
        "ratio_mae": safe_div(state["ratio_abs_error_sum"], count),
        "ratio_mae_pp": 100 * safe_div(state["ratio_abs_error_sum"], count),
        "ratio_mae_positive_pp": 100 * safe_div(state["ratio_abs_error_positive_sum"], positive_count),
        "ratio_bias_pp": 100 * safe_div(state["ratio_signed_error_sum"], count),
        "image_damage_precision": precision,
        "image_damage_recall": recall,
        "image_damage_f2": f2,
        "severity_accuracy": safe_div(float(np.trace(confusion)), float(confusion.sum())),
        "severity_macro_f1": macro_f1(confusion),
        "severe_recall": safe_div(float(confusion[3, 3]), severe_total),
        "severity_confusion": confusion.tolist(),
        "missing_block_predictions": state["missing_block_predictions"],
    }


def empty_state() -> dict:
    return {
        "count": 0,
        "positive_count": 0,
        "block_dice_sum": 0.0,
        "damage_dice_sum": 0.0,
        "damage_dice_positive_sum": 0.0,
        "ratio_abs_error_sum": 0.0,
        "ratio_abs_error_positive_sum": 0.0,
        "ratio_signed_error_sum": 0.0,
        "image_tp": 0,
        "image_fp": 0,
        "image_fn": 0,
        "missing_block_predictions": 0,
        "confusion": np.zeros((4, 4), dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mask Dice, damage-area error and severity.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path.home() / "roady/datasets/server_roi_seg_v3")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50"),
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--mask-size", type=int, default=512)
    parser.add_argument("--normal-cutoff", type=float, default=0.005)
    parser.add_argument("--mild-limit", type=float, default=0.05)
    parser.add_argument("--moderate-limit", type=float, default=0.15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    output = args.output or Path.home() / f"roady/runs/area_severity_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    states = {threshold: empty_state() for threshold in args.thresholds}
    model = YOLO(str(args.model))
    minimum_confidence = min(args.thresholds)

    results = model.predict(
        source=str(image_dir),
        stream=True,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=minimum_confidence,
        device=args.device,
        verbose=False,
    )
    for index, result in enumerate(results, 1):
        label = label_dir / f"{Path(result.path).stem}.txt"
        gt_block, gt_damage = ground_truth_masks(label, args.mask_size)
        gt_damage &= gt_block
        gt_ratio = damage_ratio(gt_block, gt_damage)
        gt_severity = severity(gt_ratio, args.normal_cutoff, args.mild_limit, args.moderate_limit)
        predictions = prediction_polygons(result, args.mask_size)

        for threshold, state in states.items():
            pred_block = union_at(predictions[0], threshold, args.mask_size)
            pred_damage = union_at(predictions[1], threshold, args.mask_size)
            pred_damage &= pred_block
            pred_ratio = damage_ratio(pred_block, pred_damage)
            pred_severity = severity(pred_ratio, args.normal_cutoff, args.mild_limit, args.moderate_limit)
            gt_positive = bool(gt_damage.any())
            pred_positive = bool(pred_damage.any())

            state["count"] += 1
            state["positive_count"] += int(gt_positive)
            state["block_dice_sum"] += dice(pred_block, gt_block)
            state["damage_dice_sum"] += dice(pred_damage, gt_damage)
            state["damage_dice_positive_sum"] += dice(pred_damage, gt_damage) if gt_positive else 0.0
            absolute_error = abs(pred_ratio - gt_ratio)
            state["ratio_abs_error_sum"] += absolute_error
            state["ratio_abs_error_positive_sum"] += absolute_error if gt_positive else 0.0
            state["ratio_signed_error_sum"] += pred_ratio - gt_ratio
            state["image_tp"] += int(gt_positive and pred_positive)
            state["image_fp"] += int(not gt_positive and pred_positive)
            state["image_fn"] += int(gt_positive and not pred_positive)
            state["missing_block_predictions"] += int(not pred_block.any())
            state["confusion"][gt_severity, pred_severity] += 1
        if index % 250 == 0:
            print(f"processed={index}", flush=True)

    summaries = []
    for threshold, state in states.items():
        item = {"confidence": threshold, **summarize(state)}
        summaries.append(item)
    csv_columns = [key for key in summaries[0] if key != "severity_confusion"]
    with (output / "threshold_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    best = {
        "ratio_mae": min(summaries, key=lambda item: item["ratio_mae"]),
        "severity_macro_f1": max(summaries, key=lambda item: item["severity_macro_f1"]),
        "damage_dice_positive": max(summaries, key=lambda item: item["damage_dice_positive_mean"]),
        "image_damage_f2": max(summaries, key=lambda item: item["image_damage_f2"]),
    }
    report = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "split": args.split,
        "severity_thresholds": {
            "normal_cutoff": args.normal_cutoff,
            "mild_limit": args.mild_limit,
            "moderate_limit": args.moderate_limit,
        },
        "metrics_by_confidence": summaries,
        "best": best,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"split": args.split, "best": best}, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
