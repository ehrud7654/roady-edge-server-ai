from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .analyzer import DAMAGE_CLASSES, ServerDamageAnalyzer
from .policy import SEVERITY_ORDER, evaluate_unit_severity


SEVERITIES = ("normal", "minor", "moderate", "severe")
PREDICTION_COLUMNS = (*SEVERITIES, "not_estimable")
CLASS_IDS = {"tactile_block": 0, "missing": 1, "crack": 2, "wear": 3}


def _mask_from_polygons(polygons: list[list[tuple[float, float]]], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        points = np.asarray(
            [[round(x * (width - 1)), round(y * (height - 1))] for x, y in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _ground_truth(label_path: Path, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    polygons: dict[int, list[list[tuple[float, float]]]] = {key: [] for key in CLASS_IDS.values()}
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7 or len(values) % 2 == 0:
            continue
        class_id = int(values[0])
        if class_id not in polygons:
            continue
        coordinates = [float(value) for value in values[1:]]
        polygons[class_id].append(list(zip(coordinates[0::2], coordinates[1::2])))
    return {
        name: _mask_from_polygons(polygons[class_id], shape)
        for name, class_id in CLASS_IDS.items()
    }


def _prediction_masks(
    result: Any,
    shape: tuple[int, int],
    *,
    tactile_confidence: float,
    damage_confidence: float,
) -> dict[str, np.ndarray]:
    masks = {name: np.zeros(shape, dtype=bool) for name in CLASS_IDS}
    if result.boxes is None or result.masks is None:
        return masks
    for class_value, confidence, mask_value in zip(
        result.boxes.cls.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.masks.data.cpu().numpy(),
    ):
        class_id = int(class_value)
        name = next((key for key, value in CLASS_IDS.items() if value == class_id), None)
        if name is None:
            continue
        threshold = tactile_confidence if name == "tactile_block" else damage_confidence
        if float(confidence) < threshold:
            continue
        resized = cv2.resize(mask_value, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        masks[name] |= resized > 0.5
    return masks


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * float((prediction & target).sum()) / denominator


def _ratio(tactile: np.ndarray, damage: np.ndarray) -> float | None:
    denominator = int(tactile.sum())
    return None if denominator == 0 else float((damage & tactile).sum() / denominator * 100.0)


def _ground_truth_decision(masks: dict[str, np.ndarray], policy: dict[str, Any]) -> tuple[float, str]:
    tactile = masks["tactile_block"]
    damage_types: dict[str, dict[str, Any]] = {}
    damage_union = np.zeros(tactile.shape, dtype=bool)
    for name in DAMAGE_CLASSES:
        valid = masks[name] & tactile
        damage_union |= valid
        ratio = _ratio(tactile, valid)
        damage_types[name] = {
            "detected": bool(valid.any()),
            "ratio_percent": 0.0 if ratio is None else ratio,
        }
    total_ratio = _ratio(tactile, damage_union)
    decision = evaluate_unit_severity(damage_types, policy)
    return (0.0 if total_ratio is None else total_ratio), decision.severity


def _macro_f1(confusion: np.ndarray) -> float:
    scores: list[float] = []
    for index in range(len(SEVERITIES)):
        true_positive = float(confusion[index, index])
        false_positive = float(confusion[:, index].sum() - true_positive)
        false_negative = float(confusion[index, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ROADY area, severity and review policy.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary-margin-percent", type=float, default=1.0)
    parser.add_argument("--tactile-confidence", type=float, default=0.25)
    parser.add_argument("--damage-confidence", type=float, default=0.05)
    args = parser.parse_args()

    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    args.output.mkdir(parents=True, exist_ok=True)
    analyzer = ServerDamageAnalyzer(args.model)
    analyzer.policy["review"]["tactile_confidence"] = args.tactile_confidence
    analyzer.policy["review"]["damage_confidence"] = args.damage_confidence
    model = analyzer.model
    confusion = np.zeros((len(SEVERITIES), len(PREDICTION_COLUMNS)), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    dice_values = {name: [] for name in CLASS_IDS}
    positive_dice_values = {name: [] for name in CLASS_IDS}
    positive_damage_dice: list[float] = []
    ratio_errors: list[float] = []
    boundary_errors: list[float] = []
    boundary_correct = 0
    estimable_count = 0
    thresholds = [
        float(analyzer.policy["ratio_thresholds"][key])
        for key in ("normal_max_percent", "minor_max_percent", "moderate_max_percent")
    ]

    results = model.predict(
        source=str(image_dir), stream=True, imgsz=args.imgsz, conf=0.05,
        device=args.device, verbose=False,
    )
    for index, result in enumerate(results, start=1):
        image_path = Path(result.path)
        label_path = label_dir / f"{image_path.stem}.txt"
        gt_masks = _ground_truth(label_path, tuple(result.orig_shape))
        pred_masks = _prediction_masks(
            result,
            tuple(result.orig_shape),
            tactile_confidence=args.tactile_confidence,
            damage_confidence=args.damage_confidence,
        )
        gt_ratio, gt_severity = _ground_truth_decision(gt_masks, analyzer.policy)
        gt_damage_union = np.zeros(tuple(result.orig_shape), dtype=bool)
        pred_damage_union = np.zeros(tuple(result.orig_shape), dtype=bool)
        for name in DAMAGE_CLASSES:
            gt_damage_union |= gt_masks[name] & gt_masks["tactile_block"]
            pred_damage_union |= pred_masks[name] & pred_masks["tactile_block"]
        payload, overlay, analyzed_damage_mask = analyzer.analyze_result(
            result,
            input_metadata={
                "roi_source": "tactile_block",
                "frame_quality_verified": True,
            },
        )
        if gt_damage_union.any():
            positive_damage_dice.append(_dice(analyzed_damage_mask, gt_damage_union))
        summary = payload["summary"]
        reference_pred_ratio = summary["max_damage_ratio_percent"]
        fully_estimable = summary.get("ratio_status") in {
            "estimated",
            "measured_no_damage",
        }
        pred_ratio = reference_pred_ratio if fully_estimable else None
        pred_severity = summary["estimated_severity"]
        pred_column = "not_estimable" if pred_severity is None else pred_severity
        confusion[SEVERITIES.index(gt_severity), PREDICTION_COLUMNS.index(pred_column)] += 1
        near_boundary = any(
            abs(gt_ratio - threshold) <= args.boundary_margin_percent
            for threshold in thresholds
        )
        absolute_error = None if pred_ratio is None else abs(float(pred_ratio) - gt_ratio)
        if absolute_error is not None:
            estimable_count += 1
            ratio_errors.append(absolute_error)
            if near_boundary:
                boundary_errors.append(absolute_error)
                boundary_correct += int(pred_severity == gt_severity)
        for name in CLASS_IDS:
            class_dice = _dice(pred_masks[name], gt_masks[name])
            dice_values[name].append(class_dice)
            if gt_masks[name].any():
                positive_dice_values[name].append(class_dice)
        rows.append(
            {
                "image": image_path.name,
                "gt_ratio_percent": round(gt_ratio, 4),
                "pred_ratio_percent": pred_ratio,
                "reference_pred_ratio_percent": reference_pred_ratio,
                "ratio_status": summary.get("ratio_status", "not_estimable"),
                "ratio_abs_error_pp": None if absolute_error is None else round(absolute_error, 4),
                "gt_severity": gt_severity,
                "pred_severity": pred_column,
                "near_boundary": near_boundary,
                "review_required": summary["review_required"],
                "review_codes": ",".join(reason["code"] for reason in summary["review_reasons"]),
            }
        )
        cv2.imwrite(str(args.output / f"{image_path.stem}_overlay.jpg"), overlay)
        if index % 25 == 0:
            print(f"processed={index}", flush=True)

    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savetxt(args.output / "severity_confusion.csv", confusion, delimiter=",", fmt="%d")
    boundary_count = sum(bool(row["near_boundary"]) for row in rows)
    ratio_mae_pp = float(np.mean(ratio_errors)) if ratio_errors else None
    positive_damage_dice_mean = (
        float(np.mean(positive_damage_dice)) if positive_damage_dice else None
    )
    severity_macro_f1 = _macro_f1(confusion)
    gate = analyzer.policy["review"]["model_quality_gate"]
    quality_gate_passed = bool(
        positive_damage_dice_mean is not None
        and ratio_mae_pp is not None
        and positive_damage_dice_mean >= float(gate["minimum_positive_damage_dice"])
        and ratio_mae_pp <= float(gate["maximum_ratio_mae_pp"])
        and severity_macro_f1 >= float(gate["minimum_severity_macro_f1"])
        and estimable_count / len(rows) >= float(gate["minimum_estimable_coverage"])
    )
    report = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "split": args.split,
        "tactile_confidence": args.tactile_confidence,
        "damage_confidence": args.damage_confidence,
        "images": len(rows),
        "estimable_count": estimable_count,
        "estimable_coverage": estimable_count / len(rows) if rows else 0.0,
        "ratio_mae_pp": ratio_mae_pp,
        "ratio_p95_error_pp": float(np.percentile(ratio_errors, 95)) if ratio_errors else None,
        "positive_damage_dice_mean": positive_damage_dice_mean,
        "severity_macro_f1": severity_macro_f1,
        "severity_accuracy": float(np.trace(confusion[:, :4]) / confusion.sum()) if confusion.sum() else 0.0,
        "severity_confusion_rows": list(SEVERITIES),
        "severity_confusion_columns": list(PREDICTION_COLUMNS),
        "severity_confusion": confusion.tolist(),
        "boundary_margin_percent": args.boundary_margin_percent,
        "boundary_image_count": boundary_count,
        "boundary_estimable_count": len(boundary_errors),
        "boundary_ratio_mae_pp": float(np.mean(boundary_errors)) if boundary_errors else None,
        "boundary_severity_accuracy": boundary_correct / len(boundary_errors) if boundary_errors else None,
        "mask_dice_mean": {
            name: float(np.mean(values)) if values else None
            for name, values in dice_values.items()
        },
        "mask_dice_positive_mean": {
            name: float(np.mean(values)) if values else None
            for name, values in positive_dice_values.items()
        },
        "quality_gate": {
            "minimum_positive_damage_dice": float(gate["minimum_positive_damage_dice"]),
            "maximum_ratio_mae_pp": float(gate["maximum_ratio_mae_pp"]),
            "minimum_severity_macro_f1": float(gate["minimum_severity_macro_f1"]),
            "minimum_estimable_coverage": float(gate["minimum_estimable_coverage"]),
            "passed": quality_gate_passed,
            "automatic_recommendation_enabled": quality_gate_passed,
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
