from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .expected_region import (
    estimate_expected_region,
    estimate_missing_expected_region,
    mask_bbox,
)
from .policy import (
    ModelQuality,
    SEVERITY_ORDER,
    evaluate_unit_severity,
    load_severity_policy,
)
from .review_reasons import review_reason_list


REQUIRED_CLASSES = ("tactile_block", "missing", "crack", "wear")
DAMAGE_CLASSES = ("missing", "crack", "wear")
# Optional 5th class. When the model has it (5-class model), a confident
# obstruction mask covering the ROI routes the whole ROI to OBSTRUCTION_SUSPECTED
# (판단 보류) instead of confirming missing/crack/wear. Absent on 4-class models,
# in which case obstruction handling falls back to the metadata flag only.
OBSTRUCTION_CLASS = "obstruction"
COLORS = {
    "tactile_block": (64, 220, 120),
    "missing": (40, 40, 230),
    "crack": (235, 220, 40),
    "wear": (40, 180, 240),
}


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return 0.0 if union == 0 else float(np.logical_and(left, right).sum() / union)


def _deduplicate(items: list[tuple[float, np.ndarray]], threshold: float) -> list[tuple[float, np.ndarray]]:
    kept: list[tuple[float, np.ndarray]] = []
    for confidence, mask in sorted(items, key=lambda item: item[0], reverse=True):
        if all(_mask_iou(mask, previous) < threshold for _, previous in kept):
            kept.append((confidence, mask))
    return kept


def _union(items: list[tuple[float, np.ndarray]], shape: tuple[int, int]) -> np.ndarray:
    merged = np.zeros(shape, dtype=bool)
    for _, mask in items:
        merged |= mask
    return merged


def _polygons(mask: np.ndarray) -> list[list[list[int]]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: list[list[list[int]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 2:
            continue
        epsilon = 0.001 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        result.append([[int(x), int(y)] for [[x, y]] in simplified.tolist()])
    return result


def _bbox_list(mask: np.ndarray) -> list[int] | None:
    bbox = mask_bbox(mask)
    return None if bbox is None else list(bbox)


def _bbox_intersects_mask(bbox: tuple[int, int, int, int], mask: np.ndarray) -> bool:
    x1, y1, x2, y2 = bbox
    return bool(mask[max(0, y1):min(mask.shape[0], y2), max(0, x1):min(mask.shape[1], x2)].any())


def _mask_components(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return max(0, count - 1)


def _confidence(items: list[tuple[float, np.ndarray]], area: np.ndarray | None = None) -> float | None:
    values = [confidence for confidence, mask in items if area is None or np.logical_and(mask, area).any()]
    return max(values, default=None)


class ServerDamageAnalyzer:
    """Convert a four-class YOLO segmentation result into conservative analysis units."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        policy_path: str | Path | None = None,
        quality: ModelQuality = ModelQuality(),
    ) -> None:
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.policy = load_severity_policy(policy_path)
        self.quality = quality
        raw_names = self.model.names
        self.class_names = {
            int(key): str(value).strip().lower()
            for key, value in (raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names))
        }
        present = set(self.class_names.values())
        self.class_mapping_valid = all(name in present for name in REQUIRED_CLASSES)
        self.class_ids = {name: next((key for key, value in self.class_names.items() if value == name), None) for name in REQUIRED_CLASSES}

    def predict(
        self,
        image: str | Path | np.ndarray,
        *,
        imgsz: int = 1024,
        device: str | int = "cpu",
        image_quality_ok: bool = True,
        input_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        minimum_confidence = float(self.policy["review"]["damage_confidence"])
        result = self.model.predict(
            source=image,
            imgsz=imgsz,
            conf=minimum_confidence,
            device=device,
            verbose=False,
        )[0]
        return self.analyze_result(
            result,
            image_quality_ok=image_quality_ok,
            input_metadata=input_metadata,
        )

    def analyze_result(
        self,
        result: Any,
        *,
        image_quality_ok: bool = True,
        input_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        height, width = result.orig_shape
        shape = (height, width)
        original = result.orig_img.copy()
        input_payload = self._input_payload(input_metadata)
        if not self.class_mapping_valid:
            payload = self._invalid_mapping_payload(input_payload)
            return payload, original, np.zeros(shape, dtype=bool)

        items = self._collect_instances(result, shape)
        # 5-class model: a confident obstruction mask covering enough of the ROI
        # means the block is occluded -> route to 판단 보류 via the existing
        # possible_obstruction path (never auto-confirm missing/crack/wear on it).
        obstruction_union = _union(items.get(OBSTRUCTION_CLASS, []), shape)
        if obstruction_union.any():
            coverage = float(obstruction_union.sum()) / float(max(1, height * width))
            if coverage >= float(self.policy["review"].get("obstruction_coverage_min", 0.05)):
                input_payload["possible_obstruction"] = True
                if not input_payload.get("obstruction_source"):
                    input_payload["obstruction_source"] = "server_model"
        tactile_items = items["tactile_block"]
        type_items = {name: items[name] for name in DAMAGE_CLASSES}
        type_unions = {name: _union(type_items[name], shape) for name in DAMAGE_CLASSES}
        units: list[dict[str, Any]] = []
        unit_total_masks: list[np.ndarray] = []
        assigned_missing = np.zeros(shape, dtype=bool)
        tactile_bboxes = [mask_bbox(mask) for _, mask in tactile_items]
        tactile_bboxes = [bbox for bbox in tactile_bboxes if bbox is not None]

        for index, (tactile_confidence, tactile_mask) in enumerate(tactile_items, start=1):
            unit, total_mask, unit_missing = self._analyze_tactile_unit(
                index=index,
                tactile_confidence=tactile_confidence,
                tactile_mask=tactile_mask,
                tactile_bboxes=tactile_bboxes,
                type_items=type_items,
                type_unions=type_unions,
                image_quality_ok=image_quality_ok,
                input_payload=input_payload,
            )
            units.append(unit)
            unit_total_masks.append(total_mask)
            assigned_missing |= unit_missing

        unresolved_missing = type_unions["missing"] & ~assigned_missing
        if unresolved_missing.any():
            expected_cfg = self.policy["analysis_units"]["expected_region"]
            expected = estimate_missing_expected_region(
                missing_mask=unresolved_missing,
                neighbor_bboxes=tactile_bboxes,
                shape=shape,
                minimum_neighbor_count=int(expected_cfg["minimum_neighbor_count"]),
                maximum_size_cv=float(expected_cfg["maximum_size_cv"]),
                maximum_spacing_cv=float(expected_cfg["maximum_spacing_cv"]),
            )
            valid_missing = (
                unresolved_missing & expected.mask
                if expected.mask is not None
                else unresolved_missing
            )
            units.append(
                self._unresolved_missing_unit(
                    valid_missing,
                    type_items["missing"],
                    len(units) + 1,
                    expected_region=expected,
                )
            )
            unit_total_masks.append(valid_missing)

        if not tactile_items and not unresolved_missing.any():
            units = []

        total_damage_mask = np.zeros(shape, dtype=bool)
        for mask in unit_total_masks:
            total_damage_mask |= mask
        payload = self._payload(units, input_payload, total_damage_mask)
        overlay = self._overlay(original, units)
        return payload, overlay, total_damage_mask

    def _collect_instances(self, result: Any, shape: tuple[int, int]) -> dict[str, list[tuple[float, np.ndarray]]]:
        candidates = {name: [] for name in REQUIRED_CLASSES + (OBSTRUCTION_CLASS,)}
        if result.boxes is not None and result.masks is not None:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
            masks = result.masks.data.detach().cpu().numpy()
            for class_id, confidence, mask in zip(classes, confidences, masks):
                name = self.class_names.get(int(class_id))
                if name not in candidates:
                    continue
                if name == "tactile_block":
                    threshold = float(self.policy["review"]["tactile_confidence"])
                elif name == OBSTRUCTION_CLASS:
                    threshold = float(self.policy["review"].get("obstruction_confidence", 0.25))
                else:
                    threshold = float(self.policy["review"]["damage_confidence"])
                if confidence >= threshold:
                    candidates[name].append((float(confidence), _resize_mask(mask, shape)))
        duplicate_iou = float(self.policy["review"]["duplicate_mask_iou"])
        return {name: _deduplicate(values, duplicate_iou) for name, values in candidates.items()}

    def _analyze_tactile_unit(
        self,
        *,
        index: int,
        tactile_confidence: float,
        tactile_mask: np.ndarray,
        tactile_bboxes: list[tuple[int, int, int, int]],
        type_items: dict[str, list[tuple[float, np.ndarray]]],
        type_unions: dict[str, np.ndarray],
        image_quality_ok: bool,
        input_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        bbox = mask_bbox(tactile_mask)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        aspect = (x2 - x1) / max(y2 - y1, 1)
        unit_cfg = self.policy["analysis_units"]
        is_block = (
            _mask_components(tactile_mask) == 1
            and float(unit_cfg["block_aspect_ratio_min"]) <= aspect <= float(unit_cfg["block_aspect_ratio_max"])
        )
        analysis_unit = "block" if is_block else "block_group"
        local_id = f"{analysis_unit}_{index}"
        reasons: list[str] = []
        if not is_block:
            reasons.append("BLOCK_INSTANCE_UNRESOLVED")
        if tactile_confidence < float(self.policy["review"]["tactile_confidence"]):
            reasons.append("LOW_TACTILE_CONFIDENCE")
        if not image_quality_ok or not input_payload["frame_quality_verified"]:
            reasons.append("FRAME_QUALITY_NOT_VERIFIED")
        if input_payload["roi_fallback_used"]:
            reasons.append("ROI_DAMAGE_FALLBACK_USED")

        expected_cfg = unit_cfg["expected_region"]
        expected = estimate_expected_region(
            unit_bbox=bbox,
            neighbor_bboxes=tactile_bboxes,
            shape=tactile_mask.shape,
            minimum_neighbor_count=int(expected_cfg["minimum_neighbor_count"]),
            maximum_size_cv=float(expected_cfg["maximum_size_cv"]),
            maximum_spacing_cv=float(expected_cfg["maximum_spacing_cv"]),
        ) if is_block else None

        damage_types: dict[str, dict[str, Any]] = {}
        total_mask = np.zeros(tactile_mask.shape, dtype=bool)
        assigned_missing = np.zeros(tactile_mask.shape, dtype=bool)
        for name in DAMAGE_CLASSES:
            raw = type_unions[name]
            confidence = _confidence(type_items[name], tactile_mask)
            if name in ("crack", "wear"):
                valid = raw & tactile_mask
                detected = bool(valid.any())
                ratio = float(valid.sum() / tactile_mask.sum() * 100.0) if tactile_mask.any() else None
                ratio_status = "estimated" if detected else "measured_no_damage"
                area_uncertain = False
            else:
                detected_near_unit = _bbox_intersects_mask(bbox, raw)
                if detected_near_unit and expected is not None and expected.mask is not None:
                    valid = raw & expected.mask
                    assigned_missing |= valid
                    detected = bool(valid.any())
                    ratio = float(valid.sum() / expected.mask.sum() * 100.0) if detected else 0.0
                    ratio_status = "estimated" if detected else "measured_no_damage"
                    area_uncertain = False
                elif detected_near_unit:
                    valid = raw & _bbox_mask(bbox, tactile_mask.shape)
                    assigned_missing |= valid
                    detected = True
                    ratio = None
                    ratio_status = "not_estimable"
                    area_uncertain = True
                    reasons.extend(["EXPECTED_BLOCK_REGION_UNAVAILABLE", "MISSING_AREA_UNCERTAIN"])
                else:
                    valid = np.zeros(tactile_mask.shape, dtype=bool)
                    detected = False
                    ratio = 0.0
                    ratio_status = "measured_no_damage"
                    area_uncertain = False
            total_mask |= valid
            if detected and (confidence is None or confidence < float(self.policy["review"]["minimum_damage_confidence_for_auto"])):
                reasons.append("LOW_DAMAGE_CONFIDENCE")
            damage_types[name] = {
                "detected": detected,
                "confidence": confidence,
                "raw_pixels": int(raw.sum()),
                "pixels": int(valid.sum()),
                "ratio_percent": None if ratio is None else round(ratio, 4),
                "ratio_status": ratio_status,
                "area_uncertain": area_uncertain,
                "bbox_xyxy": _bbox_list(valid),
                "polygons": _polygons(valid),
            }

        ratios = [
            value["ratio_percent"]
            for value in damage_types.values()
            if value["detected"]
            and value["ratio_status"] == "estimated"
            and value["ratio_percent"] is not None
        ]
        if any(self._ratio_near_threshold(float(value)) for value in ratios):
            reasons.append("RATIO_NEAR_THRESHOLD")
        decision = evaluate_unit_severity(damage_types, self.policy)
        missing_unknown = damage_types["missing"]["ratio_status"] == "not_estimable"
        missing_estimable = (
            damage_types["missing"]["detected"]
            and damage_types["missing"]["ratio_status"] == "estimated"
            and expected is not None
            and expected.mask is not None
        )
        denominator_mask = expected.mask if missing_estimable else tactile_mask
        total_ratio = (
            None
            if missing_unknown or not denominator_mask.any()
            else float((total_mask & denominator_mask).sum() / denominator_mask.sum() * 100.0)
        )
        total_status = "not_estimable" if missing_unknown else ("estimated" if total_mask.any() else "measured_no_damage")
        unit = {
            "local_unit_id": local_id,
            "analysis_unit": analysis_unit,
            "bbox_xyxy": list(bbox),
            "polygon": _polygons(tactile_mask),
            "tactile": {"confidence": tactile_confidence, "pixels": int(tactile_mask.sum())},
            "expected_region_status": expected.status if expected is not None else "not_available_for_group",
            "damage_types": damage_types,
            "analysis": {
                "damage_detected": any(value["detected"] for value in damage_types.values()),
                "total_damage_ratio_percent": None if total_ratio is None else round(total_ratio, 4),
                "ratio_status": total_status,
                "estimated_severity": decision.severity,
                "repair_priority": decision.repair_priority,
                "repair_priority_label": decision.repair_priority_label,
                "severity_reason": (
                    {
                        "dominant_damage_type": "missing",
                        "rule": "missing_ratio_not_estimable_default_severity",
                        "measured_ratio_percent": None,
                        "policy_version": self.policy["policy"]["version"],
                    }
                    if missing_unknown
                    else decision.reason
                ),
                "review_required": bool(reasons),
                "review_reasons": review_reason_list(reasons),
                "advisory_only": bool(self.policy["policy"]["advisory_only"]),
            },
        }
        return unit, total_mask, assigned_missing

    def _unresolved_missing_unit(
        self,
        mask: np.ndarray,
        items: list[tuple[float, np.ndarray]],
        index: int,
        expected_region: Any | None = None,
    ) -> dict[str, Any]:
        estimable = expected_region is not None and expected_region.mask is not None
        reasons = ["TACTILE_BLOCK_NOT_DETECTED", "BLOCK_INSTANCE_UNRESOLVED"]
        if not estimable:
            reasons.extend(["EXPECTED_BLOCK_REGION_UNAVAILABLE", "MISSING_AREA_UNCERTAIN"])
        damage_types = {
            "missing": {
                "detected": True,
                "confidence": _confidence(items, mask),
                "raw_pixels": int(mask.sum()),
                "pixels": int(mask.sum()),
                "ratio_percent": (
                    round(float(mask.sum() / expected_region.mask.sum() * 100.0), 4)
                    if estimable
                    else None
                ),
                "ratio_status": "estimated" if estimable else "not_estimable",
                "area_uncertain": not estimable,
                "bbox_xyxy": _bbox_list(mask),
                "polygons": _polygons(mask),
            },
            "crack": _empty_damage_type(),
            "wear": _empty_damage_type(),
        }
        decision = evaluate_unit_severity(damage_types, self.policy)
        return {
            "local_unit_id": f"block_group_{index}",
            "analysis_unit": "block_group",
            "bbox_xyxy": _bbox_list(mask),
            "polygon": [],
            "tactile": {"confidence": None, "pixels": 0},
            "expected_region_status": "estimated" if estimable else "unavailable",
            "damage_types": damage_types,
            "analysis": {
                "damage_detected": True,
                "total_damage_ratio_percent": damage_types["missing"]["ratio_percent"],
                "ratio_status": damage_types["missing"]["ratio_status"],
                "estimated_severity": decision.severity,
                "repair_priority": decision.repair_priority,
                "repair_priority_label": decision.repair_priority_label,
                "severity_reason": {
                    "dominant_damage_type": "missing",
                    "rule": "missing_ratio_not_estimable_default_severity",
                    "measured_ratio_percent": None,
                    "policy_version": self.policy["policy"]["version"],
                },
                "review_required": True,
                "review_reasons": review_reason_list(reasons),
                "advisory_only": True,
            },
        }

    def _payload(self, units: list[dict[str, Any]], input_payload: dict[str, Any], total_mask: np.ndarray) -> dict[str, Any]:
        if units:
            uncertain_units = [
                unit
                for unit in units
                if unit["analysis"]["damage_detected"]
                and unit["analysis"]["ratio_status"] == "not_estimable"
            ]
            worst = (
                uncertain_units[0]
                if uncertain_units
                else max(
                    units,
                    key=lambda unit: SEVERITY_ORDER[unit["analysis"]["estimated_severity"]],
                )
            )
            estimable = [unit["analysis"]["total_damage_ratio_percent"] for unit in units if unit["analysis"]["total_damage_ratio_percent"] is not None]
            summary_reasons = _merge_review_reasons(units)
            summary = {
                "damage_detected": any(unit["analysis"]["damage_detected"] for unit in units),
                "ratio_status": "not_estimable" if uncertain_units else (
                    "estimated" if any(value is not None and value > 0 for value in estimable)
                    else "measured_no_damage"
                ),
                "estimated_severity": None if uncertain_units else worst["analysis"]["estimated_severity"],
                "repair_priority": "inspection_required" if uncertain_units else worst["analysis"]["repair_priority"],
                "repair_priority_label": "담당자 검토 필요" if uncertain_units else worst["analysis"]["repair_priority_label"],
                "worst_unit_id": worst["local_unit_id"],
                "dominant_damage_type": worst["analysis"]["severity_reason"]["dominant_damage_type"],
                "max_damage_ratio_percent": max(estimable) if estimable else None,
                "mean_damage_ratio_percent": round(float(np.mean(estimable)), 4) if estimable else None,
                "not_estimable_unit_count": sum(unit["analysis"]["ratio_status"] == "not_estimable" for unit in units),
                "review_required_unit_count": sum(unit["analysis"]["review_required"] for unit in units),
                "review_required": bool(summary_reasons),
                "review_reasons": summary_reasons,
                "advisory_only": True,
            }
        else:
            reasons = review_reason_list(["TACTILE_BLOCK_NOT_DETECTED"])
            summary = {
                "damage_detected": False,
                "ratio_status": "not_estimable",
                "estimated_severity": None,
                "repair_priority": None,
                "repair_priority_label": None,
                "worst_unit_id": None,
                "dominant_damage_type": None,
                "max_damage_ratio_percent": None,
                "mean_damage_ratio_percent": None,
                "not_estimable_unit_count": 0,
                "review_required_unit_count": 0,
                "review_required": True,
                "review_reasons": reasons,
                "advisory_only": True,
            }
        if input_payload["possible_obstruction"]:
            obstruction_reason = review_reason_list(["OBSTRUCTION_SUSPECTED"])[0]
            for unit in units:
                analysis = unit["analysis"]
                detected = bool(analysis["damage_detected"])
                analysis.update(
                    {
                        "damage_detected": True if detected else None,
                        "total_damage_ratio_percent": None,
                        "ratio_status": "not_estimable",
                        "estimated_severity": None,
                        "repair_priority": "inspection_required",
                        "repair_priority_label": "담당자 검토 필요",
                        "severity_reason": {
                            "dominant_damage_type": None,
                            "rule": "obstruction_suspected",
                            "measured_ratio_percent": None,
                            "policy_version": self.policy["policy"]["version"],
                        },
                        "review_required": True,
                    }
                )
                if all(
                    reason["code"] != obstruction_reason["code"]
                    for reason in analysis["review_reasons"]
                ):
                    analysis["review_reasons"].append(obstruction_reason)
            detected = any(
                unit["analysis"]["damage_detected"] is True for unit in units
            )
            summary.update(
                {
                    "damage_detected": True if detected else None,
                    "ratio_status": "not_estimable",
                    "estimated_severity": None,
                    "repair_priority": "inspection_required",
                    "repair_priority_label": "담당자 검토 필요",
                    "max_damage_ratio_percent": None,
                    "mean_damage_ratio_percent": None,
                    "not_estimable_unit_count": len(units),
                    "review_required_unit_count": len(units),
                    "review_required": True,
                    "advisory_only": True,
                }
            )
            if all(
                reason["code"] != obstruction_reason["code"]
                for reason in summary["review_reasons"]
            ):
                summary["review_reasons"].append(obstruction_reason)
        if input_payload["edge_damage_candidate_detected"] and not summary["damage_detected"]:
            disagreement = review_reason_list(["EDGE_SERVER_DISAGREEMENT"])[0]
            if all(reason["code"] != disagreement["code"] for reason in summary["review_reasons"]):
                summary["review_reasons"].append(disagreement)
            summary["review_required"] = True
        return {
            "schema_version": "2.0",
            "model": {
                "name": self.model_path.stem,
                "task": "segmentation",
                "class_names": list(self.class_names.values()),
                "policy_version": self.policy["policy"]["version"],
                "class_mapping_valid": True,
            },
            "input": input_payload,
            "artifacts": {"original_image": None, "overlay_image": None, "mask_image": None},
            "units": units,
            "summary": summary,
            "regions": {
                "tactile_block": {"polygons": [unit["polygon"] for unit in units], "pixels": sum(unit["tactile"]["pixels"] for unit in units)},
                "damage": {"bbox_xyxy": _bbox_list(total_mask), "polygons": _polygons(total_mask), "pixels": int(total_mask.sum())},
            },
            "analysis": {
                "damage_detected": summary["damage_detected"],
                "damage_ratio_percent": summary["max_damage_ratio_percent"],
                "estimated_severity": summary["estimated_severity"],
                "repair_priority": summary["repair_priority"],
                "review_required": summary["review_required"],
                "review_reasons": summary["review_reasons"],
                "advisory_only": True,
                "deprecated": True,
            },
            "quality": {
                "positive_damage_dice": self.quality.positive_damage_dice,
                "damage_f2": self.quality.damage_f2,
                "ratio_mae_pp": self.quality.ratio_mae_pp,
                "severity_macro_f1": self.quality.severity_macro_f1,
            },
        }

    def _invalid_mapping_payload(self, input_payload: dict[str, Any]) -> dict[str, Any]:
        reasons = review_reason_list(["MODEL_CLASS_MAPPING_INVALID"])
        return {
            "schema_version": "2.0",
            "model": {
                "name": self.model_path.stem,
                "task": "segmentation",
                "class_names": list(self.class_names.values()),
                "required_class_names": list(REQUIRED_CLASSES),
                "policy_version": self.policy["policy"]["version"],
                "class_mapping_valid": False,
            },
            "input": input_payload,
            "artifacts": {"original_image": None, "overlay_image": None, "mask_image": None},
            "units": [],
            "summary": {
                "damage_detected": None,
                "ratio_status": "not_estimable",
                "estimated_severity": None,
                "repair_priority": None,
                "worst_unit_id": None,
                "dominant_damage_type": None,
                "max_damage_ratio_percent": None,
                "mean_damage_ratio_percent": None,
                "not_estimable_unit_count": 0,
                "review_required_unit_count": 0,
                "review_required": True,
                "review_reasons": reasons,
                "advisory_only": True,
            },
            "regions": {"tactile_block": {}, "damage": {}},
            "analysis": {
                "damage_detected": None,
                "damage_ratio_percent": None,
                "estimated_severity": None,
                "repair_priority": None,
                "review_required": True,
                "review_reasons": reasons,
                "advisory_only": True,
                "deprecated": True,
            },
            "quality": {},
        }

    def _input_payload(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        metadata = metadata or {}
        return {
            "original_image": metadata.get("original_image"),
            "analysis_roi": metadata.get("analysis_roi"),
            "roi_source": metadata.get("roi_source", "unknown"),
            "roi_fallback_used": bool(metadata.get("roi_fallback_used", False)),
            "frame_quality_verified": bool(metadata.get("frame_quality_verified", False)),
            "edge_damage_candidate_detected": bool(
                metadata.get("edge_damage_candidate_detected", False)
            ),
            "possible_obstruction": bool(metadata.get("possible_obstruction", False)),
            "obstruction_source": metadata.get("obstruction_source"),
        }

    def _ratio_near_threshold(self, ratio_percent: float) -> bool:
        margin = float(self.policy["review"]["boundary_margin_percent"])
        thresholds = self.policy["ratio_thresholds"]
        return any(abs(ratio_percent - float(thresholds[key])) <= margin for key in ("normal_max_percent", "minor_max_percent", "moderate_max_percent"))

    def _overlay(self, original: np.ndarray, units: list[dict[str, Any]]) -> np.ndarray:
        overlay = original.copy()
        for unit in units:
            for polygon in unit.get("polygon", []):
                points = np.asarray(polygon, dtype=np.int32)
                if len(points) >= 2:
                    cv2.polylines(overlay, [points], True, COLORS["tactile_block"], 2)
            for damage_type in DAMAGE_CLASSES:
                for polygon in unit["damage_types"][damage_type].get("polygons", []):
                    points = np.asarray(polygon, dtype=np.int32)
                    if len(points) < 3:
                        continue
                    tint = overlay.copy()
                    cv2.fillPoly(tint, [points], COLORS[damage_type])
                    overlay = cv2.addWeighted(tint, 0.30, overlay, 0.70, 0)
                    cv2.polylines(overlay, [points], True, COLORS[damage_type], 2)
            bbox = unit.get("bbox_xyxy")
            if bbox:
                ratio = unit["analysis"]["total_damage_ratio_percent"]
                ratio_text = "ratio=?" if ratio is None else f"{ratio:.1f}%"
                review = " REVIEW" if unit["analysis"]["review_required"] else ""
                label = f"{unit['local_unit_id']} {ratio_text} {unit['analysis']['estimated_severity']}{review}"
                cv2.putText(overlay, label, (bbox[0], max(20, bbox[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return overlay


def _bbox_mask(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    result = np.zeros(shape, dtype=bool)
    result[max(0, y1):min(shape[0], y2), max(0, x1):min(shape[1], x2)] = True
    return result


def _empty_damage_type() -> dict[str, Any]:
    return {
        "detected": False,
        "confidence": None,
        "raw_pixels": 0,
        "pixels": 0,
        "ratio_percent": 0.0,
        "ratio_status": "measured_no_damage",
        "area_uncertain": False,
        "bbox_xyxy": None,
        "polygons": [],
    }


def _merge_review_reasons(units: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_code: dict[str, dict[str, str]] = {}
    for unit in units:
        for reason in unit["analysis"]["review_reasons"]:
            by_code.setdefault(reason["code"], reason)
    return list(by_code.values())
