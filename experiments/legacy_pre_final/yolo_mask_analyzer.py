from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ai.server.severity import (
    ModelQuality,
    RepairContext,
    SeverityPolicy,
    SeverityResult,
    assess_severity,
)


TACTILE_BLOCK_CLASS_ID = 0
DAMAGE_CLASS_ID = 1


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def analyze_yolo_segmentation(
    result: Any,
    *,
    tactile_threshold: float = 0.50,
    damage_threshold: float = 0.05,
    image_quality_ok: bool = True,
    missing_area_uncertain: bool = False,
    model_quality: ModelQuality = ModelQuality(),
    known_tactile_area_cm2: float | None = None,
    pixel_area_cm2: float | None = None,
    context: RepairContext = RepairContext(),
    policy: SeverityPolicy = SeverityPolicy(),
) -> SeverityResult:
    """Convert one Ultralytics segmentation Result into a severity result."""

    height, width = result.orig_shape
    tactile_mask = np.zeros((height, width), dtype=bool)
    damage_mask = np.zeros((height, width), dtype=bool)
    tactile_confidences: list[float] = []
    damage_confidences: list[float] = []

    if result.boxes is not None and result.masks is not None:
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
        masks = result.masks.data.detach().cpu().numpy()
        for class_id, confidence, mask in zip(classes, confidences, masks):
            if class_id == TACTILE_BLOCK_CLASS_ID and confidence >= tactile_threshold:
                tactile_mask |= _resize_mask(mask, height, width)
                tactile_confidences.append(float(confidence))
            elif class_id == DAMAGE_CLASS_ID and confidence >= damage_threshold:
                damage_mask |= _resize_mask(mask, height, width)
                damage_confidences.append(float(confidence))

    return assess_severity(
        tactile_mask.ravel(),
        damage_mask.ravel(),
        tactile_confidence=max(tactile_confidences, default=None),
        damage_confidence=max(damage_confidences, default=None),
        image_quality_ok=image_quality_ok,
        missing_area_uncertain=missing_area_uncertain,
        model_quality=model_quality,
        known_tactile_area_cm2=known_tactile_area_cm2,
        pixel_area_cm2=pixel_area_cm2,
        context=context,
        policy=policy,
    )
