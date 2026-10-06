from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ExpectedRegionResult:
    mask: np.ndarray | None
    status: str
    reason: str | None
    size_cv: float | None = None
    spacing_cv: float | None = None


def estimate_expected_region(
    *,
    unit_bbox: tuple[int, int, int, int],
    neighbor_bboxes: list[tuple[int, int, int, int]],
    shape: tuple[int, int],
    minimum_neighbor_count: int = 3,
    maximum_size_cv: float = 0.20,
    maximum_spacing_cv: float = 0.25,
) -> ExpectedRegionResult:
    """Validate a unit-sized expected region from stable neighboring detections.

    This function never creates a grid or splits a group. It only accepts the
    current unit rectangle as an expected region when neighboring instance sizes
    and center spacing provide enough evidence that the scale is stable.
    """
    if len(neighbor_bboxes) < minimum_neighbor_count:
        return ExpectedRegionResult(None, "unavailable", "insufficient_neighbors")
    widths = np.asarray([max(1, x2 - x1) for x1, _, x2, _ in neighbor_bboxes], dtype=float)
    heights = np.asarray([max(1, y2 - y1) for _, y1, _, y2 in neighbor_bboxes], dtype=float)
    size_cv = max(_cv(widths), _cv(heights))
    if size_cv > maximum_size_cv:
        return ExpectedRegionResult(None, "unavailable", "neighbor_size_unstable", size_cv=size_cv)

    centers = np.asarray(
        [[(x1 + x2) / 2.0, (y1 + y2) / 2.0] for x1, y1, x2, y2 in neighbor_bboxes]
    )
    axis = int(np.argmax(np.ptp(centers, axis=0)))
    ordered = np.sort(centers[:, axis])
    spacing = np.diff(ordered)
    spacing = spacing[spacing > 1.0]
    if len(spacing) < minimum_neighbor_count - 1:
        return ExpectedRegionResult(None, "unavailable", "neighbor_spacing_insufficient", size_cv=size_cv)
    spacing_cv = _cv(spacing)
    if spacing_cv > maximum_spacing_cv:
        return ExpectedRegionResult(
            None,
            "unavailable",
            "neighbor_spacing_unstable",
            size_cv=size_cv,
            spacing_cv=spacing_cv,
        )

    height, width = shape
    x1, y1, x2, y2 = unit_bbox
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return ExpectedRegionResult(None, "unavailable", "unit_bbox_empty")
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return ExpectedRegionResult(
        mask,
        "estimated",
        None,
        size_cv=size_cv,
        spacing_cv=spacing_cv,
    )


def estimate_missing_expected_region(
    *,
    missing_mask: np.ndarray,
    neighbor_bboxes: list[tuple[int, int, int, int]],
    shape: tuple[int, int],
    minimum_neighbor_count: int = 3,
    maximum_size_cv: float = 0.20,
    maximum_spacing_cv: float = 0.25,
) -> ExpectedRegionResult:
    """Estimate a completely missing block from stable neighboring blocks.

    The missing mask is used only for its center. Region dimensions come from
    neighboring tactile instances, so the damage bbox is never expanded into a
    denominator. A candidate is accepted only when neighboring scale and the
    dominant row/column spacing are stable.
    """
    if not missing_mask.any():
        return ExpectedRegionResult(None, "unavailable", "missing_mask_empty")
    if len(neighbor_bboxes) < minimum_neighbor_count:
        return ExpectedRegionResult(None, "unavailable", "insufficient_neighbors")

    widths = np.asarray([max(1, x2 - x1) for x1, _, x2, _ in neighbor_bboxes], dtype=float)
    heights = np.asarray([max(1, y2 - y1) for _, y1, _, y2 in neighbor_bboxes], dtype=float)
    size_cv = max(_cv(widths), _cv(heights))
    if size_cv > maximum_size_cv:
        return ExpectedRegionResult(None, "unavailable", "neighbor_size_unstable", size_cv=size_cv)

    centers = np.asarray(
        [[(x1 + x2) / 2.0, (y1 + y2) / 2.0] for x1, y1, x2, y2 in neighbor_bboxes],
        dtype=float,
    )
    axis = int(np.argmax(np.ptp(centers, axis=0)))
    cross_axis = 1 - axis
    if _cv(np.asarray([max(1.0, value) for value in centers[:, cross_axis]], dtype=float)) > 0.50:
        return ExpectedRegionResult(None, "unavailable", "neighbor_alignment_unstable", size_cv=size_cv)
    ordered = np.sort(centers[:, axis])
    spacing = np.diff(ordered)
    spacing = spacing[spacing > 1.0]
    if len(spacing) < minimum_neighbor_count - 1:
        return ExpectedRegionResult(None, "unavailable", "neighbor_spacing_insufficient", size_cv=size_cv)
    base_spacing = float(np.min(spacing))
    normalized_spacing = spacing / base_spacing
    spacing_error = np.abs(normalized_spacing - np.maximum(1.0, np.rint(normalized_spacing)))
    spacing_cv = float(spacing_error.mean())
    if spacing_cv > maximum_spacing_cv:
        return ExpectedRegionResult(
            None,
            "unavailable",
            "neighbor_spacing_unstable",
            size_cv=size_cv,
            spacing_cv=spacing_cv,
        )

    ys, xs = np.nonzero(missing_mask)
    missing_center = np.asarray([float(xs.mean()), float(ys.mean())])
    nearest = centers[np.argmin(np.linalg.norm(centers - missing_center, axis=1))]
    projected_steps = abs(missing_center[axis] - nearest[axis]) / max(base_spacing, 1.0)
    step_error = abs(projected_steps - max(1.0, round(projected_steps)))
    cross_limit = float(np.median(heights if axis == 0 else widths)) * 0.75
    if step_error > maximum_spacing_cv or abs(missing_center[cross_axis] - nearest[cross_axis]) > cross_limit:
        return ExpectedRegionResult(
            None,
            "unavailable",
            "missing_center_off_grid",
            size_cv=size_cv,
            spacing_cv=spacing_cv,
        )

    expected_width = int(round(float(np.median(widths))))
    expected_height = int(round(float(np.median(heights))))
    height, width = shape
    x1 = max(0, int(round(missing_center[0] - expected_width / 2)))
    y1 = max(0, int(round(missing_center[1] - expected_height / 2)))
    x2 = min(width, x1 + expected_width)
    y2 = min(height, y1 + expected_height)
    if x2 <= x1 or y2 <= y1:
        return ExpectedRegionResult(None, "unavailable", "expected_region_empty")
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return ExpectedRegionResult(mask, "estimated", None, size_cv=size_cv, spacing_cv=spacing_cv)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, width, height = cv2.boundingRect(np.concatenate(contours))
    return x, y, x + width, y + height


def _cv(values: np.ndarray) -> float:
    mean = float(values.mean())
    return float(values.std() / mean) if mean > 0 else float("inf")
