from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence


Severity = Literal["normal", "minor", "moderate", "severe", "review_required"]
EstimatedSeverity = Literal["normal", "minor", "moderate", "severe"]
RepairNeed = Literal[
    "none",
    "monitor",
    "inspection_required",
    "repair_candidate",
    "manual_review",
]
PriorityLevel = Literal["low", "medium", "high", "urgent", "manual_review"]
DamageType = Literal["unknown", "crack", "breakage", "missing", "wear"]


@dataclass(frozen=True)
class SeverityPolicy:
    """Operational policy. Thresholds are ratios in the [0, 1] range."""

    normal_ratio: float = 0.005
    minor_ratio: float = 0.05
    severe_ratio: float = 0.15
    normal_boundary_margin: float = 0.002
    boundary_margin: float = 0.02
    tactile_confidence: float = 0.50
    damage_confidence: float = 0.05
    minimum_tactile_pixels: int = 100
    minimum_mask_map50: float = 0.65
    minimum_severity_macro_f1: float = 0.80
    minimum_severe_recall: float = 0.90

    def __post_init__(self) -> None:
        if not 0 <= self.normal_ratio < self.minor_ratio < self.severe_ratio <= 1:
            raise ValueError("severity ratio thresholds must be increasing within [0, 1]")
        if self.normal_boundary_margin < 0 or self.boundary_margin < 0:
            raise ValueError("boundary margins must be non-negative")


@dataclass(frozen=True)
class ModelQuality:
    """Frozen validation metrics used to decide whether auto-decision is safe."""

    damage_mask_map50: float | None = None
    severity_macro_f1: float | None = None
    severe_recall: float | None = None


@dataclass(frozen=True)
class RepairContext:
    """Optional operational context. Each weight is normalized to [0, 1]."""

    damage_type: DamageType = "unknown"
    location_criticality: float = 0.0
    pedestrian_usage: float = 0.0
    recurrence_count: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.location_criticality <= 1:
            raise ValueError("location_criticality must be within [0, 1]")
        if not 0 <= self.pedestrian_usage <= 1:
            raise ValueError("pedestrian_usage must be within [0, 1]")
        if self.recurrence_count < 0:
            raise ValueError("recurrence_count must be non-negative")


@dataclass(frozen=True)
class SeverityResult:
    # `severity` is review_required when an automatic operational decision is unsafe.
    severity: Severity
    estimated_severity: EstimatedSeverity
    repair_need: RepairNeed
    priority_level: PriorityLevel
    priority_score: int
    damage_ratio: float
    damage_ratio_percent: float
    damage_pixels_in_block: int
    raw_damage_pixels: int
    damage_pixels_outside_block: int
    tactile_block_pixels: int
    damage_area_cm2: float | None
    tactile_block_area_cm2: float | None
    area_calculation_method: str
    tactile_block_confidence: float | None
    damage_confidence: float | None
    severity_confidence: float
    review_required: bool
    review_reasons: tuple[str, ...]
    advisory_only: bool

    @property
    def review_reason(self) -> str | None:
        """Backward-compatible first review reason."""

        return self.review_reasons[0] if self.review_reasons else None

    @property
    def repair_priority(self) -> str:
        """Backward-compatible Korean display label."""

        labels = {
            "low": "이상 없음",
            "medium": "경과 관찰",
            "high": "보수 확인 필요",
            "urgent": "보수 필수 후보",
            "manual_review": "판정 보류",
        }
        return labels[self.priority_level]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["review_reason"] = self.review_reason
        result["repair_priority"] = self.repair_priority
        return result


def classify_damage_ratio(
    ratio: float,
    policy: SeverityPolicy = SeverityPolicy(),
) -> EstimatedSeverity:
    if ratio < policy.normal_ratio:
        return "normal"
    if ratio < policy.minor_ratio:
        return "minor"
    if ratio < policy.severe_ratio:
        return "moderate"
    return "severe"


def _base_priority(severity: EstimatedSeverity) -> int:
    return {"normal": 0, "minor": 25, "moderate": 60, "severe": 90}[severity]


def calculate_priority_score(
    severity: EstimatedSeverity,
    context: RepairContext = RepairContext(),
) -> int:
    """Return a transparent 0-100 provisional repair-priority score."""

    type_bonus = {
        "unknown": 0,
        "wear": 2,
        "crack": 4,
        "breakage": 6,
        "missing": 7,
    }[context.damage_type]
    recurrence_bonus = min(context.recurrence_count, 5) / 5 * 5
    score = (
        _base_priority(severity)
        + type_bonus
        + context.location_criticality * 8
        + context.pedestrian_usage * 5
        + recurrence_bonus
    )
    return min(100, max(0, round(score)))


def _priority_level(score: int) -> PriorityLevel:
    if score >= 80:
        return "urgent"
    if score >= 50:
        return "high"
    if score >= 20:
        return "medium"
    return "low"


def _repair_need(severity: EstimatedSeverity) -> RepairNeed:
    return {
        "normal": "none",
        "minor": "monitor",
        "moderate": "inspection_required",
        "severe": "repair_candidate",
    }[severity]


def _model_quality_reasons(
    quality: ModelQuality,
    policy: SeverityPolicy,
) -> list[str]:
    reasons: list[str] = []
    if quality.damage_mask_map50 is None:
        reasons.append("mask_quality_not_measured")
    elif quality.damage_mask_map50 < policy.minimum_mask_map50:
        reasons.append("mask_quality_below_target")
    if quality.severity_macro_f1 is None:
        reasons.append("severity_quality_not_measured")
    elif quality.severity_macro_f1 < policy.minimum_severity_macro_f1:
        reasons.append("severity_macro_f1_below_target")
    if quality.severe_recall is None:
        reasons.append("severe_recall_not_measured")
    elif quality.severe_recall < policy.minimum_severe_recall:
        reasons.append("severe_recall_below_target")
    return reasons


def _area_values(
    ratio: float,
    damage_pixels: int,
    tactile_pixels: int,
    *,
    known_tactile_area_cm2: float | None,
    pixel_area_cm2: float | None,
) -> tuple[float | None, float | None, str]:
    if known_tactile_area_cm2 is not None:
        if known_tactile_area_cm2 <= 0:
            raise ValueError("known_tactile_area_cm2 must be positive")
        return (
            ratio * known_tactile_area_cm2,
            known_tactile_area_cm2,
            "known_tactile_area_ratio",
        )
    if pixel_area_cm2 is not None:
        if pixel_area_cm2 <= 0:
            raise ValueError("pixel_area_cm2 must be positive")
        return (
            damage_pixels * pixel_area_cm2,
            tactile_pixels * pixel_area_cm2,
            "pixel_calibration",
        )
    return None, None, "pixel_ratio_only"


def assess_severity(
    tactile_mask: Sequence[bool | int],
    damage_mask: Sequence[bool | int],
    *,
    tactile_confidence: float | None,
    damage_confidence: float | None,
    image_quality_ok: bool = True,
    missing_area_uncertain: bool = False,
    model_quality: ModelQuality | None = None,
    measured_damage_mask_map50: float | None = None,
    measured_severity_macro_f1: float | None = None,
    measured_severe_recall: float | None = None,
    known_tactile_area_cm2: float | None = None,
    pixel_area_cm2: float | None = None,
    context: RepairContext = RepairContext(),
    policy: SeverityPolicy = SeverityPolicy(),
) -> SeverityResult:
    """Analyze segmentation masks and produce an advisory repair decision.

    Damage ratio is intersection(damage, tactile_block) / tactile_block.
    Area in cm² is returned only when physical calibration is supplied.
    """

    if len(tactile_mask) != len(damage_mask):
        raise ValueError("tactile_mask and damage_mask must have the same number of pixels")
    if model_quality is None:
        model_quality = ModelQuality(
            damage_mask_map50=measured_damage_mask_map50,
            severity_macro_f1=measured_severity_macro_f1,
            severe_recall=measured_severe_recall,
        )

    tactile = [bool(value) for value in tactile_mask]
    damage = [bool(value) for value in damage_mask]
    tactile_pixels = sum(tactile)
    raw_damage_pixels = sum(damage)
    damage_pixels = sum(block and defect for block, defect in zip(tactile, damage))
    outside_pixels = raw_damage_pixels - damage_pixels
    ratio = 0.0 if tactile_pixels == 0 else damage_pixels / tactile_pixels
    estimated = classify_damage_ratio(ratio, policy)

    damage_area, tactile_area, area_method = _area_values(
        ratio,
        damage_pixels,
        tactile_pixels,
        known_tactile_area_cm2=known_tactile_area_cm2,
        pixel_area_cm2=pixel_area_cm2,
    )

    reasons: list[str] = []
    if not image_quality_ok:
        reasons.append("image_quality_insufficient")
    if missing_area_uncertain:
        reasons.append("missing_area_uncertain")
    if tactile_confidence is None or tactile_confidence < policy.tactile_confidence:
        reasons.append("tactile_block_confidence_low")
    if tactile_pixels < policy.minimum_tactile_pixels:
        reasons.append("tactile_block_mask_missing_or_small")
    if damage_pixels and (damage_confidence is None or damage_confidence < policy.damage_confidence):
        reasons.append("damage_confidence_low")

    boundary_pairs = (
        (policy.normal_ratio, policy.normal_boundary_margin),
        (policy.minor_ratio, policy.boundary_margin),
        (policy.severe_ratio, policy.boundary_margin),
    )
    nearest_boundary = min(abs(ratio - boundary) for boundary, _ in boundary_pairs)
    if any(margin > 0 and abs(ratio - boundary) <= margin for boundary, margin in boundary_pairs):
        reasons.append("severity_boundary_ambiguous")

    reasons.extend(_model_quality_reasons(model_quality, policy))
    reasons = list(dict.fromkeys(reasons))
    review_required = bool(reasons)

    priority_score = calculate_priority_score(estimated, context)
    if review_required:
        severity: Severity = "review_required"
        priority_level: PriorityLevel = "manual_review"
        repair_need: RepairNeed = "manual_review"
    else:
        severity = estimated
        priority_level = _priority_level(priority_score)
        repair_need = _repair_need(estimated)

    confidence_values = [
        value
        for value in (tactile_confidence, damage_confidence if damage_pixels else 1.0)
        if value is not None
    ]
    model_confidence = min(confidence_values) if confidence_values else 0.0
    relevant_margin = min(
        margin for boundary, margin in boundary_pairs if abs(ratio - boundary) == nearest_boundary
    )
    boundary_confidence = min(1.0, nearest_boundary / max(relevant_margin, 1e-9))
    severity_confidence = 0.0 if review_required else min(model_confidence, boundary_confidence)

    return SeverityResult(
        severity=severity,
        estimated_severity=estimated,
        repair_need=repair_need,
        priority_level=priority_level,
        priority_score=priority_score,
        damage_ratio=ratio,
        damage_ratio_percent=round(ratio * 100, 4),
        damage_pixels_in_block=damage_pixels,
        raw_damage_pixels=raw_damage_pixels,
        damage_pixels_outside_block=outside_pixels,
        tactile_block_pixels=tactile_pixels,
        damage_area_cm2=None if damage_area is None else round(damage_area, 4),
        tactile_block_area_cm2=None if tactile_area is None else round(tactile_area, 4),
        area_calculation_method=area_method,
        tactile_block_confidence=tactile_confidence,
        damage_confidence=damage_confidence,
        severity_confidence=round(severity_confidence, 4),
        review_required=review_required,
        review_reasons=tuple(reasons),
        advisory_only=True,
    )
