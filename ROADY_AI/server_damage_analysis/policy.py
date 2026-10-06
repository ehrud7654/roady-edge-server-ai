from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AnalysisPolicy:
    """Deprecated v1 policy whose ratio values use the 0~1 range."""

    normal_limit: float = 0.005
    minor_limit: float = 0.05
    moderate_limit: float = 0.15
    boundary_margin: float = 0.01
    tactile_confidence: float = 0.25
    damage_confidence: float = 0.05
    duplicate_mask_iou: float = 0.70
    minimum_tactile_pixels: int = 100
    minimum_damage_confidence_for_auto: float = 0.70


@dataclass(frozen=True)
class ModelQuality:
    """Fixed validation metrics exposed for diagnostics only."""

    positive_damage_dice: float = 0.08300269501207208
    damage_f2: float = 0.826086956521739
    ratio_mae_pp: float = 6.084992637708795
    severity_macro_f1: float = 0.21873215305539692


SEVERITY_LABELS = {
    "normal": "정상 추정",
    "minor": "경미 추정",
    "moderate": "보통 추정",
    "severe": "심각 추정",
}

PRIORITY_BY_SEVERITY = {
    "normal": ("none", "이상 없음"),
    "minor": ("monitor", "경과 관찰"),
    "moderate": ("inspection_required", "보수 확인 필요"),
    "severe": ("repair_candidate", "보수 필수 후보"),
}


def classify_severity(ratio: float, policy: AnalysisPolicy) -> str:
    if ratio < policy.normal_limit:
        return "normal"
    if ratio < policy.minor_limit:
        return "minor"
    if ratio < policy.moderate_limit:
        return "moderate"
    return "severe"


def decide_review(
    *,
    ratio: float,
    tactile_pixels: int,
    tactile_confidence: float | None,
    damage_confidence: float | None,
    image_quality_ok: bool,
    quality: ModelQuality,
    policy: AnalysisPolicy,
) -> list[str]:
    """Deprecated v1 review policy retained for API compatibility."""
    reasons: list[str] = []
    if not image_quality_ok:
        reasons.append("image_quality_insufficient")
    if tactile_pixels < policy.minimum_tactile_pixels:
        reasons.append("tactile_mask_missing_or_small")
    if tactile_confidence is None or tactile_confidence < policy.tactile_confidence:
        reasons.append("tactile_confidence_low")
    if ratio > 0 and (
        damage_confidence is None
        or damage_confidence < policy.minimum_damage_confidence_for_auto
    ):
        reasons.append("damage_confidence_low")
    if any(
        abs(ratio - threshold) <= policy.boundary_margin
        for threshold in (policy.normal_limit, policy.minor_limit, policy.moderate_limit)
    ):
        reasons.append("severity_boundary_ambiguous")
    return list(dict.fromkeys(reasons))


SEVERITY_ORDER = {"normal": 0, "minor": 1, "moderate": 2, "severe": 3}


@dataclass(frozen=True)
class SeverityDecision:
    severity: str
    repair_priority: str
    repair_priority_label: str
    reason: dict[str, Any]


def load_severity_policy(path: str | Path | None = None) -> dict[str, Any]:
    policy_path = Path(path) if path else Path(__file__).with_name("severity_policy.yaml")
    with policy_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError("severity policy must be a YAML object")
    return payload


def severity_for_percent(ratio_percent: float, policy: dict[str, Any]) -> str:
    thresholds = policy["ratio_thresholds"]
    if ratio_percent < float(thresholds["normal_max_percent"]):
        return "normal"
    if ratio_percent < float(thresholds["minor_max_percent"]):
        return "minor"
    if ratio_percent < float(thresholds["moderate_max_percent"]):
        return "moderate"
    return "severe"


def evaluate_unit_severity(
    damage_types: dict[str, dict[str, Any]], policy: dict[str, Any]
) -> SeverityDecision:
    """Return the maximum subtype severity and an explicit decision reason."""
    decisions: list[tuple[int, str, str, float | None, str]] = []
    for damage_type, result in damage_types.items():
        if not result.get("detected", False):
            continue
        ratio = result.get("ratio_percent")
        type_policy = policy["damage_types"][damage_type]
        floor = str(type_policy["minimum_severity"])
        measured = severity_for_percent(float(ratio), policy) if ratio is not None else "normal"
        severity = max((floor, measured), key=SEVERITY_ORDER.__getitem__)
        rule = (
            f"{damage_type}_ratio_above_{severity}_threshold"
            if ratio is not None and SEVERITY_ORDER[measured] >= SEVERITY_ORDER[floor]
            else f"{damage_type}_minimum_severity"
        )
        decisions.append((SEVERITY_ORDER[severity], severity, damage_type, ratio, rule))

    if not decisions:
        severity = "normal"
        dominant = None
        ratio = 0.0
        rule = "no_damage_detected"
    else:
        _, severity, dominant, ratio, rule = max(decisions, key=lambda item: item[0])
    priority, priority_label = PRIORITY_BY_SEVERITY[severity]
    return SeverityDecision(
        severity=severity,
        repair_priority=priority,
        repair_priority_label=priority_label,
        reason={
            "dominant_damage_type": dominant,
            "rule": rule,
            "measured_ratio_percent": ratio,
            "policy_version": policy["policy"]["version"],
        },
    )
