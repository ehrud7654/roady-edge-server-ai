from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .policy import load_severity_policy
from .schemas import (
    AggregationDetail,
    AnalysisDetail,
    DamageAnalysisResponse,
    ImageAnalysisSummary,
    ModelDetail,
    ReviewReason,
)


PRIORITY_BY_SEVERITY = {
    "normal": None,
    "minor": "LOW",
    "moderate": "NORMAL",
    "severe": "HIGH",
}

DEFAULT_MINOR_MAX_PIXELS = 10_000
DEFAULT_MODERATE_MAX_PIXELS = 50_000
DEFAULT_MAX_SCORE_PIXELS = 300_000

MISSING_LARGE_THRESHOLD_PERCENT = 15.0

_DAMAGE_TYPE_POLICY = load_severity_policy()["damage_types"]
DAMAGE_TYPE_SELECTION_WEIGHTS = {
    damage_type: float(settings["weight"])
    for damage_type, settings in _DAMAGE_TYPE_POLICY.items()
}
DAMAGE_TYPE_RESPONSE_NAMES = {
    "crack": "CRACK",
    "wear": "WEAR",
}

SEVERITY_LABELS = {
    "normal": "정상 추정",
    "minor": "경미 추정",
    "moderate": "보통 추정",
    "severe": "심각 추정",
}


class Analyzer(Protocol):
    def predict(
        self,
        image: np.ndarray,
        *,
        imgsz: int,
        device: str | int,
        image_quality_ok: bool = True,
        input_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]: ...


class AnalysisInputError(ValueError):
    pass


@dataclass(frozen=True)
class InputImage:
    filename: str
    image: np.ndarray
    input_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class _AnalyzedImage:
    index: int
    filename: str
    payload: dict[str, Any]
    damage_pixels: int | None
    damage_score: int | None
    damage_ratio: float | None
    damage_ratio_percent: float | None
    severity: str | None
    confidence_score: float | None
    damage_detected: bool | None

    @property
    def damaged(self) -> bool | None:
        return self.damage_detected


def calculate_damage_score(
    damage_pixels: int,
    *,
    minor_max_pixels: int = DEFAULT_MINOR_MAX_PIXELS,
    moderate_max_pixels: int = DEFAULT_MODERATE_MAX_PIXELS,
    max_score_pixels: int = DEFAULT_MAX_SCORE_PIXELS,
) -> int:
    """Map the integrated damage-mask pixel count to a 0-100 score."""
    if not 1 < minor_max_pixels < moderate_max_pixels < max_score_pixels:
        raise ValueError(
            "Damage score pixel thresholds must start above 1 and be strictly increasing."
        )

    pixels = max(0, int(damage_pixels))
    if pixels == 0:
        return 0
    if pixels <= minor_max_pixels:
        return round(1 + ((pixels - 1) / (minor_max_pixels - 1)) * 29)
    if pixels <= moderate_max_pixels:
        return round(
            31
            + ((pixels - minor_max_pixels) / (moderate_max_pixels - minor_max_pixels))
            * 39
        )
    if pixels <= max_score_pixels:
        return round(
            71
            + ((pixels - moderate_max_pixels) / (max_score_pixels - moderate_max_pixels))
            * 29
        )
    return 100


def severity_from_damage_score(damage_score: int | None) -> str | None:
    """Derive the service severity used for repair decisions from a damage score."""
    if damage_score is None or damage_score <= 0:
        return None
    if damage_score <= 30:
        return "minor"
    if damage_score <= 70:
        return "moderate"
    return "severe"


def read_and_verify_sha256(
    model_path: Path,
    sha256_path: Path,
    *,
    verify: bool,
) -> str:
    if not model_path.is_file():
        raise FileNotFoundError(f"AI model not found: {model_path}")
    if not sha256_path.is_file():
        raise FileNotFoundError(f"AI model SHA-256 file not found: {sha256_path}")

    expected = sha256_path.read_text(encoding="utf-8").strip().split()[0].lower()
    if len(expected) != 64:
        raise ValueError(f"Invalid AI model SHA-256: {sha256_path}")
    if verify:
        digest = hashlib.sha256()
        with model_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"AI model SHA-256 mismatch: expected={expected}, actual={actual}"
            )
    return expected


class DamageAnalysisService:
    def __init__(
        self,
        analyzer: Analyzer,
        *,
        model_path: Path,
        model_sha256: str,
        imgsz: int,
        device: str,
        score_minor_max_pixels: int = DEFAULT_MINOR_MAX_PIXELS,
        score_moderate_max_pixels: int = DEFAULT_MODERATE_MAX_PIXELS,
        score_max_pixels: int = DEFAULT_MAX_SCORE_PIXELS,
    ) -> None:
        self.analyzer = analyzer
        self.model_path = model_path
        self.model_sha256 = model_sha256
        self.imgsz = imgsz
        self.device = device
        if not 1 < score_minor_max_pixels < score_moderate_max_pixels < score_max_pixels:
            raise ValueError(
                "Damage score pixel thresholds must start above 1 and be strictly increasing."
            )
        self.score_minor_max_pixels = score_minor_max_pixels
        self.score_moderate_max_pixels = score_moderate_max_pixels
        self.score_max_pixels = score_max_pixels

    def analyze_images(self, images: Sequence[InputImage]) -> DamageAnalysisResponse:
        if not images:
            raise AnalysisInputError("At least one image is required.")

        started_at = time.perf_counter()
        analyzed = [self._analyze_one(index, item) for index, item in enumerate(images)]
        selected = max(
            analyzed,
            key=lambda item: (
                item.damage_score if item.damage_score is not None else -1,
                item.damage_pixels if item.damage_pixels is not None else -1,
                item.damage_ratio if item.damage_ratio is not None else -1.0,
                item.confidence_score if item.confidence_score is not None else -1.0,
            ),
        )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)

        analysis = selected.payload["analysis"]
        summary = selected.payload.get("summary", analysis)
        model = selected.payload["model"]
        response_severity = self._response_severity(selected)
        repair_required = (
            None if selected.damaged is None else selected.damaged
        )
        severity_label = analysis.get("estimated_severity_label")
        if response_severity != selected.severity:
            severity_label = None
        if severity_label is None and response_severity is not None:
            severity_label = SEVERITY_LABELS[response_severity]

        return DamageAnalysisResponse(
            damaged=selected.damaged,
            damage_score=selected.damage_score,
            damage_type=self._damage_type(selected.payload),
            repair_required=repair_required,
            repair_priority=(
                None
                if selected.damaged is not True or response_severity is None
                else PRIORITY_BY_SEVERITY[response_severity]
            ),
            confidence_score=(
                selected.confidence_score if selected.damaged is True else None
            ),
            analysis_detail=AnalysisDetail(
                model=ModelDetail(
                    name=str(model["name"]),
                    weights=str(model.get("weights", self.model_path.name)),
                    weights_sha256=self.model_sha256,
                    classes=self._model_classes(model),
                ),
                aggregation=AggregationDetail(
                    image_count=len(analyzed),
                    selected_image_index=selected.index,
                ),
                damage_ratio=selected.damage_ratio,
                damage_ratio_percent=selected.damage_ratio_percent,
                estimated_severity=response_severity,
                estimated_severity_label=(
                    None if severity_label is None else str(severity_label)
                ),
                review_required=bool(summary.get("review_required", False)),
                review_reasons=self._review_reasons(summary),
                advisory_only=bool(summary.get("advisory_only", True)),
                regions=selected.payload["regions"],
                units=list(selected.payload.get("units", [])),
                summary=dict(selected.payload.get("summary", analysis)),
                quality={
                    str(key): float(value)
                    for key, value in selected.payload.get("quality", {}).items()
                },
                images=[self._summary(item) for item in analyzed],
                inference_ms=elapsed_ms,
            ),
        )

    def warmup(self) -> None:
        image = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.analyzer.predict(image, imgsz=self.imgsz, device=self.device)

    def _analyze_one(self, index: int, item: InputImage) -> _AnalyzedImage:
        payload, _, _ = self.analyzer.predict(
            item.image,
            imgsz=self.imgsz,
            device=self.device,
            input_metadata=item.input_metadata,
        )
        analysis = payload["analysis"]
        summary = payload.get("summary", analysis)
        raw_ratio_percent = summary.get("max_damage_ratio_percent")
        if raw_ratio_percent is None:
            raw_ratio_percent = analysis.get("damage_ratio_percent")
        if raw_ratio_percent is None and analysis.get("damage_ratio") is not None:
            raw_ratio_percent = float(analysis["damage_ratio"]) * 100
        ratio_percent = (
            None
            if raw_ratio_percent is None
            else round(max(0.0, min(float(raw_ratio_percent), 100.0)), 2)
        )
        ratio = None if ratio_percent is None else round(ratio_percent / 100, 6)
        raw_severity = summary.get("estimated_severity", analysis.get("estimated_severity"))
        severity = None if raw_severity is None else str(raw_severity)
        if severity is not None and severity not in PRIORITY_BY_SEVERITY:
            raise RuntimeError(f"Unknown estimated severity: {severity}")
        damage_pixels = self._damage_pixels(payload)
        provisional_score = (
            None
            if damage_pixels is None
            else calculate_damage_score(
                damage_pixels,
                minor_max_pixels=self.score_minor_max_pixels,
                moderate_max_pixels=self.score_moderate_max_pixels,
                max_score_pixels=self.score_max_pixels,
            )
        )
        damage_detected = self._damage_detected(
            payload,
            summary,
            analysis,
            provisional_score,
        )
        raw_confidence = payload.get("regions", {}).get("damage", {}).get("confidence")
        if raw_confidence is None:
            raw_confidence = self._maximum_damage_confidence(payload)
        confidence = (
            None
            if raw_confidence is None
            else round(max(0.0, min(float(raw_confidence), 1.0)), 4)
        )
        return _AnalyzedImage(
            index=index,
            filename=item.filename,
            payload=payload,
            damage_pixels=damage_pixels,
            damage_score=provisional_score,
            damage_ratio=ratio,
            damage_ratio_percent=ratio_percent,
            severity=severity,
            confidence_score=confidence,
            damage_detected=damage_detected,
        )

    @staticmethod
    def _damage_detected(
        payload: dict[str, Any],
        summary: dict[str, Any],
        analysis: dict[str, Any],
        score: int | None,
    ) -> bool | None:
        if str(payload.get("schema_version", "1.0")) != "2.0":
            return bool(score and score > 0)
        value = summary.get("damage_detected", analysis.get("damage_detected"))
        return value if isinstance(value, bool) else None

    @staticmethod
    def _damage_pixels(payload: dict[str, Any]) -> int | None:
        value = payload.get("regions", {}).get("damage", {}).get("pixels")
        if value is None:
            return None
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"Invalid damage pixel count: {value}") from exc

    @staticmethod
    def _damage_type(payload: dict[str, Any]) -> str | None:
        if str(payload.get("schema_version", "1.0")) != "2.0":
            return None

        detected_types: dict[str, dict[str, Any]] = {}
        for unit in payload.get("units", []):
            for damage_type, details in unit.get("damage_types", {}).items():
                if (
                    damage_type not in DAMAGE_TYPE_SELECTION_WEIGHTS
                    or details.get("detected") is not True
                ):
                    continue

                ratio_percent = DamageAnalysisService._bounded_damage_type_value(
                    details.get("ratio_percent"),
                    maximum=100.0,
                    field_name=f"{damage_type}.ratio_percent",
                    preserve_none=True,
                )
                pixels = DamageAnalysisService._damage_type_pixels(
                    details.get("pixels"),
                    damage_type=damage_type,
                )
                aggregate = detected_types.setdefault(
                    damage_type,
                    {
                        "pixels": 0,
                        "ratios": [],
                        "ratio_not_estimable": False,
                        "tactile_pixels": 0,
                    },
                )
                aggregate["pixels"] += pixels
                if ratio_percent is not None:
                    aggregate["ratios"].append(ratio_percent)
                else:
                    aggregate["ratio_not_estimable"] = True
                if damage_type == "missing":
                    aggregate["tactile_pixels"] += (
                        DamageAnalysisService._damage_type_pixels(
                            unit.get("tactile", {}).get("pixels"),
                            damage_type="tactile",
                        )
                    )

        candidates: list[tuple[float, float, str, float | None]] = []
        for damage_type, aggregate in detected_types.items():
            weight = DAMAGE_TYPE_SELECTION_WEIGHTS[damage_type]
            selection_score = aggregate["pixels"] * weight
            ratios = aggregate["ratios"]
            ratio = max(ratios) if ratios else None
            if damage_type == "missing" and aggregate["ratio_not_estimable"]:
                ratio = DamageAnalysisService._fallback_missing_ratio_percent(
                    missing_pixels=aggregate["pixels"],
                    tactile_pixels=aggregate["tactile_pixels"],
                )
            candidates.append(
                (
                    selection_score,
                    weight,
                    damage_type,
                    ratio,
                )
            )

        if not candidates:
            return None

        _, _, selected_type, ratio = max(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        if selected_type in DAMAGE_TYPE_RESPONSE_NAMES:
            return DAMAGE_TYPE_RESPONSE_NAMES[selected_type]

        if ratio is None:
            return "LARGE_MISSING"
        return (
            "LARGE_MISSING"
            if ratio >= MISSING_LARGE_THRESHOLD_PERCENT
            else "SMALL_MISSING"
        )

    @staticmethod
    def _fallback_missing_ratio_percent(
        *,
        missing_pixels: int,
        tactile_pixels: int,
    ) -> float | None:
        total_pixels = missing_pixels + tactile_pixels
        if total_pixels <= 0:
            return None
        return round(missing_pixels / total_pixels * 100.0, 4)

    @staticmethod
    def _damage_type_pixels(value: Any, *, damage_type: str) -> int:
        if value is None:
            return 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Invalid damage type pixel count for {damage_type}: {value}"
            ) from exc

    @staticmethod
    def _bounded_damage_type_value(
        value: Any,
        *,
        maximum: float,
        field_name: str,
        preserve_none: bool = False,
    ) -> float | None:
        if value is None:
            return None if preserve_none else 0.0
        try:
            return max(0.0, min(float(value), maximum))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Invalid damage type value for {field_name}: {value}"
            ) from exc

    @staticmethod
    def _review_reasons(summary: dict[str, Any]) -> list[ReviewReason]:
        normalized: list[ReviewReason] = []
        for reason in summary.get("review_reasons", []):
            if isinstance(reason, dict):
                code = str(reason.get("code", "UNKNOWN"))
                message = str(reason.get("message", code))
            else:
                code = str(reason)
                message = code
            normalized.append(ReviewReason(code=code, message=message))
        return normalized

    @staticmethod
    def _model_classes(model: dict[str, Any]) -> dict[str, str]:
        classes = model.get("classes")
        if isinstance(classes, dict):
            return {str(key): str(value) for key, value in classes.items()}
        return {
            str(index): str(name)
            for index, name in enumerate(model.get("class_names", []))
        }

    @staticmethod
    def _maximum_damage_confidence(payload: dict[str, Any]) -> float | None:
        values = [
            details.get("confidence")
            for unit in payload.get("units", [])
            for name, details in unit.get("damage_types", {}).items()
            if name in {"missing", "crack", "wear"}
            and details.get("confidence") is not None
        ]
        return max(values) if values else None

    def _summary(self, item: _AnalyzedImage) -> ImageAnalysisSummary:
        analysis = item.payload["analysis"]
        return ImageAnalysisSummary(
            index=item.index,
            filename=item.filename,
            damaged=item.damaged,
            damage_score=item.damage_score,
            damage_ratio=item.damage_ratio,
            damage_ratio_percent=item.damage_ratio_percent,
            estimated_severity=self._response_severity(item),
            confidence_score=item.confidence_score if item.damaged is True else None,
            review_required=bool(
                item.payload.get("summary", analysis).get("review_required", False)
            ),
        )

    @staticmethod
    def _response_severity(item: _AnalyzedImage) -> str | None:
        if item.damaged is not True:
            return item.severity

        score_severity = severity_from_damage_score(item.damage_score)
        if score_severity is not None:
            return score_severity

        # A positive model decision must produce a usable repair priority even
        # when the damage ratio or mask pixel count cannot be estimated.
        return "moderate"
