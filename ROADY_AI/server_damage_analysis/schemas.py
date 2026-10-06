from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


Severity = Literal["normal", "minor", "moderate", "severe"]
RepairPriority = Literal["LOW", "NORMAL", "HIGH", "URGENT"]
DamageType = Literal["SMALL_MISSING", "LARGE_MISSING", "CRACK", "WEAR"]


class ReviewReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class AnalysisInputMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    original_image: str | None = Field(
        default=None,
        validation_alias=AliasChoices("originalImage", "original_image"),
    )
    analysis_roi: str | None = Field(
        default=None,
        validation_alias=AliasChoices("analysisRoi", "analysis_roi"),
    )
    roi_source: str = Field(
        default="unknown",
        validation_alias=AliasChoices("roiSource", "roi_source"),
    )
    roi_fallback_used: bool = Field(
        default=False,
        validation_alias=AliasChoices("roiFallbackUsed", "roi_fallback_used"),
    )
    frame_quality_verified: bool = Field(
        default=False,
        validation_alias=AliasChoices("frameQualityVerified", "frame_quality_verified"),
    )
    edge_damage_candidate_detected: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "edgeDamageCandidateDetected",
            "edge_damage_candidate_detected",
        ),
    )
    analysis_unit_hint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("analysisUnitHint", "analysis_unit_hint"),
    )
    tactile_detection_count: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "tactileDetectionCount",
            "tactile_detection_count",
        ),
    )
    frame_selection_status: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "frameSelectionStatus",
            "frame_selection_status",
        ),
    )


class ModelDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    weights: str
    weights_sha256: str
    classes: dict[str, str]


class AggregationDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_count: int = Field(ge=1)
    selected_image_index: int = Field(ge=0)
    strategy: Literal["max_damage_score"] = "max_damage_score"


class ImageAnalysisSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    filename: str
    damaged: bool | None
    damage_score: int | None = Field(default=None, ge=0, le=100)
    damage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    damage_ratio_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    estimated_severity: Severity | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    review_required: bool


class AnalysisDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    model: ModelDetail
    aggregation: AggregationDetail
    damage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    damage_ratio_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    estimated_severity: Severity | None = None
    estimated_severity_label: str | None = None
    review_required: bool
    review_reasons: list[ReviewReason]
    advisory_only: bool
    regions: dict[str, Any]
    units: list[dict[str, Any]]
    summary: dict[str, Any]
    quality: dict[str, float]
    images: list[ImageAnalysisSummary]
    inference_ms: float = Field(ge=0.0)


class DamageAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    damaged: bool | None
    damage_score: int | None = Field(default=None, ge=0, le=100)
    damage_type: DamageType | None = None
    repair_required: bool | None
    repair_priority: RepairPriority | None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    analysis_detail: AnalysisDetail


class HealthResponse(BaseModel):
    status: Literal["ok", "ready"]


class ModelInfoResponse(BaseModel):
    name: str
    weights: str
    weights_sha256: str
    classes: dict[str, str]
    device: str
    imgsz: int
