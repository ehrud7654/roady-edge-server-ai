from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ROADY_AI.server_damage_analysis.service import (
    AnalysisInputError,
    DamageAnalysisService,
    InputImage,
    calculate_damage_score,
)


@pytest.mark.parametrize(
    ("pixels", "expected"),
    [
        (0, 0),
        (1, 1),
        (5_000, 15),
        (10_000, 30),
        (10_001, 31),
        (19_260, 40),
        (50_000, 70),
        (50_001, 71),
        (75_000, 74),
        (100_000, 77),
        (175_000, 86),
        (300_000, 100),
        (300_001, 100),
    ],
)
def test_damage_score_uses_only_damage_mask_pixels(pixels: int, expected: int):
    assert calculate_damage_score(pixels) == expected


def test_service_maps_model_result_to_database_contract():
    service = make_service(
        FakeAnalyzer([payload(0.0859, 0.84321, damage_pixels=19_260)])
    )

    response = service.analyze_images([input_image("damage.jpg")])

    assert response.damaged is True
    assert response.damage_score == 40
    assert response.damage_type is None
    assert response.repair_required is True
    assert response.repair_priority == "NORMAL"
    assert response.confidence_score == 0.8432
    assert response.analysis_detail.damage_ratio_percent == 8.59
    assert response.analysis_detail.estimated_severity == "moderate"
    assert response.analysis_detail.model.weights_sha256 == "a" * 64


def test_service_selects_highest_risk_image():
    service = make_service(
        FakeAnalyzer(
            [
                payload(0.02, 0.91),
                payload(0.20, 0.81, damage_pixels=80_000),
                payload(0.08, 0.95, damage_pixels=30_000),
            ]
        )
    )

    response = service.analyze_images(
        [
            input_image("minor.jpg"),
            input_image("severe.jpg"),
            input_image("moderate.jpg"),
        ]
    )

    assert response.damage_score == 74
    assert response.repair_required is True
    assert response.repair_priority == "HIGH"
    assert response.confidence_score == 0.81
    assert response.analysis_detail.aggregation.image_count == 3
    assert response.analysis_detail.aggregation.selected_image_index == 1
    assert len(response.analysis_detail.images) == 3


def test_service_returns_normal_contract_without_damage_pixels():
    service = make_service(FakeAnalyzer([payload(0.004, 0.88, damage_pixels=0)]))

    response = service.analyze_images([input_image("normal.jpg")])

    assert response.damaged is False
    assert response.damage_score == 0
    assert response.repair_required is False
    assert response.repair_priority is None
    assert response.confidence_score is None


def test_service_returns_review_contract_without_tactile_block():
    service = make_service(FakeAnalyzer([payload(0.0, None, tactile_pixels=0)]))

    response = service.analyze_images([input_image("missing.jpg")])
    assert response.damaged is False
    assert response.analysis_detail.review_required is True


def test_service_scores_unknown_missing_from_damage_pixels():
    service = make_service(FakeAnalyzer([v2_unknown_missing_payload()]))

    response = service.analyze_images([input_image("unknown-missing.jpg")])

    assert response.damaged is True
    assert response.damage_score == 40
    assert response.damage_type == "LARGE_MISSING"
    assert response.repair_required is True
    assert response.repair_priority == "NORMAL"
    assert response.analysis_detail.damage_ratio is None
    assert response.analysis_detail.damage_ratio_percent is None
    assert response.analysis_detail.estimated_severity == "moderate"
    assert response.analysis_detail.review_required is True
    assert response.analysis_detail.schema_version == "2.0"


@pytest.mark.parametrize(
    ("tactile_pixels", "missing_pixels", "expected_type"),
    [
        (189_516, 9_845, "SMALL_MISSING"),
        (188_467, 85_555, "LARGE_MISSING"),
    ],
)
def test_service_falls_back_to_tactile_and_missing_pixel_ratio(
    tactile_pixels: int,
    missing_pixels: int,
    expected_type: str,
):
    payload_value = v2_unknown_missing_payload()
    unit = payload_value["units"][0]
    unit["tactile"] = {"confidence": 0.9, "pixels": tactile_pixels}
    unit["damage_types"]["missing"]["pixels"] = missing_pixels
    service = make_service(FakeAnalyzer([payload_value]))

    response = service.analyze_images([input_image("unknown-missing-ratio.jpg")])

    assert response.damage_type == expected_type


@pytest.mark.parametrize(
    ("dominant", "ratio_percent", "expected_type"),
    [
        ("missing", 14.99, "SMALL_MISSING"),
        ("missing", 15.0, "LARGE_MISSING"),
        ("crack", 0.12, "CRACK"),
        ("wear", 3.0, "WEAR"),
    ],
)
def test_service_maps_v2_dominant_damage_type(
    dominant: str,
    ratio_percent: float,
    expected_type: str,
):
    service = make_service(FakeAnalyzer([v2_payload(dominant, ratio_percent)]))

    response = service.analyze_images([input_image("v2-damage.jpg")])

    assert response.damaged is True
    assert response.damage_type == expected_type


@pytest.mark.parametrize(
    ("missing_pixels", "wear_pixels", "expected_type"),
    [
        (15_750, 666, "WEAR"),
        (85_555, 574, "LARGE_MISSING"),
        (28_399, 379, "LARGE_MISSING"),
    ],
)
def test_service_selects_type_by_weighted_pixels(
    missing_pixels: int,
    wear_pixels: int,
    expected_type: str,
):
    payload_value = v2_payload("missing", 90.0, confidence=1.0)
    damage_types = payload_value["units"][0]["damage_types"]
    damage_types["missing"]["pixels"] = missing_pixels
    damage_types["wear"] = {
        "detected": True,
        "confidence": 0.01,
        "ratio_percent": 0.01,
        "pixels": wear_pixels,
    }
    service = make_service(FakeAnalyzer([payload_value]))

    response = service.analyze_images([input_image("weighted-damage-types.jpg")])

    assert response.damage_type == expected_type


def test_service_selects_larger_weighted_pixels_between_crack_and_wear():
    payload_value = v2_payload("wear", 90.0, confidence=1.0)
    payload_value["units"][0]["damage_types"]["wear"]["pixels"] = 900
    payload_value["units"][0]["damage_types"]["crack"] = {
        "detected": True,
        "confidence": 0.01,
        "ratio_percent": 0.01,
        "pixels": 1_000,
    }
    service = make_service(FakeAnalyzer([payload_value]))

    response = service.analyze_images([input_image("crack-and-wear.jpg")])

    assert response.damage_type == "CRACK"


def test_service_uses_v2_detection_independently_from_damage_score():
    service = make_service(FakeAnalyzer([v2_payload("crack", 0.12)]))

    response = service.analyze_images([input_image("small-crack.jpg")])

    assert response.damaged is True
    assert response.damage_score == 1
    assert response.damage_type == "CRACK"
    assert response.repair_required is True
    assert response.repair_priority == "LOW"
    assert response.analysis_detail.estimated_severity == "minor"


def test_service_keeps_unknown_model_decision_null():
    payload_value = v2_payload("crack", 1.0)
    payload_value["summary"]["damage_detected"] = None
    payload_value["analysis"]["damage_detected"] = None
    service = make_service(FakeAnalyzer([payload_value]))

    response = service.analyze_images([input_image("invalid-model.jpg")])

    assert response.damaged is None
    assert response.confidence_score is None


class FakeAnalyzer:
    def __init__(self, payloads: list[dict]):
        self.payloads = iter(payloads)
        self.received_metadata: list[dict | None] = []

    def predict(
        self,
        image,
        *,
        imgsz,
        device,
        image_quality_ok=True,
        input_metadata=None,
    ):
        del image, imgsz, device, image_quality_ok
        self.received_metadata.append(input_metadata)
        return next(self.payloads), np.zeros((2, 2, 3)), np.zeros((2, 2))


def make_service(analyzer: FakeAnalyzer) -> DamageAnalysisService:
    return DamageAnalysisService(
        analyzer,
        model_path=Path("yolo26s_seg_v1_best.pt"),
        model_sha256="a" * 64,
        imgsz=1024,
        device="cpu",
    )


def input_image(filename: str) -> InputImage:
    return InputImage(filename=filename, image=np.zeros((10, 10, 3), dtype=np.uint8))


def payload(
    ratio: float,
    confidence: float | None,
    *,
    tactile_pixels: int = 1000,
    damage_pixels: int | None = None,
) -> dict:
    if ratio < 0.005:
        severity = "normal"
    elif ratio < 0.05:
        severity = "minor"
    elif ratio < 0.15:
        severity = "moderate"
    else:
        severity = "severe"
    resolved_damage_pixels = (
        round(tactile_pixels * ratio) if damage_pixels is None else damage_pixels
    )
    return {
        "schema_version": "1.0",
        "model": {
            "name": "yolo26s-seg-server-v1",
            "weights": "yolo26s_seg_v1_best.pt",
            "classes": {"0": "tactile_block", "1": "damage"},
        },
        "regions": {
            "tactile_block": {
                "bbox_xyxy": [0, 0, 10, 10],
                "polygons": [],
                "confidence": 0.9,
                "pixels": tactile_pixels,
            },
            "damage": {
                "bbox_xyxy": [1, 1, 2, 2] if resolved_damage_pixels else None,
                "polygons": [],
                "confidence": confidence,
                "pixels": resolved_damage_pixels,
                "raw_pixels": resolved_damage_pixels,
                "outside_tactile_pixels": 0,
                "merged_instance_count": 1 if resolved_damage_pixels else 0,
            },
        },
        "analysis": {
            "damage_detected": resolved_damage_pixels > 0,
            "damage_ratio": ratio,
            "damage_ratio_percent": ratio * 100,
            "estimated_severity": severity,
            "estimated_severity_label": severity,
            "repair_priority": "inspection_required",
            "repair_priority_label": "보수 확인 필요",
            "review_required": True,
            "review_reasons": ["severity_boundary_ambiguous"],
            "advisory_only": True,
        },
        "quality": {
            "positive_damage_dice": 0.083,
            "damage_f2": 0.8261,
            "ratio_mae_pp": 6.085,
            "severity_macro_f1": 0.2187,
        },
    }


def v2_unknown_missing_payload() -> dict:
    reasons = [{"code": "MISSING_AREA_UNCERTAIN", "message": "결손 범위를 계산할 수 없습니다."}]
    return {
        "schema_version": "2.0",
        "model": {
            "name": "server-multiclass-v2",
            "task": "segmentation",
            "class_names": ["tactile_block", "missing", "crack", "wear"],
            "policy_version": "draft-1",
            "class_mapping_valid": True,
        },
        "regions": {
            "tactile_block": {"pixels": 1000, "polygons": []},
            "damage": {"pixels": 19_260, "polygons": []},
        },
        "units": [
            {
                "local_unit_id": "block_group_1",
                "damage_types": {
                    "missing": {"detected": True, "confidence": 0.82},
                    "crack": {"detected": False, "confidence": None},
                    "wear": {"detected": False, "confidence": None},
                }
            }
        ],
        "summary": {
            "damage_detected": True,
            "estimated_severity": None,
            "repair_priority": "inspection_required",
            "worst_unit_id": "block_group_1",
            "dominant_damage_type": "missing",
            "max_damage_ratio_percent": None,
            "review_required": True,
            "review_reasons": reasons,
            "advisory_only": True,
        },
        "analysis": {
            "damage_detected": True,
            "damage_ratio_percent": None,
            "estimated_severity": None,
            "review_required": True,
            "review_reasons": reasons,
            "advisory_only": True,
        },
        "quality": {},
    }


def v2_payload(
    dominant: str,
    ratio_percent: float,
    confidence: float = 0.82,
) -> dict:
    if ratio_percent < 0.5:
        severity = "normal"
    elif ratio_percent < 5.0:
        severity = "minor"
    elif ratio_percent < 15.0:
        severity = "moderate"
    else:
        severity = "severe"
    damage_types = {
        name: {
            "detected": name == dominant,
            "confidence": confidence if name == dominant else None,
            "ratio_percent": ratio_percent if name == dominant else 0.0,
            "pixels": max(1, round(ratio_percent * 10)) if name == dominant else 0,
        }
        for name in ("missing", "crack", "wear")
    }
    reasons = [{"code": "RATIO_NEAR_THRESHOLD", "message": "review"}]
    return {
        "schema_version": "2.0",
        "model": {
            "name": "yolo26s_seg_multiclass_v2_best",
            "task": "segmentation",
            "class_names": ["tactile_block", "missing", "crack", "wear"],
            "policy_version": "draft-1",
            "class_mapping_valid": True,
        },
        "regions": {
            "tactile_block": {"pixels": 1000, "polygons": []},
            "damage": {"pixels": max(1, round(ratio_percent * 10)), "polygons": []},
        },
        "units": [
            {
                "local_unit_id": "block_1",
                "damage_types": damage_types,
            }
        ],
        "summary": {
            "damage_detected": True,
            "estimated_severity": severity,
            "repair_priority": "inspection_required",
            "worst_unit_id": "block_1",
            "dominant_damage_type": dominant,
            "max_damage_ratio_percent": ratio_percent,
            "review_required": True,
            "review_reasons": reasons,
            "advisory_only": True,
        },
        "analysis": {
            "damage_detected": True,
            "damage_ratio_percent": ratio_percent,
            "estimated_severity": severity,
            "review_required": True,
            "review_reasons": reasons,
            "advisory_only": True,
        },
        "quality": {},
    }
