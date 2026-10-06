from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ROADY_AI.server_damage_analysis.analyzer import ServerDamageAnalyzer
from ROADY_AI.server_damage_analysis.expected_region import (
    estimate_expected_region,
    estimate_missing_expected_region,
)
from ROADY_AI.server_damage_analysis.policy import (
    ModelQuality,
    evaluate_unit_severity,
    load_severity_policy,
)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def analyzer():
    instance = ServerDamageAnalyzer.__new__(ServerDamageAnalyzer)
    instance.model_path = Path("fake_multiclass.pt")
    instance.policy = load_severity_policy()
    instance.quality = ModelQuality(
        positive_damage_dice=0.9,
        damage_f2=0.9,
        ratio_mae_pp=1.0,
        severity_macro_f1=0.9,
    )
    instance.class_names = {0: "tactile_block", 1: "missing", 2: "crack", 3: "wear"}
    instance.class_mapping_valid = True
    instance.class_ids = {value: key for key, value in instance.class_names.items()}
    return instance


def fake_result(instances, shape=(100, 120)):
    classes, confidences, masks = zip(*instances) if instances else ([], [], [])
    return SimpleNamespace(
        orig_shape=shape,
        orig_img=np.zeros((*shape, 3), dtype=np.uint8),
        boxes=SimpleNamespace(cls=FakeTensor(classes), conf=FakeTensor(confidences)),
        masks=SimpleNamespace(data=FakeTensor(masks)),
    )


def rectangle(shape, x1, y1, x2, y2):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def test_crack_and_wear_are_clipped_to_tactile_unit():
    shape = (100, 120)
    tactile = rectangle(shape, 20, 20, 60, 60)
    crack = rectangle(shape, 40, 40, 80, 80)
    wear = rectangle(shape, 0, 0, 10, 10)
    result = fake_result([(0, 0.9, tactile), (2, 0.9, crack), (3, 0.9, wear)], shape)

    payload, _, _ = analyzer().analyze_result(
        result, input_metadata={"frame_quality_verified": True, "roi_source": "tactile_block"}
    )
    unit = payload["units"][0]

    assert unit["damage_types"]["crack"]["pixels"] == 400
    assert unit["damage_types"]["wear"]["pixels"] == 0
    assert unit["damage_types"]["wear"]["ratio_percent"] == 0.0


def test_model_quality_metrics_are_informational_and_do_not_require_review():
    instance = analyzer()
    instance.quality = ModelQuality()
    shape = (100, 120)
    tactile = rectangle(shape, 20, 20, 60, 60)
    result = fake_result([(0, 0.9, tactile)], shape)

    payload, _, _ = instance.analyze_result(
        result, input_metadata={"frame_quality_verified": True, "roi_source": "tactile_block"}
    )

    assert payload["summary"]["review_required"] is False
    assert payload["summary"]["review_reasons"] == []
    assert payload["quality"]["positive_damage_dice"] == ModelQuality().positive_damage_dice


def test_missing_is_not_deleted_by_tactile_intersection_when_expected_region_is_valid():
    shape = (100, 120)
    tactile_masks = [rectangle(shape, x, 20, x + 20, 40) for x in (10, 40, 70)]
    tactile_masks[0][25:35, 15:25] = 0
    missing = rectangle(shape, 15, 25, 25, 35)
    result = fake_result(
        [(0, 0.9, mask) for mask in tactile_masks] + [(1, 0.9, missing)], shape
    )

    payload, _, _ = analyzer().analyze_result(
        result, input_metadata={"frame_quality_verified": True, "roi_source": "tactile_block"}
    )
    first = payload["units"][0]["damage_types"]["missing"]

    assert first["detected"]
    assert first["pixels"] == 100
    assert first["ratio_percent"] == 25.0
    assert first["ratio_status"] == "estimated"
    assert payload["units"][0]["analysis"]["total_damage_ratio_percent"] == 25.0


def test_missing_ratio_is_null_when_expected_region_is_unavailable():
    shape = (100, 120)
    tactile = rectangle(shape, 10, 20, 30, 40)
    tactile[25:35, 15:25] = 0
    missing = rectangle(shape, 15, 25, 25, 35)
    result = fake_result([(0, 0.9, tactile), (1, 0.9, missing)], shape)

    payload, _, _ = analyzer().analyze_result(
        result, input_metadata={"frame_quality_verified": True, "roi_source": "tactile_block"}
    )
    unit = payload["units"][0]

    assert unit["damage_types"]["missing"]["ratio_percent"] is None
    assert unit["damage_types"]["missing"]["ratio_status"] == "not_estimable"
    assert unit["analysis"]["total_damage_ratio_percent"] is None
    assert unit["analysis"]["estimated_severity"] == "moderate"
    assert unit["analysis"]["repair_priority"] == "inspection_required"
    assert unit["analysis"]["review_required"]
    assert "RATIO_NEAR_THRESHOLD" not in {
        reason["code"] for reason in unit["analysis"]["review_reasons"]
    }


def test_wide_tactile_instance_remains_block_group_and_requires_review():
    shape = (100, 120)
    tactile = rectangle(shape, 10, 20, 100, 40)
    result = fake_result([(0, 0.9, tactile)], shape)

    payload, _, _ = analyzer().analyze_result(
        result, input_metadata={"frame_quality_verified": True, "roi_source": "tactile_block"}
    )
    unit = payload["units"][0]

    assert unit["analysis_unit"] == "block_group"
    assert unit["analysis"]["review_required"]
    assert "BLOCK_INSTANCE_UNRESOLVED" in {
        reason["code"] for reason in unit["analysis"]["review_reasons"]
    }


def test_expected_region_rejects_unstable_neighbor_sizes():
    result = estimate_expected_region(
        unit_bbox=(0, 0, 20, 20),
        neighbor_bboxes=[(0, 0, 20, 20), (30, 0, 70, 20), (80, 0, 100, 60)],
        shape=(100, 120),
    )

    assert result.mask is None
    assert result.status == "unavailable"


def test_complete_missing_region_uses_stable_neighbor_scale_and_spacing():
    shape = (80, 150)
    # The predicted missing mask is deliberately smaller than a neighboring
    # tactile block. This proves the denominator size comes from the stable
    # 20x20 neighbor median rather than from the 10x10 damage mask bbox.
    missing = rectangle(shape, 105, 25, 115, 35)
    result = estimate_missing_expected_region(
        missing_mask=missing.astype(bool),
        neighbor_bboxes=[(10, 20, 30, 40), (40, 20, 60, 40), (70, 20, 90, 40)],
        shape=shape,
    )

    assert result.status == "estimated"
    assert result.mask is not None
    assert int(result.mask.sum()) == 400
    assert int(missing.sum()) == 100


def test_complete_missing_is_not_estimable_with_unstable_neighbor_scale():
    shape = (100, 150)
    missing = rectangle(shape, 100, 20, 120, 40)
    result = estimate_missing_expected_region(
        missing_mask=missing.astype(bool),
        neighbor_bboxes=[(10, 20, 30, 40), (40, 20, 80, 40), (90, 20, 110, 80)],
        shape=shape,
    )

    assert result.status == "unavailable"
    assert result.mask is None


def test_type_minimum_severity_and_wear_can_reach_severe():
    policy = load_severity_policy()
    missing = evaluate_unit_severity(
        {
            "missing": {"detected": True, "ratio_percent": 1.0},
            "crack": {"detected": False, "ratio_percent": 0.0},
            "wear": {"detected": False, "ratio_percent": 0.0},
        },
        policy,
    )
    wear = evaluate_unit_severity(
        {
            "missing": {"detected": False, "ratio_percent": 0.0},
            "crack": {"detected": False, "ratio_percent": 0.0},
            "wear": {"detected": True, "ratio_percent": 20.0},
        },
        policy,
    )

    assert missing.severity == "moderate"
    assert wear.severity == "severe"
    assert wear.reason["dominant_damage_type"] == "wear"
    assert wear.reason["policy_version"] == "draft-1"


def test_summary_uses_worst_unit_and_excludes_not_estimable_from_mean():
    instance = analyzer()
    estimable = {
        "local_unit_id": "block_1",
        "tactile": {"pixels": 100},
        "polygon": [],
        "analysis": {
            "damage_detected": True,
            "total_damage_ratio_percent": 2.0,
            "ratio_status": "estimated",
            "estimated_severity": "minor",
            "repair_priority": "monitor",
            "repair_priority_label": "monitor",
            "severity_reason": {"dominant_damage_type": "crack"},
            "review_required": False,
            "review_reasons": [],
        },
    }
    unknown = {
        "local_unit_id": "block_group_2",
        "tactile": {"pixels": 0},
        "polygon": [],
        "analysis": {
            "damage_detected": True,
            "total_damage_ratio_percent": None,
            "ratio_status": "not_estimable",
            "estimated_severity": "moderate",
            "repair_priority": "inspection_required",
            "repair_priority_label": "inspection",
            "severity_reason": {"dominant_damage_type": "missing"},
            "review_required": True,
            "review_reasons": [{"code": "MISSING_AREA_UNCERTAIN", "message": "uncertain"}],
        },
    }

    payload = instance._payload(
        [estimable, unknown],
        instance._input_payload({"frame_quality_verified": True}),
        np.zeros((10, 10), dtype=bool),
    )

    assert payload["summary"]["worst_unit_id"] == "block_group_2"
    assert payload["summary"]["estimated_severity"] is None
    assert payload["summary"]["repair_priority"] == "inspection_required"
    assert payload["summary"]["ratio_status"] == "not_estimable"
    assert payload["summary"]["review_required"] is True
    assert payload["summary"]["mean_damage_ratio_percent"] == 2.0
    assert payload["summary"]["not_estimable_unit_count"] == 1
    assert payload["summary"]["review_required"]


def test_invalid_model_class_mapping_returns_review_instead_of_analysis():
    instance = analyzer()
    instance.class_names = {0: "tactile_block", 1: "damage"}
    instance.class_mapping_valid = False
    result = fake_result([], shape=(20, 20))

    payload, _, _ = instance.analyze_result(result)

    assert payload["model"]["class_mapping_valid"] is False
    assert payload["summary"]["review_required"] is True
    assert payload["summary"]["max_damage_ratio_percent"] is None
    assert "MODEL_CLASS_MAPPING_INVALID" in {
        reason["code"] for reason in payload["summary"]["review_reasons"]
    }


def test_edge_candidate_without_server_damage_requires_review():
    instance = analyzer()

    payload = instance._payload(
        [],
        instance._input_payload(
            {
                "frame_quality_verified": True,
                "edge_damage_candidate_detected": True,
            }
        ),
        np.zeros((10, 10), dtype=bool),
    )

    assert payload["summary"]["damage_detected"] is False
    assert payload["summary"]["review_required"] is True
    assert "EDGE_SERVER_DISAGREEMENT" in {
        reason["code"] for reason in payload["summary"]["review_reasons"]
    }


def test_obstruction_suspected_never_confirms_ratio_or_severity():
    shape = (100, 120)
    tactile = rectangle(shape, 20, 20, 80, 80)
    wear = rectangle(shape, 30, 30, 50, 50)
    result = fake_result([(0, 0.9, tactile), (3, 0.95, wear)], shape)

    payload, _, _ = analyzer().analyze_result(
        result,
        input_metadata={
            "frame_quality_verified": True,
            "roi_source": "tactile_block",
            "edge_damage_candidate_detected": True,
            "possible_obstruction": True,
            "obstruction_source": "edge_ood_gate",
        },
    )

    unit = payload["units"][0]
    assert unit["damage_types"]["wear"]["detected"] is True
    assert unit["analysis"]["damage_detected"] is True
    assert unit["analysis"]["ratio_status"] == "not_estimable"
    assert unit["analysis"]["total_damage_ratio_percent"] is None
    assert unit["analysis"]["estimated_severity"] is None
    assert payload["summary"]["max_damage_ratio_percent"] is None
    assert payload["summary"]["repair_priority"] == "inspection_required"
    assert "OBSTRUCTION_SUSPECTED" in {
        reason["code"] for reason in payload["summary"]["review_reasons"]
    }


def test_obstruction_without_damage_returns_unknown_not_clean():
    shape = (100, 120)
    tactile = rectangle(shape, 20, 20, 80, 80)
    result = fake_result([(0, 0.9, tactile)], shape)

    payload, _, _ = analyzer().analyze_result(
        result,
        input_metadata={
            "frame_quality_verified": True,
            "possible_obstruction": True,
        },
    )

    assert payload["summary"]["damage_detected"] is None
    assert payload["summary"]["max_damage_ratio_percent"] is None
    assert payload["summary"]["estimated_severity"] is None
    assert payload["summary"]["review_required"] is True


def test_yaml_policy_change_is_applied_without_code_change(tmp_path):
    source = Path(__file__).parents[1] / "server_damage_analysis" / "severity_policy.yaml"
    content = source.read_text(encoding="utf-8").replace(
        "moderate_max_percent: 15.0", "moderate_max_percent: 25.0"
    )
    custom = tmp_path / "custom_policy.yaml"
    custom.write_text(content, encoding="utf-8")
    policy = load_severity_policy(custom)

    decision = evaluate_unit_severity(
        {
            "missing": {"detected": False, "ratio_percent": 0.0},
            "crack": {"detected": True, "ratio_percent": 20.0},
            "wear": {"detected": False, "ratio_percent": 0.0},
        },
        policy,
    )

    assert decision.severity == "moderate"


def analyzer_5class():
    instance = ServerDamageAnalyzer.__new__(ServerDamageAnalyzer)
    instance.model_path = Path("fake_obstruction.pt")
    instance.policy = load_severity_policy()
    instance.quality = ModelQuality(
        positive_damage_dice=0.9,
        damage_f2=0.9,
        ratio_mae_pp=1.0,
        severity_macro_f1=0.9,
    )
    instance.class_names = {
        0: "tactile_block", 1: "missing", 2: "crack", 3: "wear", 4: "obstruction"
    }
    instance.class_mapping_valid = True
    instance.class_ids = {value: key for key, value in instance.class_names.items()}
    return instance


def test_obstruction_class_mask_routes_to_review_without_metadata_flag():
    # 5-class model detects an obstruction mask covering the ROI. Even though it
    # ALSO spuriously predicts missing at high confidence (the observed 86% failure),
    # the block must go to 판단 보류 - never auto-confirmed as damage. No metadata
    # flag is set here: the routing must be driven by the model's obstruction class.
    shape = (100, 120)
    tactile = rectangle(shape, 20, 20, 60, 60)
    missing = rectangle(shape, 25, 25, 55, 55)          # scooter mistaken for missing
    obstruction = rectangle(shape, 10, 10, 100, 90)     # ~60% of ROI
    result = fake_result(
        [(0, 0.9, tactile), (1, 0.95, missing), (4, 0.92, obstruction)], shape
    )

    payload, _, _ = analyzer_5class().analyze_result(
        result,
        input_metadata={"frame_quality_verified": True, "roi_source": "tactile_block"},
    )
    summary = payload["summary"]

    assert payload["input"]["possible_obstruction"] is True
    assert payload["input"]["obstruction_source"] == "server_model"
    assert summary["review_required"] is True
    assert summary["estimated_severity"] is None
    assert summary["max_damage_ratio_percent"] is None
    assert "OBSTRUCTION_SUSPECTED" in {r["code"] for r in summary["review_reasons"]}
    for unit in payload["units"]:
        assert unit["analysis"]["ratio_status"] == "not_estimable"
        assert unit["analysis"]["estimated_severity"] is None


def test_tiny_obstruction_blip_below_coverage_does_not_trigger_review():
    # An obstruction mask below obstruction_coverage_min must NOT flip a genuine
    # damage analysis into 판단 보류.
    shape = (100, 120)
    tactile = rectangle(shape, 20, 20, 60, 60)
    crack = rectangle(shape, 30, 30, 50, 50)
    tiny = rectangle(shape, 0, 0, 4, 4)                 # ~0.13% of ROI
    result = fake_result([(0, 0.9, tactile), (2, 0.9, crack), (4, 0.9, tiny)], shape)

    payload, _, _ = analyzer_5class().analyze_result(
        result,
        input_metadata={"frame_quality_verified": True, "roi_source": "tactile_block"},
    )

    assert payload["input"]["possible_obstruction"] is False
    assert "OBSTRUCTION_SUSPECTED" not in {
        r["code"] for r in payload["summary"]["review_reasons"]
    }
