from ROADY_AI.server_damage_analysis.policy import (
    AnalysisPolicy,
    ModelQuality,
    classify_severity,
    decide_review,
)


def test_severity_boundaries():
    policy = AnalysisPolicy()
    assert classify_severity(0.004, policy) == "normal"
    assert classify_severity(0.01, policy) == "minor"
    assert classify_severity(0.10, policy) == "moderate"
    assert classify_severity(0.20, policy) == "severe"


def test_v1_quality_metrics_do_not_require_review():
    reasons = decide_review(
        ratio=0.10, tactile_pixels=1000, tactile_confidence=0.9,
        damage_confidence=0.9, image_quality_ok=True,
        quality=ModelQuality(), policy=AnalysisPolicy(),
    )
    assert reasons == []
