import unittest

from ai.server.severity import (
    ModelQuality,
    RepairContext,
    SeverityPolicy,
    assess_severity,
)


POLICY = SeverityPolicy(
    minimum_tactile_pixels=10,
    normal_boundary_margin=0.0,
    boundary_margin=0.0,
)
QUALIFIED = ModelQuality(
    damage_mask_map50=0.70,
    severity_macro_f1=0.85,
    severe_recall=0.92,
)


def assess(damage_count: int, tactile_count: int = 100, **kwargs):
    tactile = [1] * tactile_count
    damage = [1] * damage_count + [0] * (tactile_count - damage_count)
    return assess_severity(
        tactile,
        damage,
        tactile_confidence=0.9,
        damage_confidence=0.9 if damage_count else None,
        model_quality=QUALIFIED,
        policy=POLICY,
        **kwargs,
    )


class SeverityTests(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(assess(0).severity, "normal")

    def test_minor(self):
        result = assess(2)
        self.assertEqual(result.severity, "minor")
        self.assertEqual(result.repair_need, "monitor")

    def test_moderate(self):
        result = assess(10)
        self.assertEqual(result.severity, "moderate")
        self.assertEqual(result.repair_need, "inspection_required")

    def test_severe(self):
        result = assess(20)
        self.assertEqual(result.severity, "severe")
        self.assertEqual(result.repair_need, "repair_candidate")
        self.assertEqual(result.priority_level, "urgent")

    def test_boundary_requires_review(self):
        policy = SeverityPolicy(minimum_tactile_pixels=10)
        result = assess_severity(
            [1] * 1000,
            [1] * 50 + [0] * 950,
            tactile_confidence=0.9,
            damage_confidence=0.9,
            model_quality=QUALIFIED,
            policy=policy,
        )
        self.assertEqual(result.severity, "review_required")
        self.assertEqual(result.estimated_severity, "moderate")
        self.assertIn("severity_boundary_ambiguous", result.review_reasons)

    def test_model_quality_gate_preserves_reference_estimate(self):
        result = assess_severity(
            [1] * 100,
            [1] * 20 + [0] * 80,
            tactile_confidence=0.9,
            damage_confidence=0.9,
            model_quality=ModelQuality(0.496, 0.581, 0.4389),
            policy=POLICY,
        )
        self.assertEqual(result.severity, "review_required")
        self.assertEqual(result.estimated_severity, "severe")
        self.assertIn("mask_quality_below_target", result.review_reasons)
        self.assertIn("severity_macro_f1_below_target", result.review_reasons)
        self.assertIn("severe_recall_below_target", result.review_reasons)

    def test_damage_outside_tactile_block_is_excluded(self):
        result = assess_severity(
            [1] * 100 + [0] * 100,
            [0] * 100 + [1] * 100,
            tactile_confidence=0.9,
            damage_confidence=0.9,
            model_quality=QUALIFIED,
            policy=POLICY,
        )
        self.assertEqual(result.damage_ratio, 0)
        self.assertEqual(result.damage_pixels_outside_block, 100)
        self.assertEqual(result.severity, "normal")

    def test_known_block_area_calculates_cm2(self):
        result = assess(20, known_tactile_area_cm2=900.0)
        self.assertEqual(result.damage_area_cm2, 180.0)
        self.assertEqual(result.tactile_block_area_cm2, 900.0)
        self.assertEqual(result.area_calculation_method, "known_tactile_area_ratio")

    def test_priority_context_is_transparent_and_bounded(self):
        result = assess(
            10,
            context=RepairContext(
                damage_type="missing",
                location_criticality=1.0,
                pedestrian_usage=1.0,
                recurrence_count=5,
            ),
        )
        self.assertEqual(result.priority_score, 85)
        self.assertEqual(result.priority_level, "urgent")

    def test_to_dict_contains_backward_compatible_fields(self):
        payload = assess(2).to_dict()
        self.assertEqual(payload["repair_priority"], "경과 관찰")
        self.assertIn("review_reason", payload)


if __name__ == "__main__":
    unittest.main()
