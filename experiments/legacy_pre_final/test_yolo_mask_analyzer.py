import unittest
from types import SimpleNamespace

import numpy as np

from ai.server.severity import ModelQuality, SeverityPolicy
from ai.server.yolo_mask_analyzer import analyze_yolo_segmentation


class TensorLike:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class YoloMaskAnalyzerTests(unittest.TestCase):
    def test_unions_yolo_masks_and_clips_damage_to_block(self):
        tactile = np.zeros((10, 10), dtype=np.uint8)
        tactile[1:9, 1:9] = 1
        damage = np.zeros((10, 10), dtype=np.uint8)
        damage[1:5, 1:5] = 1
        damage[0, 0] = 1  # outside the tactile block
        result = SimpleNamespace(
            orig_shape=(10, 10),
            boxes=SimpleNamespace(
                cls=TensorLike([0, 1]),
                conf=TensorLike([0.95, 0.85]),
            ),
            masks=SimpleNamespace(data=TensorLike([tactile, damage])),
        )

        output = analyze_yolo_segmentation(
            result,
            model_quality=ModelQuality(0.70, 0.85, 0.92),
            policy=SeverityPolicy(
                minimum_tactile_pixels=10,
                normal_boundary_margin=0,
                boundary_margin=0,
            ),
        )
        self.assertEqual(output.tactile_block_pixels, 64)
        self.assertEqual(output.damage_pixels_in_block, 16)
        self.assertEqual(output.damage_pixels_outside_block, 1)
        self.assertEqual(output.damage_ratio, 0.25)
        self.assertEqual(output.estimated_severity, "severe")


if __name__ == "__main__":
    unittest.main()
