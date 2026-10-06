from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]


def _boolean_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppSettings:
    model_path: Path = WORKSPACE / "ROADY_AI/models/server/yolo26s_seg_multiclass_v4_best.pt"
    model_sha256_path: Path = (
        WORKSPACE / "ROADY_AI/models/server/yolo26s_seg_multiclass_v4_best.sha256"
    )
    device: str = "cpu"
    imgsz: int = 768
    max_images: int = 50
    max_image_bytes: int = 20 * 1024 * 1024
    max_request_bytes: int = 200 * 1024 * 1024
    inference_concurrency: int = 1
    score_minor_max_pixels: int = 10_000
    score_moderate_max_pixels: int = 50_000
    score_max_pixels: int = 300_000
    verify_model_hash: bool = True
    warmup_enabled: bool = True

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            model_path=Path(
                os.getenv(
                    "ROADY_AI_MODEL_PATH",
                    str(cls.model_path),
                )
            ),
            model_sha256_path=Path(
                os.getenv(
                    "ROADY_AI_MODEL_SHA256_PATH",
                    str(cls.model_sha256_path),
                )
            ),
            device=os.getenv("ROADY_AI_DEVICE", cls.device),
            imgsz=int(os.getenv("ROADY_AI_IMGSZ", cls.imgsz)),
            max_images=int(os.getenv("ROADY_AI_MAX_IMAGES", cls.max_images)),
            max_image_bytes=int(
                os.getenv("ROADY_AI_MAX_IMAGE_BYTES", cls.max_image_bytes)
            ),
            max_request_bytes=int(
                os.getenv("ROADY_AI_MAX_REQUEST_BYTES", cls.max_request_bytes)
            ),
            inference_concurrency=max(
                1,
                int(
                    os.getenv(
                        "ROADY_AI_INFERENCE_CONCURRENCY",
                        cls.inference_concurrency,
                    )
                ),
            ),
            score_minor_max_pixels=int(
                os.getenv(
                    "ROADY_AI_SCORE_MINOR_MAX_PIXELS",
                    cls.score_minor_max_pixels,
                )
            ),
            score_moderate_max_pixels=int(
                os.getenv(
                    "ROADY_AI_SCORE_MODERATE_MAX_PIXELS",
                    cls.score_moderate_max_pixels,
                )
            ),
            score_max_pixels=int(
                os.getenv("ROADY_AI_SCORE_MAX_PIXELS", cls.score_max_pixels)
            ),
            verify_model_hash=_boolean_env("ROADY_AI_VERIFY_MODEL_HASH", True),
            warmup_enabled=_boolean_env("ROADY_AI_WARMUP_ENABLED", True),
        )
