from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from ROADY_AI.server_damage_analysis.api import create_app
from ROADY_AI.server_damage_analysis.config import AppSettings
from ROADY_AI.tests.test_server_damage_service import (
    FakeAnalyzer,
    payload,
    v2_unknown_missing_payload,
)


def test_analyze_accepts_spring_multipart_contract(tmp_path: Path):
    app = build_test_app(
        tmp_path,
        [payload(0.0859, 0.84321, damage_pixels=19_260)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            data={
                "damageId": "42",
                "latitude": "",
                "longitude": "",
                "capturedAt": "",
            },
            files=[("images", ("damage.jpg", jpeg_bytes(), "application/octet-stream"))],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["damaged"] is True
    assert body["damage_score"] == 40
    assert body["damage_type"] is None
    assert body["repair_required"] is True
    assert body["repair_priority"] == "NORMAL"
    assert body["confidence_score"] == 0.8432
    assert body["analysis_detail"]["aggregation"]["image_count"] == 1


def test_health_and_model_info_are_ready(tmp_path: Path):
    app = build_test_app(tmp_path, [payload(0.0, None)])

    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").json() == {"status": "ready"}
        model_info = client.get("/model-info").json()

    assert model_info["name"] == "yolo26s-seg-server-v1"
    assert model_info["weights"] == "model.pt"
    assert model_info["weights_sha256"] == "a" * 64


def test_analyze_rejects_non_image_payload(tmp_path: Path):
    app = build_test_app(tmp_path, [payload(0.0, None)])

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            data={"damageId": "42"},
            files=[("images", ("not-image.txt", b"hello", "text/plain"))],
        )

    assert response.status_code == 415
    assert response.json()["detail"] == "Image index 0 cannot be decoded."


def test_analyze_rejects_invalid_damage_id(tmp_path: Path):
    app = build_test_app(tmp_path, [payload(0.0, None)])

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            data={"damageId": "invalid"},
            files=[("images", ("damage.jpg", jpeg_bytes(), "image/jpeg"))],
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "damageId must be a positive integer."


def test_analyze_scores_missing_from_pixels_when_ratio_is_unavailable(tmp_path: Path):
    app = build_test_app(tmp_path, [v2_unknown_missing_payload()])

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            data={"damageId": "42"},
            files=[("images", ("damage.jpg", jpeg_bytes(), "image/jpeg"))],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["damaged"] is True
    assert body["damage_score"] == 40
    assert body["damage_type"] == "LARGE_MISSING"
    assert body["repair_required"] is True
    assert body["repair_priority"] == "NORMAL"
    assert body["analysis_detail"]["damage_ratio"] is None
    assert body["analysis_detail"]["estimated_severity"] == "moderate"
    assert body["analysis_detail"]["summary"]["estimated_severity"] is None
    assert body["analysis_detail"]["summary"]["repair_priority"] == "inspection_required"


def test_analyze_passes_image_metadata_to_model(tmp_path: Path):
    analyzer = FakeAnalyzer([payload(0.0, None)])
    app = build_test_app(tmp_path, analyzer=analyzer)
    metadata = {
        "originalImage": "original.jpg",
        "roiSource": "tactile_block",
        "roiFallbackUsed": False,
        "frameQualityVerified": True,
        "edgeDamageCandidateDetected": True,
        "tactileDetectionCount": 2,
    }

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            data={
                "damageId": "42",
                "analysisMetadata": json.dumps(metadata),
            },
            files=[("images", ("roi.jpg", jpeg_bytes(), "image/jpeg"))],
        )

    assert response.status_code == 200
    assert analyzer.received_metadata == [
        {
            "original_image": "original.jpg",
            "analysis_roi": "roi.jpg",
            "roi_source": "tactile_block",
            "roi_fallback_used": False,
            "frame_quality_verified": True,
            "edge_damage_candidate_detected": True,
            "tactile_detection_count": 2,
        }
    ]


def test_analyze_rejects_invalid_analysis_metadata_json(tmp_path: Path):
    app = build_test_app(tmp_path, [payload(0.0, None)])

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            data={"damageId": "42", "analysisMetadata": "not-json"},
            files=[("images", ("roi.jpg", jpeg_bytes(), "image/jpeg"))],
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "analysisMetadata must be valid JSON."


def test_analyze_rejects_metadata_count_mismatch(tmp_path: Path):
    app = build_test_app(tmp_path, [payload(0.0, None), payload(0.0, None)])

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            data={"damageId": "42", "analysisMetadata": "{}"},
            files=[
                ("images", ("roi-1.jpg", jpeg_bytes(), "image/jpeg")),
                ("images", ("roi-2.jpg", jpeg_bytes(), "image/jpeg")),
            ],
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "analysisMetadata count must match images count."


def build_test_app(
    tmp_path: Path,
    payloads: list[dict] | None = None,
    *,
    analyzer: FakeAnalyzer | None = None,
):
    model_path = tmp_path / "model.pt"
    sha_path = tmp_path / "model.sha256"
    model_path.write_bytes(b"fake-model")
    sha_path.write_text(("a" * 64) + "  model.pt\n", encoding="utf-8")
    settings = AppSettings(
        model_path=model_path,
        model_sha256_path=sha_path,
        verify_model_hash=False,
        warmup_enabled=False,
    )
    selected_analyzer = analyzer or FakeAnalyzer(payloads or [])
    return create_app(settings, analyzer_factory=lambda _: selected_analyzer)


def jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((10, 10, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()
