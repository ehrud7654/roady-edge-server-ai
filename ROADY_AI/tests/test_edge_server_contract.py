import json

import pytest

from ROADY_AI.server_damage_analysis.edge_event import load_edge_analysis_request


def write_event(tmp_path, *, images, ai_metadata=None):
    pending = tmp_path / "pending"
    image_dir = tmp_path / "images"
    pending.mkdir()
    image_dir.mkdir()
    for name in images:
        (image_dir / name).write_bytes(b"image")
    payload = {
        "eventId": "damage_001",
        "images": images,
        "metadata": {"ai": ai_metadata or {}},
    }
    path = pending / "damage_001.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_edge_event_uses_second_image_as_server_analysis_roi(tmp_path):
    event_path = write_event(
        tmp_path,
        images=["damage_001_01.jpg", "damage_001_02.jpg"],
        ai_metadata={
            "roi_source": "tactile_block",
            "roi_fallback_used": False,
            "frame_quality_verified": True,
            "analysis_unit_hint": "block_or_block_group",
            "tactile_detection_count": 2,
            "frame_selection_status": "quality_gate_passed",
        },
    )

    request = load_edge_analysis_request(event_path)

    assert request.original_image.name == "damage_001_01.jpg"
    assert request.analysis_roi.name == "damage_001_02.jpg"
    assert request.input_metadata == {
        "original_image": "damage_001_01.jpg",
        "analysis_roi": "damage_001_02.jpg",
        "roi_source": "tactile_block",
        "roi_fallback_used": False,
        "frame_quality_verified": True,
        "analysis_unit_hint": "block_or_block_group",
        "tactile_detection_count": 2,
        "frame_selection_status": "quality_gate_passed",
        "edge_damage_candidate_detected": True,
        "possible_obstruction": False,
    }


def test_edge_fallback_and_unverified_quality_propagate_to_server(tmp_path):
    event_path = write_event(
        tmp_path,
        images=["original.jpg", "fallback_roi.jpg"],
        ai_metadata={
            "roi_source": "damage_fallback",
            "roi_fallback_used": True,
            "frame_quality_verified": False,
        },
    )

    request = load_edge_analysis_request(event_path)

    assert request.input_metadata["roi_fallback_used"] is True
    assert request.input_metadata["frame_quality_verified"] is False
    assert request.input_metadata["edge_damage_candidate_detected"] is True


def test_event_without_analysis_roi_is_rejected_instead_of_using_original(tmp_path):
    event_path = write_event(tmp_path, images=["original.jpg"])

    with pytest.raises(ValueError, match="original frame and analysis ROI"):
        load_edge_analysis_request(event_path)


def test_event_rejects_image_path_traversal(tmp_path):
    pending = tmp_path / "pending"
    pending.mkdir()
    path = pending / "damage_001.json"
    path.write_text(
        json.dumps({"eventId": "damage_001", "images": ["original.jpg", "../roi.jpg"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="plain filenames"):
        load_edge_analysis_request(path)
