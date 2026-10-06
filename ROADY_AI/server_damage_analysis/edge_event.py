from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EdgeAnalysisRequest:
    event_id: str
    original_image: Path
    analysis_roi: Path
    input_metadata: dict[str, Any]


def load_edge_analysis_request(
    event_json: str | Path,
    *,
    image_dir: str | Path | None = None,
) -> EdgeAnalysisRequest:
    """Convert the persisted Edge event contract into a Server AI request.

    Edge stores images in this stable order: original frame, analysis ROI. The
    adapter validates the contract instead of silently analyzing the original
    frame when an ROI is missing.
    """
    event_path = Path(event_json)
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list) or len(images) < 2:
        raise ValueError("Edge event must contain original frame and analysis ROI")
    if any(not isinstance(name, str) or Path(name).name != name for name in images[:2]):
        raise ValueError("Edge event image names must be plain filenames")

    root = Path(image_dir) if image_dir is not None else event_path.parent.parent / "images"
    original_path = root / images[0]
    roi_path = root / images[1]
    if not original_path.is_file():
        raise FileNotFoundError(f"original image does not exist: {original_path}")
    if not roi_path.is_file():
        raise FileNotFoundError(f"analysis ROI does not exist: {roi_path}")

    metadata = payload.get("metadata") or {}
    ai_metadata = metadata.get("ai") if isinstance(metadata, dict) else None
    if not isinstance(ai_metadata, dict):
        ai_metadata = {}
    input_metadata = {
        "original_image": original_path.name,
        "analysis_roi": roi_path.name,
        "roi_source": ai_metadata.get("roi_source", "unknown"),
        "roi_fallback_used": bool(ai_metadata.get("roi_fallback_used", False)),
        "frame_quality_verified": bool(ai_metadata.get("frame_quality_verified", False)),
        # A persisted damage event means Edge observed a candidate. Keep this
        # true for older event JSON that predates the explicit metadata field.
        "edge_damage_candidate_detected": bool(
            ai_metadata.get("edge_damage_candidate_detected", True)
        ),
        # This flag must come from an upstream obstruction/OOD gate. The
        # four-class segmentation model does not contain an obstruction class.
        "possible_obstruction": bool(ai_metadata.get("possible_obstruction", False)),
    }
    for key in (
        "analysis_unit_hint",
        "tactile_detection_count",
        "frame_selection_status",
        "obstruction_source",
    ):
        if key in ai_metadata:
            input_metadata[key] = ai_metadata[key]

    return EdgeAnalysisRequest(
        event_id=str(payload.get("eventId", event_path.stem)),
        original_image=original_path,
        analysis_roi=roi_path,
        input_metadata=input_metadata,
    )
