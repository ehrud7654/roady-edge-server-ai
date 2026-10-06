from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from .analyzer import ServerDamageAnalyzer
from .edge_event import load_edge_analysis_request


def main() -> None:
    parser = argparse.ArgumentParser(description="ROADY 서버 파손 분석")
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--event-json", help="Edge pending event JSON path")
    parser.add_argument("--event-image-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--policy", default=None, help="Severity policy YAML path")
    parser.add_argument(
        "--input-metadata",
        default=None,
        help="Edge ROI and representative-frame metadata JSON",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    input_metadata = None
    image_path = args.image
    original_source = args.image
    if args.event_json:
        request = load_edge_analysis_request(
            args.event_json,
            image_dir=args.event_image_dir,
        )
        image_path = str(request.analysis_roi)
        original_source = str(request.original_image)
        input_metadata = request.input_metadata
    elif args.input_metadata:
        input_metadata = json.loads(Path(args.input_metadata).read_text(encoding="utf-8"))
    analyzer = ServerDamageAnalyzer(args.model, policy_path=args.policy)
    payload, overlay, damage_mask = analyzer.predict(
        image_path,
        imgsz=args.imgsz,
        device=args.device,
        input_metadata=input_metadata,
    )
    original_name = Path(original_source).name
    roi_name = Path(image_path).name
    overlay_name = f"{Path(image_path).stem}_analysis.jpg"
    mask_name = f"{Path(image_path).stem}_damage_mask.png"
    payload["artifacts"].update({
        "original_image": original_name,
        "analysis_roi": roi_name,
        "overlay_image": overlay_name,
        "mask_image": mask_name,
    })
    shutil.copy2(original_source, output / original_name)
    if Path(image_path).resolve() != Path(original_source).resolve():
        shutil.copy2(image_path, output / roi_name)
    cv2.imwrite(str(output / overlay_name), overlay)
    cv2.imwrite(str(output / mask_name), damage_mask.astype("uint8") * 255)
    (output / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output / "analysis.json")


if __name__ == "__main__":
    main()
