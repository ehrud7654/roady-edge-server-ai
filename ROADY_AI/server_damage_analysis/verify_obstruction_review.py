"""Verify obstruction -> 판단 보류(OBSTRUCTION_SUSPECTED) behavior of a 5-class model.

Runs ServerDamageAnalyzer on a set of 씽씽이(obstruction) ROI images and reports:
  * how many route to OBSTRUCTION_SUSPECTED / review_required (the desired behavior)
  * how many the pipeline would AUTO-CONFIRM as damage (the failure mode we are
    eliminating - the "86%" that the 4-class model produced)

This is the real check for the obstruction pilot: target is
    OBSTRUCTION_SUSPECTED ~ 100%   and   auto-confirmed damage = 0%.

Note: obstruction handling is analyzer/policy logic, not a YOLO metric, so this is
run through the analyzer rather than `yolo val` / `evaluate_policy` (4-class only).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .analyzer import ServerDamageAnalyzer

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _label_has_class(label_path: Path, class_id: int) -> bool:
    if not label_path.is_file():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0].isdigit() and int(parts[0]) == class_id:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="directory of ROI images")
    parser.add_argument("--labels", type=Path, default=None,
                        help="optional label dir; if given, only images whose label "
                             "contains the obstruction class are evaluated")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    analyzer = ServerDamageAnalyzer(args.model)
    obstruction_id = next(
        (key for key, value in analyzer.class_names.items() if value == "obstruction"), None
    )
    print("class_names:", analyzer.class_names)
    print("class_mapping_valid:", analyzer.class_mapping_valid, "| obstruction_class_id:", obstruction_id)
    if obstruction_id is None:
        print("WARNING: model has no 'obstruction' class - this is not a 5-class model.")

    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.labels is not None and obstruction_id is not None:
        # labels may live in split subdirs (labels/train, labels/val, ...) -> index by stem
        label_index = {p.stem: p for p in args.labels.rglob("*.txt")}
        images = [p for p in images if _label_has_class(label_index.get(p.stem, Path("/nonexistent")), obstruction_id)]

    tally: Counter = Counter()
    rows: list[dict] = []
    for path in images:
        payload, _, _ = analyzer.predict(
            str(path),
            imgsz=args.imgsz,
            device=args.device,
            input_metadata={
                "roi_source": "tactile_block",
                "frame_quality_verified": True,
                "edge_damage_candidate_detected": True,
            },
        )
        summary = payload["summary"]
        codes = {reason["code"] for reason in summary["review_reasons"]}
        obstruction_flagged = bool(payload["input"].get("possible_obstruction"))
        obstruction_suspected = "OBSTRUCTION_SUSPECTED" in codes
        # FAILURE mode: pipeline confirms damage on a scooter without deferring.
        auto_confirmed = bool(summary.get("damage_detected")) and not summary["review_required"]

        tally["total"] += 1
        tally["obstruction_flagged"] += int(obstruction_flagged)
        tally["obstruction_suspected"] += int(obstruction_suspected)
        tally["review_required"] += int(summary["review_required"])
        tally["auto_confirmed_damage"] += int(auto_confirmed)
        rows.append(
            {
                "image": path.name,
                "obstruction_flagged": obstruction_flagged,
                "obstruction_source": payload["input"].get("obstruction_source"),
                "review_required": summary["review_required"],
                "obstruction_suspected": obstruction_suspected,
                "damage_detected": summary.get("damage_detected"),
                "estimated_severity": summary.get("estimated_severity"),
                "review_codes": ",".join(sorted(codes)),
            }
        )

    n = max(1, tally["total"])
    print(f"\n=== obstruction verification on {tally['total']} images ===")
    print(f"OBSTRUCTION_SUSPECTED (판단보류) : {tally['obstruction_suspected']:4d}  ({tally['obstruction_suspected']/n*100:5.1f}%)")
    print(f"review_required                 : {tally['review_required']:4d}  ({tally['review_required']/n*100:5.1f}%)")
    print(f"obstruction flagged by model    : {tally['obstruction_flagged']:4d}  ({tally['obstruction_flagged']/n*100:5.1f}%)")
    print(f"AUTO-CONFIRMED as damage (FAIL)  : {tally['auto_confirmed_damage']:4d}  ({tally['auto_confirmed_damage']/n*100:5.1f}%)   <- target 0%")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"tally": dict(tally), "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("saved:", args.output)


if __name__ == "__main__":
    main()
