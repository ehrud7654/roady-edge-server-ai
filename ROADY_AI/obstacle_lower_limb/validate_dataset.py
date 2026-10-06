from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_CLASS_IDS = {0, 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the ROADY lower-limb YOLO dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--decode-all", action="store_true")
    return parser.parse_args()


def validate_label(path: Path) -> tuple[int, list[str]]:
    box_count = 0
    errors: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"line {line_number}: invalid numeric value")
            continue
        if class_id not in VALID_CLASS_IDS:
            errors.append(f"line {line_number}: class must be 0 or 1, got {class_id}")
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            errors.append(f"line {line_number}: coordinate outside [0, 1]")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"line {line_number}: non-positive box")
        if x - width / 2 < -1e-6 or x + width / 2 > 1.0 + 1e-6:
            errors.append(f"line {line_number}: x bounds outside image")
        if y - height / 2 < -1e-6 or y + height / 2 > 1.0 + 1e-6:
            errors.append(f"line {line_number}: y bounds outside image")
        box_count += 1
    return box_count, errors


def main() -> None:
    args = parse_args()
    summary: dict[str, object] = {"dataset": str(args.dataset), "splits": {}, "errors": []}
    global_stems: dict[str, str] = {}

    for split in ("train", "val", "test"):
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            summary["errors"].append(f"{split}: missing images or labels directory")
            continue

        images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {p.stem: p for p in label_dir.glob("*.txt")}
        missing_labels = sorted(set(images) - set(labels))
        orphan_labels = sorted(set(labels) - set(images))
        counts = Counter(images=len(images), labels=len(labels), boxes=0, positive_images=0)

        for stem, image_path in images.items():
            if stem in global_stems:
                summary["errors"].append(
                    f"{split}/{stem}: duplicate stem also appears in {global_stems[stem]}"
                )
            global_stems[stem] = split
            if args.decode_all:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as error:
                    summary["errors"].append(f"{split}/{stem}: image decode failed: {error}")

        for stem, label_path in labels.items():
            box_count, label_errors = validate_label(label_path)
            counts["boxes"] += box_count
            counts["positive_images"] += int(box_count > 0)
            for error in label_errors:
                summary["errors"].append(f"{split}/{stem}: {error}")

        summary["splits"][split] = {
            **counts,
            "negative_images": len(images) - counts["positive_images"],
            "missing_labels": missing_labels[:50],
            "orphan_labels": orphan_labels[:50],
        }
        if not images:
            summary["errors"].append(f"{split}: dataset contains no images")
        if missing_labels:
            summary["errors"].append(f"{split}: {len(missing_labels)} images have no label file")
        if orphan_labels:
            summary["errors"].append(f"{split}: {len(orphan_labels)} labels have no image")

    summary["passed"] = len(summary["errors"]) == 0
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
