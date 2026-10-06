from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
COLORS = {0: (50, 220, 80), 1: (255, 70, 70)}


def parse_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes = []
    errors = []
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
        if class_id not in {0, 1}:
            errors.append(f"line {line_number}: invalid class {class_id}")
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            errors.append(f"line {line_number}: coordinate outside [0, 1]")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"line {line_number}: non-positive box")
        if x - width / 2 < -1e-6 or x + width / 2 > 1.0 + 1e-6:
            errors.append(f"line {line_number}: x bounds outside image")
        if y - height / 2 < -1e-6 or y + height / 2 > 1.0 + 1e-6:
            errors.append(f"line {line_number}: y bounds outside image")
        boxes.append((class_id, x, y, width, height))
    return boxes, errors


def render_overlay(image_path: Path, boxes: list[tuple[int, float, float, float, float]], output: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for class_id, x, y, box_width, box_height in boxes:
        left = (x - box_width / 2) * width
        top = (y - box_height / 2) * height
        right = (x + box_width / 2) * width
        bottom = (y + box_height / 2) * height
        color = COLORS[class_id]
        draw.rectangle((left, top, right, bottom), outline=color, width=max(2, width // 500))
        draw.text((left + 3, max(0, top - 14)), "tactile" if class_id == 0 else "damage", fill=color)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((1280, 1280))
    image.save(output, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Detect 데이터셋의 파일·좌표·클래스를 검증한다.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overlays", type=Path, required=True)
    parser.add_argument("--overlay-per-split-class", type=int, default=12)
    parser.add_argument("--decode-all", action="store_true")
    args = parser.parse_args()

    summary = {"splits": {}, "errors": [], "decode_errors": [], "overlay_files": []}
    global_image_stems: dict[str, str] = {}
    for split in ("train", "val", "test"):
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        images = {path.stem: path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        counts = Counter()
        area_bins = Counter()
        candidates = {0: [], 1: []}
        for stem, label_path in labels.items():
            boxes, errors = parse_label(label_path)
            for error in errors:
                summary["errors"].append({"split": split, "sample_id": stem, "error": error})
            present = set()
            for class_id, _, _, width, height in boxes:
                counts[f"class_{class_id}_boxes"] += 1
                present.add(class_id)
                area = width * height
                area_bins["tiny_<0.1%" if area < 0.001 else "small_<1%" if area < 0.01 else "medium_<10%" if area < 0.1 else "large_>=10%"] += 1
            for class_id in present:
                counts[f"class_{class_id}_images"] += 1
                if len(candidates[class_id]) < args.overlay_per_split_class:
                    candidates[class_id].append((stem, boxes))

        for stem, image_path in images.items():
            previous = global_image_stems.get(stem)
            if previous:
                summary["errors"].append({"split": split, "sample_id": stem, "error": f"duplicate stem also in {previous}"})
            else:
                global_image_stems[stem] = split
            if args.decode_all:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as error:
                    summary["decode_errors"].append({"split": split, "sample_id": stem, "error": str(error)})

        rendered = set()
        for class_id, samples in candidates.items():
            for stem, boxes in samples:
                if stem in rendered:
                    continue
                output = args.overlays / split / f"{stem}.jpg"
                render_overlay(images[stem], boxes, output)
                summary["overlay_files"].append(str(output))
                rendered.add(stem)

        summary["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": len(missing_labels),
            "missing_images": len(missing_images),
            "counts": dict(sorted(counts.items())),
            "box_area_bins": dict(sorted(area_bins.items())),
            "missing_label_samples": missing_labels[:20],
            "missing_image_samples": missing_images[:20],
        }

    summary["passed_structural_validation"] = not summary["errors"] and not summary["decode_errors"] and all(
        item["missing_labels"] == 0 and item["missing_images"] == 0 for item in summary["splits"].values()
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": summary["passed_structural_validation"],
        "errors": len(summary["errors"]),
        "decode_errors": len(summary["decode_errors"]),
        "splits": summary["splits"],
        "overlays": len(summary["overlay_files"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
