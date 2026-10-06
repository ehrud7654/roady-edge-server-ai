from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageOps


WORKSPACE = Path(__file__).resolve().parents[2]
TACTILE_NAMES = {"점자블록", "점자블럭"}


def bbox(points: list[tuple[float, float]], width: float, height: float) -> tuple[float, float, float, float] | None:
    if len(points) < 3:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, right = max(0.0, min(xs)), min(width, max(xs))
    top, bottom = max(0.0, min(ys)), min(height, max(ys))
    if right <= left or bottom <= top:
        return None
    return (
        ((left + right) / 2) / width,
        ((top + bottom) / 2) / height,
        (right - left) / width,
        (bottom - top) / height,
    )


def yolo_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    return str(class_id) + " " + " ".join(f"{min(1.0, max(0.0, value)):.6f}" for value in box)


def road_boxes(obj: dict, width: int, height: int) -> list[str]:
    lines = []
    for annotation in obj.get("annotations") or []:
        if annotation.get("label_name") not in TACTILE_NAMES or annotation.get("annotation_type") != "polygon":
            continue
        try:
            points = [(float(x), float(y)) for x, y in (annotation.get("annotation_info") or [])]
        except (TypeError, ValueError):
            continue
        box = bbox(points, width, height)
        if box is None:
            continue
        class_id = 1 if annotation.get("is_defect") == "불량부분" else 0
        lines.append(yolo_line(class_id, box))
    return lines


def surface_boxes(root: ET.Element, image_name: str, width: int, height: int) -> list[str]:
    image = next((item for item in root.findall("image") if item.get("name") == image_name), None)
    if image is None:
        return []
    lines = []
    for polygon in image.findall("polygon"):
        if polygon.get("label") != "braille_guide_blocks":
            continue
        try:
            points = [tuple(map(float, pair.split(","))) for pair in polygon.get("points", "").split(";")]
        except (TypeError, ValueError):
            continue
        box = bbox(points, width, height)
        if box is None:
            continue
        state = "unknown"
        for attribute in polygon.findall("attribute"):
            if attribute.get("name") == "attribute":
                state = (attribute.text or "unknown").strip()
        # 손상된 점자블록도 점자블록 클래스에 포함하고, 같은 영역을
        # 파손 후보 클래스로 한 번 더 제공한다.
        lines.append(yolo_line(0, box))
        if state == "damaged":
            lines.append(yolo_line(1, box))
    return lines


def select_rows(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return rows
    selected = []
    per_split = Counter()
    for row in rows:
        if per_split[row["split"]] >= limit:
            continue
        selected.append(row)
        per_split[row["split"]] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="통합 원본에서 Edge YOLO Detect 데이터셋을 생성한다.")
    parser.add_argument("--manifest", type=Path, default=WORKSPACE / "ROADY_AI/datasets/tactile_combined_v1.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--include-unreviewed-damage", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"출력 폴더가 이미 존재함: {args.output}")

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    rows = select_rows(rows, args.limit_per_split)
    archives: dict[str, zipfile.ZipFile] = {}
    xml_cache: dict[tuple[str, str], ET.Element] = {}
    counts = Counter()
    records = []
    try:
        for index, row in enumerate(rows, 1):
            split = row["split"]
            image_path = args.output / "images" / split / f"{row['sample_id']}.jpg"
            label_file = args.output / "labels" / split / f"{row['sample_id']}.txt"
            if args.resume and image_path.exists() and label_file.exists():
                lines = [line for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                class_counts = Counter(int(line.split()[0]) for line in lines)
                counts[f"{split}/images"] += 1
                counts[f"{split}/tactile_block"] += class_counts[0]
                counts[f"{split}/damage_candidate"] += class_counts[1]
                records.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": split,
                        "source_dataset": row["source_dataset"],
                        "state_original": row["state_original"],
                        "tactile_boxes": class_counts[0],
                        "damage_boxes": class_counts[1],
                        "provisional_damage": row["mvp_label"] == "review",
                    }
                )
                continue
            source_path = row["source_archive"]
            label_path = row["label_archive"]
            if source_path not in archives:
                archives[source_path] = zipfile.ZipFile(source_path)
            if label_path not in archives:
                archives[label_path] = zipfile.ZipFile(label_path)
            source_archive = archives[source_path]
            label_archive = archives[label_path]

            with source_archive.open(row["source_member"]) as stream:
                image = ImageOps.exif_transpose(Image.open(io.BytesIO(stream.read()))).convert("RGB")
            width, height = image.size
            if row["source_dataset"] == "road_facility_aihub_513":
                obj = json.loads(label_archive.read(row["label_member"]).decode("utf-8-sig"))
                lines = road_boxes(obj, width, height)
            else:
                key = (label_path, row["label_member"])
                if key not in xml_cache:
                    xml_cache[key] = ET.fromstring(label_archive.read(row["label_member"]))
                lines = surface_boxes(xml_cache[key], row["label_image_name"], width, height)

            if not args.include_unreviewed_damage:
                lines = [line for line in lines if not line.startswith("1 ")]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_file.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, quality=args.jpeg_quality)
            label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            class_counts = Counter(int(line.split()[0]) for line in lines)
            counts[f"{split}/images"] += 1
            counts[f"{split}/tactile_block"] += class_counts[0]
            counts[f"{split}/damage_candidate"] += class_counts[1]
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "source_dataset": row["source_dataset"],
                    "state_original": row["state_original"],
                    "tactile_boxes": class_counts[0],
                    "damage_boxes": class_counts[1],
                    "provisional_damage": row["mvp_label"] == "review",
                }
            )
            if index % 500 == 0:
                print(f"processed={index}/{len(rows)}", flush=True)
    finally:
        for archive in archives.values():
            archive.close()

    yaml = (
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: tactile_block\n  1: damage_candidate\n"
    )
    (args.output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "images": len(records),
        "include_unreviewed_damage": args.include_unreviewed_damage,
        "counts": dict(sorted(counts.items())),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
