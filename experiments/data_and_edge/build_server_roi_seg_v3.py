from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def parse_label(path: Path, width: int, height: int) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    blocks: list[list[tuple[float, float]]] = []
    damages: list[list[tuple[float, float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7 or len(values) % 2 == 0:
            continue
        class_id = int(values[0])
        coords = [float(value) for value in values[1:]]
        points = [(x * width, y * height) for x, y in zip(coords[0::2], coords[1::2])]
        (blocks if class_id == 0 else damages).append(points)
    return blocks, damages


def bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def square_crop(cx: float, cy: float, side: float, width: int, height: int) -> tuple[int, int, int, int]:
    side = min(max(32.0, side), float(width), float(height))
    left = min(max(cx - side / 2, 0.0), width - side)
    top = min(max(cy - side / 2, 0.0), height - side)
    return (
        int(round(left)),
        int(round(top)),
        int(round(left + side)),
        int(round(top + side)),
    )


def clip_edge(points, inside, intersect):
    if not points:
        return []
    output = []
    previous = points[-1]
    for current in points:
        current_inside = inside(current)
        previous_inside = inside(previous)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous = current
    return output


def clip_polygon(points, left, top, right, bottom):
    def vertical(edge):
        def intersect(a, b):
            if abs(b[0] - a[0]) < 1e-9:
                return edge, a[1]
            t = (edge - a[0]) / (b[0] - a[0])
            return edge, a[1] + t * (b[1] - a[1])
        return intersect

    def horizontal(edge):
        def intersect(a, b):
            if abs(b[1] - a[1]) < 1e-9:
                return a[0], edge
            t = (edge - a[1]) / (b[1] - a[1])
            return a[0] + t * (b[0] - a[0]), edge
        return intersect

    result = clip_edge(points, lambda p: p[0] >= left, vertical(left))
    result = clip_edge(result, lambda p: p[0] <= right, vertical(right))
    result = clip_edge(result, lambda p: p[1] >= top, horizontal(top))
    return clip_edge(result, lambda p: p[1] <= bottom, horizontal(bottom))


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    ) / 2


def yolo_line(class_id: int, points, left, top, crop_width, crop_height) -> str | None:
    clipped = clip_polygon(points, left, top, left + crop_width, top + crop_height)
    if len(clipped) < 3 or polygon_area(clipped) < 4:
        return None
    coords: list[float] = []
    for x, y in clipped:
        coords.extend(
            (
                min(1.0, max(0.0, (x - left) / crop_width)),
                min(1.0, max(0.0, (y - top) / crop_height)),
            )
        )
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in coords)


def find_image(directory: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found: {directory}/{stem}")


def stable_rng(key: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Edge-like ROI crops for two-class server segmentation.")
    parser.add_argument("--source", type=Path, default=Path.home() / "roady/datasets/server_seg_v2")
    parser.add_argument("--image-source", type=Path, help="Optional image dataset root; defaults to --source.")
    parser.add_argument("--output", type=Path, default=Path.home() / "roady/datasets/server_roi_seg_v3")
    parser.add_argument("--min-side", type=int, default=640)
    parser.add_argument("--context", type=float, default=4.0)
    parser.add_argument("--block-context", type=float, default=2.0)
    parser.add_argument("--negative-crops", type=int, default=1)
    parser.add_argument("--train-jitter", type=float, default=0.08)
    parser.add_argument("--train-scale-min", type=float, default=0.9)
    parser.add_argument("--train-scale-max", type=float, default=1.2)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-split", type=int, default=0)
    args = parser.parse_args()
    image_source = args.image_source or args.source
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    counts = Counter()
    records: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        labels = sorted((args.source / "labels" / split).glob("*.txt"))
        if args.limit_per_split > 0:
            labels = labels[: args.limit_per_split]
        for number, label_path in enumerate(labels, 1):
            image_path = find_image(image_source / "images" / split, label_path.stem)
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            width, height = image.size
            blocks, damages = parse_label(label_path, width, height)
            if not blocks:
                counts[f"{split}/skipped_no_block"] += 1
                continue
            block_box = bbox(blocks[0])
            block_short = max(1.0, min(block_box[2] - block_box[0], block_box[3] - block_box[1]))
            crops: list[tuple[str, tuple[int, int, int, int], str]] = []

            if damages:
                for damage_index, damage in enumerate(damages):
                    damage_box = bbox(damage)
                    cx = (damage_box[0] + damage_box[2]) / 2
                    cy = (damage_box[1] + damage_box[3]) / 2
                    damage_side = max(damage_box[2] - damage_box[0], damage_box[3] - damage_box[1])
                    side = max(args.min_side, damage_side * args.context, block_short * args.block_context)
                    if split == "train":
                        rng = stable_rng(f"{label_path.stem}:{damage_index}", args.seed)
                        cx += rng.uniform(-args.train_jitter, args.train_jitter) * side
                        cy += rng.uniform(-args.train_jitter, args.train_jitter) * side
                        side *= rng.uniform(args.train_scale_min, args.train_scale_max)
                    crops.append(
                        (
                            f"p{damage_index:02d}",
                            square_crop(cx, cy, side, width, height),
                            "positive",
                        )
                    )
            else:
                long_vertical = (block_box[3] - block_box[1]) >= (block_box[2] - block_box[0])
                for negative_index in range(args.negative_crops):
                    fraction = (negative_index + 1) / (args.negative_crops + 1)
                    cx = (block_box[0] + block_box[2]) / 2 if long_vertical else block_box[0] + (block_box[2] - block_box[0]) * fraction
                    cy = block_box[1] + (block_box[3] - block_box[1]) * fraction if long_vertical else (block_box[1] + block_box[3]) / 2
                    side = max(args.min_side, block_short * 2.5)
                    crops.append(
                        (
                            f"n{negative_index:02d}",
                            square_crop(cx, cy, side, width, height),
                            "negative",
                        )
                    )

            for crop_id, (left, top, right, bottom), kind in crops:
                crop_width, crop_height = right - left, bottom - top
                stem = f"{label_path.stem}_{crop_id}"
                output_image = args.output / "images" / split / f"{stem}.jpg"
                output_label = args.output / "labels" / split / f"{stem}.txt"
                output_image.parent.mkdir(parents=True, exist_ok=True)
                output_label.parent.mkdir(parents=True, exist_ok=True)
                image.crop((left, top, right, bottom)).save(output_image, quality=args.jpeg_quality)

                lines: list[str] = []
                for class_id, polygons in ((0, blocks), (1, damages)):
                    for polygon in polygons:
                        line = yolo_line(class_id, polygon, left, top, crop_width, crop_height)
                        if line:
                            lines.append(line)
                output_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
                class_counts = Counter(int(line.split()[0]) for line in lines)
                counts[f"{split}/crops"] += 1
                counts[f"{split}/{kind}"] += 1
                counts[f"{split}/tactile_instances"] += class_counts[0]
                counts[f"{split}/damage_instances"] += class_counts[1]
                records.append(
                    {
                        "split": split,
                        "source": label_path.stem,
                        "roi": stem,
                        "kind": kind,
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "tactile_instances": class_counts[0],
                        "damage_instances": class_counts[1],
                    }
                )
            if number % 1000 == 0:
                print(f"{split}: source={number}/{len(labels)}, crops={counts[f'{split}/crops']}", flush=True)

    yaml_text = (
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: tactile_block\n  1: damage\n"
    )
    (args.output / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    with (args.output / "roi_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "parameters": vars(args)
        | {
            "source": str(args.source),
            "image_source": str(image_source),
            "output": str(args.output),
        },
        "counts": dict(sorted(counts.items())),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
