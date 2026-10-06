from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_NAMES = {"foot": 0, "lower leg": 1, "lower_leg": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Foot and Lower Leg from the COCO Body Part YOLO dataset."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.4)
    return parser.parse_args()


def find_data_yaml(source: Path) -> Path:
    for candidate in [source / "data.yaml", source / "dataset.yaml", *source.glob("*.yaml")]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No dataset YAML found under {source}")


def read_names(data_yaml: Path) -> dict[int, str]:
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


def choose_output_split(source_split: str, stem: str, test_fraction: float) -> str:
    if source_split == "train":
        return "train"
    digest = hashlib.sha256(stem.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "test" if value < test_fraction else "val"


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    if not 0.1 <= args.test_fraction <= 0.5:
        raise ValueError("--test-fraction must be between 0.1 and 0.5")
    if args.output.exists() and any(args.output.rglob("*")):
        raise FileExistsError(f"Output must be absent or empty: {args.output}")

    names = read_names(find_data_yaml(args.source))
    source_to_target: dict[int, int] = {}
    for class_id, name in names.items():
        normalized = name.strip().lower().replace("-", " ")
        if normalized in TARGET_NAMES:
            source_to_target[class_id] = TARGET_NAMES[normalized]
    if set(source_to_target.values()) != {0, 1}:
        raise RuntimeError(f"Could not resolve Foot and Lower Leg. Source names: {names}")

    stats: Counter[str] = Counter()
    for source_split in ("train", "val"):
        image_dir = args.source / "images" / source_split
        label_dir = args.source / "labels" / source_split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing source split directories for {source_split}")

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            output_split = choose_output_split(source_split, image_path.stem, args.test_fraction)
            output_image = args.output / "images" / output_split / image_path.name
            output_label = args.output / "labels" / output_split / f"{image_path.stem}.txt"
            link_or_copy(image_path, output_image)

            selected: list[str] = []
            source_label = label_dir / f"{image_path.stem}.txt"
            if source_label.is_file():
                for line_number, raw in enumerate(
                    source_label.read_text(encoding="utf-8").splitlines(), 1
                ):
                    parts = raw.split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        raise ValueError(
                            f"{source_label}:{line_number}: expected 5 fields"
                        )
                    source_class = int(parts[0])
                    if source_class in source_to_target:
                        target_class = source_to_target[source_class]
                        selected.append(" ".join([str(target_class), *parts[1:]]))
                        stats[f"class_{target_class}_boxes"] += 1

            output_label.parent.mkdir(parents=True, exist_ok=True)
            output_label.write_text(
                "\n".join(selected) + ("\n" if selected else ""),
                encoding="utf-8",
            )
            stats[f"{output_split}_images"] += 1
            stats[f"{output_split}_positive_images"] += int(bool(selected))
            stats[f"{output_split}_negative_images"] += int(not selected)

    dataset_yaml = {
        "path": args.output.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "foot", 1: "lower_leg"},
    }
    (args.output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    report = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "source_class_mapping": source_to_target,
        "stats": dict(sorted(stats.items())),
    }
    (args.output / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
