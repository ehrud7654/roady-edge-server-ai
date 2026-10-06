from __future__ import annotations

"""Build a balanced replay list without moving or duplicating image files."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "ai/datasets/server_edge_roi_multiclass_v3_crack_replay"
MANIFEST = DATASET / "metadata/manifest.json"
BACKUP = DATASET / "metadata/source_backup.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    backup = json.loads(BACKUP.read_text(encoding="utf-8"))
    source_keys = {
        record["image_file"]: {
            polygon.get("class_key") for polygon in record.get("polygons", [])
        }
        for record in backup["records"]
    }

    train = [item for item in manifest if item.get("split") == "train"]
    paths: list[Path] = []
    replay_counts = {
        "base_train": 0,
        "small_missing": 0,
        "soil_negative": 0,
        "large_missing": 0,
        "crack": 0,
        "wear": 0,
        "small_missing_with_soil": 0,
    }

    for item in train:
        image = DATASET / str(item["image"])
        keys = source_keys.get(str(item.get("source_image")), set())
        role = item.get("sample_role")
        variant = item.get("variant")
        paths.append(image)
        replay_counts["base_train"] += 1

        # Replay only base views for damage classes so existing jittered copies
        # do not dominate merely through repeated frames.
        if role == "edge_group_roi" and variant == "base":
            if "missing_small" in keys:
                paths.append(image)
                replay_counts["small_missing"] += 1
                if "soil" in keys:
                    paths.append(image)
                    replay_counts["small_missing_with_soil"] += 1
            if "missing_large" in keys:
                paths.append(image)
                replay_counts["large_missing"] += 1
            if "crack" in keys:
                paths.append(image)
                replay_counts["crack"] += 1
            if "wear" in keys:
                paths.append(image)
                replay_counts["wear"] += 1

        # Soil-on-intact-block is the paired hard negative for exposed
        # substrate in a small missing corner.
        if role == "no_damage_block_negative" and "soil" in keys:
            paths.extend((image, image))
            replay_counts["soil_negative"] += 2

    list_path = DATASET / "train_small_missing_replay_v5.txt"
    list_path.write_text(
        "\n".join(path.resolve().as_posix() for path in paths) + "\n",
        encoding="utf-8",
    )
    yaml_path = DATASET / "dataset_small_missing_replay_v5_local.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {DATASET.resolve().as_posix()}",
            f"train: {list_path.resolve().as_posix()}",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: tactile_block",
            "  1: missing",
            "  2: crack",
            "  3: wear",
            "  4: obstruction",
            "",
        ]),
        encoding="utf-8",
    )
    small_test = [
        DATASET / str(item["image"])
        for item in manifest
        if item.get("split") == "test"
        and "missing_small" in source_keys.get(str(item.get("source_image")), set())
    ]
    small_test_path = DATASET / "test_small_missing_v5.txt"
    small_test_path.write_text(
        "\n".join(path.resolve().as_posix() for path in small_test) + "\n",
        encoding="utf-8",
    )
    (DATASET / "dataset_small_missing_test_local.yaml").write_text(
        "\n".join([
            f"path: {DATASET.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            f"test: {small_test_path.resolve().as_posix()}",
            "names:",
            "  0: tactile_block",
            "  1: missing",
            "  2: crack",
            "  3: wear",
            "  4: obstruction",
            "",
        ]),
        encoding="utf-8",
    )
    report = {
        "total_train_entries": len(paths),
        "small_missing_test_images": len(small_test),
        "replay_counts": replay_counts,
        "warning": "Replay improves known demo confusions; it does not create new physical defect diversity.",
    }
    (DATASET / "metadata/small_missing_replay_v5.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
