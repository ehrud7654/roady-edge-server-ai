from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path.cwd()  # Set the working directory to a local ROADY data workspace.
SOURCE = ROOT / "ai/edge_review_v2_v3/out"
SOURCE_VIDEO = ROOT / "ai/edge_review_v2_v3/source/tactile_v1_v2_source.mp4"
BASE = ROOT / "ai/datasets/edge_balanced_replay_v3"
OUTPUT = ROOT / "ai/datasets/edge_large_missing_replay_v4"
MODEL = ROOT / "ai/runs/edge_yolo26n_balanced_replay_v3-2/weights/best.pt"

TACTILE_CONFIDENCE = 0.45
DAMAGE_CONFIDENCE = 0.50
MANUAL_IOU_THRESHOLD = 0.50
HARD_REPLAY_REPEAT = 5
NEGATIVE_REPLAY_REPEAT = 4
REVIEWED_NEGATIVE_FRAMES = (1256, 1383)


def xywh_to_xyxy(box: list[float]) -> list[float]:
    xc, yc, w, h = box
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(first)
    bx1, by1, bx2, by2 = xywh_to_xyxy(second)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def read_manual(path: Path) -> list[tuple[int, list[float]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) == 5:
            rows.append((int(values[0]), [float(value) for value in values[1:]]))
    return rows


def result_boxes(result) -> list[tuple[int, float, list[float]]]:
    height, width = result.orig_shape
    rows = []
    if result.boxes is None:
        return rows
    for cls_value, conf_value, xyxy in zip(
        result.boxes.cls.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.xyxy.cpu().tolist(),
    ):
        x1, y1, x2, y2 = xyxy
        rows.append(
            (
                int(cls_value),
                float(conf_value),
                [
                    (x1 + x2) / (2 * width),
                    (y1 + y2) / (2 * height),
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ],
            )
        )
    return rows


def write_label(path: Path, rows: list[tuple[int, list[float]]]) -> None:
    text = "\n".join(
        f"{class_id} " + " ".join(f"{value:.6f}" for value in box)
        for class_id, box in rows
    )
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def extract_video_frames(frame_numbers: tuple[int, ...], target: Path) -> list[Path]:
    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(SOURCE_VIDEO)
    paths = []
    for number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {number}")
        path = target / f"negative_frame_{number:06d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    cap.release()
    return paths


def main() -> None:
    images_dir = OUTPUT / "images/train"
    labels_dir = OUTPUT / "labels/train"
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    model = YOLO(str(MODEL))
    if model.names != {0: "tactile_block", 1: "damage_candidate"}:
        raise ValueError(f"Unexpected class mapping: {model.names}")

    summary = Counter()
    hard_paths: list[Path] = []
    image_paths = sorted((SOURCE / "images").glob("*.jpg"))
    for source_image in image_paths:
        target_image = images_dir / source_image.name
        shutil.copy2(source_image, target_image)
        manual = read_manual(SOURCE / "labels" / f"{source_image.stem}.txt")
        prediction = model.predict(
            source=str(source_image), imgsz=768, conf=0.05, iou=0.7, device=0, verbose=False
        )[0]
        merged = list(manual)
        for class_id, confidence, box in result_boxes(prediction):
            threshold = TACTILE_CONFIDENCE if class_id == 0 else DAMAGE_CONFIDENCE
            if confidence < threshold:
                continue
            if class_id == 1 and any(
                manual_class == 1 and iou(box, manual_box) >= MANUAL_IOU_THRESHOLD
                for manual_class, manual_box in manual
            ):
                summary["pseudo_damage_suppressed_by_manual"] += 1
                continue
            merged.append((class_id, box))
            summary[f"pseudo_class_{class_id}"] += 1
        write_label(labels_dir / f"{source_image.stem}.txt", merged)
        hard_paths.append(target_image.resolve())
        summary["manual_damage"] += len(manual)

    negative_paths = extract_video_frames(REVIEWED_NEGATIVE_FRAMES, images_dir)
    for negative_image in negative_paths:
        prediction = model.predict(
            source=str(negative_image), imgsz=768, conf=0.05, iou=0.7, device=0, verbose=False
        )[0]
        tactile_rows = [
            (class_id, box)
            for class_id, confidence, box in result_boxes(prediction)
            if class_id == 0 and confidence >= TACTILE_CONFIDENCE
        ]
        write_label(labels_dir / f"{negative_image.stem}.txt", tactile_rows)
        summary["negative_tactile"] += len(tactile_rows)

    base_entries = [
        Path(line.strip()).resolve()
        for line in (BASE / "train_balanced.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_entries = list(base_entries)
    for path in hard_paths:
        train_entries.extend([path] * HARD_REPLAY_REPEAT)
    for path in negative_paths:
        train_entries.extend([path.resolve()] * NEGATIVE_REPLAY_REPEAT)
    (OUTPUT / "train_replay_v4.txt").write_text(
        "\n".join(str(path) for path in train_entries) + "\n", encoding="utf-8"
    )
    dataset = "\n".join(
        [
            f"path: {BASE.as_posix()}",
            f"train: {(OUTPUT / 'train_replay_v4.txt').as_posix()}",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: tactile_block",
            "  1: damage_candidate",
            "",
        ]
    )
    (OUTPUT / "dataset_local.yaml").write_text(dataset, encoding="utf-8")
    report = {
        "base_model": str(MODEL),
        "base_train_entries": len(base_entries),
        "hard_frames": len(hard_paths),
        "hard_replay_repeat": HARD_REPLAY_REPEAT,
        "reviewed_negative_frames": list(REVIEWED_NEGATIVE_FRAMES),
        "negative_replay_repeat": NEGATIVE_REPLAY_REPEAT,
        "final_train_entries": len(train_entries),
        "pseudo_thresholds": {
            "tactile_block": TACTILE_CONFIDENCE,
            "damage_candidate": DAMAGE_CONFIDENCE,
        },
        "counts": dict(summary),
        "warning": "The reviewed source video is now training data and cannot be used as an unbiased final test.",
    }
    (OUTPUT / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
