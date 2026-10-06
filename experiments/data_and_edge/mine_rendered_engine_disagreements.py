from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine exact v2/v3 TensorRT differences from an aligned rendered comparison video."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--minimum-frame-gap", type=int, default=5)
    return parser.parse_args()


def damage_overlay(frame: np.ndarray, raw: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    raw_hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    raw_hue, raw_saturation, raw_value = cv2.split(raw_hsv)
    candidate = (
        ((hue <= 12) | (hue >= 170))
        & (saturation >= 90)
        & (value >= 90)
    )
    raw_red = (
        ((raw_hue <= 15) | (raw_hue >= 165))
        & (raw_saturation >= 55)
        & (raw_value >= 60)
    ).astype(np.uint8)
    raw_red = cv2.dilate(raw_red, np.ones((5, 5), np.uint8)) > 0
    difference = np.max(
        np.abs(frame.astype(np.int16) - raw.astype(np.int16)), axis=2
    ) >= 12
    mask = candidate & difference & ~raw_red
    # Ignore the fixed performance header; detection labels and boxes remain below it.
    mask[:36, :] = False
    return mask


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    gallery = args.output / "gallery"
    gallery.mkdir(exist_ok=True)
    source = cv2.VideoCapture(str(args.source))
    comparison = cv2.VideoCapture(str(args.comparison))
    if not source.isOpened() or not comparison.isOpened():
        raise RuntimeError("Could not open source or comparison video")
    source_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    comparison_frames = int(comparison.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_frames != comparison_frames:
        raise ValueError(f"Frame mismatch: source={source_frames}, comparison={comparison_frames}")
    fps = float(source.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(comparison.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(comparison.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width % 2:
        raise ValueError(f"Comparison width must be even: {width}")
    half_width = width // 2
    rows: list[dict[str, object]] = []
    index = 0
    while True:
        source_ok, raw = source.read()
        comparison_ok, rendered = comparison.read()
        if not source_ok or not comparison_ok:
            break
        resized = cv2.resize(raw, (half_width, height), interpolation=cv2.INTER_LINEAR)
        left, right = rendered[:, :half_width], rendered[:, half_width:]
        v2_mask = damage_overlay(left, resized)
        v3_mask = damage_overlay(right, resized)
        # Dilation turns thin rendered outlines into comparable regions and tolerates codec jitter.
        kernel = np.ones((5, 5), np.uint8)
        v2_region = cv2.dilate(v2_mask.astype(np.uint8), kernel) > 0
        v3_region = cv2.dilate(v3_mask.astype(np.uint8), kernel) > 0
        union = int((v2_region | v3_region).sum())
        difference = int((v2_region ^ v3_region).sum())
        overlap = int((v2_region & v3_region).sum())
        score = difference / union if union else 0.0
        rows.append(
            {
                "frame_index": index,
                "timestamp_sec": round(index / fps, 6) if fps else 0.0,
                "v2_overlay_pixels": int(v2_mask.sum()),
                "v3_overlay_pixels": int(v3_mask.sum()),
                "overlap_region_pixels": overlap,
                "symmetric_difference_pixels": difference,
                "difference_score": round(score, 6),
            }
        )
        index += 1
    source.release()
    comparison.release()

    ranked = sorted(rows, key=lambda row: (row["difference_score"], row["symmetric_difference_pixels"]), reverse=True)
    selected: list[dict[str, object]] = []
    for row in ranked:
        frame_index = int(row["frame_index"])
        if any(abs(frame_index - int(item["frame_index"])) < args.minimum_frame_gap for item in selected):
            continue
        selected.append(row)
        if len(selected) >= args.top:
            break
    selected.sort(key=lambda row: int(row["frame_index"]))
    selected_by_frame = {int(row["frame_index"]): (rank, row) for rank, row in enumerate(selected, start=1)}
    comparison = cv2.VideoCapture(str(args.comparison))
    frame_index = 0
    while True:
        ok, rendered = comparison.read()
        if not ok:
            break
        match = selected_by_frame.get(frame_index)
        if match is not None:
            rank, row = match
            filename = f"rank_{rank:03d}_frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(gallery / filename), rendered)
            row["gallery_image"] = f"gallery/{filename}"
        frame_index += 1
    comparison.release()
    for rank, row in enumerate(selected, start=1):
        frame_index = int(row["frame_index"])
        if "gallery_image" not in row:
            raise RuntimeError(f"Failed to extract selected frame: {frame_index}")

    fieldnames = list(rows[0]) if rows else []
    with (args.output / "all_frame_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    selected_fields = list(dict.fromkeys(fieldnames + ["gallery_image"]))
    with (args.output / "selected_engine_disagreements.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fields)
        writer.writeheader()
        writer.writerows(selected)

    cards = "".join(
        f'<article><img src="{row["gallery_image"]}" loading="lazy"><p>'
        f'frame {row["frame_index"]} · {row["timestamp_sec"]:.2f}s · '
        f'difference {row["difference_score"]:.3f}</p></article>'
        for row in selected
    )
    (args.output / "index.html").write_text(
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>TensorRT v2/v3 검수</title>"
        "<style>body{font-family:system-ui;background:#10131a;color:#eee;margin:24px}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:16px}"
        "article{background:#1c2332;padding:10px;border-radius:10px}img{width:100%}</style></head>"
        f"<body><h1>실제 TensorRT v2/v3 불일치 상위 {len(selected)}프레임</h1>"
        "<p>왼쪽 v2 · 오른쪽 v3. 후보는 렌더된 빨간 파손 영역 차이로 정렬했습니다.</p>"
        f"<main>{cards}</main></body></html>",
        encoding="utf-8",
    )
    print(f"frames={index} selected={len(selected)}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
