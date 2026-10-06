from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO


ROOT = Path.cwd()  # Set the working directory to a local ROADY data workspace.
INPUT = ROOT / "ai/analysis/server_v3_v4_v5_video_test/tactile_v1_v2_source.mp4"
OUTPUT = ROOT / "ai/analysis/server_v3_v4_v5_video_test"
MODELS = {
    "v3": ROOT / "ai/models/server/server_yolo26s_seg_multiclass_v3/yolo26s_seg_multiclass_v3_best.pt",
    "v4": ROOT / "ai/runs/server_yolo26s_edge_roi_multiclass_v4_instances_local-3/weights/best.pt",
    "v5": ROOT / "ai/runs/server_yolo26s_edge_roi_multiclass_v5_small_missing_replay_local/weights/best.pt",
}
CONF = 0.05
IMGSZ = 1024


def video_info(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, frames


def run_model(label: str, weight: Path, fps: float, width: int, height: int) -> dict:
    model = YOLO(str(weight))
    names = model.names
    output_video = OUTPUT / f"server_{label}_prediction.mp4"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    counts = defaultdict(int)
    frame_hits = defaultdict(set)
    confidences = defaultdict(list)
    frame_rows = []
    started = perf_counter()

    results = model.predict(
        source=str(INPUT),
        stream=True,
        imgsz=IMGSZ,
        conf=CONF,
        iou=0.7,
        device=0,
        verbose=False,
    )
    for frame_index, result in enumerate(results):
        writer.write(result.plot())
        detections = []
        if result.boxes is not None:
            for cls_value, conf_value, xyxy_value in zip(
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.xyxy.cpu().tolist(),
            ):
                cls_id = int(cls_value)
                name = names[cls_id]
                confidence = float(conf_value)
                counts[name] += 1
                frame_hits[name].add(frame_index)
                confidences[name].append(confidence)
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": name,
                        "confidence": confidence,
                        "bbox_xyxy": [round(float(v), 2) for v in xyxy_value],
                    }
                )
        frame_rows.append({"frame_index": frame_index, "detections": detections})
        if (frame_index + 1) % 100 == 0:
            print(f"{label}: processed {frame_index + 1} frames", flush=True)

    writer.release()
    elapsed = perf_counter() - started
    summary = {
        "model": str(weight),
        "output_video": str(output_video),
        "confidence_threshold": CONF,
        "imgsz": IMGSZ,
        "processed_frames": len(frame_rows),
        "elapsed_seconds": elapsed,
        "throughput_fps": len(frame_rows) / elapsed if elapsed else None,
        "classes": {},
    }
    for name in names.values():
        values = confidences[name]
        summary["classes"][name] = {
            "detections": counts[name],
            "detected_frames": len(frame_hits[name]),
            "confidence_mean": sum(values) / len(values) if values else None,
            "confidence_max": max(values) if values else None,
        }
    (OUTPUT / f"server_{label}_detections.json").write_text(
        json.dumps(frame_rows, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def make_comparison(fps: float, width: int, height: int) -> Path:
    paths = [OUTPUT / f"server_{label}_prediction.mp4" for label in MODELS]
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    panel_width = 640
    panel_height = max(1, round(height * panel_width / width))
    output = OUTPUT / "server_v3_v4_v5_comparison.mp4"
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_width * len(captures), panel_height),
    )
    labels = list(MODELS)
    while True:
        frames = []
        for label, cap in zip(labels, captures):
            ok, frame = cap.read()
            if not ok:
                frames = []
                break
            frame = cv2.resize(frame, (panel_width, panel_height))
            cv2.rectangle(frame, (0, 0), (150, 42), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"SERVER {label.upper()}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frames.append(frame)
        if not frames:
            break
        writer.write(cv2.hconcat(frames))
    writer.release()
    for cap in captures:
        cap.release()
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fps, width, height, frames = video_info(INPUT)
    print(f"input: {width}x{height}, fps={fps:.3f}, frames={frames}", flush=True)
    report = {
        "input": str(INPUT),
        "video": {"fps": fps, "width": width, "height": height, "frames": frames},
        "models": {},
    }
    for label, weight in MODELS.items():
        print(f"starting {label}: {weight}", flush=True)
        report["models"][label] = run_model(label, weight, fps, width, height)
    report["comparison_video"] = str(make_comparison(fps, width, height))
    report_path = OUTPUT / "server_v3_v4_v5_video_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
