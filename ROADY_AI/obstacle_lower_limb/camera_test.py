from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE))

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Test lower-limb detection with the USB obstacle camera.")
    parser.add_argument("--model", type=Path, required=True, help=".engine on Jetson or .pt for development")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    model = YOLO(str(args.model), task="detect")
    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open USB camera index {args.camera}")

    latencies: deque[float] = deque(maxlen=120)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            started = time.perf_counter()
            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=0,
                verbose=False,
            )[0]
            latencies.append((time.perf_counter() - started) * 1000.0)
            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)
            mean_latency = sum(latencies) / len(latencies)
            print(
                f"\rdetections={count:2d} latency_mean_ms={mean_latency:7.2f}",
                end="",
                flush=True,
            )

            if args.display:
                annotated = result.plot()
                cv2.putText(
                    annotated,
                    f"{mean_latency:.1f} ms",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("ROADY lower-limb obstacle detector", annotated)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break
    finally:
        print()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
