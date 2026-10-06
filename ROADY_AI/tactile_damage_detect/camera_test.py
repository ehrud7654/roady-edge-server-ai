from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE))

import cv2
from ultralytics import YOLO


COLORS = {
    0: (255, 140, 0),
    1: (0, 0, 255),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="카메라에서 점자블록과 파손 후보를 실시간 확인한다.")
    parser.add_argument(
        "--model",
        type=Path,
        default=WORKSPACE / "ROADY_AI/models/edge/tactile_damage_candidate_yolo26n_best.pt",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--device", default=0)
    parser.add_argument("--damage-only", action="store_true")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {args.camera}")

    previous = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]

            for box in result.boxes:
                class_id = int(box.cls.item())
                if args.damage_only and class_id != 1:
                    continue
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                name = result.names[class_id]
                color = COLORS.get(class_id, (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{name} {confidence:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            now = time.perf_counter()
            fps = 1.0 / max(now - previous, 1e-6)
            previous = now
            cv2.putText(
                frame,
                f"FPS {fps:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("ROADY tactile damage detect", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
