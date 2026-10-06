from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture training frames from the mounted USB obstacle camera.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=0.0,
        help="Automatic capture interval. Use 0 for manual S-key capture only.",
    )
    parser.add_argument("--scene-id", default="scene")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open USB camera index {args.camera}")

    last_saved_at = 0.0
    saved_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue

            now = time.monotonic()
            auto_save = args.interval_sec > 0 and now - last_saved_at >= args.interval_sec
            preview = frame.copy()
            cv2.putText(
                preview,
                f"saved={saved_count}  S=save  Q/Esc=quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
            cv2.imshow("ROADY obstacle-camera data capture", preview)
            key = cv2.waitKey(1) & 0xFF
            manual_save = key in {ord("s"), ord("S")}

            if auto_save or manual_save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output = args.output / f"{args.scene_id}_{timestamp}.jpg"
                if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Failed to save {output}")
                saved_count += 1
                last_saved_at = now
                print(output)

            if key in {27, ord("q"), ord("Q")}:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print(f"Captured {saved_count} frames into {args.output}")


if __name__ == "__main__":
    main()
