from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ROADY lower-limb weights on the target Jetson.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--allow-non-jetson", action="store_true")
    args = parser.parse_args()

    machine = platform.machine().lower()
    is_jetson_arch = machine in {"aarch64", "arm64"}
    if not is_jetson_arch and not args.allow_non_jetson:
        raise RuntimeError(
            "TensorRT engines must be exported on the target Jetson. "
            "Run this script on Jetson Orin Nano or pass --allow-non-jetson only for diagnostics."
        )
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    model = YOLO(str(args.weights))
    output = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=True,
        int8=False,
        dynamic=False,
        batch=1,
        workspace=args.workspace_gb,
        device=0,
    )
    print(f"TensorRT engine: {output}")


if __name__ == "__main__":
    main()
