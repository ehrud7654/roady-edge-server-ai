from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE))
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / "ROADY_AI/.matplotlib"))

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="점자블록·파손 후보 YOLO26 Detect 모델을 학습한다.")
    parser.add_argument(
        "--data",
        type=Path,
        default=WORKSPACE / "ROADY_AI/datasets/edge_yolo26n_detect_review_ft_v3/dataset.yaml",
    )
    parser.add_argument(
        "--model",
        default=str(WORKSPACE / "ROADY_AI/models/edge/tactile_damage_candidate_yolo26n_best.pt"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="tactile_damage_yolo26n")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.00015)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device: str | int
    if args.device is not None:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else "cpu"

    print(
        json.dumps(
            {
                "cuda": torch.cuda.is_available(),
                "device": str(device),
                "data": str(args.data.resolve()),
                "model": args.model,
            },
            ensure_ascii=False,
        )
    )

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(WORKSPACE / "ROADY_AI/runs"),
        name=args.name,
        workers=args.workers,
        seed=42,
        deterministic=True,
        plots=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        amp=True,
        cos_lr=True,
        close_mosaic=5,
        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.15,
        degrees=3.0,
        translate=0.04,
        scale=0.1,
        perspective=0.0,
        fliplr=0.5,
        mosaic=0.1,
        mixup=0.0,
    )


if __name__ == "__main__":
    main()
