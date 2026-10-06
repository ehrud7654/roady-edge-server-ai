from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE))
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / "ROADY_AI/.matplotlib"))

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ROADY person lower-limb obstacle detector.")
    parser.add_argument(
        "--data",
        type=Path,
        default=WORKSPACE / "ROADY_AI/obstacle_lower_limb/dataset.yaml",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE / "ROADY_AI/datasets/obstacle_lower_limb_v1",
    )
    parser.add_argument("--model", default=str(WORKSPACE / "yolo11n.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--name", default="obstacle_lower_limb_yolo11n_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    dataset_root = args.dataset_root.resolve()
    for split in ("train", "val", "test"):
        if not (dataset_root / "images" / split).is_dir():
            raise FileNotFoundError(f"Missing image split: {dataset_root / 'images' / split}")
        if not (dataset_root / "labels" / split).is_dir():
            raise FileNotFoundError(f"Missing label split: {dataset_root / 'labels' / split}")

    data_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    data_config["path"] = dataset_root.as_posix()
    runtime_yaml = WORKSPACE / "ROADY_AI/obstacle_lower_limb/.runtime_dataset.yaml"
    runtime_yaml.write_text(
        yaml.safe_dump(data_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    print(
        json.dumps(
            {
                "data": str(runtime_yaml),
                "dataset_root": str(dataset_root),
                "model": args.model,
                "device": str(device),
                "imgsz": args.imgsz,
                "batch": args.batch,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    model = YOLO(args.model)
    model.train(
        data=str(runtime_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(WORKSPACE / "ROADY_AI/runs"),
        name=args.name,
        workers=args.workers,
        seed=42,
        deterministic=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        patience=args.patience,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
        degrees=3.0,
        translate=0.1,
        scale=0.35,
        fliplr=0.5,
        perspective=0.0005,
        mosaic=0.5,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        plots=True,
        save=True,
        val=True,
    )


if __name__ == "__main__":
    main()
