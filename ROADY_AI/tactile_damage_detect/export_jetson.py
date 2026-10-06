from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jetson에서 YOLO26n ONNX와 TensorRT FP16 엔진을 순서대로 생성한다."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=WORKSPACE
        / "ROADY_AI/models/edge/tactile_damage_candidate_yolo26n_best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    onnx_path = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=1,
            dynamic=False,
            simplify=True,
            opset=args.opset,
        )
    ).resolve()

    trtexec = shutil.which(args.trtexec)
    if trtexec is None:
        raise FileNotFoundError(
            f"trtexec를 찾을 수 없습니다: {args.trtexec}. "
            "Jetson의 TensorRT bin 경로를 확인하세요."
        )
    engine_path = (args.output or onnx_path.with_suffix(".engine")).resolve()
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--memPoolSize=workspace:{args.workspace_mib}",
    ]
    print("ONNX:", onnx_path)
    print("TensorRT command:", " ".join(command))
    subprocess.run(command, check=True)
    print("ENGINE:", engine_path)


if __name__ == "__main__":
    main()
