from __future__ import annotations

"""Wait for v4, test v3/v4, choose the safer checkpoint, then train v5."""

import json
import subprocess
import time
from pathlib import Path

import psutil
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "ai/runs"
DATASET = ROOT / "ai/datasets/server_edge_roi_multiclass_v3_crack_replay"
V3 = ROOT / "ai/models/server/server_yolo26s_seg_multiclass_v3/yolo26s_seg_multiclass_v3_best.pt"
V4_RUN = RUNS / "server_yolo26s_edge_roi_multiclass_v4_instances_local-3"
V4 = V4_RUN / "weights/best.pt"
REPORT = RUNS / "server_v3_v4_selection.json"
TRAINING_PID = 109352


def wait_for_v4() -> None:
    while psutil.pid_exists(TRAINING_PID):
        time.sleep(30)
    if not V4.is_file():
        raise FileNotFoundError(f"v4 did not produce best.pt: {V4}")


def evaluate(model_path: Path, name: str) -> dict[str, object]:
    metrics = YOLO(str(model_path)).val(
        data=str(DATASET / "dataset_local.yaml"),
        split="test",
        imgsz=1024,
        batch=1,
        device=0,
        workers=0,
        plots=True,
        project=str(RUNS),
        name=name,
    )
    names = metrics.names
    seg_recall = {names[i]: float(value) for i, value in enumerate(metrics.seg.r)}
    seg_ap50 = {names[i]: float(value) for i, value in enumerate(metrics.seg.ap50)}
    overall_map = float(metrics.seg.map)
    small_metrics = YOLO(str(model_path)).val(
        data=str(DATASET / "dataset_small_missing_test_local.yaml"),
        split="test",
        imgsz=1024,
        batch=1,
        device=0,
        workers=0,
        plots=False,
        project=str(RUNS),
        name=f"{name}_small_missing",
    )
    small_names = small_metrics.names
    small_recall = {
        small_names[i]: float(value) for i, value in enumerate(small_metrics.seg.r)
    }.get("missing", 0.0)
    # Preserve the difficult classes and block denominator. Obstruction has
    # only four test instances, so it is reported but not allowed to dominate.
    score = (
        0.20 * small_recall
        + 0.20 * seg_recall.get("missing", 0.0)
        + 0.15 * seg_recall.get("crack", 0.0)
        + 0.15 * seg_recall.get("wear", 0.0)
        + 0.15 * seg_recall.get("tactile_block", 0.0)
        + 0.15 * overall_map
    )
    return {
        "model": str(model_path),
        "selection_score": score,
        "mask_recall": seg_recall,
        "small_missing_subset_mask_recall": small_recall,
        "mask_ap50": seg_ap50,
        "mask_map50_95": overall_map,
    }


def main() -> None:
    wait_for_v4()
    subprocess.run(
        [str(ROOT / ".venv-yolo/Scripts/python.exe"),
         str(ROOT / "ai/scripts/build_server_small_missing_replay_v5.py")],
        check=True,
    )
    v3_result = evaluate(V3, "server_v3_fixed_test_local")
    v4_result = evaluate(V4, "server_v4_instances_fixed_test_local")
    # Demand a real improvement; within 0.5 percentage points keep v3 to
    # avoid replacing a proven checkpoint due to small-test noise.
    if float(v4_result["selection_score"]) >= float(v3_result["selection_score"]) + 0.005:
        selected = V4
        selected_name = "v4"
    else:
        selected = V3
        selected_name = "v3"
    report = {
        "v3": v3_result,
        "v4": v4_result,
        "selected": selected_name,
        "selected_model": str(selected),
        "selection_rule": "v4 must exceed v3 composite score by at least 0.005",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    model = YOLO(str(selected))
    model.train(
        data=str(DATASET / "dataset_small_missing_replay_v5_local.yaml"),
        epochs=25,
        imgsz=1024,
        batch=1,
        device=0,
        workers=0,
        optimizer="AdamW",
        lr0=0.00005,
        cos_lr=True,
        patience=7,
        degrees=2,
        translate=0.04,
        scale=0.10,
        perspective=0.0001,
        mosaic=0.02,
        close_mosaic=6,
        hsv_h=0.006,
        hsv_s=0.20,
        hsv_v=0.15,
        mask_ratio=2,
        overlap_mask=False if selected_name == "v4" else True,
        seed=42,
        project=str(RUNS),
        name="server_yolo26s_edge_roi_multiclass_v5_small_missing_replay_local",
    )
    v5 = RUNS / "server_yolo26s_edge_roi_multiclass_v5_small_missing_replay_local/weights/best.pt"
    if not v5.is_file():
        raise FileNotFoundError(f"v5 did not produce best.pt: {v5}")
    v5_result = evaluate(v5, "server_v5_small_missing_fixed_test_local")
    final_report = json.loads(REPORT.read_text(encoding="utf-8"))
    final_report["v5"] = v5_result
    candidates = {
        "v3": final_report["v3"],
        "v4": final_report["v4"],
        "v5": final_report["v5"],
    }
    final_name = max(candidates, key=lambda key: float(candidates[key]["selection_score"]))
    final_report["final_selected"] = final_name
    final_report["final_selected_model"] = candidates[final_name]["model"]
    REPORT.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
