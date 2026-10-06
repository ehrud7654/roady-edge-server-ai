from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


TACTILE_NAMES = {"점자블록", "점자블럭"}


def is_tactile_annotation(annotation: dict[str, Any]) -> bool:
    return str(annotation.get("label_name", "")).strip() in TACTILE_NAMES


def iter_label_zips(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".zip":
            paths.add(root.resolve())
            continue
        if root.exists():
            paths.update(path.resolve() for path in root.rglob("*.zip"))
    return sorted(paths, key=lambda path: str(path).lower())


def polygon_quality(annotation: dict[str, Any]) -> str:
    if annotation.get("annotation_type") != "polygon":
        return "not_polygon"
    points = annotation.get("annotation_info") or []
    if len(points) < 3:
        return "too_few_points"
    try:
        unique = {(float(point[0]), float(point[1])) for point in points}
    except (TypeError, ValueError, IndexError):
        return "invalid_coordinate"
    return "valid" if len(unique) >= 3 else "degenerate"


def audit_zip(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "json_members": 0,
        "invalid_json": 0,
        "tactile_files": 0,
        "states": Counter(),
        "defect_labels": Counter(),
        "annotation_types": Counter(),
        "polygon_quality": Counter(),
        "sample_filenames": [],
    }
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".json"):
                continue
            result["json_members"] += 1
            try:
                obj = json.loads(archive.read(member).decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                result["invalid_json"] += 1
                continue
            annotations = [
                annotation
                for annotation in (obj.get("annotations") or [])
                if is_tactile_annotation(annotation)
            ]
            if not annotations:
                continue
            result["tactile_files"] += 1
            description = obj.get("description") or {}
            state = str(description.get("state", "unknown"))
            result["states"][state] += 1
            if len(result["sample_filenames"]) < 5:
                result["sample_filenames"].append((obj.get("info") or {}).get("filename"))
            for annotation in annotations:
                result["defect_labels"][str(annotation.get("is_defect", "unknown"))] += 1
                result["annotation_types"][str(annotation.get("annotation_type", "unknown"))] += 1
                result["polygon_quality"][polygon_quality(annotation)] += 1

    for key in ("states", "defect_labels", "annotation_types", "polygon_quality"):
        result[key] = dict(sorted(result[key].items()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="압축 해제 없이 AI Hub 도로시설물 라벨 ZIP의 점자블록 구성을 집계한다."
    )
    parser.add_argument("roots", nargs="+", type=Path, help="ZIP 파일 또는 ZIP을 포함한 폴더")
    parser.add_argument("--output", type=Path, help="JSON 결과 저장 경로")
    args = parser.parse_args()

    archives = []
    for path in iter_label_zips(args.roots):
        try:
            with zipfile.ZipFile(path) as archive:
                has_json = any(
                    not member.is_dir() and member.filename.lower().endswith(".json")
                    for member in archive.infolist()
                )
            if has_json:
                archives.append(audit_zip(path))
        except (zipfile.BadZipFile, OSError) as error:
            archives.append({"path": str(path), "error": str(error)})

    totals = {
        "archives": len(archives),
        "tactile_files": sum(item.get("tactile_files", 0) for item in archives),
        "invalid_json": sum(item.get("invalid_json", 0) for item in archives),
    }
    payload = {"totals": totals, "archives": archives}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
