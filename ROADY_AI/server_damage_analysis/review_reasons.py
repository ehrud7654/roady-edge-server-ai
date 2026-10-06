from __future__ import annotations

from typing import Iterable


REVIEW_MESSAGES = {
    "ROI_DAMAGE_FALLBACK_USED": "점자블록 기준 영역을 찾지 못해 파손 후보 영역을 대신 사용했습니다.",
    "FRAME_QUALITY_NOT_VERIFIED": "대표 프레임이 품질 검증 조건을 충족하지 못했습니다.",
    "TACTILE_BLOCK_NOT_DETECTED": "점자블록 영역이 탐지되지 않았습니다.",
    "BLOCK_INSTANCE_UNRESOLVED": "점자블록을 개별 블록으로 분리하지 못해 블록 그룹 단위로 분석했습니다.",
    "EXPECTED_BLOCK_REGION_UNAVAILABLE": "결손 비율 계산에 필요한 예상 블록 영역을 신뢰성 있게 생성하지 못했습니다.",
    "MISSING_AREA_UNCERTAIN": "결손은 탐지되었으나 결손 면적 비율을 확정할 수 없습니다.",
    "MODEL_CLASS_MAPPING_INVALID": "모델 클래스 구성이 서버 분석 규격과 일치하지 않습니다.",
    "LOW_TACTILE_CONFIDENCE": "점자블록 탐지 신뢰도가 기준보다 낮습니다.",
    "LOW_DAMAGE_CONFIDENCE": "파손 탐지 신뢰도가 자동 분석 기준보다 낮습니다.",
    "RATIO_NEAR_THRESHOLD": "파손 비율이 심각도 등급 경계에 가까워 담당자 확인이 필요합니다.",
    "EDGE_SERVER_DISAGREEMENT": "Edge에서 파손 후보가 탐지되었으나 서버에서 파손 유형을 확정하지 못했습니다.",
    "OBSTRUCTION_SUSPECTED": "점자블록이 물체에 가려져 파손 여부와 파손 비율을 확정할 수 없습니다.",
}


def review_reason(code: str) -> dict[str, str]:
    return {"code": code, "message": REVIEW_MESSAGES.get(code, code)}


def review_reason_list(codes: Iterable[str]) -> list[dict[str, str]]:
    return [review_reason(code) for code in dict.fromkeys(codes)]
