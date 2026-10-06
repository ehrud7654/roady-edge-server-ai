# ROADY Edge 분석 ROI 및 대표 프레임 선택

이 문서는 Edge 파손 후보 이벤트가 Server AI 분석용 이미지를 선택할 때 사용하는 기본 계약을 설명한다. 현재 Git의 AI 산출물은 Edge 모델 가중치와 Server 분석 모델, 그리고 양쪽이 주고받는 메타데이터 규격을 제공한다. 실제 ROS 이벤트 생성, 대표 프레임 선택, ROI 후처리 튜닝은 Embedded/Server 연동 구현에서 최종 적용한다.

즉 이 문서는 후처리 구현을 강제하는 운영 코드 명세가 아니라, AI 모델이 기대하는 입력 분포와 연동 시 유지해야 할 계약을 정리한 문서이다.

## 입력과 기존 인터페이스

- 카메라 토픽: `/camera/wide/image_raw`
- 모델 출력 클래스: `tactile_block`, `damage_candidate`
- 이벤트 토픽: `/damage/events`
- 이벤트 이미지 순서:
  1. 원본 전체 프레임
  2. Server AI 분석용 ROI

ROS 토픽과 이미지 순서는 유지한다. 과거 코드의 `tactile_roi` 접근자는 호환용 alias로 남아 있지만, 내부 표준 이름은 `analysis_roi`이다.

## 분석 ROI 생성 권장 기준

Server AI는 원본 전체 프레임보다 점자블록 기준 ROI에서 가장 안정적으로 동작한다. 따라서 분석 ROI는 가능한 한 파손 후보와 관련된 점자블록 영역을 기준으로 생성하는 것을 권장한다.

기본 권장 흐름은 다음과 같다.

1. 파손 후보와 교차하거나 중심점 포함 관계가 있는 `tactile_block`을 선택한다.
2. 관련 점자블록이 여러 개면 합집합 Bounding Box를 만든다.
3. 합집합의 가로·세로에 기본 20% 여백을 추가한다.
4. 이미지 경계를 넘어간 좌표를 보정한다.
5. 관련 점자블록이 없을 때만 파손 후보 Bounding Box 1.5배 Crop을 사용한다.

Detect Bounding Box가 점자블록 한 장을 항상 분리한다고 가정하지 않는다. 따라서 tactile 기반 ROI의 분석 단위 힌트는 `block_or_block_group`이다. ROI가 지나치게 tight하면 Server AI의 결손 기준 영역과 블록 분리가 불안정해질 수 있고, 반대로 지나치게 넓으면 벽·바닥 등 학습 분포 밖 배경이 많이 들어갈 수 있다. 최종 ROI 크기와 품질 게이트는 실제 주행 영상에서 Embedded/Server 담당 구현과 함께 조정한다.

이벤트 JSON의 `metadata.ai`에는 다음 값이 추가된다.

```json
{
  "roi_source": "tactile_block",
  "analysis_unit_hint": "block_or_block_group",
  "tactile_detection_count": 2,
  "roi_fallback_used": false
}
```

점자블록을 찾지 못한 경우에는 다음처럼 기록한다.

```json
{
  "roi_source": "damage_fallback",
  "analysis_unit_hint": "unknown",
  "tactile_detection_count": 0,
  "roi_fallback_used": true
}
```

## 대표 프레임 품질 게이트 권장 기준

대표 프레임은 파손 Box가 아닌 분석 ROI를 기준으로 평가하는 것을 권장한다.

- tactile 기반 ROI 존재 여부
- 프레임 가장자리 접촉 여부
- 최소 ROI 너비·높이·면적
- Laplacian 기반 선명도
- 파손 후보와 분석 ROI의 공간적 관계
- 비어 있지 않은 Crop
- 이전 프레임 대비 ROI 면적 변화율
- 이전 프레임 대비 ROI 중심 이동량

안정된 ROI가 품질 게이트를 통과하면 다음 상태를 기록한다.

```json
{
  "frame_selection_status": "quality_gate_passed",
  "frame_quality_verified": true
}
```

이벤트 종료 전까지 검증된 프레임이 없으면 기존 후보 중 품질 점수가 가장 높은 이미지를 전송하되 다음처럼 불확실성을 명시한다.

```json
{
  "frame_selection_status": "fallback_best_available",
  "frame_quality_verified": false
}
```

이 상태는 Server AI가 확정적인 면적·심각도를 반환할 수 있는지 판단하는 입력 조건으로 사용할 수 있다. 다만 최종 프로젝트 기준으로 대표 프레임 품질 게이트와 ROI 후처리는 AI 모델 자체가 아니라 Embedded/Server 연동 구현의 책임 범위이다. AI 서버는 전달받은 메타데이터를 해석해 불확실성을 `review_required`로 전파한다.

## ROS 파라미터

| 파라미터 | 기본값 | 의미 |
|---|---:|---|
| `tactile_roi_margin_ratio` | 0.20 | tactile 합집합 ROI 여백 |
| `damage_fallback_scale` | 1.5 | tactile 미검출 시 파손 Box 확대 비율 |
| `tactile_relation_iou` | 0.01 | 공간 연관성 보조 IoU 기준 |
| `frame_edge_margin_px` | 3 | 프레임 가장자리 제외 여백 |
| `minimum_roi_width_px` | 32 | 최소 ROI 너비 |
| `minimum_roi_height_px` | 32 | 최소 ROI 높이 |
| `minimum_roi_area_px` | 1024 | 최소 ROI 면적 |
| `minimum_roi_sharpness` | 10.0 | Laplacian 분산 최소값 |
| `stable_area_change_ratio` | 0.10 | 안정 프레임 최대 면적 변화율 |
| `stable_center_shift_ratio` | 0.03 | 화면 대각선 대비 최대 중심 이동률 |
| `stable_observation_count` | 2 | 품질 검증에 필요한 연속 안정 관측 수 |

기본값은 시연환경 영상으로 재측정해야 하는 초기값이며 행정 또는 운영 기준이 아니다.

## 하위 호환성

- 기존 `ReadyDamageEvent.tactile_roi`는 deprecated alias로 유지한다.
- 기존 이벤트 JSON 필드는 유지하고 선택적 `metadata` 필드만 추가한다.
- 기존 이벤트 확정 횟수, 시간 창, timeout, 중복 방지 방식은 변경하지 않는다.

## 현재 역할 분담

| 영역 | 담당 |
|---|---|
| Edge AI 모델 가중치와 클래스 계약 | AI |
| Server AI Segmentation 모델과 분석 응답 계약 | AI |
| ROS 토픽 수신, 이벤트 생성, 연속 프레임 추적 | Embedded |
| 최종 ROI 크롭 정책과 대표 프레임 후처리 | Embedded/Server 연동 |
| 관리자 화면 표시와 결과 저장 | Server/Frontend |

AI 문서는 위 연동이 깨지지 않도록 모델 입력 크기, 클래스 순서, 이미지 순서, 메타데이터 의미를 제공한다.

## Server AI 연결 계약

Edge 이벤트에 저장되는 이미지 순서는 다음과 같다.

1. `images[0]`: 원본 전체 프레임 — 관리자 확인 및 보관용
2. `images[1]`: tactile 기준 분석 ROI — Server AI 주 분석 입력

Server는 [edge_event.py](server_damage_analysis/edge_event.py)를 통해 이 계약을 검증한다. 두 번째 이미지가 없으면 원본을 대신 분석하지 않으며, `metadata.ai`의 ROI source·fallback·대표 프레임 품질 상태를 Server 입력으로 전달한다.
