# ROADY 서버 AI 파손 분석

Jira: `S15P11A404-168`

서버 AI는 Edge가 전송한 점자블록 기준 ROI를 YOLO26s Segmentation 모델로 분석합니다. 현재 운영 후보는 점자블록과 파손 3유형, 가림 물체를 포함한 5클래스 모델입니다. 분석 전제조건을 검증할 수 없으면 비율이나 심각도를 확정하지 않고 `null`, `review_required=true`, `advisory_only=true`로 반환합니다.

## 현재 연결 모델

- 모델: `YOLO26s-seg`
- 가중치: `models/server/yolo26s_seg_multiclass_v4_best.pt`
- 입력 크기: 768
- 학습 클래스: `tactile_block`, `missing`, `crack`, `wear`, `obstruction`
- 모델 SHA-256은 같은 폴더의 `.sha256` 파일로 검증합니다.

큰 결손과 작은 결손은 별도 학습 클래스가 아닙니다. 모델은 두 유형을 `missing`으로
분할하고, 개별 점자블록 대비 결손 면적 비율을 계산할 수 있을 때 15% 기준으로
`SMALL_MISSING`과 `LARGE_MISSING`을 API 변환 단계에서 구분합니다. 기준 영역을 신뢰할 수 없으면
보수적으로 `LARGE_MISSING`, `moderate` 판정을 적용하고 비율만 `null`로 유지합니다. `damage_score`는
비율이나 점자블록 검출 여부와 무관하게 통합 파손 마스크의 픽셀 수로 계산합니다.

Edge에서 파손 후보가 탐지됐지만 서버가 `missing`, `crack`, `wear` 중 신뢰 가능한 근거를
확정하지 못한 경우에는 `EDGE_SERVER_DISAGREEMENT` 사유와 함께 `review_required=true`로
전환합니다. 씽씽이처럼 점자블록을 가리는 물체도 정상 또는 파손으로 임의 확정하지 않습니다.

## 모델 계약

필수 파손 분석 클래스는 `tactile_block`, `missing`, `crack`, `wear`입니다. `obstruction`은 시연환경에서 점자블록 위 거치물, 예를 들어 씽씽이를 판단 보류로 보내기 위한 선택 클래스입니다. 클래스 ID를 고정하지 않고 모델의 `names` 메타데이터를 검사합니다. 필수 클래스가 없으면 `MODEL_CLASS_MAPPING_INVALID`로 검토 전환합니다. 기존 통합 `damage` 결과는 `missing ∪ crack ∪ wear`로 계산한 deprecated 호환 필드에서만 제공합니다.

## 분석 단위

- `block`: 모델이 개별 점자블록 인스턴스로 분리했고 형태 검사를 통과한 경우
- `block_group`: 여러 블록이 연결되었거나 개별 블록 여부가 불명확한 경우

블록을 임의 격자로 강제 분할하지 않습니다. `block_group`에는 `BLOCK_INSTANCE_UNRESOLVED` 검토 사유를 추가하며 `local_unit_id`는 한 이미지 안에서만 유효합니다.

## Mask와 비율

- 균열과 마모: `type_mask ∩ tactile_unit_mask`
- 결손: tactile 교집합을 사용하지 않습니다. 주변 개별 블록의 크기와 간격이 안정적인 경우에만 `expected_block_region`을 사용합니다.
- expected region을 신뢰할 수 없으면 결손 비율은 `null`, 상태는 `not_estimable`로 유지합니다. 파손 판정은 보류하지 않고 결손 유형의 최소 심각도 `moderate`를 적용합니다.

| 비율 상태 | 의미 |
|---|---|
| `measured_no_damage` | 해당 유형 파손이 없어 0% |
| `estimated` | 유효 기준 영역에서 계산한 추정 비율 |
| `not_estimable` | 파손은 있으나 기준 영역을 검증할 수 없어 계산 불가 |

## 심각도 정책

정책은 [severity_policy.yaml](severity_policy.yaml)에 있습니다. 현재 `draft-1`은 검증된 행정 기준이 아닌 잠정 정책이므로 모든 결과는 `advisory_only=true`입니다. 유형별 심각도와 `minimum_severity`를 적용한 뒤 가장 높은 등급을 분석 단위 등급으로 사용합니다. 마모에도 상한을 두지 않습니다.

## Edge 입력 계약

서버는 `roi_source`, `roi_fallback_used`, `frame_quality_verified`를 입력 메타데이터로 받습니다. damage fallback ROI 또는 품질 미검증 프레임은 검토 사유로 전파됩니다. 이벤트 JSON의 `images[0]`은 원본 프레임, `images[1]`은 분석 ROI입니다. 분석 ROI가 없으면 원본을 대신 분석하지 않습니다.

```json
{
  "original_image": "event_001_original.jpg",
  "analysis_roi": "event_001_analysis_roi.jpg",
  "roi_source": "tactile_block",
  "roi_fallback_used": false,
  "frame_quality_verified": true
}
```

## 명령행 실행

Edge 이벤트 JSON:

```bash
python -m ROADY_AI.server_damage_analysis.analyze_image \
  --model ROADY_AI/models/server/yolo26s_seg_multiclass_v4_best.pt \
  --event-json /path/to/pending/damage_event.json \
  --output output --device cpu --imgsz 768
```

개별 ROI:

```bash
python -m ROADY_AI.server_damage_analysis.analyze_image \
  --model ROADY_AI/models/server/yolo26s_seg_multiclass_v4_best.pt \
  --image analysis_roi.jpg \
  --input-metadata input_metadata.json \
  --policy ROADY_AI/server_damage_analysis/severity_policy.yaml \
  --output output --device cpu --imgsz 768
```

출력은 `analysis.json`, ROI 사본, Overlay, 통합 damage mask입니다. `units`와 `summary`가 현재 기준 응답이며 `regions`와 최상위 `analysis`는 deprecated 호환 필드입니다.

## 면적·심각도 정책 평가

고정 Test에서 실제 운영 후처리를 포함한 면적 비율 MAE, 심각도 혼동행렬,
Macro-F1과 등급 경계 오차를 계산합니다. Test는 최종 보고에만 사용하고 모델이나
임계값 선택에는 Validation을 사용합니다.

```bash
python -m ROADY_AI.server_damage_analysis.evaluate_policy \
  --model ~/roady/models/server_yolo26s_seg_multiclass_v4/best.pt \
  --dataset ~/roady/datasets/server_edge_roi_multiclass_v4 \
  --split test --imgsz 768 --device 0 \
  --output ~/roady/runs/server_yolo26s_seg_multiclass_v4_policy_test
```

산출물은 `report.json`, `per_image.csv`, `severity_confusion.csv`와 이미지별
Overlay입니다. 결손의 예상 블록 영역을 만들 수 없는 경우는 0%로 바꾸지 않고
`not_estimable` 열에 집계합니다.

## FastAPI

Spring의 Redis Queue Worker가 호출하는 동기식 분석 API입니다. Redis Queue와 작업 상태는 Spring이 관리하며 AI 서버는 DB나 Redis에 직접 연결하지 않습니다.

```bash
uvicorn ROADY_AI.server_damage_analysis.api:app \
  --host 0.0.0.0 --port 8000 --workers 1
```

- `POST /analyze`: Spring multipart 요청 분석
- `GET /health/live`: 프로세스 생존 확인
- `GET /health/ready`: 모델 로딩 완료 확인
- `GET /model-info`: 실제 모델 파일명, SHA-256, 클래스 매핑 확인

`POST /analyze`는 기존 `damageId`, `images` 필드와 함께 선택적인 `analysisMetadata` JSON을 받습니다.
이미지가 한 장이면 객체 하나, 여러 장이면 이미지 순서와 개수가 같은 배열을 전달합니다. camelCase와
snake_case를 모두 허용하며 메타데이터가 없으면 기존 Spring 요청과 동일하게 품질 미검증 ROI로 처리합니다.

```json
{
  "originalImage": "event_001_original.jpg",
  "analysisRoi": "event_001_analysis_roi.jpg",
  "roiSource": "tactile_block",
  "roiFallbackUsed": false,
  "frameQualityVerified": true,
  "edgeDamageCandidateDetected": true
}
```

Spring은 최상위 요약 필드를 구조화된 컬럼으로 파싱하고 응답 전체를 `raw_result`에 보존합니다. 비율 계산이 불가능해도 `regions.damage.pixels`가 있으면 픽셀 수로 `damage_score`를 계산합니다.

| 통합 파손 마스크 픽셀 수 | `damage_score` |
| ---: | ---: |
| 0 | 0 |
| 1~10,000 | 1~30 선형 변환 |
| 10,001~50,000 | 31~70 선형 변환 |
| 50,001~300,000 | 71~100 선형 변환 |
| 300,000 초과 | 100 |

구간은 `ROADY_AI_SCORE_MINOR_MAX_PIXELS`, `ROADY_AI_SCORE_MODERATE_MAX_PIXELS`,
`ROADY_AI_SCORE_MAX_PIXELS` 환경변수로 조정할 수 있습니다. 세 값은 반드시 오름차순이어야 합니다.

`damaged`는 `damage_score > 0`이 아니라 `summary.damage_detected`를 사용합니다. `missing`,
`crack`, `wear` 중 하나라도 유효하게 탐지되면 `true`입니다. 모델 클래스 계약이
유효하지 않아 판단할 수 없으면 `damaged=null`입니다. 최상위 `damage_type`은 대표 유형을
`SMALL_MISSING`, `LARGE_MISSING`, `CRACK`, `WEAR` 중 하나로 변환합니다. 결손 크기를 계산할 수
없으면 `missing / (tactile + missing) × 100`으로 fallback 비율을 계산하며, 이 값도 계산할 수
없을 때만 보수적으로 `LARGE_MISSING`을 사용합니다. 여러 유형이 탐지되면 유형별 탐지 픽셀에
`CRACK(40.0)`, `WEAR(40.0)`, `MISSING(1.0)` 가중치를 곱하고 가장 큰 값을 대표 유형으로
선정합니다. confidence는 대표 유형 선정에 사용하지 않습니다. 이 가중치는 대표 유형
선정에만 사용하며 `damage_score`와 `repair_priority`에는 영향을 주지 않습니다. 전체 유형별 탐지 결과는
`analysis_detail.units[].damage_types`에 그대로 보존합니다.

`damaged=true`이면 보류 상태가 되지 않도록 `repair_required=true`와 비어 있지 않은
`repair_priority`를 반환합니다. 보수 우선순위는 픽셀 기반 `damage_score`에 따라 1~30은
`LOW`, 31~70은 `NORMAL`, 71~100은 `HIGH`로 결정합니다. 모델이 비율 기반 심각도를
확정하지 못하더라도 이 서비스 판정은 적용하며, 원본 모델 판단과 검토 사유는
`analysis_detail.summary`에 그대로 보존합니다. `URGENT`는 자동 판정하지 않습니다.

```json
{
  "damaged": true,
  "damage_score": 40,
  "damage_type": "LARGE_MISSING",
  "repair_required": true,
  "repair_priority": "NORMAL",
  "confidence_score": 0.82,
  "analysis_detail": {
    "schema_version": "2.0",
    "damage_ratio": null,
    "damage_ratio_percent": null,
    "estimated_severity": "moderate",
    "review_required": true,
    "advisory_only": true
  }
}
```

외부 API 스키마는 [api_response.schema.json](api_response.schema.json), 상세 분석 스키마는 [response.schema.json](response.schema.json)을 참고합니다.

## Docker

```bash
docker build -f ROADY_AI/Dockerfile -t roady-ai:latest .
docker run --rm --gpus all -e ROADY_AI_DEVICE=0 -p 8000:8000 roady-ai:latest
```

CPU에서는 `ROADY_AI_DEVICE=cpu`를 사용합니다. GPU 서버는 시작 단계에서 모델 SHA-256 검증과 워밍업을 완료한 뒤 `/health/ready`를 활성화합니다.

## Overlay와 하위 호환성

- 검토 코드와 한국어 메시지는 [review_reasons.py](review_reasons.py) 한곳에서 관리합니다.
- Overlay는 tactile 외곽선과 missing/crack/wear mask, 분석 단위 ID, 비율, 심각도, 검토 여부를 표시합니다. 계산 불가 비율은 `ratio=?`입니다.
- 기존 이미지 저장 흐름과 `regions.damage`, `analysis.damage_ratio_percent`는 유지하지만 deprecated입니다.
- 기본 배포 모델은 5클래스 `yolo26s_seg_multiclass_v4_best.pt`입니다. `obstruction`은 점자블록 위 거치물을 검출해 파손 비율·심각도를 확정하지 않고 `OBSTRUCTION_SUSPECTED` 판단 보류로 전환합니다. 기존 v3 가중치는 롤백용으로 유지합니다. 필수 4클래스의 모델 메타데이터 매핑이 달라지면 자동 분석하지 않고 `MODEL_CLASS_MAPPING_INVALID`로 검토 전환합니다.
- 모델 품질 게이트 미달 시 `MODEL_QUALITY_GATE_NOT_MET` 검토 사유가 추가됩니다.
- `quality`는 모델 검증 성능을 보여주는 참고 정보이며 이미지별 판정이나 `review_required`를 변경하지 않습니다.

## 테스트

PowerShell:

```powershell
$env:PYTHONPATH='Roady_Embedded/src/perception;Roady_Embedded/src/storage'
& 'python' -m pytest `
  ROADY_AI/tests `
  Roady_Embedded/src/perception/test `
  Roady_Embedded/src/storage/test -q
```

Linux/Jupyter:

```bash
export PYTHONPATH="Roady_Embedded/src/perception:Roady_Embedded/src/storage"
python -m pytest ROADY_AI/tests \
  Roady_Embedded/src/perception/test \
  Roady_Embedded/src/storage/test -q
```

필수 테스트 패키지는 `pytest`, `numpy`, `opencv-python`, `PyYAML`입니다. 실제 모델 추론에는 `ultralytics`와 대응 PyTorch 환경이 추가로 필요합니다.
