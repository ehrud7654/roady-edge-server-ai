# Jetson 전달용 YOLO26n 변환 안내

## 모델 사양

- 모델: `YOLO26n Detect`
- 클래스: `tactile_block`, `damage_candidate`
- 입력: `batch=1`, `768x768`
- 기본 운영: 별도 손상 분류기 없이 단일 Detect 모델 사용
- PyTorch 가중치: `ROADY_AI/models/edge/tactile_damage_candidate_yolo26n_best.pt`

## Jetson에서 ONNX → TensorRT FP16 변환

TensorRT 엔진은 JetPack, CUDA, TensorRT 및 Jetson 하드웨어에 종속되므로 Jetson Orin Nano에서 직접 생성한다.

```bash
python ROADY_AI/tactile_damage_detect/export_jetson.py \
  --weights ROADY_AI/models/edge/tactile_damage_candidate_yolo26n_best.pt \
  --imgsz 768 \
  --workspace-mib 2048 \
  --output ROADY_AI/models/edge/tactile_damage_candidate_yolo26n_fp16.engine
```

스크립트는 고정 입력 ONNX를 생성한 뒤 `trtexec --fp16`으로 TensorRT 엔진을 생성한다. 생성된 `.engine`은 Git에 올리지 않는다.

## 임베디드 팀 전달 사항

- 모델 경로: `ROADY_AI/models/edge/tactile_damage_candidate_yolo26n_best.pt`
- TensorRT 엔진 권장명: `tactile_damage_candidate_yolo26n_fp16.engine`
- 입력 크기: `768`
- 클래스 0: `tactile_block`
- 클래스 1: `damage_candidate`
- TensorRT 출력 호환 처리와 ROS 파라미터 연결은 임베디드 담당 범위

## Jetson 검증 기준

- FP16 엔진 생성 및 로드 성공
- `damage_candidate` 좌표와 신뢰도 정상 출력
- 768 입력 기준 15 FPS 이상
- 전체 로봇 프로세스 포함 메모리 사용량 4 GB 이하
- PyTorch 대비 고정 검증 영상 이벤트 Recall 하락 2%p 이내
