# ROADY 사람 다리·발 장애물 탐지 모델

Jetson Orin Nano 8GB에 연결된 장애물용 USB 카메라에서 사람의 다리와 발을 탐지하기 위한 YOLO11n Detect 프로젝트다.

## 현재 상태

- 모델 구조: YOLO11n Detect
- 학습 클래스: `foot`, `lower_leg` 2개
- 로봇 출력: 둘 중 하나라도 탐지되면 `human_lower_limb_detected=true`
- 입력 크기: 640
- 배포 정밀도: TensorRT FP16 우선
- 현재 전용 데이터셋: COCO Body Part의 `Foot`, `Lower Leg` 변환본 + Hard Example 보강본
- 현재 전용 학습 가중치: `ROADY_AI/models/edge/obstacle_lower_limb_yolo11n_best.pt`

Ultralytics의 COCO 사전학습 `yolo11n.pt`를 초기 가중치로 사용해 COCO Body Part의 `Foot`, `Lower Leg` 클래스만 파인튜닝했다. 기본 가중치는 필요할 때 자동 다운로드하므로 저장소에 포함하지 않는다.

## 라벨 기준

공개 COCO Body Part의 원래 부위 박스를 유지한다.

- 클래스 `0`: `foot`
- 클래스 `1`: `lower_leg`
- 발과 종아리 박스를 억지로 하나의 큰 박스로 합치지 않는다.
- 여러 사람과 여러 신체 부위는 각각 라벨링한다.
- 발만 보이거나 하체 일부만 보이는 경우도 양성으로 라벨링한다.
- 신발 종류, 긴 바지, 반바지, 치마 아래 하체, 맨발을 다양하게 포함한다.
- 의자·책상 다리, 기둥, 마네킹 하체, 그림자, 바닥 무늬는 음성 이미지로 포함한다.
- 같은 연속 영상의 인접 프레임이 train/val/test에 나뉘지 않도록 주행 클립 단위로 분리한다.

## 데이터셋 구조

```text
ROADY_AI/datasets/obstacle_lower_limb_v1/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ labels/
   ├─ train/
   ├─ val/
   └─ test/
```

음성 이미지에도 같은 stem의 빈 `.txt` 라벨 파일을 만든다.

## 데이터 수집 최소 조건

초기 학습 권장량은 전용 USB 카메라에서 추출한 서로 다른 장면 기준이다.

- 양성 이미지: 최소 1,500장
- 음성/Hard Negative 이미지: 최소 1,500장
- 사람: 최소 20명 이상
- 거리: 근거리·중거리·최대 요구 거리
- 방향: 정면, 후면, 측면, 횡단
- 상태: 정지, 보행, 부분 가림, 복수 인원
- 환경: 실내외, 역광, 그림자, 밝기 변화

수량보다 실제 로봇의 카메라 높이·각도·해상도와 동일한 데이터가 우선이다.

## USB 카메라 데이터 수집

카메라를 로봇의 실제 장착 위치와 각도로 고정한 뒤 실행한다.

```bash
python3 ROADY_AI/obstacle_lower_limb/capture_frames.py \
  --camera 0 \
  --output collected/obstacle_scene_01 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --scene-id indoor_corridor_01
```

- `S`: 현재 프레임 저장
- `Q` 또는 `Esc`: 종료
- 자동 수집이 필요하면 `--interval-sec 0.5`처럼 간격을 지정한다.
- 연속 프레임을 과도하게 저장하지 말고 사람, 거리, 방향, 배경과 조명 조건을 바꿔 수집한다.
- 장면 또는 연속 클립 ID를 파일명에 유지해 train/val/test 누수를 방지한다.

## 데이터 검증

```powershell
.\ai\.venv\Scripts\python.exe .\ai\obstacle_lower_limb\validate_dataset.py `
  --dataset .\ai\datasets\obstacle_lower_limb_v1 `
  --report .\reports\obstacle_lower_limb_dataset_v1.json `
  --decode-all
```

## 학습

```powershell
.\ai\.venv\Scripts\python.exe .\ai\obstacle_lower_limb\train.py
```

결과 가중치:

```text
ROADY_AI/models/edge/obstacle_lower_limb_yolo11n_best.pt
```

## 평가 게이트

전체 mAP만으로 배포하지 않는다.

- 사람 다리·발 Recall 우선
- 거리 구간별 Recall
- 정면·측면·부분 가림·복수 인원 Recall
- 의자·책상 다리, 기둥, 그림자 오탐률
- 오탐 감속 횟수/km
- Jetson end-to-end latency P95
- 30분 지속 실행 온도와 throttling

정확한 통과 기준은 로봇 속도, 탐지 거리와 LiDAR 정책을 확정한 뒤 수치화한다.

## Jetson TensorRT FP16 export

TensorRT 엔진은 반드시 실제 Jetson Orin Nano의 JetPack/TensorRT 환경에서 생성한다.

```bash
python3 ROADY_AI/obstacle_lower_limb/export_jetson.py \
  --weights ROADY_AI/models/edge/obstacle_lower_limb_yolo11n_best.pt
```

Windows나 다른 GPU에서 생성한 `.engine`을 Jetson 배포물로 사용하지 않는다.

## USB 카메라 시험

```bash
python3 ROADY_AI/obstacle_lower_limb/camera_test.py \
  --model ROADY_AI/models/edge/obstacle_lower_limb_yolo11n_best.engine \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --display
```

화면이 없는 서비스 환경에서는 `--display`를 제거한다.

## 하드웨어 팀 인계 파일

최종 인계 패키지는 다음을 포함한다.

```text
obstacle_lower_limb_handoff/
├─ best.pt
├─ best.engine
├─ dataset.yaml
├─ camera_test.py
├─ capture_frames.py
├─ metrics.json
├─ test_predictions/
└─ README.md
```

`best.engine`은 하드웨어 팀의 대상 Jetson에서 export하거나 동일 JetPack 환경에서 생성한다.
