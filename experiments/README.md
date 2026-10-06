# 개인 AI 실험 코드

- `data_and_edge/`: 로컬 dataset audit, builder, hard example mining, Edge v4 replay 평가 코드.
- `server_v5/`: 최종 배포 v4 이후 진행한 로컬 v5 replay·비교 실험 코드. 데이터와 weight는 없음.
- `legacy_pre_final/`: 미병합 초기 Server mask/severity 분석 코드. 현재 운영 API와 호환되지 않을 수 있음.

이 도구는 원래 삼성 노트북의 절대 데이터 경로와 당시 환경을 전제로 작성됐다. 공개 후보에서는 경로/주소를 익명화했으며, 실제 실행 전 사용자 데이터 경로와 의존성을 설정해야 한다. 최종 성능이나 배포 코드는 `ROADY_AI/`와 루트 README를 기준으로 한다.
