# 코드 출처와 기여 경계

이 후보의 `ROADY_AI/`는 GitLab `origin/master`의 현재 파일에서 추출했다. 모델·데이터·사진은 제외하고 주소·이메일은 익명화했다. 원본 Git history는 복사하지 않았다. 황도경은 팀 내 AI 단독 담당자였지만 AI 경로의 모든 변경을 혼자 작성하지 않았다.

| 코드 영역 | 근거와 표시 방법 |
|---|---|
| Edge `tactile_damage_detect/` | 황도경의 `e45d809`, `9b7e229`, `b741f5f` 등. 학습/모델 준비/평가 중심 개인 기여. ROS 최종 런타임은 팀 연동. |
| Server `server_damage_analysis/analyzer.py`, `expected_region.py`, `policy.py` | `48ca382`, `691069c`, `edf2624`, `40a77be` 등 AI 모델·mask/검토 정책 기여. 팀원 후속 수정이 있으므로 현 파일 전체를 단독 작성으로 주장하지 않음. |
| Server `api.py`, `service.py`, Docker/API 연결 | 팀원 `5556a69` 및 후속 커밋의 FastAPI wrapper·점수/대표 유형 수정 포함. 협업 구현으로 표시. |
| `experiments/data_and_edge/` | `Desktop/ROADY/ai` 로컬 AI 작업공간에서 선별. Git 추적 이력은 없으며, 파일 경로와 SHA-256을 Mac 이전 자료에서 보존. 개인 실험으로 소개하되 최종 운영 코드 아님. |
| `experiments/server_v5/` | 로컬 v5 후속 실험. 최종 GitLab 모델은 v4이며 동일 Test 비교가 없으므로 성능 우월성 주장 없음. |
| `experiments/legacy_pre_final/` | 미병합 `feat/roady-ai`의 `4497e07` 등에 대응하는 초기 서버 분석. 최종 코드에 다른 형태로 발전했으나 해당 파일은 final에 없음. |
| Backend/Embedded/Frontend | 이 후보에 전체 소스를 포함하지 않음. Spring/Redis, ROS/로봇, Vue 화면은 해당 담당자의 팀 결과물. |

`feat/roady-ai` 4커밋의 분류: `a140b31`의 장애물/손상 분류 초기 구현은 후속 AI 코드에 일부 반영됐고 손상 분류기는 final에서 제외됐다. `4497e07`의 초기 mask·severity 구현은 final의 Server 분석 정책과 별개로 남아 archive 가치가 있다. `29ac38a`의 데이터 빌더·검수 도구 상당수는 final에 없는 로컬 실험이다. `68ac238`의 README는 현재 final 문서로 대체되어 중복 복사하지 않았다. 미병합 실험의 파일을 final 경로에 섞지 않았다.

정확한 개인 commit 해시·일시·파일 목록은 별도 보존 자료 `ROADY_PRESERVATION/PERSONAL_COMMIT_EVIDENCE.md`를 참조한다. 이 공개 후보에는 내부 GitLab URL이나 전체 이력을 포함하지 않는다.
