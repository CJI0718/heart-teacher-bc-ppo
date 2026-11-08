# 실행 순서

이 문서는 **향후 사용 명령**이다. 저장소 구축 시 학습을 실행하지 않았다. 아래 학습 명령들은 `--execute-training`이 없어서 계획만 출력한다.

설치 후 가상환경을 활성화하거나 `.venv/Scripts/heart-study.exe`를 사용한다.

```text
heart-study plan
heart-study components --deals 2000 --output runs/components
heart-study select --checkpoint runs/components --output runs/selected.json
heart-study teacher --features runs/selected.json --deals 2000 --output runs/teacher
heart-study demos --teacher runs/teacher --deals 500 --output runs/demos
heart-study bc --data runs/demos --epochs 10 --output runs/bc
heart-study ppo --checkpoint runs/bc --iterations 100 --output runs/ppo
heart-study export --checkpoint runs/ppo --output submissions/my-entry
```

`select`, `demos`, `export`는 실제 입력 파일이 있어야 한다. 따라서 아직 학습하지 않은 상태에서 위 블록을 통째로 실행하지 않는다. 실제 학습을 결정한 뒤 **components, teacher, bc, ppo의 네 명령에만** `--execute-training`을 추가하고 순서대로 실행한다. 각 명령의 출력 폴더는 새 경로를 사용한다.

| 단계 | 입력 | 저장되는 것 |
|---|---|---|
| components | 공식 환경, 고정 규칙 정책 | 4개 선형 예측기, 특징 활성도, 종료 사건 수 |
| select | 학습된 components | 선택 특징 목록, 과제별 선택 index |
| teacher | 선택 특징, 새 환경 경험 | 전체 유효 비용의 선형 Sarsa 정책 |
| demos | 고정한 학습 teacher | encoded 관측·특징·선택 행동 |
| bc | teacher 행동 자료 | attention 학생의 가중치 |
| ppo | BC 학생 | 새 경험으로 갱신한 actor–critic 가중치 |
| export | 학생 가중치 | 공식 제출 그래프와 검증 파일 |

처음 승인받아 실행할 때는 위 숫자 그대로 긴 학습을 시작하지 말고, 아주 작은 deal/epoch/iteration 수로 경계·유한값·손실·저장을 확인한다. 현재 수치는 튜닝된 설정이 아니다. 프로그램의 `train_*` 함수를 직접 호출하면 CLI의 실행 옵션을 거치지 않으므로, 학습을 원하지 않을 때 직접 호출하지 않는다.

## 반복 실행과 시간

JAX의 첫 실행에는 컴파일 비용이 든다. 이 저장소는 단일 환경을 Python 루프로 진행한다. 처음부터 Colab GPU가 CPU보다 빠르다고 가정하지 않는다. 처리량 측정·vectorized rollout·학습 재개 기능은 후속 구현 범위다.

## 평가

현재 `check`는 설치/계약 검사이며 경쟁 성능 평가기가 아니다. 학습한 뒤 규칙·선형 teacher·BC·PPO를 동일한 PASS, 상대, 좌석, seed 조건으로 평가하는 절차를 추가한다. 아직 승률이나 개선 결과를 보고하지 않는다.
