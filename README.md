# HEART: Linear Teacher → Attention BC → PPO

하트·스페이드 퀸·문샷의 부분 예측 과제에서 특징을 고른 뒤, **전체 유효 벌점을 최소화하는 선형 teacher**를 학습하고 그 행동을 **attention 정책에 BC로 이전**, 이후 **PPO로 개선**하는 연구용 구현입니다.

**현재 상태: 코드 구축 및 학습 없는 검증. 학습은 실행하지 않았습니다.** 학습된 가중치, 성능 곡선, 승률 결과는 제공하지 않습니다. 이 저장소는 공식 HEART의 제출작이나 원논문의 완전한 재현이 아닙니다.

## 구조

```text
공식 HEART 관측 (상대 손패 제외)
    │
    ├─ 관측 가능한 기본 특징 + 2개 특징 조합
    │      ↓
    │  하트 / 퀸 / 자기 문샷 / 상대 문샷: 선형 TD 예측
    │      ↓
    │  유용한 특징 선택 → 중복 제거한 합집합
    │      ↓
    │  전체 유효 벌점의 선형 Sarsa(λ) teacher
    │      ↓ 합법 카드별 비용 비교, argmin
    │  (관측, teacher 선택 행동) 자료
    │      ↓
    └─ 카드 공유 encoder + 공개 이력 attention
           ↓ BC: teacher의 행동을 지도 모방
           ↓ PPO: 새 경험으로 actor와 critic 개선
         공식 PASS / PLAY logits 출력
```

- **Teacher:** 비용 `C(x,a)=wᵀφ(x,a)`, 합법 행동 `argmin`. λ 초기값 0.75.
- **Attention:** 폭 32, head 2. 카드별 공유 encoder, 공개 이력 cross-attention, pooled critic. 손패 슬롯 위치 embedding 없음.
- **BC:** 학습한 통합 teacher의 행동 label로 cross-entropy. 선형 가중치를 attention 가중치에 복사하지 않음.
- **PPO:** on-policy PPO-Clip, TD 기반 GAE(초기 λ=0.95), actor/value/entropy loss, gradient clipping.
- **환경:** 제어 좌석 0, 고정 easy/medium/hard 상대, PLAY만 학습. PASS는 모든 단계에서 동일한 고정 규칙.
- **보상:** 공식 deal 종료 보상 `−effective penalty`, γ=1. 추가 보상은 없음.

KL anchor, curriculum, 과거 정책 상대 풀, self-play league, best-checkpoint 실험은 포함하지 않았습니다. 단순한 단계별 저장·불러오기는 구현에 필요한 기본 기능입니다.

## 설치

Python **3.12**, JAX CPU 기준입니다. Windows PowerShell 예:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,export]"
$env:JAX_PLATFORMS = 'cpu'
.venv\Scripts\heart-study.exe plan
```

공식 환경은 [funny-rl/Heart](https://github.com/funny-rl/Heart)의 commit `ad122dba1e1d4c0aaee9bee8846ceba60ed680fe`에 고정합니다. Git 설치가 필요합니다. 버전을 임의로 올릴 때는 관측·슬롯·deal 경계 계약을 다시 검증해야 합니다. GPU 설치와 대규모 병렬화는 아직 다루지 않습니다.

## 학습하지 않고 확인하기

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\ruff.exe check src tests tools
.venv\Scripts\heart-study.exe check --report verification/local_smoke.json
```

테스트는 수식, 합법 mask, 카드 순열, 관측 인코딩, 손실과 gradient의 유한성 등을 확인합니다. **optimizer step이나 Sarsa 가중치 갱신을 실행하지 않습니다.** `check`는 무작위 초기 가중치를 고정한 채 한 deal을 실행하는 인터페이스 검사입니다. 얻는 점수는 학습 성능 결과가 아닙니다.

```powershell
# 이 명령도 학습을 시작하지 않고 계획만 출력합니다.
.venv\Scripts\heart-study.exe teacher --features runs/selected.json --output runs/teacher
```

학습 관련 네 명령(`components`, `teacher`, `bc`, `ppo`)은 **`--execute-training`을 명시해야** 실제 가중치를 갱신합니다. 이번 구축 과정에서는 이 옵션으로 실행하지 않았습니다.

## 나중에 학습을 승인한 뒤의 순서

실행 순서와 입력/출력은 [실행 안내](docs/RUNBOOK_KO.md)에 있습니다. 현재 저장소에는 학습된 teacher가 없으므로 `demos`, BC, PPO를 바로 실행할 수는 없습니다. 먼저 부분 과제와 통합 teacher를 학습해야 합니다.

## 코드와 공부 순서

| 파일 | 역할 |
|---|---|
| [`observations.py`](src/heart_study/observations.py) | 공개 이력의 카드·좌석·순서, 보유 카드, 공개 void 정보 |
| [`features.py`](src/heart_study/features.py) | 기본 30개 + pairwise 406개 = 후보 436개, 특징 선택·통합 |
| [`teacher.py`](src/heart_study/teacher.py) | 선형 비용, 합법 argmin, Sarsa(λ) 수식 |
| [`model.py`](src/heart_study/model.py) | 작은 attention actor–critic |
| [`losses.py`](src/heart_study/losses.py) | BC, GAE, PPO 손실 |
| [`environment.py`](src/heart_study/environment.py) | 공식 환경 연결, 고정 PASS, 종료 target |
| [`training.py`](src/heart_study/training.py) | 부분 학습 → 선택 → teacher → demos → BC → PPO 실행 루프 |
| [`checkpoints.py`](src/heart_study/checkpoints.py) | 배열과 설정을 분리한 저장/불러오기 |
| [`export.py`](src/heart_study/export.py) | 공식 제출용 두 head 출력 |

처음에는 [코드·수식 해설](docs/METHOD_KO.md)을 읽고 `teacher.py`, `losses.py`, 마지막으로 `training.py`를 보는 순서를 권합니다.

## 검증 범위와 남은 작업

- 환경·관측·수식·forward·미분 계산을 검증합니다.
- 무작위 초기 모델의 제출 그래프도 검사하며, 이 파일은 공개 저장소에 넣지 않습니다.
- **실제 optimizer 업데이트와 전체 학습 루프는 실행 검증 전**입니다. 수렴, 시간, 성능 개선을 주장하지 않습니다.
- 구현은 읽기 쉬운 **단일 환경 참조 루프**입니다. JAX를 쓴다는 것만으로 GPU 병렬학습이 구현된 것은 아닙니다.
- 특징 선택은 현재 `|가중치| × RMS 활성도`를 이용한 후보 선별입니다. 특징 제거 후 재학습 검증은 후속 작업이며 구현 완료로 표현하지 않습니다.
- 저장 파일은 단계 간 전달용 가중치입니다. optimizer·환경 상태까지 포함하는 정확한 학습 재개 checkpoint가 아닙니다.

검증 내역은 [`verification/README.md`](verification/README.md), 출처와 원논문에서 달라진 부분은 [`docs/SOURCES.md`](docs/SOURCES.md)에 있습니다.
