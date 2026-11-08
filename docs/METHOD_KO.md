# 수식이 실제 파일에서 어떻게 작동하는가

## 1. 먼저 함수의 입력과 출력을 구분한다

`x`는 **자기 관측과 공개 이력의 표현**, `a`는 **13개 손패 슬롯 중 합법 카드 하나**다. 환경의 모든 상태 `s`나 상대의 숨은 손패를 뜻하지 않는다.

| 함수 | 출력 의미 | 선택 방향 |
|---|---|---|
| Teacher `Cw(x,a)` | 이 행동 이후의 예상 유효 벌점 | 작을수록 좋음 |
| Student actor `πθ(a\|x)` | 합법 행동 선택 확률 | 확률로 sampling |
| Student critic `Vφ(x)` | 현재 학생 정책의 예상 reward return | 보상은 벌점의 음수 |

Teacher의 카드별 비용과 student의 카드별 logit은 서로 다른 숫자다. 최종 제출 때 host는 logit을 mask한 뒤 argmax를 사용한다. 학습 중 PPO는 합법 행동 분포에서 sampling한다.

## 2. 선형이라고 해서 단순한 조건만 표현하는 것은 아니다

기본 특징이 `Q♠ 보유`, `낮은 스페이드 수`, `선도 여부`라면 둘의 곱도 별도 특징으로 만든다.

\[
C_w(x,a)=w^T\phi(x,a)
\]

선형은 **가중치 w에 대해 선형**이라는 의미다. 특징 자체에는 AND 같은 복합 조건을 넣을 수 있다. 이 코드에서는 30개 기본 특징과 bias를 제외한 29개 중 두 개 조합 406개, 총 436개로 작게 시작한다. 연속 특징의 곱은 수치 상호작용이고, 두 이진 특징의 곱은 AND다.

현재 정보로 확실하게 계산할 수 있는 것만 쓴다. 상대가 클럽을 따르지 못한 사실, 공개된 벌점의 소유자 수 등은 포함하지만 상대가 실제 가진 카드 목록은 넣지 않는다.

## 3. 부분 과제 네 개와 특징 통합

각 deal 종료에서 다음 target을 얻는다.

1. 내가 가져온 하트 수 / 13
2. 내가 가져온 Q♠ 비용 / 13: 0 또는 1
3. 자기 문샷: 0 또는 1
4. 상대 중 한 명의 문샷: 0 또는 1

부분 모델들은 **공통의 규칙 행동 정책 아래 종료 target을 예측하는 TD 학습**을 한다. 네 정책을 따로 최적화해 투표하는 구현이 아니다. 문샷 head의 1은 예측 정답이며 PPO에 +1 보상을 더하는 것이 아니다.

`components`는 네 가중치 벡터를 저장한다. `select`는 각 과제에서 `|w_j| × RMS(φ_j)`가 큰 활성 조합을 골라 **특징 함수 목록의 합집합**을 만든다. 기본 30개는 유지한다. 과제당 기본 32개 조합이므로 최종 특징은 중복 여부에 따라 30–158개다. 이 순위는 초기 선별용 지표이고 인과적 중요도나 최종 게임 성능을 보장하지 않는다.

그 다음 `teacher`는 **새 가중치를 0으로 초기화**하고 전체 유효 벌점 `d`를 target으로 다시 학습한다. 네 가중치를 더하거나 평균하지 않는다.

문샷 성공이 거의 없는 데이터에서는 관련 head가 유용한 특징을 얻지 못할 수 있다. component metadata의 `raw_target_sums`에서 사례 수를 확인해야 한다. 현재 코드는 희귀 사례용 curriculum을 추가하지 않았다.

## 4. Teacher: 비용 표기의 Sarsa(λ)

초기 설정은 γ=1, λ=.75다. 중간 비용은 0, 종료 비용은 `d`다.

\[
\delta_t^C=c_{t+1}+\gamma m_t C_w(x_{t+1},a_{t+1})-C_w(x_t,a_t)
\]
\[
e_t=\gamma\lambda e_{t-1}+\phi(x_t,a_t),\qquad
w\leftarrow w+\alpha_{\mathrm{eff}}\delta_t^C e_t
\]

`a(t+1)`은 **실제 다음에 선택할 행동**이다. 비용을 최소화하는 ε-greedy로 수집하므로 on-policy Sarsa다. 여기서 `max`나 `min`으로 다음 값을 직접 선택하는 Q-learning target과 구분한다.

코드의 학습률은 `αeff = α / max(1, φᵀφ)`로 활성 특징의 크기에 맞춰 줄인다. deal 종료에서는 bootstrap과 trace를 모두 끊는다. 관측 요약의 함수 근사이므로 수렴을 보장하는 설계라고 주장하지 않는다.

예상 비용이 카드 A=7, B=3, C=5라면 frozen teacher는 B를 고른다. `C`는 비용이라 **argmin**이고, reward를 최대화하는 표현은 `Q=-C`로 바꾸면 된다.

## 5. 선형 teacher에서 attention 학생으로 무엇을 옮기나?

teacher 가중치를 attention에 복사하지 않는다. 고정한 teacher가 플레이하며 선택한 `(x, aT)`를 저장한다.

\[
L_{BC}=-\mathbb{E}_{(x,a_T)}[\log\pi_\theta(a_T\mid x)]
\]

B가 정답일 때 학생이 B의 확률을 .2로 주면 loss 1.609, .6이면 .511이다. `aT`는 teacher의 선택이지 최적 행동이라는 뜻이 아니다. Critic 출력층은 BC target에 포함하지 않고 PPO의 value loss로 학습한다.

## 6. 왜 작은 attention인가?

손패는 집합이다. 카드 순서를 바꾸면 출력도 해당 카드 순서대로 바뀌어야 한다. 그래서 모든 카드에 **같은 encoder와 actor 함수**를 쓴다. 손패 슬롯 자체의 embedding은 없다.

공개 이력은 13개 완료 trick과 현재 trick, 총 56개 카드 위치다. 각 token에 문양·랭크·상대 좌석·trick 내 순서·trick 번호를 넣는다. 카드 query가 이 이력과 context token을 참조하는 cross-attention이다. 이력에 카드가 하나도 없을 때도 context token 하나는 남아 attention이 비정상 값이 되는 일을 막는다.

정확히 계산 가능한 void 정보는 직접 제공한다. Attention이 부분 관측 문제를 완전 관측으로 바꾸는 것은 아니다. 호출 사이에 RNN hidden state를 저장하지 않는다.

## 7. PPO: 이번에는 teacher가 아니라 학생 자신의 경험

매 iteration에서 현재 학생 정책을 고정해 새 rollout을 수집한다. 행동, 보상, 당시 log probability, 당시 value를 저장한다. 이후 정해진 여러 epoch만 업데이트하고 자료를 버린다.

\[
\delta_t=r_{t+1}+\gamma m_tV(x_{t+1})-V(x_t),\quad
A_t=\delta_t+\gamma\lambda m_t A_{t+1}
\]
\[
\rho_t=\frac{\pi_\theta(a_t\mid x_t)}{\pi_{old}(a_t\mid x_t)},\quad
L_{actor}=-\mathbb{E}[\min(\rho_tA_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)]
\]

여기에 value 오차와 entropy를 결합한다. `λ=.95`는 GAE의 초기값, PPO clip의 `ε=.2`는 **teacher의 ε-greedy 탐험 확률 .1과 다른 값**이다. 일반적인 gradient clipping도 적용한다. KL anchor나 다른 논문의 특수 전략은 넣지 않았다.

## 8. 가장 중요한 경계: deal 종료와 버퍼 종료

- deal 종료: 보상이 0이어도 `deal_completed=True`. 다음 deal의 value를 더하지 않는다.
- rollout 256개 수집 종료: 아직 deal 안이면 마지막 `next_value`를 사용한다.
- match 종료: 환경을 새 match로 초기화하되, 이미 종료한 deal의 target에는 영향을 주지 않는다.

`info.discount`는 match 기준일 수 있으므로 이 코드의 deal 학습에서는 `info.deal_completed`를 직접 사용한다. PASS는 고정 규칙으로 건너뛰고 모든 학습 전이는 PLAY 결정 사이에 만든다.

## 9. 파일의 의미

- `selected.json`: 어떤 특징 함수를 계산할 것인가.
- `arrays.npz`: 선형 w 또는 attention의 행렬·bias. pickle을 사용하지 않는다.
- `metadata.json`: 단계, 설정, 특징 이름, 학습 수행 여부.
- `entry.bin`: 공식 host가 읽는 직렬화 계산 그래프. 학습 재개용 파일이 아니다.

현재 저장 방식은 단계 간 전달용이다. optimizer와 환경 상태까지 저장하는 정확한 중단·재개 기능은 없다. 실제 학습 전 작은 검증 실행으로 업데이트와 저장 파일을 추가 확인해야 한다.
