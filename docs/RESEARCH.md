# 연구 배경 및 이론

Weighted Multi-Token Prediction (WMTP)의 이론적 배경 및 실험 설계.

---

## 핵심 가설

**"Not All Tokens Are What You Need"**

표준 MTP는 모든 미래 토큰을 균등 가중으로 다루어, 쉬운/비핵심 토큰에도 동일한 학습 자원을 배분한다. 중요 토큰에 계산을 집중하는 WMTP는 동일 FLOPs에서 더 높은 성능과 안정적 수렴을 달성한다.

---

## Related Work

WMTP는 Advantage-Weighted Regression을 MTP에 확장한 방법이다. 관련 연구를 방법론적 유사성 기준으로 분류한다.

### 1. Advantage-Weighted Methods

WMTP의 직접적인 이론적 기반이 되는 방법론들이다.

**AWR (Advantage-Weighted Regression)** (Peng et al., 2019)은 off-policy RL을 weighted supervised learning으로 변환한 원조 방법이다:

```
L_AWR = E[exp(A(s,a) / β) · log π(a|s)]
```

Advantage를 지수 변환하여 가중치로 사용, cross-entropy loss에 적용한다. WMTP는 이 원리를 MTP에 확장했다.

**APA (Advantage-Induced Policy Alignment)** (Zhu et al., 2023)는 AWR을 LLM RLHF에 적용했다:

```
π*(a|s) ∝ π_init(a|s) · exp(A/β)           (Target policy)
L_APA = E[(log π_θ - log π_init + A/β)²]   (Squared error)
```

AWR과 달리 squared error를 사용하여 학습 안정성을 높였다. 단, online RLHF 환경에서 동작한다.

### 2. Implicit Q-Learning Methods

Value를 명시적으로 학습하지 않고 policy 자체에서 Q-value를 추출하는 방법들이다.

**Q-SFT** (Hong et al., 2024, ICLR 2025)는 token probability를 Q-value로 직접 해석한다:

```
Q(s,a) ≈ p_θ(a|s) / π_β(a|s)                    (Implicit Q)
weight = r + γ · max Q(s',a')                    (Bellman target)
L = weight · log p_θ(a|s)                        (Weighted SFT)
```

Q value 생성 시 별도 Value Model 없이 단일 모델로 동작한다.

### 3. Value-Based Offline RL

별도의 Value function을 학습하여 policy 개선에 활용하는 방법들이다.

**ILQL** (Snell et al., 2022)은 IQL을 LLM에 확장했다. Q와 V를 joint training하고, inference 시 advantage로 logit perturbation한다:

```
logits_new = logits + β · (Q(s,a) - V(s))       (Inference-time)
```

Inference마다 Q, V forward가 필요하여 추론 비용이 발생한다.

**OREO** (Hao et al., 2024)는 Soft Actor-Critic 스타일의 policy gradient로 학습한다:

```
∇L_π ∝ ∇log π(a|s) · [Q(s,a) - β·log π(a|s)]   (Policy Gradient)
```

Policy gradient 기반이므로 weighted SFT 대비 학습이 불안정할 수 있다.

### 4. Multi-Token Prediction

**Meta MTP** (Gloeckle et al., 2024)는 n개의 독립 head로 미래 토큰을 예측한다:

```
L_MTP = (1/n) Σ_{k=1}^{n} CE(head_k, y_{t+k})
```

논문에서 "MTP가 암묵적으로 consequential tokens에 높은 가중치를 부여한다"고 언급했으나, 명시적 가중화는 시도되지 않았다.

### 연구 위치 및 비교

| 방법 | Loss 형태 | 가중치 | Value 학습 | 별도 모델 | MTP |
|------|----------|--------|-----------|----------|-----|
| AWR | Weighted CE | `exp(A/β)` | Joint | X | X |
| APA | Squared Error | `exp(A/β)` | Online RM | X | X |
| Q-SFT | Weighted CE | Bellman target | Implicit | X | X |
| ILQL | - | - | Joint | X | X |
| OREO | Policy Gradient | - | Joint | X | X |
| **WMTP** | Weighted CE | `exp(GAE/β)` | 독립→frozen | O | **O** |

### WMTP의 기여

| 기여 | 대비 논문 | 설명 |
|------|----------|------|
| **MTP 확장** | AWR, APA, Q-SFT | 기존 방법들은 single-token prediction만 지원 |
| **Weighted SFT** | OREO | Policy gradient 대비 학습 안정성 |
| **Decoupled Training** | AWR, ILQL, OREO | Value를 독립 학습 후 frozen 사용 |
| **Training-Time** | ILQL | Inference-time perturbation 대비 추론 비용 제로 |
| **Explicit Value** | Q-SFT | Implicit Q 대비 해석 가능성, 디버깅 용이 |
| **Pairwise Ranking** | AWR, Q-SFT | 오답 데이터를 explicit하게 활용 |

**한 줄 요약**: WMTP = AWR의 `exp(A/β)·CE` + Decoupled Value + MTP 확장 + Pairwise Ranking

---

## 이론적 배경

WMTP는 세 가지 이론적 구성 요소를 결합한다.

### 1. AWR 원리의 MTP 확장

AWR의 `exp(A/β) · CE` 원리를 MTP의 n개 head에 적용한다:

```
L_AWR  = E[exp(A(s,a)/β) · log π(a|s)]           (단일 토큰)
L_WMTP = Σ_{t,k} exp(A_t/β) · CE(head_k, y_{t+k})  (MTP 확장)
```

가중치 `exp(A/β)`는 advantage가 높은 토큰에 더 큰 학습 신호를 부여한다.

### 2. GAE 기반 Advantage 계산

GAE (Schulman et al., 2015)로 토큰별 advantage를 추정한다:

```
δ_t = γV(s_{t+1}) - V(s_t)                    (TD error, r_t=0 for t<T)
A_t^GAE = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}          (역방향 누적)
```

| λ 값 | 특성 | Variance | Bias |
|------|------|----------|------|
| λ=0 | TD(0), 단기 | 낮음 | 높음 |
| λ=1 | Monte Carlo, 장기 | 높음 | 낮음 |
| **λ=0.95** | WMTP 설정 | 균형 | 균형 |

### 3. Pairwise Ranking으로 Value 학습

Bradley-Terry 모델로 정답/오답의 상대적 선호를 학습한다:

```
P(correct > incorrect) = σ(V_pos - V_neg)
L_pairwise = -log σ(V_pos - V_neg)
```

**장점**:
- 절대 스케일 불변 → 학습 안정성
- 오답 데이터 explicit 활용 → AWR/Q-SFT 대비 차별점

### WMTP 전체 수식

```
# Phase 1: Value Model 학습 (Pairwise Ranking)
L_value = -log σ(V(correct) - V(incorrect))

# Phase 2: Policy 학습 (AWR + GAE + MTP)
A_t = GAE(V, γ=1.0, λ=0.95)
w_t = exp(A_t / β)
L_policy = Σ_{t,k} w_t · CE(head_k, y_{t+k})
```

---

## 아키텍처 개요

```
weighted_mtp/
├── pipelines/
│   ├── run_baseline.py      # Baseline MTP (균등 가중치)
│   ├── run_critic.py        # Value Model 독립 학습 (Pairwise Ranking)
│   ├── run_verifiable.py    # Verifiable WMTP (GAE 가중화)
│   ├── run_ref_tuning.py    # Reference Model 도메인 적응 (Backup)
│   ├── run_rho1.py          # Rho-1 WMTP (Backup Plan)
│   └── run_evaluation.py    # Pass@K 평가
├── value_weighting/
│   ├── td_weighting.py      # GAE/TD error 기반 가중화
│   ├── td_stats_ema.py      # EMA 기반 통계 정규화
│   └── rho1_weighting.py    # Rho-1 selection (Backup)
├── data/
│   ├── datasets.py          # 메타데이터 기반 샘플링 (Pairwise, Length-balanced)
│   └── collators.py         # Alpaca 템플릿 적용 및 Loss Masking
└── utils/
    ├── loss_utils.py        # MTP CE Loss 계산
    └── pairwise_utils.py    # Pairwise Ranking Loss, λ-return
```

---

## 가중화 방식

### 방식 1: Baseline MTP (Control Group)

```
w_{t,k} = 1.0 (모든 k)
```

- **파이프라인**: `run_baseline.py`
- **데이터**: 정답만 학습
- **목적**: 비교 기준선
- **특징**: 표준 SFT, `compute_mtp_ce_loss_unweighted()` 사용

### 방식 2: Verifiable Critic WMTP (주요 방식)

데이터셋의 검증 가능한 레이블(is_correct)을 reward signal로 사용.
**독립 Value Model + GAE Advantage 기반 가중화** 구조.

#### Phase 1: Value Model 학습 (run_critic.py)

독립적인 Value Model을 **Pairwise Ranking Loss** 기반으로 학습.

```python
# Value Model: Sheared-LLaMA 2.7B + Value Head
# Policy Model과 분리하여 별도 학습

# Pairwise Ranking Loss (Bradley-Terry Model)
# P(correct > incorrect) = sigmoid(V_pos - V_neg)
# 정답/오답 쌍의 상대적 선호를 모델링
pairwise_loss = -log(sigmoid(V_pos_mean - V_neg_mean))

# Loss 계산 (loss_type="pairwise_ranking")
total_loss = pairwise_coef * pairwise_loss
```

**Pairwise Ranking의 장점**:
- **스케일 불변성**: 절대 Value 대신 상대 비교로 학습 안정성 확보
- **간결한 구조**: 복잡한 TD 타겟 계산 없이 직접적인 선호 학습
- **실험적 유효성**: Lambda Return 방식 대비 유의미한 성능 향상 확인

**특징**:
- **독립 모델**: Policy Model과 분리된 Value Model (Sheared-LLaMA 2.7B)
- **Pairwise Ranking**: 정답/오답 쌍 비교로 상대적 가치 학습
- **LoRA 학습**: Backbone frozen, LoRA adapter + Value Head만 학습

**OREO와의 차이점**:
- OREO: Policy와 Value를 Soft Bellman으로 joint training
- WMTP: Value를 독립 학습 후 frozen하여 weight 계산에만 사용

**Lambda Return (대안)**: `loss_type="lambda_return"` 설정 시 TD(λ) 기반 타겟으로 학습 가능하나, 실험 결과 Pairwise Ranking이 더 효과적이었음.

#### Phase 2: Weighted Policy Training (run_verifiable.py)

학습된 Value Model을 frozen 상태로 로드하여 GAE Advantage 기반 가중치 계산.

```python
# 1. Frozen Value Model 로드
value_model = load_critic_checkpoint(checkpoint_path)
value_model.eval()  # 평가 모드 고정

# 2. Value 추론 (gradient 차단)
with torch.no_grad():
    value_logits = value_model(input_ids, attention_mask)

# 3. GAE Advantage 계산
# δ_t = γV(t) - V(t-1)  (한계 가치)
# A_t = δ_t + γλ * A_{t+1}  (역방향 GAE)
advantages = compute_gae_advantage(
    value_logits=value_logits,
    rewards=rewards,
    loss_mask=loss_mask,
    gamma=1.0,
    lam=0.95,  # td_lambda
    initial_value=0.5,
)

# 4. Advantage Whitening + Exponential Weighting (AWR 원리)
# 표준화: A_norm = (A - mean) / (std + eps)
# 가중치: w = exp(A_norm / β)  ← IQL/AWR에서 차용
# EMA 정규화로 학습 안정성 확보
weights = build_weights(
    td_errors=advantages,
    loss_mask=loss_mask,
    beta=1.0,
    min_weight=0.1,
    max_weight=3.0,
    external_mean=ema_mean,
    external_std=ema_std,
)

# 5. Weighted MTP Loss
weighted_ce_loss = compute_mtp_ce_loss(
    logits=logits,
    labels=labels,
    attention_mask=attention_mask,
    weights=weights,
)
```

**특징**:
- **Frozen Value Model**: Critic에서 학습된 가중치 고정 사용
- **GAE Advantage**: 토큰별 한계 가치(marginal value) 기반 가중화
- **Advantage Whitening**: 스케일 불변성 확보
- **EMA 정규화**: 배치 간 통계 안정화
- **LoRA 학습**: 효율적인 파라미터 업데이트

**ILQL과의 차이점**:
- ILQL: Inference 시 `logits += β(Q - V)`로 perturbation
- WMTP: Training 시 `loss = Σ exp(A/β) · CE`로 가중화 → Inference 비용 없음

### 방식 3: Rho-1 WMTP (Backup Plan)

Verifiable Critic이 유의미한 결과를 보여 실제 실험 검증은 수행하지 않았으나, 대안 접근법으로 파이프라인을 유지한다.

**핵심 아이디어**: Reference Model과 Policy Model의 loss 차이 기반 토큰 선택

```
excess_loss = CE_policy - CE_reference
weights = TopK(excess_loss, k_percent)  # binary selection
```

**파이프라인**: `run_ref_tuning.py` → `run_rho1.py`

---

## 비교 테이블

| 특성 | Baseline | Verifiable Critic | Rho-1 (Backup) |
|------|----------|-------------------|----------------|
| **가중치 산출** | 상수 (1.0) | GAE Advantage → exp weighting | Excess loss Top-k |
| **Value 학습** | 없음 | 독립 모델 (Pairwise Ranking) | 없음 |
| **가중치 차원** | 없음 | [batch, seq] 2D | [batch, seq, n_future] 3D |
| **외부 모델** | 불필요 | Value Model (2.7B) | Reference Model (2.7B) |
| **데이터 요구** | 정답만 | 정답+오답 (Pairwise) | 정답만 |
| **학습 파이프라인** | 1단계 | 2단계 (Critic → Policy) | 2단계 (Ref → Policy) |
| **구현 복잡도** | 낮음 | 높음 | 중간 |
| **이론적 기반** | 표준 SFT | GAE + Pairwise Ranking + AWR | 정보 이론 |
| **Negative signal** | 미사용 | 활용 (Pairwise 비교) | 미사용 |

---

## 핵심 알고리즘

### GAE Advantage 계산 (td_weighting.py)

```python
def compute_gae_advantage(value_logits, rewards, loss_mask, gamma=1.0, lam=0.95):
    """GAE 기반 Advantage 계산

    수식:
        δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
        A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}  (GAE)

    Pairwise Ranking Value Model과 호환:
    - δ_t = γ*v_t - v_{t-1}로 통일 (한계 가치)
    - 외부 reward 대신 V 자체가 "정답 확률" 반영
    """
    # δ_t = γV(t) - V(t-1): 토큰 t의 한계 가치
    # 역방향 GAE 누적: A_t = δ_t + γλ * A_{t+1}
    # Target_t = V(t) + A_t
    # Advantage = Target - V
```

### Pairwise Ranking Loss (pairwise_utils.py)

```python
def pairwise_ranking_loss(v_pos, v_neg, mask_pos, mask_neg):
    """Bradley-Terry Pairwise Ranking Loss

    P(pos > neg) = sigmoid(V_pos - V_neg)
    Loss = -log(sigmoid(V_pos - V_neg))

    Output 토큰만 사용하여 시퀀스 평균 비교 (Instruction 제외)
    """
    # 시퀀스 평균 value 계산 (Output 토큰만)
    v_pos_mean = (v_pos * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    v_neg_mean = (v_neg * mask_neg).sum(dim=1) / (mask_neg.sum(dim=1) + 1e-8)

    return -F.logsigmoid(v_pos_mean - v_neg_mean).mean()
```

### Advantage Whitening + Exponential Weighting (td_weighting.py)

```python
def build_weights(td_errors, loss_mask, beta=1.0, min_weight=0.1, max_weight=3.0):
    """TD error 기반 토큰 가중치 (AWR 원리 적용)

    수식:
        A_norm = (A - μ) / (σ + ε)  (Whitening)
        w = exp(A_norm / β)         (AWR-style weighting)
        w = clamp(w, min, max)      (Stability)

    IQL/AWR의 exp(advantage/β) 원리를 training loss에 적용
    EMA 통계 사용 시 배치 간 안정성 확보
    """
```

---

## 데이터셋 전처리

CodeContests 데이터셋을 WMTP 학습에 적합한 형태로 변환하는 과정.

### Alpaca 포맷 변환 (collators.py)

모든 샘플은 Alpaca 표준 템플릿으로 변환된다:

```python
def apply_alpaca_template(instruction, input_text, output):
    """Alpaca 표준 템플릿 적용

    Input이 있는 경우:
        Below is an instruction that describes a task, paired with an input
        that provides further context. Write a response that appropriately
        completes the request.

        ### Instruction:
        {problem_description}

        ### Input:
        {example_input}

        ### Response:
        {solution_code}

    Input이 없는 경우:
        Below is an instruction that describes a task. Write a response that
        appropriately completes the request.

        ### Instruction:
        {problem_description}

        ### Response:
        {solution_code}
    """
```

**Loss Masking** (`AlpacaDataCollator`, `PairwiseDataCollator`):
- Instruction + Input + "### Response:\n" 부분: `labels = -100` (학습 제외)
- Response (Output) 부분: `labels = token_ids` (학습 대상)
- Padding 토큰: `labels = -100` (학습 제외)

### Pairwise 샘플링 (datasets.py)

Value Model 학습을 위해 정답/오답 쌍을 구성한다.

#### Unique Pair 샘플링 (`_sample_unique_pairs`)

```python
# 알고리즘 (O(n) 복잡도):
# 1. 각 bin에서 problem별 correct/incorrect 리스트 셔플
# 2. min(correct수, incorrect수, max_pairs_per_problem) 개수만큼 1:1 매핑
# 3. difficulty weight 비례로 bin별 할당
# 4. 전체 셔플 후 n_samples개 반환

def _sample_unique_pairs(problem_index_map, n_samples, ...):
    for problem in problems:
        correct_list = shuffle(problem.correct_indices)
        incorrect_list = shuffle(problem.incorrect_indices)

        n_pairs = min(len(correct_list), len(incorrect_list), max_pairs_per_problem)
        for i in range(n_pairs):
            pairs.append({
                "correct_idx": correct_list[i],
                "incorrect_idx": incorrect_list[i],
            })
```

**Difficulty 기반 가중 샘플링**:
- 난이도 구간(bins) 정의: `{"low": [1, 5], "medium": [6, 10], "high": [11, 25]}`
- 각 bin별 가중치 할당: `{"low": 0.5, "medium": 0.3, "high": 0.2}`
- 가중치 비례로 샘플 수 배분

### Length-Balanced 샘플링 (datasets.py)

토큰 길이 편향을 제거하여 모델이 시퀀스 길이로 정답/오답을 구분하는 것을 방지.

#### 층화 추출 (`_sample_length_balanced_pairs`)

```python
# 알고리즘:
# 동일 problem 내 + 동일 length bin 내에서 1:1 매칭

length_bins = [0, 100, 150, 200, 300, 500, 1000, 2000]

def _sample_length_balanced_pairs(problem_index_map, n_samples, ...):
    for problem in problems:
        # 토큰 길이별로 correct/incorrect 그룹핑
        correct_by_bin = group_by_length_bin(problem.correct_indices,
                                              problem.correct_token_lengths)
        incorrect_by_bin = group_by_length_bin(problem.incorrect_indices,
                                                problem.incorrect_token_lengths)

        # 동일 bin 내에서만 매칭 (길이 편향 제거)
        for bin_label in common_bins:
            pairs.extend(match_within_bin(correct_by_bin[bin_label],
                                          incorrect_by_bin[bin_label]))
```

**층화 추출(Stratified Sampling)의 효과**:
- 정답이 짧고 오답이 긴 편향 제거
- 모델이 "내용"으로 가치를 학습하도록 유도
- Value Model의 토큰 수준 변별력 향상

### 메타데이터 기반 효율적 로딩

전체 데이터를 메모리에 로드하지 않고 필요한 샘플만 선택적으로 읽는다:

```python
# 1. 메타데이터에서 problem_index_map 로드 (캐싱)
#    {problem_id: {difficulty, correct_indices, incorrect_indices,
#                  correct_token_lengths, incorrect_token_lengths}}

# 2. 샘플링 알고리즘으로 필요한 인덱스만 선택

# 3. JSONL 파일에서 선택된 인덱스만 순차 읽기 (O(n) 최적화)
```

---

## 실험 설계

### 모델 구성

| 모델 | 용도 | 크기 |
|------|------|------|
| Meta-LLaMA MTP | Policy Model | 7B (4-head MTP) |
| Sheared-LLaMA | Value Model | 2.7B |

### 학습 설정 (Production)

| 항목 | Baseline | Critic | Verifiable |
|------|----------|--------|------------|
| **Epochs** | 1.0 | 3.0 | 1.0 |
| **Batch Size** | 12/GPU | 8/GPU | 12/GPU |
| **Learning Rate** | 1e-4 | 5e-5 | 1e-4 |
| **LoRA Rank** | 64 | 64 | 64 |
| **Gradient Accum** | 2 | 2 | 2 |

### Verifiable 핵심 Hyperparameters

```yaml
training:
  td_lambda: 0.95          # GAE lambda
  beta: 1.0                # Temperature (AWR)
  weight_clip_min: 0.1     # 최소 가중치
  weight_clip_max: 3.0     # 최대 가중치
  td_ema_momentum: 0.1     # EMA 모멘텀
  td_ema_warmup_steps: 10  # EMA 워밍업
```

### 데이터셋

- **CodeContests Train**: 정답+오답 페어
- **샘플 크기**:
  - Baseline/Verifiable: 200K samples
  - Critic: 50K pairwise samples
- **Max Length**: 2048 tokens

### 평가 벤치마크

- **코드 생성**: MBPP, HumanEval
- **In-domain**: CodeContests test
- **Pass@K**: K=1, 5, 10, 20

---

## 안정화 메커니즘

### Pairwise Ranking 학습

- **Bradley-Terry 모델**: P(pos > neg) = sigmoid(V_pos - V_neg)
- 상대 비교로 절대 스케일 불변
- Output 토큰만 사용하여 시퀀스 평균 비교

### EMA 정규화

- 배치 간 통계 안정화 (momentum=0.1)
- Warmup steps로 초기 불안정 방지

### Advantage Whitening

- 표준화로 스케일 불변성 확보
- Clipping으로 극단값 제한 (0.1~3.0)

---

## 파이프라인 실행 순서

### Baseline
```bash
python -m weighted_mtp.pipelines.run_baseline --config configs/production/baseline.yaml
```

### Verifiable (2단계)
```bash
# Phase 1: Critic 학습 (Pairwise Ranking)
python -m weighted_mtp.pipelines.run_critic --config configs/production/critic_mlp.yaml

# Phase 2: Policy 학습 (Critic checkpoint 필요)
python -m weighted_mtp.pipelines.run_verifiable --config configs/production/verifiable.yaml
```

### Rho-1 (Backup Plan)
Verifiable 접근이 실패할 경우를 대비한 대안 파이프라인.
```bash
# Phase 1: Reference 도메인 적응
python -m weighted_mtp.pipelines.run_ref_tuning --config configs/production/ref_tuning.yaml

# Phase 2: Policy 학습 (Reference checkpoint 필요)
python -m weighted_mtp.pipelines.run_rho1 --config configs/production/rho1.yaml
```

### 평가
```bash
python -m weighted_mtp.pipelines.run_evaluation \
  --checkpoint storage/checkpoints/verifiable/checkpoint_best.pt \
  --dataset humaneval \
  --temperature 0.2
```

---

## 참고문헌

### 핵심 참고문헌

- **AWR**: Peng et al. (2019). Advantage-Weighted Regression: Simple and Scalable Off-Policy RL. [arXiv](https://arxiv.org/abs/1910.00177)

- **APA**: Zhu et al. (2023). Fine-Tuning Language Models with Advantage-Induced Policy Alignment. [arXiv](https://arxiv.org/abs/2306.02231)

- **Q-SFT**: Hong et al. (2024). Q-Learning for Language Models via Supervised Fine-Tuning. ICLR 2025. [arXiv](https://arxiv.org/abs/2411.05193)

- **Meta MTP**: Gloeckle et al. (2024). Better & Faster LLM via Multi-Token Prediction. ICML 2024. [arXiv](https://arxiv.org/abs/2404.19737)

- **GAE**: Schulman et al. (2015). High-Dimensional Continuous Control Using GAE. ICLR 2016. [arXiv](https://arxiv.org/abs/1506.02438)

### 관련 연구

- **IQL**: Kostrikov et al. (2021). Offline RL with Implicit Q-Learning. ICLR 2022. [arXiv](https://arxiv.org/abs/2110.06169)

- **ILQL**: Snell et al. (2022). Offline RL for Natural Language Generation with Implicit Language Q Learning. [arXiv](https://arxiv.org/abs/2206.11871)

- **OREO**: Hao et al. (2024). Offline Reinforcement Learning for LLM Multi-Step Reasoning. [arXiv](https://arxiv.org/abs/2412.16145)

- **TLCR**: Token-Level Continuous Reward for Fine-grained RLHF. (2024). [arXiv](https://arxiv.org/abs/2407.16574)

- **Token Weighting for Long-Range LM**: NAACL 2025. [arXiv](https://arxiv.org/abs/2503.09202)

- **APA**: Zhu et al. (2023). Fine-Tuning Language Models with Advantage-Induced Policy Alignment. [arXiv](https://arxiv.org/abs/2306.02231)

- **Rho-1**: Lin et al. (2024). Not All Tokens Are What You Need. NeurIPS 2024 Oral. [arXiv](https://arxiv.org/abs/2404.07965)

### 기초 이론

- **PPO**: Schulman et al. (2017). Proximal Policy Optimization. [arXiv](https://arxiv.org/abs/1707.06347)

- **InstructGPT**: Ouyang et al. (2022). Training language models to follow instructions. [arXiv](https://arxiv.org/abs/2203.02155)

- **Sutton & Barto**: Reinforcement Learning: An Introduction (2nd Ed). [PDF](http://incompleteideas.net/book/RLbook2020.pdf)

- **Bradley-Terry Model**: Bradley & Terry (1952). Rank Analysis of Incomplete Block Designs.
