# Weighted MTP 아키텍처 문서

## 개요

Weighted Multi-Token Prediction (WMTP) 프로젝트는 코드 생성 모델의 성능 향상을 위해 **토큰별 가중치 학습**을 적용합니다. 핵심 아이디어는 모든 토큰을 동일하게 학습하는 대신, **올바른 코드 생성에 중요한 토큰에 더 높은 가중치**를 부여하는 것입니다.

### 핵심 목표
- **MTP (Multi-Token Prediction)**: 다음 토큰만 예측하는 NTP 대신, 여러 미래 토큰을 동시에 예측
- **Value-Weighted Learning**: TD(Temporal Difference) 기반 value function으로 토큰별 중요도 추정
- **Verifiable Reward**: 코드 실행 결과를 binary reward로 활용 (pass=1, fail=0)

---

## 디렉토리 구조

```
weighted_mtp/
├── src/weighted_mtp/
│   ├── core/                 # 핵심 인프라 (logging, env)
│   ├── data/                 # 데이터셋 및 DataLoader
│   │   ├── datasets.py       # JSONL 데이터셋 로딩, 샘플링
│   │   ├── collators.py      # Batch collation, Alpaca 템플릿
│   │   └── dataloader.py     # DataLoader 팩토리
│   ├── models/               # 모델 정의
│   │   ├── meta_mtp/         # Meta LLaMA MTP 모델
│   │   │   ├── adapter.py    # MetaLlamaMTPAdapter (Policy Model)
│   │   │   ├── transformer.py# Transformer 구현
│   │   │   └── checkpoints.py# 체크포인트 로딩
│   │   ├── value_model.py    # ValueModel (독립 Critic)
│   │   ├── value_head.py     # Value Head (Linear/MLP/Sigmoid)
│   │   └── lora.py           # LoRA 구현
│   ├── pipelines/            # 학습/평가 파이프라인
│   │   ├── run_baseline.py   # Baseline MTP 학습
│   │   ├── run_critic.py     # Critic (Value Model) 학습
│   │   ├── run_verifiable.py # Verifiable WMTP 학습 (핵심)
│   │   ├── run_rho1.py       # Rho-1 Selective 학습
│   │   ├── run_ref_tuning.py # Reference Model SFT
│   │   └── run_evaluation.py # 벤치마크 평가
│   ├── runtime/              # 분산 학습 인프라
│   │   ├── fsdp.py           # FSDP wrapping, all-reduce
│   │   ├── distributed.py    # 분산 환경 초기화
│   │   └── environment.py    # 환경 설정
│   ├── value_weighting/      # 가중치 계산 로직
│   │   ├── td_weighting.py   # TD error 기반 가중치
│   │   └── rho1_weighting.py # Rho-1 selective 가중치
│   └── utils/                # 유틸리티
│       ├── loss_utils.py     # MTP CE Loss 계산
│       ├── pairwise_utils.py # Pairwise ranking, λ-return
│       ├── checkpoint_utils.py# 체크포인트 저장/로딩
│       └── generation_utils.py# MTP 생성 유틸
├── configs/                  # YAML 설정 파일
│   ├── local/                # 로컬 개발용
│   └── production/           # 프로덕션 학습용
├── scripts/                  # 실행 스크립트
│   └── vessl/                # VESSL AI 학습 스크립트
├── storage/                  # 모델/데이터 저장소
│   ├── models/               # Pretrained 모델
│   ├── datasets/             # 처리된 데이터셋
│   └── checkpoints/          # 학습 체크포인트
└── tests/                    # 테스트 코드
```

---

## 모델 아키텍처

### 1. Policy Model (MetaLlamaMTPAdapter)

MTP 학습을 위한 Transformer 모델입니다. Meta LLaMA3를 기반으로 하며, `n_future_tokens`개의 미래 토큰을 동시에 예측합니다.

**파일**: `src/weighted_mtp/models/meta_mtp/adapter.py`

```
Transformer 구조:
├── tok_embeddings (Embedding)
├── layers[0..N-k] (trunk layers)
├── extra_heads[0..k-1] (MTP heads)
├── norm (RMSNorm)
└── output (Linear → vocab)

Forward:
  input_ids → embedding → trunk layers → [head_0, head_1, ..., head_k]
                                              ↓
                                    logits [batch, seq, k, vocab]
```

**주요 메서드**:
- `from_pretrained()`: Safetensors 또는 checkpoint에서 로드
- `apply_lora()`: LoRA (Low-Rank Adaptation) 적용
- `forward()`: MTP logits 반환, `return_hidden_states=True`로 hidden states 반환

**Flash Attention 구현**:
```python
# Training 시 Flash Attention (start_pos=0, seqlen>1)
if start_pos == 0 and seqlen > 1:
    output = F.scaled_dot_product_attention(
        xq, keys, values,
        is_causal=True,  # Causal masking 자동 적용
    )
```

### 2. Value Model (독립 Critic)

Pairwise Ranking으로 학습되는 독립 Value Model입니다. HuggingFace LlamaModel을 backbone으로 사용합니다.

**파일**: `src/weighted_mtp/models/value_model.py`

```
ValueModel 구조:
├── backbone (HuggingFace LlamaModel, 2.7B)
└── value_head (MLPValueHead)

Forward:
  input_ids → backbone → hidden_states → value_head → V(t) [batch, seq, 1]
```

**주요 특징**:
- Policy Model과 완전히 분리된 별도 모델 (별도 backbone)
- Pairwise Ranking Loss로 학습 (correct > incorrect 순서 학습)
- LoRA로 효율적인 fine-tuning 지원
- Verifiable 파이프라인에서 eval only로 사용

### 3. Value Head

토큰별 value 예측을 위한 head입니다.

**파일**: `src/weighted_mtp/models/value_head.py`

| 타입 | 구조 | 용도 |
|------|------|------|
| `LinearValueHead` | hidden → 1 | 기본 MSE 학습 |
| `SigmoidValueHead` | hidden → 1 → σ | BCE Loss (확률 출력) |
| `MLPValueHead` | hidden → h/4 → h/8 → 1 | 표현력 향상, Pairwise 학습 권장 |

---

## 학습 파이프라인

### 전체 학습 순서

```
1. Reference Model SFT (run_ref_tuning.py)
   └─> storage/checkpoints/ref_tuned/

2. Critic 학습 (run_critic.py)
   └─> storage/checkpoints/critic_best/

3. Verifiable WMTP 학습 (run_verifiable.py)
   └─> storage/checkpoints/verifiable/

4. 평가 (run_evaluation.py)
   └─> MLflow metrics
```

### 1. Baseline (run_baseline.py)

가중치 없이 MTP CE Loss만으로 학습하는 baseline입니다.

```python
# 모든 head에 대해 CE Loss 평균
Loss = Σ_{k=1}^{n_future} CE(logits_k, labels_{t+k}) / n_future
```

### 2. Critic (run_critic.py)

독립 Value Model을 Pairwise Ranking Loss로 학습합니다.

**Pairwise Ranking Loss (Bradley-Terry)**:
```python
# P(correct > incorrect) = σ(V_correct - V_incorrect)
# Loss = -log σ(V_correct - V_incorrect)
v_pos_mean = (v_pos * mask_pos).sum(dim=1) / mask_pos.sum(dim=1)
v_neg_mean = (v_neg * mask_neg).sum(dim=1) / mask_neg.sum(dim=1)
loss = -F.logsigmoid(v_pos_mean - v_neg_mean).mean()
```

**λ-Return 기반 Pointwise Loss (보조)**:
```python
# Fitted λ-Return: G_t^λ = (1-λ)γV_{t+1} + λγG_{t+1}^λ
lambda_targets = compute_lambda_return(values, rewards, loss_mask, gamma, lam)
value_loss = F.smooth_l1_loss(values, lambda_targets)  # Huber loss
```

**학습 흐름**:
1. 같은 problem의 correct/incorrect solution 쌍 로드
2. 각각 forward하여 sequence-level value 계산
3. Correct의 평균 value가 incorrect보다 높도록 학습

### 3. Verifiable WMTP (run_verifiable.py)

**핵심 파이프라인**. 학습된 Critic으로 토큰별 가중치를 계산하여 Policy Model을 학습합니다.

**학습 흐름**:
```python
# 1. Critic (frozen) forward → V(t) 예측
value_logits = critic_model(input_ids)  # [batch, seq, 1]

# 2. TD error 계산: δ_t = γV(t) - V(t-1)
td_errors = compute_td_errors(value_logits, rewards, loss_mask, gamma)

# 3. 가중치 변환: Advantage Whitening + Exponential
weights = build_weights(td_errors, loss_mask, beta, min_weight, max_weight)

# 4. Weighted CE Loss
loss_dict = compute_mtp_ce_loss(logits, labels, attention_mask, weights)
```

**TD Error 계산** (`value_weighting/td_weighting.py`):
```python
# 중간 토큰: marginal value contribution
δ_t = γ·V(t) - V(t-1)

# terminal 토큰: 실제 reward와의 차이
δ_T = R - V(T)
```

**가중치 변환**:
```python
# Advantage Whitening: 스케일 불변성 확보
δ_normalized = (δ - mean) / (std + eps)

# Exponential transformation
w = exp(δ_normalized / beta)

# Clipping: 안정성 보장
w = clamp(w, min_weight, max_weight)
```

**GAE (Generalized Advantage Estimation)**:
```python
# 역방향 계산: A_t = δ_t + γλ·A_{t+1}
# Target_t = V_t + A_t
# λ=0.0: TD(0), λ=0.95: GAE 권장값, λ=1.0: Monte Carlo
```

### 4. Rho-1 (run_rho1.py)

Reference Model과의 perplexity 차이로 토큰 선택 학습합니다.

**Selective Token Loss**:
```python
# Reference (frozen) vs Policy forward
ppl_ref = CE(ref_logits, labels)
ppl_policy = CE(policy_logits, labels)

# 선택 기준: Reference 대비 어려운 토큰 (상위 k%)
excess_loss = ppl_policy - ppl_ref
weights = compute_mtp_selective_weights(excess_loss, k_percent=0.6)

# 선택된 토큰만 학습
loss = compute_mtp_ce_loss(logits, labels, attention_mask, weights)
```

---

## 데이터 흐름

### 데이터셋 구조

**파일**: `src/weighted_mtp/data/datasets.py`

```
storage/datasets/{dataset_name}/processed/
├── train.jsonl          # Solution-level 샘플
├── train_metadata.json  # Problem-level 인덱스 맵
├── test_eval.jsonl      # 평가용 (problem-level)
└── ...
```

**JSONL 샘플 형식**:
```json
{
  "instruction": "Write a function that...",
  "input": "",
  "output": "def solve(...):\n    ...",
  "problem_id": "codecontests_123",
  "is_correct": true
}
```

**메타데이터 구조** (`train_metadata.json`):
```json
{
  "problem_index_map": {
    "codecontests_123": {
      "difficulty": 12,
      "correct_indices": [0, 5, 23],
      "incorrect_indices": [1, 2, 3, 4],
      "correct_token_lengths": [150, 200, 180],
      "incorrect_token_lengths": [120, 140, 130, 160]
    }
  }
}
```

### 샘플링 전략

**1. Difficulty-based Sampling**:
```yaml
difficulty_bins:
  easy: [1, 8]
  medium: [9, 15]
  hard: [16, 25]
difficulty_weights:
  easy: 0.2
  medium: 0.5
  hard: 0.3
```

**2. Length-balanced Sampling** (Critic 전용):
- 같은 length bin 내에서 correct/incorrect 쌍 매칭
- 길이 편향 제거하여 모델이 sequence length로 학습하는 것 방지

**3. Unique Pair Sampling**:
- correct/incorrect 인덱스 1:1 매칭 (중복 방지)
- `max_pairs_per_problem`으로 problem당 다양성 확보

### Collator (Batch 처리)

**파일**: `src/weighted_mtp/data/collators.py`

```python
# Alpaca 템플릿 적용
text = f"""Below is an instruction...
### Instruction:
{instruction}
### Input:
{input}
### Response:
{output}"""

# Tokenization + Padding
input_ids, attention_mask, labels = tokenize_and_pad(batch)

# Instruction 마스킹 (Output만 학습)
labels[:instruction_end] = -100
```

---

## 분산 학습 (FSDP)

**파일**: `src/weighted_mtp/runtime/fsdp.py`

### FSDP Wrapping

```python
wrapped_model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    sharding_strategy=FULL_SHARD,  # ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    use_orig_params=True,
    sync_module_states=True,
)
```

**Transformer Layer 자동 감지**:
```python
# HuggingFace LlamaModel → LlamaDecoderLayer
# MetaLlamaMTPAdapter → TransformerBlock
transformer_layer_cls = _detect_transformer_layer_cls(model)
```

### 샤딩 전략

| 전략 | 설명 | 용도 |
|------|------|------|
| `FULL_SHARD` | ZeRO-3 (Model + Optimizer + Gradient 샤딩) | Verifiable/Baseline |
| `SHARD_GRAD_OP` | ZeRO-2 (Optimizer + Gradient만) | 메모리 중간 |
| `NO_SHARD` | DDP (복제) | Critic (Value Head만 학습) |

### Activation Checkpointing

```python
# FSDP wrapping 후 적용 (메모리 절감)
apply_activation_checkpointing(
    wrapped_model,
    checkpoint_wrapper_fn=non_reentrant_wrapper,
    check_fn=lambda m: isinstance(m, layer_cls_tuple),
)
```

### Metric Aggregation

```python
# 모든 GPU에서 loss 평균
avg_loss = all_reduce_scalar(loss.item(), op="mean")

# 여러 metric 한 번에 집계 (단일 통신으로 효율화)
metrics = all_reduce_scalars({"loss": 0.5, "acc": 0.8})
```

### Rank-aware 데이터 분산

```python
# 모든 rank가 동일 seed로 전체 샘플 계산
all_indices = _compute_sampling_indices(...)

# rank::world_size 패턴으로 분할
if world_size > 1:
    rank_indices = all_indices[rank::world_size]
```

---

## 설정 파일 구조

**위치**: `configs/production/verifiable.yaml`

```yaml
experiment:
  name: verifiable_wmtp
  run_prefix: verifiable

models:
  policy:
    path: storage/models/meta-llama-mtp
    dtype: bfloat16
    params_override:
      max_seq_len: 2048

  critic:
    checkpoint_path: storage/checkpoints/critic_best/checkpoint.pt
    base_model_path: storage/models/ref-sheared-llama-2.7b/raw

training:
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 1e-5
  num_epochs: 3
  max_steps: 10000

  # TD Weighting 설정
  gamma: 1.0
  lam: 0.95
  beta: 1.0
  min_weight: 0.1
  max_weight: 5.0

  # LoRA 설정
  use_lora: true
  lora:
    rank: 8
    alpha: 16.0
    dropout: 0.0
    target_modules: [wq, wk, wv, wo]

data:
  dataset: codecontests
  split: train
  sampling:
    n_samples: 50000
    difficulty_bins:
      easy: [1, 8]
      medium: [9, 15]
      hard: [16, 25]
    difficulty_weights:
      easy: 0.2
      medium: 0.5
      hard: 0.3

runtime:
  fsdp:
    sharding_strategy: FULL_SHARD
    mixed_precision: true
    activation_checkpointing: true
```

---

## 평가 파이프라인

**파일**: `src/weighted_mtp/pipelines/run_evaluation.py`

### 지원 벤치마크

| 벤치마크 | 메트릭 | 테스트 방식 |
|----------|--------|-------------|
| HumanEval | Pass@K | 함수 시그니처 완성 |
| MBPP | Pass@K | 전체 코드 생성 |
| CodeContests | Pass@K | stdin/stdout 테스트 |
| GSM8K | Accuracy | 정답 추출 비교 |

### 평가 흐름

```python
# 1. Checkpoint 로드
adapter = load_checkpoint_for_evaluation(checkpoint_path)

# 2. 데이터셋 로드
dataset = load_evaluation_dataset("humaneval", split="test")

# 3. 생성 (MTP 활용)
for task in dataset:
    completions = generate_with_mtp(
        adapter, tokenizer, task["prompt"],
        num_samples=100,  # Pass@100
        temperature=0.8,
    )

# 4. 실행 및 평가
pass_at_k = evaluate_pass_at_k(completions, tests, k=[1, 10, 100])
```

---

## 핵심 개념 요약

### TD Learning과 Value Weighting

1. **Value Function V(t)**: 토큰 t까지 생성했을 때 최종 성공 확률 예측
2. **TD Error δ_t**: 토큰 t의 marginal contribution (γV(t) - V(t-1))
3. **가중치 w_t**: TD error가 큰 토큰 = 중요한 결정 포인트

### Pairwise Ranking vs Pointwise

- **Pairwise**: Correct/Incorrect 순서 학습 (상대적 비교)
- **Pointwise**: 개별 토큰의 절대 value 학습 (λ-return)

### Lambda Return (λ-Return)

```
G_t^λ = (1-λ)γV_{t+1} + λγG_{t+1}^λ

λ=0.0: TD(0), 한 스텝 bootstrap
λ=0.95: GAE 권장값, 빠른 수렴 + 안정성
λ=1.0: Monte Carlo, 종단 reward만 사용
```

---

## 성능 최적화

| 최적화 | 메모리 효과 | 속도 효과 | 적용 대상 |
|--------|-----------|----------|----------|
| FSDP FULL_SHARD | 75% 절감 | 6-10% 느림 | Verifiable/Baseline |
| BFloat16 | A100 최적화 | 네이티브 속도 | 모든 파이프라인 |
| Flash Attention | 30-40% 절감 | 2-4배 빠름 | Training (start_pos=0) |
| Activation Checkpointing | 추가 절감 | 약간 느림 | 대용량 모델 |

**시너지 효과** (FULL_SHARD + BFloat16 + Flash Attention):
- 메모리: 90GB → 18-21GB (77% 절감)
- Batch size: 4 → 8-12 증가 가능

---

## 참고

- **Transformer 구현**: `src/weighted_mtp/models/meta_mtp/transformer.py`
- **Adapter**: `src/weighted_mtp/models/meta_mtp/adapter.py`
- **Value Model**: `src/weighted_mtp/models/value_model.py`
- **메타데이터 로딩**: `src/weighted_mtp/data/datasets.py`
- **분산학습**: `src/weighted_mtp/runtime/fsdp.py`, `distributed.py`
- **Value Weighting**: `src/weighted_mtp/value_weighting/td_weighting.py`
- **Pairwise Utils**: `src/weighted_mtp/utils/pairwise_utils.py`
- **MTP Loss**: `src/weighted_mtp/utils/loss_utils.py`
