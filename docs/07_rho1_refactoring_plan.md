# Rho-1 Weighting 전면 개편 계획서

## 1. 현황 분석

### 1.1 현재 구현의 문제점

**Rho-1 논문과의 주요 차이:**

1. **CRITICAL: Softmax Weighting vs Top-k Selection**
   - 논문: Binary hard selection (top k% = 1, rest = 0)
   - 현재: Continuous softmax weights (모든 토큰 기여)
   - 위치: `rho1_weighting.py:105-138 build_weights()`

2. **CRITICAL: Absolute Difference vs Signed Difference**
   - 논문: `ℒ_Δ = ℒ_θ - ℒ_ref` (부호 보존)
   - 현재: `torch.abs(ℒ_θ - ℒ_ref)` (부호 제거)
   - 위치: `rho1_weighting.py:85`
   - 문제: Policy가 Reference보다 좋은 토큰도 선택될 수 있음

3. **MODERATE: Temperature Parameter**
   - 논문: 없음
   - 현재: `temperature=1.0` hyperparameter 추가
   - 위치: `rho1_weighting.py:107`, `run_rho1.py:508`

4. **구조적 문제: Position-wise Averaging**
   - 현재: 4개 head의 excess loss를 평균 → [batch, seq]
   - 문제: Head별 독립적 선택 불가능
   - 위치: `rho1_weighting.py:94-100`

### 1.2 기존 코드 구조

**`rho1_weighting.py` (191 lines):**
```python
compute_excess_loss(policy_logits, ref_logits, labels, attention_mask)
    → excess_loss: [batch, seq]  # 4개 head 평균

build_weights(excess_loss, temperature, attention_mask)
    → weights: [batch, seq]  # Softmax

compute_rho1_stats(excess_loss, weights, attention_mask)
    → stats: dict
```

**`run_rho1.py` - Validation (lines 200-277):**
```python
1. ref_logits = ref_model.forward(input_ids)  # [batch, seq, vocab] - 1번만!
2. policy_logits = adapter.forward(input_ids)  # [batch, seq, 4, vocab]
3. excess_loss = compute_excess_loss(...)     # [batch, seq]
4. weights = build_weights(...)                # [batch, seq]
5. Loop over k=1~4: weighted CE loss 계산
```

**`run_rho1.py` - Training (lines 479-559):**
- Validation과 동일한 구조
- Backward pass 추가

### 1.3 Reference Inference 방식 확인 ✅

**중요 발견: Reference는 배치당 1번만 inference**

```python
# Validation (line 207-208)
ref_logits_mtp = ref_model.transformer.forward(input_ids, return_all_heads=False)
ref_logits = ref_logits_mtp.squeeze(2)  # [batch, seq, vocab]

# Training (line 489-490)
with torch.no_grad():
    ref_logits_mtp = ref_model.transformer.forward(input_ids, return_all_heads=False)
    ref_logits = ref_logits.squeeze(2)  # [batch, seq, vocab]
```

**각 head k에 대해:**
- Head 0 (t+1): `ref_logits[:, :, :]` 사용
- Head 1 (t+2): `ref_logits[:, 1:, :]` 사용 (slice)
- Head 2 (t+3): `ref_logits[:, 2:, :]` 사용 (slice)
- Head 3 (t+4): `ref_logits[:, 3:, :]` 사용 (slice)

→ **추가 inference 없음, indexing만 다름**

---

## 2. 목표

### 2.1 사용자 제안 방식 구현

**MTP-specific Rho-1 Weighting:**
```
t+1 토큰 (Head 0): 무조건 학습 (weight = 1)
t+2~t+4 토큰 (Head 1,2,3): Rho-1 방식 적용
    - Signed excess loss 계산
    - Batch-wise top-k% selection
    - Binary drop (0 or 1)

결과: 각 position에서 최소 1개, 최대 4개 토큰 학습
```

**이론적 근거:**
- t+1은 NTP baseline과 동일 task → 항상 학습 필수
- t+2~t+4는 장기 예측 → 선택적 학습으로 효율화
- Gradient flow 안정성 확보 (최소 1개 보장)

### 2.2 Rho-1 논문 방식 정확 구현

1. **Signed difference**: `excess_loss = policy_ce - ref_ce` (NOT abs)
2. **Top-k selection**: Binary weights (1 if top k%, else 0)
3. **No temperature**: 불필요한 hyperparameter 제거

### 2.3 코드 간결화

- `compute_excess_loss()` + `build_weights()` → 단일 함수로 통합
- Per-head binary selection으로 명확한 로직
- Validation/Training loop 중복 제거

---

## 3. 설계 원칙 (개발원칙 준수)

### 원칙 1: 앞/뒤 흐름 확인 후 구조 파악 ✅
- `rho1_weighting.py` 전체 읽음
- `run_rho1.py` validation/training loop 확인
- Reference inference가 1번만 수행됨을 확인

### 원칙 2: 기존 구조 존중, 중복 제거 ✅
- **유지**: Timestep alignment 로직 (정확함)
- **유지**: Reference 1번 inference 구조
- **삭제**: `build_weights()` (softmax 방식 제거)
- **개편**: `compute_excess_loss()` → `compute_mtp_selective_weights()`

### 원칙 3: 잘못된 구조 삭제, 새로 생성 ✅
- Softmax weighting → Top-k selection으로 전면 교체
- Averaged excess loss → Per-head binary selection
- `torch.abs()` 제거

### 원칙 4: 하위 호환성 무시, 깔끔하게 ✅
- Config breaking change 허용: `temperature` → `k_percent`
- 함수 시그니처 변경
- 기존 checkpoint와는 호환 (model weights 동일)

### 원칙 5: 계획과 비교 검토 ✅
- 이전 분석: Rho-1이 논문과 다름 (softmax, abs)
- 현재 계획: 논문 방식 + MTP 확장 (t+1 always)
- 방향성 일치

### 원칙 6: 패키지 도구 활용 ✅
- uv 기반 프로젝트 구조 유지
- pytest integration test 재실행
- 순수 PyTorch 기능만 사용 (추가 의존성 없음)

---

## 4. 구현 계획

### Phase 1: `rho1_weighting.py` 전면 개편

**삭제할 코드:**
```python
# ❌ 완전 삭제
def compute_excess_loss(...) -> torch.Tensor:
    # Lines 21-102
    # Averaged excess loss 방식

def build_weights(...) -> torch.Tensor:
    # Lines 105-138
    # Softmax weighting 방식
```

**신규 작성:**
```python
# ✅ 새로운 함수
def compute_mtp_selective_weights(
    policy_logits: torch.Tensor,  # [batch, seq, 4, vocab]
    ref_logits: torch.Tensor,      # [batch, seq, vocab] - 이미 계산된 것!
    labels: torch.Tensor,           # [batch, seq]
    attention_mask: torch.Tensor,  # [batch, seq]
    k_percent: float = 0.6,         # Top 60% selection for t+2~t+4
) -> tuple[torch.Tensor, dict]:
    """MTP-specific Rho-1 weighting

    Strategy:
    - Head 0 (t+1): 무조건 weight=1 (Reference 비교 안 함)
    - Head 1,2,3 (t+2~t+4): Batch-wise top-k selection

    Returns:
        weights: [batch, seq, 4] - Binary (0 or 1) per head
        stats: dict - Selection statistics
    """
```

**수정할 코드:**
```python
# ⚠️ 수정
def compute_rho1_stats(...):
    # Per-head statistics로 변경
    # weights: [batch, seq, 4] 입력 받도록
```

### Phase 2: `run_rho1.py` 수정

**Validation loop (lines 200-277):**

```python
# BEFORE
excess_loss = compute_excess_loss(policy_logits, ref_logits, labels, attention_mask)
weights = build_weights(excess_loss, temperature, attention_mask)

# Loop over k=1~4
for k in range(1, n_future + 1):
    weights_k = weights[:, :valid_len]  # Same weight for all heads!
    weighted_ce_k = ce_loss_k * weights_k * mask_k
```

```python
# AFTER
weights, stats = compute_mtp_selective_weights(
    policy_logits, ref_logits, labels, attention_mask, k_percent=0.6
)
# weights: [batch, seq, 4]

# Loop over k=1~4
for k in range(1, n_future + 1):
    weights_k = weights[:, :valid_len, k-1]  # Different weight per head!
    weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.reshape(-1)
```

**Training loop (lines 479-559):**
- Validation과 동일한 수정 적용

**Logging 추가:**
```python
if is_main_process() and step % log_interval == 0:
    logger.info(f"Selection stats:")
    logger.info(f"  Overall ratio: {stats['selection_ratio']:.2%}")
    logger.info(f"  Head 0 (t+1): {stats['head_0_count']} (100%)")
    logger.info(f"  Head 1 (t+2): {stats['head_1_count']} ({stats['head_1_ratio']:.1%})")
    logger.info(f"  Head 2 (t+3): {stats['head_2_count']} ({stats['head_2_ratio']:.1%})")
    logger.info(f"  Head 3 (t+4): {stats['head_3_count']} ({stats['head_3_ratio']:.1%})")
```

### Phase 3: Config 업데이트

**`configs/rho1/rho1_local.yaml`:**

```yaml
training:
  # ❌ 삭제
  # temperature: 1.0

  # ✅ 추가
  k_percent: 0.6  # Top 60% selection for t+2~t+4 heads
```

**`configs/defaults.yaml`:**

```yaml
training:
  # Rho-1 specific
  k_percent: 0.6  # Top-k selection ratio (0.5-0.7 range)
```

### Phase 4: 테스트 및 검증

**Unit test (선택적):**
```python
# tests/unit/test_rho1_weighting.py
def test_mtp_selective_weights_head0_always_selected():
    """Head 0는 항상 100% 선택되어야 함"""

def test_mtp_selective_weights_topk_ratio():
    """Head 1,2,3는 k_percent만큼 선택되어야 함"""

def test_signed_excess_loss():
    """Excess loss가 signed difference인지 확인"""
```

**Integration test:**
```bash
# 기존 test 재실행
pytest tests/integration/test_pipeline_rho1.py -v

# 출력 확인:
# - Selection ratio가 60-70% 범위인지
# - Head 0가 항상 100%인지
# - Loss가 정상적으로 감소하는지
```

---

## 5. 상세 구현

### 5.1 `compute_mtp_selective_weights()` 전체 코드

```python
def compute_mtp_selective_weights(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    k_percent: float = 0.6,
) -> tuple[torch.Tensor, dict]:
    """MTP-specific Rho-1 weighting: t+1 always, t+2~4 selective

    Reference inference는 이미 완료되어 ref_logits로 전달됨.
    각 head에 대해 ref_logits의 다른 slice만 사용.

    Args:
        policy_logits: [batch, seq, n_future=4, vocab]
        ref_logits: [batch, seq, vocab] - 이미 계산된 것!
        labels: [batch, seq]
        attention_mask: [batch, seq]
        k_percent: Top k% selection ratio (0~1)

    Returns:
        weights: [batch, seq, 4] - Binary selection mask
        stats: dict - Selection statistics
    """
    batch_size, seq_len, n_future, vocab_size = policy_logits.shape
    device = policy_logits.device

    # Initialize: all zeros
    weights = torch.zeros(batch_size, seq_len, n_future, device=device)

    # Statistics
    stats = {}

    # ===== HEAD 0 (t+1): 무조건 선택 =====
    weights[:, :, 0] = attention_mask.float()
    stats['head_0_count'] = attention_mask.sum().item()
    stats['head_0_ratio'] = 1.0

    # ===== HEAD 1,2,3 (t+2~t+4): Rho-1 selection =====
    for k in range(2, n_future + 1):  # k = 2, 3, 4
        head_idx = k - 1  # 1, 2, 3
        valid_len = seq_len - k

        if valid_len <= 0:
            stats[f'head_{head_idx}_count'] = 0
            stats[f'head_{head_idx}_ratio'] = 0.0
            continue

        # Timestep alignment (기존 정확한 로직 유지)
        policy_logits_k = policy_logits[:, :valid_len, head_idx, :]  # [batch, valid_len, vocab]
        ref_logits_k = ref_logits[:, k-1:k-1+valid_len, :]            # [batch, valid_len, vocab]
        labels_k = labels[:, k:k+valid_len]                            # [batch, valid_len]
        mask_k = attention_mask[:, k:k+valid_len]                      # [batch, valid_len]

        # Per-token CE loss
        policy_ce = F.cross_entropy(
            policy_logits_k.reshape(-1, vocab_size),
            labels_k.reshape(-1),
            reduction='none'
        ).view(batch_size, valid_len)

        ref_ce = F.cross_entropy(
            ref_logits_k.reshape(-1, vocab_size),
            labels_k.reshape(-1),
            reduction='none'
        ).view(batch_size, valid_len)

        # ✅ Signed excess loss (NOT abs!)
        excess_loss = policy_ce - ref_ce  # [batch, valid_len]

        # Batch-wise top-k selection (논문 방식)
        valid_mask = mask_k.bool()
        valid_excess = excess_loss[valid_mask]

        if valid_excess.numel() == 0:
            stats[f'head_{head_idx}_count'] = 0
            stats[f'head_{head_idx}_ratio'] = 0.0
            continue

        # Top k% threshold
        threshold = torch.quantile(valid_excess, 1 - k_percent)

        # Binary selection: 1 if >= threshold, 0 otherwise
        selected = (excess_loss >= threshold).float() * mask_k.float()
        weights[:, :valid_len, head_idx] = selected

        # Statistics
        stats[f'head_{head_idx}_count'] = selected.sum().item()
        total_valid = mask_k.sum().item()
        stats[f'head_{head_idx}_ratio'] = selected.sum().item() / (total_valid + 1e-8)
        stats[f'head_{head_idx}_excess_mean'] = valid_excess.mean().item()
        stats[f'head_{head_idx}_threshold'] = threshold.item()

    # Overall statistics
    total_possible = attention_mask.sum().item() * n_future
    total_selected = weights.sum().item()
    stats['selection_ratio'] = total_selected / (total_possible + 1e-8)
    stats['avg_heads_per_position'] = total_selected / (attention_mask.sum().item() + 1e-8)

    return weights, stats
```

### 5.2 Validation/Training Loop 수정

**핵심 변경점:**

```python
# BEFORE: Position-wise weights [batch, seq]
weights_k = weights[:, :valid_len]  # Same for all heads
weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.reshape(-1)

# AFTER: Per-head weights [batch, seq, 4]
weights_k = weights[:, :valid_len, k-1]  # Different per head
weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.reshape(-1)
```

**전체 수정 (Validation/Training 공통):**

```python
# Reference forward (1번만!)
with torch.no_grad():
    ref_logits_mtp = ref_model.transformer.forward(input_ids, return_all_heads=False)
    ref_logits = ref_logits_mtp.squeeze(2)  # [batch, seq, vocab]

# Policy forward
policy_logits = adapter.transformer.forward(input_ids, return_all_heads=True)
# [batch, seq, 4, vocab]

# ✅ NEW: Per-head weights
weights, selection_stats = compute_mtp_selective_weights(
    policy_logits, ref_logits, labels, attention_mask, k_percent=config.training.k_percent
)
# weights: [batch, seq, 4]

# Weighted CE loss
batch_weighted_ce_loss = 0.0

for k in range(1, n_future + 1):
    valid_len = seq_len - k
    if valid_len <= 0:
        continue

    policy_logits_k = policy_logits[:, :valid_len, k-1, :]
    labels_k = labels[:, k:k+valid_len]
    weights_k = weights[:, :valid_len, k-1]  # ✅ Per-head indexing
    mask_k = attention_mask[:, k:k+valid_len]

    ce_loss_k = F.cross_entropy(
        policy_logits_k.reshape(-1, vocab_size),
        labels_k.reshape(-1),
        reduction='none'
    )

    weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * mask_k.float().reshape(-1)

    mask_sum_k = mask_k.float().sum()
    if mask_sum_k > 0:
        batch_weighted_ce_loss += weighted_ce_k.sum() / mask_sum_k

weighted_ce_loss = batch_weighted_ce_loss / n_future
```

### 5.3 파일별 수정 요약

| 파일 | 변경 유형 | 주요 내용 |
|------|----------|----------|
| `rho1_weighting.py` | 전면 개편 | - `compute_excess_loss()` 삭제<br>- `build_weights()` 삭제<br>- `compute_mtp_selective_weights()` 신규<br>- `compute_rho1_stats()` 수정 |
| `run_rho1.py` | 부분 수정 | - Validation loop: weights 계산 변경<br>- Training loop: weights 계산 변경<br>- Logging 추가 (selection stats) |
| `configs/rho1/rho1_local.yaml` | Config 변경 | - `temperature` 삭제<br>- `k_percent: 0.6` 추가 |
| `configs/defaults.yaml` | Config 변경 | - `k_percent: 0.6` 추가 |

---

## 6. 검증 계획

### 6.1 Integration Test 실행

```bash
# Rho-1 integration test 재실행
PYTHONPATH=src pytest tests/integration/test_pipeline_rho1.py -v -s

# 예상 출력:
# ✓ Config validation passed
# ✓ Model loading successful
# ✓ Training step 0: selection_ratio=72.5%
#   - Head 0: 100.0% (always)
#   - Head 1: 62.3% (top-k)
#   - Head 2: 58.9% (top-k)
#   - Head 3: 64.1% (top-k)
# ✓ Loss decreased: 3.45 → 2.98
# ✓ Test passed
```

### 6.2 검증 체크리스트

**기능적 검증:**
- [ ] Head 0가 항상 100% 선택됨
- [ ] Head 1,2,3이 k_percent ± 5% 범위로 선택됨
- [ ] Selection ratio가 0.6-0.8 범위 (평균 2.8개/position)
- [ ] Loss가 정상적으로 감소함
- [ ] Gradient flow 확인 (모든 position에서 최소 1개)

**성능 검증:**
- [ ] Reference inference는 배치당 1번만 수행
- [ ] Memory 사용량 변화 없음 (weights만 [batch,seq] → [batch,seq,4])
- [ ] Training speed 변화 없음

**코드 품질:**
- [ ] 함수 시그니처 명확함
- [ ] 주석이 정확함 (이모지 없음, 한글)
- [ ] 불필요한 wrapper 없음
- [ ] 네이밍 일관성

### 6.3 로그 예시

**정상 동작 로그:**
```
[Step 100] Training metrics:
  Weighted CE loss: 2.847
  Selection statistics:
    Overall ratio: 72.3% (2.89 heads/position avg)
    Head 0 (t+1): 1024 tokens (100.0% - always selected)
    Head 1 (t+2): 650 tokens (63.5% - top-k, threshold=0.42)
    Head 2 (t+3): 592 tokens (57.8% - top-k, threshold=0.51)
    Head 3 (t+4): 655 tokens (64.0% - top-k, threshold=0.38)
  Avg excess loss: t+2=0.42, t+3=0.51, t+4=0.38
```

---

## 7. 예상 효과

### 7.1 이론적 정확성
- ✅ Rho-1 논문 방식 정확히 구현 (signed diff + top-k)
- ✅ MTP 특성에 맞는 합리적 확장 (t+1 always)
- ✅ 불필요한 hyperparameter 제거 (temperature)

### 7.2 학습 안정성
- ✅ 모든 position에서 최소 1개 토큰 학습 보장
- ✅ Gradient flow 안정적
- ✅ 초기 학습 단계 안정성 향상

### 7.3 효율성
- ✅ Reference inference: 배치당 1번 (변화 없음, 이미 최적)
- ✅ Excess loss 계산: 3회로 절감 (4회 → 3회)
- ⚠️ Memory 미미한 증가: weights [batch,seq] → [batch,seq,4]

### 7.4 Interpretability
- ✅ 어떤 head가 얼마나 선택되는지 명확히 추적
- ✅ Selection threshold 확인 가능
- ✅ Per-head 난이도 분석 가능

---

## 8. 실행 순서

1. **MD 파일 검토 및 승인** ← 현재 단계
2. **`rho1_weighting.py` 전면 개편**
3. **`run_rho1.py` 수정**
4. **Config 파일 업데이트**
5. **Integration test 실행 및 검증**
6. **결과 분석 및 보고**

---

## 9. 리스크 및 대응

### 9.1 Breaking Changes
- **Config 호환성**: `temperature` 제거 → 기존 config 수정 필요
- **대응**: `rho1_local.yaml` 즉시 업데이트

### 9.2 학습 결과 변화
- **예상**: Softmax → Top-k로 인해 초기 loss 높을 수 있음
- **대응**: k_percent 조정 (0.5 ~ 0.7 실험)

### 9.3 Selection 불균형
- **예상**: 일부 batch에서 선택 비율 편차
- **대응**: Batch-wise top-k가 아닌 sample-wise top-k 실험 가능

---

## 10. 참고 자료

**Rho-1 논문:**
- Lin et al. "Rho-1: Not All Tokens Are What You Need" (NeurIPS 2024)
- https://arxiv.org/abs/2404.07965
- Excess Loss: ℒ_Δ(x_i) = ℒ_θ(x_i) − ℒ_ref(x_i)
- Top-k selection: I_k%(x_i) = 1 if top k%, else 0

**MTP 특성:**
- n_future_tokens = 4
- Position t에서 t+1, t+2, t+3, t+4 동시 예측
- Reference는 NTP (single head)

**개발원칙:**
- 앞/뒤 흐름 확인 후 구조 파악
- 기존 정확한 로직 유지, 중복/오류만 제거
- 하위 호환성 무시, 깔끔하게 재작성
- 계획과 비교하여 방향성 검증
