# GAE 기반 Advantage Weighting 개선 계획

## 1. 현황 분석

### 1.1 현재 구현 (1-step Marginal Value)

```python
# compute_td_errors() - td_weighting.py:110-181
δ_t = V(t) - V(t-1)      # Intermediate
δ_0 = V(0) - 0.5         # First token
δ_T = R - V(T)           # Terminal
```

**의미**: "t번째 토큰을 선택함으로써 가치가 얼마나 증가했는가"

### 1.2 문제점

| 문제 | 설명 |
|------|------|
| **Noise 증폭** | Pairwise ranking loss로 학습된 value model은 시퀀스 간 상대 순서만 학습하므로, token-level V(t) trajectory가 smooth하지 않음 |
| **Local 정보만 사용** | V(t) - V(t-1)은 인접 토큰 간 차이만 반영, 미래 정보 미활용 |
| **Terminal 신호 전파 부족** | 마지막 토큰의 R - V(T) 신호가 이전 토큰들에 직접 전파되지 않음 |

### 1.3 Pairwise Ranking Value Model 특성

```
Pairwise Ranking Loss:
  max(0, margin - (V_correct - V_incorrect))

학습 목표: 시퀀스 간 상대적 순서
  → token-level smoothness 보장 안 됨
  → V(t)가 급격히 변동할 수 있음
  → 1-step δ_t = V(t) - V(t-1)이 noisy
```

---

## 2. 개선 방향: GAE (Generalized Advantage Estimation)

### 2.1 GAE 수식

```
δ_t = r_t + γV(t+1) - V(t)     # 1-step TD error
A_t = Σ_{k=0}^{T-t} (γλ)^k δ_{t+k}  # GAE advantage

역방향 계산:
  A_T = δ_T = R - V(T)
  A_t = δ_t + γλ * A_{t+1}
```

### 2.2 λ 파라미터 효과

| λ | 방식 | Bias | Variance | 설명 |
|---|------|------|----------|------|
| 0 | TD(0) | High | Low | 1-step bootstrap, 빠르지만 biased |
| 0.95 | GAE | Medium | Medium | 권장값, bias-variance 균형 |
| 1.0 | MC | Zero | High | Monte Carlo, unbiased but noisy |

### 2.3 LLM 환경 특수성

```
r_t = 0 for t < T (intermediate reward 없음)
r_T = R (terminal reward만 존재)

따라서:
  δ_t = γV(t+1) - V(t)  for t < T
  δ_T = R - V(T)        for t = T

GAE 효과:
  - Terminal의 R - V(T) 신호가 역전파됨
  - λ=0.95면 10 step 이전까지 신호 전달 (0.95^10 ≈ 0.60)
  - Noise 평활화 효과
```

---

## 3. 기존 코드 분석

### 3.1 이미 구현된 함수

```python
# td_weighting.py:15-107
def compute_td_targets(value_logits, rewards, loss_mask, gamma=1.0, lam=0.0):
    """GAE 기반 TD targets 계산

    Returns:
        td_targets: [batch, seq, 1] = V(t) + A_t
    """
```

**핵심**: `compute_td_targets()`는 이미 GAE를 구현하고 있음. 단, 현재 사용되지 않음.

### 3.2 현재 사용 흐름

```
run_verifiable.py:
  compute_td_errors() → δ_t = V(t) - V(t-1)
                     ↓
  build_weights(td_errors, ...) → exp(normalized_δ / β)
```

### 3.3 개선 후 흐름

```
run_verifiable.py:
  compute_td_targets(lam=0.95) → targets = V(t) + A_t
                              ↓
  advantage = targets - V(t) = A_t (GAE advantage)
                              ↓
  build_weights(advantage, ...) → exp(normalized_A / β)
```

---

## 4. Phase별 개선 계획

### Phase 1: GAE Advantage 함수 추가 (신규)

**목표**: `compute_gae_advantage()` 함수 추가

```python
# td_weighting.py에 추가
def compute_gae_advantage(
    value_logits: torch.Tensor,
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> torch.Tensor:
    """GAE 기반 Advantage 계산

    A_t = Σ (γλ)^k δ_{t+k}

    Args:
        lam: GAE lambda (0=TD(0), 0.95=GAE 권장, 1.0=MC)

    Returns:
        advantages: [batch, seq] GAE advantage
    """
    targets = compute_td_targets(value_logits, rewards, loss_mask, gamma, lam)
    values = value_logits.squeeze(-1).detach()
    advantages = targets.squeeze(-1) - values
    return advantages * loss_mask.float()
```

**변경 파일**: `src/weighted_mtp/value_weighting/td_weighting.py`

---

### Phase 2: Config 확장

**목표**: `td_lambda` 파라미터 추가

```yaml
# configs/production/verifiable.yaml
training:
  # 기존
  beta: 1.0
  weight_clip_min: 0.1
  weight_clip_max: 3.0

  # 추가
  td_lambda: 0.95  # GAE lambda (0=TD(0), 0.95=GAE, 1.0=MC)
```

**변경 파일**: `configs/production/verifiable.yaml`

---

### Phase 3: Pipeline 수정

**목표**: `compute_td_errors` → `compute_gae_advantage` 교체

```python
# run_verifiable.py 수정

# Before
from weighted_mtp.value_weighting.td_weighting import (
    compute_td_errors,
    ...
)

td_errors = compute_td_errors(value_logits, rewards, loss_mask, gamma=1.0)

# After
from weighted_mtp.value_weighting.td_weighting import (
    compute_gae_advantage,
    ...
)

td_lambda = config.training.get("td_lambda", 0.95)
advantages = compute_gae_advantage(
    value_logits, rewards, loss_mask, gamma=1.0, lam=td_lambda
)
```

**변경 파일**: `src/weighted_mtp/pipelines/run_verifiable.py`

**수정 위치**:
1. import 문 (51-55줄)
2. Validation loop 내 TD error 계산 (181-186줄)
3. Training loop 내 TD error 계산 (504-510줄)
4. 변수명 `td_errors` → `advantages` (일관성)

---

### Phase 4: EMA 및 로깅 호환

**목표**: TDStatsEMA가 advantage에도 동작하도록 확인

```python
# 현재 EMA 사용 (변경 불필요)
td_ema.update(advantages, loss_mask, distributed=True)  # td_errors 대신 advantages

# 로깅 메트릭명 변경 (선택적)
"td/mean" → "advantage/mean"
"td/std" → "advantage/std"
```

**변경 파일**: `src/weighted_mtp/pipelines/run_verifiable.py` (로깅 부분)

---

## 5. 하위 호환성 고려

### 5.1 원칙 4 적용 (하위 호환성 미고려)

```
기존: compute_td_errors() - 1-step marginal value
신규: compute_gae_advantage() - GAE advantage

→ compute_td_errors()는 그대로 유지 (다른 용도로 사용 가능)
→ verifiable pipeline에서만 compute_gae_advantage() 사용
→ 불필요한 fallback 없음
```

### 5.2 Config 기본값

```yaml
td_lambda: 0.95  # 기본값 (config에 없으면 0.95 사용)
```

---

## 6. 검증 계획

### 6.1 단위 테스트

```python
def test_gae_advantage():
    # λ=0: TD(0)과 동일해야 함
    # λ=1: MC와 동일해야 함 (A_t = R - V(t))
    # λ=0.95: 중간값
```

### 6.2 학습 검증 메트릭

| 메트릭 | 기대 효과 |
|--------|----------|
| `advantage/std` | 1-step δ 대비 감소 예상 (smoothing) |
| `weight/clipping_ratio` | 극단값 감소로 clipping 비율 감소 예상 |
| `train/loss_ratio` | 더 의미있는 weighting으로 1.0에서 벗어남 예상 |

### 6.3 A/B 비교 실험

```
실험 A: td_lambda=0 (기존 1-step과 유사)
실험 B: td_lambda=0.95 (GAE)
실험 C: td_lambda=1.0 (MC)

비교 지표:
- Validation CE loss
- Weight distribution
- 학습 안정성 (grad_norm 변동)
```

---

## 7. 예상 효과

### 7.1 Noise 감소

```
Before (1-step):
  δ_t = V(t) - V(t-1)
  → V(t)가 noisy하면 δ_t도 noisy

After (GAE λ=0.95):
  A_t = δ_t + 0.95*A_{t+1}
  → 미래 신호의 가중 평균으로 smoothing
```

### 7.2 Terminal 신호 전파

```
Before: δ_T = R - V(T)는 terminal에만 영향
After: A_{t} = ... + (0.95)^{T-t} * δ_T
       → Terminal 신호가 역전파됨
```

### 7.3 의미론적 일관성

```
Before: "이 토큰이 가치를 증가시켰는가" (local)
After: "이 시점의 선택이 장기적으로 좋았는가" (global)
```

---

## 8. 구현 우선순위

| Phase | 작업 | 복잡도 | 우선순위 |
|-------|------|--------|----------|
| 1 | `compute_gae_advantage()` 추가 | Low | 1 |
| 2 | Config `td_lambda` 추가 | Low | 2 |
| 3 | Pipeline 수정 | Medium | 3 |
| 4 | 로깅 메트릭명 정리 | Low | 4 |

**총 예상 변경**: 약 50줄 수정/추가

---

## 9. Pairwise Ranking Value Model 호환 (추가)

### 9.1 문제

Pairwise Ranking으로 학습된 Value Model은:
- 절대값이 아닌 상대적 순서만 학습 (V_correct > V_incorrect)
- V의 스케일이 [0, 1] 보장 없음 (예: 0.3 ~ 0.8 범위)
- 외부 reward R=1.0과 스케일 불일치

### 9.2 해결: Terminal을 V_T로 통일

```
모든 δ를 동일한 공식으로:
  δ_t = γ*V_t - V_{t-1}  (terminal 포함)

의미:
  - 외부 reward(R) 대신 V_T 자체가 "정답 확률" 반영
  - Correct 시퀀스: V_T 높음 → δ_T = V_T - V_{T-1} (긍정적 기여 가능)
  - 스케일 일관성 확보
```

### 9.3 구현 변경

```python
# 기존 (R 사용)
if t == term_idx:
    delta = rewards[b].item() - prev_value  # R - V_{T-1}

# 변경 (V_T 사용)
delta = gamma * current_value - prev_value  # 모든 t에 동일
```

### 9.4 시점 정렬 수정

기존 코드에서 발견된 시점 오류도 함께 수정:

| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| Intermediate | δ_t = γ*V_{t+1} - V_t | δ_t = γ*V_t - V_{t-1} |
| Terminal | δ_T = R - V_T | δ_T = γ*V_T - V_{T-1} |
| 의미 | 토큰 t+1의 기여가 t에 저장 | 토큰 t의 기여가 t에 저장 |

---

## 10. 결론

1. **현재 문제**: 1-step δ = V(t) - V(t-1)은 pairwise ranking value model의 noisy V(t) trajectory에서 증폭됨

2. **해결책**: GAE (λ=0.95)로 미래 정보를 역전파하여 smoothing

3. **Pairwise Ranking 호환**: Terminal reward를 V_T로 통일하여 스케일 일관성 확보

4. **시점 정렬**: δ_t = V_t - V_{t-1}로 토큰 t의 기여가 올바르게 t 위치에 저장

5. **리스크**: 낮음 - 기존 코드 구조 유지, 일관된 수식 적용
