# Random Window + Terminal Position 분리 구현 계획서

## 배경

### 문제 상황
1. **길이 편향**: correct 평균 171 토큰, incorrect 평균 226 토큰 (24% 차이)
2. **Position correlation**: -0.4 ~ -0.5 (시퀀스 뒤로 갈수록 V 감소 패턴)
3. **현재 한계**: `preserve_eos=True` 시 고정 윈도우 (마지막 192토큰만 학습)

### 요구사항
- 고정 윈도우 크기 (192) 유지
- 윈도우 **시작 위치를 랜덤**하게 선택
- Lambda Return의 **terminal reward는 진짜 EOS**에서 시작

### 핵심 아이디어
```
Terminal Position과 학습 윈도우를 분리:

┌─────────────────────────────────────────────────────────────┐
│ [Instruction]  [Output 토큰들...]                    [PAD]  │
│ ◀── masked ──▶ ◀─────── output 영역 ──────────────▶        │
│                 ◀── 학습 윈도우 (192) ──▶                   │
│                                              ▲              │
│                                         Terminal (EOS)      │
│                                         G_T = R 설정        │
└─────────────────────────────────────────────────────────────┘

- loss_mask: 학습 윈도우 내 토큰만 1
- attention_mask: 전체 output 영역 (EOS 위치 결정용)
- Lambda Return: attention_mask 기준 EOS에서 역방향 전파
- Loss 계산: loss_mask 범위만 사용
```

---

## 마스킹 체계 정의

### 1. 마스크 종류 및 역할

| 마스크 | 생성 위치 | 역할 | 값=1 범위 |
|--------|----------|------|-----------|
| `attention_mask` | tokenizer | Padding 제외, FSDP 호환 | BOS + Instruction + Output |
| `labels` | collator | Loss 계산 대상 | Output 중 학습 윈도우만 |
| `loss_mask` | run_critic | `labels != -100` | 학습 윈도우 |
| `output_end_mask` | 신규 | Terminal position 식별 | 전체 Output (윈도우 무관) |

### 2. 시퀀스 구조 예시

```
Position:     0   1   2   ...  150  151  ...  340  341  ...  2048
              ├───────────────┼─────────────────┼────────────────┤
Token:        BOS [Instruction] [   Output    ] [    PAD       ]

attention_mask: 1   1   1   ...   1    1   ...   1    0   ...   0
labels:        -100 -100 -100 ... -100  tok ...  tok -100 ... -100
                                   ↑              ↑
                             window_start    window_end (≠ EOS)

output_end_mask: 0   0   0   ...   0    0   ...   1    0   ...   0
                                              ↑
                                        진짜 EOS (Terminal)
```

---

## 구현 계획

### Phase 1: 기존 코드 정리

**목표**: `min_window_size` 완전 제거, 테스트 정상화

**수정 파일**:

| 파일 | 변경 내용 |
|------|----------|
| `collators.py` | `min_window_size` 파라미터 제거 (완료) |
| `dataloader.py` | `min_window_size` 전달 제거 (완료) |
| `run_critic.py` | 로깅에서 `min_window_size` 제거 (완료) |
| `critic_mlp.yaml` | `min_window_size` 설정 제거 (완료) |
| `test_collators.py` | 테스트에서 `min_window_size` 제거 |

**검증**: 모든 테스트 통과

---

### Phase 2: Terminal Position 분리

**목표**: Lambda Return 계산 시 진짜 EOS 위치 사용

**파일**: `src/weighted_mtp/utils/pairwise_utils.py`

**2-1. `create_output_end_mask` 함수 추가**

```python
def create_output_end_mask(
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Output 영역의 마지막 토큰(EOS) 위치 마스크 생성

    Random Window 적용 여부와 무관하게 진짜 output 끝 위치를 식별.
    Lambda Return의 terminal position (G_T = R) 설정에 사용.

    Args:
        attention_mask: [batch, seq] Padding 제외 마스크
        labels: [batch, seq] 원본 labels (윈도우 마스킹 적용 전 또는 후)

    Returns:
        output_end_mask: [batch, seq] output 끝 위치만 1

    로직:
        1. attention_mask로 유효 토큰 범위 파악
        2. attention_mask 기준 마지막 유효 토큰 = output 끝 (EOS)
        3. 해당 위치만 1인 마스크 반환
    """
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device

    # 각 시퀀스의 마지막 유효 토큰 위치 계산
    # attention_mask.sum() - 1 = 마지막 유효 인덱스
    seq_lengths = attention_mask.sum(dim=1)  # [batch]
    last_valid_indices = (seq_lengths - 1).clamp(min=0).long()  # [batch]

    # EOS 마스크 생성
    output_end_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
    batch_indices = torch.arange(batch_size, device=device)

    # 유효 토큰이 있는 시퀀스만 마킹
    has_valid = seq_lengths > 0
    output_end_mask[batch_indices[has_valid], last_valid_indices[has_valid]] = 1.0

    return output_end_mask
```

**2-2. `compute_lambda_return` 수정**

```python
def compute_lambda_return(
    values: torch.Tensor,
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    output_end_mask: Optional[torch.Tensor] = None,  # 추가
) -> torch.Tensor:
    """Fitted λ-Return 타겟 계산

    Args:
        values: [batch, seq] Value model 예측
        rewards: [batch] Terminal reward
        loss_mask: [batch, seq] 학습 대상 토큰 마스크 (윈도우 적용됨)
        gamma: Discount factor
        lam: GAE smoothing factor
        output_end_mask: [batch, seq] 진짜 EOS 위치 마스크 (옵션)
            - 제공되면: EOS 위치에서 G_T = R 설정
            - 미제공: loss_mask 기준 마지막 위치 사용 (기존 동작)

    Returns:
        lambda_returns: [batch, seq] 위치별 λ-return 타겟
    """
    batch_size, seq_len = values.shape
    device = values.device
    dtype = values.dtype

    lambda_returns = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

    for b in range(batch_size):
        # Terminal position 결정
        if output_end_mask is not None:
            # 진짜 EOS 위치 사용
            eos_pos = output_end_mask[b].nonzero(as_tuple=True)[0]
            if len(eos_pos) == 0:
                continue
            terminal_pos = eos_pos[0].item()
        else:
            # 기존 동작: loss_mask 기준
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
            terminal_pos = valid_positions[-1].item()

        # G_T = R (terminal position)
        lambda_returns[b, terminal_pos] = rewards[b]

        # 역방향 전파: output 전체 범위에서 계산
        # loss_mask와 무관하게 모든 output 토큰에 대해 λ-return 계산
        G_next = rewards[b]

        # terminal_pos부터 역방향으로 전파
        for t in range(terminal_pos - 1, -1, -1):
            V_next = values[b, t + 1]
            td_component = (1 - lam) * gamma * V_next
            mc_component = lam * gamma * G_next
            G_t = td_component + mc_component

            lambda_returns[b, t] = G_t
            G_next = G_t

    return lambda_returns
```

**2-3. `compute_lambda_value_loss` 수정**

```python
def compute_lambda_value_loss(
    value_logits: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    loss_type: str = "huber",
    huber_delta: float = 0.5,
    output_end_mask: Optional[torch.Tensor] = None,  # 추가
) -> torch.Tensor:
    """λ-Return 기반 Value Loss

    Args:
        ...기존 파라미터...
        output_end_mask: [batch, seq] 진짜 EOS 위치 마스크 (옵션)
    """
    values = value_logits.squeeze(-1)

    # λ-return 타겟 계산 (output_end_mask 전달)
    with torch.no_grad():
        lambda_targets = compute_lambda_return(
            values.detach(), rewards, loss_mask, gamma, lam,
            output_end_mask=output_end_mask,  # 추가
        )

    # Loss 계산: loss_mask 범위만 사용 (윈도우 내 토큰)
    combined_mask = attention_mask * loss_mask

    if loss_type == "huber":
        loss = F.smooth_l1_loss(values, lambda_targets, reduction='none', beta=huber_delta)
    else:
        loss = (values - lambda_targets) ** 2

    # 시퀀스별 평균 후 배치 평균
    seq_lengths = combined_mask.sum(dim=1)
    valid_seq_mask = seq_lengths > 0
    n_valid = valid_seq_mask.sum()

    if n_valid == 0:
        return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

    seq_loss = (loss * combined_mask).sum(dim=1) / (seq_lengths + 1e-8)
    masked_loss = seq_loss[valid_seq_mask].sum() / n_valid

    return masked_loss
```

---

### Phase 3: Random Window 시작 위치 랜덤화

**목표**: 윈도우 시작 위치를 랜덤하게 선택 (EOS는 별도 관리)

**파일**: `src/weighted_mtp/data/collators.py`

**3-1. `apply_random_window_mask` 수정**

```python
def apply_random_window_mask(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    len_prompt: int,
    window_size: int = 192,
) -> torch.Tensor:
    """Output 토큰 중 랜덤 위치의 고정 길이 윈도우만 학습 대상으로 선택

    Terminal position(EOS)은 Lambda Return 계산에서 별도 관리되므로,
    윈도우 시작 위치를 자유롭게 랜덤 선택 가능.

    Args:
        labels: [seq_len] 원본 labels (instruction/padding은 이미 -100)
        attention_mask: [seq_len] 유효 토큰 마스크
        len_prompt: instruction + input 토큰 수 (output 시작 위치)
        window_size: 학습 대상 윈도우 크기

    Returns:
        윈도우 외 output 토큰이 -100으로 마스킹된 labels

    동작:
        1. output 범위 계산: [len_prompt, seq_end)
        2. output_len <= window_size: 전체 output 학습
        3. output_len > window_size: 랜덤 시작 위치에서 window_size만큼 학습
    """
    seq_end = int(attention_mask.sum().item())
    output_start = len_prompt
    output_end = seq_end
    output_len = output_end - output_start

    # window_size 이하면 전체 학습
    if output_len <= window_size:
        return labels

    # 랜덤 시작 위치 선택
    max_start = output_end - window_size
    window_start = random.randint(output_start, max_start)
    window_end = window_start + window_size

    # 윈도우 외 output 토큰 마스킹
    new_labels = labels.clone()
    if window_start > output_start:
        new_labels[output_start:window_start] = -100
    if window_end < output_end:
        new_labels[window_end:output_end] = -100

    return new_labels
```

**3-2. `preserve_eos` 파라미터 제거**

- `PairwiseDataCollator`에서 `preserve_eos` 필드 제거
- `dataloader.py`에서 `preserve_eos` 전달 제거
- `critic_mlp.yaml`에서 `preserve_eos` 설정 제거

---

### Phase 4: run_critic.py 통합

**목표**: Lambda Return 계산 시 `output_end_mask` 전달

**파일**: `src/weighted_mtp/pipelines/run_critic.py`

**4-1. Training loop 수정**

```python
# Training loop 내부
pos_loss_mask = (pos_labels != -100)
neg_loss_mask = (neg_labels != -100)

# output_end_mask 생성 (진짜 EOS 위치)
from weighted_mtp.utils.pairwise_utils import create_output_end_mask
pos_output_end_mask = create_output_end_mask(pos_attention_mask, pos_labels)
neg_output_end_mask = create_output_end_mask(neg_attention_mask, neg_labels)

# Lambda Return Loss 계산
if loss_type == "lambda_return":
    pos_lambda_loss = compute_lambda_value_loss(
        pos_value_logits, pos_rewards, pos_attention_mask, pos_loss_mask.float(),
        gamma=lambda_gamma, lam=current_lam,
        loss_type=value_loss_fn, huber_delta=huber_delta,
        output_end_mask=pos_output_end_mask,  # 추가
    )
    neg_lambda_loss = compute_lambda_value_loss(
        neg_value_logits, neg_rewards, neg_attention_mask, neg_loss_mask.float(),
        gamma=lambda_gamma, lam=current_lam,
        loss_type=value_loss_fn, huber_delta=huber_delta,
        output_end_mask=neg_output_end_mask,  # 추가
    )
```

**4-2. Validation loop 동일하게 수정**

---

### Phase 5: 테스트 및 검증

**5-1. 단위 테스트**

```python
class TestRandomWindowWithEOS:
    """Random Window + EOS 분리 테스트"""

    def test_output_end_mask_creation(self):
        """output_end_mask가 정확한 EOS 위치를 가리키는지 검증"""
        attention_mask = torch.tensor([[1,1,1,1,1,0,0]])  # 5개 유효 토큰
        labels = torch.tensor([[-100,-100,1,2,-100,-100,-100]])  # 윈도우: [2,3]

        output_end_mask = create_output_end_mask(attention_mask, labels)

        # EOS는 attention_mask 기준 마지막 = index 4
        assert output_end_mask[0, 4] == 1.0
        assert output_end_mask.sum() == 1.0

    def test_lambda_return_with_random_window(self):
        """랜덤 윈도우에서도 terminal reward가 EOS에 설정되는지 검증"""
        values = torch.zeros(1, 10)
        rewards = torch.tensor([1.0])
        loss_mask = torch.tensor([[0,0,1,1,1,0,0,0,0,0]])  # 윈도우: [2,3,4]
        output_end_mask = torch.tensor([[0,0,0,0,0,0,0,1,0,0]])  # EOS: 7

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.95,
            output_end_mask=output_end_mask,
        )

        # Terminal reward는 EOS(7)에 설정
        assert lambda_returns[0, 7] == 1.0
        # 윈도우 내 토큰도 역방향 전파된 값을 가짐
        assert lambda_returns[0, 4] > 0  # EOS 근처

    def test_loss_only_on_window(self):
        """Loss 계산이 윈도우 내 토큰에서만 이루어지는지 검증"""
        # ... loss_mask 범위만 loss 계산에 포함되는지 검증
```

**5-2. FSDP 호환성 검증**

```python
def test_fsdp_compatibility():
    """분산 환경에서 마스크 처리가 올바른지 검증"""
    # 각 GPU에서 독립적으로 처리되는지 확인
    # attention_mask, loss_mask, output_end_mask 모두 동일 shape
```

**5-3. 통합 테스트 (실제 학습)**

```
검증 항목:
1. position_correlation 개선: -0.4 → -0.1 목표
2. pairwise_accuracy 유지
3. Loss 수렴 양상 정상
4. Terminal reward가 EOS에서 시작하는지 로그 확인
```

---

## 마스킹 흐름 요약

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Data Flow                                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1. Tokenizer                                                             │
│     ├── input_ids: [BOS, Inst..., Output..., PAD...]                     │
│     └── attention_mask: [1,1,1,...,1,1,0,0,0]                            │
│                                                                           │
│  2. Collator (apply_random_window_mask)                                  │
│     ├── len_prompt 계산 (Instruction 끝 위치)                            │
│     ├── labels 생성: Instruction=-100, Output=token_ids, Padding=-100    │
│     └── Random Window 적용: 윈도우 외 Output도 -100                      │
│                                                                           │
│  3. run_critic.py                                                         │
│     ├── loss_mask = (labels != -100)  # 윈도우 내 토큰만                  │
│     ├── output_end_mask = create_output_end_mask(attention_mask)         │
│     │                     # attention_mask 기준 마지막 유효 토큰 = EOS    │
│     │                                                                     │
│     ├── compute_lambda_return(...)                                        │
│     │   ├── Terminal: output_end_mask 기준 EOS에 G_T = R                 │
│     │   └── 역방향 전파: 전체 output 범위                                │
│     │                                                                     │
│     └── compute_lambda_value_loss(...)                                    │
│         └── Loss: loss_mask 범위만 (윈도우 내 토큰)                       │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 구현 순서

| Phase | 작업 | 파일 | 예상 변경량 |
|-------|------|------|------------|
| 1 | min_window_size 정리 | test_collators.py | -10 lines |
| 2 | Terminal Position 분리 | pairwise_utils.py | +50 lines |
| 3 | Random Window 랜덤화 | collators.py, dataloader.py, yaml | +10/-20 lines |
| 4 | run_critic.py 통합 | run_critic.py | +20 lines |
| 5 | 테스트 및 검증 | tests/ | +80 lines |

---

## 개발 원칙 체크리스트

- [x] **원칙 1**: 앞/뒤 흐름 분석 (attention_mask → labels → loss_mask → Lambda Return)
- [x] **원칙 2**: 기존 구조 존중 (create_dataloader, compute_lambda_return 확장)
- [x] **원칙 3**: 불필요한 중복 제거 (preserve_eos 파라미터 → output_end_mask로 대체)
- [x] **원칙 4**: 하위 호환성 미고려, 깨끗한 구현
- [x] **원칙 4-1**: 변수명 통일 (output_end_mask, loss_mask, attention_mask)
- [x] **원칙 4-2**: wrapper 최소화 (create_output_end_mask만 추가)
- [x] **원칙 4-3**: 한글 주석, 불필요한 버전 주석 제외
- [ ] **원칙 5**: Phase별 구현 후 계획서와 비교 검토
- [x] **원칙 6**: 패키지 의존성 도구 활용 (uv)

---

## 구현 전제 조건 (Hidden Cost)

### 물리적 전제
**"모델은 반드시 EOS 위치까지 Forward Pass를 수행해야 한다."**

### Input Truncation 불가

```
❌ 잘못된 방식 (Input 자르기):
   input_ids = input_ids[:192]  # 물리적으로 자름
   → 모델은 EOS($t=350$)의 V(s_350)를 알 수 없음
   → Lambda Return의 역방향 전파 불가능

✅ 올바른 방식 (Mask만 적용):
   input_ids = 전체 시퀀스 (Padding 전까지)
   labels = 윈도우 외 토큰만 -100 (학습 제외)
   → 모델은 전체 시퀀스 Forward
   → EOS까지의 V 값 모두 계산됨
   → Lambda Return 정확히 역방향 전파
```

### 비용 vs 이득

| 항목 | 비용 | 이득 |
|------|------|------|
| **연산** | 192개만 학습해도 전체 길이(500~1000) Forward/Backward | 완벽한 랜덤성 |
| **메모리** | Last-N Truncation보다 비효율적 | 정확한 보상 전파 |
| **속도** | 전체 시퀀스 연산 필요 | 길이 편향 완전 제거 |

### 인프라 적합성

```
2.7B 모델 + H200 3장 (240GB VRAM):
- 전체 시퀀스 Forward/Backward 충분히 감당 가능
- Activation Checkpointing으로 메모리 최적화
- FSDP로 파라미터 샤딩
```

### 현재 구현 검증

현재 코드가 이미 이 전제를 만족하는지 확인:

```python
# collators.py - apply_random_window_mask
# input_ids는 변경하지 않음, labels만 마스킹
new_labels = labels.clone()
new_labels[output_start:window_start] = -100  # 학습 제외할 뿐, 입력은 유지

# dataloader.py - PairwiseDataCollator
# input_ids 전체 반환, labels만 마스킹됨
return {
    "input_ids": input_ids,       # 전체 시퀀스 (자르지 않음)
    "attention_mask": attention_mask,
    "labels": labels,             # 윈도우만 학습 대상
}
```

**결론**: 현재 구현이 이미 "전체 시퀀스 입력 + 마스크 기반 학습"을 사용하므로, 추가 수정 없이 이 전제를 만족합니다.

---

## 기대 효과

| 메트릭 | 현재 | 예상 |
|--------|------|------|
| pos_position_corr | -0.4 ~ -0.5 | -0.1 ~ 0.1 |
| neg_position_corr | -0.4 ~ -0.5 | -0.1 ~ 0.1 |
| 학습 토큰 균등성 | 불균등 (171 vs 226) | 균등 (최대 192) |
| Terminal reward 정확성 | 윈도우 끝 | 진짜 EOS |
| 윈도우 다양성 | 고정 (마지막 192) | 랜덤 위치 |
