# Random Window Chunking 구현 계획서

## 배경

### 문제 상황
- **데이터 길이 편향**: correct 평균 171 토큰, incorrect 평균 226 토큰 (24% 차이)
- **Position correlation**: -0.4 ~ -0.5 (80step) - 시퀀스 뒤로 갈수록 V 감소 패턴 학습
- **기존 대응**: per-sequence mean 이미 적용됨 → 불충분

### 해결 방향
긴 시퀀스가 loss에 과도하게 기여하는 것을 방지하기 위해, output 토큰에서 **랜덤한 고정 길이 창**만 학습 대상으로 선택.

업계 표준 방식:
- LLaMA 사전학습: concat-and-chunk로 고정 길이 서브시퀀스 학습
- TRL/RLHF: response length normalization, reward 정규화

---

## 데이터 통계 기반 파라미터 설계

### Train 데이터 output_tokens 분포

| 메트릭 | Correct | Incorrect |
|--------|---------|-----------|
| mean | 170.77 | 225.88 |
| median | 120 | 165 |
| p25 | 76 | 95 |
| p75 | 201 | 285 |
| min | 5 | 2 |
| max | 1853 | 1848 |

### 파라미터 결정 근거

| 파라미터 | 값 | 근거 |
|----------|------|------|
| `min_window_size` | **128** | correct median(120) 근처, 50% 샘플에 윈도우 적용 |
| `window_size` | **192** | correct p75(201) 이하, 학습 토큰 수 균등화 |
| `preserve_eos` | **True** | Lambda Return terminal reward 학습 보장 |

### 예상 윈도우 적용률

```
min_window_size=128 기준:
- correct: ~50% (128+ 토큰)에 윈도우 적용
- incorrect: ~65% (128+ 토큰)에 윈도우 적용

window_size=192 적용 후:
- 모든 윈도우 적용 샘플이 동일한 192 토큰으로 균등화
- 길이 편향 효과 대폭 감소
```

---

## 현재 구현 분석

### 1. 데이터 흐름
```
datasets.py → PairwiseDataCollator → DataLoader → run_critic.py
     ↓                 ↓                              ↓
 pair 샘플링      labels 마스킹 생성          loss_mask = (labels != -100)
```

### 2. 현재 마스킹 로직 (`collators.py:254-304`)
```python
def _tokenize_sample(self, instruction, input_text, output):
    # 1. prompt 길이 계산
    len_prompt = len(prompt_tokens["input_ids"])

    # 2. 전체 토큰화
    tokenized = self.tokenizer(full_text, max_length=self.max_length, ...)

    # 3. labels 마스킹
    labels[:len_prompt] = -100          # instruction/input 마스킹
    labels[attention_mask == 0] = -100  # padding 마스킹
    # → output 토큰 전체가 학습 대상
```

### 3. Loss 계산 (`pairwise_utils.py`)
```python
# loss_mask = labels != -100 (output 토큰만 True)
seq_loss = (loss * combined_mask).sum(dim=1) / (seq_lengths + 1e-8)  # 시퀀스별 평균
masked_loss = seq_loss[valid_seq_mask].sum() / n_valid               # 배치 평균
```

### 4. 마스크 호환성 검증

```
시퀀스 구조:
[BOS][instruction...][### Response:][output tokens...][PAD...]
     |<-- len_prompt -->|<-- output 영역 -->|<-- attention=0 -->|
     0                  150                 350                 2048

attention_mask.sum() = 350 (유효 토큰 수)
output_start = len_prompt = 150
output_end = 350 (마지막 유효 토큰 + 1)
output_len = 200

→ attention_mask, padding 마스크와 완전 호환
```

---

## 구현 계획

### Phase 1: 핵심 함수 구현

**파일**: `src/weighted_mtp/data/collators.py`

**새 함수**: `apply_random_window_mask`

```python
import random


def apply_random_window_mask(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    len_prompt: int,
    window_size: int = 192,
    min_window_size: int = 128,
    preserve_eos: bool = True,
) -> torch.Tensor:
    """Output 토큰 중 랜덤 윈도우만 학습 대상으로 선택

    Args:
        labels: [seq_len] 원본 labels (-100으로 마스킹된 상태)
        attention_mask: [seq_len] 유효 토큰 마스크
        len_prompt: instruction + input 토큰 수 (제외 대상)
        window_size: 학습 대상 윈도우 크기 (기본 192)
        min_window_size: 최소 윈도우 크기 (기본 128, 이하면 전체 학습)
        preserve_eos: True면 EOS를 항상 윈도우에 포함 (terminal reward 학습 보장)

    Returns:
        labels: 윈도우 외 토큰이 -100으로 마스킹된 labels

    동작:
    1. output 토큰 범위 계산: [len_prompt, seq_end)
    2. output_len <= min_window_size: 전체 output 학습 (변경 없음)
    3. output_len > min_window_size:
       - preserve_eos=True: 윈도우 끝을 output_end에 고정, 앞쪽만 랜덤
       - preserve_eos=False: 윈도우 시작점 완전 랜덤
    """
    # output 범위 계산 (padding 제외)
    seq_end = attention_mask.sum().item()
    output_start = len_prompt
    output_end = seq_end
    output_len = output_end - output_start

    # 짧은 시퀀스는 전체 학습
    if output_len <= min_window_size:
        return labels

    # 윈도우 크기 결정 (output_len 이하로 제한)
    actual_window = min(window_size, output_len)

    if preserve_eos:
        # EOS를 항상 포함: 윈도우 끝을 output_end에 고정
        window_end = output_end
        window_start = window_end - actual_window
    else:
        # 완전 랜덤: 윈도우 시작점 랜덤 선택
        max_start = output_end - actual_window
        window_start = random.randint(output_start, max_start)
        window_end = window_start + actual_window

    # 윈도우 외 output 토큰 마스킹
    new_labels = labels.clone()
    if window_start > output_start:
        new_labels[output_start:window_start] = -100
    if window_end < output_end:
        new_labels[window_end:output_end] = -100

    return new_labels
```

**EOS 보존 동작 예시**:
```
preserve_eos=True, window_size=192:

원본: [instruction 150토큰][output 300토큰][PAD...]
                          |<-- 300 tokens -->|

윈도우 적용 후:
      [instruction 150토큰][masked 108토큰][학습 192토큰][PAD...]
                          |<-- -100 -->|<-- labels -->|

→ output 끝(EOS 포함)이 항상 학습 대상에 포함됨
→ Lambda Return의 terminal reward (G_T = R) 학습 보장
```

---

### Phase 2: PairwiseDataCollator 수정

**파일**: `src/weighted_mtp/data/collators.py`

**변경 사항**:

```python
@dataclass
class PairwiseDataCollator:
    """Pairwise Ranking 학습용 Collator"""

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    padding: str = "max_length"
    # Random Window 설정
    use_random_window: bool = False
    window_size: int = 192
    min_window_size: int = 128
    preserve_eos: bool = True

    def _tokenize_sample(
        self,
        instruction: str,
        input_text: str,
        output: str,
    ) -> dict[str, torch.Tensor]:
        """단일 샘플 토큰화"""
        # 1. Prompt 부분 토큰화 (output 제외) → 길이 계산용
        prompt_text = apply_alpaca_template(
            instruction, input_text, output="", include_response_header=True
        )
        prompt_tokens = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        len_prompt = len(prompt_tokens["input_ids"])

        # 2. 전체 텍스트 토큰화
        full_text = apply_alpaca_template(instruction, input_text, output)
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # 3. Labels 생성 및 기본 Masking
        labels = input_ids.clone()
        labels[:len_prompt] = -100          # Instruction + Input 마스킹
        labels[attention_mask == 0] = -100  # Padding 마스킹

        # 4. Random Window 적용 (옵션)
        if self.use_random_window:
            labels = apply_random_window_mask(
                labels=labels,
                attention_mask=attention_mask,
                len_prompt=len_prompt,
                window_size=self.window_size,
                min_window_size=self.min_window_size,
                preserve_eos=self.preserve_eos,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
```

---

### Phase 3: 설정 통합

**파일**: `configs/production/critic_mlp.yaml`

```yaml
data:
  sampling:
    use_pairwise: true
    n_samples: 200000
    max_pairs_per_problem: 50

  # Random Window 설정 (길이 편향 완화)
  collator:
    use_random_window: true
    window_size: 192          # 학습 윈도우 크기 (correct p75=201 기준)
    min_window_size: 128      # 이보다 짧으면 전체 학습 (correct median=120 기준)
    preserve_eos: true        # EOS 항상 포함 (terminal reward 학습 보장)
```

**파일**: `src/weighted_mtp/pipelines/run_critic.py`

```python
# DataLoader 생성 시 collator 설정 전달
collator_config = config.data.get("collator", {})
collator = PairwiseDataCollator(
    tokenizer=tokenizer,
    max_length=config.model.max_length,
    use_random_window=collator_config.get("use_random_window", False),
    window_size=collator_config.get("window_size", 192),
    min_window_size=collator_config.get("min_window_size", 128),
    preserve_eos=collator_config.get("preserve_eos", True),
)
```

---

### Phase 4: 검증 및 테스트

**테스트 케이스**:

1. **단위 테스트**: `apply_random_window_mask` 함수
   ```python
   def test_short_sequence_unchanged():
       """min_window_size 이하 시퀀스는 변경 없음"""
       labels = torch.tensor([1, 2, 3, 4, 5])  # 5 토큰
       attention_mask = torch.ones(5)
       result = apply_random_window_mask(labels, attention_mask, len_prompt=2,
                                         window_size=192, min_window_size=128)
       assert torch.equal(result, labels)

   def test_preserve_eos():
       """preserve_eos=True면 마지막 토큰 항상 포함"""
       labels = torch.arange(300)
       attention_mask = torch.ones(300)
       result = apply_random_window_mask(labels, attention_mask, len_prompt=50,
                                         window_size=192, preserve_eos=True)
       # 마지막 192개 토큰만 학습 대상
       assert (result[108:300] != -100).all()  # 윈도우 내
       assert (result[50:108] == -100).all()   # 윈도우 외

   def test_window_size_respected():
       """학습 대상 토큰 수가 window_size 이하"""
       labels = torch.arange(500)
       attention_mask = torch.ones(500)
       result = apply_random_window_mask(labels, attention_mask, len_prompt=100,
                                         window_size=192, min_window_size=128)
       n_train_tokens = (result != -100).sum().item()
       assert n_train_tokens == 192
   ```

2. **통합 테스트**: `PairwiseDataCollator`
   ```python
   def test_collator_random_window():
       """use_random_window=True시 labels 마스킹 검증"""
       collator = PairwiseDataCollator(tokenizer, use_random_window=True)
       batch = collator([sample])
       pos_labels = batch["pos_labels"][0]
       n_train = (pos_labels != -100).sum().item()
       assert n_train <= 192  # window_size 이하
   ```

3. **학습 메트릭 검증**:
   - `position_correlation` 개선 여부: -0.4 → -0.1 목표
   - `pairwise_accuracy` 유지 여부
   - loss 수렴 양상 비교

---

## 기대 효과

| 메트릭 | 현재 | 예상 |
|--------|------|------|
| pos_position_corr | -0.4 ~ -0.5 | -0.1 ~ 0.1 |
| neg_position_corr | -0.4 ~ -0.5 | -0.1 ~ 0.1 |
| 학습 토큰 수 균등성 | 불균등 (171 vs 226) | 균등 (최대 192) |
| EOS 학습 보장 | O | O (preserve_eos) |

---

## 구현 순서

| Phase | 작업 | 파일 | 예상 변경량 |
|-------|------|------|------------|
| 1 | `apply_random_window_mask` 함수 구현 | `collators.py` | +50 lines |
| 2 | `PairwiseDataCollator` 수정 | `collators.py` | +15 lines |
| 3 | config 및 run_critic.py 수정 | yaml, py | +15 lines |
| 4 | 테스트 작성 및 검증 | `tests/` | +60 lines |

---

## 설계 결정 사항

### Q1: preserve_eos=True가 기본값인 이유
Lambda Return 계산에서 terminal position의 타겟 G_T = R (reward)입니다.
EOS가 윈도우에서 제외되면 terminal reward 학습이 불가능해져 Value Model의
핵심 학습 신호가 손실됩니다.

### Q2: window_size=192 선택 이유
- correct p75 = 201: 75% 샘플이 201 이하
- 192로 설정하면 대부분의 correct 샘플 정보를 보존하면서
- incorrect의 긴 부분 (평균 226)을 효과적으로 truncate

### Q3: min_window_size=128 선택 이유
- correct median = 120: 50% 샘플이 120 이하
- 128 미만 샘플은 이미 충분히 짧아 길이 편향 영향 적음
- 너무 작은 윈도우는 정보 손실 우려

---

## 대안 비교

| 방안 | 장점 | 단점 | 채택 |
|------|------|------|------|
| **Random Window (본 계획)** | 구현 단순, 길이 균등화, EOS 보존 | 일부 정보 손실 | O |
| Length-matched Sampling | 데이터 분포 유지 | 데이터 활용률 감소 | X |
| Length Conditioning | 모델 레벨 해결 | 구조 변경 필요 | X |
| Relative Position Metric | 진단 개선 | 편향 자체 미해결 | 보조 |
