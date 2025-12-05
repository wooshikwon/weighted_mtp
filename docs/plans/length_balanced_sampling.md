# Length-Balanced Pairwise Sampling 구현 계획서

## 배경

### 문제 상황
- **길이 편향**: Correct 평균 169.6 토큰, Incorrect 평균 222.1 토큰 (31% 차이)
- **Length Shortcut**: 300+ 토큰 구간에서 Incorrect 비율 65-69%
- **Random Window 한계**: Loss 계산 범위만 제한, Forward Pass는 전체 시퀀스 → 모델이 attention sequence length 차이 학습 가능

### 분석 결과
```
2D Binning (Difficulty x Length Bin) 적용 시:
- 총 unique 쌍: 711,584 (목표 20만의 3.6배)
- 길이 편향: 52.5 토큰 (31%) → 2.2 토큰 (1.0%)
- 97% 편향 제거
```

### 핵심 아이디어
```
같은 problem 내 + 같은 length bin 내에서만 correct/incorrect 쌍 생성
→ attention sequence length 차이 최소화
→ 모델이 길이 shortcut 학습 불가

기존 방식:
  Problem A: correct(150 tokens) ↔ incorrect(400 tokens)  # 길이 차이 큼

새 방식:
  Problem A, Bin 100-150: correct(120) ↔ incorrect(135)   # 같은 bin 내 매칭
  Problem A, Bin 300-500: correct(350) ↔ incorrect(380)   # 같은 bin 내 매칭
```

---

## 설계

### 1. 기존 방식 보존
- `_sample_unique_pairs` 함수는 그대로 유지
- `length_bins` 옵션이 없으면 기존 동작 (difficulty만 고려)
- critic_mlp 전용 옵션으로 추가

### 2. 새로운 함수 추가
```python
def _sample_length_balanced_pairs(
    problem_index_map: dict,
    n_samples: int,
    length_bins: list[int],           # [0, 100, 150, 200, 300, 500, 1000]
    seed: int,
    max_pairs_per_problem: int = None,
) -> list[dict]:
    """Length-balanced unique pair 샘플링

    같은 problem 내 + 같은 length bin 내에서 1:1 unique 매칭.
    가용량 비례로 샘플링하여 데이터 효율 최대화.
    """
```

### 3. Config 확장
```yaml
data_sampling:
  # 기존 옵션 유지
  use_pairwise: true
  n_samples: 200000

  # Length-balanced 샘플링 (critic_mlp 전용)
  use_length_balanced: true
  length_bins: [0, 100, 150, 200, 300, 500, 1000, 2000]
```

---

## 구현 계획

### Phase 1: 메타데이터 확장

**목표**: problem_index_map에 output 토큰 길이 정보 추가

**파일**: `scripts/create_storage/extract_metadata.py`

**변경 사항**:
```python
# 기존 메타데이터 구조
{
    "problem_id": {
        "difficulty": 7,
        "correct_indices": [0, 1, 2],
        "incorrect_indices": [3, 4, 5],
    }
}

# 확장된 구조
{
    "problem_id": {
        "difficulty": 7,
        "correct_indices": [0, 1, 2],
        "incorrect_indices": [3, 4, 5],
        "correct_token_lengths": [150, 200, 180],    # 각 인덱스별 토큰 길이
        "incorrect_token_lengths": [220, 350, 190],
    }
}
```

**검증**: 메타데이터 재생성 후 길이 정보 확인

---

### Phase 2: Length-Balanced 샘플링 함수 구현

**목표**: 2D binning 기반 샘플링 로직 구현

**파일**: `src/weighted_mtp/data/datasets.py`

**새 함수**: `_sample_length_balanced_pairs`

```python
def _sample_length_balanced_pairs(
    problem_index_map: dict[str, dict],
    n_samples: int,
    length_bins: list[int],
    seed: int,
    max_pairs_per_problem: Optional[int] = None,
) -> list[dict]:
    """Length-balanced unique pair 샘플링

    같은 problem 내 + 같은 length bin 내에서 1:1 unique 매칭.
    각 샘플은 최대 1회만 사용 (uniqueness 보장).

    Args:
        problem_index_map: {problem_id: {correct_indices, incorrect_indices,
                                         correct_token_lengths, incorrect_token_lengths}}
        n_samples: 목표 쌍 수
        length_bins: 길이 구간 경계 [0, 100, 150, 200, 300, 500, 1000]
        seed: 랜덤 시드
        max_pairs_per_problem: problem당 최대 쌍 수

    Returns:
        [{"correct_idx": int, "incorrect_idx": int, "problem_id": str, "length_bin": str}, ...]

    알고리즘:
        1. 각 problem 내에서 correct/incorrect를 length bin별로 그룹핑
        2. 같은 bin 내에서 1:1 unique 매칭 (셔플 후 zip)
        3. 모든 셀의 쌍을 수집
        4. 가용량 비례로 n_samples개 샘플링
    """
    random.seed(seed)

    def get_bin_label(token_len: int) -> str:
        for i, b in enumerate(length_bins[1:]):
            if token_len < b:
                return f"{length_bins[i]}-{b}"
        return f"{length_bins[-1]}+"

    # 1. 모든 가능한 쌍 수집 (같은 problem + 같은 length bin)
    all_pairs = []

    for pid, info in problem_index_map.items():
        correct_indices = info.get("correct_indices", [])
        incorrect_indices = info.get("incorrect_indices", [])
        correct_lengths = info.get("correct_token_lengths", [])
        incorrect_lengths = info.get("incorrect_token_lengths", [])

        if not correct_indices or not incorrect_indices:
            continue
        if not correct_lengths or not incorrect_lengths:
            continue

        # bin별 그룹핑
        correct_by_bin = defaultdict(list)
        incorrect_by_bin = defaultdict(list)

        for idx, length in zip(correct_indices, correct_lengths):
            bin_label = get_bin_label(length)
            correct_by_bin[bin_label].append({"idx": idx, "length": length})

        for idx, length in zip(incorrect_indices, incorrect_lengths):
            bin_label = get_bin_label(length)
            incorrect_by_bin[bin_label].append({"idx": idx, "length": length})

        # 같은 bin 내에서 1:1 매칭
        for bin_label in set(correct_by_bin.keys()) & set(incorrect_by_bin.keys()):
            c_list = correct_by_bin[bin_label]
            i_list = incorrect_by_bin[bin_label]

            random.shuffle(c_list)
            random.shuffle(i_list)

            n_pairs = min(len(c_list), len(i_list))
            if max_pairs_per_problem:
                n_pairs = min(n_pairs, max_pairs_per_problem)

            for j in range(n_pairs):
                all_pairs.append({
                    "correct_idx": c_list[j]["idx"],
                    "incorrect_idx": i_list[j]["idx"],
                    "problem_id": pid,
                    "length_bin": bin_label,
                })

    # 2. 가용량 비례 샘플링
    random.shuffle(all_pairs)

    if len(all_pairs) < n_samples:
        logger.warning(
            f"Length-balanced 쌍 부족: 요청={n_samples:,}, 가용={len(all_pairs):,}"
        )

    selected = all_pairs[:n_samples]

    # 3. 로깅
    logger.info(f"=== Length-Balanced 샘플링 ===")
    logger.info(f"전체 가용 쌍: {len(all_pairs):,}")
    logger.info(f"샘플링된 쌍: {len(selected):,}")

    # bin별 통계
    bin_counts = defaultdict(int)
    for p in selected:
        bin_counts[p["length_bin"]] += 1
    for bin_label in sorted(bin_counts.keys()):
        logger.info(f"  {bin_label}: {bin_counts[bin_label]:,} 쌍")

    return selected
```

---

### Phase 3: load_dataset 통합

**목표**: `use_length_balanced` 옵션 추가

**파일**: `src/weighted_mtp/data/datasets.py`

**변경 사항** (`load_dataset` 함수):

```python
def load_dataset(...):
    # 기존 코드 유지
    use_pairwise = sampling_config.get("use_pairwise", False)
    use_length_balanced = sampling_config.get("use_length_balanced", False)

    # Length-balanced 샘플링 (critic_mlp 전용)
    if use_pairwise and use_length_balanced:
        length_bins = sampling_config.get("length_bins", [0, 100, 150, 200, 300, 500, 1000, 2000])
        all_pairs = _sample_length_balanced_pairs(
            problem_index_map=problem_index_map,
            n_samples=n_samples,
            length_bins=length_bins,
            seed=seed,
            max_pairs_per_problem=max_pairs_per_problem,
        )
    else:
        # 기존 difficulty 기반 샘플링
        all_pairs = _sample_unique_pairs(
            problem_index_map=problem_index_map,
            n_samples=n_samples,
            difficulty_weights=difficulty_weights,
            difficulty_bins=difficulty_bins,
            seed=seed,
            max_pairs_per_problem=max_pairs_per_problem,
        )

    # 이하 기존 코드 동일
```

---

### Phase 4: Config 업데이트

**목표**: critic_mlp.yaml에 length-balanced 옵션 추가

**파일**: `configs/production/critic_mlp.yaml`

**변경 사항**:
```yaml
data_sampling:
  seed: 84
  use_pairwise: true
  n_samples: 200000
  max_pairs_per_problem: 50

  # 기존 difficulty 설정 (length_balanced 사용 시 무시됨)
  difficulty_bins:
    zero: [0, 25]
  difficulty_weights:
    zero: 1.0

  # Length-Balanced 샘플링 (길이 편향 제거)
  use_length_balanced: true
  length_bins: [0, 100, 150, 200, 300, 500, 1000, 2000]

  collator:
    use_random_window: false  # length-balanced 시 불필요
    window_size: 192
```

---

### Phase 5: 테스트 및 검증

**5-1. 단위 테스트**

**파일**: `tests/unit/test_datasets.py`

```python
class TestLengthBalancedSampling:
    """Length-Balanced 샘플링 테스트"""

    def test_same_bin_matching(self):
        """같은 length bin 내에서만 매칭되는지 검증"""
        problem_index_map = {
            "prob1": {
                "correct_indices": [0, 1, 2],
                "incorrect_indices": [3, 4, 5],
                "correct_token_lengths": [80, 150, 350],
                "incorrect_token_lengths": [90, 160, 380],
            }
        }

        pairs = _sample_length_balanced_pairs(
            problem_index_map, n_samples=3,
            length_bins=[0, 100, 200, 500], seed=42
        )

        # 각 쌍이 같은 bin 내에 있는지 확인
        for pair in pairs:
            c_idx = pair["correct_idx"]
            i_idx = pair["incorrect_idx"]
            # bin 검증 로직

    def test_uniqueness(self):
        """각 샘플이 최대 1회만 사용되는지 검증"""
        # correct_idx와 incorrect_idx 모두 unique해야 함

    def test_proportional_sampling(self):
        """가용량 비례 샘플링 검증"""
        # 큰 셀에서 더 많이 샘플링되는지 확인
```

**5-2. 통합 테스트**

```bash
# 샘플링 결과 통계 확인
PYTHONPATH=src uv run python -c "
from weighted_mtp.data.datasets import load_dataset
dataset = load_dataset('codecontests', 'train', {
    'use_pairwise': True,
    'use_length_balanced': True,
    'n_samples': 1000,
    'length_bins': [0, 100, 150, 200, 300, 500, 1000],
}, seed=42)
print(f'Loaded {len(dataset)} pairs')
"
```

**5-3. 길이 편향 검증**

```python
# 샘플링된 쌍의 correct/incorrect 토큰 길이 비교
# 목표: 평균 차이 < 5%
```

---

## 파일 변경 요약

| Phase | 파일 | 변경 내용 | 예상 LOC |
|-------|------|----------|---------|
| 1 | `extract_metadata.py` | 토큰 길이 정보 추가 | +30 |
| 2 | `datasets.py` | `_sample_length_balanced_pairs` 함수 | +80 |
| 3 | `datasets.py` | `load_dataset` 분기 추가 | +15 |
| 4 | `critic_mlp.yaml` | length_balanced 옵션 | +5 |
| 5 | `test_datasets.py` | 테스트 케이스 | +60 |

---

## 호환성

### 기존 파이프라인 영향 없음
- `use_length_balanced: false` (기본값) → 기존 동작
- baseline, rho1, verifiable 파이프라인 변경 없음
- critic_mlp만 `use_length_balanced: true` 설정

### 하위 호환성
- `length_bins` 미설정 시 기본값 사용
- 메타데이터에 토큰 길이 없으면 기존 방식으로 fallback (경고 출력)

---

## 기대 효과

| 메트릭 | 현재 | 예상 |
|--------|------|------|
| Correct-Incorrect 길이 차이 | 52.5 토큰 (31%) | < 5 토큰 (2%) |
| Length Shortcut 학습 가능성 | 높음 | 낮음 |
| position_correlation | -0.4 ~ -0.5 | -0.1 ~ 0.1 |
| 데이터 활용률 | 100% | 711k/895k (79%) |

---

## 개발 원칙 체크리스트

- [x] **원칙 1**: 기존 `_sample_unique_pairs` 함수 분석 완료
- [x] **원칙 2**: 기존 구조 존중 (새 함수 추가, 기존 함수 수정 최소화)
- [x] **원칙 3**: 중복 없음 (length_balanced는 완전히 새로운 로직)
- [ ] **원칙 4**: 구현 시 변수명 통일, wrapper 최소화
- [ ] **원칙 5**: Phase별 완료 후 계획서와 비교 검토
- [x] **원칙 6**: 패키지 의존성 도구 활용 (uv)

---

## 구현 순서

1. **Phase 1** (메타데이터): extract_metadata.py 수정 → 메타데이터 재생성
2. **Phase 2** (핵심 로직): `_sample_length_balanced_pairs` 구현
3. **Phase 3** (통합): `load_dataset` 분기 추가
4. **Phase 4** (설정): critic_mlp.yaml 업데이트
5. **Phase 5** (검증): 테스트 및 길이 편향 확인
