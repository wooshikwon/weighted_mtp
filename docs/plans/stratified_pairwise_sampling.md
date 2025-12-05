# Stratified Pairwise Sampling 구현 계획

## 배경

### 문제점
Value Model 학습 시 **길이 편향(length bias)** 발생:
- Incorrect 샘플이 평균 31% 더 김 (222.8 vs 169.9 output tokens)
- 모델이 attention_mask 길이만으로 correct/incorrect 구분 가능
- position_correlation이 -0.4 ~ -0.5로 높은 음의 상관관계

### 해결 방안
**Problem 단위 길이 기반 매칭 + 난이도별 층화추출**:
1. 같은 Problem 내에서 길이 차이가 작은 (correct, incorrect) 쌍 매칭
2. 난이도별 쿼터로 분포 균형 유지
3. 기존 데이터 로딩 방식 유지, critic_mlp용만 새 방식 적용

---

## 데이터 분석 결과

### 길이 구간별 Correct 비율
| 길이 구간 | Correct 비율 |
|----------|-------------|
| 0-100 | 56.9% |
| 101-150 | 52.4% |
| 151-200 | 46.2% |
| 201-300 | 38.8% |
| 301-500 | 32.9% |
| 500+ | 33.2% |

### 난이도별 가용 쌍 (length_tolerance=50)
| 난이도 | 가용 쌍 (추정) | 최대 비율 |
|--------|---------------|----------|
| zero | 183,269 | 91.6% |
| easy | 180,796 | 90.4% |
| medium | 166,914 | 83.5% |
| hard | 29,683 | 14.8% |

### 확정 설정 (18만 쌍, medium 강조)
```yaml
data_sampling:
  stratified_pairwise:
    enabled: true
    target_pairs: 180000
    length_tolerance: 50
    difficulty_weights:
      zero: 0.20
      easy: 0.25
      medium: 0.45
      hard: 0.10
```

---

## 현재 구조 분석

### 데이터 로딩 흐름
```
train.jsonl
    ↓
create_dataloader() [dataloader.py]
    ↓
PairwiseSampler [samplers.py]  ← 현재: 난이도 가중치만 적용
    ↓
PairwiseDataCollator [collators.py]
    ↓
DataLoader
```

### 수정 대상 파일
| 파일 | 역할 | 수정 내용 |
|------|------|----------|
| `samplers.py` | 샘플 인덱스 선택 | 새 StratifiedPairwiseSampler 클래스 추가 |
| `dataloader.py` | DataLoader 생성 | stratified_pairwise 설정 분기 추가 |
| `critic_mlp.yaml` | 설정 파일 | stratified_pairwise 섹션 추가 |

### 유지 대상 (하위 호환성)
- 기존 `PairwiseSampler`: 다른 파이프라인에서 계속 사용
- 기존 `create_dataloader` 인터페이스: 설정으로 분기

---

## Phase별 구현 계획

### Phase 1: 메타데이터 전처리

**목표**: Problem별 (correct, incorrect) 그룹화 및 토큰 길이 계산

**파일**: `src/weighted_mtp/data/metadata_builder.py` (신규)

```python
def build_stratified_metadata(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    output_path: str,
) -> dict:
    """
    Problem별 correct/incorrect 샘플을 길이와 함께 그룹화

    Returns:
        {
            'problems': {
                problem_id: {
                    'difficulty': int,
                    'correct': [{'idx': int, 'output_tokens': int}, ...],
                    'incorrect': [{'idx': int, 'output_tokens': int}, ...],
                }
            },
            'difficulty_groups': {
                'zero': [problem_id, ...],
                'easy': [...],
                'medium': [...],
                'hard': [...],
            }
        }
    """
```

**출력**: `train_stratified_metadata.json`

---

### Phase 2: StratifiedPairwiseSampler 구현

**목표**: Problem 내 길이 매칭 + 난이도별 쿼터 샘플링

**파일**: `src/weighted_mtp/data/samplers.py`

```python
class StratifiedPairwiseSampler:
    """Problem 단위 길이 기반 매칭 + 난이도별 층화추출

    Args:
        metadata_path: stratified metadata JSON 경로
        target_pairs: 목표 쌍 수
        length_tolerance: 허용 길이 차이 (토큰)
        difficulty_weights: 난이도별 가중치
        seed: 랜덤 시드
    """

    def __init__(
        self,
        metadata_path: str,
        target_pairs: int = 180000,
        length_tolerance: int = 50,
        difficulty_weights: dict = None,
        seed: int = 42,
    ):
        self.metadata = self._load_metadata(metadata_path)
        self.target_pairs = target_pairs
        self.length_tolerance = length_tolerance
        self.difficulty_weights = difficulty_weights or {
            'zero': 0.20, 'easy': 0.25, 'medium': 0.45, 'hard': 0.10
        }
        self.seed = seed

        # 매칭된 쌍 인덱스 생성
        self.pairs = self._build_pairs()

    def _match_pairs_in_problem(
        self,
        correct_list: list,
        incorrect_list: list,
    ) -> list[tuple[int, int]]:
        """Problem 내에서 길이 차이가 tolerance 이내인 쌍 매칭"""
        pairs = []
        used_incorrect = set()

        # 길이 순 정렬로 효율적 매칭
        correct_sorted = sorted(correct_list, key=lambda x: x['output_tokens'])
        incorrect_sorted = sorted(incorrect_list, key=lambda x: x['output_tokens'])

        for c in correct_sorted:
            best_match = None
            best_diff = float('inf')

            for i in incorrect_sorted:
                if i['idx'] in used_incorrect:
                    continue

                diff = abs(c['output_tokens'] - i['output_tokens'])
                if diff <= self.length_tolerance and diff < best_diff:
                    best_match = i
                    best_diff = diff

            if best_match:
                pairs.append((c['idx'], best_match['idx']))
                used_incorrect.add(best_match['idx'])

        return pairs

    def _build_pairs(self) -> list[tuple[int, int]]:
        """난이도별 쿼터에 맞춰 쌍 생성"""
        all_pairs = {dg: [] for dg in ['zero', 'easy', 'medium', 'hard']}

        # 각 Problem에서 매칭 가능한 쌍 수집
        for problem_id, problem_data in self.metadata['problems'].items():
            diff_group = self._difficulty_group(problem_data['difficulty'])
            pairs = self._match_pairs_in_problem(
                problem_data['correct'],
                problem_data['incorrect'],
            )
            all_pairs[diff_group].extend(pairs)

        # 난이도별 쿼터 적용
        random.seed(self.seed)
        selected_pairs = []

        for dg, weight in self.difficulty_weights.items():
            n_target = int(self.target_pairs * weight)
            available = all_pairs[dg]

            if len(available) < n_target:
                selected = available
            else:
                selected = random.sample(available, n_target)

            selected_pairs.extend(selected)

        random.shuffle(selected_pairs)
        return selected_pairs

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        for correct_idx, incorrect_idx in self.pairs:
            yield correct_idx, incorrect_idx
```

---

### Phase 3: DataLoader 통합

**목표**: create_dataloader에 stratified_pairwise 옵션 추가

**파일**: `src/weighted_mtp/data/dataloader.py`

```python
def create_dataloader(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int,
    sampling_config: dict,
    seed: int = 42,
    shuffle: bool = True,
    collator_config: dict = None,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    # 기존 로직 유지...

    use_pairwise = sampling_config.get("use_pairwise", False)

    if use_pairwise:
        # 층화추출 모드 확인
        stratified_config = sampling_config.get("stratified_pairwise", {})

        if stratified_config.get("enabled", False):
            # 새로운 층화추출 방식
            sampler = StratifiedPairwiseSampler(
                metadata_path=stratified_config.get("metadata_path"),
                target_pairs=stratified_config.get("target_pairs", 180000),
                length_tolerance=stratified_config.get("length_tolerance", 50),
                difficulty_weights=stratified_config.get("difficulty_weights"),
                seed=seed,
            )
            dataset = StratifiedPairwiseDataset(dataset_path, sampler.pairs)
        else:
            # 기존 방식 유지
            dataset = PairwiseSampler(...)
```

---

### Phase 4: Dataset 클래스 구현

**목표**: 사전 매칭된 쌍을 효율적으로 로드

**파일**: `src/weighted_mtp/data/datasets.py`

```python
class StratifiedPairwiseDataset(Dataset):
    """사전 매칭된 (correct, incorrect) 쌍 로드

    Args:
        dataset_path: train.jsonl 경로
        pairs: [(correct_idx, incorrect_idx), ...] 리스트
    """

    def __init__(self, dataset_path: str, pairs: list[tuple[int, int]]):
        self.dataset_path = dataset_path
        self.pairs = pairs

        # 전체 데이터를 메모리에 로드 (인덱스 접근용)
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict]:
        with open(self.dataset_path, 'r') as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        correct_idx, incorrect_idx = self.pairs[idx]

        correct_sample = self.samples[correct_idx]
        incorrect_sample = self.samples[incorrect_idx]

        return {
            'instruction': correct_sample['instruction'],
            'input': correct_sample.get('input', ''),
            'correct_output': correct_sample['output'],
            'incorrect_output': incorrect_sample['output'],
        }
```

---

### Phase 5: Config 수정

**파일**: `configs/production/critic_mlp.yaml`

```yaml
# 데이터 샘플링 (Pairwise + 층화추출)
data_sampling:
  seed: 84
  use_pairwise: true

  # 층화추출 설정 (길이 편향 완화)
  stratified_pairwise:
    enabled: true
    metadata_path: storage/datasets/codecontests/processed/train_stratified_metadata.json
    target_pairs: 180000
    length_tolerance: 50
    difficulty_weights:
      zero: 0.20
      easy: 0.25
      medium: 0.45
      hard: 0.10

  # Random Window 설정 (기존 유지)
  collator:
    use_random_window: true
    window_size: 192
```

---

### Phase 6: 테스트

**6-1. 단위 테스트**
```python
class TestStratifiedPairwiseSampler:
    def test_length_matching(self):
        """길이 차이가 tolerance 이내인지 검증"""

    def test_difficulty_quota(self):
        """난이도별 쿼터가 정확히 적용되는지 검증"""

    def test_unique_samples(self):
        """모든 샘플이 unique한지 검증"""

    def test_pair_count(self):
        """목표 쌍 수 달성 검증"""
```

**6-2. 통합 테스트**
```bash
# 메타데이터 빌드
python -m weighted_mtp.data.metadata_builder \
    --input storage/datasets/codecontests/processed/train.jsonl \
    --output storage/datasets/codecontests/processed/train_stratified_metadata.json

# 샘플링 검증
python -c "
from weighted_mtp.data.samplers import StratifiedPairwiseSampler
sampler = StratifiedPairwiseSampler(
    metadata_path='...',
    target_pairs=180000,
)
print(f'Total pairs: {len(sampler)}')
"
```

---

## 개발 원칙 체크리스트

- [ ] **원칙 1**: 기존 dataloader.py, samplers.py 구조 분석 후 수정
- [ ] **원칙 2**: 기존 PairwiseSampler 유지, 새 클래스로 분리
- [ ] **원칙 3**: 중복 로직 없이 깔끔한 분기 처리
- [ ] **원칙 4**: 하위 호환성 유지 (enabled=false시 기존 동작)
- [ ] **원칙 5**: 각 Phase 완료 후 계획서 대비 검증
- [ ] **원칙 6**: uv 패키지 관리 사용

---

## 예상 효과

| 지표 | 현재 | 목표 |
|------|------|------|
| position_correlation | -0.4 ~ -0.5 | -0.1 ~ 0.0 |
| 길이 차이 (correct vs incorrect) | 31% | ≤ 50 tokens |
| 학습 데이터 수 | 20만 쌍 | 18만 쌍 |
| 난이도 분포 | 불균형 | 쿼터 적용 |
