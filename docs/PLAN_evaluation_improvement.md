# 평가 파이프라인 개선 계획서

## 개요

4가지 데이터셋(humaneval, mbpp, gsm8k, codecontests) 모두를 지원하는 평가 파이프라인 구축.
Meta MTP 2024 논문의 평가 방식(Pass@K, Temperature Search)을 참조하여 개선.

---

## 현재 상태 분석

### 데이터셋별 현황

| 데이터셋 | 샘플 수 | 테스트 데이터 | 평가 지원 |
|----------|---------|--------------|-----------|
| humaneval | 164 | `metadata.test` + `metadata.entry_point` | O |
| mbpp | 500 | `metadata.test_list` | O |
| gsm8k | 1,319 | `metadata.final_answer` | O (CLI 누락) |
| codecontests | 14,851 | 테스트 케이스 없음 | X |

### 문제점

1. **CLI choices 불일치** (`cli/evaluate.py:36`)
   - `codecontests` 포함 → 실제 평가 로직 없음 (ValueError 발생)
   - `gsm8k` 누락 → 이미 구현된 평가 로직 존재

2. **codecontests 테스트 케이스 부재**
   - `setup_datasets.py`에서 `public_tests`를 input 예시로만 저장
   - 실제 input/output 쌍을 평가용으로 저장하지 않음
   - DeepMind 원본 데이터: `public_tests`, `private_tests`, `generated_tests` 포함

3. **Temperature Search 미지원**
   - Meta MTP 논문: {0.5, 0.6, 0.7, 0.8, 0.9} 범위에서 최적 temperature 탐색
   - 현재: 단일 temperature만 지원

---

## Phase별 개선 계획

### Phase 1: CLI 수정 (즉시)

**목표**: CLI choices를 실제 지원 데이터셋과 일치시킴

**수정 파일**: `src/weighted_mtp/cli/evaluate.py`

```python
# 변경 전 (line 36)
choices=["humaneval", "mbpp", "codecontests"]

# 변경 후
choices=["humaneval", "mbpp", "gsm8k"]
```

**검증**: `--help` 출력 확인, gsm8k 평가 테스트

---

### Phase 2: codecontests 테스트 케이스 다운로드 스크립트

**목표**: DeepMind 원본 데이터에서 실제 테스트 케이스 추출

**신규 파일**: `scripts/create_storage/download_codecontests_tests.py`

**DeepMind 원본 데이터 구조**:
```
gs://dm-code_contests/
├── test/
│   └── *.riegeli (protobuf 형식)
├── valid/
└── train/

각 problem에 포함된 테스트 필드:
- public_tests: {input: [], output: []}
- private_tests: {input: [], output: []}
- generated_tests: {input: [], output: []}
```

**다운로드 방식 (2가지 옵션)**:

| 옵션 | 방식 | 장점 | 단점 |
|------|------|------|------|
| A | HuggingFace deepmind/code_contests 재활용 | 추가 설치 없음 | 이미 다운로드 중이므로 중복 |
| B | GCS 직접 다운로드 | 최신 데이터, 완전한 테스트 | gsutil 설치 필요 |

**권장: 옵션 A** - HuggingFace 데이터에서 테스트 케이스 추출

**구현 내용**:
```python
def download_codecontests_tests():
    """HuggingFace에서 codecontests 테스트 케이스 추출"""
    dataset = load_dataset("deepmind/code_contests")

    for split in ["train", "valid", "test"]:
        tests_data = []
        for example in dataset[split]:
            task_name = example["name"]
            tests_data.append({
                "task_name": task_name,
                "public_tests": example.get("public_tests", {}),
                "private_tests": example.get("private_tests", {}),
                "generated_tests": example.get("generated_tests", {}),
            })

        # JSON 저장
        output_path = f"storage/datasets/codecontests/tests/{split}_tests.json"
        save_json(tests_data, output_path)
```

**출력 구조**:
```
storage/datasets/codecontests/
├── processed/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
└── tests/              # 신규
    ├── train_tests.json
    ├── valid_tests.json
    └── test_tests.json
```

---

### Phase 3: codecontests 평가 유틸리티 구현

**목표**: codecontests 코드 실행 및 테스트 평가 함수 구현

**수정 파일**: `src/weighted_mtp/utils/evaluation_utils.py`

**신규 함수**:
```python
def execute_codecontests_tests(
    code: str,
    tests: dict,  # {"input": [...], "output": [...]}
    timeout: int = 10,
) -> dict:
    """CodeContests 형식의 테스트 실행

    Args:
        code: 생성된 Python 코드
        tests: {"input": [str, ...], "output": [str, ...]}
        timeout: 테스트당 실행 제한 시간 (초)

    Returns:
        {
            "passed": int,
            "total": int,
            "pass_rate": float,
            "details": [{"input": str, "expected": str, "actual": str, "passed": bool}, ...]
        }
    """
    pass
```

**실행 방식**:
- 생성된 코드를 임시 파일로 저장
- subprocess로 stdin에 input 전달
- stdout 캡처 후 expected output과 비교
- 정확히 일치하면 pass

---

### Phase 4: run_evaluation.py에 codecontests 통합

**목표**: 평가 파이프라인에서 codecontests 분기 처리

**수정 파일**: `src/weighted_mtp/pipelines/run_evaluation.py`

**변경 내용**:

1. **테스트 케이스 로드 함수 추가**:
```python
def _load_codecontests_tests(split: str) -> dict:
    """codecontests 테스트 케이스 로드

    Returns:
        {task_name: {"public_tests": {...}, "private_tests": {...}, ...}}
    """
    tests_path = Path(f"storage/datasets/codecontests/tests/{split}_tests.json")
    with open(tests_path) as f:
        tests_list = json.load(f)
    return {t["task_name"]: t for t in tests_list}
```

2. **평가 루프 분기 추가** (line 162-194 사이):
```python
elif dataset_name == "codecontests":
    # task_id에서 task_name 추출 (예: "task_0_correct_1" → "task_0")
    task_name = "_".join(task_id.split("_")[:2])

    # 테스트 케이스 로드 (public_tests만 사용, private는 비공개)
    test_data = codecontests_tests.get(task_name, {})
    public_tests = test_data.get("public_tests", {"input": [], "output": []})

    result = execute_codecontests_tests(
        code=code,
        tests=public_tests,
        timeout=10,
    )
    passed = result["pass_rate"] >= 1.0  # 모든 테스트 통과
```

3. **CLI choices 업데이트**:
```python
choices=["humaneval", "mbpp", "gsm8k", "codecontests"]
```

---

### Phase 5: Temperature Search 기능 추가 (선택)

**목표**: Meta MTP 논문 방식의 temperature 최적화 지원

**수정 파일**: `src/weighted_mtp/cli/evaluate.py`, `run_evaluation.py`

**CLI 옵션 추가**:
```python
parser.add_argument(
    "--temperature-search",
    type=str,
    default=None,
    help="Temperature 범위 (예: '0.5,0.6,0.7,0.8,0.9')",
)
```

**동작 방식**:
1. 지정된 temperature 범위에서 각각 평가 수행
2. 가장 높은 Pass@K를 달성한 temperature 선택
3. 결과에 최적 temperature 포함

**참고**: Meta MTP 논문에서는 {0.5, 0.6, 0.7, 0.8, 0.9} 범위에서 탐색

---

## 구현 우선순위

| 순위 | Phase | 예상 소요 | 이유 |
|------|-------|----------|------|
| 1 | Phase 1 | 5분 | 즉시 수정 가능, 버그 해결 |
| 2 | Phase 2 | 30분 | codecontests 평가의 전제 조건 |
| 3 | Phase 3 | 1시간 | 핵심 평가 로직 구현 |
| 4 | Phase 4 | 30분 | 통합 및 테스트 |
| 5 | Phase 5 | 1시간 | 선택 사항, 논문 재현 필요 시 |

---

## 파일 변경 요약

| 파일 | Phase | 변경 내용 |
|------|-------|----------|
| `cli/evaluate.py` | 1, 4 | choices 수정, temperature-search 옵션 추가 |
| `scripts/create_storage/download_codecontests_tests.py` | 2 | 신규 생성 |
| `utils/evaluation_utils.py` | 3 | `execute_codecontests_tests()` 추가 |
| `pipelines/run_evaluation.py` | 4 | codecontests 분기 추가, 테스트 로더 추가 |
| `data/datasets.py` | - | 변경 없음 |

---

## 검증 계획

### Phase 1 검증
```bash
# CLI help 확인
uv run wmtp evaluate --help
# gsm8k 평가 테스트
uv run wmtp evaluate --checkpoint <path> --dataset gsm8k --max-tasks 5
```

### Phase 2 검증
```bash
# 테스트 케이스 다운로드
uv run python scripts/create_storage/download_codecontests_tests.py

# 출력 파일 확인
ls -la storage/datasets/codecontests/tests/
```

### Phase 3-4 검증
```bash
# codecontests 평가 테스트
uv run wmtp evaluate --checkpoint <path> --dataset codecontests --max-tasks 5
```

### Phase 5 검증
```bash
# Temperature search 테스트
uv run wmtp evaluate --checkpoint <path> --dataset humaneval \
    --temperature-search "0.5,0.6,0.7,0.8,0.9" --max-tasks 10
```

---

## 참고 자료

- **Meta MTP 2024 논문**: Pass@K (Chen et al. 2021), 200 samples, temperature search
- **DeepMind Code Contests**: https://github.com/google-deepmind/code_contests
- **HuggingFace Dataset**: deepmind/code_contests

---

## 작성 정보

- **작성일**: 2025-12-02
- **상태**: 구현 완료
- **완료일**: 2025-12-02
