# Storage Directory

Weighted MTP 프로젝트의 모든 모델 및 데이터셋 자산을 관리합니다.

---

## 디렉터리 구조

```
storage/
├── models/
│   ├── meta-llama-mtp/              # Base MTP model (7B, ~25GB)
│   │   ├── raw/                     # HuggingFace 원본
│   │   │   ├── 7B_1T_4/
│   │   │   │   ├── consolidated.pth
│   │   │   │   └── params.json
│   │   │   └── tokenizer.model
│   │   ├── safetensors/
│   │   │   ├── model.safetensors   # 변환된 모델
│   │   │   └── SHA256SUMS
│   │   ├── configs/
│   │   │   ├── params.json
│   │   │   └── meta_adapter.yaml
│   │   └── tokenizer/
│   │       └── tokenizer.model
│   ├── ref-sheared-llama-2.7b/      # Reference model (2.7B, ~10GB)
│   │   ├── raw/
│   │   │   └── reference_model/
│   │   ├── safetensors/
│   │   │   ├── model.safetensors
│   │   │   └── SHA256SUMS
│   │   ├── configs/
│   │   │   └── config.json
│   │   └── tokenizer/
│   │       └── tokenizer.model
│   ├── micro-mtp/                   # 로컬 테스트용 MTP (~180MB)
│   │   ├── safetensors/
│   │   │   ├── model.safetensors
│   │   │   └── SHA256SUMS
│   │   ├── configs/
│   │   │   └── config.json
│   │   └── metadata.json
│   └── micro-reference/             # 로컬 테스트용 Reference
│       ├── safetensors/
│       ├── configs/
│       └── metadata.json
├── datasets/
│   ├── codecontests/
│   │   ├── processed/
│   │   │   ├── train.jsonl          # 학습용 (correct + incorrect)
│   │   │   ├── valid.jsonl
│   │   │   ├── test.jsonl
│   │   │   ├── valid_eval.jsonl     # 평가용 (문제만)
│   │   │   ├── test_eval.jsonl
│   │   │   ├── train_metadata.json  # problem_index_map 포함
│   │   │   ├── valid_metadata.json
│   │   │   ├── test_metadata.json
│   │   │   └── schema.json
│   │   ├── tests/
│   │   │   ├── train_tests.json     # 테스트 케이스
│   │   │   ├── valid_tests.json
│   │   │   └── test_tests.json
│   │   └── stats/
│   ├── mbpp/
│   │   └── processed/
│   │       ├── train.jsonl
│   │       ├── validation.jsonl
│   │       ├── test.jsonl
│   │       └── *_metadata.json
│   ├── humaneval/
│   │   └── processed/
│   │       ├── test.jsonl           # test split만 존재
│   │       └── test_metadata.json
│   └── gsm8k/
│       └── processed/
│           ├── train.jsonl
│           ├── test.jsonl
│           └── *_metadata.json
├── datasets_local_small/            # 로컬 테스트용 축소 데이터셋
│   ├── codecontests_small/
│   ├── mbpp_small/
│   ├── humaneval_small/
│   └── gsm8k_small/
└── checkpoints/                     # 학습 체크포인트
    ├── baseline/
    ├── verifiable/
    └── critic/
```

---

## 모델 자산

### meta-llama-mtp (Base Model)

| 항목 | 값 |
|------|-----|
| 크기 | ~25GB |
| 파라미터 | 7B |
| 형식 | safetensors |
| 소스 | `facebook/multi-token-prediction` |
| n_future_tokens | 4 |
| vocab_size | 32000 |

**주요 파일**:
- `safetensors/model.safetensors` - 모델 가중치
- `safetensors/SHA256SUMS` - 무결성 검증용
- `tokenizer/tokenizer.model` - SentencePiece 토크나이저
- `configs/meta_adapter.yaml` - MTP adapter 설정
- `configs/params.json` - 원본 파라미터

### ref-sheared-llama-2.7b (Reference Model)

| 항목 | 값 |
|------|-----|
| 크기 | ~10GB |
| 파라미터 | 2.7B |
| 형식 | safetensors (sharded에서 병합) |
| 소스 | `microsoft/rho-1` (reference_model) |
| 용도 | Rho-1 Weighted 실험용 |

**토크나이저**: `meta-llama-mtp`와 공유 (vocab_size: 32000)

### micro-mtp / micro-reference (로컬 테스트용)

| 항목 | 값 |
|------|-----|
| 크기 | ~180MB |
| 레이어 | 4 |
| hidden_size | 512 |
| intermediate_size | 1365 |
| 생성 방법 | slice (원본에서) 또는 random (랜덤 초기화) |

---

## 데이터셋 자산

### codecontests

| 항목 | 값 |
|------|-----|
| 소스 | `deepmind/code_contests` |
| train | ~3.2M 샘플 (correct + incorrect) |
| valid | ~15K 샘플 |
| test | ~8K 샘플 |
| is_correct 필드 | O |
| 용도 | 모든 실험 (Baseline, Verifiable Critic, Rho-1) |

**특수 파일**:
- `*_eval.jsonl` - 평가용 (문제만, 솔루션 없음)
- `tests/*.json` - 테스트 케이스 (public, private, generated)
- `*_metadata.json` - problem_index_map (학습용 샘플링)

### mbpp

| 항목 | 값 |
|------|-----|
| 소스 | `google-research-datasets/mbpp` (full) |
| train | 374 샘플 |
| validation | 90 샘플 |
| test | 500 샘플 |
| is_correct 필드 | X |
| 용도 | Baseline, Rho-1만 가능 |

### humaneval

| 항목 | 값 |
|------|-----|
| 소스 | `openai/openai_humaneval` |
| test | 164 샘플 |
| is_correct 필드 | X |
| 용도 | 평가 전용 (test split만 존재) |

### gsm8k

| 항목 | 값 |
|------|-----|
| 소스 | `openai/gsm8k` (main) |
| train | ~7.5K 샘플 |
| test | ~1.3K 샘플 |
| is_correct 필드 | X |
| 용도 | 수학 추론 평가 |

---

## Alpaca 형식 (통일 스키마)

모든 데이터셋은 다음 형식으로 저장:

```json
{
  "instruction": "문제 설명",
  "input": "",
  "output": "정답 코드 또는 풀이",
  "task_id": "고유 ID",
  "is_correct": true,
  "metadata": {
    "source": "code_contests",
    "difficulty": 7
  }
}
```

**필드 설명**:
- `instruction`: 문제 설명 (예: Example I/O 포함)
- `input`: 빈 문자열 (중복 방지)
- `output`: 솔루션 코드
- `is_correct`: 정답 여부 (codecontests만)
- `metadata.difficulty`: 난이도 (1-25, codecontests)

---

## 메타데이터 형식

`*_metadata.json` 파일 구조:

```json
{
  "metadata": [...],
  "problem_index_map": {
    "problem_id": {
      "difficulty": 7,
      "correct_indices": [0, 5, 12],
      "incorrect_indices": [1, 2, 3],
      "correct_token_lengths": [150, 200, 180],
      "incorrect_token_lengths": [120, 90, 300]
    }
  },
  "stats": {
    "total": 100000,
    "correct": 30000,
    "incorrect": 70000,
    "n_problems": 500,
    "n_valid_problems": 480,
    "total_possible_pairs": 5000000
  }
}
```

**학습 파이프라인에서 사용**:
- Difficulty 기반 가중 샘플링
- Length-balanced 샘플링 (Critic MLP용)
- Pairwise 쌍 생성

---

## 설정 방법

### 모델 설정

```bash
# Meta MTP 전체 설정
uv run python scripts/create_storage/setup_models.py \
  --model meta-llama-mtp --steps all

# Micro 모델 생성 (랜덤 초기화)
uv run python scripts/create_storage/setup_models.py \
  --model meta-llama-mtp --create-micro --micro-type mtp --micro-init random
```

### 데이터셋 설정

```bash
# 전체 데이터셋 설정
uv run python scripts/create_storage/setup_datasets.py \
  --datasets all --steps all

# CodeContests 평가용 + 테스트 케이스
uv run python scripts/create_storage/setup_datasets.py \
  --datasets codecontests --steps eval,tests
```

### 무결성 검증

```bash
uv run python scripts/create_storage/verify_storage.py --check all
```

---

## 무결성 검증

각 모델의 safetensors 파일 무결성 확인:

```bash
cd storage/models/meta-llama-mtp/safetensors
sha256sum -c SHA256SUMS

cd storage/models/ref-sheared-llama-2.7b/safetensors
sha256sum -c SHA256SUMS
```

---

## 로컬 테스트용 축소 데이터셋

`datasets_local_small/` - 빠른 로컬 테스트용:

| 데이터셋 | train | validation/valid | test |
|---------|-------|------------------|------|
| codecontests_small | 100 | 32 | 32 |
| mbpp_small | 100 | 32 | 32 |
| humaneval_small | - | - | 32 |
| gsm8k_small | 100 | - | 32 |

---

## 참고

- 자세한 설정 방법: `docs/SETUP.md`
- 모델 다운로드에 HuggingFace 토큰 필요 (`HF_TOKEN`)
- Meta MTP 모델은 라이선스 동의 필요
