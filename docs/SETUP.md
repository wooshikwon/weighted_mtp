# 환경 설정

Weighted MTP 프로젝트의 로컬 개발 환경 구성 및 데이터 준비 가이드.

---

## 1. 의존성 설치

### uv 패키지 관리자

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치
uv sync

# 개발 의존성 포함
uv sync --dev
```

### 환경변수 설정

프로젝트 루트에 `.env` 파일 생성:

```bash
# HuggingFace (필수)
HF_TOKEN=<your-huggingface-token>
```

HuggingFace 토큰은 https://huggingface.co/settings/tokens 에서 발급.
`facebook/multi-token-prediction` 모델 접근을 위해 Meta 라이선스 동의 필요.

---

## 2. 모델 준비

HuggingFace에서 모델을 다운로드하고 SafeTensors 형식으로 변환합니다.

### 2.1 Meta LLaMA MTP (7B)

```bash
# 다운로드 + 변환 + config 동기화 + 검증
uv run python scripts/create_storage/setup_models.py \
  --model meta-llama-mtp \
  --steps all
```

### 2.2 Sheared-LLaMA 2.7B (Reference Model)

```bash
uv run python scripts/create_storage/setup_models.py \
  --model ref-sheared-llama \
  --steps all
```

### 2.3 Micro 모델 생성 (로컬 테스트용)

원본 모델 없이 랜덤 초기화된 경량 모델 생성:

```bash
# Micro-MTP (4 layers, 512 hidden)
uv run python scripts/create_storage/setup_models.py \
  --model meta-llama-mtp \
  --create-micro \
  --micro-type mtp \
  --micro-init random

# Micro-Reference
uv run python scripts/create_storage/setup_models.py \
  --model ref-sheared-llama \
  --create-micro \
  --micro-type reference \
  --micro-init random
```

### 모델 디렉터리 구조

```
storage/models/
├── meta-llama-mtp/              # Base MTP (약 25GB)
│   ├── raw/                     # HuggingFace 원본
│   │   ├── 7B_1T_4/
│   │   │   ├── consolidated.pth
│   │   │   └── params.json
│   │   └── tokenizer.model
│   ├── safetensors/
│   │   ├── model.safetensors    # 변환된 모델
│   │   └── SHA256SUMS
│   ├── configs/
│   │   ├── params.json
│   │   └── meta_adapter.yaml
│   └── tokenizer/
│       └── tokenizer.model
├── ref-sheared-llama-2.7b/      # Reference model (약 10GB)
│   ├── raw/
│   ├── safetensors/
│   └── configs/
├── micro-mtp/                   # 로컬 테스트용 (약 180MB)
│   ├── safetensors/
│   └── configs/
└── micro-reference/             # 로컬 테스트용
```

---

## 3. 데이터셋 준비

HuggingFace에서 데이터셋을 다운로드하고 Alpaca 형식으로 전처리합니다.

### 3.1 전체 데이터셋 준비

```bash
# 전체 데이터셋 (codecontests, mbpp, humaneval, gsm8k)
uv run python scripts/create_storage/setup_datasets.py \
  --datasets all \
  --steps all
```

### 3.2 개별 데이터셋

```bash
# CodeContests
uv run python scripts/create_storage/setup_datasets.py \
  --datasets codecontests \
  --steps all

# MBPP
uv run python scripts/create_storage/setup_datasets.py \
  --datasets mbpp \
  --steps all

# HumanEval
uv run python scripts/create_storage/setup_datasets.py \
  --datasets humaneval \
  --steps all

# GSM8K
uv run python scripts/create_storage/setup_datasets.py \
  --datasets gsm8k \
  --steps all
```

### 3.3 단계별 실행

```bash
# 다운로드 + 전처리만
uv run python scripts/create_storage/setup_datasets.py --datasets all --steps process

# 메타데이터 추출
uv run python scripts/create_storage/setup_datasets.py --datasets codecontests --steps metadata

# Small 버전 생성
uv run python scripts/create_storage/setup_datasets.py --datasets all --steps small

# 통계 생성
uv run python scripts/create_storage/setup_datasets.py --datasets all --steps stats

# CodeContests 평가용 데이터셋 생성 (valid_eval.jsonl, test_eval.jsonl)
uv run python scripts/create_storage/setup_datasets.py --datasets codecontests --steps eval

# CodeContests 테스트 케이스 추출 (tests/*.json)
uv run python scripts/create_storage/setup_datasets.py --datasets codecontests --steps tests
```

### 데이터셋 디렉터리 구조

```
storage/datasets/
├── codecontests/
│   ├── processed/
│   │   ├── train.jsonl           # 학습용 (correct + incorrect 솔루션)
│   │   ├── valid.jsonl
│   │   ├── test.jsonl
│   │   ├── test_eval.jsonl       # 평가용 (문제만)
│   │   ├── train_metadata.json   # is_correct, difficulty, problem_index_map
│   │   └── schema.json
│   ├── tests/
│   │   ├── train_tests.json      # 테스트 케이스
│   │   ├── valid_tests.json
│   │   └── test_tests.json
│   └── stats/
├── mbpp/
│   └── processed/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
├── humaneval/
│   └── processed/
│       └── test.jsonl
├── gsm8k/
│   └── processed/
│       ├── train.jsonl
│       └── test.jsonl
└── datasets_local_small/         # 로컬 테스트용 샘플
    ├── codecontests_small/
    ├── mbpp_small/
    ├── humaneval_small/
    └── gsm8k_small/
```

### Alpaca 형식

모든 데이터셋은 통일된 Alpaca 형식으로 저장:

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

---

## 4. 검증

### Storage 무결성 검증

```bash
# 전체 검증 (모델 + 데이터셋)
uv run python scripts/create_storage/verify_storage.py --check all

# 모델만 검증
uv run python scripts/create_storage/verify_storage.py --check models

# 데이터셋만 검증
uv run python scripts/create_storage/verify_storage.py --check datasets

# 검증 리포트 생성
uv run python scripts/create_storage/verify_storage.py --check all --generate-report
```

### 단위 테스트

```bash
# 전체 테스트
uv run pytest tests/unit/

# 특정 모듈
uv run pytest tests/unit/test_adapter.py
uv run pytest tests/unit/test_datasets.py
```

---

## 5. 빠른 시작 (Quick Start)

처음 설정 시 전체 과정:

```bash
# 1. 의존성 설치
uv sync --dev

# 2. 환경변수 설정
echo "HF_TOKEN=<your-token>" > .env

# 3. Micro 모델 생성 (로컬 테스트용)
uv run python scripts/create_storage/setup_models.py \
  --model meta-llama-mtp \
  --create-micro --micro-type mtp --micro-init random

# 4. Small 데이터셋 준비
uv run python scripts/create_storage/setup_datasets.py \
  --datasets all --steps process,small

# 5. 검증
uv run python scripts/create_storage/verify_storage.py --check all
```

---

## 6. Troubleshooting

### uv 설치 오류

```bash
# macOS Homebrew 대안
brew install uv
```

### HuggingFace 다운로드 실패

```bash
# 토큰 확인
echo $HF_TOKEN

# 토큰 재설정
huggingface-cli login

# Meta 모델 라이선스 동의 확인
# https://huggingface.co/facebook/multi-token-prediction
```

### 메모리 부족

```bash
# Micro 모델 사용 (configs에서 설정)
model:
  use_micro: true

# 또는 --create-micro --micro-init random 으로 생성
```

### Safetensors 변환 오류

```bash
# 기존 safetensors 삭제 후 재생성
rm -rf storage/models/*/safetensors/
uv run python scripts/create_storage/setup_models.py --model all --steps convert
```

---

## 참고 스크립트

| 스크립트 | 용도 |
|---------|------|
| `scripts/create_storage/setup_models.py` | 모델 다운로드/변환/micro 생성 |
| `scripts/create_storage/setup_datasets.py` | 데이터셋 다운로드/전처리/metadata/eval/tests 통합 |
| `scripts/create_storage/verify_storage.py` | 무결성 검증 |
