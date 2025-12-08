# Docker 실행 가이드

단독 Docker 환경에서 Weighted MTP 학습/평가를 실행하는 방법.

---

## 구성 개요

```
Docker Container
├── /app/                    # 코드 (이미지에 포함)
│   ├── src/weighted_mtp/
│   ├── configs/
│   ├── storage -> /storage  # 심볼릭 링크 (기존 config 호환)
│   └── pyproject.toml
└── /storage/                # 데이터/모델 (Volume Mount)
    ├── models/
    ├── datasets/
    └── checkpoints/
```

**원칙**:
- **코드**: Docker 이미지에 포함 (빌드 시)
- **데이터/모델**: Volume mount (런타임 시) - 대용량이므로 이미지에 포함하지 않음
- **Config 호환**: `/app/storage → /storage` 심볼릭 링크로 기존 `configs/production/` 그대로 사용

---

## 1. 이미지 빌드

```bash
# 로컬 빌드
docker build -t weighted-mtp:latest .

# 태그 지정
docker build -t weighted-mtp:v0.2.0 .
```

---

## 3. 데이터/모델 준비 (호스트)

호스트 머신에서 storage 디렉토리 준비:

```bash
# 프로젝트 루트에서
./storage/
├── models/
│   ├── meta-llama-mtp/          # 25GB
│   └── ref-sheared-llama-2.7b/  # 10GB
├── datasets/
│   └── codecontests/processed/
└── checkpoints/                  # 학습 결과 저장
```

### 모델 준비

```bash
# 호스트에서 실행
uv run python scripts/create_storage/setup_models.py \
  --model meta-llama-mtp --steps all

uv run python scripts/create_storage/setup_models.py \
  --model ref-sheared-llama --steps all
```

### 데이터셋 준비

```bash
uv run python scripts/create_storage/setup_datasets.py \
  --datasets codecontests --steps all
```

---

## 4. 실행

### 기본 실행 (Volume Mount)

```bash
# 현재 디렉토리의 storage를 /storage로 마운트
docker run --gpus all -it \
  -v $(pwd)/storage:/storage \
  weighted-mtp:latest \
  python src/weighted_mtp/pipelines/run_baseline.py \
    --config configs/production/baseline.yaml
```

### 환경변수 전달

```bash
docker run --gpus all -it \
  -v $(pwd)/storage:/storage \
  -e HF_TOKEN=${HF_TOKEN} \
  -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
  weighted-mtp:latest \
  python src/weighted_mtp/pipelines/run_baseline.py \
    --config configs/production/baseline.yaml
```

### Interactive Shell

```bash
docker run --gpus all -it \
  -v $(pwd)/storage:/storage \
  weighted-mtp:latest \
  /bin/bash
```

---

## 5. 멀티 GPU 실행

```bash
# 4-GPU 분산 학습
docker run --gpus all -it \
  -v $(pwd)/storage:/storage \
  --shm-size=16g \
  weighted-mtp:latest \
  torchrun --nproc_per_node=4 \
    src/weighted_mtp/pipelines/run_verifiable.py \
    --config configs/docker/verifiable.yaml
```

**주의**: `--shm-size=16g`로 shared memory 확장 필요 (NCCL 통신용)

---

## 6. Docker Compose (권장)

프로젝트 루트에 `docker-compose.yaml` 포함됨:

```bash
# 빌드
docker compose build

# Baseline 학습
docker compose up baseline

# Critic 학습
docker compose up critic

# Verifiable WMTP 학습
docker compose up verifiable

# 분산 학습 (4-GPU)
docker compose up distributed

# 평가
docker compose up evaluate

# Interactive shell
docker compose run --rm shell

# 백그라운드 실행
docker compose up -d baseline
docker compose logs -f baseline
```

---

## 7. 클라우드 스토리지 연동 (선택)

대용량 모델/데이터를 S3에서 런타임 다운로드:

```bash
docker run --gpus all -it \
  -v $(pwd)/storage:/storage \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  weighted-mtp:latest \
  bash -c "
    aws s3 sync s3://your-bucket/models /storage/models --quiet &&
    python src/weighted_mtp/pipelines/run_baseline.py --config configs/docker/baseline.yaml
  "
```

---

## 8. 체크리스트

### 빌드 전
- [ ] `pyproject.toml`, `uv.lock` 최신 상태
- [ ] `src/`, `configs/` 디렉토리 존재

### 실행 전
- [ ] `storage/models/` 에 필요한 모델 존재
- [ ] `storage/datasets/` 에 필요한 데이터셋 존재
- [ ] Docker에서 GPU 접근 가능 (`docker run --gpus all nvidia-smi`)

### 분산 학습 시
- [ ] `--shm-size=16g` 이상 설정
- [ ] `--gpus all` 또는 특정 GPU 지정

---

## 요약

| 항목 | 위치 | 방식 |
|------|------|------|
| 코드 | `/app/src/` | 이미지에 포함 |
| 설정 | `/app/configs/` | 이미지에 포함 |
| 모델 | `/storage/models/` | Volume mount |
| 데이터 | `/storage/datasets/` | Volume mount |
| 체크포인트 | `/storage/checkpoints/` | Volume mount |
