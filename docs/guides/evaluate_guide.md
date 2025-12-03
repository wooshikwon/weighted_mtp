# Evaluation 파이프라인 가이드

Checkpoint를 로드하여 벤치마크 데이터셋에서 Pass@K 평가를 수행하는 가이드

## 지원 데이터셋

| Dataset | 원본 Tasks | 평가 Tasks | 설명 |
|---------|-----------|-----------|------|
| HumanEval | 164 | 100 | OpenAI 코드 생성 벤치마크 |
| MBPP | 500 | 100 | Google 기초 Python 프로그래밍 |
| GSM8K | 1,319 | 100 | 초등학교 수준 수학 문제 |
| CodeContests | 165 | 100 | DeepMind 경쟁 프로그래밍 문제 |

> **Note**: 평가 시간 단축을 위해 각 데이터셋당 최대 100개 문제만 평가합니다. 재현성을 위해 `seed=42`로 고정 샘플링됩니다.

## 사전 준비

### 1. 필수 파일 확인

```bash
# 서버에서 확인
ls -la /workspace/weighted_mtp/storage/models/meta-llama-mtp/safetensors/model.safetensors
ls -la /workspace/weighted_mtp/storage/models/meta-llama-mtp/tokenizer/
ls -la /workspace/weighted_mtp/storage/datasets/codecontests/tests/  # codecontests 평가 시
```

### 2. 모델 파일이 없는 경우 S3에서 다운로드

```bash
# AWS 환경변수 설정
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY='<your-secret>'
export AWS_DEFAULT_REGION=eu-north-1

# 모델 다운로드
aws s3 cp s3://wmtp/weighted-mtp/models/meta-llama-mtp/safetensors/model.safetensors \
    /workspace/weighted_mtp/storage/models/meta-llama-mtp/safetensors/

# 토크나이저 다운로드
aws s3 cp s3://wmtp/weighted-mtp/models/meta-llama-mtp/tokenizer/ \
    /workspace/weighted_mtp/storage/models/meta-llama-mtp/tokenizer/ --recursive

# CodeContests 테스트 케이스 다운로드 (codecontests 평가 시)
aws s3 cp s3://wmtp/weighted-mtp/datasets/codecontests/tests/ \
    /workspace/weighted_mtp/storage/datasets/codecontests/tests/ --recursive
```

### 3. CodeContests 평가용 데이터셋 생성

CodeContests 평가에는 별도의 평가용 데이터셋이 필요합니다 (165개 문제).

```bash
cd /workspace/weighted_mtp

# 평가용 데이터셋 생성 (HuggingFace에서 다운로드)
uv run python scripts/create_storage/create_codecontests_eval.py

# 생성 확인
ls -la storage/datasets/codecontests/processed/test_eval.jsonl
```

## Evaluation 실행

### 단일 데이터셋 실행

```bash
cd /workspace/weighted_mtp

# MLflow 로컬 저장 설정
export MLFLOW_TRACKING_URI=file:///workspace/weighted_mtp/mlruns

# HumanEval 평가
CUDA_VISIBLE_DEVICES=0 uv run python -m weighted_mtp.cli.evaluate \
    --checkpoint storage/checkpoints/baseline/lora-mtp-baseline/checkpoint_epoch_1.80.pt \
    --dataset humaneval \
    --dtype bfloat16
```

### 4개 GPU 병렬 실행

```bash
cd /workspace/weighted_mtp

# MLflow 로컬 저장 설정
export MLFLOW_TRACKING_URI=file:///workspace/weighted_mtp/mlruns

# 4개 데이터셋 병렬 실행 (각 GPU에 할당)
CUDA_VISIBLE_DEVICES=0 nohup uv run python -m weighted_mtp.cli.evaluate \
    --checkpoint storage/checkpoints/baseline/lora-mtp-baseline/checkpoint_epoch_1.80.pt \
    --dataset humaneval --dtype bfloat16 > logs/eval_humaneval.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup uv run python -m weighted_mtp.cli.evaluate \
    --checkpoint storage/checkpoints/baseline/lora-mtp-baseline/checkpoint_epoch_1.80.pt \
    --dataset mbpp --dtype bfloat16 > logs/eval_mbpp.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup uv run python -m weighted_mtp.cli.evaluate \
    --checkpoint storage/checkpoints/baseline/lora-mtp-baseline/checkpoint_epoch_1.80.pt \
    --dataset gsm8k --dtype bfloat16 > logs/eval_gsm8k.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup uv run python -m weighted_mtp.cli.evaluate \
    --checkpoint storage/checkpoints/baseline/lora-mtp-baseline/checkpoint_epoch_1.80.pt \
    --dataset codecontests --dtype bfloat16 > logs/eval_codecontests.log 2>&1 &
```

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--checkpoint` | (필수) | Checkpoint 경로 |
| `--dataset` | humaneval | humaneval, mbpp, gsm8k, codecontests |
| `--num-samples` | 20 | 문제당 생성 샘플 수 (Pass@K 계산용) |
| `--temperature` | 0.2 | Sampling temperature |
| `--max-tokens` | 512 | 최대 생성 토큰 수 |
| `--dtype` | auto | float16, bfloat16 |
| `--no-mlflow` | - | MLflow 로깅 비활성화 |
| `--max-tasks` | 전체 | 평가 태스크 수 제한 (테스트용) |

## 진행 상황 모니터링

### 로그 확인

```bash
# 전체 로그 실시간 확인
tail -f logs/eval_*.log

# 진행률만 확인
watch -n 10 'tail -n 1 logs/eval_*.log'

# 특정 데이터셋만
tail -f logs/eval_humaneval.log
```

### 프로세스 확인

```bash
# 실행 중인 프로세스 확인
ps aux | grep evaluate

# GPU 사용량 확인
nvidia-smi
```

### 프로세스 중단

```bash
# 전체 중단
pkill -f "weighted_mtp.cli.evaluate"

# 특정 PID 중단
kill <PID>
```

## 예상 소요 시간 (H100 기준)

| Dataset | Tasks | 예상 시간 |
|---------|-------|----------|
| HumanEval | 164 | 30분 ~ 1시간 |
| MBPP | 500 | 1 ~ 2시간 |
| GSM8K | 1,319 | 2 ~ 4시간 |
| CodeContests | 165 | 30분 ~ 1시간 |

## MLflow 결과 확인

### 1. 서버에서 mlruns/ 다운로드

```bash
# 로컬에서 실행
rsync -avz --progress -e "ssh -i ~/.ssh/for_personal -p <PORT>" \
    root@ssh4.vast.ai:/workspace/weighted_mtp/mlruns/ \
    /Users/wesley/Desktop/wooshikwon/weighted_mtp/mlruns/
```

### 2. 경로 수정 (서버 경로 -> 로컬 경로)

```bash
# 로컬에서 실행
find /Users/wesley/Desktop/wooshikwon/weighted_mtp/mlruns \
    -name "meta.yaml" -exec sed -i '' \
    's|file:///workspace/weighted_mtp/mlruns|file:///Users/wesley/Desktop/wooshikwon/weighted_mtp/mlruns|g' {} \;
```

### 3. MLflow UI 실행

```bash
cd /Users/wesley/Desktop/wooshikwon/weighted_mtp
mlflow ui --port 5000
```

브라우저에서 `http://localhost:5000` 접속

### MLflow에 저장되는 항목

**Parameters:**
- checkpoint, dataset, num_samples_per_task, temperature
- max_new_tokens, checkpoint_epoch, checkpoint_val_loss

**Metrics:**
- pass@1, pass@5, pass@10, pass@20

**Artifacts:**
- `results_{dataset}.csv`: 태스크별 Pass@K 결과
- `samples_{dataset}.jsonl`: 처음 10개 태스크의 생성 샘플

## 결과 확인 (MLflow 없이)

평가 완료 후 로그에서 직접 확인:

```bash
grep -A 10 "Evaluation Results\|pass@" logs/eval_*.log
```

## 문제 해결

### ModuleNotFoundError: No module named 'weighted_mtp'

```bash
# uv run 사용 (권장)
uv run python -m weighted_mtp.cli.evaluate ...

# 또는 PYTHONPATH 설정
PYTHONPATH=src python -m weighted_mtp.cli.evaluate ...
```

### MLflow enabled but MLFLOW_TRACKING_URI not set

```bash
# 로컬 파일 기반 MLflow 사용
export MLFLOW_TRACKING_URI=file:///workspace/weighted_mtp/mlruns
```

### safetensors not found

S3에서 모델 다운로드 필요 (위 "사전 준비" 섹션 참조)

### CodeContests 테스트 케이스 파일이 없습니다

```bash
aws s3 cp s3://wmtp/weighted-mtp/datasets/codecontests/tests/ \
    /workspace/weighted_mtp/storage/datasets/codecontests/tests/ --recursive
```

### CodeContests 평가용 데이터셋 파일이 없습니다

```bash
uv run python scripts/create_storage/create_codecontests_eval.py
```
