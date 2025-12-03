# Baseline MTP 학습 가이드

Vast.ai A100 x4 환경에서 Baseline MTP 실험을 실행하는 단계별 가이드입니다.

## 실험 개요

| 항목 | 값 |
|------|-----|
| 실험명 | lora-mtp-baseline |
| 모델 | meta-llama-mtp (7B) |
| 데이터 | CodeContests 120,000 samples |
| Epochs | 2 |
| Effective Batch Size | 192 (12 × 4 GPU × 4 accum) |
| LoRA | rank=64, alpha=128 |
| 예상 학습 시간 | ~1-2시간 |
| 예상 비용 | ~$8-10 |

---

## Step 1: S3 데이터 업로드 (최초 1회)

로컬에서 실행:

```bash
cd /Users/wesley/Desktop/wooshikwon/weighted_mtp
./scripts/vastai/upload_to_s3.sh
```

업로드 확인:
```bash
aws s3 ls s3://wmtp/weighted-mtp/ --recursive --summarize
```

---

## Step 2: 인스턴스 대여

```bash
# A100 80GB x4 인스턴스 검색
vastai search offers \
    'gpu_name=A100_SXM4 num_gpus=4 gpu_ram>=80 reliability>0.95' \
    -o 'dph+'

# 인스턴스 생성 (OFFER_ID는 검색 결과에서 확인)
vastai create instance OFFER_ID \
    --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
    --disk 250 \
    --ssh \
    --direct
```

---

## Step 3: SSH 접속

```bash
# 인스턴스 ID 확인
vastai show instances

# SSH 접속 정보 확인
vastai ssh-url INSTANCE_ID

# 접속
ssh -i ~/.ssh/for_personal root@<IP> -p <PORT>
```

---

## Step 4: 인스턴스 환경 설정

### 4.1 시스템 패키지 설치

```bash
apt-get update && apt-get install -y curl git vim htop tmux
```

### 4.2 UV 패키지 매니저 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 4.3 AWS CLI 설치 및 설정

```bash
pip install awscli -q

# AWS credentials 설정
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
export AWS_DEFAULT_REGION="eu-north-1"
```

### 4.4 S3에서 데이터 다운로드

```bash
mkdir -p /workspace/storage/{models,datasets,checkpoints}

# 모델 다운로드 (~60GB, 약 10분)
aws s3 sync s3://wmtp/weighted-mtp/models/ /workspace/storage/models/ \
    --cli-read-timeout 300

# 데이터셋 다운로드
aws s3 sync s3://wmtp/weighted-mtp/datasets/ /workspace/storage/datasets/
```

### 4.5 프로젝트 클론 및 설정

```bash
cd /workspace
git clone https://github.com/wooshikwon/weighted_mtp.git
cd weighted_mtp

# Storage 심볼릭 링크
ln -sf /workspace/storage/models storage/models
ln -sf /workspace/storage/datasets storage/datasets
ln -sf /workspace/storage/checkpoints storage/checkpoints

# 의존성 설치
uv sync --frozen
```

### 4.6 환경변수 설정

```bash
cat > .env << 'EOF'
NCCL_DEBUG=WARN
NCCL_TIMEOUT=3600
NCCL_IB_DISABLE=1
EOF

source .env
```

### 4.7 GPU 확인

```bash
nvidia-smi
uv run python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

---

## Step 5: Baseline 학습 실행

### 5.1 tmux 세션 생성

```bash
tmux new -s training
```

### 5.2 학습 실행

```bash
cd /workspace/weighted_mtp

torchrun --nproc_per_node=4 \
    -m weighted_mtp.pipelines.run_baseline \
    --config configs/production/baseline.yaml
```

### 5.3 tmux 사용법

- **Detach** (세션 유지하며 나가기): `Ctrl+B`, `D`
- **Re-attach** (세션 복귀): `tmux attach -t training`

---

## Step 6: 학습 모니터링 (선택)

### MLflow 로컬 서버 실행

인스턴스 내 새 터미널에서:
```bash
cd /workspace/weighted_mtp
uv run mlflow ui --host 0.0.0.0 --port 5000 &
```

로컬에서 포트 포워딩:
```bash
ssh -i ~/.ssh/for_personal -L 5000:localhost:5000 root@<IP> -p <PORT>
```

브라우저에서 `http://localhost:5000` 접속

---

## Step 7: 학습 완료 후 처리

### 7.1 인스턴스 Stop (GPU 해제)

로컬에서 실행:
```bash
vastai stop instance INSTANCE_ID
```

### 7.2 결과 다운로드

인스턴스 재시작 후:
```bash
# 인스턴스 재시작
vastai start instance INSTANCE_ID

# 결과 다운로드
scp -r -i ~/.ssh/for_personal -P <PORT> \
    root@<IP>:/workspace/weighted_mtp/storage/checkpoints/baseline ./local_checkpoints/

scp -r -i ~/.ssh/for_personal -P <PORT> \
    root@<IP>:/workspace/weighted_mtp/mlruns ./local_mlruns/
```

### 7.3 인스턴스 완전 삭제

```bash
vastai destroy instance INSTANCE_ID
```

---

## Config 설정 참조

```yaml
# configs/production/baseline.yaml 주요 설정

experiment:
  name: lora-mtp-baseline

models:
  policy:
    name: meta-llama-mtp
    path: storage/models/meta-llama-mtp

dataset:
  name: codecontests
  train: storage/datasets/codecontests/processed/train.jsonl
  max_length: 2048

data_sampling:
  n_samples: 120000

training:
  n_epochs: 2.0
  batch_size: 12
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  use_lora: true
  lora:
    rank: 64
    alpha: 128.0

checkpoint:
  save_dir: storage/checkpoints/baseline/${experiment.name}
  save_checkpoint_every: 0.2
  save_lora_only: true
```

---

## 문제 해결

### NCCL 타임아웃

```bash
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=1
```

### GPU OOM

```bash
# batch_size 줄이기
torchrun ... --override training.batch_size=8
```

### 학습 재개 (체크포인트에서)

```bash
# TODO: resume 기능 구현 시 추가
```

---

## 비용 요약

| 단계 | 시간 | 비용 |
|------|------|------|
| 환경 설정 + S3 다운로드 | ~20분 | ~$1.5 |
| Baseline 학습 | ~1-2시간 | ~$5-9 |
| 결과 다운로드 | ~10분 | ~$0.7 |
| **총계** | **~2시간** | **~$8-10** |

---

*Last Updated: 2025-12-01*
