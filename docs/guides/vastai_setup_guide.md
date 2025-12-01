# Vast.ai GPU 렌탈 가이드: Weighted MTP 학습 환경 설정

Vast.ai에서 A100 80GB x4 환경으로 Weighted MTP 학습을 실행하기 위한 가이드입니다.

## 목차

1. [사전 요구사항](#1-사전-요구사항)
2. [Vast.ai CLI 설치 및 설정](#2-vastai-cli-설치-및-설정)
3. [인스턴스 검색 및 대여](#3-인스턴스-검색-및-대여)
4. [데이터 준비 및 전송](#4-데이터-준비-및-전송)
5. [인스턴스 환경 설정](#5-인스턴스-환경-설정)
6. [학습 파이프라인 실행](#6-학습-파이프라인-실행)
7. [Checkpoint 관리](#7-checkpoint-관리)
8. [비용 최적화](#8-비용-최적화)
9. [문제 해결](#9-문제-해결)

---

## 1. 사전 요구사항

### 1.1 필요 리소스

| 항목 | 요구사항 |
|------|----------|
| GPU | A100 80GB x4 (SXM4 권장) |
| VRAM | 320GB 총합 |
| 시스템 RAM | 256GB+ 권장 |
| 스토리지 | 200GB+ (모델 + 데이터 + 체크포인트) |
| 네트워크 | 1Gbps+ (데이터 전송용) |

### 1.2 데이터 크기

```
storage/
├── models/
│   ├── meta-llama-mtp/           # 25GB (7B MTP 모델)
│   └── ref-sheared-llama-2.7b/   # 10GB (Value Model 베이스)
├── datasets/
│   └── codecontests/             # ~5GB (학습 데이터)
└── checkpoints/
    └── critic/                   # ~500MB (Critic 체크포인트)

총 필요 용량: ~50GB (초기) + 체크포인트 저장 공간
```

### 1.3 로컬 환경 준비

```bash
# Vast.ai CLI 설치
pip install vastai

# AWS CLI 설치 (S3 데이터 전송용)
pip install awscli
aws configure  # AWS credentials 설정
```

---

## 2. Vast.ai CLI 설치 및 설정

### 2.1 CLI 설치

```bash
# PyPI에서 설치 (권장)
pip install vastai

# 또는 최신 버전 직접 다운로드
wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast
chmod +x vast
```

### 2.2 API 키 설정

1. [Vast.ai Console](https://cloud.vast.ai/cli/)에서 API 키 확인
2. API 키 등록:

```bash
vastai set api-key YOUR_API_KEY_HERE
```

### 2.3 CLI 동작 확인

```bash
# 계정 정보 확인
vastai show user

# 사용 가능한 GPU 목록 확인
vastai search offers 'gpu_name=A100' --limit 5
```

---

## 3. 인스턴스 검색 및 대여

### 3.1 A100 80GB x4 인스턴스 검색

```bash
# 기본 검색: A100 SXM4 80GB x4
vastai search offers \
    'gpu_name=A100_SXM4 num_gpus=4 gpu_ram>=80 reliability>0.95 inet_down>=500' \
    -o 'dph+'

# 또는 PCIE 버전 (약간 저렴)
vastai search offers \
    'gpu_name=A100_PCIE num_gpus=4 gpu_ram>=80 reliability>0.95' \
    -o 'dph+'
```

### 3.2 검색 결과 필터 옵션

| 필터 | 설명 | 예시 |
|------|------|------|
| `gpu_name` | GPU 모델명 | `A100_SXM4`, `A100_PCIE` |
| `num_gpus` | GPU 개수 | `4` |
| `gpu_ram` | GPU 메모리 (GB) | `>=80` |
| `reliability` | 안정성 점수 | `>0.95` |
| `inet_down` | 다운로드 속도 (Mbps) | `>=500` |
| `disk_space` | 디스크 용량 (GB) | `>=200` |

### 3.3 인스턴스 대여

```bash
# 검색 결과에서 offer_id 확인 후 대여
vastai create instance OFFER_ID \
    --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
    --disk 250 \
    --ssh \
    --direct
```

**주요 옵션:**
- `--image`: Docker 이미지 (PyTorch 2.5.1 + CUDA 12.4 권장)
- `--disk`: 디스크 용량 (GB)
- `--ssh`: SSH 접속 활성화
- `--direct`: 직접 SSH 연결 (프록시 우회, 빠른 전송)

### 3.4 인스턴스 상태 확인

```bash
# 내 인스턴스 목록
vastai show instances

# 특정 인스턴스 상세 정보
vastai show instance INSTANCE_ID
```

### 3.5 SSH 접속

```bash
# SSH 접속 정보 확인
vastai ssh-url INSTANCE_ID

# 접속 (출력된 명령어 실행)
ssh -p PORT root@HOST -L 8080:localhost:8080
```

---

## 4. 데이터 준비 및 전송

### 4.1 S3에 데이터 업로드 (로컬에서)

먼저 로컬의 데이터를 S3에 업로드합니다:

```bash
# 프로젝트 루트에서 실행
cd /path/to/weighted_mtp

# 모델 업로드
aws s3 sync storage/models/ s3://YOUR_BUCKET/weighted-mtp/models/ \
    --exclude "*.DS_Store"

# 데이터셋 업로드
aws s3 sync storage/datasets/ s3://YOUR_BUCKET/weighted-mtp/datasets/ \
    --exclude "*.DS_Store"

# Critic 체크포인트 업로드 (verifiable 학습 시 필요)
aws s3 sync storage/checkpoints/critic/ s3://YOUR_BUCKET/weighted-mtp/checkpoints/critic/
```

### 4.2 인스턴스에서 S3 데이터 다운로드

SSH로 인스턴스에 접속한 후:

```bash
# AWS CLI 설치 및 설정
pip install awscli
aws configure  # credentials 입력

# 작업 디렉터리 생성
mkdir -p /workspace/storage/{models,datasets,checkpoints}

# 모델 다운로드 (병렬 전송으로 속도 향상)
aws s3 sync s3://YOUR_BUCKET/weighted-mtp/models/ /workspace/storage/models/ \
    --cli-read-timeout 300

# 데이터셋 다운로드
aws s3 sync s3://YOUR_BUCKET/weighted-mtp/datasets/ /workspace/storage/datasets/

# Critic 체크포인트 다운로드
aws s3 sync s3://YOUR_BUCKET/weighted-mtp/checkpoints/ /workspace/storage/checkpoints/
```

### 4.3 Vast.ai Cloud Sync 사용 (대안)

Vast.ai의 내장 S3 동기화 기능:

```bash
# S3 연결 설정 (웹 콘솔에서 먼저 인증 필요)
vastai cloud copy s3://YOUR_BUCKET/weighted-mtp/ C.INSTANCE_ID:/workspace/storage/
```

### 4.4 데이터 무결성 확인

```bash
# 모델 파일 확인
ls -lh /workspace/storage/models/meta-llama-mtp/
ls -lh /workspace/storage/models/ref-sheared-llama-2.7b/

# 데이터셋 확인
wc -l /workspace/storage/datasets/codecontests/processed/*.jsonl
```

---

## 5. 인스턴스 환경 설정

### 5.1 시스템 패키지 설치

```bash
# 기본 패키지
apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    htop \
    tmux
```

### 5.2 UV 패키지 매니저 설치

```bash
# UV 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 설치 확인
uv --version
```

### 5.3 프로젝트 클론 및 의존성 설치

```bash
# 프로젝트 클론
cd /workspace
git clone https://github.com/wooshikwon/weighted_mtp.git
cd weighted_mtp

# 의존성 설치 (lock 파일 사용)
uv sync --frozen

# 또는 fresh install
uv sync
```

### 5.4 Storage 심볼릭 링크 설정

```bash
# 프로젝트 내 storage 디렉터리 연결
cd /workspace/weighted_mtp
mkdir -p storage
ln -sf /workspace/storage/models storage/models
ln -sf /workspace/storage/datasets storage/datasets
ln -sf /workspace/storage/checkpoints storage/checkpoints

# 확인
ls -la storage/
```

### 5.5 환경변수 설정

```bash
# .env 파일 생성
cat > .env << 'EOF'
# HuggingFace (optional)
HF_TOKEN=your_hf_token_here

# AWS S3 (체크포인트 업로드용)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=eu-north-1

# MLflow (optional)
MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

# NCCL 설정 (Multi-GPU)
NCCL_DEBUG=WARN
NCCL_TIMEOUT=3600
NCCL_IB_DISABLE=1
EOF

# 환경변수 로드
source .env
export $(grep -v '^#' .env | xargs)
```

### 5.6 GPU 상태 확인

```bash
# GPU 확인
nvidia-smi

# PyTorch CUDA 확인
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

---

## 6. 학습 파이프라인 실행

### 6.1 Single GPU 실행

```bash
# Baseline MTP
uv run python -m weighted_mtp.pipelines.run_baseline \
    --config configs/production/baseline.yaml

# Critic Pre-training
uv run python -m weighted_mtp.pipelines.run_critic \
    --config configs/production/critic_mlp.yaml

# Verifiable WMTP
uv run python -m weighted_mtp.pipelines.run_verifiable \
    --config configs/production/verifiable.yaml
```

### 6.2 Multi-GPU 실행 (4-GPU)

```bash
# tmux 세션에서 실행 권장
tmux new -s training

# Baseline MTP (4-GPU)
uv run torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    -m weighted_mtp.pipelines.run_baseline \
    --config configs/production/baseline.yaml

# Verifiable WMTP (4-GPU)
uv run torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    -m weighted_mtp.pipelines.run_verifiable \
    --config configs/production/verifiable.yaml
```

### 6.3 Config Override

```bash
# Batch size, gradient accumulation 조정
uv run torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    -m weighted_mtp.pipelines.run_verifiable \
    --config configs/production/verifiable.yaml \
    --override training.batch_size=24 \
    --override training.gradient_accumulation_steps=2
```

### 6.4 백그라운드 실행

```bash
# nohup으로 백그라운드 실행
nohup uv run torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    -m weighted_mtp.pipelines.run_verifiable \
    --config configs/production/verifiable.yaml \
    > training.log 2>&1 &

# 로그 모니터링
tail -f training.log

# 또는 tmux에서 실행 후 detach (Ctrl+B, D)
```

### 6.5 학습 모니터링

```bash
# GPU 사용량 실시간 모니터링
watch -n 1 nvidia-smi

# 메모리 사용량
htop

# 학습 로그 확인
tail -f training.log
```

---

## 7. Checkpoint 관리

### 7.1 로컬 체크포인트 위치

```bash
# 체크포인트 디렉터리 구조
storage/checkpoints/
├── baseline/
│   └── {experiment_name}/
│       ├── checkpoint_epoch_0.20.pt
│       ├── checkpoint_best.pt
│       └── checkpoint_final.pt
├── critic/
│   └── {experiment_name}/
└── verifiable/
    └── {experiment_name}/
```

### 7.2 체크포인트 S3 자동 업로드

Config에서 S3 업로드 활성화:

```yaml
# configs/production/verifiable.yaml
checkpoint:
  save_dir: storage/checkpoints/verifiable/${experiment.name}
  s3_upload: true  # S3 자동 업로드 활성화
```

### 7.3 수동 체크포인트 백업

```bash
# S3로 체크포인트 업로드
aws s3 sync storage/checkpoints/ s3://YOUR_BUCKET/weighted-mtp/checkpoints/ \
    --exclude "*.tmp"

# 특정 실험만 업로드
aws s3 sync storage/checkpoints/verifiable/lora-verifiable/ \
    s3://YOUR_BUCKET/weighted-mtp/checkpoints/verifiable/lora-verifiable/
```

### 7.4 인스턴스 종료 전 체크포인트 저장

```bash
#!/bin/bash
# save_and_shutdown.sh

echo "=== Syncing checkpoints to S3 ==="
aws s3 sync storage/checkpoints/ s3://YOUR_BUCKET/weighted-mtp/checkpoints/

echo "=== Syncing MLflow logs ==="
aws s3 sync mlruns/ s3://YOUR_BUCKET/weighted-mtp/mlruns/

echo "=== Done! Safe to shutdown ==="
```

---

## 8. 비용 최적화

### 8.1 예상 비용

| GPU 타입 | 개수 | 시간당 비용 (P25) | 24시간 |
|----------|------|------------------|--------|
| A100 SXM4 80GB | 4 | ~$3.5-5.0 | ~$85-120 |
| A100 PCIE 80GB | 4 | ~$2.5-4.0 | ~$60-95 |
| H100 SXM5 80GB | 4 | ~$8.0-12.0 | ~$190-290 |

### 8.2 비용 절감 팁

1. **On-Demand vs Reserved**
   - 단기 (< 24시간): On-Demand
   - 장기 (> 3일): Reserved 고려

2. **인스턴스 선택**
   - `reliability > 0.98`: 안정성 높은 호스트
   - `dph+` 정렬: 가격순 정렬

3. **효율적인 작업 흐름**
   ```bash
   # 1. 데이터 전송 완료 후 학습 시작
   # 2. tmux로 학습 실행
   # 3. 주기적 체크포인트 S3 동기화
   # 4. 학습 완료 후 즉시 인스턴스 종료
   ```

4. **Interruptible 인스턴스**
   - 최대 50% 저렴
   - 언제든 중단 가능
   - 체크포인트 자주 저장 필수

### 8.3 인스턴스 종료

```bash
# 인스턴스 중지 (데이터 유지, 요금 감소)
vastai stop instance INSTANCE_ID

# 인스턴스 완전 삭제 (데이터 삭제됨!)
vastai destroy instance INSTANCE_ID
```

---

## 9. 문제 해결

### 9.1 SSH 연결 실패

```bash
# 직접 연결 포트 확인
vastai show instance INSTANCE_ID | grep -i port

# SSH 키 권한 확인
chmod 600 ~/.ssh/id_rsa

# Proxy 연결 (직접 연결 실패 시)
vastai ssh INSTANCE_ID
```

### 9.2 NCCL 타임아웃

```bash
# 환경변수 설정
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# 또는 학습 명령어에 직접 추가
NCCL_TIMEOUT=3600 NCCL_IB_DISABLE=1 uv run torchrun ...
```

### 9.3 GPU 메모리 부족 (OOM)

```bash
# Batch size 줄이기
--override training.batch_size=16

# Gradient accumulation 늘리기
--override training.gradient_accumulation_steps=4

# Effective batch size = batch_size × grad_accum × num_gpus
# 예: 16 × 4 × 4 = 256
```

### 9.4 데이터 전송 느림

```bash
# 병렬 전송 활성화
aws s3 sync ... --cli-read-timeout 300 --cli-connect-timeout 300

# 또는 다운로드 속도 빠른 인스턴스 선택
vastai search offers '... inet_down>=1000'
```

### 9.5 학습 중단 후 재개

```bash
# 체크포인트에서 재개 (config에서 설정)
# TODO: resume 기능 구현 시 추가
```

---

## 부록: 빠른 시작 스크립트

### setup_vastai.sh

```bash
#!/bin/bash
# Vast.ai 인스턴스 초기 설정 스크립트

set -e

echo "=== 시스템 패키지 설치 ==="
apt-get update && apt-get install -y curl git vim htop tmux

echo "=== UV 설치 ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== AWS CLI 설치 ==="
pip install awscli

echo "=== 프로젝트 클론 ==="
cd /workspace
git clone https://github.com/wooshikwon/weighted_mtp.git
cd weighted_mtp

echo "=== 의존성 설치 ==="
uv sync --frozen

echo "=== Storage 심볼릭 링크 ==="
mkdir -p storage /workspace/storage/{models,datasets,checkpoints}
ln -sf /workspace/storage/models storage/models
ln -sf /workspace/storage/datasets storage/datasets
ln -sf /workspace/storage/checkpoints storage/checkpoints

echo "=== 완료! ==="
echo "다음 단계:"
echo "1. aws configure로 AWS credentials 설정"
echo "2. S3에서 데이터 다운로드"
echo "3. .env 파일 생성"
echo "4. 학습 시작"
```

---

## 참고 자료

- [Vast.ai Documentation](https://docs.vast.ai/)
- [Vast.ai PyTorch Guide](https://docs.vast.ai/pytorch)
- [Vast.ai Data Movement](https://docs.vast.ai/instances/data-movement)
- [Vast.ai CLI Reference](https://vast.ai/docs/cli/quickstart)
- [Vast.ai Pricing](https://vast.ai/pricing/gpu/A100-SXM4)
- [vastai-client PyPI](https://pypi.org/project/vastai/)

---

*Last Updated: 2025-12-01*
