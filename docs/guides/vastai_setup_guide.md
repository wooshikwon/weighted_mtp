# Vast.ai GPU 렌탈 가이드: Weighted MTP 학습

Vast.ai에서 A100 80GB x4 환경으로 Weighted MTP 학습을 실행하기 위한 실전 가이드입니다.

## 현재 환경 상태

| 항목 | 상태 | 비고 |
|------|------|------|
| Vast.ai 계정 | 설정 완료 | $25 충전, 자동 충전 활성화 |
| API 키 | 등록 완료 | `vastai set api-key` |
| SSH 키 | 등록 완료 | `~/.ssh/for_personal.pub` |
| S3 버킷 | 데이터 업로드 완료 | `s3://wmtp/weighted-mtp/` |
| MLflow 서버 | 없음 | 인스턴스 내 로컬 실행 |

---

## 전체 워크플로우

```
[로컬] S3 업로드 (1회)
    ↓
[Vast.ai] 인스턴스 대여 ($4.32/hr)
    ↓
[Vast.ai] S3에서 데이터 다운로드
    ↓
[Vast.ai] 학습 실행 + MLflow 로컬 모니터링
    ↓
[Vast.ai] 학습 완료 → stop (GPU 해제, 디스크 유지)
    ↓
[로컬] 결과 다운로드 (SCP)
    ↓
[Vast.ai] destroy (완전 삭제)
```

---

## 1. 사전 준비 (로컬)

### 1.1 필요 도구 설치

```bash
# Vast.ai CLI
pip install vastai

# AWS CLI (S3 데이터 전송용)
pip install awscli
```

### 1.2 Vast.ai API 키 설정

```bash
# API 키 등록 (https://cloud.vast.ai/cli/ 에서 확인)
vastai set api-key YOUR_API_KEY

# 확인
vastai show user
```

### 1.3 SSH 키 등록

1. Vast.ai 웹 콘솔 접속: https://cloud.vast.ai/account/
2. "SSH Keys" 섹션에서 공개키 등록
3. 로컬 공개키 확인: `cat ~/.ssh/for_personal.pub`

---

## 2. S3 데이터 업로드 (최초 1회)

### 2.1 업로드 스크립트 실행

```bash
cd /Users/wesley/Desktop/wooshikwon/weighted_mtp
./scripts/vastai/upload_to_s3.sh
```

### 2.2 업로드 내용

| 디렉터리 | 크기 | 용도 |
|----------|------|------|
| `models/meta-llama-mtp/` | ~50GB | 7B MTP 모델 |
| `models/ref-sheared-llama-2.7b/raw/` | ~10GB | Reference 모델 |
| `datasets/` | ~8GB | CodeContests 데이터셋 |
| `checkpoints/critic/` | ~99MB | Critic 사전학습 체크포인트 |

### 2.3 업로드 확인

```bash
aws s3 ls s3://wmtp/weighted-mtp/ --recursive --summarize
```

---

## 3. 인스턴스 대여

### 3.1 A100 80GB x4 검색

```bash
vastai search offers \
    'gpu_name=A100_SXM4 num_gpus=4 gpu_ram>=80 reliability>0.95 inet_down>=500' \
    -o 'dph+'
```

### 3.2 인스턴스 생성

```bash
# OFFER_ID는 검색 결과에서 확인
vastai create instance OFFER_ID \
    --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
    --disk 250 \
    --ssh \
    --direct
```

### 3.3 인스턴스 상태 확인

```bash
# 인스턴스 목록
vastai show instances

# SSH 접속 정보
vastai ssh-url INSTANCE_ID
```

---

## 4. 인스턴스 환경 설정

### 4.1 SSH 접속

```bash
# vastai ssh-url 출력 결과 사용
ssh -i ~/.ssh/for_personal root@<IP> -p <PORT>
```

### 4.2 자동 설정 스크립트 실행

인스턴스 접속 후:

```bash
# 시스템 패키지 설치
apt-get update && apt-get install -y curl git vim htop tmux

# UV 패키지 매니저 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# AWS CLI 설치
pip install awscli -q
```

### 4.3 AWS Credentials 설정

```bash
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
export AWS_DEFAULT_REGION="eu-north-1"
```

### 4.4 S3에서 데이터 다운로드

```bash
mkdir -p /workspace/storage/{models,datasets,checkpoints}

# 모델 다운로드 (~60GB, 약 10-15분)
aws s3 sync s3://wmtp/weighted-mtp/models/ /workspace/storage/models/ --cli-read-timeout 300
aws s3 sync s3://wmtp/weighted-mtp/datasets/ /workspace/storage/datasets/
aws s3 sync s3://wmtp/weighted-mtp/checkpoints/ /workspace/storage/checkpoints/
```

### 4.5 프로젝트 클론 및 설정

```bash
cd /workspace
git clone https://github.com/wooshikwon/weighted_mtp.git
cd weighted_mtp

# Storage 심볼릭 링크
rm -rf storage 2>/dev/null || true
mkdir -p storage
ln -sf /workspace/storage/models storage/models
ln -sf /workspace/storage/datasets storage/datasets
ln -sf /workspace/storage/checkpoints storage/checkpoints

# 의존성 설치
uv sync --frozen
```

### 4.6 환경변수 설정

```bash
cat > .env << 'EOF'
# NCCL 설정 (Multi-GPU)
NCCL_DEBUG=WARN
NCCL_TIMEOUT=3600
NCCL_IB_DISABLE=1
EOF
```

### 4.7 GPU 상태 확인

```bash
nvidia-smi
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"
```

---

## 5. 학습 실행

### 5.1 tmux 세션 생성

```bash
tmux new -s training
```

### 5.2 학습 파이프라인 실행

```bash
cd /workspace/weighted_mtp
source .env

# Baseline MTP (A100 x4)
torchrun --nproc_per_node=4 \
    -m weighted_mtp.pipelines.run_baseline \
    --config configs/production/baseline.yaml

# Critic Pre-training
torchrun --nproc_per_node=4 \
    -m weighted_mtp.pipelines.run_critic \
    --config configs/production/critic_mlp.yaml

# Verifiable WMTP
torchrun --nproc_per_node=4 \
    -m weighted_mtp.pipelines.run_verifiable \
    --config configs/production/verifiable.yaml

# Rho-1 WMTP
torchrun --nproc_per_node=4 \
    -m weighted_mtp.pipelines.run_rho1 \
    --config configs/production/rho1.yaml
```

### 5.3 tmux 사용법

- **Detach** (세션 유지하며 나가기): `Ctrl+B`, `D`
- **Re-attach** (세션 복귀): `tmux attach -t training`
- **세션 목록**: `tmux ls`

---

## 6. MLflow 모니터링

### 6.1 인스턴스에서 MLflow 서버 실행

```bash
# 새 tmux 세션 또는 새 터미널에서
cd /workspace/weighted_mtp
mlflow ui --host 0.0.0.0 --port 5000 &
```

### 6.2 로컬에서 포트 포워딩

로컬 터미널에서:

```bash
ssh -i ~/.ssh/for_personal -L 5000:localhost:5000 root@<IP> -p <PORT>
```

### 6.3 브라우저에서 접속

```
http://localhost:5000
```

### 6.4 모니터링 가능 메트릭

| 메트릭 | 설명 |
|--------|------|
| `train/loss` | 학습 손실 |
| `train/mtp_loss` | MTP 손실 |
| `train/ntp_loss` | NTP 손실 |
| `train/learning_rate` | 학습률 |
| `eval/*` | 검증 메트릭 |

---

## 7. 학습 완료 후 처리

### 7.1 인스턴스 Stop (GPU 해제)

```bash
# 로컬에서 실행
vastai stop instance INSTANCE_ID
```

**Stop 상태:**
- GPU 해제 (과금 중지)
- 디스크 유지 (~$0.05/GB/월)
- 데이터 보존됨

### 7.2 결과 다운로드 (로컬에서)

인스턴스 재시작 후 결과 다운로드:

```bash
# 인스턴스 재시작 (다운로드용)
vastai start instance INSTANCE_ID

# SSH 접속 후 또는 직접 SCP
scp -r -i ~/.ssh/for_personal -P <PORT> \
    root@<IP>:/workspace/weighted_mtp/outputs ./local_outputs/

scp -r -i ~/.ssh/for_personal -P <PORT> \
    root@<IP>:/workspace/weighted_mtp/mlruns ./local_mlruns/
```

### 7.3 인스턴스 완전 삭제

```bash
vastai destroy instance INSTANCE_ID
```

---

## 8. 비용 관리

### 8.1 예상 비용

| 상태 | 비용 |
|------|------|
| 학습 중 (A100x4) | ~$4.32/hr |
| Stop 상태 (250GB 디스크) | ~$0.40/일 |
| S3 저장 (68GB) | ~$1.56/월 |

### 8.2 비용 최적화 전략

1. **학습 완료 즉시 Stop** - GPU 비용 절감
2. **결과 다운로드 후 Destroy** - 디스크 비용 절감
3. **S3는 초기 데이터 전송용으로만 사용** - 체크포인트는 인스턴스 로컬 저장

### 8.3 인스턴스 관리 명령어

```bash
# 목록 확인
vastai show instances

# 중지 (디스크 유지)
vastai stop instance INSTANCE_ID

# 재시작
vastai start instance INSTANCE_ID

# 완전 삭제
vastai destroy instance INSTANCE_ID
```

---

## 9. 문제 해결

### 9.1 SSH 연결 실패

```bash
# SSH 키 권한 확인
chmod 600 ~/.ssh/for_personal

# 포트 확인
vastai show instance INSTANCE_ID | grep -i port
```

### 9.2 NCCL 타임아웃

```bash
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
```

### 9.3 GPU OOM

```bash
# Config에서 batch_size 줄이기
--override training.batch_size=16
--override training.gradient_accumulation_steps=4
```

### 9.4 데이터 전송 느림

```bash
# 다운로드 속도 빠른 인스턴스 선택
vastai search offers '... inet_down>=1000'
```

---

## 빠른 참조

### 인스턴스 대여부터 학습까지

```bash
# 1. 인스턴스 검색 및 대여
vastai search offers 'gpu_name=A100_SXM4 num_gpus=4 gpu_ram>=80' -o 'dph+'
vastai create instance OFFER_ID --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel --disk 250 --ssh --direct

# 2. SSH 접속
vastai ssh-url INSTANCE_ID

# 3. 환경 설정 (인스턴스 내)
bash /workspace/weighted_mtp/scripts/vastai/setup_instance.sh

# 4. 학습 실행
tmux new -s training
torchrun --nproc_per_node=4 -m weighted_mtp.pipelines.run_baseline --config configs/production/baseline.yaml
```

### 학습 완료 후

```bash
# 1. Stop (GPU 해제)
vastai stop instance INSTANCE_ID

# 2. 결과 다운로드
vastai start instance INSTANCE_ID
scp -r -i ~/.ssh/for_personal -P <PORT> root@<IP>:/workspace/weighted_mtp/outputs ./

# 3. 완전 삭제
vastai destroy instance INSTANCE_ID
```

---

## 참고 자료

- [Vast.ai Documentation](https://docs.vast.ai/)
- [Vast.ai CLI Reference](https://vast.ai/docs/cli/quickstart)
- [Vast.ai Pricing](https://vast.ai/pricing/gpu/A100-SXM4)

---

*Last Updated: 2025-12-01*
