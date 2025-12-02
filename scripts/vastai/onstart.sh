#!/bin/bash
# Vast.ai onstart 스크립트
# 인스턴스 시작 시 자동 실행되어 환경 설정

set -e

LOG_FILE="/workspace/setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Vast.ai 자동 환경 설정 시작 ==="
echo "시작 시간: $(date)"

# 1. 시스템 패키지
echo "[1/6] 시스템 패키지 설치..."
apt-get update && apt-get install -y curl git vim htop tmux > /dev/null 2>&1

# 2. UV 설치
echo "[2/6] UV 패키지 매니저 설치..."
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"

# 3. AWS CLI 설치
echo "[3/6] AWS CLI 설치..."
pip install awscli -q

# 4. S3에서 데이터 다운로드 (credentials는 환경변수로 전달)
echo "[4/6] S3에서 데이터 다운로드... (약 10-15분 소요)"
mkdir -p /workspace/storage/{models,datasets,checkpoints}

aws s3 sync s3://wmtp/weighted-mtp/models/ /workspace/storage/models/ --cli-read-timeout 300
aws s3 sync s3://wmtp/weighted-mtp/datasets/ /workspace/storage/datasets/
aws s3 sync s3://wmtp/weighted-mtp/checkpoints/ /workspace/storage/checkpoints/

# 5. 프로젝트 클론 및 설정
echo "[5/6] 프로젝트 클론..."
cd /workspace
git clone https://github.com/wooshikwon/weighted_mtp.git || (cd weighted_mtp && git pull)
cd weighted_mtp

# Storage 심볼릭 링크
rm -rf storage 2>/dev/null || true
mkdir -p storage
ln -sf /workspace/storage/models storage/models
ln -sf /workspace/storage/datasets storage/datasets
ln -sf /workspace/storage/checkpoints storage/checkpoints

# 6. 의존성 설치
echo "[6/6] Python 의존성 설치..."
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen

# .env 파일 생성 (환경변수에서 가져옴)
cat > .env << EOF
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-eu-north-1}
MLFLOW_TRACKING_URI=http://13.50.240.176:5000
MLFLOW_TRACKING_USERNAME=wmtp_admin
MLFLOW_TRACKING_PASSWORD=wmtp_secure_2025
NCCL_DEBUG=WARN
NCCL_TIMEOUT=3600
NCCL_IB_DISABLE=1
EOF

echo ""
echo "=== 환경 설정 완료 ==="
echo "완료 시간: $(date)"
echo ""
echo "GPU 상태:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""
echo "Storage 확인:"
ls -la /workspace/weighted_mtp/storage/
echo ""
echo "학습 실행 명령어:"
echo "  cd /workspace/weighted_mtp"
echo "  torchrun --nproc_per_node=4 -m weighted_mtp.pipelines.run_baseline --config configs/production/baseline.yaml"
