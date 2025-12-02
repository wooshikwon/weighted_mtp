#!/bin/bash
# Vast.ai 인스턴스 환경 설정 스크립트
# SSH 접속 후 실행: bash setup_instance.sh

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Vast.ai 인스턴스 환경 설정 ===${NC}"
echo ""

# 1. 시스템 패키지 설치
echo -e "${YELLOW}[1/7] 시스템 패키지 설치...${NC}"
apt-get update && apt-get install -y curl git vim htop tmux > /dev/null 2>&1
echo "완료"

# 2. UV 패키지 매니저 설치
echo -e "${YELLOW}[2/7] UV 패키지 매니저 설치...${NC}"
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
echo "UV version: $(uv --version)"

# 3. 프로젝트 클론
echo -e "${YELLOW}[3/7] 프로젝트 클론...${NC}"
cd /workspace
if [ ! -d "weighted_mtp" ]; then
    git clone https://github.com/wooshikwon/weighted_mtp.git
else
    echo "프로젝트가 이미 존재합니다. pull 진행..."
    cd weighted_mtp && git pull && cd ..
fi
cd weighted_mtp
echo "완료"

# 4. 의존성 설치
echo -e "${YELLOW}[4/7] Python 의존성 설치...${NC}"
uv sync --frozen
echo "완료"

# 5. AWS CLI 설치 및 설정
echo -e "${YELLOW}[5/7] AWS CLI 설정...${NC}"
pip install awscli -q

# AWS credentials 입력 요청
echo ""
echo -e "${RED}AWS credentials를 입력해주세요:${NC}"
read -p "AWS_ACCESS_KEY_ID: " AWS_ACCESS_KEY_ID
read -s -p "AWS_SECRET_ACCESS_KEY: " AWS_SECRET_ACCESS_KEY
echo ""
read -p "AWS_DEFAULT_REGION [eu-north-1]: " AWS_DEFAULT_REGION
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-eu-north-1}

export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION

# 6. S3에서 데이터 다운로드
echo -e "${YELLOW}[6/7] S3에서 데이터 다운로드...${NC}"
mkdir -p /workspace/storage/{models,datasets,checkpoints}

S3_BUCKET="s3://wmtp/weighted-mtp"

echo "  - models 다운로드 중... (약 60GB, 시간이 걸립니다)"
aws s3 sync "${S3_BUCKET}/models/" /workspace/storage/models/ \
    --cli-read-timeout 300

echo "  - datasets 다운로드 중..."
aws s3 sync "${S3_BUCKET}/datasets/" /workspace/storage/datasets/

echo "  - checkpoints 다운로드 중..."
aws s3 sync "${S3_BUCKET}/checkpoints/" /workspace/storage/checkpoints/

echo "다운로드 완료"

# 7. Storage 심볼릭 링크 설정
echo -e "${YELLOW}[7/7] Storage 심볼릭 링크 설정...${NC}"
cd /workspace/weighted_mtp
rm -rf storage 2>/dev/null || true
mkdir -p storage
ln -sf /workspace/storage/models storage/models
ln -sf /workspace/storage/datasets storage/datasets
ln -sf /workspace/storage/checkpoints storage/checkpoints

# .env 파일 생성
cat > .env << EOF
# AWS S3
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

# MLflow (optional)
MLFLOW_TRACKING_URI=http://13.50.240.176:5000
MLFLOW_TRACKING_USERNAME=wmtp_admin
MLFLOW_TRACKING_PASSWORD=wmtp_secure_2025

# NCCL 설정 (Multi-GPU)
NCCL_DEBUG=WARN
NCCL_TIMEOUT=3600
NCCL_IB_DISABLE=1
EOF

echo ""
echo -e "${GREEN}=== 환경 설정 완료 ===${NC}"
echo ""

# GPU 상태 확인
echo "GPU 상태:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo ""
echo "PyTorch CUDA 확인:"
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "Storage 디렉터리 확인:"
ls -la storage/

echo ""
echo -e "${GREEN}=== 학습 실행 명령어 ===${NC}"
echo ""
echo "# Baseline MTP (A100 x4):"
echo "  torchrun --nproc_per_node=4 -m weighted_mtp.pipelines.run_baseline --config configs/production/baseline.yaml"
echo ""
echo "# Critic Pre-training:"
echo "  torchrun --nproc_per_node=4 -m weighted_mtp.pipelines.run_critic --config configs/production/critic_mlp.yaml"
echo ""
echo "# Verifiable WMTP:"
echo "  torchrun --nproc_per_node=4 -m weighted_mtp.pipelines.run_verifiable --config configs/production/verifiable.yaml"
echo ""
