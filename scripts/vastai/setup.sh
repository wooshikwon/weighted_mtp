#!/bin/bash
# Vast.ai 인스턴스 초기 설정 스크립트
# 사용법: curl -sSL https://raw.githubusercontent.com/wooshikwon/weighted_mtp/main/scripts/vastai/setup.sh | bash

set -e

echo "=============================================="
echo "  Weighted MTP - Vast.ai 환경 설정"
echo "=============================================="

# 1. 시스템 패키지
echo ""
echo "[1/6] 시스템 패키지 설치..."
apt-get update -qq && apt-get install -y -qq curl git vim htop tmux > /dev/null

# 2. UV 패키지 매니저
echo "[2/6] UV 패키지 매니저 설치..."
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# 3. AWS CLI
echo "[3/6] AWS CLI 설치..."
pip install -q awscli

# 4. 프로젝트 클론
echo "[4/6] 프로젝트 클론..."
cd /workspace
if [ ! -d "weighted_mtp" ]; then
    git clone -q https://github.com/wooshikwon/weighted_mtp.git
fi
cd weighted_mtp

# 5. 의존성 설치
echo "[5/6] Python 의존성 설치..."
uv sync --frozen > /dev/null 2>&1

# 6. Storage 디렉터리 설정
echo "[6/6] Storage 디렉터리 설정..."
mkdir -p storage /workspace/storage/{models,datasets,checkpoints,mlruns}
rm -rf storage/models storage/datasets storage/checkpoints mlruns 2>/dev/null || true
ln -sf /workspace/storage/models storage/models
ln -sf /workspace/storage/datasets storage/datasets
ln -sf /workspace/storage/checkpoints storage/checkpoints
ln -sf /workspace/storage/mlruns mlruns

echo ""
echo "=============================================="
echo "  설정 완료!"
echo "=============================================="
echo ""
echo "다음 단계:"
echo ""
echo "1. AWS credentials 설정:"
echo "   aws configure"
echo ""
echo "2. S3에서 데이터 다운로드:"
echo "   aws s3 sync s3://YOUR_BUCKET/weighted-mtp/models/ /workspace/storage/models/"
echo "   aws s3 sync s3://YOUR_BUCKET/weighted-mtp/datasets/ /workspace/storage/datasets/"
echo "   aws s3 sync s3://YOUR_BUCKET/weighted-mtp/checkpoints/ /workspace/storage/checkpoints/"
echo ""
echo "3. 환경변수 설정 (.env 파일 생성):"
echo "   cd /workspace/weighted_mtp"
echo "   cp .env.example .env  # 또는 직접 생성"
echo "   vim .env"
echo ""
echo "4. GPU 확인:"
echo "   nvidia-smi"
echo ""
echo "5. 학습 시작 (4-GPU):"
echo "   cd /workspace/weighted_mtp"
echo "   source .env"
echo "   uv run torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \\"
echo "       -m weighted_mtp.pipelines.run_verifiable \\"
echo "       --config configs/production/verifiable.yaml"
echo ""
