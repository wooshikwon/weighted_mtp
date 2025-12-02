#!/bin/bash
# S3 업로드 스크립트 (Vast.ai 학습용)
# 테스트 모델 제외, 필수 파일만 업로드

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 프로젝트 루트
PROJECT_ROOT="/Users/wesley/Desktop/wooshikwon/weighted_mtp"
BUCKET="s3://wmtp/weighted-mtp"

# AWS credentials 로드
source "${PROJECT_ROOT}/.env"
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION

echo -e "${GREEN}=== S3 업로드 시작 ===${NC}"
echo "Bucket: ${BUCKET}"
echo ""

# 1. Models 업로드 (테스트 모델 제외)
echo -e "${YELLOW}[1/4] Models 업로드 중... (meta-llama-mtp: 50GB)${NC}"
aws s3 sync "${PROJECT_ROOT}/storage/models/meta-llama-mtp/" "${BUCKET}/models/meta-llama-mtp/" \
    --exclude "*.DS_Store"

echo -e "${YELLOW}[2/4] Models 업로드 중... (ref-sheared-llama-2.7b: 10GB)${NC}"
aws s3 sync "${PROJECT_ROOT}/storage/models/ref-sheared-llama-2.7b/" "${BUCKET}/models/ref-sheared-llama-2.7b/" \
    --exclude "*.DS_Store" \
    --exclude "safetensors/*"

# 2. Datasets 업로드
echo -e "${YELLOW}[3/4] Datasets 업로드 중... (~8GB)${NC}"
aws s3 sync "${PROJECT_ROOT}/storage/datasets/" "${BUCKET}/datasets/" \
    --exclude "*.DS_Store"

# 3. Critic checkpoint 업로드
echo -e "${YELLOW}[4/4] Critic checkpoint 업로드 중... (~99MB)${NC}"
aws s3 sync "${PROJECT_ROOT}/storage/critic/" "${BUCKET}/checkpoints/critic/" \
    --exclude "*.DS_Store"

echo ""
echo -e "${GREEN}=== 업로드 완료 ===${NC}"
echo ""

# 업로드 결과 확인
echo "업로드된 파일 목록:"
aws s3 ls "${BUCKET}/" --recursive --summarize | tail -5
