#!/bin/bash
# VESSL Storage에서 micro 모델 다운로드

set -e

STORAGE_NAME="vessl-storage"

echo "=== Micro 모델 다운로드 시작 ==="

# Micro-MTP 다운로드
echo "1. Micro-MTP 다운로드 중..."
mkdir -p storage/models/micro-mtp/{configs,safetensors}

vessl storage copy-file \
  volume://$STORAGE_NAME/models/micro-mtp/configs/config.json \
  storage/models/micro-mtp/configs/config.json

vessl storage copy-file \
  volume://$STORAGE_NAME/models/micro-mtp/metadata.json \
  storage/models/micro-mtp/metadata.json

vessl storage copy-file \
  volume://$STORAGE_NAME/models/micro-mtp/safetensors/SHA256SUMS \
  storage/models/micro-mtp/safetensors/SHA256SUMS

vessl storage copy-file \
  volume://$STORAGE_NAME/models/micro-mtp/safetensors/model.safetensors \
  storage/models/micro-mtp/safetensors/model.safetensors

echo "✓ Micro-MTP 다운로드 완료"

# Micro-Ref 다운로드
echo "2. Micro-Ref 다운로드 중..."
mkdir -p storage/models/micro-ref/{configs,safetensors}

vessl storage copy-file \
  volume://$STORAGE_NAME/models/micro-ref/configs/config.json \
  storage/models/micro-ref/configs/config.json

vessl storage copy-file \
  volume://$STORAGE_NAME/models/micro-ref/metadata.json \
  storage/models/micro-ref/metadata.json

vessl storage copy-file \
  volume://$STORAGE_NAME/models/micro-ref/safetensors/SHA256SUMS \
  storage/models/micro-ref/safetensors/SHA256SUMS

vessl storage copy-file \
  volume://$STORAGE_NAME/models/micro-ref/safetensors/model.safetensors \
  storage/models/micro-ref/safetensors/model.safetensors

echo "✓ Micro-Ref 다운로드 완료"

echo ""
echo "=== 다운로드 확인 ==="
du -sh storage/models/micro-*

echo ""
echo "=== ✓ Micro 모델 다운로드 완료 ==="
