#!/bin/bash
# MTP 7B_1T_4 모델 다운로드 및 변환 통합 스크립트

set -e  # 에러 발생 시 중단

echo "======================================================================="
echo "MTP 7B_1T_4 모델 설정 (다운로드 → 변환 → 검증)"
echo "======================================================================="
echo ""

# 환경 변수 확인
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN 환경변수가 설정되지 않았습니다."
    echo "   .env 파일을 확인하거나 다음 명령어로 설정하세요:"
    echo "   export HF_TOKEN=your_token_here"
    exit 1
fi

echo "✅ HF_TOKEN 확인됨"
echo ""

# Step 1: 다운로드
echo "======================================================================="
echo "Step 1/4: HuggingFace에서 MTP 7B_1T_4 다운로드"
echo "======================================================================="
echo "파일: consolidated.pth (25.1GB), params.json, tokenizer.model"
echo "이어받기 지원: 중단된 다운로드는 자동으로 재개됩니다"
echo ""
echo "다운로드를 시작합니다..."
echo ""

uv run python scripts/download_mtp_model.py

echo ""
echo "✅ 다운로드 완료"
echo ""

# Step 2: PyTorch → SafeTensors 변환
echo "======================================================================="
echo "Step 2/4: PyTorch → SafeTensors 변환"
echo "======================================================================="
echo ""

uv run python scripts/convert_mtp_to_safetensors.py

echo ""
echo "✅ 변환 완료"
echo ""

# Step 3: Config 동기화
echo "======================================================================="
echo "Step 3/4: meta_adapter.yaml을 실제 params.json 값으로 업데이트"
echo "======================================================================="
echo ""

uv run python scripts/sync_mtp_config.py

echo ""
echo "✅ Config 동기화 완료"
echo ""

# Step 4: 검증
echo "======================================================================="
echo "Step 4/4: 모델 무결성 검증"
echo "======================================================================="
echo ""

uv run python scripts/verify_mtp_model.py

echo ""
echo "======================================================================="
echo "✅ MTP 7B_1T_4 모델 설정 완료!"
echo "======================================================================="
echo ""
echo "다음 단계:"
echo "  - storage/models_v2/meta-llama-mtp/ 디렉터리 확인"
echo "  - Phase 2 (코드 스켈레톤 구축) 진행 가능"
echo ""
