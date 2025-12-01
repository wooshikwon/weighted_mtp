#!/bin/bash
# Vast.ai 학습 실행 스크립트
#
# 사용법:
#   ./scripts/vastai/train.sh --pipeline verifiable --ngpus 4
#   ./scripts/vastai/train.sh --pipeline baseline --ngpus 4 --batch-size 24
#   ./scripts/vastai/train.sh --pipeline critic --ngpus 4

set -e

# 기본값
PIPELINE="verifiable"
NGPUS=4
BATCH_SIZE=""
GRAD_ACCUM=""
N_SAMPLES=""
S3_UPLOAD="true"

# CLI 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --pipeline)
            PIPELINE="$2"
            shift 2
            ;;
        --ngpus)
            NGPUS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --no-s3-upload)
            S3_UPLOAD="false"
            shift 1
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "사용법: $0 --pipeline <baseline|critic|verifiable> --ngpus <N> [options]"
            exit 1
            ;;
    esac
done

# 파이프라인별 설정
case $PIPELINE in
    baseline)
        CONFIG="configs/production/baseline.yaml"
        MODULE="weighted_mtp.pipelines.run_baseline"
        ;;
    critic)
        CONFIG="configs/production/critic_mlp.yaml"
        MODULE="weighted_mtp.pipelines.run_critic"
        ;;
    verifiable)
        CONFIG="configs/production/verifiable.yaml"
        MODULE="weighted_mtp.pipelines.run_verifiable"
        ;;
    *)
        echo "오류: 지원하지 않는 파이프라인: $PIPELINE"
        echo "지원 파이프라인: baseline, critic, verifiable"
        exit 1
        ;;
esac

# Override 인자 생성
OVERRIDE_ARGS=""
if [ -n "$BATCH_SIZE" ]; then
    OVERRIDE_ARGS="$OVERRIDE_ARGS --override training.batch_size=$BATCH_SIZE"
fi
if [ -n "$GRAD_ACCUM" ]; then
    OVERRIDE_ARGS="$OVERRIDE_ARGS --override training.gradient_accumulation_steps=$GRAD_ACCUM"
fi
if [ -n "$N_SAMPLES" ]; then
    OVERRIDE_ARGS="$OVERRIDE_ARGS --override data_sampling.n_samples=$N_SAMPLES"
fi
if [ "$S3_UPLOAD" = "true" ]; then
    OVERRIDE_ARGS="$OVERRIDE_ARGS --override checkpoint.s3_upload=true"
fi

# 환경변수 설정
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# NCCL 설정 (Multi-GPU)
if [ "$NGPUS" -gt 1 ]; then
    export NCCL_TIMEOUT=3600
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=1
    export NCCL_ASYNC_ERROR_HANDLING=1
fi

# .env 로드
if [ -f .env ]; then
    source .env
    export $(grep -v '^#' .env | xargs 2>/dev/null) || true
fi

# 설정 출력
echo "=============================================="
echo "  Weighted MTP Training - Vast.ai"
echo "=============================================="
echo ""
echo "Pipeline: $PIPELINE"
echo "Config: $CONFIG"
echo "GPU Count: $NGPUS"
[ -n "$BATCH_SIZE" ] && echo "Batch Size: $BATCH_SIZE (override)"
[ -n "$GRAD_ACCUM" ] && echo "Gradient Accumulation: $GRAD_ACCUM (override)"
[ -n "$N_SAMPLES" ] && echo "N Samples: $N_SAMPLES (override)"
echo "S3 Upload: $S3_UPLOAD"
echo ""

# GPU 확인
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo ""

# 학습 명령어 생성 및 실행
if [ "$NGPUS" -eq 1 ]; then
    CMD="uv run python -m $MODULE --config $CONFIG$OVERRIDE_ARGS"
else
    CMD="uv run torchrun --nproc_per_node=$NGPUS --nnodes=1 --node_rank=0 -m $MODULE --config $CONFIG$OVERRIDE_ARGS"
fi

echo "=== Running Command ==="
echo "$CMD"
echo ""

# 실행
eval $CMD
