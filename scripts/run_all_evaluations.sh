#!/bin/bash
# 3개 실험(pure, baseline, verifiable) x 4개 데이터셋 순차 평가
# 각 GPU에서 데이터셋별로 3개 실험을 순차 실행
# 사용법: nohup ./scripts/run_all_evaluations.sh > logs/all_eval.log 2>&1 &

set -e

# MLflow 설정
export MLFLOW_TRACKING_URI=file:///workspace/weighted_mtp/mlruns

# 모델 경로
BASE_MODEL="storage/models/meta-llama-mtp"  # Pure (SFT 전) 모델
BASELINE_CKPT="storage/checkpoints/baseline/last-final-mtp-baseline/checkpoint_final.pt"
VERIFIABLE_CKPT="storage/checkpoints/verifiable/last-final-verifiable/checkpoint_final.pt"

# 평가 문제 수 제한 (전체 평가 시 주석 처리)
# GPU 균형을 위해 실행 시간 기준 조정:
# - CodeContests: 가장 느림 (stdin/stdout + 다수 테스트 케이스)
# - HumanEval/MBPP: 중간 (코드 실행)
# - GSM8K: 빠름 (텍스트 비교만)
MAX_TASKS_HUMANEVAL=80      # 전체 164
MAX_TASKS_MBPP=100          # 전체 500
MAX_TASKS_GSM8K=300         # 전체 1319
MAX_TASKS_CODECONTESTS=20   # 전체 165 (가장 느림)

# Pass@K를 위한 샘플 수 (모든 데이터셋 동일)
NUM_SAMPLES=20              # Pass@1, Pass@5, Pass@10, Pass@20 계산 가능

# 로그 디렉토리 생성
mkdir -p logs

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting all evaluations..."
echo "Pure (base): $BASE_MODEL"
echo "Baseline: $BASELINE_CKPT"
echo "Verifiable: $VERIFIABLE_CKPT"
echo ""

# GPU 0: HumanEval (pure → baseline → verifiable)
(
    echo "[GPU 0] HumanEval evaluation starting..."

    echo "[GPU 0] Pure - HumanEval"
    CUDA_VISIBLE_DEVICES=0 uv run python -m weighted_mtp.cli.evaluate \
        --base-model "$BASE_MODEL" \
        --dataset humaneval \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_HUMANEVAL \
        > logs/eval_humaneval_pure.log 2>&1

    echo "[GPU 0] Baseline - HumanEval"
    CUDA_VISIBLE_DEVICES=0 uv run python -m weighted_mtp.cli.evaluate \
        --checkpoint "$BASELINE_CKPT" \
        --dataset humaneval \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_HUMANEVAL \
        > logs/eval_humaneval_baseline.log 2>&1

    echo "[GPU 0] Verifiable - HumanEval"
    CUDA_VISIBLE_DEVICES=0 uv run python -m weighted_mtp.cli.evaluate \
        --checkpoint "$VERIFIABLE_CKPT" \
        --dataset humaneval \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_HUMANEVAL \
        > logs/eval_humaneval_verifiable.log 2>&1

    echo "[GPU 0] HumanEval done!"
) &

# GPU 1: MBPP (pure → baseline → verifiable)
(
    echo "[GPU 1] MBPP evaluation starting..."

    echo "[GPU 1] Pure - MBPP"
    CUDA_VISIBLE_DEVICES=1 uv run python -m weighted_mtp.cli.evaluate \
        --base-model "$BASE_MODEL" \
        --dataset mbpp \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_MBPP \
        > logs/eval_mbpp_pure.log 2>&1

    echo "[GPU 1] Baseline - MBPP"
    CUDA_VISIBLE_DEVICES=1 uv run python -m weighted_mtp.cli.evaluate \
        --checkpoint "$BASELINE_CKPT" \
        --dataset mbpp \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_MBPP \
        > logs/eval_mbpp_baseline.log 2>&1

    echo "[GPU 1] Verifiable - MBPP"
    CUDA_VISIBLE_DEVICES=1 uv run python -m weighted_mtp.cli.evaluate \
        --checkpoint "$VERIFIABLE_CKPT" \
        --dataset mbpp \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_MBPP \
        > logs/eval_mbpp_verifiable.log 2>&1

    echo "[GPU 1] MBPP done!"
) &

# GPU 2: GSM8K (pure → baseline → verifiable)
(
    echo "[GPU 2] GSM8K evaluation starting..."

    echo "[GPU 2] Pure - GSM8K"
    CUDA_VISIBLE_DEVICES=2 uv run python -m weighted_mtp.cli.evaluate \
        --base-model "$BASE_MODEL" \
        --dataset gsm8k \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_GSM8K \
        > logs/eval_gsm8k_pure.log 2>&1

    echo "[GPU 2] Baseline - GSM8K"
    CUDA_VISIBLE_DEVICES=2 uv run python -m weighted_mtp.cli.evaluate \
        --checkpoint "$BASELINE_CKPT" \
        --dataset gsm8k \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_GSM8K \
        > logs/eval_gsm8k_baseline.log 2>&1

    echo "[GPU 2] Verifiable - GSM8K"
    CUDA_VISIBLE_DEVICES=2 uv run python -m weighted_mtp.cli.evaluate \
        --checkpoint "$VERIFIABLE_CKPT" \
        --dataset gsm8k \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_GSM8K \
        > logs/eval_gsm8k_verifiable.log 2>&1

    echo "[GPU 2] GSM8K done!"
) &

# GPU 3: CodeContests (pure → baseline → verifiable)
(
    echo "[GPU 3] CodeContests evaluation starting..."

    echo "[GPU 3] Pure - CodeContests"
    CUDA_VISIBLE_DEVICES=3 uv run python -m weighted_mtp.cli.evaluate \
        --base-model "$BASE_MODEL" \
        --dataset codecontests \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_CODECONTESTS \
        > logs/eval_codecontests_pure.log 2>&1

    echo "[GPU 3] Baseline - CodeContests"
    CUDA_VISIBLE_DEVICES=3 uv run python -m weighted_mtp.cli.evaluate \
        --checkpoint "$BASELINE_CKPT" \
        --dataset codecontests \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_CODECONTESTS \
        > logs/eval_codecontests_baseline.log 2>&1

    echo "[GPU 3] Verifiable - CodeContests"
    CUDA_VISIBLE_DEVICES=3 uv run python -m weighted_mtp.cli.evaluate \
        --checkpoint "$VERIFIABLE_CKPT" \
        --dataset codecontests \
        --dtype bfloat16 \
        --num-samples $NUM_SAMPLES \
        --temperature-search "0.2,0.8" \
        --max-tasks $MAX_TASKS_CODECONTESTS \
        > logs/eval_codecontests_verifiable.log 2>&1

    echo "[GPU 3] CodeContests done!"
) &

# 모든 백그라운드 작업 대기
wait

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All evaluations completed!"
echo ""
echo "=== Logs ==="
ls -la logs/eval_*.log
echo ""

# 결과 요약 파싱
echo "=== Parsing Results ==="
uv run python scripts/parse_eval_results.py --logs-dir logs | tee logs/eval_summary.txt

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Summary saved to logs/eval_summary.txt"
