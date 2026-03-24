"""Baseline SFT checkpoint → Merged HF model for Value Model initialization.

Baseline SFT는 LoRA로 학습되므로, checkpoint에서 base model + LoRA를 로드한 뒤
LoRA를 base weights에 merge하여 순수 HF 모델로 저장한다.
이 모델은 Critic 학습 시 ValueModel.from_pretrained()의 입력으로 사용된다.

Usage:
    # Baseline 학습 완료 후 실행
    python scripts/merge_sft_for_value.py \
        --checkpoint storage/checkpoints/ntp_baseline/checkpoint_best.pt \
        --output storage/models/llama3-8b-code-merged

    # 이후 Critic 학습
    torchrun --nproc_per_node=4 python -m weighted_mtp train \
        --config configs/production/critic_mlp.yaml
"""

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Merge Baseline SFT LoRA into base model for Value Model initialization"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Baseline SFT checkpoint path (e.g., storage/checkpoints/ntp_baseline/checkpoint_best.pt)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model path (default: extracted from checkpoint config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="storage/models/llama3-8b-code-merged",
        help="Output directory for merged HF model",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 1. Load checkpoint and model using the evaluation loader
    #    (handles both full and LoRA checkpoint formats, applies merge)
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    from weighted_mtp.utils.checkpoint_utils import load_checkpoint_for_evaluation

    model, metadata = load_checkpoint_for_evaluation(
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
        inference_only=True,
    )
    logger.info(f"Model loaded — epoch: {metadata['epoch']}")

    # 2. Resolve base model path for tokenizer
    base_model_path = args.base_model
    if base_model_path is None:
        config = metadata.get("config", {})
        if "models" in config and "policy" in config["models"]:
            base_model_path = config["models"]["policy"]["path"]
        elif "model" in config:
            base_model_path = config["model"]["path"]
        else:
            base_model_path = "meta-llama/Meta-Llama-3-8B"
            logger.warning(f"Could not extract model path from checkpoint, using default: {base_model_path}")

    # 3. Save merged model as HF format
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    logger.info(f"Merged model saved to {output_path}")

    # 4. Copy tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    logger.info(f"Tokenizer saved to {output_path}")

    # 5. Summary
    param_count = sum(p.numel() for p in model.parameters())
    size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    logger.info(f"Done — {param_count/1e9:.2f}B params, {size_gb:.1f}GB")
    logger.info(f"Next: update critic_mlp.yaml → models.value_model.path: {output_path}")


if __name__ == "__main__":
    main()
