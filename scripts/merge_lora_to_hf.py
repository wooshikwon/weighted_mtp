#!/usr/bin/env python3
"""LoRA checkpoint를 base model에 merge하여 HuggingFace 형식으로 저장

SFT 학습 후 LoRA 가중치를 base model에 병합하여
다음 단계(Critic, Verifiable 등)에서 사용할 수 있는 full model 생성.

Usage:
    uv run python scripts/merge_lora_to_hf.py \
        --base-model storage/models/ref-sheared-llama-2.7b/raw \
        --lora-checkpoint storage/checkpoints/ref-tuning/ultimate-ref-tuning/checkpoint_best.pt \
        --output-dir storage/models/ref-sheared-llama-2.7b/sft-merged
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weighted_mtp.models.lora import (
    apply_lora_to_hf_model,
    load_hf_lora_state_dict,
    merge_lora_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="LoRA checkpoint를 base model에 merge하여 HuggingFace 형식으로 저장"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model path (e.g., storage/models/ref-sheared-llama-2.7b/raw)",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        required=True,
        help="LoRA checkpoint path (e.g., storage/checkpoints/.../checkpoint_best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    args = parser.parse_args()

    # Paths
    base_model_path = Path(args.base_model)
    lora_checkpoint_path = Path(args.lora_checkpoint)
    output_dir = Path(args.output_dir)

    # Validate paths
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    if not lora_checkpoint_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_checkpoint_path}")

    # dtype 설정
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # 1. Base model 로드
    logger.info(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",  # CPU에서 merge (메모리 효율)
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    logger.info(f"Base model loaded: {model.config.hidden_size}d, {model.config.num_hidden_layers}L")

    # 2. LoRA checkpoint 로드
    logger.info(f"Loading LoRA checkpoint: {lora_checkpoint_path}")
    checkpoint = torch.load(lora_checkpoint_path, map_location="cpu", weights_only=False)

    # Checkpoint 구조 확인
    if "lora_config" in checkpoint:
        lora_config = checkpoint["lora_config"]
        lora_state_dict = checkpoint.get("lora_state_dict", checkpoint.get("model_state_dict"))
    elif "model_state_dict" in checkpoint:
        # Legacy 형식: lora_config가 별도로 없는 경우
        lora_state_dict = checkpoint["model_state_dict"]
        # state_dict에서 LoRA config 추론
        lora_config = _infer_lora_config_from_state_dict(lora_state_dict)
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")

    logger.info(f"LoRA config: rank={lora_config.get('rank')}, "
                f"alpha={lora_config.get('alpha')}, "
                f"targets={lora_config.get('target_modules')}")

    # 3. LoRA 적용
    logger.info("Applying LoRA to model...")
    apply_lora_to_hf_model(model, lora_config)

    # 4. LoRA state_dict 로드
    logger.info("Loading LoRA weights...")
    load_hf_lora_state_dict(model, lora_state_dict)

    # 5. Merge
    logger.info("Merging LoRA weights into base model...")
    merge_lora_weights(model)

    # 6. 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving merged model to: {output_dir}")

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Checkpoint 메타데이터 저장 (참조용)
    metadata = {
        "base_model": str(base_model_path),
        "lora_checkpoint": str(lora_checkpoint_path),
        "lora_config": lora_config,
        "dtype": args.dtype,
    }
    if "epoch" in checkpoint:
        metadata["source_epoch"] = checkpoint["epoch"]
    if "val_metrics" in checkpoint:
        metadata["source_val_metrics"] = checkpoint["val_metrics"]

    import json
    with open(output_dir / "merge_info.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("=" * 50)
    logger.info("Merge completed successfully!")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 50)


def _infer_lora_config_from_state_dict(state_dict: dict) -> dict:
    """state_dict에서 LoRA config 추론 (legacy checkpoint 지원)"""
    # LoRA 키 패턴: layers.0.self_attn.q_proj.lora_A 등
    lora_keys = [k for k in state_dict.keys() if "lora_A" in k or "lora_B" in k]

    if not lora_keys:
        raise ValueError("No LoRA weights found in state_dict")

    # rank 추론 (lora_A의 shape에서)
    sample_key = next(k for k in lora_keys if "lora_A" in k)
    rank = state_dict[sample_key].shape[0]

    # target_modules 추론
    target_modules = set()
    for key in lora_keys:
        # 예: model.layers.0.self_attn.q_proj.lora_A → q_proj
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                target_modules.add(part)

    return {
        "rank": rank,
        "alpha": rank * 2,  # 기본값
        "dropout": 0.0,
        "target_modules": list(target_modules),
    }


if __name__ == "__main__":
    main()
