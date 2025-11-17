"""Micro 모델 생성 스크립트

Meta-LLaMA-MTP와 Ref-Sheared-LLaMA에서 경량 테스트용 micro 모델 생성

Usage:
    # Micro-MTP 생성 (512 hidden, 4 layers)
    python scripts/create_micro_models.py --type mtp

    # Micro-Ref 생성 (512 hidden, 4 layers)
    python scripts/create_micro_models.py --type ref

    # 둘 다 생성
    python scripts/create_micro_models.py --type all
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from weighted_mtp.models.meta_mtp import ModelArgs, Transformer


def create_micro_config(model_type: str) -> dict:
    """Micro 모델 config 생성"""
    base_config = {
        "n_future_tokens": 4,
        "hidden_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 1365,
        "rope_theta": 10000.0,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-05,
        "vocab_size": 32000,
        "cache": {"freqs_cis": True, "causal_mask": True},
        "dtype": "float16",
        "target_device": "mps"
    }

    if model_type == "mtp":
        base_config["model_name"] = "micro-mtp"
    else:
        base_config["model_name"] = "micro-ref"

    return base_config


def create_micro_model(model_type: str):
    """Micro 모델 생성 및 저장

    Args:
        model_type: "mtp" 또는 "ref"
    """

    print(f"\n{'='*60}")
    print(f"Micro-{model_type.upper()} 모델 생성")
    print(f"{'='*60}\n")

    # 1. 출력 디렉토리 설정
    output_dir = Path(f"storage/models/micro-{model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs_dir = output_dir / "configs"
    safetensors_dir = output_dir / "safetensors"
    configs_dir.mkdir(exist_ok=True)
    safetensors_dir.mkdir(exist_ok=True)

    print(f"[1/5] 출력 디렉토리: {output_dir}")

    # 2. Config 생성
    config = create_micro_config(model_type)
    config_path = configs_dir / "config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[2/5] Config 저장: {config_path}")
    print(f"  - Hidden: {config['hidden_size']}")
    print(f"  - Layers: {config['num_hidden_layers']}")
    print(f"  - Heads: {config['num_attention_heads']}")

    # 3. ModelArgs 생성
    model_args = ModelArgs(
        dim=config["hidden_size"],
        n_layers=config["num_hidden_layers"],
        n_heads=config["num_attention_heads"],
        n_kv_heads=config["num_key_value_heads"],
        vocab_size=config["vocab_size"],
        n_future_tokens=config["n_future_tokens"],
        rope_theta=config["rope_theta"],
        max_seq_len=config["max_position_embeddings"],
        norm_eps=config["rms_norm_eps"],
        intermediate_size=config["intermediate_size"],
    )

    print(f"[3/5] ModelArgs 생성 완료")

    # 4. 모델 생성 및 랜덤 초기화
    model = Transformer(model_args)

    # 랜덤 초기화 (deterministic)
    torch.manual_seed(42)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)

    print(f"[4/5] Transformer 모델 생성 완료 (랜덤 초기화)")

    # 5. Safetensors 저장
    state_dict = model.state_dict()
    safetensors_path = safetensors_dir / "model.safetensors"

    save_file(state_dict, safetensors_path)

    print(f"[5/5] Safetensors 저장: {safetensors_path}")

    # 6. SHA256 체크섬 생성
    sha256_hash = hashlib.sha256()
    with open(safetensors_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    checksum_path = safetensors_dir / "SHA256SUMS"
    with open(checksum_path, "w") as f:
        f.write(f"{sha256_hash.hexdigest()}  model.safetensors\n")

    print(f"     체크섬 저장: {checksum_path}")

    # 7. Metadata 생성
    metadata = {
        "model_name": f"micro-{model_type}",
        "type": "test",
        "description": f"Lightweight test model for {model_type.upper()}",
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_hidden_layers"],
            "num_heads": config["num_attention_heads"]
        },
        "files": {
            "config": "configs/config.json",
            "weights": "safetensors/model.safetensors",
            "checksum": "safetensors/SHA256SUMS"
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"     Metadata 저장: {metadata_path}")

    # 8. 검증
    print(f"\n{'='*60}")
    print(f"검증")
    print(f"{'='*60}")

    loaded_state = load_file(safetensors_path)
    print(f"✓ Safetensors 로드 성공")
    print(f"✓ 총 파라미터: {metadata['parameters']['total']:,}")
    print(f"✓ 파일 크기: {safetensors_path.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\n✓ Micro-{model_type.upper()} 생성 완료!\n")


def main():
    parser = argparse.ArgumentParser(description="Micro 모델 생성")
    parser.add_argument(
        "--type",
        choices=["mtp", "ref", "all"],
        required=True,
        help="생성할 모델 타입"
    )

    args = parser.parse_args()

    if args.type == "all":
        create_micro_model("mtp")
        create_micro_model("ref")
    else:
        create_micro_model(args.type)


if __name__ == "__main__":
    main()
