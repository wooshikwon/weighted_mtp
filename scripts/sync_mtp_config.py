#!/usr/bin/env python3
"""
실제 MTP params.json을 기준으로 meta_adapter.yaml 동기화

Source: storage/models_v2/meta-llama-mtp/raw/7B_1T_4/params.json
Target: storage/models_v2/meta-llama-mtp/configs/meta_adapter.yaml
"""

import json
import yaml
from pathlib import Path

def sync_mtp_config():
    base_dir = Path("storage/models_v2/meta-llama-mtp")
    params_file = base_dir / "raw/7B_1T_4/params.json"
    config_dir = base_dir / "configs"
    config_file = config_dir / "meta_adapter.yaml"

    # 디렉터리 생성
    config_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MTP Config 동기화 (실제 params.json → meta_adapter.yaml)")
    print("=" * 70)
    print()

    # params.json 로드
    if not params_file.exists():
        raise FileNotFoundError(f"params.json not found: {params_file}")

    with open(params_file, 'r') as f:
        params = json.load(f)

    print("HuggingFace 공식 params.json:")
    print(json.dumps(params, indent=2))
    print()

    # intermediate_size 계산
    dim = params["dim"]
    ffn_dim_multiplier = params.get("ffn_dim_multiplier", 1.0)
    multiple_of = params.get("multiple_of", 256)

    hidden_dim = int(8 * dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    intermediate_size = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    print(f"Calculated intermediate_size: {intermediate_size}")
    print(f"  (8 * {dim} / 3 = {int(8*dim/3)} → rounded to multiple of {multiple_of})")
    print()

    # meta_adapter.yaml 생성
    adapter_config = {
        "model_name": "meta-llama-mtp",
        "n_future_tokens": params["n_future_tokens"],
        "hidden_size": params["dim"],
        "num_hidden_layers": params["n_layers"],
        "num_attention_heads": params["n_heads"],
        "num_key_value_heads": params["n_kv_heads"],
        "intermediate_size": intermediate_size,
        "rope_theta": params["rope_theta"],
        "max_position_embeddings": 2048,  # LLaMA 1 기본값
        "rms_norm_eps": params["norm_eps"],
        "vocab_size": params["vocab_size"],
        "cache": {
            "freqs_cis": True,
            "causal_mask": True
        },
        "dtype": "float16"  # MTP 공식 dtype
    }

    # YAML 저장
    with open(config_file, 'w') as f:
        yaml.dump(adapter_config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Created: {config_file}")
    print()
    print("meta_adapter.yaml 내용:")
    print("-" * 70)
    with open(config_file, 'r') as f:
        print(f.read())
    print("-" * 70)
    print()

    # 중요 파라미터 하이라이트
    print("=" * 70)
    print("⚠️  주요 파라미터 (Meta 공식 값)")
    print("=" * 70)
    print(f"  rope_theta:       {params['rope_theta']} (LLaMA 1 기본값, LLaMA 2는 500000)")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  dtype:            float16 (MTP 공식 dtype)")
    print(f"  max_position_embeddings: 2048 (LLaMA 1 기본값)")
    print()
    print("=" * 70)
    print("✅ Config 동기화 완료!")
    print("=" * 70)
    print()
    print("다음 단계:")
    print("  - docs/00_ideal_structure.md 업데이트")
    print("  - docs/03_phase1_detailed_plan.md 업데이트")
    print()

if __name__ == "__main__":
    sync_mtp_config()
