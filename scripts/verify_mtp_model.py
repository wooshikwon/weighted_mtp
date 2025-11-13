#!/usr/bin/env python3
"""
MTP 모델 무결성 검증

Checks:
1. SafeTensors 파일 존재 및 로딩
2. SHA256 해시 일치
3. Config 파일 일치
4. Tensor shape 검증
"""

from pathlib import Path
from safetensors import safe_open
import hashlib
import yaml
import json

def calculate_sha256(file_path):
    """파일의 SHA256 해시 계산"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_mtp_model():
    base_dir = Path("storage/models_v2/meta-llama-mtp")

    print("=" * 70)
    print("MTP 7B_1T_4 모델 무결성 검증")
    print("=" * 70)
    print()

    errors = []
    warnings = []

    # 1. 파일 존재 확인
    print("[1/4] 필수 파일 존재 확인...")
    required_files = {
        "safetensors": base_dir / "safetensors/model.safetensors",
        "sha256": base_dir / "safetensors/SHA256SUMS",
        "config": base_dir / "configs/meta_adapter.yaml",
        "params": base_dir / "raw/7B_1T_4/params.json",
        "metadata": base_dir / "metadata.json",
    }

    for name, file_path in required_files.items():
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024**2)
            print(f"  ✓ {name}: {file_path.name} ({size_mb:.1f} MB)")
        else:
            errors.append(f"Missing file: {file_path}")
            print(f"  ✗ {name}: NOT FOUND")

    print()

    if errors:
        print("❌ 필수 파일이 누락되었습니다!")
        for err in errors:
            print(f"  - {err}")
        return False

    # 2. SHA256 검증
    print("[2/4] SHA256 해시 검증...")
    safetensors_file = required_files["safetensors"]
    sha256_file = required_files["sha256"]

    with open(sha256_file, 'r') as f:
        expected_hash = f.read().strip().split()[0]

    print(f"  Expected: {expected_hash}")

    actual_hash = calculate_sha256(safetensors_file)
    print(f"  Actual:   {actual_hash}")

    if expected_hash == actual_hash:
        print("  ✓ SHA256 일치")
    else:
        errors.append("SHA256 mismatch")
        print("  ✗ SHA256 불일치!")

    print()

    # 3. SafeTensors 로딩 및 구조 검증
    print("[3/4] SafeTensors 구조 검증...")
    try:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"  ✓ Loaded {len(keys)} tensors")

            # 주요 텐서 확인
            important_tensors = [
                "tok_embeddings.weight",
                "layers.0.attention.wq.weight",
                "norm.weight",
                "output.weight"
            ]

            for tensor_name in important_tensors:
                if tensor_name in keys:
                    tensor = f.get_tensor(tensor_name)
                    print(f"  ✓ {tensor_name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
                else:
                    warnings.append(f"Tensor not found: {tensor_name}")

    except Exception as e:
        errors.append(f"Failed to load SafeTensors: {e}")
        print(f"  ✗ SafeTensors 로딩 실패: {e}")

    print()

    # 4. Config 일치 검증
    print("[4/4] Config 일치 검증...")

    with open(required_files["params"], 'r') as f:
        params = json.load(f)

    with open(required_files["config"], 'r') as f:
        config = yaml.safe_load(f)

    checks = [
        ("n_future_tokens", params["n_future_tokens"], config["n_future_tokens"]),
        ("hidden_size", params["dim"], config["hidden_size"]),
        ("num_layers", params["n_layers"], config["num_hidden_layers"]),
        ("num_heads", params["n_heads"], config["num_attention_heads"]),
        ("rope_theta", params["rope_theta"], config["rope_theta"]),
        ("vocab_size", params["vocab_size"], config["vocab_size"]),
    ]

    for name, expected, actual in checks:
        if expected == actual:
            print(f"  ✓ {name}: {actual}")
        else:
            errors.append(f"Config mismatch: {name} (expected={expected}, actual={actual})")
            print(f"  ✗ {name}: expected={expected}, actual={actual}")

    print()

    # 결과 출력
    print("=" * 70)
    if errors:
        print("❌ 검증 실패!")
        print("=" * 70)
        for err in errors:
            print(f"  - {err}")
        return False
    elif warnings:
        print("⚠️  경고 사항이 있지만 기본 검증은 통과했습니다")
        print("=" * 70)
        for warn in warnings:
            print(f"  - {warn}")
        return True
    else:
        print("✅ 모든 검증 통과!")
        print("=" * 70)
        print()
        print("MTP 7B_1T_4 모델이 정상적으로 설정되었습니다.")
        print()
        print("모델 정보:")
        print(f"  - Parameters: ~6.7B")
        print(f"  - rope_theta: {params['rope_theta']}")
        print(f"  - dtype: float16")
        print(f"  - Location: {base_dir}")
        return True

if __name__ == "__main__":
    import sys
    success = verify_mtp_model()
    sys.exit(0 if success else 1)
