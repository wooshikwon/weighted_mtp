#!/usr/bin/env python3
"""
MTP PyTorch 모델을 SafeTensors 형식으로 변환

Input:  storage/models_v2/meta-llama-mtp/raw/7B_1T_4/consolidated.pth
Output: storage/models_v2/meta-llama-mtp/safetensors/model.safetensors
"""

import torch
from safetensors.torch import save_file
from pathlib import Path
import hashlib
import json

def calculate_sha256(file_path):
    """파일의 SHA256 해시 계산"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def convert_mtp_to_safetensors():
    # 경로 설정
    base_dir = Path("storage/models_v2/meta-llama-mtp")
    input_file = base_dir / "raw/7B_1T_4/consolidated.pth"
    output_dir = base_dir / "safetensors"
    output_file = output_dir / "model.safetensors"
    sha256_file = output_dir / "SHA256SUMS"

    # 디렉터리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MTP PyTorch → SafeTensors 변환")
    print("=" * 70)
    print()
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()

    # 입력 파일 확인
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    input_size_gb = input_file.stat().st_size / (1024**3)
    print(f"Input file size: {input_size_gb:.2f} GB")
    print()

    # PyTorch 모델 로드
    print("Loading PyTorch checkpoint...")
    state_dict = torch.load(input_file, map_location='cpu', weights_only=True)
    print(f"✓ Loaded {len(state_dict)} tensors")
    print()

    # Tensor 정보 출력
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print()

    # SafeTensors로 저장
    print("Converting to SafeTensors format...")
    save_file(state_dict, str(output_file))

    output_size_gb = output_file.stat().st_size / (1024**3)
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Output file size: {output_size_gb:.2f} GB")
    print()

    # SHA256 해시 계산
    print("Calculating SHA256 hash...")
    sha256_hash = calculate_sha256(output_file)

    # SHA256SUMS 파일 생성
    with open(sha256_file, 'w') as f:
        f.write(f"{sha256_hash}  model.safetensors\n")

    print(f"✓ SHA256: {sha256_hash}")
    print(f"✓ Saved to: {sha256_file}")
    print()

    # 검증
    print("Verifying conversion...")
    from safetensors import safe_open

    with safe_open(output_file, framework="pt", device="cpu") as f:
        safetensors_keys = set(f.keys())

    pytorch_keys = set(state_dict.keys())

    if safetensors_keys == pytorch_keys:
        print(f"✓ All {len(safetensors_keys)} tensors verified")
    else:
        print("❌ Key mismatch!")
        print(f"  Missing in SafeTensors: {pytorch_keys - safetensors_keys}")
        print(f"  Extra in SafeTensors: {safetensors_keys - pytorch_keys}")
        raise ValueError("Conversion verification failed")

    print()
    print("=" * 70)
    print("✅ Conversion completed successfully!")
    print("=" * 70)
    print()

    # metadata.json 업데이트
    metadata_file = base_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        metadata['sha256']['checkpoint'] = sha256_hash
        metadata['files']['checkpoint'] = 'safetensors/model.safetensors'
        metadata['notes'] = f"Converted from facebook/multi-token-prediction 7B_1T_4. Original params: rope_theta=10000.0, dtype=float16"

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Updated metadata.json with SHA256 hash")
        print()

if __name__ == "__main__":
    convert_mtp_to_safetensors()
