#!/usr/bin/env python3
"""
Storage 무결성 검증 스크립트

기능:
- 모든 모델 무결성 검증 (SHA256, Config, SafeTensors)
- 모든 데이터셋 검증 (파일 존재, Schema, 메타데이터)
- problem_index_map 검증 (Pairwise 학습 필수)
- 검증 리포트 생성

Usage:
    # 전체 검증
    uv run python scripts/create_storage/verify_storage.py --check all

    # 모델만 검증
    uv run python scripts/create_storage/verify_storage.py --check models

    # 데이터셋만 검증
    uv run python scripts/create_storage/verify_storage.py --check datasets

    # 리포트 생성
    uv run python scripts/create_storage/verify_storage.py --check all --generate-report
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
from safetensors import safe_open


# ============================================================================
# 검증 대상 정의
# ============================================================================

MODELS_TO_VERIFY = {
    "meta-llama-mtp": {
        "base_dir": "storage/models/meta-llama-mtp",
        "required_files": [
            "safetensors/model.safetensors",
            "safetensors/SHA256SUMS",
            "configs/params.json",
            "configs/meta_adapter.yaml",
        ],
        "config_checks": [
            ("n_future_tokens", 4),
            ("rope_theta", 10000.0),
            ("intermediate_size", 11008),
            ("dtype", "float16"),
        ],
    },
    "ref-sheared-llama-2.7b": {
        "base_dir": "storage/models/ref-sheared-llama-2.7b",
        "required_files": [
            "safetensors/model.safetensors",
            "safetensors/SHA256SUMS",
            "configs/config.json",
        ],
        "config_checks": [
            ("hidden_size", 2560),
            ("num_hidden_layers", 32),
        ],
    },
    "micro-mtp": {
        "base_dir": "storage/models/micro-mtp",
        "required_files": [
            "safetensors/model.safetensors",
            "safetensors/SHA256SUMS",
            "configs/config.json",
        ],
        "config_checks": [
            ("num_hidden_layers", 4),
            ("hidden_size", 512),
        ],
    },
}

# 전체 데이터셋 (processed 디렉토리)
DATASETS_TO_VERIFY = {
    "codecontests": {
        "processed_dir": "storage/datasets/codecontests/processed",
        "splits": ["train", "valid", "test"],
        "required_fields": ["instruction", "input", "output", "task_id", "is_correct"],
        "metadata_required": True,
        "problem_index_map_required": True,  # Pairwise 학습 필수
    },
    "mbpp": {
        "processed_dir": "storage/datasets/mbpp/processed",
        "splits": ["train", "validation", "test"],
        "required_fields": ["instruction", "input", "output", "task_id"],
        "metadata_required": False,
        "problem_index_map_required": False,
    },
    "humaneval": {
        "processed_dir": "storage/datasets/humaneval/processed",
        "splits": ["test"],
        "required_fields": ["instruction", "input", "output", "task_id"],
        "metadata_required": False,
        "problem_index_map_required": False,
    },
    "gsm8k": {
        "processed_dir": "storage/datasets/gsm8k/processed",
        "splits": ["train", "test"],
        "required_fields": ["instruction", "input", "output", "task_id"],
        "metadata_required": False,
        "problem_index_map_required": False,
    },
}

# Small 데이터셋 (로컬 테스트용)
SMALL_DATASETS_TO_VERIFY = {
    "codecontests_small": {
        "small_dir": "storage/datasets_local_small/codecontests_small",
        "files": ["train_small.jsonl", "valid_small.jsonl"],
        "required_fields": ["instruction", "input", "output", "task_id"],
    },
    "mbpp_small": {
        "small_dir": "storage/datasets_local_small/mbpp_small",
        "files": ["train_small.jsonl", "validation_small.jsonl"],
        "required_fields": ["instruction", "input", "output", "task_id"],
    },
    "humaneval_small": {
        "small_dir": "storage/datasets_local_small/humaneval_small",
        "files": ["test_small.jsonl"],
        "required_fields": ["instruction", "input", "output", "task_id"],
    },
}


# ============================================================================
# 유틸리티 함수
# ============================================================================

def calculate_sha256(file_path: Path) -> str:
    """파일의 SHA256 해시 계산"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def format_size(size_bytes: int) -> str:
    """파일 크기를 읽기 쉬운 형식으로 변환"""
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


# ============================================================================
# StorageVerifier 클래스
# ============================================================================

class StorageVerifier:
    def __init__(self):
        self.results = {
            "models": {},
            "datasets": {},
            "small_datasets": {},
            "timestamp": datetime.now().isoformat(),
        }

    def verify_models(self, model_list: List[str]) -> bool:
        """모델 검증"""
        print("=" * 70)
        print("Verifying Models")
        print("=" * 70)
        print()

        all_passed = True

        for model_name in model_list:
            if model_name not in MODELS_TO_VERIFY:
                print(f"  Unknown model: {model_name}, skipping...")
                continue

            print(f"[{model_name}]")
            print("-" * 70)

            config = MODELS_TO_VERIFY[model_name]
            base_dir = Path(config["base_dir"])

            model_passed, model_errors = self._verify_single_model(
                model_name, base_dir, config
            )

            self.results["models"][model_name] = {
                "passed": model_passed,
                "errors": model_errors,
            }

            if model_passed:
                print(f"  {model_name}: All checks passed!")
            else:
                print(f"  {model_name}: Verification failed!")
                for err in model_errors:
                    print(f"    - {err}")
                all_passed = False

            print()

        return all_passed

    def _verify_single_model(
        self, model_name: str, base_dir: Path, config: Dict
    ) -> Tuple[bool, List[str]]:
        """단일 모델 검증"""
        errors = []

        # 1. 파일 존재 확인
        print("  [1/3] Checking required files...")
        for file_rel in config["required_files"]:
            file_path = base_dir / file_rel
            if file_path.exists():
                size = format_size(file_path.stat().st_size)
                print(f"    [OK] {file_rel} ({size})")
            else:
                errors.append(f"Missing: {file_rel}")
                print(f"    [FAIL] {file_rel} NOT FOUND")

        # 2. SHA256 검증
        safetensors_file = base_dir / "safetensors/model.safetensors"
        sha256_file = base_dir / "safetensors/SHA256SUMS"

        if safetensors_file.exists() and sha256_file.exists():
            print("  [2/3] Verifying SHA256...")

            with open(sha256_file, "r") as f:
                expected_hash = f.read().strip().split()[0]

            actual_hash = calculate_sha256(safetensors_file)

            if expected_hash == actual_hash:
                print(f"    [OK] SHA256 match: {actual_hash[:16]}...")
            else:
                errors.append("SHA256 mismatch")
                print(f"    [FAIL] SHA256 mismatch!")
                print(f"      Expected: {expected_hash}")
                print(f"      Actual:   {actual_hash}")
        else:
            print("  [2/3] Skipping SHA256 (files missing)")

        # 3. Config 검증
        if config["config_checks"]:
            print("  [3/3] Verifying config...")

            yaml_path = base_dir / "configs/meta_adapter.yaml"
            json_path = base_dir / "configs/config.json"

            config_data = None

            if yaml_path.exists():
                with open(yaml_path, "r") as f:
                    config_data = yaml.safe_load(f)
            elif json_path.exists():
                with open(json_path, "r") as f:
                    config_data = json.load(f)
            else:
                errors.append("Config file not found")
                print("    [FAIL] Config file not found")
                return (False, errors)

            for key, expected_value in config["config_checks"]:
                actual_value = config_data.get(key)

                if actual_value == expected_value:
                    print(f"    [OK] {key}: {actual_value}")
                else:
                    errors.append(
                        f"Config mismatch: {key} (expected={expected_value}, actual={actual_value})"
                    )
                    print(f"    [FAIL] {key}: expected={expected_value}, actual={actual_value}")
        else:
            print("  [3/3] Skipping config checks")

        return (len(errors) == 0, errors)

    def verify_datasets(self, dataset_list: List[str]) -> bool:
        """전체 데이터셋 검증 (processed 디렉토리)"""
        print("=" * 70)
        print("Verifying Datasets (Full)")
        print("=" * 70)
        print()

        all_passed = True

        for dataset_name in dataset_list:
            if dataset_name not in DATASETS_TO_VERIFY:
                print(f"  Unknown dataset: {dataset_name}, skipping...")
                continue

            print(f"[{dataset_name}]")
            print("-" * 70)

            config = DATASETS_TO_VERIFY[dataset_name]
            processed_dir = Path(config["processed_dir"])

            if not processed_dir.exists():
                print(f"  [SKIP] Directory not found: {processed_dir}")
                self.results["datasets"][dataset_name] = {
                    "passed": False,
                    "errors": ["Directory not found"],
                }
                all_passed = False
                print()
                continue

            dataset_passed, dataset_errors = self._verify_single_dataset(
                dataset_name, processed_dir, config
            )

            self.results["datasets"][dataset_name] = {
                "passed": dataset_passed,
                "errors": dataset_errors,
            }

            if dataset_passed:
                print(f"  {dataset_name}: All checks passed!")
            else:
                print(f"  {dataset_name}: Verification failed!")
                for err in dataset_errors:
                    print(f"    - {err}")
                all_passed = False

            print()

        return all_passed

    def _verify_single_dataset(
        self, dataset_name: str, processed_dir: Path, config: Dict
    ) -> Tuple[bool, List[str]]:
        """단일 데이터셋 검증"""
        errors = []
        splits = config["splits"]
        required_fields = config["required_fields"]
        metadata_required = config.get("metadata_required", False)
        problem_index_map_required = config.get("problem_index_map_required", False)

        # 1. JSONL 파일 존재 확인
        print("  [1/3] Checking JSONL files...")
        for split in splits:
            jsonl_path = processed_dir / f"{split}.jsonl"
            if jsonl_path.exists():
                size = format_size(jsonl_path.stat().st_size)
                # 라인 수 카운트 (대용량 파일은 추정)
                if jsonl_path.stat().st_size > 100 * 1024 * 1024:  # 100MB 이상
                    line_count = "large"
                else:
                    with open(jsonl_path, "r") as f:
                        line_count = sum(1 for _ in f)
                print(f"    [OK] {split}.jsonl ({size}, {line_count} samples)")
            else:
                errors.append(f"Missing: {split}.jsonl")
                print(f"    [FAIL] {split}.jsonl NOT FOUND")

        # 2. Schema 검증 (샘플 3개)
        print("  [2/3] Validating schema...")
        for split in splits:
            jsonl_path = processed_dir / f"{split}.jsonl"
            if not jsonl_path.exists():
                continue

            with open(jsonl_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break

                    try:
                        data = json.loads(line)
                        missing = [field for field in required_fields if field not in data]

                        if missing:
                            errors.append(f"{split}.jsonl: missing fields {missing}")
                            print(f"    [FAIL] {split} line {i+1}: missing {missing}")
                        elif i == 0:
                            print(f"    [OK] {split}: schema valid (checked 3 samples)")

                    except json.JSONDecodeError:
                        errors.append(f"{split}.jsonl line {i+1}: JSON decode error")
                        print(f"    [FAIL] {split} line {i+1}: JSON error")

        # 3. 메타데이터 검증
        print("  [3/3] Validating metadata...")
        for split in splits:
            metadata_path = processed_dir / f"{split}_metadata.json"

            if not metadata_path.exists():
                if metadata_required:
                    errors.append(f"Missing: {split}_metadata.json")
                    print(f"    [FAIL] {split}_metadata.json NOT FOUND")
                else:
                    print(f"    [SKIP] {split}_metadata.json (optional)")
                continue

            size = format_size(metadata_path.stat().st_size)

            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # problem_index_map 검증
                if problem_index_map_required:
                    if "problem_index_map" not in metadata:
                        errors.append(f"{split}_metadata.json: missing problem_index_map")
                        print(f"    [FAIL] {split}_metadata.json: problem_index_map NOT FOUND")
                        print(f"           Run: uv run python scripts/create_storage/setup_datasets.py --datasets {dataset_name} --steps metadata")
                    else:
                        problem_count = len(metadata["problem_index_map"])
                        stats = metadata.get("stats", {})
                        valid_problems = stats.get("n_valid_problems", "N/A")
                        total_pairs = stats.get("total_possible_pairs", "N/A")
                        print(f"    [OK] {split}_metadata.json ({size})")
                        print(f"         problems={problem_count}, valid={valid_problems}, pairs={total_pairs}")
                else:
                    print(f"    [OK] {split}_metadata.json ({size})")

            except json.JSONDecodeError:
                errors.append(f"{split}_metadata.json: JSON decode error")
                print(f"    [FAIL] {split}_metadata.json: invalid JSON")

        return (len(errors) == 0, errors)

    def verify_small_datasets(self) -> bool:
        """Small 데이터셋 검증 (로컬 테스트용)"""
        print("=" * 70)
        print("Verifying Small Datasets")
        print("=" * 70)
        print()

        all_passed = True

        for dataset_name, config in SMALL_DATASETS_TO_VERIFY.items():
            print(f"[{dataset_name}]")
            print("-" * 70)

            small_dir = Path(config["small_dir"])

            if not small_dir.exists():
                print(f"  [SKIP] Directory not found: {small_dir}")
                self.results["small_datasets"][dataset_name] = {
                    "passed": False,
                    "errors": ["Directory not found"],
                }
                print()
                continue

            errors = []

            # 파일 존재 확인
            print("  Checking files...")
            for file_name in config["files"]:
                file_path = small_dir / file_name
                if file_path.exists():
                    size = format_size(file_path.stat().st_size)
                    print(f"    [OK] {file_name} ({size})")
                else:
                    errors.append(f"Missing: {file_name}")
                    print(f"    [FAIL] {file_name} NOT FOUND")

            passed = len(errors) == 0
            self.results["small_datasets"][dataset_name] = {
                "passed": passed,
                "errors": errors,
            }

            if passed:
                print(f"  {dataset_name}: All checks passed!")
            else:
                print(f"  {dataset_name}: Verification failed!")
                all_passed = False

            print()

        return all_passed

    def generate_report(self) -> None:
        """검증 리포트 생성"""
        print("=" * 70)
        print("Generating Verification Report")
        print("=" * 70)
        print()

        report_dir = Path("storage/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"verification_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"  Report saved to {report_file}")
        print()

        # 요약
        total_models = len(self.results["models"])
        passed_models = sum(1 for m in self.results["models"].values() if m.get("passed"))

        total_datasets = len(self.results["datasets"])
        passed_datasets = sum(1 for d in self.results["datasets"].values() if d.get("passed"))

        total_small = len(self.results["small_datasets"])
        passed_small = sum(1 for d in self.results["small_datasets"].values() if d.get("passed"))

        print("Summary:")
        print(f"  Models:         {passed_models}/{total_models} passed")
        print(f"  Datasets:       {passed_datasets}/{total_datasets} passed")
        print(f"  Small Datasets: {passed_small}/{total_small} passed")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Storage 무결성 검증 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 전체 검증
  uv run python scripts/create_storage/verify_storage.py --check all

  # 모델만 검증
  uv run python scripts/create_storage/verify_storage.py --check models

  # 데이터셋만 검증
  uv run python scripts/create_storage/verify_storage.py --check datasets

  # 리포트 생성
  uv run python scripts/create_storage/verify_storage.py --check all --generate-report
        """,
    )

    parser.add_argument(
        "--check",
        required=True,
        choices=["models", "datasets", "all"],
        help="What to verify",
    )

    parser.add_argument(
        "--include-small",
        action="store_true",
        help="Also verify small datasets",
    )

    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate verification report (JSON)",
    )

    args = parser.parse_args()

    verifier = StorageVerifier()

    all_passed = True

    try:
        if args.check in ["models", "all"]:
            models_passed = verifier.verify_models(list(MODELS_TO_VERIFY.keys()))
            all_passed = all_passed and models_passed

        if args.check in ["datasets", "all"]:
            datasets_passed = verifier.verify_datasets(list(DATASETS_TO_VERIFY.keys()))
            all_passed = all_passed and datasets_passed

            if args.include_small:
                small_passed = verifier.verify_small_datasets()
                all_passed = all_passed and small_passed

        if args.generate_report:
            verifier.generate_report()

        # 최종 결과
        print("=" * 70)
        if all_passed:
            print("All verifications PASSED")
        else:
            print("Some verifications FAILED")
        print("=" * 70)

        sys.exit(0 if all_passed else 1)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
