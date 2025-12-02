#!/usr/bin/env python3
"""CodeContests 테스트 케이스 추출 스크립트

HuggingFace deepmind/code_contests에서 테스트 케이스를 추출하여
평가용 JSON 파일로 저장합니다.

테스트 종류:
- public_tests: 공개 테스트 (평가에 사용)
- private_tests: 비공개 테스트 (대회 채점용)
- generated_tests: 자동 생성 테스트

Usage:
    uv run python scripts/create_storage/extract_codecontests_tests.py
    uv run python scripts/create_storage/extract_codecontests_tests.py --split test
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def extract_codecontests_tests(
    output_dir: str = "storage/datasets/codecontests/tests",
    target_split: str | None = None,
) -> None:
    """CodeContests 테스트 케이스 추출 (split별 로드)

    Args:
        output_dir: 출력 디렉토리 경로
        target_split: 특정 split만 처리 (None이면 전체)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CodeContests 테스트 케이스 추출")
    print("=" * 70)
    print()

    splits = [target_split] if target_split else ["train", "valid", "test"]

    for split in splits:
        print(f"Processing {split} split...")
        print(f"  Loading deepmind/code_contests ({split})...")

        # split별로 로드하여 메모리 절약
        try:
            split_data = load_dataset("deepmind/code_contests", split=split)
        except Exception as e:
            print(f"  Failed to load {split}: {e}")
            continue

        tests_data = {}
        stats = {
            "total_problems": 0,
            "problems_with_public": 0,
            "problems_with_private": 0,
            "problems_with_generated": 0,
            "total_public_tests": 0,
            "total_private_tests": 0,
            "total_generated_tests": 0,
        }

        for idx, example in enumerate(split_data):
            if idx % 100 == 0:
                print(f"  Processing: {idx}/{len(split_data)}", end="\r")

            task_name = example.get("name", "")
            if not task_name:
                continue

            # 테스트 케이스 추출
            public_tests = example.get("public_tests", {})
            private_tests = example.get("private_tests", {})
            generated_tests = example.get("generated_tests", {})

            # input/output 리스트 추출
            public_inputs = list(public_tests.get("input", []))
            public_outputs = list(public_tests.get("output", []))
            private_inputs = list(private_tests.get("input", []))
            private_outputs = list(private_tests.get("output", []))
            generated_inputs = list(generated_tests.get("input", []))
            generated_outputs = list(generated_tests.get("output", []))

            # 유효한 테스트만 저장
            tests_entry = {
                "task_name": task_name,
                "public_tests": {
                    "input": public_inputs,
                    "output": public_outputs,
                },
                "private_tests": {
                    "input": private_inputs,
                    "output": private_outputs,
                },
                "generated_tests": {
                    "input": generated_inputs,
                    "output": generated_outputs,
                },
            }

            tests_data[task_name] = tests_entry

            # 통계 업데이트
            stats["total_problems"] += 1
            if len(public_inputs) > 0:
                stats["problems_with_public"] += 1
                stats["total_public_tests"] += len(public_inputs)
            if len(private_inputs) > 0:
                stats["problems_with_private"] += 1
                stats["total_private_tests"] += len(private_inputs)
            if len(generated_inputs) > 0:
                stats["problems_with_generated"] += 1
                stats["total_generated_tests"] += len(generated_inputs)

        # JSON 저장
        output_file = output_path / f"{split}_tests.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(tests_data, f, ensure_ascii=False)

        print()
        print(f"  Saved: {output_file}")
        print(f"  Problems: {stats['total_problems']}")
        print(f"  With public tests: {stats['problems_with_public']} ({stats['total_public_tests']} tests)")
        print(f"  With private tests: {stats['problems_with_private']} ({stats['total_private_tests']} tests)")
        print(f"  With generated tests: {stats['problems_with_generated']} ({stats['total_generated_tests']} tests)")
        print()

        # 메모리 해제
        del split_data

    print("=" * 70)
    print("Extraction completed!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CodeContests 테스트 케이스 추출")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "valid", "test"],
        help="특정 split만 처리 (기본: 전체)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="storage/datasets/codecontests/tests",
        help="출력 디렉토리",
    )
    args = parser.parse_args()

    extract_codecontests_tests(
        output_dir=args.output_dir,
        target_split=args.split,
    )


if __name__ == "__main__":
    main()
