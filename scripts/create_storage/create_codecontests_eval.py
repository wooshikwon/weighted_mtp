#!/usr/bin/env python3
"""CodeContests 평가용 데이터셋 생성 스크립트

HuggingFace deepmind/code_contests test split에서 165개 문제를 추출하여
평가용 JSONL 파일로 저장합니다.

기존 processed/test.jsonl은 학습용 (problem+solution 쌍, 14,851개)이며,
이 스크립트는 평가용 (problem description only, 165개)을 생성합니다.

출력 형식:
{
    "task_id": "problem_name",
    "instruction": "문제 설명 (description)",
    "metadata": {
        "source": "codeforces",
        "difficulty": 1500,
        "public_tests": {"input": [...], "output": [...]},
        "private_tests": {"input": [...], "output": [...]},
        "generated_tests": {"input": [...], "output": [...]}
    }
}

Usage:
    uv run python scripts/create_storage/create_codecontests_eval.py
    uv run python scripts/create_storage/create_codecontests_eval.py --split valid
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def create_codecontests_eval(
    output_dir: str = "storage/datasets/codecontests/processed",
    target_split: str = "test",
) -> None:
    """CodeContests 평가용 데이터셋 생성

    Args:
        output_dir: 출력 디렉토리 경로
        target_split: 대상 split (test 또는 valid)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CodeContests 평가용 데이터셋 생성")
    print("=" * 70)
    print()

    print(f"Loading deepmind/code_contests ({target_split})...")
    split_data = load_dataset("deepmind/code_contests", split=target_split)
    print(f"Loaded {len(split_data)} problems")
    print()

    # 평가용 데이터 생성
    eval_samples = []
    stats = {
        "total": 0,
        "with_public_tests": 0,
        "with_private_tests": 0,
        "sources": {},
        "difficulties": [],
    }

    for idx, example in enumerate(split_data):
        if idx % 50 == 0:
            print(f"Processing: {idx}/{len(split_data)}", end="\r")

        task_name = example.get("name", "")
        description = example.get("description", "")

        if not task_name or not description:
            continue

        # 테스트 케이스 추출
        public_tests = example.get("public_tests", {})
        private_tests = example.get("private_tests", {})
        generated_tests = example.get("generated_tests", {})

        public_inputs = list(public_tests.get("input", []))
        public_outputs = list(public_tests.get("output", []))
        private_inputs = list(private_tests.get("input", []))
        private_outputs = list(private_tests.get("output", []))
        generated_inputs = list(generated_tests.get("input", []))
        generated_outputs = list(generated_tests.get("output", []))

        # 메타데이터 추출
        source = example.get("source", 0)
        difficulty = example.get("difficulty", 0)
        cf_rating = example.get("cf_rating", 0)
        time_limit = example.get("time_limit", {})
        memory_limit = example.get("memory_limit_bytes", 0)

        # 평가용 샘플 생성
        eval_sample = {
            "task_id": task_name,
            "instruction": description,
            "metadata": {
                "source": source,
                "difficulty": difficulty,
                "cf_rating": cf_rating,
                "time_limit": time_limit,
                "memory_limit_bytes": memory_limit,
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
            },
        }
        eval_samples.append(eval_sample)

        # 통계 업데이트
        stats["total"] += 1
        if len(public_inputs) > 0:
            stats["with_public_tests"] += 1
        if len(private_inputs) > 0:
            stats["with_private_tests"] += 1

        source_str = str(source)
        stats["sources"][source_str] = stats["sources"].get(source_str, 0) + 1

        if cf_rating > 0:
            stats["difficulties"].append(cf_rating)

    print()
    print()

    # JSONL 저장
    output_file = output_path / f"{target_split}_eval.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved: {output_file}")
    print()
    print("=== Statistics ===")
    print(f"Total problems: {stats['total']}")
    print(f"With public tests: {stats['with_public_tests']}")
    print(f"With private tests: {stats['with_private_tests']}")
    print()
    print("Source distribution:")
    for source, count in sorted(stats["sources"].items()):
        print(f"  {source}: {count}")

    if stats["difficulties"]:
        avg_diff = sum(stats["difficulties"]) / len(stats["difficulties"])
        min_diff = min(stats["difficulties"])
        max_diff = max(stats["difficulties"])
        print()
        print(f"CF Rating: avg={avg_diff:.0f}, min={min_diff}, max={max_diff}")

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CodeContests 평가용 데이터셋 생성")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["valid", "test"],
        help="대상 split (기본: test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="storage/datasets/codecontests/processed",
        help="출력 디렉토리",
    )
    args = parser.parse_args()

    create_codecontests_eval(
        output_dir=args.output_dir,
        target_split=args.split,
    )


if __name__ == "__main__":
    main()
