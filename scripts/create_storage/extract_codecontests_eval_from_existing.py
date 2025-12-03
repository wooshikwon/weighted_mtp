#!/usr/bin/env python3
"""기존 test.jsonl에서 CodeContests 평가용 데이터 추출

HuggingFace 다운로드 없이 기존 데이터에서 고유 문제만 추출합니다.

Usage:
    uv run python scripts/create_storage/extract_codecontests_eval_from_existing.py
"""

import json
import re
from pathlib import Path


def main():
    input_path = Path("storage/datasets/codecontests/processed/test.jsonl")
    output_path = Path("storage/datasets/codecontests/processed/test_eval.jsonl")

    if not input_path.exists():
        print(f"입력 파일이 없습니다: {input_path}")
        return

    seen_problems = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            task_id = sample["task_id"]

            match = re.match(r"^(.+?)_(correct|incorrect)_\d+$", task_id)
            problem_name = match.group(1) if match else task_id

            if problem_name not in seen_problems:
                seen_problems[problem_name] = {
                    "task_id": problem_name,
                    "instruction": sample["instruction"],
                    "metadata": sample.get("metadata", {}),
                }

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in seen_problems.values():
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"추출 완료: {len(seen_problems)}개 문제")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    main()
