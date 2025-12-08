#!/usr/bin/env python3
"""
Phase1 데이터셋 설정 통합 스크립트

HuggingFace datasets 라이브러리를 사용하여:
1. 데이터셋 다운로드 및 Alpaca 형식으로 변환 (process)
2. 메타데이터 추출 (metadata)
3. Small 버전 생성 (small)
4. 통계 생성 (stats)
5. 평가용 데이터셋 생성 - codecontests 전용 (eval)
6. 테스트 케이스 추출 - codecontests 전용 (tests)
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import sentencepiece as spm
from datasets import load_dataset


# ============================================================================
# 데이터셋 설정 정의
# ============================================================================

DATASET_CONFIGS = {
    "codecontests": {
        "repo": "deepmind/code_contests",
        "base_dir": "storage/datasets/codecontests",
        "splits": ["train", "valid", "test"],
        "small_sizes": {"train": 100, "valid": 32, "test": 32},
    },
    "mbpp": {
        "repo": "google-research-datasets/mbpp",
        "config": "full",  # "full" or "sanitized"
        "base_dir": "storage/datasets/mbpp",
        "splits": ["train", "validation", "test"],
        "small_sizes": {"train": 100, "validation": 32, "test": 32},
    },
    "humaneval": {
        "repo": "openai/openai_humaneval",
        "base_dir": "storage/datasets/humaneval",
        "splits": ["test"],  # HumanEval only has test split
        "small_sizes": {"test": 32},
    },
    "gsm8k": {
        "repo": "openai/gsm8k",
        "config": "main",
        "base_dir": "storage/datasets/gsm8k",
        "splits": ["train", "test"],
        "small_sizes": {"train": 100, "test": 32},
    },
}


# ============================================================================
# DatasetSetup 클래스
# ============================================================================

class DatasetSetup:
    def __init__(self, dataset_name: str, max_tokens: int = 2048):
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.base_dir = Path(self.config["base_dir"])
        self.max_tokens = max_tokens

        # Tokenizer 로딩
        tokenizer_path = Path("storage/models/meta-llama-mtp/tokenizer/tokenizer.model")
        if tokenizer_path.exists():
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(tokenizer_path))
            print(f"✓ Loaded tokenizer: {tokenizer_path.name} (vocab_size={self.tokenizer.vocab_size()})")
        else:
            self.tokenizer = None
            print(f"⚠️  Tokenizer not found at {tokenizer_path}, skipping token-based filtering")
        print()

    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer.encode(text))

    def _filter_by_token_length(self, samples: List[Dict]) -> List[Dict]:
        """토큰 길이 기준으로 샘플 필터링"""
        if self.tokenizer is None:
            return samples

        filtered_samples = []

        for sample in samples:
            inst_tokens = self._count_tokens(sample["instruction"])
            input_tokens = self._count_tokens(sample["input"])
            output_tokens = self._count_tokens(sample["output"])
            total_tokens = inst_tokens + input_tokens + output_tokens

            if total_tokens <= self.max_tokens:
                # 토큰 정보를 샘플에 저장 (stats에서 재사용)
                sample["_token_counts"] = {
                    "instruction": inst_tokens,
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens,
                }
                filtered_samples.append(sample)

        filtered_count = len(samples) - len(filtered_samples)
        if filtered_count > 0:
            print(f"  ⚠️  Filtered {filtered_count} samples (>{self.max_tokens} tokens)")
            print(f"  ✓ Remaining: {len(filtered_samples)} samples")

        return filtered_samples

    def download_and_process(self) -> None:
        """HuggingFace에서 데이터셋을 다운로드하고 Alpaca 형식으로 변환"""
        print("=" * 70)
        print(f"Downloading and processing {self.dataset_name}")
        print("=" * 70)
        print()

        # 디렉터리 생성
        processed_dir = self.base_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # HuggingFace에서 데이터셋 로드
        repo = self.config["repo"]
        config_name = self.config.get("config", None)

        print(f"Loading from HuggingFace: {repo}")
        if config_name:
            print(f"  Config: {config_name}")
            dataset = load_dataset(repo, config_name)
        else:
            dataset = load_dataset(repo)

        print(f"✓ Loaded dataset")
        print(f"  Splits: {list(dataset.keys())}")
        print()

        # 각 split을 Alpaca 형식으로 변환
        for split in self.config["splits"]:
            if split not in dataset:
                print(f"⚠️  Split '{split}' not found, skipping")
                continue

            print(f"Processing {split} split...")
            split_data = dataset[split]

            # 변환 함수 호출
            if self.dataset_name == "codecontests":
                alpaca_samples = self._convert_codecontests(split_data)
            elif self.dataset_name == "mbpp":
                alpaca_samples = self._convert_mbpp(split_data)
            elif self.dataset_name == "humaneval":
                alpaca_samples = self._convert_humaneval(split_data)
            elif self.dataset_name == "gsm8k":
                alpaca_samples = self._convert_gsm8k(split_data)
            else:
                raise NotImplementedError(f"No converter for {self.dataset_name}")

            print(f"  Before filtering: {len(alpaca_samples)} samples")

            # 토큰 길이 필터링
            alpaca_samples = self._filter_by_token_length(alpaca_samples)

            # JSONL 저장
            output_file = processed_dir / f"{split}.jsonl"
            with open(output_file, "w") as f:
                for sample in alpaca_samples:
                    # _token_counts는 내부 캐시용이므로 저장하지 않음
                    sample_to_save = {k: v for k, v in sample.items() if k != "_token_counts"}
                    f.write(json.dumps(sample_to_save) + "\n")

            print(f"✓ Saved {split}.jsonl ({len(alpaca_samples)} samples)")

        # Schema 생성
        self._generate_schema(processed_dir)

        print()
        print("✅ Processing completed!")
        print()

    def _convert_codecontests(self, split_data) -> List[Dict]:
        """CodeContests를 Alpaca 형식으로 변환 (correct + incorrect 모두 포함)"""
        alpaca_samples = []

        for idx, example in enumerate(split_data):
            # CodeContests 필드: name, description, public_tests, solutions, incorrect_solutions
            instruction = example.get("description", "")
            task_name = example.get("name", f"task_{idx}")

            # input 필드는 빈 문자열로 설정
            # 문제 설명(instruction)에 이미 Example I/O가 포함되어 있음
            # 중복 제거 + 정답 유출 방지
            input_text = ""

            # 테스트 케이스 존재 여부 확인 (metadata용)
            public_tests = example.get("public_tests", {})
            input_list = public_tests.get("input", [])

            # 1. Correct solutions 처리
            solutions = example.get("solutions", {})
            language_list = solutions.get("language", [])
            solution_list = solutions.get("solution", [])

            for sol_idx, (lang, sol) in enumerate(zip(language_list, solution_list)):
                if lang in [1, 3]:  # PYTHON or PYTHON3
                    alpaca_sample = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": sol,
                        "task_id": f"{task_name}_correct_{sol_idx}",
                        "is_correct": True,
                        "metadata": {
                            "source": "code_contests",
                            "difficulty": example.get("difficulty", -1),
                            "has_tests": len(input_list) > 0,
                        }
                    }
                    alpaca_samples.append(alpaca_sample)

            # 2. Incorrect solutions 처리
            incorrect_solutions = example.get("incorrect_solutions", {})
            incorrect_language_list = incorrect_solutions.get("language", [])
            incorrect_solution_list = incorrect_solutions.get("solution", [])

            for sol_idx, (lang, sol) in enumerate(zip(incorrect_language_list, incorrect_solution_list)):
                if lang in [1, 3]:  # PYTHON or PYTHON3
                    alpaca_sample = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": sol,
                        "task_id": f"{task_name}_incorrect_{sol_idx}",
                        "is_correct": False,
                        "metadata": {
                            "source": "code_contests",
                            "difficulty": example.get("difficulty", -1),
                            "has_tests": len(input_list) > 0,
                        }
                    }
                    alpaca_samples.append(alpaca_sample)

        return alpaca_samples

    def _convert_mbpp(self, split_data) -> List[Dict]:
        """MBPP를 Alpaca 형식으로 변환"""
        alpaca_samples = []

        for example in split_data:
            # MBPP 필드: task_id, text, code, test_list, test_setup_code, challenge_test_list
            test_list = example.get("test_list", [])
            if isinstance(test_list, np.ndarray):
                test_list = test_list.tolist()

            # test_list에서 함수명 추출 (예: "assert remove_Occ(...)" → "remove_Occ")
            instruction = example.get("text", "")
            if test_list:
                func_match = re.search(r'assert\s+(\w+)\s*\(', test_list[0])
                if func_match:
                    func_name = func_match.group(1)
                    instruction = f"{instruction}\n\nThe function should be named '{func_name}'."

            alpaca_sample = {
                "instruction": instruction,
                "input": "",
                "output": example.get("code", ""),
                "task_id": str(example.get("task_id", "")),
                "metadata": {
                    "source": "mbpp",
                    "test_list": test_list,
                    "test_setup_code": example.get("test_setup_code", ""),
                    "has_tests": len(test_list) > 0,
                }
            }
            alpaca_samples.append(alpaca_sample)

        return alpaca_samples

    def _convert_humaneval(self, split_data) -> List[Dict]:
        """HumanEval을 Alpaca 형식으로 변환"""
        alpaca_samples = []

        for example in split_data:
            # HumanEval 필드: task_id, prompt, canonical_solution, test, entry_point
            alpaca_sample = {
                "instruction": example.get("prompt", ""),
                "input": "",
                "output": example.get("canonical_solution", ""),
                "task_id": example.get("task_id", ""),
                "metadata": {
                    "source": "humaneval",
                    "test": example.get("test", ""),
                    "entry_point": example.get("entry_point", ""),
                    "has_tests": bool(example.get("test", "")),
                }
            }
            alpaca_samples.append(alpaca_sample)

        return alpaca_samples

    def _convert_gsm8k(self, split_data) -> List[Dict]:
        """GSM8K를 Alpaca 형식으로 변환"""
        alpaca_samples = []

        for idx, example in enumerate(split_data):
            # GSM8K 필드: question, answer
            # answer 형식: "Step1...\nStep2...\n#### 42" (####가 최종 답)
            question = example.get("question", "")
            answer = example.get("answer", "")

            alpaca_sample = {
                "instruction": question,
                "input": "",
                "output": answer,
                "task_id": f"gsm8k_{idx}",
                "metadata": {
                    "source": "gsm8k",
                }
            }
            alpaca_samples.append(alpaca_sample)

        return alpaca_samples

    def _generate_schema(self, processed_dir: Path) -> None:
        """스키마 생성"""
        # CodeContests는 is_correct 필드 포함
        required_fields = ["instruction", "input", "output", "task_id"]
        if self.dataset_name == "codecontests":
            required_fields.append("is_correct")

        schema = {
            "dataset": self.dataset_name,
            "format": "alpaca",
            "required_fields": required_fields,
            "optional_fields": ["metadata"],
            "description": f"{self.dataset_name} dataset in Alpaca format",
            "source": self.config["repo"],
            "notes": "CodeContests includes is_correct field for correct/incorrect solution classification" if self.dataset_name == "codecontests" else None
        }

        schema_file = processed_dir / "schema.json"
        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=2)

        print(f"✓ Generated schema.json")

    def create_small(self) -> None:
        """Small 버전 생성"""
        print("=" * 70)
        print(f"Creating small version of {self.dataset_name}")
        print("=" * 70)
        print()

        processed_dir = self.base_dir / "processed"
        small_dir = Path(f"storage/datasets_local_small/{self.dataset_name}_small")
        small_dir.mkdir(parents=True, exist_ok=True)

        small_sizes = self.config["small_sizes"]

        for split in self.config["splits"]:
            input_file = processed_dir / f"{split}.jsonl"
            output_file = small_dir / f"{split}_small.jsonl"

            if not input_file.exists():
                print(f"⚠️  {input_file} not found, skipping {split}")
                continue

            size = small_sizes.get(split, 32)
            count = 0

            with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
                for line in f_in:
                    if count >= size:
                        break
                    f_out.write(line)
                    count += 1

            print(f"✓ Created {split}_small.jsonl ({count} samples)")

        print()
        print("✅ Small version created!")
        print()

    def _extract_problem_id(self, task_id: str) -> str:
        """task_id에서 problem_id 추출

        task_id 형식: "{problem_id}_correct_{N}" 또는 "{problem_id}_incorrect_{N}"
        """
        if not task_id:
            return task_id

        match = re.match(r"^(.+?)_(correct|incorrect)_\d+$", task_id)
        if match:
            return match.group(1)

        return task_id

    def extract_metadata(self) -> None:
        """메타데이터 추출 (problem_index_map 포함)

        학습 파이프라인에서 사용하는 problem_index_map 생성:
        - correct_indices: 정답 샘플 인덱스 목록
        - incorrect_indices: 오답 샘플 인덱스 목록
        - correct_token_lengths: 정답 샘플 토큰 길이
        - incorrect_token_lengths: 오답 샘플 토큰 길이
        """
        print("=" * 70)
        print(f"Extracting metadata for {self.dataset_name}")
        print("=" * 70)
        print()

        processed_dir = self.base_dir / "processed"

        for split in self.config["splits"]:
            file = processed_dir / f"{split}.jsonl"

            if not file.exists():
                print(f"⚠️  {file} not found, skipping {split}")
                continue

            print(f"Processing {split}...")

            metadata_list = []
            problem_index_map = {}
            stats = {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "difficulty_dist": {},
                "has_is_correct": False,
                "has_difficulty": False,
                "has_problem_id": False,
            }

            with open(file, "r") as f:
                for idx, line in enumerate(f):
                    if idx % 100000 == 0 and idx > 0:
                        print(f"  진행: {idx:,} 샘플")

                    item = json.loads(line.strip())

                    meta = {}
                    is_correct = None
                    difficulty = None
                    problem_id = None

                    # is_correct 필드
                    if "is_correct" in item:
                        is_correct = item["is_correct"]
                        meta["is_correct"] = is_correct
                        stats["has_is_correct"] = True
                        if is_correct:
                            stats["correct"] += 1
                        else:
                            stats["incorrect"] += 1

                    # difficulty 필드
                    if "metadata" in item and "difficulty" in item["metadata"]:
                        difficulty = item["metadata"]["difficulty"]
                        meta["difficulty"] = difficulty
                        stats["has_difficulty"] = True

                        diff_str = str(difficulty)
                        stats["difficulty_dist"][diff_str] = stats["difficulty_dist"].get(diff_str, 0) + 1

                    # problem_id 추출 및 problem_index_map 구성
                    task_id = item.get("task_id")
                    if task_id:
                        problem_id = self._extract_problem_id(task_id)
                        if problem_id:
                            meta["problem_id"] = problem_id
                            stats["has_problem_id"] = True

                            if problem_id not in problem_index_map:
                                problem_index_map[problem_id] = {
                                    "difficulty": difficulty,
                                    "correct_indices": [],
                                    "incorrect_indices": [],
                                    "correct_token_lengths": [],
                                    "incorrect_token_lengths": [],
                                }

                            # output 토큰 길이 계산
                            output_text = item.get("output", "")
                            token_length = self._count_tokens(output_text) if self.tokenizer else 0

                            if is_correct is True:
                                problem_index_map[problem_id]["correct_indices"].append(idx)
                                problem_index_map[problem_id]["correct_token_lengths"].append(token_length)
                            elif is_correct is False:
                                problem_index_map[problem_id]["incorrect_indices"].append(idx)
                                problem_index_map[problem_id]["incorrect_token_lengths"].append(token_length)

                    metadata_list.append(meta)
                    stats["total"] += 1

            # problem_index_map 통계
            if stats["has_problem_id"]:
                n_problems = len(problem_index_map)
                n_valid_problems = sum(
                    1 for p in problem_index_map.values()
                    if len(p["correct_indices"]) > 0 and len(p["incorrect_indices"]) > 0
                )
                total_possible_pairs = sum(
                    len(p["correct_indices"]) * len(p["incorrect_indices"])
                    for p in problem_index_map.values()
                )
                stats["n_problems"] = n_problems
                stats["n_valid_problems"] = n_valid_problems
                stats["total_possible_pairs"] = total_possible_pairs

            # 메타데이터 저장
            output_data = {
                "metadata": metadata_list,
                "problem_index_map": problem_index_map,
                "stats": stats,
                "source_file": str(file),
            }

            output_file = processed_dir / f"{split}_metadata.json"
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"✓ {split}_metadata.json ({stats['total']:,} samples)")
            if stats["has_is_correct"]:
                print(f"  correct={stats['correct']:,}, incorrect={stats['incorrect']:,}")
            if stats["has_problem_id"]:
                print(f"  problems={stats['n_problems']:,}, valid={stats['n_valid_problems']:,}, pairs={stats['total_possible_pairs']:,}")

        print()
        print("✅ Metadata extraction completed!")
        print()

    def generate_stats(self) -> None:
        """통계 생성 (토큰 길이 포함)"""
        print("=" * 70)
        print(f"Generating stats for {self.dataset_name}")
        print("=" * 70)
        print()

        processed_dir = self.base_dir / "processed"
        stats_dir = self.base_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "dataset": self.dataset_name,
            "max_tokens": self.max_tokens,
            "tokenizer": "meta-llama-mtp" if self.tokenizer else None,
            "splits": {},
        }

        for split in self.config["splits"]:
            file = processed_dir / f"{split}.jsonl"

            if not file.exists():
                print(f"⚠️  {file} not found, skipping {split}")
                continue

            samples = []
            with open(file, "r") as f:
                for line in f:
                    samples.append(json.loads(line))

            # 문자 길이 통계
            inst_lengths = [len(s["instruction"]) for s in samples]
            out_lengths = [len(s["output"]) for s in samples]

            # is_correct 분류 (CodeContests만)
            correct_count = sum(1 for s in samples if s.get("is_correct", True))
            incorrect_count = sum(1 for s in samples if not s.get("is_correct", True))

            split_stats = {
                "count": len(samples),
                "correct_count": correct_count,
                "incorrect_count": incorrect_count,
                "correct_ratio": round(correct_count / len(samples), 3) if samples else 0.0,
                "avg_instruction_length": int(np.mean(inst_lengths)) if inst_lengths else 0,
                "avg_output_length": int(np.mean(out_lengths)) if out_lengths else 0,
                "max_instruction_length": int(np.max(inst_lengths)) if inst_lengths else 0,
                "max_output_length": int(np.max(out_lengths)) if out_lengths else 0,
            }

            # 토큰 길이 통계 (tokenizer 사용 가능한 경우)
            if self.tokenizer:
                # process 단계에서 저장한 토큰 정보 재사용 (중복 계산 방지)
                if samples and "_token_counts" in samples[0]:
                    # 이미 계산된 토큰 정보 사용
                    inst_tokens = [s["_token_counts"]["instruction"] for s in samples]
                    out_tokens = [s["_token_counts"]["output"] for s in samples]
                    total_tokens = [s["_token_counts"]["total"] for s in samples]
                else:
                    # 토큰 정보가 없는 경우만 계산
                    inst_tokens = [self._count_tokens(s["instruction"]) for s in samples]
                    out_tokens = [self._count_tokens(s["output"]) for s in samples]
                    total_tokens = [
                        self._count_tokens(s["instruction"]) +
                        self._count_tokens(s["input"]) +
                        self._count_tokens(s["output"])
                        for s in samples
                    ]

                split_stats.update({
                    "avg_instruction_tokens": int(np.mean(inst_tokens)) if inst_tokens else 0,
                    "avg_output_tokens": int(np.mean(out_tokens)) if out_tokens else 0,
                    "avg_total_tokens": int(np.mean(total_tokens)) if total_tokens else 0,
                    "max_instruction_tokens": int(np.max(inst_tokens)) if inst_tokens else 0,
                    "max_output_tokens": int(np.max(out_tokens)) if out_tokens else 0,
                    "max_total_tokens": int(np.max(total_tokens)) if total_tokens else 0,
                })

            stats["splits"][split] = split_stats
            print(f"✓ {split}: {len(samples)} samples")

        # 저장
        timestamp = datetime.now().strftime("%Y-%m-%d")
        stats_file = stats_dir / f"{timestamp}_summary.json"

        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print()
        print(f"✓ Stats saved to {stats_file}")
        print()

        print("✅ Stats generated!")
        print()

    def create_eval(self) -> None:
        """평가용 데이터셋 생성 (codecontests 전용)

        학습용 데이터와 달리 솔루션 없이 문제만 포함.
        테스트 케이스는 metadata에 포함하여 평가 시 사용.
        출력: processed/{split}_eval.jsonl
        """
        if self.dataset_name != "codecontests":
            print(f"⚠️  eval step is only for codecontests, skipping {self.dataset_name}")
            return

        print("=" * 70)
        print(f"Creating eval dataset for {self.dataset_name}")
        print("=" * 70)
        print()

        processed_dir = self.base_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        repo = self.config["repo"]

        for split in ["valid", "test"]:
            print(f"Processing {split} split...")

            try:
                split_data = load_dataset(repo, split=split)
            except Exception as e:
                print(f"  Failed to load {split}: {e}")
                continue

            eval_samples = []
            stats = {
                "total": 0,
                "with_public_tests": 0,
                "with_private_tests": 0,
            }

            for idx, example in enumerate(split_data):
                if idx % 50 == 0:
                    print(f"  Processing: {idx}/{len(split_data)}", end="\r")

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

                # 평가용 샘플 생성
                eval_sample = {
                    "task_id": task_name,
                    "instruction": description,
                    "input": "",
                    "metadata": {
                        "source": example.get("source", 0),
                        "difficulty": example.get("difficulty", 0),
                        "cf_rating": example.get("cf_rating", 0),
                        "time_limit": example.get("time_limit", {}),
                        "memory_limit_bytes": example.get("memory_limit_bytes", 0),
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

                stats["total"] += 1
                if len(public_inputs) > 0:
                    stats["with_public_tests"] += 1
                if len(private_inputs) > 0:
                    stats["with_private_tests"] += 1

            print()

            # JSONL 저장
            output_file = processed_dir / f"{split}_eval.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for sample in eval_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            print(f"✓ Saved {split}_eval.jsonl ({stats['total']} problems)")
            print(f"  with_public_tests: {stats['with_public_tests']}")
            print(f"  with_private_tests: {stats['with_private_tests']}")

            del split_data

        print()
        print("✅ Eval dataset created!")
        print()

    def extract_tests(self) -> None:
        """테스트 케이스 추출 (codecontests 전용)

        public_tests, private_tests, generated_tests를 별도 JSON으로 저장.
        평가 시 코드 실행 검증에 사용.
        출력: tests/{split}_tests.json
        """
        if self.dataset_name != "codecontests":
            print(f"⚠️  tests step is only for codecontests, skipping {self.dataset_name}")
            return

        print("=" * 70)
        print(f"Extracting tests for {self.dataset_name}")
        print("=" * 70)
        print()

        tests_dir = self.base_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        repo = self.config["repo"]

        for split in self.config["splits"]:
            print(f"Processing {split} split...")

            try:
                split_data = load_dataset(repo, split=split)
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

                public_inputs = list(public_tests.get("input", []))
                public_outputs = list(public_tests.get("output", []))
                private_inputs = list(private_tests.get("input", []))
                private_outputs = list(private_tests.get("output", []))
                generated_inputs = list(generated_tests.get("input", []))
                generated_outputs = list(generated_tests.get("output", []))

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

            print()

            # JSON 저장
            output_file = tests_dir / f"{split}_tests.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(tests_data, f, ensure_ascii=False)

            print(f"✓ Saved {split}_tests.json ({stats['total_problems']} problems)")
            print(f"  public: {stats['problems_with_public']} problems, {stats['total_public_tests']} tests")
            print(f"  private: {stats['problems_with_private']} problems, {stats['total_private_tests']} tests")
            print(f"  generated: {stats['problems_with_generated']} problems, {stats['total_generated_tests']} tests")

            del split_data

        print()
        print("✅ Tests extracted!")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase1 데이터셋 설정 스크립트 (HuggingFace datasets 사용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 전체 데이터셋 설정 (다운로드 + 변환 + 메타데이터 + small + stats + eval + tests)
  uv run python scripts/create_storage/setup_datasets.py --datasets all --steps all

  # MBPP만 처리
  uv run python scripts/create_storage/setup_datasets.py --datasets mbpp --steps all

  # 다운로드+전처리만
  uv run python scripts/create_storage/setup_datasets.py --datasets all --steps process

  # 메타데이터만 추출
  uv run python scripts/create_storage/setup_datasets.py --datasets codecontests --steps metadata

  # CodeContests 평가용 데이터셋 생성
  uv run python scripts/create_storage/setup_datasets.py --datasets codecontests --steps eval

  # CodeContests 테스트 케이스 추출
  uv run python scripts/create_storage/setup_datasets.py --datasets codecontests --steps tests

  # 단계별 실행
  uv run python scripts/create_storage/setup_datasets.py --datasets codecontests --steps process,metadata,small,stats,eval,tests
        """,
    )

    parser.add_argument(
        "--datasets",
        required=True,
        help="Datasets to setup (comma-separated or 'all'): codecontests,mbpp,humaneval,all",
    )

    parser.add_argument(
        "--steps",
        default="all",
        help="Steps to run (comma-separated): process,metadata,small,stats,eval,tests,all",
    )

    args = parser.parse_args()

    # Datasets 파싱
    if args.datasets == "all":
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]

    # Steps 파싱
    if args.steps == "all":
        steps = ["process", "metadata", "small", "stats", "eval", "tests"]
    else:
        steps = [s.strip() for s in args.steps.split(",")]

    # Max tokens 파라미터 추가 (필요 시 argparse에서 받을 수 있음)
    max_tokens = 2048

    # 각 데이터셋 처리
    for dataset_name in datasets:
        try:
            setup = DatasetSetup(dataset_name, max_tokens=max_tokens)

            if "process" in steps:
                setup.download_and_process()

            if "metadata" in steps:
                setup.extract_metadata()

            if "small" in steps:
                setup.create_small()

            if "stats" in steps:
                setup.generate_stats()

            if "eval" in steps:
                setup.create_eval()

            if "tests" in steps:
                setup.extract_tests()

        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print()
    print("=" * 70)
    print("✅ All datasets processed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
