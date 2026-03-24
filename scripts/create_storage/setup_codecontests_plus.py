#!/usr/bin/env python3
"""CodeContests+ 데이터셋 전처리 — ByteDance-Seed/Code-Contests-Plus

기존 CodeContests 대비:
- 11,690 problems, 13M+ solutions (correct + incorrect)
- 고품질 테스트케이스 (TPR/TNR 대폭 향상)
- difficulty 필드 없음 → 원본 CodeContests와 ID 매칭으로 보완

출력 포맷은 기존 파이프라인과 동일:
- train.jsonl / valid.jsonl (Alpaca 형식, solution-level rows)
- train_metadata.json / valid_metadata.json (problem_index_map)

Usage:
    # 전체 데이터셋 (GPU 서버에서 실행)
    python scripts/create_storage/setup_codecontests_plus.py

    # Go/No-Go 테스트용 (문제 500개, 솔루션 cap 적용)
    python scripts/create_storage/setup_codecontests_plus.py \
        --max-problems 500 --max-solutions-per-problem 100

    # 메타데이터만 재생성
    python scripts/create_storage/setup_codecontests_plus.py --steps metadata
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 출력 디렉토리
OUTPUT_DIR = Path("storage/datasets/codecontests/processed")

# Python 언어 필터
PYTHON_LANGUAGES = {"Python", "Python 3", "Python3", "PyPy", "PyPy 3", "PyPy3"}


def download_dataset(max_problems: int | None = None):
    """HuggingFace에서 CodeContests+ 다운로드

    Args:
        max_problems: 최대 문제 수 (None이면 전체)

    Returns:
        HuggingFace Dataset
    """
    from datasets import load_dataset

    logger.info("Downloading ByteDance-Seed/Code-Contests-Plus (default subset)...")
    ds = load_dataset("ByteDance-Seed/Code-Contests-Plus", "default", split="train")
    logger.info(f"Downloaded: {len(ds):,} problems")

    if max_problems and max_problems < len(ds):
        # 재현 가능한 샘플링
        ds = ds.shuffle(seed=42).select(range(max_problems))
        logger.info(f"Subsampled to {len(ds):,} problems (--max-problems={max_problems})")

    return ds


def load_original_difficulty_map() -> dict[str, int]:
    """원본 CodeContests에서 difficulty 매핑 로드 (ID 기반)

    Returns:
        {problem_name: difficulty} 딕셔너리
    """
    # 원본 메타데이터가 있으면 활용
    original_meta_path = OUTPUT_DIR / "train_metadata_original.json"
    if original_meta_path.exists():
        with open(original_meta_path) as f:
            meta = json.load(f)
        pim = meta.get("problem_index_map", {})
        return {pid: info.get("difficulty", 5) for pid, info in pim.items()}

    # 없으면 원본 데이터셋에서 직접 추출 시도
    try:
        from datasets import load_dataset as ld

        logger.info("Loading original deepmind/code_contests for difficulty mapping...")
        orig = ld("deepmind/code_contests", split="train")
        diff_map = {}
        for row in orig:
            name = row.get("name", "")
            diff = row.get("difficulty", 5)
            if name:
                diff_map[name] = diff
        logger.info(f"Loaded {len(diff_map):,} difficulty mappings from original dataset")
        return diff_map
    except Exception as e:
        logger.warning(f"Could not load original difficulty: {e}")
        logger.warning("All problems will be assigned difficulty=5 (default)")
        return {}


def process_dataset(
    ds,
    difficulty_map: dict[str, int],
    max_solutions_per_problem: int | None = None,
    val_ratio: float = 0.05,
    seed: int = 42,
):
    """CodeContests+ → Alpaca JSONL 변환

    Args:
        ds: HuggingFace Dataset
        difficulty_map: {problem_name: difficulty}
        max_solutions_per_problem: correct/incorrect 각각 최대 솔루션 수
        val_ratio: validation split 비율
        seed: 랜덤 시드

    Returns:
        (train_samples, valid_samples) 튜플
    """
    rng = random.Random(seed)

    # Problem-level train/valid split
    n_problems = len(ds)
    indices = list(range(n_problems))
    rng.shuffle(indices)
    n_val = max(1, int(n_problems * val_ratio))
    val_indices = set(indices[:n_val])

    train_samples = []
    valid_samples = []

    # 언어 통계
    lang_stats = defaultdict(int)
    skipped_no_python = 0

    for prob_idx in range(n_problems):
        row = ds[prob_idx]
        problem_id = row.get("title", "") or row.get("id", f"problem_{prob_idx}")
        description = row.get("description", "")

        # Difficulty: 원본 매칭 → 기본값 5
        difficulty = difficulty_map.get(problem_id, 5)

        # Python 솔루션 필터링
        correct_subs = row.get("correct_submissions", [])
        incorrect_subs = row.get("incorrect_submissions", [])

        correct_python = []
        incorrect_python = []

        for sub in correct_subs:
            lang = sub.get("language", "")
            lang_stats[lang] += 1
            if lang in PYTHON_LANGUAGES:
                correct_python.append(sub["code"])

        for sub in incorrect_subs:
            lang = sub.get("language", "")
            lang_stats[lang] += 1
            if lang in PYTHON_LANGUAGES:
                incorrect_python.append(sub["code"])

        # Python 솔루션이 없으면 스킵
        if not correct_python and not incorrect_python:
            skipped_no_python += 1
            continue

        # 솔루션 수 제한
        if max_solutions_per_problem:
            rng.shuffle(correct_python)
            rng.shuffle(incorrect_python)
            correct_python = correct_python[:max_solutions_per_problem]
            incorrect_python = incorrect_python[:max_solutions_per_problem]

        # 대상 리스트 선택
        target = valid_samples if prob_idx in val_indices else train_samples

        # Correct solutions → Alpaca rows
        for sol_idx, code in enumerate(correct_python):
            target.append({
                "instruction": description,
                "input": "",
                "output": code,
                "task_id": f"{problem_id}_correct_{sol_idx}",
                "is_correct": True,
                "metadata": {
                    "source": "codecontests_plus",
                    "difficulty": difficulty,
                    "problem_id": problem_id,
                },
            })

        # Incorrect solutions → Alpaca rows
        for sol_idx, code in enumerate(incorrect_python):
            target.append({
                "instruction": description,
                "input": "",
                "output": code,
                "task_id": f"{problem_id}_incorrect_{sol_idx}",
                "is_correct": False,
                "metadata": {
                    "source": "codecontests_plus",
                    "difficulty": difficulty,
                    "problem_id": problem_id,
                },
            })

    # 통계 출력
    logger.info(f"=== Language Distribution ===")
    for lang, count in sorted(lang_stats.items(), key=lambda x: -x[1])[:15]:
        marker = " ← included" if lang in PYTHON_LANGUAGES else ""
        logger.info(f"  {lang:>20s}: {count:>10,}{marker}")

    logger.info(f"Problems with no Python solutions: {skipped_no_python}")
    logger.info(f"Train: {len(train_samples):,} samples")
    logger.info(f"Valid: {len(valid_samples):,} samples")

    return train_samples, valid_samples


def save_jsonl(samples: list[dict], path: Path):
    """JSONL 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples):,} samples → {path}")


def generate_metadata(jsonl_path: Path, output_path: Path, tokenizer=None):
    """JSONL에서 problem_index_map 메타데이터 생성

    Args:
        jsonl_path: 입력 JSONL 경로
        output_path: 메타데이터 JSON 출력 경로
        tokenizer: 토큰 길이 계산용 (None이면 문자 수 사용)
    """
    logger.info(f"Generating metadata: {jsonl_path} → {output_path}")

    problem_index_map = {}
    total = 0
    n_correct = 0
    n_incorrect = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            total += 1

            # Problem ID 추출
            metadata = item.get("metadata", {})
            problem_id = metadata.get("problem_id")
            if not problem_id:
                task_id = item.get("task_id", "")
                # "problemname_correct_0" → "problemname"
                parts = task_id.rsplit("_", 2)
                problem_id = parts[0] if len(parts) >= 3 else task_id

            if not problem_id:
                continue

            difficulty = metadata.get("difficulty", 5)

            if problem_id not in problem_index_map:
                problem_index_map[problem_id] = {
                    "difficulty": difficulty,
                    "correct_indices": [],
                    "incorrect_indices": [],
                    "correct_token_lengths": [],
                    "incorrect_token_lengths": [],
                }

            # 토큰 길이 계산
            output_text = item.get("output", "")
            if tokenizer:
                token_length = len(tokenizer.encode(output_text))
            else:
                # fallback: 문자 수 / 4 (대략적 토큰 추정)
                token_length = len(output_text) // 4

            is_correct = item.get("is_correct", False)
            if is_correct:
                problem_index_map[problem_id]["correct_indices"].append(idx)
                problem_index_map[problem_id]["correct_token_lengths"].append(token_length)
                n_correct += 1
            else:
                problem_index_map[problem_id]["incorrect_indices"].append(idx)
                problem_index_map[problem_id]["incorrect_token_lengths"].append(token_length)
                n_incorrect += 1

    # 통계
    n_problems = len(problem_index_map)
    n_valid = sum(
        1 for p in problem_index_map.values()
        if len(p["correct_indices"]) > 0 and len(p["incorrect_indices"]) > 0
    )

    logger.info(f"=== Metadata Statistics ===")
    logger.info(f"Total samples: {total:,} (correct: {n_correct:,}, incorrect: {n_incorrect:,})")
    logger.info(f"Problems: {n_problems:,} (with both correct+incorrect: {n_valid:,})")

    # Length-balanced 시뮬레이션
    length_bins = [0, 50, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1500, 2000]

    def get_bin(tl):
        for i, b in enumerate(length_bins[1:]):
            if tl < b:
                return f"{length_bins[i]}-{b}"
        return f"{length_bins[-1]}+"

    total_pairs_50 = 0
    total_pairs_unlimited = 0
    bin_pair_counts = defaultdict(int)

    for pid, info in problem_index_map.items():
        cl = info["correct_token_lengths"]
        il = info["incorrect_token_lengths"]
        ci = info["correct_indices"]
        ii = info["incorrect_indices"]

        if not cl or not il:
            continue

        cb = defaultdict(list)
        ib = defaultdict(list)
        for idx, length in zip(ci, cl):
            cb[get_bin(length)].append(idx)
        for idx, length in zip(ii, il):
            ib[get_bin(length)].append(idx)

        pp = 0
        for bl in set(cb.keys()) & set(ib.keys()):
            n = min(len(cb[bl]), len(ib[bl]))
            pp += n
            bin_pair_counts[bl] += n

        total_pairs_unlimited += pp
        total_pairs_50 += min(pp, 50)

    logger.info(f"\n=== Length-Balanced Pair Capacity ===")
    logger.info(f"Total pairs (unlimited): {total_pairs_unlimited:,}")
    logger.info(f"Total pairs (cap=50): {total_pairs_50:,}")
    logger.info(f"Bin distribution:")
    for bl in sorted(bin_pair_counts.keys(), key=lambda x: int(x.split("-")[0]) if "-" in x else 9999):
        logger.info(f"  {bl:>10s}: {bin_pair_counts[bl]:>10,} pairs")

    # 저장
    meta = {
        "dataset": "codecontests_plus",
        "total_samples": total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_problems": n_problems,
        "n_valid_problems": n_valid,
        "problem_index_map": problem_index_map,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    logger.info(f"Metadata saved → {output_path}")


def load_tokenizer():
    """LLaMA-3 토크나이저 로드 (가능하면)"""
    try:
        from transformers import AutoTokenizer
        tokenizer_path = "storage/models/meta-llama/Meta-Llama-3-8B"
        if Path(tokenizer_path).exists():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Loaded LLaMA-3 tokenizer (vocab={tokenizer.vocab_size})")
            return tokenizer
    except Exception:
        pass

    try:
        import sentencepiece as spm
        sp_path = Path("storage/models/meta-llama-mtp/tokenizer/tokenizer.model")
        if sp_path.exists():
            sp = spm.SentencePieceProcessor()
            sp.load(str(sp_path))
            logger.info(f"Loaded SentencePiece tokenizer (vocab={sp.vocab_size()})")
            return sp
    except Exception:
        pass

    logger.warning("No tokenizer found — using char-length estimation (len/4)")
    return None


def main():
    parser = argparse.ArgumentParser(description="CodeContests+ 전처리")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["process", "metadata"],
        choices=["process", "metadata"],
        help="실행할 단계 (기본: process metadata)",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="최대 문제 수 (Go/No-Go: 500, 전체: None)",
    )
    parser.add_argument(
        "--max-solutions-per-problem",
        type=int,
        default=None,
        help="문제당 correct/incorrect 각각 최대 솔루션 수 (기본: 무제한)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation split 비율 (기본: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )
    args = parser.parse_args()

    if "process" in args.steps:
        logger.info("=" * 70)
        logger.info("Step 1: Download & Process CodeContests+")
        logger.info("=" * 70)

        ds = download_dataset(max_problems=args.max_problems)
        difficulty_map = load_original_difficulty_map()

        train_samples, valid_samples = process_dataset(
            ds,
            difficulty_map=difficulty_map,
            max_solutions_per_problem=args.max_solutions_per_problem,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        save_jsonl(train_samples, OUTPUT_DIR / "train.jsonl")
        save_jsonl(valid_samples, OUTPUT_DIR / "valid.jsonl")

    if "metadata" in args.steps:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Step 2: Generate Metadata")
        logger.info("=" * 70)

        tokenizer = load_tokenizer()

        for split, jsonl_name, meta_name in [
            ("train", "train.jsonl", "train_metadata.json"),
            ("valid", "valid.jsonl", "valid_metadata.json"),
        ]:
            jsonl_path = OUTPUT_DIR / jsonl_name
            if not jsonl_path.exists():
                logger.warning(f"{jsonl_path} not found, skipping metadata for {split}")
                continue

            meta_path = OUTPUT_DIR / meta_name
            generate_metadata(jsonl_path, meta_path, tokenizer)

    logger.info("")
    logger.info("Done!")
    logger.info(f"Output: {OUTPUT_DIR}/")
    logger.info("")
    logger.info("Config 조정 참고:")
    logger.info("  critic_mlp.yaml → data_sampling.n_samples 를 메타데이터의 pair capacity에 맞춰 조정")
    logger.info("  baseline.yaml   → data_sampling.n_samples 를 correct 샘플 수에 맞춰 조정")


if __name__ == "__main__":
    main()
