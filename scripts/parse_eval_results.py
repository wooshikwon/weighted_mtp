#!/usr/bin/env python3
"""평가 결과 파싱 및 요약

로그 파일에서 Pass@K 결과를 추출하여 테이블로 출력합니다.
Pure, Baseline, Verifiable 3개 실험 결과를 비교합니다.

Usage:
    uv run python scripts/parse_eval_results.py
    uv run python scripts/parse_eval_results.py --logs-dir /path/to/logs
"""

import argparse
import re
from pathlib import Path


def parse_log_file(log_path: Path, dataset: str = "") -> dict | None:
    """로그 파일에서 최적 Temperature 결과 추출

    Temperature search 모드에서는 최적 결과만 반환
    CodeContests는 partial_pass_rate도 추출
    """
    if not log_path.exists():
        return None

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None

    # 최적 Temperature 결과 찾기 (Temperature Search 결과 섹션)
    best_temp_match = re.search(r"최적 Temperature: ([\d.]+)", content)

    result = {
        "temperature": None,
        "pass@1": None,
        "pass@5": None,
        "pass@10": None,
        "pass@20": None,
        "partial_pass_rate": None,  # CodeContests용
    }

    if best_temp_match:
        result["temperature"] = float(best_temp_match.group(1))

        # 최적 Temperature 이후의 pass@k 결과 추출
        best_section = content[best_temp_match.end():]
        for k in [1, 5, 10, 20]:
            match = re.search(rf"pass@{k}: ([\d.]+)%", best_section)
            if match:
                result[f"pass@{k}"] = float(match.group(1))
    else:
        # Temperature search 없는 경우 (단일 temperature)
        # 평가 결과 요약 섹션에서 추출
        for k in [1, 5, 10, 20]:
            match = re.search(rf"pass@{k}: ([\d.]+)%", content)
            if match:
                result[f"pass@{k}"] = float(match.group(1))

    # CodeContests: partial_pass_rate 추출
    if dataset == "codecontests":
        partial_match = re.search(r"partial_pass_rate: ([\d.]+)%", content)
        if partial_match:
            result["partial_pass_rate"] = float(partial_match.group(1))

    # 결과가 하나도 없으면 None 반환
    if result["pass@1"] is None:
        return None

    return result


def format_value(val, suffix="%"):
    """값을 포맷팅"""
    if val is None:
        return "-"
    return f"{val:.2f}{suffix}"


def print_results_table(all_results: dict, datasets: list[str], experiments: list[str]):
    """3개 실험 결과 테이블 출력"""
    print("=" * 100)
    print("Pass@K Results Summary")
    print("=" * 100)
    print()

    # Pass@1 비교 테이블
    print("[ Pass@1 ]")
    print(f"{'Dataset':<15} {'Pure':>12} {'Baseline':>12} {'Verifiable':>12} {'B-P':>8} {'V-B':>8}")
    print("-" * 75)

    for dataset in datasets:
        pure_val = all_results.get(f"{dataset}_pure", {}).get("pass@1")
        baseline_val = all_results.get(f"{dataset}_baseline", {}).get("pass@1")
        verifiable_val = all_results.get(f"{dataset}_verifiable", {}).get("pass@1")

        # 차이 계산
        diff_bp = None
        diff_vb = None
        if pure_val is not None and baseline_val is not None:
            diff_bp = baseline_val - pure_val
        if baseline_val is not None and verifiable_val is not None:
            diff_vb = verifiable_val - baseline_val

        print(f"{dataset:<15} {format_value(pure_val):>12} {format_value(baseline_val):>12} "
              f"{format_value(verifiable_val):>12} {format_value(diff_bp, ''):>8} {format_value(diff_vb, ''):>8}")

    print("-" * 75)
    print()

    # Pass@5 비교 테이블
    print("[ Pass@5 ]")
    print(f"{'Dataset':<15} {'Pure':>12} {'Baseline':>12} {'Verifiable':>12} {'B-P':>8} {'V-B':>8}")
    print("-" * 75)

    for dataset in datasets:
        pure_val = all_results.get(f"{dataset}_pure", {}).get("pass@5")
        baseline_val = all_results.get(f"{dataset}_baseline", {}).get("pass@5")
        verifiable_val = all_results.get(f"{dataset}_verifiable", {}).get("pass@5")

        diff_bp = None
        diff_vb = None
        if pure_val is not None and baseline_val is not None:
            diff_bp = baseline_val - pure_val
        if baseline_val is not None and verifiable_val is not None:
            diff_vb = verifiable_val - baseline_val

        print(f"{dataset:<15} {format_value(pure_val):>12} {format_value(baseline_val):>12} "
              f"{format_value(verifiable_val):>12} {format_value(diff_bp, ''):>8} {format_value(diff_vb, ''):>8}")

    print("-" * 75)
    print()

    # Pass@10 비교 테이블
    print("[ Pass@10 ]")
    print(f"{'Dataset':<15} {'Pure':>12} {'Baseline':>12} {'Verifiable':>12} {'B-P':>8} {'V-B':>8}")
    print("-" * 75)

    for dataset in datasets:
        pure_val = all_results.get(f"{dataset}_pure", {}).get("pass@10")
        baseline_val = all_results.get(f"{dataset}_baseline", {}).get("pass@10")
        verifiable_val = all_results.get(f"{dataset}_verifiable", {}).get("pass@10")

        diff_bp = None
        diff_vb = None
        if pure_val is not None and baseline_val is not None:
            diff_bp = baseline_val - pure_val
        if baseline_val is not None and verifiable_val is not None:
            diff_vb = verifiable_val - baseline_val

        print(f"{dataset:<15} {format_value(pure_val):>12} {format_value(baseline_val):>12} "
              f"{format_value(verifiable_val):>12} {format_value(diff_bp, ''):>8} {format_value(diff_vb, ''):>8}")

    print("-" * 75)
    print()

    # Pass@20 비교 테이블
    print("[ Pass@20 ]")
    print(f"{'Dataset':<15} {'Pure':>12} {'Baseline':>12} {'Verifiable':>12} {'B-P':>8} {'V-B':>8}")
    print("-" * 75)

    for dataset in datasets:
        pure_val = all_results.get(f"{dataset}_pure", {}).get("pass@20")
        baseline_val = all_results.get(f"{dataset}_baseline", {}).get("pass@20")
        verifiable_val = all_results.get(f"{dataset}_verifiable", {}).get("pass@20")

        diff_bp = None
        diff_vb = None
        if pure_val is not None and baseline_val is not None:
            diff_bp = baseline_val - pure_val
        if baseline_val is not None and verifiable_val is not None:
            diff_vb = verifiable_val - baseline_val

        print(f"{dataset:<15} {format_value(pure_val):>12} {format_value(baseline_val):>12} "
              f"{format_value(verifiable_val):>12} {format_value(diff_bp, ''):>8} {format_value(diff_vb, ''):>8}")

    print("-" * 75)
    print()

    # CodeContests Partial Pass Rate 테이블
    print("[ CodeContests Partial Pass Rate (평균 테스트 통과율) ]")
    print(f"{'Dataset':<15} {'Pure':>12} {'Baseline':>12} {'Verifiable':>12} {'B-P':>8} {'V-B':>8}")
    print("-" * 75)

    dataset = "codecontests"
    pure_val = all_results.get(f"{dataset}_pure", {}).get("partial_pass_rate")
    baseline_val = all_results.get(f"{dataset}_baseline", {}).get("partial_pass_rate")
    verifiable_val = all_results.get(f"{dataset}_verifiable", {}).get("partial_pass_rate")

    diff_bp = None
    diff_vb = None
    if pure_val is not None and baseline_val is not None:
        diff_bp = baseline_val - pure_val
    if baseline_val is not None and verifiable_val is not None:
        diff_vb = verifiable_val - baseline_val

    print(f"{dataset:<15} {format_value(pure_val):>12} {format_value(baseline_val):>12} "
          f"{format_value(verifiable_val):>12} {format_value(diff_bp, ''):>8} {format_value(diff_vb, ''):>8}")

    print("-" * 75)


def print_summary(all_results: dict, datasets: list[str]):
    """실험별 평균 요약"""
    print()
    print("=" * 100)
    print("Average Across Datasets (Pass@1)")
    print("=" * 100)

    for exp in ["pure", "baseline", "verifiable"]:
        values = []
        for dataset in datasets:
            key = f"{dataset}_{exp}"
            if key in all_results and all_results[key].get("pass@1") is not None:
                values.append(all_results[key]["pass@1"])

        if values:
            avg = sum(values) / len(values)
            print(f"  {exp.capitalize():12}: {avg:.2f}% (n={len(values)})")
        else:
            print(f"  {exp.capitalize():12}: N/A")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="평가 결과 파싱")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="로그 디렉토리 경로",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    datasets = ["humaneval", "mbpp", "gsm8k", "codecontests"]
    experiments = ["pure", "baseline", "verifiable"]

    # 모든 결과 수집
    all_results = {}

    for dataset in datasets:
        for exp in experiments:
            log_file = logs_dir / f"eval_{dataset}_{exp}.log"
            result = parse_log_file(log_file, dataset=dataset)
            if result:
                all_results[f"{dataset}_{exp}"] = result
                # CodeContests는 partial pass rate도 표시
                if dataset == "codecontests" and result.get("partial_pass_rate") is not None:
                    print(f"[OK] {log_file.name}: pass@1={result['pass@1']:.2f}%, partial={result['partial_pass_rate']:.2f}%")
                else:
                    print(f"[OK] {log_file.name}: pass@1={result['pass@1']:.2f}%")
            else:
                print(f"[--] {log_file.name}: not found or no results")

    print()

    if not all_results:
        print("No evaluation results found.")
        return

    # 결과 테이블 출력
    print_results_table(all_results, datasets, experiments)

    # 평균 요약
    print_summary(all_results, datasets)


if __name__ == "__main__":
    main()
