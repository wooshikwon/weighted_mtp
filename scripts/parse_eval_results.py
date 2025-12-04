#!/usr/bin/env python3
"""평가 결과 파싱 및 요약

로그 파일에서 Pass@K 결과를 추출하여 테이블로 출력합니다.
Baseline과 Verifiable 결과를 구분하여 표시합니다.

Usage:
    uv run python scripts/parse_eval_results.py
    uv run python scripts/parse_eval_results.py --logs-dir /path/to/logs
"""

import argparse
import re
from pathlib import Path


def parse_log_file(log_path: Path) -> list[dict]:
    """로그 파일에서 평가 결과 추출"""
    results = []
    current_temp = None
    current_result = {}

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Temperature별 블록 찾기
    temp_pattern = r"Temperature ([\d.]+) 평가 시작"
    result_pattern = r"pass@(\d+): ([\d.]+)%"

    lines = content.split("\n")
    for line in lines:
        # Temperature 시작
        temp_match = re.search(temp_pattern, line)
        if temp_match:
            if current_result:
                results.append(current_result)
            current_temp = float(temp_match.group(1))
            current_result = {"temperature": current_temp}

        # Pass@K 결과
        result_match = re.search(result_pattern, line)
        if result_match and current_temp is not None:
            k = int(result_match.group(1))
            value = float(result_match.group(2))
            current_result[f"pass@{k}"] = value

    # 마지막 결과 추가
    if current_result:
        results.append(current_result)

    return results


def print_results_table(title: str, all_results: dict, datasets: list[str]):
    """결과 테이블 출력"""
    print("=" * 80)
    print(f"{title} Results")
    print("=" * 80)
    print()
    print(f"{'Dataset':<15} {'Temp':<6} {'Pass@1':>10} {'Pass@5':>10} {'Pass@10':>10} {'Pass@20':>10}")
    print("-" * 80)

    for dataset in datasets:
        if dataset not in all_results:
            print(f"{dataset:<15} {'N/A':<6} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
            continue

        for result in all_results[dataset]:
            temp = result.get("temperature", "?")
            p1 = result.get("pass@1", "-")
            p5 = result.get("pass@5", "-")
            p10 = result.get("pass@10", "-")
            p20 = result.get("pass@20", "-")

            p1_str = f"{p1:.2f}%" if isinstance(p1, float) else str(p1)
            p5_str = f"{p5:.2f}%" if isinstance(p5, float) else str(p5)
            p10_str = f"{p10:.2f}%" if isinstance(p10, float) else str(p10)
            p20_str = f"{p20:.2f}%" if isinstance(p20, float) else str(p20)

            print(f"{dataset:<15} {temp:<6} {p1_str:>10} {p5_str:>10} {p10_str:>10} {p20_str:>10}")

    print("-" * 80)


def print_temperature_avg(all_results: dict):
    """Temperature별 평균 출력"""
    print("Temperature별 평균:")
    for temp in [0.2, 0.8]:
        values = {"pass@1": [], "pass@5": [], "pass@10": [], "pass@20": []}
        for dataset, results in all_results.items():
            for r in results:
                if r.get("temperature") == temp:
                    for k in ["pass@1", "pass@5", "pass@10", "pass@20"]:
                        if k in r and isinstance(r[k], float):
                            values[k].append(r[k])

        if values["pass@1"]:
            avg_p1 = sum(values["pass@1"]) / len(values["pass@1"])
            avg_p5 = sum(values["pass@5"]) / len(values["pass@5"]) if values["pass@5"] else 0
            avg_p10 = sum(values["pass@10"]) / len(values["pass@10"]) if values["pass@10"] else 0
            avg_p20 = sum(values["pass@20"]) / len(values["pass@20"]) if values["pass@20"] else 0
            print(f"  T={temp}: Pass@1={avg_p1:.2f}%, Pass@5={avg_p5:.2f}%, Pass@10={avg_p10:.2f}%, Pass@20={avg_p20:.2f}%")


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

    # Baseline 결과 수집
    baseline_results = {}
    for dataset in datasets:
        log_file = logs_dir / f"eval_{dataset}.log"
        if log_file.exists():
            results = parse_log_file(log_file)
            if results:
                baseline_results[dataset] = results

    # Verifiable 결과 수집
    verifiable_results = {}
    for dataset in datasets:
        log_file = logs_dir / f"eval_verifiable_{dataset}.log"
        if log_file.exists():
            results = parse_log_file(log_file)
            if results:
                verifiable_results[dataset] = results

    # Baseline 결과 출력
    if baseline_results:
        print_results_table("BASELINE", baseline_results, datasets)
        print()
        print_temperature_avg(baseline_results)
    else:
        print("=" * 80)
        print("[INFO] Baseline 결과 없음")
        print("=" * 80)

    print()

    # Verifiable 결과 출력
    if verifiable_results:
        print_results_table("VERIFIABLE", verifiable_results, datasets)
        print()
        print_temperature_avg(verifiable_results)
    else:
        print("=" * 80)
        print("[INFO] Verifiable 결과 없음")
        print("=" * 80)

    print()

    # 비교 테이블 (최적 Temperature 기준)
    if baseline_results and verifiable_results:
        print("=" * 80)
        print("COMPARISON (Best Temperature)")
        print("=" * 80)
        print()
        print(f"{'Dataset':<15} {'Baseline':>12} {'Verifiable':>12} {'Diff':>10}")
        print("-" * 50)

        for dataset in datasets:
            # Baseline 최고 pass@1
            baseline_best = "-"
            if dataset in baseline_results:
                p1_values = [r.get("pass@1", 0) for r in baseline_results[dataset]]
                if p1_values:
                    baseline_best = max(p1_values)

            # Verifiable 최고 pass@1
            verifiable_best = "-"
            if dataset in verifiable_results:
                p1_values = [r.get("pass@1", 0) for r in verifiable_results[dataset]]
                if p1_values:
                    verifiable_best = max(p1_values)

            # 차이 계산
            if isinstance(baseline_best, float) and isinstance(verifiable_best, float):
                diff = verifiable_best - baseline_best
                diff_str = f"{diff:+.2f}%"
                baseline_str = f"{baseline_best:.2f}%"
                verifiable_str = f"{verifiable_best:.2f}%"
            else:
                diff_str = "-"
                baseline_str = str(baseline_best) if baseline_best != "-" else "-"
                verifiable_str = str(verifiable_best) if verifiable_best != "-" else "-"

            print(f"{dataset:<15} {baseline_str:>12} {verifiable_str:>12} {diff_str:>10}")

        print("-" * 50)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
