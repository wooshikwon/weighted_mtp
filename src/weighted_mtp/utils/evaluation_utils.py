"""Code Evaluation 유틸리티

생성된 코드의 execution-based 평가 및 Pass@K 메트릭 계산
GSM8K 등 Math 벤치마크의 exact match 평가 지원
"""

import re
import subprocess
import tempfile
from pathlib import Path

from scipy.special import comb


# HumanEval 표준 stop sequences (OpenAI/EvalPlus best practice)
HUMANEVAL_STOP_SEQUENCES = [
    "\nclass ",
    "\ndef ",
    "\n#",
    "\nif __name__",
    "\nprint(",
    "\nassert ",
    '\n"""',
    "\n'''",
]


def postprocess_humaneval_completion(completion: str, prompt: str = "") -> str:
    """HumanEval 생성 코드 후처리 (OpenAI/EvalPlus best practice)

    1. Stop sequences에서 truncate
    2. 중복 함수 시그니처 제거
    3. 불필요한 trailing content 제거

    Args:
        completion: 생성된 코드 (prompt 제외)
        prompt: 원본 프롬프트 (함수 시그니처+docstring)

    Returns:
        후처리된 함수 본문
    """
    # 1. 마크다운 코드 블록 제거
    completion = strip_markdown_code_block(completion)

    # 2. Stop sequences에서 truncate
    for stop in HUMANEVAL_STOP_SEQUENCES:
        if stop in completion:
            completion = completion[:completion.index(stop)]

    # 3. 중복 함수 시그니처 제거
    # 모델이 프롬프트의 함수 시그니처를 다시 생성하는 경우
    if prompt:
        # 함수 이름 추출 (def function_name)
        func_match = re.search(r'def\s+(\w+)\s*\(', prompt)
        if func_match:
            func_name = func_match.group(1)
            # 생성된 코드에 같은 함수 정의가 있으면 제거
            dup_pattern = rf'\ndef\s+{func_name}\s*\([^)]*\)\s*(?:->.*?)?:'
            completion = re.sub(dup_pattern, '', completion)

    # 4. Trailing whitespace 정리 (빈 줄 유지)
    lines = completion.rstrip().split('\n')
    # 마지막 빈 줄들 제거하되, 최소 하나의 내용은 유지
    while len(lines) > 1 and not lines[-1].strip():
        lines.pop()
    completion = '\n'.join(lines)

    return completion


def truncate_at_stop_sequences(
    text: str,
    stop_sequences: list[str] | None = None,
) -> str:
    """Stop sequences에서 텍스트 truncate

    Args:
        text: 원본 텍스트
        stop_sequences: stop sequence 리스트 (None이면 HUMANEVAL_STOP_SEQUENCES 사용)

    Returns:
        Truncate된 텍스트
    """
    if stop_sequences is None:
        stop_sequences = HUMANEVAL_STOP_SEQUENCES

    for stop in stop_sequences:
        if stop in text:
            text = text[:text.index(stop)]

    return text


def strip_markdown_code_block(code: str) -> str:
    """마크다운 코드 블록 제거

    LLM이 생성한 코드에서 ```python 또는 ``` 블록을 제거.

    Args:
        code: 생성된 코드 문자열

    Returns:
        마크다운 블록이 제거된 순수 코드
    """
    code = code.strip()

    # ``` 로 시작하는 경우
    if code.startswith("```"):
        lines = code.split("\n")
        # 첫 줄 제거 (```python, ```py, ``` 등)
        lines = lines[1:]
        # 마지막 줄이 ``` 면 제거
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return code


def execute_code_with_tests(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 5,
) -> bool:
    """생성된 코드 실행 및 테스트

    Args:
        code: 생성된 함수 코드
        test_code: 채점용 test case 코드
        entry_point: 함수 이름
        timeout: 실행 제한 시간 (초)

    Returns:
        정답 여부 (True=pass, False=fail)

    Examples:
        >>> code = '''
        ... def add(a, b):
        ...     return a + b
        ... '''
        >>> test = '''
        ... def check(candidate):
        ...     assert candidate(1, 2) == 3
        ... '''
        >>> execute_code_with_tests(code, test, "add", timeout=5)
        True
    """
    # 마크다운 코드 블록 제거
    code = strip_markdown_code_block(code)

    # 전체 코드 조합
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        # subprocess로 안전하게 실행
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )

        # 정상 종료 여부 확인
        passed = (result.returncode == 0)

    except subprocess.TimeoutExpired:
        # Timeout 발생 시 오답 처리
        passed = False
    except Exception:
        # 기타 예외 발생 시 오답 처리
        passed = False
    finally:
        # 임시 파일 삭제
        Path(temp_path).unlink(missing_ok=True)

    return passed


def execute_mbpp_tests(
    code: str,
    test_list: list[str],
    test_setup_code: str = "",
    timeout: int = 5,
) -> bool:
    """MBPP 형식의 테스트 실행

    Args:
        code: 생성된 함수 코드
        test_list: assert 문 리스트 (예: ["assert func(x) == y", ...])
        test_setup_code: 테스트 실행 전 설정 코드
        timeout: 실행 제한 시간 (초)

    Returns:
        정답 여부 (True=pass, False=fail)
    """
    # 마크다운 코드 블록 제거
    code = strip_markdown_code_block(code)

    # 테스트 코드 조합
    test_code = "\n".join(test_list)
    full_code = f"{test_setup_code}\n{code}\n\n{test_code}\n"

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        passed = (result.returncode == 0)

    except subprocess.TimeoutExpired:
        passed = False
    except Exception:
        passed = False
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return passed


def execute_codecontests_tests(
    code: str,
    tests: dict,
    timeout: int = 10,
) -> dict:
    """CodeContests 형식의 테스트 실행

    stdin으로 입력을 전달하고 stdout 출력을 expected와 비교.

    Args:
        code: 생성된 Python 코드
        tests: {"input": [str, ...], "output": [str, ...]}
        timeout: 테스트당 실행 제한 시간 (초)

    Returns:
        {
            "passed": int,
            "total": int,
            "pass_rate": float,
            "details": [{"input": str, "expected": str, "actual": str, "passed": bool}, ...]
        }
    """
    # 마크다운 코드 블록 제거
    code = strip_markdown_code_block(code)

    inputs = tests.get("input", [])
    expected_outputs = tests.get("output", [])

    if not inputs or not expected_outputs:
        return {
            "passed": 0,
            "total": 0,
            "pass_rate": 0.0,
            "details": [],
        }

    # 임시 파일로 코드 저장
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    results = []
    passed_count = 0

    # stdin용 임시 파일 생성 (파이프 데드락 방지)
    input_file = None

    try:
        for test_input, expected in zip(inputs, expected_outputs):
            try:
                # stdin을 파일로 전달 (파이프 블로킹 문제 해결)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(test_input)
                    input_file = f.name

                with open(input_file, 'r') as stdin_f:
                    proc = subprocess.Popen(
                        ['python', temp_path],
                        stdin=stdin_f,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )

                    try:
                        stdout, stderr = proc.communicate(timeout=timeout)
                        actual = stdout
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                        raise

                # 출력 비교 (trailing whitespace 무시)
                is_passed = actual.strip() == expected.strip()

                if is_passed:
                    passed_count += 1

                results.append({
                    "input": test_input[:100] + "..." if len(test_input) > 100 else test_input,
                    "expected": expected[:100] + "..." if len(expected) > 100 else expected,
                    "actual": actual[:100] + "..." if len(actual) > 100 else actual,
                    "passed": is_passed,
                })

            except subprocess.TimeoutExpired:
                results.append({
                    "input": test_input[:100] + "..." if len(test_input) > 100 else test_input,
                    "expected": expected[:100] + "..." if len(expected) > 100 else expected,
                    "actual": "[TIMEOUT]",
                    "passed": False,
                })
            except Exception as e:
                results.append({
                    "input": test_input[:100] + "..." if len(test_input) > 100 else test_input,
                    "expected": expected[:100] + "..." if len(expected) > 100 else expected,
                    "actual": f"[ERROR: {str(e)[:50]}]",
                    "passed": False,
                })
            finally:
                if input_file:
                    Path(input_file).unlink(missing_ok=True)
                    input_file = None
    finally:
        Path(temp_path).unlink(missing_ok=True)

    total = len(inputs)
    return {
        "passed": passed_count,
        "total": total,
        "pass_rate": passed_count / total if total > 0 else 0.0,
        "details": results,
    }


def extract_gsm8k_answer(text: str) -> str | None:
    """GSM8K 형식의 텍스트에서 최종 답 추출

    Alpaca 템플릿 사용 시 '### Response:' 이후 부분만 분석.
    여러 패턴을 우선순위대로 시도하여 최종 답 추출.

    Args:
        text: 생성된 텍스트 (풀이 과정 포함)

    Returns:
        추출된 숫자 문자열 또는 None
    """
    # Alpaca 템플릿: '### Response:' 이후 부분만 분석
    response_marker = "### Response:"
    if response_marker in text:
        text = text.split(response_marker, 1)[1]

    # 숫자 패턴 (정수, 소수, 콤마 포함)
    num_pattern = r'-?\d+(?:,\d+)*(?:\.\d+)?'

    # 1순위: #### 뒤의 숫자 (GSM8K 표준 형식)
    match = re.search(rf'####\s*({num_pattern})', text)
    if match:
        return match.group(1).replace(',', '')

    # 2순위: "the answer is" 뒤의 숫자
    match = re.search(rf'the\s+answer\s+is\s*:?\s*({num_pattern})', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')

    # 3순위: "answer:" 또는 "answer =" 뒤의 숫자
    match = re.search(rf'answer\s*[:=]\s*({num_pattern})', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')

    # 4순위: "therefore" 뒤의 숫자
    match = re.search(rf'therefore[,\s]+.*?({num_pattern})', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')

    # 5순위: "= $숫자" 또는 "=$숫자" 패턴 (수식 결과)
    match = re.search(rf'=\s*\$?\s*({num_pattern})', text)
    if match:
        return match.group(1).replace(',', '')

    # 6순위: Response 부분의 마지막 숫자 (fallback)
    numbers = re.findall(num_pattern, text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def evaluate_gsm8k_answer(
    generated_text: str,
    ground_truth: str,
) -> bool:
    """GSM8K 답변 평가 (exact match)

    Args:
        generated_text: 모델이 생성한 텍스트
        ground_truth: 정답 숫자 문자열

    Returns:
        정답 여부 (True=correct, False=incorrect)
    """
    extracted = extract_gsm8k_answer(generated_text)
    if extracted is None:
        return False

    # 숫자 비교 (정수/실수 모두 처리)
    try:
        extracted_num = float(extracted)
        ground_truth_num = float(ground_truth)
        # 부동소수점 비교 허용 오차
        return abs(extracted_num - ground_truth_num) < 1e-6
    except ValueError:
        # 숫자 변환 실패 시 문자열 비교
        return extracted.strip() == ground_truth.strip()


def compute_pass_at_k(
    n: int,
    c: int,
    k: int,
) -> float:
    """Pass@K 메트릭 계산 (unbiased estimator)

    Args:
        n: 총 생성 개수
        c: 정답 개수
        k: 평가할 개수

    Returns:
        Pass@K 확률 [0.0, 1.0]

    Formula:
        Pass@K = 1 - C(n-c, k) / C(n, k)

    Interpretation:
        n개 중 k개를 무작위로 선택했을 때, 최소 1개가 정답일 확률

    References:
        Chen et al. (2021) "Evaluating Large Language Models Trained on Code"

    Examples:
        >>> compute_pass_at_k(n=10, c=3, k=1)  # 30%
        0.3
        >>> compute_pass_at_k(n=10, c=3, k=5)  # ~83%
        0.8333...
    """
    # k가 n보다 크면 모든 샘플 선택 → 정답 보장
    if n - c < k:
        return 1.0

    # Pass@K 공식: 1 - (오답만 k개 선택할 확률)
    return 1.0 - float(comb(n - c, k, exact=True) / comb(n, k, exact=True))


def evaluate_pass_at_k(
    results: list[bool],
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """Pass@K 메트릭 계산

    Args:
        results: 샘플별 정답 여부 리스트 [True, False, True, ...]
        k_values: 계산할 K 값들

    Returns:
        {"pass@1": 0.2, "pass@5": 0.65, "pass@10": 0.85}

    Examples:
        >>> results = [True, True, False, False, False]
        >>> evaluate_pass_at_k(results, k_values=[1, 5])
        {'pass@1': 0.4, 'pass@5': 1.0}
    """
    n = len(results)
    c = sum(results)

    metrics = {}
    for k in k_values:
        # k가 n보다 크면 skip
        if k > n:
            continue
        metrics[f"pass@{k}"] = compute_pass_at_k(n, c, k)

    return metrics
