"""pytest 공통 fixture"""

import json
import pytest
import sys
from pathlib import Path

from weighted_mtp.utils import s3_utils


@pytest.fixture(scope="session", autouse=True)
def setup_vendor_path():
    """vendor 모듈 접근을 위한 PYTHONPATH 설정

    vendor/는 프로젝트 루트에 위치하므로 PYTHONPATH에 추가 필요
    """
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


@pytest.fixture
def project_root() -> Path:
    """프로젝트 루트 경로"""
    return Path(__file__).parent.parent


@pytest.fixture
def storage_root(project_root: Path) -> Path:
    """storage/ 경로"""
    return project_root / "storage"


@pytest.fixture
def micro_model_path(storage_root: Path) -> Path:
    """Micro MTP 모델 경로"""
    return storage_root / "models/micro-mtp"


@pytest.fixture(autouse=True)
def ensure_s3_executor_available():
    """각 테스트 전에 s3_upload_executor가 사용 가능한지 확인

    이전 테스트에서 shutdown된 경우 재생성
    """
    if s3_utils.s3_upload_executor._shutdown:
        s3_utils.reset_s3_executor()


# ---------------------------------------------------------------------------
# Fixture 데이터 생성 — CodeContests + MBPP
# ---------------------------------------------------------------------------

def _generate_fixture_jsonl_and_metadata(
    dataset_dir: Path,
    dataset_name: str,
    n_problems: int = 50,
    correct_per_problem: int = 20,
    incorrect_per_problem: int = 10,
    splits: tuple[str, ...] = ("train", "valid"),
    generate_eval: bool = False,
):
    """테스트용 JSONL + metadata 생성

    실제 setup_codecontests_plus.py 출력과 동일한 스키마.
    충분한 데이터를 생성하여 pairwise sampling, difficulty-based sampling,
    distributed loading 등 모든 테스트를 커버.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        records = []
        problem_index_map = {}
        global_idx = 0

        for pid in range(n_problems):
            problem_id = f"test_problem_{pid:04d}"
            # difficulty 분포: ~33% difficulty=7, 나머지 8-20 범위
            difficulty = 7 if pid % 3 == 0 else (8 + pid % 13)

            correct_indices = []
            correct_token_lengths = []
            incorrect_indices = []
            incorrect_token_lengths = []

            # Correct solutions — 각각 고유한 코드
            for sol_idx in range(correct_per_problem):
                token_len = 50 + (pid * 7 + sol_idx * 13) % 200
                records.append({
                    "instruction": f"Solve problem {pid}: compute f({pid})",
                    "input": "",
                    "output": (
                        f"def solve_{pid}_{sol_idx}():\n"
                        f"    # solution {sol_idx} for problem {pid}\n"
                        f"    result = {pid * 1000 + sol_idx}\n"
                        f"    return result\n"
                    ),
                    "task_id": f"{problem_id}_correct_{sol_idx}",
                    "is_correct": True,
                    "metadata": {
                        "source": f"{dataset_name}_test_fixture",
                        "difficulty": difficulty,
                        "problem_id": problem_id,
                    },
                })
                correct_indices.append(global_idx)
                correct_token_lengths.append(token_len)
                global_idx += 1

            # Incorrect solutions — 각각 고유한 코드 (problem_id + sol_idx)
            for sol_idx in range(incorrect_per_problem):
                token_len = 40 + (pid * 5 + sol_idx * 11) % 180
                records.append({
                    "instruction": f"Solve problem {pid}: compute f({pid})",
                    "input": "",
                    "output": (
                        f"def wrong_{pid}_{sol_idx}():\n"
                        f"    # incorrect attempt {sol_idx} for problem {pid}\n"
                        f"    return {-(pid * 1000 + sol_idx + 1)}\n"
                    ),
                    "task_id": f"{problem_id}_incorrect_{sol_idx}",
                    "is_correct": False,
                    "metadata": {
                        "source": f"{dataset_name}_test_fixture",
                        "difficulty": difficulty,
                        "problem_id": problem_id,
                    },
                })
                incorrect_indices.append(global_idx)
                incorrect_token_lengths.append(token_len)
                global_idx += 1

            problem_index_map[problem_id] = {
                "difficulty": difficulty,
                "correct_indices": correct_indices,
                "incorrect_indices": incorrect_indices,
                "correct_token_lengths": correct_token_lengths,
                "incorrect_token_lengths": incorrect_token_lengths,
            }

        # Write JSONL
        jsonl_path = dataset_dir / f"{split}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Write metadata
        n_correct = sum(
            len(v["correct_indices"]) for v in problem_index_map.values()
        )
        n_incorrect = sum(
            len(v["incorrect_indices"]) for v in problem_index_map.values()
        )
        metadata = {
            "dataset": f"{dataset_name}_test_fixture",
            "total_samples": len(records),
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_problems": n_problems,
            "n_valid_problems": n_problems,
            "problem_index_map": problem_index_map,
        }
        metadata_path = dataset_dir / f"{split}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    # CodeContests 평가용 test_eval.jsonl (problem-level, 별도 포맷)
    if generate_eval:
        eval_records = []
        for pid in range(min(n_problems, 20)):
            problem_id = f"test_problem_{pid:04d}"
            eval_records.append({
                "instruction": f"Solve problem {pid}: compute f({pid})",
                "input": "",
                "output": "",
                "task_id": problem_id,
                "metadata": {
                    "source": f"{dataset_name}_test_fixture",
                    "problem_id": problem_id,
                },
            })
        eval_path = dataset_dir / "test_eval.jsonl"
        with open(eval_path, "w", encoding="utf-8") as f:
            for record in eval_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


@pytest.fixture(scope="session", autouse=True)
def create_test_fixture_data():
    """테스트 실행 전에 fixture 데이터셋을 생성하고, 종료 후 정리

    storage/datasets/ 아래에 codecontests와 mbpp fixture를 생성.
    실제 데이터가 이미 존재하면 건드리지 않음.
    """
    project_root = Path(__file__).parent.parent
    storage_base = project_root / "storage" / "datasets"
    created_dirs = []

    # CodeContests fixture (50 problems × 25 correct + 25 incorrect = 2500 samples)
    # max_pairs_per_problem=20 기본값 기준 50×20=1000 쌍 가용
    cc_dir = storage_base / "codecontests" / "processed"
    if not cc_dir.exists():
        _generate_fixture_jsonl_and_metadata(
            dataset_dir=cc_dir,
            dataset_name="codecontests",
            n_problems=50,
            correct_per_problem=25,
            incorrect_per_problem=25,
            splits=("train", "valid"),
            generate_eval=True,
        )
        created_dirs.append(cc_dir)

    # MBPP fixture (40 problems × 15 correct + 8 incorrect = 920 samples)
    mbpp_dir = storage_base / "mbpp" / "processed"
    if not mbpp_dir.exists():
        _generate_fixture_jsonl_and_metadata(
            dataset_dir=mbpp_dir,
            dataset_name="mbpp",
            n_problems=40,
            correct_per_problem=15,
            incorrect_per_problem=8,
            splits=("train", "valid"),
        )
        created_dirs.append(mbpp_dir)

    yield

    # Cleanup: 우리가 생성한 fixture만 삭제
    import shutil
    for d in created_dirs:
        if d.exists():
            shutil.rmtree(d)
        # 빈 부모 디렉토리도 정리
        parent = d.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
