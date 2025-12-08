"""환경변수 로딩 유틸리티

프로젝트 루트의 .env 파일을 자동으로 로드하여 MLflow credentials 등을 설정
"""

from pathlib import Path

from dotenv import load_dotenv

_env_loaded = False


def ensure_env_loaded() -> None:
    """프로젝트 루트의 .env 파일을 로드 (한 번만)

    특징:
    - .env 파일이 없어도 에러 내지 않음 (CI 환경 대응)
    - 이미 설정된 환경변수는 override하지 않음 (환경 유연성)
    - 중복 호출 방지 (전역 플래그 사용)

    사용 예시:
        # CLI나 파이프라인 entry point 시작 부분에서
        from weighted_mtp.core.env import ensure_env_loaded
        ensure_env_loaded()

    환경:
    - 로컬: .env 파일 사용
    - CI: MLFLOW_TRACKING_URI 등 환경변수로 직접 설정
    - VESSL: .env.vessl 또는 Secrets로 환경변수 주입
    """
    global _env_loaded
    if not _env_loaded:
        # 프로젝트 루트 경로 계산 (src/weighted_mtp/core/env.py 기준)
        project_root = Path(__file__).parent.parent.parent.parent
        env_path = project_root / ".env"

        if env_path.exists():
            # override=False: 이미 설정된 환경변수는 덮어쓰지 않음
            load_dotenv(env_path, override=False)

        _env_loaded = True
