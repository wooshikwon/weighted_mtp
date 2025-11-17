"""통합 로깅 시스템

전체 파이프라인에서 일관된 로깅 형식과 prefix를 제공
"""

import logging
import sys
from typing import Optional


def setup_logging(
    name: str,
    level: str = "INFO",
    rank: Optional[int] = None,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """통합 로깅 설정

    Args:
        name: Pipeline 이름 (CLI, BASELINE, CRITIC, VERIFIABLE, RHO1)
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rank: 분산학습 rank (None이면 단일 프로세스)
        format_str: 로그 포맷 문자열 (None이면 기본 포맷 사용)

    Returns:
        설정된 Logger 객체

    Examples:
        # CLI에서 사용
        logger = setup_logging("CLI")
        logger.info("실행 Pipeline: baseline")

        # Pipeline에서 단일 프로세스
        logger = setup_logging("BASELINE", level="INFO")
        logger.info("학습 시작")

        # Pipeline에서 분산학습
        rank = get_rank()
        logger = setup_logging("BASELINE", level="INFO", rank=rank)
        logger.info(f"Device: cuda:{rank}")
    """
    # Logger 이름 생성
    if rank is not None:
        logger_name = f"{name}:R{rank}"
    else:
        logger_name = name

    # Root logger 설정 (한 번만)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)

        # 기본 포맷: 시간 [레벨] [이름] 메시지
        if format_str is None:
            format_str = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(getattr(logging, level.upper()))

    # Pipeline별 logger 반환
    return logging.getLogger(logger_name)
