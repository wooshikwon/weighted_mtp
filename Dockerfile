FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Dependencies (캐시 활용: 의존성 레이어를 소스보다 먼저 설치)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Project install (editable, src/ 필요)
RUN uv sync --frozen

# Environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Storage: /storage를 마운트하면 storage/ 경로로 접근 가능 (심볼릭 링크)
VOLUME ["/storage"]
RUN ln -s /storage /app/storage

# Default command
CMD ["python", "-c", "print('Weighted MTP container ready. Use: docker run --gpus all -v ./storage:/storage weighted-mtp:latest python src/weighted_mtp/pipelines/run_baseline.py --config configs/local/baseline_local.yaml')"]
