# Business Co-Founder API Server Dockerfile
#
# Mimics BPGenerationAgent container conventions (slim Debian base, pip installs, uvicorn entrypoint),
# but uses Python 3.11 because `deepagents` requires Python >= 3.11.
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11-slim

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app/libs/deepagents:/app/libs/deepagents-cli"

# System deps (kept consistent with BPGenerationAgent; safe default)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for better caching
COPY libs/deepagents/pyproject.toml libs/deepagents/uv.lock /app/libs/deepagents/
COPY libs/deepagents-cli/pyproject.toml libs/deepagents-cli/uv.lock /app/libs/deepagents-cli/

# Minimal runtime deps for the API layer (FastAPI/Uvicorn + optional langdetect).
COPY apps/business_cofounder_api/requirements.txt /app/apps/business_cofounder_api/requirements.txt

RUN pip install --no-cache-dir -r /app/apps/business_cofounder_api/requirements.txt

# Install deepagents + deepagents-cli from local source
COPY libs/deepagents /app/libs/deepagents
COPY libs/deepagents-cli /app/libs/deepagents-cli
RUN pip install --no-cache-dir -e /app/libs/deepagents -e /app/libs/deepagents-cli

# Copy application code
COPY apps/business_cofounder_api /app/apps/business_cofounder_api

# Expose port (matches dev default)
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import http.client; conn = http.client.HTTPConnection('localhost', 8001); conn.request('GET', '/health'); r = conn.getresponse(); exit(0 if r.status == 200 else 1)" || exit 1

# Run the FastAPI app (pure-python server stack to reduce segfault risk):
# - loop: asyncio (avoid uvloop)
# - http: h11 (avoid httptools)
CMD ["python", "-m", "uvicorn", "apps.business_cofounder_api.app:app", "--host", "0.0.0.0", "--port", "8001", "--loop", "asyncio", "--http", "h11"]


