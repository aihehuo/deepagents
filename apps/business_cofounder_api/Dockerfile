# Business Co-Founder API Server Dockerfile
#
# Mimics BPGenerationAgent container conventions (slim Debian base, pip installs, uvicorn entrypoint),
# but uses Python 3.11 because `deepagents` requires Python >= 3.11.
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11-slim

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app/libs/deepagents:/app/libs/deepagents-cli"

# System deps
# NOTE: Avoid apt-get in this image because some environments cannot validate Debian repo signatures
# (e.g. network MITM / restricted corporate mirrors), which breaks builds.
# Current requirements are pure-Python, so we don't need gcc/g++ here.

# Minimal runtime deps for the API layer (FastAPI/Uvicorn + optional langdetect).
COPY apps/business_cofounder_api/requirements.txt /app/apps/business_cofounder_api/requirements.txt

RUN pip install -r /app/apps/business_cofounder_api/requirements.txt

# Copy deepagents + deepagents-cli source (imported via PYTHONPATH; no pip editable install)
COPY libs/deepagents /app/libs/deepagents
COPY libs/deepagents-cli /app/libs/deepagents-cli

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


