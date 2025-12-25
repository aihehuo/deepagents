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

# Create a non-root user for running the application
# Use UID 1000 (common non-root UID) to match typical host user IDs
RUN groupadd -r appuser -g 1000 && \
    useradd -r -g appuser -u 1000 -d /home/appuser -s /bin/bash appuser && \
    mkdir -p /home/appuser && \
    mkdir -p /home/appuser/.deepagents && \
    chown -R appuser:appuser /app /home/appuser

# Minimal runtime deps for the API layer (FastAPI/Uvicorn + optional langdetect).
COPY apps/business_cofounder_api/requirements.txt /app/apps/business_cofounder_api/requirements.txt

RUN pip install -r /app/apps/business_cofounder_api/requirements.txt

# Copy deepagents + deepagents-cli source (imported via PYTHONPATH; no pip editable install)
COPY libs/deepagents /app/libs/deepagents
COPY libs/deepagents-cli /app/libs/deepagents-cli

# Copy application code
COPY apps/business_cofounder_api /app/apps/business_cofounder_api

# Copy and set up entrypoint script
COPY apps/business_cofounder_api/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Expose port (matches dev default)
EXPOSE 8001

# Health check (run as appuser via entrypoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import http.client; conn = http.client.HTTPConnection('localhost', 8001); conn.request('GET', '/health'); r = conn.getresponse(); exit(0 if r.status == 200 else 1)" || exit 1

# Use entrypoint to ensure directories exist before running the app
# Note: With --user 1000:1000 in docker run, the container starts as UID 1000 (non-root)
# The entrypoint runs as UID 1000 and just ensures directories exist
# No user switching is needed since we're already running as the correct user
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run the FastAPI app (pure-python server stack to reduce segfault risk):
# - loop: asyncio (avoid uvloop)
# - http: h11 (avoid httptools)
CMD ["python", "-m", "uvicorn", "apps.business_cofounder_api.app:app", "--host", "0.0.0.0", "--port", "8001", "--loop", "asyncio", "--http", "h11"]


