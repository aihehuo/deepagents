# Wu Tanchang API Server Dockerfile
#
# Modeled after Business Co-Founder API Dockerfile:
# - slim Debian base, pip installs, uvicorn entrypoint
# - Python 3.11 because `deepagents` requires Python >= 3.11
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11-slim

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app/libs/deepagents:/app/libs/cli"

# System deps
# NOTE: Avoid apt-get in this image because some environments cannot validate Debian repo signatures
# (e.g. network MITM / restricted corporate mirrors), which breaks builds.
# Current requirements are pure-Python, so we don't need gcc/g++ here.

# Create a non-root user for running the application
RUN groupadd -r appuser -g 1000 && \
    useradd -r -g appuser -u 1000 -d /home/appuser -s /bin/bash appuser && \
    mkdir -p /home/appuser && \
    mkdir -p /home/appuser/.deepagents && \
    chown -R appuser:appuser /app /home/appuser

# Minimal runtime deps for the API layer
COPY apps/wu_tanchang_api/requirements.txt /app/apps/wu_tanchang_api/requirements.txt

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/apps/wu_tanchang_api/requirements.txt

# Copy deepagents + cli source (imported via PYTHONPATH; no pip editable install)
COPY libs/deepagents /app/libs/deepagents
COPY libs/cli /app/libs/cli

# Copy application code
COPY apps/wu_tanchang_api /app/apps/wu_tanchang_api

# Copy and set up entrypoint script
COPY apps/wu_tanchang_api/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Expose port (container internal, matching CONTAINER_PORT default in shared scripts)
EXPOSE 8001

# Health check (run as appuser via entrypoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import http.client; conn = http.client.HTTPConnection('localhost', 8001); conn.request('GET', '/health'); r = conn.getresponse(); exit(0 if r.status == 200 else 1)" || exit 1

# Use entrypoint to ensure directories exist before running the app
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run the FastAPI app (pure-python server stack to reduce segfault risk):
# - loop: asyncio (avoid uvloop)
# - http: h11 (avoid httptools)
CMD ["python", "-m", "uvicorn", "apps.wu_tanchang_api.app:app", "--host", "0.0.0.0", "--port", "8001", "--loop", "asyncio", "--http", "h11"]
