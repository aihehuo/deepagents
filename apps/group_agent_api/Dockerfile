# Group Agent API Server Dockerfile
# Python >= 3.11 base image for deepagents + FastAPI group_agent_api
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11-slim

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app/libs/deepagents"

# Create non-root user appuser (UID 1000, GID 1000)
RUN groupadd -r appuser -g 1000 && \
    useradd -r -g appuser -u 1000 -d /home/appuser -s /bin/bash appuser && \
    mkdir -p /home/appuser && \
    mkdir -p /home/appuser/.deepagents && \
    chown -R appuser:appuser /app /home/appuser

# Install requirements
COPY apps/group_agent_api/requirements.txt /app/apps/group_agent_api/requirements.txt

RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r /app/apps/group_agent_api/requirements.txt

# Copy deepagents library
COPY libs/deepagents /app/libs/deepagents

# Copy application code
COPY apps/group_agent_api /app/apps/group_agent_api

# Copy and set up entrypoint script
COPY apps/group_agent_api/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh && chown -R appuser:appuser /app /home/appuser

# Switch to non-root user appuser (1000:1000) by default
USER 1000:1000

# Expose container port
EXPOSE 8001

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import http.client; conn = http.client.HTTPConnection('localhost', 8001); conn.request('GET', '/health'); r = conn.getresponse(); exit(0 if r.status == 200 else 1)" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run Uvicorn with single worker (in-memory nonce cache requires single worker)
CMD ["python", "-m", "uvicorn", "apps.group_agent_api.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--loop", "asyncio", "--http", "h11"]
