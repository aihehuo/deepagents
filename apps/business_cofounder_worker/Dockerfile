# Business Co-Founder Worker - Celery Worker Container
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11

WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app/libs/deepagents:/app/libs/deepagents-cli"

# Install Python dependencies
COPY apps/business_cofounder_worker/requirements.txt /app/apps/business_cofounder_worker/requirements.txt
RUN pip install --no-cache-dir -r /app/apps/business_cofounder_worker/requirements.txt

# Copy deepagents + deepagents-cli source (imported via PYTHONPATH; no pip editable install)
COPY libs/deepagents /app/libs/deepagents
COPY libs/deepagents-cli /app/libs/deepagents-cli

# Copy application code
COPY apps/business_cofounder_worker /app/apps/business_cofounder_worker
COPY apps/business_cofounder_api/agent_factory.py /app/apps/business_cofounder_api/agent_factory.py
COPY apps/business_cofounder_api/checkpointer.py /app/apps/business_cofounder_api/checkpointer.py
COPY apps/business_cofounder_api/docs_backend.py /app/apps/business_cofounder_api/docs_backend.py

# Create necessary directories
RUN mkdir -p /app/apps/business_cofounder_api

# Create a non-root user for Celery worker
# Use UID 1000 (common non-root UID) to match typical host user IDs
RUN groupadd -r celery -g 1000 && \
    useradd -r -g celery -u 1000 -d /home/celery -s /bin/bash celery && \
    mkdir -p /home/celery && \
    mkdir -p /home/celery/.deepagents && \
    chown -R celery:celery /app /home/celery

# Copy entrypoint script to fix permissions at startup
COPY apps/business_cofounder_worker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Note: The entrypoint script will ensure .deepagents/business_cofounder_api exists
# and has correct permissions, even if the volume is mounted as root.

# Health check (simple Python check, run as celery user)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

# Use entrypoint to fix permissions before switching to celery user
# The entrypoint runs as root to fix permissions, then uses gosu to switch to celery user
ENTRYPOINT ["/docker-entrypoint.sh"]

# Keep as root for entrypoint to work, entrypoint will switch to celery user
# USER celery

# Run Celery worker as non-root user
# Use --pool=solo for single-threaded execution (avoids thread creation issues)
# Use --pool=prefork for multi-process (if threading works)
# Note: No need for --uid/--gid since we're already running as celery user via USER directive
CMD ["celery", "-A", "apps.business_cofounder_worker.celery_app", "worker", "--loglevel=info", "--pool=prefork"]

