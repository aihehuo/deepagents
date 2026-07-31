# Group Agent Celery Worker (REQ-032-FIX1)
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app/libs/deepagents" \
    HOME=/home/celery

RUN groupadd -r celery -g 1000 && \
    useradd -r -g celery -u 1000 -d /home/celery -s /bin/bash celery && \
    mkdir -p /home/celery/.deepagents/group_agent_api && \
    chown -R celery:celery /home/celery

COPY apps/group_agent_api/requirements.txt /app/apps/group_agent_api/requirements.txt
COPY apps/group_agent_worker/requirements.txt /app/apps/group_agent_worker/requirements.txt

RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r /app/apps/group_agent_api/requirements.txt \
    -r /app/apps/group_agent_worker/requirements.txt

COPY libs/deepagents /app/libs/deepagents
COPY apps/group_agent_api /app/apps/group_agent_api
COPY apps/group_agent_worker /app/apps/group_agent_worker

COPY apps/group_agent_worker/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh && chown -R celery:celery /app /home/celery

# FIX1 BLOCKER-5: run as non-root celery user (no privilege drop helper).
USER celery

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "from apps.group_agent_worker.runtime import worker_readiness; import sys; r=worker_readiness(); sys.exit(0 if r.get('status')=='ready' else 1)"

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["celery", "-A", "apps.group_agent_worker.celery_app", "worker", "--loglevel=info", "--pool=prefork", "--concurrency=1"]
