# WeChat Greeter Celery Worker (UC-35)
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app/libs/deepagents:/app/libs" \
    HOME=/home/celery

RUN groupadd -r celery -g 1000 && \
    useradd -r -g celery -u 1000 -d /home/celery -s /bin/bash celery && \
    mkdir -p /home/celery/.deepagents && \
    chown -R celery:celery /app /home/celery

COPY apps/wechat_greeter_api/requirements.txt /app/apps/wechat_greeter_api/requirements.txt
COPY apps/wechat_greeter_worker/requirements.txt /app/apps/wechat_greeter_worker/requirements.txt
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r /app/apps/wechat_greeter_api/requirements.txt \
    -r /app/apps/wechat_greeter_worker/requirements.txt

COPY libs/deepagents /app/libs/deepagents
COPY libs/wechat_greeter /app/libs/wechat_greeter
COPY apps/wechat_greeter_api /app/apps/wechat_greeter_api
COPY apps/wechat_greeter_worker /app/apps/wechat_greeter_worker

RUN chown -R celery:celery /app /home/celery
USER celery

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD celery -A apps.wechat_greeter_worker.celery_app inspect ping --timeout 5 -d celery@$HOSTNAME | grep -q pong || exit 1

CMD ["celery", "-A", "apps.wechat_greeter_worker.celery_app", "worker", "--loglevel=info", "--pool=prefork", "--concurrency=2", "-Q", "wechat_greeter"]
