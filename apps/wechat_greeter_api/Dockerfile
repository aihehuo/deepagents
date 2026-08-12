# WeChat Greeter API (UC-35)
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app/libs/deepagents:/app/libs" \
    HOME=/home/appuser

RUN groupadd -r appuser -g 1000 && \
    useradd -r -g appuser -u 1000 -d /home/appuser -s /bin/bash appuser && \
    mkdir -p /home/appuser/.deepagents && \
    chown -R appuser:appuser /app /home/appuser

COPY apps/wechat_greeter_api/requirements.txt /app/apps/wechat_greeter_api/requirements.txt
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r /app/apps/wechat_greeter_api/requirements.txt

COPY libs/deepagents /app/libs/deepagents
COPY libs/wechat_greeter /app/libs/wechat_greeter
COPY apps/wechat_greeter_api /app/apps/wechat_greeter_api
# /call_async imports the task to enqueue it on the configured Celery app.
COPY apps/wechat_greeter_worker /app/apps/wechat_greeter_worker

RUN chown -R appuser:appuser /app /home/appuser
USER 1000:1000

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import http.client; c=http.client.HTTPConnection('localhost',8005); c.request('GET','/healthz'); r=c.getresponse(); raise SystemExit(0 if r.status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "apps.wechat_greeter_api.main:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "1", "--loop", "asyncio", "--http", "h11"]
