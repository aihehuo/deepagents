# Community Manager V1

Internal staff-only manager. Does not send customer messages or approve customer guidance.

- API: `python -m uvicorn apps.community_manager.api:app --host 0.0.0.0 --port 8092`
- Worker: `python -m celery -A apps.community_manager.celery_app:app worker -Q community_manager_v1 --concurrency=1`
- Scheduler: `python -m celery -A apps.community_manager.celery_app:app beat --schedule=/tmp/manager-beat`
- Private configuration: copy `config.env.example`; do not commit actual keys.
- Build: `bash apps/community_manager/build.sh` with an immutable `MANAGER_RUNTIME_IMAGE` and `COMMUNITY_MANAGER_IMAGE_REPOSITORY`.
- Deploy: `bash apps/community_manager/deploy.sh` with `COMMUNITY_MANAGER_IMAGE` and an absolute `COMMUNITY_MANAGER_ENV_FILE`.

The image reuses installed runtime dependencies from a pinned image and copies only this application. If Redis is reachable only on an existing Docker network, set `COMMUNITY_MANAGER_NETWORK` in the deployment shell to include `compose.network.yml`; otherwise configure a Redis address reachable from the default manager network. Only new manager containers join the selected network.

The Compose project, queue, Redis prefix, worker and scheduler are distinct from customer agents. Rollback only targets the `community-manager-v1` project.

The API accepts only an opaque UUID run reference from Micro. Micro persists the actual employee/plan scope after its verified enterprise callback. OnePenny checks Agent Key; Micro additionally checks body HMAC, timestamp and nonce. No caller can supply a new employee scope. No ordinary worker result replaces an approved strategy.

Wire signature: HMAC SHA256 over UTF-8 `community-manager-v1\n{epoch_seconds}\n{nonce_uuid}\n` followed by the **exact request bytes**, using `COMMUNITY_MANAGER_SECRET`. Headers are `X-CM-Time`, `X-CM-Nonce`, `X-CM-Signature`; sixty-second clock window, replay nonce consumed for 120 seconds. For signed run-status GET, the final bytes are the exact path instead of a JSON body.

All enterprise employees enter manager mode directly in the configured application. Commands: `今日概览`, `下一位`, `查看 创业号`, `更新指导 创业号`, `认领/开始/等待/稍后/结束/暂停 任务号 v修订 [原因]`. Unsupported language returns help without guessing a mutation. Only guidance drafting invokes the configured model.

Tests (root runtime already installed):

```sh
uv run --no-project --python .venv/bin/python python -m pytest tests/unit_tests/community_manager/test_manager.py -q
```

Full internal contract and publication sequence live in the aggregate repository at `aihehuo_total/docs/design/community-manager-wecom-v1-implementation.md`. Both Micro entry and proactive reminder flags default off. Actual WeCom cards, contacts and host resource isolation require a production canary after deployment.
