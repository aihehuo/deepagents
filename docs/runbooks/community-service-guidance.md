# Reviewed community service guidance

Implemented 2026-10-09; not deployed in this change. Internal runbook.

The trusted asynchronous WeCom request may carry `wecom_service_guidance`. `app/community_guidance.py` validates its exact schema, 6000-byte limit, customer number, positive version IDs, approval, expiry and personal conversation/greeting action. Public metadata cannot inject it. Missing/null guidance preserves existing behavior.

`execute_wecom_run` includes the bounded snapshot in the existing reply prompt. It does not authorize search, profile writes, sending or payments. Current customer requests/refusals take precedence; inferred summary items are not verified facts. Stale summary is empty. Long-term revenue assumptions are excluded by the host allowlist. Group conversations, introductions and manual skills do not receive individual guidance.

Deploy the group-agent-api and group-agent-worker images before enabling the Micro producer flag. Production Dockerfiles COPY the application; rebuild both images. Beat uses the same worker image and should follow that version. No requirements changed. Micro controls `WECOM_SERVICE_GUIDANCE_ENABLED`; no new Python-side secret is needed.

Regression coverage: `test_community_guidance.py`, `test_wecom_manual.py`, `test_wecom_metadata.py` (48 tests passed). Tests mock models and external services; production authorization/HTTP acceptance remains a deployment check.

Full Chinese implementation/deployment guide: `/Users/yc/workspace/aihehuo/aihehuo_total/docs/design/community-manager-staff-review-implementation.md`.
