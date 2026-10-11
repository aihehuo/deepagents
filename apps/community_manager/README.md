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

All enterprise employees converse naturally in the configured application. There are no entry codes, claim commands or task version phrases. A dedicated tool-calling model loads `skills/operations/SKILL.md`, reads scoped facts and persistent memory, and updates plans, follow-ups and existing guidance drafts. Guidance still requires human review. The manager does not send customer messages or add contacts.

Automatic runs receive transactional Micro and OnePenny events through the backend ledger. Beat drains platform events and calls Micro's bounded recovery/reconciliation tick. Source events are coalesced per user and deduplicated; self-origin events cannot recursively drive guidance. Plain-text employee reminders have trusted recipient resolution, source checks, work hours and cooldown. Unknown delivery is not automatically retried.

Observation is controlled by Micro's explicit numbers/platform scope. Autonomous runs and configured operators may read scoped private data; ordinary employees require their live customer relationship or trusted assignment. Public search returns candidate numbers to Micro for scope filtering before the model sees details. No arbitrary SQL, shell, URL or recipient tool is exposed.

Model execution is bounded to six tool rounds, sixteen model tool requests and a context byte budget. All model settings are independent `COMMUNITY_MANAGER_MODEL*` variables; no customer model configuration is overwritten. `/ready` checks model configuration presence, broker and service credentials, not a real provider response.

Upward reports are produced by the independent customer assessment worker and validated by Micro. Customer replies consume the saved brief without generating a second assessment or report.

Tests (root runtime already installed):

```sh
uv run --no-project --python .venv/bin/python python -m pytest tests/unit_tests/community_manager/test_manager.py -q
```

Full internal contract and publication sequence live in the aggregate repository at `aihehuo_total/docs/design/community-manager-natural-v1-implementation.md`. Both Micro entry and proactive reminder flags default off. Real model tools, WeCom delivery, contacts and host resource isolation require a production canary after deployment.

## Customer-first background pipeline

Enable Micro `COMMUNITY_CUSTOMER_ASSESSMENT_ENABLED=1` only after migration `20261010203000` and the updated independent image are ready. Routine source events then update a platform-user dossier and enqueue `/customer/assess_async` on `community_customer_assessment_v1`. A separate `community-customer-worker` handles this queue with explicit `COMMUNITY_CUSTOMER_MODEL*` configuration. It calls no manager tools, emits no customer reply and sends no employee notification. It reuses the pure journey contract, which is explicitly copied into this image. No existing customer deployment is rebuilt by this build script.

Micro validates current sources and journey evidence, then persists meaningful reports. Only those reports enter the manager queue; timer follow-ups and employee questions can still invoke the manager directly. Manager tools `get_customer_dossier` / `get_customer_reports` are individually authorized. Existing dossiers attach to real WeCom conversations by platform user ID; no synthetic conversations are created for background assessment.

Readiness `/customer/ready` verifies customer model configuration presence and broker, not actual provider availability. The Compose project now includes a fourth, separate customer assessment worker; deployment capacity needs checking. Turn the new pipeline off before reverting an image. Previously saved in-conversation briefs remain readable for continuity; new replies do not publish them.

## Unified customer brief updates

Sidebar refresh, committed customer submissions (including record-only), and manager refresh requests retain the existing Micro API and durable `wecom_service_journey_refreshes` status records. Their Sidekiq job now links to a `community_customer_assessments` run and admits it through `/customer/assess_async`; it never calls `group-agent-api` for synthesis. Platform events use the same assessment queue and model. Identical source fingerprints share an active assessment under the user row lock. The signed result callback validates live sources, completes linked sidebar requests, saves the dossier and conditionally reports to the manager. Newer requests supersede older sidebar results; source changes prevent stale dossier publication. Uncertain admission retains its durable run link for recovery; failures and the ten-minute deadline are visible in sidebar status.

`group-agent-api` generates replies and executes existing customer skills. It reads the saved brief to guide conversation and does not generate or save another brief. Its former `/wecom/service_journey` route authenticates and returns 410 without loading a model. The dedicated customer worker performs synthesis and evaluation only; it sends neither customer messages nor staff notifications.

Release order: verify the existing customer worker and `/customer/ready`, required Micro flags (`COMMUNITY_MANAGER_ENABLED`, `COMMUNITY_MANAGER_EVENTS_ENABLED`, `COMMUNITY_CUSTOMER_ASSESSMENT_ENABLED`, `WECOM_SERVICE_JOURNEY_ENABLED`) and the signed admission configuration. Deploy Micro web and Sidekiq first; verify a sidebar refresh completes through the customer queue. Drain pre-release journey jobs before releasing the group-agent image that retires the old route. No new migration, frontend release or manager worker image is needed for this refactor. Rollback restores the older group-agent route first, then reverts Micro. Tests are offline; production canary must verify actual model completion and sidebar display.
