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

## Token Plan 首选与统一调用记账

客户评估、修正、经理推理、指导草拟和收尾总结共用 `llm.RoutedModel`。
固定首选 Token Plan / auto，备选沿用各自的 COMMUNITY_CUSTOMER_MODEL* 或 COMMUNITY_MANAGER_MODEL*（当前 qwen-plus）。
工具绑定与 JSON 输出约束同时传给两端；首选只试一次，只有网络/超时与指定HTTP错误才回退。
结构校验失败仍遵循原有一次修正逻辑，修正是一笔新的模型调用，不当作网络重试。
主备在原有单次调用超时预算内执行。内部 `client()` 返回兼容 invoke/bind_tools 的 RoutedModel；业务消息及结果协议不变。

私有 env 新增变量（config.env.example 是变量清单，不包含密钥）：

- COMMUNITY_LLM_PRIMARY_API_KEY：复用 Deep 现有 Token Plan 密钥，禁止写入仓库或日志。
- COMMUNITY_LLM_PRIMARY_BASE_URL：默认 https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
- COMMUNITY_LLM_PRIMARY_MODEL：默认 auto。

新增变量要由四个 manager 项目服务载入。部署脚本在启动服务前运行无网络凭据预检；ready和customer/ready也检查首选配置。
无需新容器、无需新队列。原有两个按量模型密钥保留作备选，不修改其他智能体的模型配置。

用量通过现有签名 RPC 的 usage 动作回传，沿用 Agent Key、HMAC、nonce 与运行 execution_token。
新增动作只写 Micro 的 micro_llm_calls，不创建消息、通知或资料更新，不扣用户credit。
每次 invoke 有独立编号，首选与备选分别记 attempt=1/2；修正和多轮推理也分别记账。
Micro 将编号绑定运行身份，客户归属只从服务端 assessment 记录读取，经理跨用户任务的user_id留空。
节点为 community_manager.reasoning / guidance / summary、community_customer.assessment / repair，OPE Micro组件标签页自动纳入。
缓存和思考是输入/输出子项，不重复计入总量。缺Token不记0，不以模型正文推断用量；实际响应模型和请求模型分开保存。

计量是尽力观测：RPC单次超时2秒、无自动网络重试，发送失败只记脱敏警告，不替代业务回复。
进程被杀或回传中断仍可能漏记，不能当完整财务账单。相同回传不会重复插入或覆盖已完成的记录。
不在提示词、业务结果或日志中附带密钥、运行token、原文或异常message。

发布顺序：Micro先支持usage动作并确认micro_llm_calls已迁移；然后配置manager私有env，再部署本项目API、两个worker和scheduler。
OnePenny门面原样转发签名字节，无需改动。旧Micro会拒绝usage但不会因遥测失败卡住模型；仍应先部署Micro以免漏账。
回退manager镜像可保留这些env变量和账表，已有用户资料与历史用量不删除。

### Bounded single-user automatic events
Automatic mode prepares `get_guidance_sources`, `resolve_owner` (including that user's follow-ups) and `get_service_journey` once through existing authenticated RPC. The model receives a compact evidence snapshot with exact refs and explicit truncation, rather than the global operations skill/history. It can only draft guidance, save follow-ups, or notify staff for the trigger user. Existing Micro authorization, evidence, revision and delivery checks remain authoritative. No new Micro API or migration is needed.

The execution budget remains six model rounds, 18 operations (three prerequisite reads plus at most 15 actions), and 64KB serialized context. A 150-second local deadline leaves room for reporting under the worker deadline. Repeated identical action arguments reuse the same in-run receipt; explicit tool rejection or exhausted budgets remain failures for bounded recovery, never successful completion. A paused/no-current-need snapshot completes without a model call. Staff-directed conversations retain the original workflow and tools. Production cost and task completion rate still require post-release observation; offline orchestration tests do not prove real-model completion.
