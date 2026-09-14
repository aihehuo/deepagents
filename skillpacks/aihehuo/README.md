# 爱合伙可组合技能包

六个技能的源码位于 `packs/`，名称和资源范围由 `catalog.json` 管理。
画像写入能力在独立的 [写技能包](../aihehuo_write/README.md) 中，单独安装、配置和鉴权，不会随六个公网读包的 `--all` 安装一起授予。
`client.py` 统一维护 HTTPS 查询、参数校验、分页参数和限流错误处理。
安装时生成各包的 `run.py`，复制客户端到包内；每个安装后的目录都能单独复制、分发和运行，互不依赖。
包使用标准库，不需要安装新的运行依赖；LangChain 接入器使用项目已有依赖。

```text
skillpacks/aihehuo/
  catalog.json           六个技能的可选清单
  client.py              统一维护的接口客户端
  manage.py              选择安装、预览及冲突检测
  integration.py         可选的智能体查询工具
  packs/<name>/SKILL.md  各任务的触发条件与工作流程

安装目录/<name>/
  SKILL.md
  run.py
  client.py
  package.json
```

## 选择安装

以下命令在仓库根目录运行，使用 SDK 的现有 uv 环境，不触发隐式依赖安装。

```bash
uv run --no-sync --project libs/deepagents python skillpacks/aihehuo/manage.py list

# 单个或组合安装，目标应是该智能体真正加载的技能源目录
uv run --no-sync --project libs/deepagents python skillpacks/aihehuo/manage.py install \
  get-ai-cofounders get-ai-communities --target /tmp/agent-a/skills

# 安装全部六个；加 --dry-run 可检查选择与冲突而不写文件
uv run --no-sync --project libs/deepagents python skillpacks/aihehuo/manage.py install \
  --all --target /tmp/agent-b/skills --dry-run
```

重复安装相同内容不会覆盖文件；任何同名内容差异都拒绝覆盖。升级时先安装到新的源目录，再切换智能体配置。
各包的发布是完整目录重命名；组合安装不是跨目录事务，磁盘等异常可能留下已经成功安装的包，可重新执行命令。
没有自动安装到所有智能体、创建后台定时任务或改变线上服务。

## 与 apps 中的智能体连接

安装文件、加载技能和授予查询工具是三个独立步骤。加载的 `skills` 路径属于智能体 backend；安装目标属于实际磁盘。
工具与技能范围应匹配；只复制文件不能让没有加载器或执行工具的智能体获得查询能力。

| 智能体 | 技能安装目标与接入方式 |
| --- | --- |
| `wu_tanchang_api` | 安装到所选源工作区的 `skills/local/`。现有工作区部署流程将 `get-ai-` 前缀包复制到运行目录的 `skills/local_aihehuo/`，由爱合伙子智能体加载并使用现有 `run_aihehuo_skill_script`。按现有工作区部署流程刷新运行副本。 |
| `business_cofounder_api` 的 business/expert | 现有加载源是各自 backend 根下的 `/skills/`；business 磁盘根为 `~/.deepagents/business_cofounder_api`，expert 的根应以创建时配置为准。安装到对应磁盘目录并提供下述查询工具。 |
| `group_agent_api` | 将选中的包安装到成员 runtime 的 `skills/aihehuo/`，设置 `GROUP_AGENT_READ_MODE=public_skills` 并重启 API 和 worker。成员读取使用公网查询工具及 `/skills/aihehuo/` 技能源；admin/cafe 不加载这些包。 |
| `wechat_greeter_api` 及其他智能体 | 检查该角色的 backend 根、加载器和工具策略，再使用相同接入方式；本包不替它们自动改变职责。 |

没有脚本执行工具的智能体可以使用可选接入器，示例中的 `runtime` 应取智能体已有运行根：

```python
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from skillpacks.aihehuo.integration import create_aihehuo_tool

runtime = Path("/tmp/agent-a")
backend = FilesystemBackend(root_dir=str(runtime), virtual_mode=True)
query_tool = create_aihehuo_tool(skills_dir=runtime / "skills")

# 在已有创建调用中合并 skills、tools；保留该角色已有的工具、中间件及其他参数。
# query_aihehuo_public_data 只暴露已安装包所需的资源。
agent = create_deep_agent(
    model=model,  # 使用应用现有模型
    backend=backend,
    skills=["/skills/"],
    tools=[query_tool],
)
```

已显式添加 `SkillsMiddleware` 的智能体可继续使用原加载器，只增加查询工具。
接入器名称为 `query_aihehuo_public_data`，技能可将其 `--resource/--keyword/--q/--page/--per/--created-since/--event-id/--format` 参数映射为同名下划线工具参数（--format 对应 output_format）。
接入器在 JSON 中补充 `_source_url`，Markdown 前补充实际官方查询接口链接，作为条目缺少 url 时的来源；保留原 data、meta 和条目字段。这是工具添加的检索信息，不是接口原始字段；独立 CLI 仍返回原始响应。
修改安装选择后重建工具及智能体，以刷新能力范围。

## 群智能体切换与回归（第四步）

仅安装需要的包；安装目录内的包共同决定可查询资源。示例路径应替换为实际 `GROUP_AGENT_RUNTIME_DIR`：

```bash
uv run --active --no-sync --project libs/deepagents python skillpacks/aihehuo/manage.py install \
  get-ai-cofounders get-ai-communities --target /tmp/group-agent/skills/aihehuo
```

将该目录作为成员运行根的一部分，配置 `GROUP_AGENT_READ_MODE=public_skills`，然后按现有部署流程重启 API 和 worker。容器镜像包含接入器源码；安装后的目录仍需放在持久化运行卷中。本步骤不自动安装包或部署服务。

公网模式中，允许读取的成员图加载已安装的技能和 `query_aihehuo_public_data`，移除 `search_candidates`；禁止读取的图不加载这些技能和工具。写入工具由可信会话的写权限单独选择。成员入口仍沿用群成员权限；匿名公开读取使用独立的可选入口，见下文。

缺少包、损坏 manifest 或未知模式会导致启动失败，查询失败不会回退到生产匹配接口。回退时设置 `GROUP_AGENT_READ_MODE=legacy` 并重启 API 和 worker，无需删除已安装文件。

公开资料通过对话正文及来源链接展示，不填充原有群内候选人、匹配状态和 @ 邀请字段。HTTP 身份校验、成员验证、画像保存、管理员工具和异步结果回调仍沿用现有流程，因此本阶段减少的是成员候选搜索对生产匹配接口的访问。

离线回归：

```bash
UV_CACHE_DIR=/tmp/deepagents-uv-cache uv run --active --no-sync --project libs/deepagents python -m pytest -q \
  tests/unit_tests/group_agent_api tests/unit_tests/skillpacks/aihehuo \
  tests/test_group_agent_api.py tests/test_group_agent_req009_async.py
```

## 匿名只读入口

`POST /chat/anonymous` 为未登录访客提供单轮公开信息对话，默认关闭。安装需要的读包到 runtime 的 `skills/aihehuo/` 后，设置 `GROUP_AGENT_ANONYMOUS_ENABLED=1` 并按现有部署流程重启 API。

该入口独立于成员的 `GROUP_AGENT_READ_MODE` 和写入模式，只加载已安装的公网读包。没有画像、文件系统、执行脚本、子智能体、管理员或生产匹配工具，没有持久化检查点。技能说明通过限定名称的 `read_aihehuo_skill` 读取，公网数据通过 `query_aihehuo_public_data` 查询。

请求只允许 `message`，不需要签名身份；不接受 user_id、group_id、metadata、membership、conversation_id 或写权限等字段。身份 headers 不参与匿名会话，不能用它们访问本人或他人的画像。响应只有服务器生成的 request_id、reply 和固定为 false 的 can_write_profile。每次请求没有前一次历史；登录不会把匿名请求自动合并到本人画像或历史。

```bash
curl -X POST http://localhost:8001/chat/anonymous \
  -H 'Content-Type: application/json' \
  -d '{"message":"请查询爱合伙公开FAQ，介绍如何寻找创业合伙人并给出来源链接。"}'
```

默认最多 2 个匿名对话同时执行，每个对话最多 3 次查询尝试、45 秒，文本最多 2000 字符。可通过 `GROUP_AGENT_ANONYMOUS_MAX_ACTIVE`（1–10）、`GROUP_AGENT_ANONYMOUS_MAX_QUERIES`（1–5）、`GROUP_AGENT_ANONYMOUS_TIMEOUT_S`（1–120）调整；匿名请求也计入应用现有总并发上限。

关闭时返回 404；启用但未配置独立图时返回 503；并发超限返回 429；超时返回 504。超时、取消和异常都会释放请求槽位。启用时缺少读包或配置无效导致启动失败，不回退到成员图。回退只需设置 `GROUP_AGENT_ANONYMOUS_ENABLED=0` 并重启，原成员接口仍使用原可信身份校验。

## 验证

公网验收使用显式开关，与离线单元测试分开。运行命令、已发现的响应差异及真实模型配置要求见 [公网验收记录](../../docs/runbooks/aihehuo-public-skills-live-acceptance.md)。

```bash
uv run --no-sync --project libs/deepagents python -m pytest -q \
  tests/unit_tests/skillpacks/aihehuo --confcutdir=tests/unit_tests/skillpacks/aihehuo --disable-socket --allow-unix-socket
```

接口规范：<https://www.aihehuo.com/skill.md>、<https://www.aihehuo.com/llms.txt>、<https://www.aihehuo.com/ai-index.json>。
接口数据量与字段可变，周报最多15篇；完整周报需要补充博客分页。
