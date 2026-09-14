# 独立画像写入技能包

`save-ai-user-profile` 是独立的写包，不属于六个公网只读包的 catalog 或 `--all` 安装范围。可单独安装，也可与任意读包组合；安装本身不会授予写权限。

每个安装目录包含 `SKILL.md`、`package.json`、`adapter.py`。接入器可复制到其他智能体，依赖宿主已有的 LangChain 以及可信 `save_group_profile` 保存工具。包内没有凭证、任意 POST 客户端或独立的用户身份参数。

## 群智能体安装与启用

以下命令使用已有 uv 环境；示例路径应替换为实际 `GROUP_AGENT_RUNTIME_DIR`：

```bash
uv run --active --no-sync --project libs/deepagents python -m skillpacks.aihehuo_write.manage \
  --target /tmp/group-agent/skills/aihehuo_write --dry-run

uv run --active --no-sync --project libs/deepagents python -m skillpacks.aihehuo_write.manage \
  --target /tmp/group-agent/skills/aihehuo_write
```

同内容重复安装保持不变；内容冲突或符号链接拒绝覆盖，升级时使用新的源目录。完整包通过目录重命名发布。

`GROUP_AGENT_WRITE_MODE` 与 `GROUP_AGENT_READ_MODE` 独立：

| 写入模式 | 行为 |
| --- | --- |
| `legacy`（默认） | 保留原保存工具和既有权限判断，便于兼容及回退 |
| `skillpack` | 需要安装写包；有写权限的成员图加载写技能和受控保存工具 |
| `disabled` | 成员会话取消画像写权限，成员图移除写工具，强制保存兜底也停止写入 |

启用时设置 `GROUP_AGENT_WRITE_MODE=skillpack`，按现有部署流程重启 API 和 worker，并确保两者的安装目录、配置一致。容器镜像已包含 `skillpacks` 接入代码，安装后的包需要放入持久化运行卷。本步骤不自动部署或修改线上 env。

缺少或损坏写包会导致可写图创建失败；运行中移除包也会阻止写入。回退时改为 `legacy` 并重启，不删除安装文件。写入模式控制成员的 `save_group_profile` 能力，不是其他业务 API 写入权限的总开关。

## 权限与保存边界

成员 HTTP 会话继续使用已有签名身份校验，写权限来自已验证的本人身份；未获得写权限的会话不加载写包。开发 stub 的身份信号仅供离线测试，不代表匿名用户的登录认证。独立的 `/chat/anonymous` 入口可配置开启，默认关闭；匿名图不加载本写包，也不访问用户画像。

模型参数只包含画像内容。身份、run、写权限和来源消息由宿主服务端注入；接入器在每次调用时要求明确的写权限、本人及群身份。宿主仍负责 ID 校验、run 绑定、签名、并发 fencing、远端 acknowledgement 和本地缓存。本包保存、更新的是已有的用户×群三维画像，不新增全站用户 profile 的写入 API。

`created`、`updated`、`idempotent` 沿用原确认路径；`stale_ignored` 保留较新画像，`fence_rejected` 或远端失败返回错误且不写本地缓存。重复请求沿用原 run 的幂等语义，不生成新 run 或另建一套存储。

## 接入其他智能体

宿主加载写技能的 backend 路径，并用 `create_profile_write_tool(skills_dir=..., writer=...)` 包装其可信保存工具。安装后的 `adapter.py` 可作为独立模块加载，无需依赖本仓库代码；但宿主必须先验证登录状态并过滤客户端 metadata，才能注入 `can_write_profile`、`user_id`、`group_id`。

共享模型图时，以可信会话权限选择可写或不可写的图，不把可写图作为未授权会话的默认图。必须保留宿主原有的写入校验；此接入器不替代身份认证或服务器端的写入鉴权。

## 离线回归

```bash
UV_CACHE_DIR=/tmp/deepagents-uv-cache uv run --active --no-sync --project libs/deepagents python -m pytest -q \
  tests/unit_tests/skillpacks/aihehuo_write \
  tests/unit_tests/group_agent_api/test_write_skills.py \
  tests/test_group_agent_profile_client.py
```

覆盖独立安装、读写组合、真实图执行、权限缺失、身份替换、更新、重复 run、远端确认及失败、包损坏和移除、禁写及兜底路径。测试不写生产数据；真实远端写入验收需另用明确的测试账号及 run。
