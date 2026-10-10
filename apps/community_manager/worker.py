"""Finite manager intents; business mutations require exact staff commands in Micro."""
import re
from apps.community_manager.celery_app import app
from apps.community_manager.protocol import rpc
from apps.community_manager.model import draft, PROMPT_VERSION

HELP = "可发送：今日概览、下一位、查看 创业号、更新指导 创业号、认领 任务号 v版本、开始/等待/稍后/结束/暂停 任务号 v版本 原因。修改任务须明确命令，不根据猜测执行。"
ACTIONS = {"认领": "claim", "开始": "start", "等待": "wait", "稍后": "defer", "结束": "complete", "暂停": "pause"}


def task_copy(row: dict) -> str:
    """Task copy.

    Args:
        row: Host-selected service task.
    """
    user = row.get("user", {})
    return (f"{user.get('name', '')}（创业号 {row['user_number']}）\n企微备注：{user.get('remark') or '未存，请先核对'}\n"
        f"需求：{row.get('need', {}).get('title', row.get('description', ''))}\n{row.get('reason', '')}\n"
        f"下一步：{row.get('suggested_action', '')}\n任务 {row['id']} v{row['revision']} · {row['status']}\n"
        f"认领请发送：认领 {row['id']} v{row['revision']}\n数据时间：{row.get('observed_at', '')}\n"
        + (user.get('detail_url') or '详情请在对应客户侧栏查看。'))


def execute(run_id: str, token: str, text: str) -> tuple[str, str]:
    """Return verified-data prose; only guidance generation invokes the independent model.

    Args:
        run_id: Opaque Micro run identifier.
        token: Micro-issued execution claim.
        text: Authenticated staff command.
    """
    def call(action: str, **values: object) -> dict:
        """Call.

        Args:
            action: Fixed business operation name.
            values: Operation payload validated by Micro.
        """
        return rpc(action, run_id=run_id, token=token, **values)
    if text in {"今日概览", "今天的概览", "概览", "今天进展如何", "今天的进度"}:
        data = call("overview")
        stages = {"founder_profile": "画像待整理", "needs_profile": "需求待明确", "ready": "待推荐", "matching": "搜索/推荐/对接", "paused": "暂停/受限", "waiting_feedback": "等待反馈", "introducing": "对接中", "processing": "客服执行中", "restricted": "身份待核验"}
        lines = [f"近期目标：{data['plan']['objective'][:160]}", f"当前账号可服务 {data['user_count']} 位用户，已记录 {data['need_count']} 个需求；未知数量不等于没有需求。"]
        lines += [f"{stages.get(k, k)}：{v} 位" for k, v in data['stages'].items()]
        lines += ["数据时间：" + data['observed_at'], "发送“下一位”领取建议；暂停或无跟进权限的用户不进入服务队列。"]
        return "\n".join(lines), "deterministic"
    if text in {"下一位", "今天优先服务谁", "接下来服务谁", "下一位是谁", "今天先找谁"}:
        data = call("queue")
        if not data['tasks']:
            return "当前没有到期且权限合格的任务。可能仍在等待反馈、已暂停或需先核验联系权限；不建议重复触达。", "deterministic"
        row = {**data['tasks'][0], "observed_at": data['observed_at']}
        return task_copy(row), "deterministic"
    match = re.fullmatch(r"查看 ([1-9][0-9]{0,14})", text)
    if match:
        data = call("user", number=match[1])
        user = data["user"]
        lines = [f"{user['name']} · {user['user_number']} · 备注：{user['remark'] or '未存'}", f"阶段：{user['phase']} / {user['state']}"]
        lines += [f"需求：{need['title']} · {need['clarity']} · {need.get('progress', {}).get('stage', '待确认')}" for need in user['needs']]
        lines += [f"任务 {t['id']} v{t['revision']} · {t['status']} · 负责人 {t['owner'] or '未分配'}" for t in data['tasks']]
        lines += [user['next_action'] or "先核实当前找人意愿。", user['detail_url'] or "请在对应客户侧栏查看。"]
        return "\n".join(lines)[:650], "deterministic"
    match = re.fullmatch(r"(认领|开始|等待|稍后|结束|暂停) ([1-9][0-9]*) v([0-9]+)(?: (.{1,500}))?", text)
    if match:
        result = call("task_action", task_id=int(match[2]), revision=int(match[3]), task_action=ACTIONS[match[1]], reason=match[4] or "")
        return f"任务 {result['id']} 已更新：{result['status']}，v{result['revision']}，负责人 {result['owner']}。\n下次处理：{result['next_action_at']}。更新的是内部安排，不表示已发送客户消息或合作成功。", "deterministic"
    match = re.fullmatch(r"更新指导 ([1-9][0-9]{0,14})", text)
    if match:
        sources = call("guidance_sources", number=match[1])
        if sources['user']['blocked']:
            return "该用户存在暂停、拒绝或身份限制。请先人工核验，不自动恢复指导。", "deterministic"
        if not sources['allowed_evidence']:
            return "没有可核验的个人证据，暂不能生成新指导；请先在侧栏整理用户信息。", "deterministic"
        output, model = draft(sources)
        context, strategy = sources['context'], sources['strategy']
        output.update(source_fingerprint=context['source_fingerprint'], context_revision=context['context_revision'],
            strategy_revision=strategy['revision'], base_active_version_id=(strategy.get('current_version') or {}).get('id'), model_id=model)
        result = call("guidance_draft", number=match[1], output=output)
        return f"已保存指导草稿 v{result['draft']['version']}。请在该用户侧栏审核接受；当前生效指导未被自动替换。", model
    return HELP, "deterministic"


@app.task(name="community_manager.run")
def run(run_id: str) -> None:
    """Claim a Micro run once; never retry unknown writes as a new business request.

    Args:
        run_id: Opaque Micro run identifier.
    """
    started = rpc("begin", run_id=run_id)
    token = started.get("token")
    if not token:
        return
    try:
        text, model = execute(run_id, token, started['text'])
        output = {"text": text[:650].encode("utf-8")[:2000].decode("utf-8", errors="ignore"), "model_id": model, "prompt_version": PROMPT_VERSION}
    except Exception:
        # Provider errors may contain tokens or user data. Fixed output only.
        output = {"text": "本次经理操作未完成，请核对任务或草稿记录后重新发起；没有向客户发送消息。", "model_id": "unknown", "prompt_version": PROMPT_VERSION, "error_code": "manager_execution_failed"}
    rpc("result", run_id=run_id, token=token, output=output)


@app.task(name="community_manager.tick")
def tick() -> None:
    """Check bounded reminders through the same authoritative business API.
    """
    rpc("tick")
