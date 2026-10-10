"""Bounded natural-dialogue manager with scoped business tools and persistent memory."""
import json
import os
import time
import uuid
from pathlib import Path
from typing import Callable
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from apps.community_manager.model import draft
from apps.community_manager.tool_schema import TOOLS

SKILL = Path(__file__).parent / "skills" / "operations" / "SKILL.md"


def client() -> ChatOpenAI:
    """Create the independently budgeted manager model, never the customer model."""
    names = ["COMMUNITY_MANAGER_MODEL", "COMMUNITY_MANAGER_MODEL_API_KEY", "COMMUNITY_MANAGER_MODEL_BASE_URL"]
    values = [os.environ.get(n, "") for n in names]
    if not all(values):
        raise ValueError("manager_model_unconfigured")
    return ChatOpenAI(model=values[0], api_key=values[1], base_url=values[2], timeout=30,
                      max_retries=0, max_tokens=2500, temperature=0)


def invoke_tool(call: Callable[..., dict], name: str, args: dict, call_id: str) -> dict:
    """Invoke only fixed business capabilities; draft synthesis is scoped to one user.

    Args:
        call: Run-bound RPC adapter.
        name: Registered tool name.
        args: Structured model arguments.
        call_id: Stable operation key, reused for all writes of this tool invocation.
    """
    allowed = {t['function']['name']: t['function']['parameters'] for t in TOOLS}
    if name not in allowed or not isinstance(args, dict):
        raise ValueError("manager_tool_forbidden")
    schema = allowed[name]
    if set(args) - set(schema['properties']) or set(schema['required']) - set(args):
        raise ValueError("manager_tool_arguments_invalid")
    if name == 'search_public_users':
        from apps.community_manager.public_users import candidate_numbers
        numbers = candidate_numbers(args)
        return call('tool', name='filter_public_candidates', arguments={'numbers': numbers}, call_id=call_id)
    if name != 'draft_guidance':
        return call('tool', name=name, arguments=args, call_id=call_id)
    sources = call('tool', name='get_guidance_sources', arguments=args, call_id=call_id + '-sources')
    if sources.get('tool_error'):
        return sources
    strategy = sources['strategy']
    if sources.get('blocked_reason'):
        return {'drafted': False, 'reason': sources['blocked_reason']}
    if strategy.get('status') in {'paused', 'revoked'}:
        return {'drafted': False, 'reason': 'user_paused'}
    if sources.get('current_source_draft'):
        return {'drafted': False, 'reason': 'same_source_draft_exists'}
    if not sources['allowed_evidence']:
        return {'drafted': False, 'reason': 'no_verified_evidence'}
    output, model = draft(sources)
    context = sources['context']
    output.update(source_fingerprint=context['source_fingerprint'], context_revision=context['context_revision'],
                  strategy_revision=strategy['revision'], base_active_version_id=(strategy.get('current_version') or {}).get('id'), model_id=model)
    return call('tool', name='save_guidance_draft', arguments={'number': args['number'], 'output': output}, call_id=call_id + '-draft')


def execute(started: dict, call: Callable[..., dict]) -> tuple[str, str]:
    """Run a bounded model/tool loop for an employee message or a backend event.

    Args:
        started: Micro-issued execution context, trigger and employee-safe history.
        call: Adapter bound to the run and execution token; cannot change employee identity.
    """
    messages = [SystemMessage(content=SKILL.read_text(encoding='utf-8') +
        "\n工具错误只说明本次未完成，不泄露系统异常。把权限拒绝作为边界，不用其他工具绕过。"
        "最终用中文自然回复，不输出任务版本暗语。每次写入以前读取当前版本；模型不补造业务执行结果。"
        "save_plan.body 必须有 objective,selection_criteria,principles,metrics,capacity,starts_at,expires_at，时间用ISO8601。"
        "随访 need_key 使用 profile/contact 或已读取需求ID；reason基于 evidence。通知需要保存安排后再调用 notify_staff。"
        "自动触发先聚焦 trigger.user_number，不因一个事件批量通知所有用户；最后总结仅内部留存。"
        "若非 global_write，不修改整体目标，记录员工建议并解释可推进的范围。")]
    # This history is already employee-scoped in Micro; no global model transcripts are mixed in.
    for row in started.get('history', [])[-12:]:
        if row['role'] == 'staff':
            messages.append(HumanMessage(content=row['text'][:2000]))
        else:
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=row['text'][:2000]))
    trigger = {k: v for k, v in started.items() if k in {'text', 'trigger', 'global_write', 'model_mode'}}
    messages.append(HumanMessage(content=json.dumps(trigger, ensure_ascii=False)))
    model = client().bind_tools(TOOLS)
    deadline = time.monotonic() + 170
    used = 0; bytes_used = sum(len(str(m.content).encode()) for m in messages)
    for _ in range(6):
        if time.monotonic() >= deadline:
            raise ValueError('manager_budget_exceeded')
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            if not isinstance(response.content, str) or not response.content.strip():
                raise ValueError('manager_output_invalid')
            return response.content, os.environ['COMMUNITY_MANAGER_MODEL']
        for tool in response.tool_calls:
            used += 1
            if used > 16 or time.monotonic() >= deadline:
                raise ValueError('manager_budget_exceeded')
            operation = str(uuid.uuid4())
            try:
                result = invoke_tool(call, tool['name'], tool['args'], operation)
            except ValueError:
                result = {'error': 'invalid_tool_request'}
            # Unknown writes/network errors abort the run instead of generating a new operation ID.
            encoded = json.dumps(result, ensure_ascii=False)
            bytes_used += len(encoded.encode())
            if bytes_used > 120_000:
                raise ValueError('manager_context_budget_exceeded')
            messages.append(ToolMessage(content=encoded, tool_call_id=tool['id']))
    raise ValueError('manager_step_budget_exceeded')
