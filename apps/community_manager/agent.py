"""Bounded natural-dialogue manager with scoped business tools and persistent memory."""
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Callable
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from apps.community_manager.llm import RoutedModel, reported_model, routed
from apps.community_manager.model import draft
from apps.community_manager.tool_schema import TOOLS

FAILURE_NOTICE = '本次经理分析未完成，已完成的操作请以记录为准；现有客服仍可继续工作，没有向客户发送消息。'

MAX_TOOL_CALLS = 48
MAX_MODEL_STEPS = 12
MAX_SECONDS = 420
MAX_CONTEXT_BYTES = 240_000

SKILL = Path(__file__).parent / "skills" / "operations" / "SKILL.md"


def client() -> RoutedModel:
    """Create the independently budgeted manager model, never the customer model."""
    names = ["COMMUNITY_MANAGER_MODEL", "COMMUNITY_MANAGER_MODEL_API_KEY", "COMMUNITY_MANAGER_MODEL_BASE_URL"]
    values = [os.environ.get(n, "") for n in names]
    if not all(values):
        raise ValueError("manager_model_unconfigured")
    standby = ChatOpenAI(model=values[0], api_key=values[1], base_url=values[2], timeout=30,
                      max_retries=0, max_tokens=2500, temperature=0)
    return routed(standby, node='manager_reasoning', model=values[0], timeout=30, max_tokens=2500)


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
        return call('tool', name='filter_public_candidates', arguments={'numbers': numbers, 'contact_scope': args.get('contact_scope', 'reachable')}, call_id=call_id)
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


def verified_candidates(messages: list, mode: str = 'reachable') -> list[dict]:
    """Collect only records matching the requested contact scope from search receipts.

    Args:
        messages: Run-local tool receipts; assistant claims are never evidence.
        mode: Explicit reachable or unbound contact scope.
    """
    found = {}
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            data = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if not isinstance(data, dict) or data.get('contact_scope') != mode:
            continue
        for user in data.get('users', []):
            if isinstance(user, dict) and user.get('bound') == ('not_bound' if mode == 'unbound' else 'bound_snapshot') and user.get('user_number'):
                found[str(user['user_number'])] = user
    return list(found.values())


def receipt_summary(messages: list) -> str:
    """Produce a factual fallback without promising further execution.

    Args:
        messages: Current scoped results, excluding remembered assistant text.
    """
    lines = []
    for mode, title in [('reachable', '已加企微候选'), ('unbound', '未加企微、主动联系候选')]:
        users = verified_candidates(messages, mode)
        if not users:
            continue
        lines.append(title + '（仅已读取资料，优先级评估尚未完成）：')
        for index, user in enumerate(users[:10], 1):
            name = str(user.get('display_name') or '未记录姓名')[:80].replace('\n', ' ')
            lines.append(f"{index}. {name}（创业号：{user['user_number']}）")
    if not lines:
        return '尚未完成具体用户名单核验；已完成操作以记录为准，未向客户发送消息。'
    lines.append('逐人关注理由、备注或联系线索及下一步尚未完成核验；本轮已停止，未向客户发送消息。')
    return '\n'.join(lines)



def partial_summary(messages: list, reason: str, deadline: float) -> tuple[str, str]:
    """Summarize verified progress without more tools when a bounded run must stop.

    Args:
        messages: Completed tool results, including refusals for unexecuted calls.
        reason: Fixed budget category, never a provider exception or customer input.
        deadline: Original execution deadline; reserve time for a tool-free summary.
    """
    prefix = '本轮已达到分析上限，已停止新增操作。'
    if time.monotonic() + 30 >= deadline:
        return prefix + '\n' + receipt_summary(messages), reported_model()
    response = client().invoke(messages + [SystemMessage(content=
        '本轮已达到' + reason + '上限。只能总结，不调用任何工具、不执行新的操作。'
        '用中文简短列出已核验的具体候选和依据、实际保存的草稿或安排、尚未完成的事项。'
        '工具报错或未执行不算完成；公开搜索候选不是已经安排服务；未读取历史明确说明。'
        '不得复述系统失败通知，不声称整项请求已完成。') , HumanMessage(content=
        '现在立即交付本轮结果，不再搜索。列出工具实际返回的具体用户，明确已核验与未核验项。禁止写“让我继续搜索”“我将继续”或任何未来操作承诺。')], usage_node='manager_summary')
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError('manager_output_invalid')
    text = response.content
    if getattr(response, 'tool_calls', None) or re.search(r'让我.*(?:搜索|查找|查询)|(?:将|会|现在|接下来|继续|再).{0,8}(?:搜索|查找)', text) or ((verified_candidates(messages) + verified_candidates(messages, 'unbound')) and not any(str(u['user_number']) in text for u in verified_candidates(messages) + verified_candidates(messages, 'unbound'))):
        text = receipt_summary(messages)
    return prefix + '\n' + text, reported_model()


def execute(started: dict, call: Callable[..., dict]) -> tuple[str, str]:
    """Run a bounded model/tool loop for an employee message or a backend event.

    Args:
        started: Micro-issued execution context, trigger and employee-safe history.
        call: Adapter bound to the run and execution token; cannot change employee identity.
    """
    from datetime import datetime, timezone
    current_time = datetime.now(timezone.utc).isoformat()
    messages = [SystemMessage(content='当前UTC时间：' + current_time + '。next_review_at须在当前时间之后、90天内，使用带时区ISO8601。\n' + SKILL.read_text(encoding='utf-8') +
        "\n工具错误只说明本次未完成，不泄露系统异常。只有403才作为权限或运营开关边界，不用其他工具绕过。400/422是参数或证据校验拒绝，不要称为权限不足；基于已读取的真实资料修正参数。409按具体tool_error处理，版本冲突先重新读取版本，源变化重新核验证据；暂停或业务阻塞不能绕过。无法修正时明确报告未保存、未通知。"
        "单轮最多48次工具调用、12轮模型推理。先用最少工具核验目标并选具体候选，避免重复搜索；优先完成用户要求的候选数量；不足时继续分页搜索，不能用未加企微的人补数。逐人指导仍需核验证据，明确未完成项。用户创业号number必须用字符串，不用数据库id代替。"
        "最终用中文自然回复，不输出任务版本暗语。完整回复须UTF8不超过12000字节，系统会分段发送。用紧凑的编号列表覆盖用户要求的全部人数，减少前言，不要遗漏后面的用户。名单每位先调用get_contact_relationships核实remark，bound_snapshot只是绑定状态，绝不能当企微备注；remark为空时写未记录备注。每次写入以前读取当前版本；模型不补造业务执行结果。"
        "save_plan.body 必须有 objective,selection_criteria,principles,metrics,capacity,starts_at,expires_at，时间用ISO8601。"
        "随访 need_key 使用 profile/contact 或已读取需求ID；reason基于 evidence。evidence必须直接复制get_guidance_sources.allowed_evidence每项的ref对象（只含source/id/field/message_index等），不能传整项ref/text/role包装，id和message_index保持整数；不能将简报内引用的字符串当证据对象。只有save_follow_up成功返回id和revision后才调用notify_staff，保存失败先按invalid_fields修正，不得先通知。"
        "自动触发先聚焦 trigger.user_number，不因一个事件批量通知所有用户；最后总结仅内部留存。"
        "若非 global_write，不修改整体目标，记录员工建议并解释可推进的范围。")]
    # This history is already employee-scoped in Micro; no global model transcripts are mixed in.
    for row in started.get('history', [])[-12:]:
        if row['role'] == 'staff':
            messages.append(HumanMessage(content=row['text'][:2000]))
        else:
            if row['text'].strip() == FAILURE_NOTICE:
                continue
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=row['text'][:2000]))
    trigger = {k: v for k, v in started.items() if k in {'text', 'trigger', 'global_write', 'model_mode'}}
    messages.append(HumanMessage(content=json.dumps(trigger, ensure_ascii=False)))
    from apps.community_manager import ranking
    rank_expected = ranking.intent(started)
    rank_receipt = None; rank_contacts = {}; rank_limit = 10; rank_refusals = 0
    if rank_expected:
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone(timedelta(hours=8))).isoformat()
        messages.append(SystemMessage(content='本轮是实时排名请求，必须先查询，不得使用历史名单和历史日期。当前北京时间：' + now + '。本轮校验要求：' + json.dumps(rank_expected, ensure_ascii=False) + '。limit为要求人数；today要求当前北京时间零点active_since；no_city表示移除city。最终姓名、顺序和日期只取本轮回执，不得编造。'))
    base_model = client()
    model = base_model.bind_tools(TOOLS)
    ranking_model = base_model.bind_tools(TOOLS, tool_choice='search_recent_users') if rank_expected else model
    deadline = time.monotonic() + MAX_SECONDS
    outreach_request = bool(re.search(r'未.{0,10}(?:企微|企业微信)|主动.{0,6}(?:联系|添加)|添加.{0,8}微信', started.get('text', '')))
    selection_mode = 'unbound' if outreach_request else 'reachable'
    ten_candidate_request = bool(re.search(r'(?:10|十).{0,6}(?:位|个|名|用户|候选)', started.get('text', '')))
    searches = 0
    used = 0; bytes_used = sum(len(str(m.content).encode()) for m in messages)
    for _ in range(MAX_MODEL_STEPS):
        if time.monotonic() >= deadline:
            return (ranking.render(rank_receipt, rank_contacts, rank_limit), reported_model()) if rank_expected else partial_summary(messages, '时间', deadline)
        response = (ranking_model if rank_expected and rank_receipt is None else model).invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            if rank_expected:
                if rank_receipt is not None:
                    return ranking.render(rank_receipt, rank_contacts, rank_limit), reported_model()
                rank_refusals += 1
                if rank_refusals >= 2:
                    return ranking.FAILURE, reported_model()
                messages.append(SystemMessage(content='本轮尚无有效实时查询回执。禁止复用历史名单或编造搜索结果。下一步必须调用search_recent_users，参数遵循：' + json.dumps(rank_expected, ensure_ascii=False) + '；today表示北京时间今天零点active_since，no_city表示不填city。'))
                continue
            if not isinstance(response.content, str) or not response.content.strip():
                raise ValueError('manager_output_invalid')
            if response.content.strip() == FAILURE_NOTICE:
                messages.append(SystemMessage(content="这是历史系统失败通知，不是当前任务的答案。依据本轮权限和工具重新处理当前请求；仅报告本轮实际核验结果与未完成事项。"))
                continue
            return response.content, reported_model()
        for index, tool in enumerate(response.tool_calls):
            if used >= MAX_TOOL_CALLS or time.monotonic() + 30 >= deadline:
                # Complete the assistant's pending call IDs with explicit non-execution results.
                for pending in response.tool_calls[index:]:
                    messages.append(ToolMessage(content='{"tool_error":"not_executed_budget_limit"}', tool_call_id=pending['id']))
                return (ranking.render(rank_receipt, rank_contacts, rank_limit), reported_model()) if rank_expected else partial_summary(messages, '工具调用或时间', deadline)
            used += 1
            operation = str(uuid.uuid4())
            try:
                if rank_expected and rank_receipt is None and tool['name'] != 'search_recent_users':
                    result = {'tool_error': 'current_ranking_receipt_required', 'note': '先调用search_recent_users取得本轮时间排序回执，再核验候选。'}
                elif tool['name'] in {'search_users', 'search_public_users'} and (ten_candidate_request or outreach_request) and searches >= 3 and len(verified_candidates(messages, selection_mode)) >= 10:
                    result = {'tool_error': 'candidate_selection_required', 'note': '已有至少10位目标联系范围内候选；停止重复搜索，从既有结果选择并核验资料、联系关系和优先级。'}
                else:
                    result = invoke_tool(call, tool['name'], tool['args'], operation)
                    if tool['name'] in {'search_users', 'search_public_users'}:
                        searches += 1
            except ValueError:
                result = {'error': 'invalid_tool_request'}
            if rank_expected and tool['name'] == 'search_recent_users':
                if ranking.valid_receipt(result, tool['args'], rank_expected):
                    rank_receipt = result; rank_limit = tool['args'].get('limit', 10)
                else:
                    result = {'tool_error': 'ranking_receipt_invalid', 'note': '未取得符合本轮地区、联系范围和排序要求的实时回执，不得声称已完成查询。'}
            if rank_expected and tool['name'] == 'get_contact_leads' and isinstance(result, dict):
                rank_contacts[str(tool['args'].get('number'))] = result
            # Unknown writes/network errors abort the run instead of generating a new operation ID.
            encoded = json.dumps(result, ensure_ascii=False)
            bytes_used += len(encoded.encode())
            if bytes_used > MAX_CONTEXT_BYTES:
                return '本轮已达到资料量上限，已停止新增操作；已完成操作以记录为准，剩余用户需分批核验，未向客户发送消息。', reported_model()
            messages.append(ToolMessage(content=encoded, tool_call_id=tool['id']))
        if (ten_candidate_request or outreach_request) and searches >= 3 and len(verified_candidates(messages, selection_mode)) >= 10:
            messages.append(SystemMessage(content='目标联系范围内候选已足够。现在停止搜索，选择所需人数，批量核验资料与联系关系。已加企微读取真实备注，未加企微单列并调用get_contact_leads读取preferred_contact与来源，优先微信号、缺失退到手机号，核验主动联系理由；随后交付完整名单，不能用继续搜索的过程话作答。'))
    return (ranking.render(rank_receipt, rank_contacts, rank_limit), reported_model()) if rank_expected else partial_summary(messages, '推理轮数', deadline)
