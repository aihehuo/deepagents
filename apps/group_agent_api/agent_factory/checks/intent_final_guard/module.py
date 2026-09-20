"""Receipt-backed deterministic guard for user-visible final replies."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from apps.group_agent_api.agent_factory.checks.intent_final_guard.ids import CHECK_ID
from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import (
    PublicReadGround,
)
from apps.group_agent_api.agent_factory.content_quality import (
    looks_like_unauthorized_action_claim,
)
from apps.group_agent_api.agent_factory.intent_routing.canonical import canonical_hash
from apps.group_agent_api.agent_factory.intent_routing.contracts import ReceiptLedgerV1

if TYPE_CHECKING:
    from apps.group_agent_api.agent_factory.intent_routing.execution import (
        IntentExecution,
    )

_logger = logging.getLogger("uvicorn.error")

_QUOTED = re.compile(r'“[^”]*”|「[^」]*」|『[^』]*』|《[^》]*》|"[^"\n]*"|\'[^\'\n]*\'')
_CLAUSE = re.compile(r"[^。！？!?；;，,\n]+")
_FUTURE_QUERY = re.compile(
    r"正在(?:查询|搜索|查找|检索|筛选|匹配|查)|"
    r"(?:马上|立即|现在)(?:为你|帮你)?(?:查询|搜索|查找|检索|筛选|匹配|查)|"
    r"请稍候|稍等|稍后.{0,12}(?:给|返回|提供|发送)|"
    r"后台.{0,16}(?:查询|搜索|匹配|筛选)|"
    r"(?:查询|搜索|匹配)结果.{0,10}(?:稍后|马上|随后)|"
    r"正在(?:保存|更新)(?:画像|资料)?|稍后.{0,8}(?:保存|更新)(?:画像|资料)?"
    r"|(?:我们|我)?(?:来|再来)(?:查|查询|搜索|筛选|匹配)"
)
_SEARCH_DONE = re.compile(
    r"(?:已经|已)(?:完成)?(?:查询|搜索|查找|检索|筛选|匹配|查过|查完)|"
    r"(?:查到|搜到|找到|匹配到)(?:了)?|"
    r"(?:查询|搜索|查找|检索|筛选|匹配)(?:完成|过了|完了)"
)
_FOUND = re.compile(r"(?:查到|搜到|找到|匹配到)(?:了)?")
_PROFILE_DONE = re.compile(
    r"(?:已经|已)(?:成功)?(?:保存|更新|修改).{0,8}(?:画像|资料)|"
    r"(?:画像|资料).{0,8}(?:已经|已)(?:保存|更新|修改)|"
    r"(?:已经|已)(?:成功)?保存(?:好了|完成)?"
)
_EXTERNAL_ACTION = re.compile(
    r"(?:已经|已)?(?:替你|帮你|帮您|帮我)?(?:发送到群|发到群里|发给群友|"
    r"在群里发了|在群里@了|@了对方|通知(?:了)?管理员|联系上对方)|"
    r"(?:对方已回复|对方同意了|对方已接受|已有\d+人触达)"
)
_NEGATED = re.compile(r"(?:没有|并未|未曾|尚未|不是|并非|不能说|不要声称|不代表).{0,14}$")
_NON_CURRENT = re.compile(
    r"(?:上次|上一轮|之前|此前|过去|历史|旧的|如果|假如|假设|例如|比如|"
    r"如何|怎么|怎样|是否|这句话|这个词|状态文案|方法)"
)
_OVERCLAIM = re.compile(r"(?:全部|所有|完整|全量|全网|毫无遗漏|没有遗漏).{0,18}(?:结果|人选|记录|资料)?")
_EXPLICIT_ENTITY = (
    re.compile(r"(?:候选人|人选|伙伴|合伙人)\s*[【\[]\s*([^】\]]{1,64})\s*[】\]]"),
    re.compile(r"(?:候选人|人选|伙伴|合伙人).{0,6}(?:名为|叫)\s*([\w\u3400-\u9fff·]{2,32})"),
)
_SCOPE = re.compile(r"(?:查询范围|本次查询覆盖|查询了)\D{0,8}\d+\s*页")
_URL = re.compile(r"https://[^\s)）\]】>，。；;]+")
# Official evidence URLs belong in receipts / debug traces, never in user reply.
_USER_VISIBLE_SOURCE_URL = re.compile(
    r"https://www\.aihehuo\.com/[^\s)）\]】>，。；;\"']+",
    re.IGNORECASE,
)
# Only strip standalone evidence labels — never compound nouns like「资金来源：」.
_STANDALONE_SOURCE_LINE = re.compile(r"(?:^|\n)\s*来源：\s*(?=\n|$)")
_ORPHAN_SOURCE_LABEL = re.compile(r"(?<![\u4e00-\u9fff])来源：\s*(?=(?:[，。；;\n]|$))")
_NO_FULL_MATCH = re.compile(
    r"(?:没有|未找到|未发现|暂无|暂未).{0,18}(?:完全|同时|全部|精确)?(?:符合|匹配|记录|人选|候选人)|"
    r"(?:证据|公开资料).{0,18}(?:不足|无法确认)|"
    r"未能生成.{0,12}通过核验.{0,12}候选摘要"
)
_ACK_ONLY = re.compile(
    r"^(?:好的|明白|收到|可以)[，,。！!\s]*(?:我们|我)?"
    r"(?:来|已|现在|再)?(?:查|查询|搜索|筛选|匹配|切回|改成).{0,40}$"
)


@dataclass(frozen=True)
class IntentFinalGuardResult:
    """Deterministic final reply and its audit outcome."""

    reply: str
    blocked: bool
    violations: tuple[str, ...] = field(default=())


def intent_final_guard_enabled(*, enabled: bool | None = None) -> bool:
    """Resolve the independent hard-guard switch; missing configuration is on."""
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import (
        intent_final_guard_enabled as configured,
    )

    return configured()


def apply_intent_final_guard(
    *,
    reply: str,
    execution: IntentExecution | None,
    public_reads: list[PublicReadGround] | None = None,
    profile_saved_this_turn: bool = False,
    profile_persisted: bool = False,
    legacy_search_completed: bool = False,
    legacy_result_count: int = 0,
    enabled: bool | None = None,
) -> IntentFinalGuardResult:
    """Enforce current-run action and evidence claims without an LLM judge."""
    raw = (reply or "").strip()
    if not intent_final_guard_enabled(enabled=enabled):
        return IntentFinalGuardResult(
            reply=_strip_user_visible_source_urls(raw),
            blocked=False,
        )

    violations: list[str] = []
    claim_text = _claim_text(raw)
    future = bool(_FUTURE_QUERY.search(claim_text))
    search_done = bool(_SEARCH_DONE.search(claim_text))
    found = bool(_FOUND.search(claim_text))
    profile_done = bool(_PROFILE_DONE.search(claim_text))

    receipts: list[Any] = []
    evidence: list[Any] = []
    execute = bool(
        execution is not None and execution.decision.action == "execute"
    )
    is_people_search = bool(
        execute
        and execution is not None
        and execution.decision.primary_intent == "people_search"
    )
    projected_names: tuple[str, ...] = ()
    projected_candidates: tuple[dict[str, Any], ...] = ()
    projected_results: tuple[tuple[str, str], ...] = ()
    recommendation_cards_frozen = False
    if execute:
        valid, receipt_violation = _validate_execution(execution)
        if not valid:
            violations.append(receipt_violation)
        else:
            receipts = list(execution.receipts)
            evidence = [
                execution.evidence_by_receipt[receipt.receipt_id]
                for receipt in receipts
            ]
            if is_people_search:
                from apps.group_agent_api.agent_factory.intent_routing.projection import (
                    project_intent_search,
                )

                projection = project_intent_search(execution)
                projected_names = _projected_candidate_names(projection)
                projected_candidates = tuple(projection.candidates)
                recommendation_cards_frozen = (
                    execution.recommendation_candidates is not None
                )
            else:
                projected_results = _evidence_result_summaries(evidence)

    public_completed = bool(public_reads)
    completed = bool(receipts) or public_completed or legacy_search_completed
    raw_result_count = (
        execution.unique_result_count()
        if receipts
        else _public_result_count(public_reads) or max(0, int(legacy_result_count))
    )
    # People-search matching count uses projected candidates; other intents use raw rows.
    visible_result_count = (
        len(projected_names) if is_people_search and receipts else raw_result_count
    )

    if future:
        violations.append("future_execution_promise")
    if search_done and not completed:
        violations.append("unverified_search_claim")
    if found and visible_result_count <= 0:
        violations.append("empty_result_misrepresented")
    if profile_done and not (profile_saved_this_turn and profile_persisted):
        violations.append("unverified_profile_claim")
    if (
        looks_like_unauthorized_action_claim(claim_text)
        or _EXTERNAL_ACTION.search(claim_text)
    ):
        violations.append("unverified_external_action")

    if execute and receipts:
        if _OVERCLAIM.search(claim_text):
            violations.append("query_scope_overclaim")
        if any(
            not url.lower().startswith("https://www.aihehuo.com/")
            for url in _URL.findall(raw)
        ):
            violations.append("unsupported_source")
        no_full_match = bool(_NO_FULL_MATCH.search(claim_text))

        if is_people_search:
            supported_names = set(projected_names)
            raw_names = _evidence_person_names(evidence)
            mentioned = _explicit_entity_names(claim_text) | {
                name for name in raw_names if _name_appears_in_reply(name, raw)
            }
            if mentioned - supported_names:
                violations.append("invented_public_entity")
            if recommendation_cards_frozen and supported_names - mentioned:
                violations.append("missing_recommendation_cards")
            mentioned_supported = any(
                len(name) >= 2 and _name_appears_in_reply(name, claim_text)
                for name in supported_names
            )
            if _ACK_ONLY.fullmatch(raw) or (
                visible_result_count <= 0 and not no_full_match
            ) or (
                visible_result_count > 0
                and supported_names
                and not mentioned_supported
                and not no_full_match
            ):
                violations.append("execution_result_omitted")
        else:
            evidence_labels = _evidence_resource_labels(evidence)
            mentioned_supported = any(
                len(label) >= 2 and _name_appears_in_reply(label, claim_text)
                for label in evidence_labels
            )
            if _ACK_ONLY.fullmatch(raw) or (
                raw_result_count <= 0 and not no_full_match
            ) or (
                raw_result_count > 0
                and evidence_labels
                and not mentioned_supported
                and not no_full_match
            ):
                violations.append("execution_result_omitted")

    violations = list(dict.fromkeys(violations))

    if violations:
        guarded = _strip_user_visible_source_urls(
            _safe_reply(
                violations=violations,
                execute=execute,
                receipts=receipts,
                result_count=visible_result_count,
                raw_result_count=raw_result_count,
                projected_names=projected_names,
                projected_candidates=projected_candidates,
                projected_results=projected_results,
                is_people_search=is_people_search,
            )
        )
        _log(violations, blocked=True)
        return IntentFinalGuardResult(
            reply=guarded,
            blocked=True,
            violations=tuple(violations),
        )

    guarded = _strip_user_visible_source_urls(raw)
    if execute and receipts:
        guarded = _append_scope(guarded, receipts, raw_result_count)
    _log([], blocked=False)
    return IntentFinalGuardResult(reply=guarded, blocked=False)


def _claim_text(reply: str) -> str:
    """Remove quoted, negated, hypothetical, methodological, and historical claims."""
    unquoted = _QUOTED.sub("", reply)
    kept: list[str] = []
    for match in _CLAUSE.finditer(unquoted):
        clause = match.group(0).strip()
        if not clause:
            continue
        claim_matches = (
            list(_FUTURE_QUERY.finditer(clause))
            + list(_SEARCH_DONE.finditer(clause))
            + list(_PROFILE_DONE.finditer(clause))
            + list(_EXTERNAL_ACTION.finditer(clause))
            + list(_OVERCLAIM.finditer(clause))
            + [
                match
                for pattern in _EXPLICIT_ENTITY
                for match in pattern.finditer(clause)
            ]
        )
        if not claim_matches:
            kept.append(clause)
            continue
        masked = list(clause)
        for claim in claim_matches:
            prefix = clause[: claim.start()]
            if _NEGATED.search(prefix) or _NON_CURRENT.search(prefix):
                masked[claim.start() : claim.end()] = " " * (
                    claim.end() - claim.start()
                )
                continue
        cleaned = "".join(masked).strip()
        if cleaned:
            kept.append(cleaned)
    return "。".join(kept)


def _validate_execution(execution: IntentExecution) -> tuple[bool, str]:
    try:
        plan = execution.plan
        if plan is None or not execution.required_steps_satisfied():
            return False, "query_plan_unsatisfied"
        steps = {step.step_id: step for step in plan.steps}
        seen: set[tuple[str, int]] = set()
        for receipt in execution.receipts:
            step = steps.get(receipt.plan_step_id)
            evidence = execution.evidence_by_receipt.get(receipt.receipt_id)
            body = receipt.model_dump(mode="json")
            body.pop("receipt_id", None)
            key = (receipt.plan_step_id, receipt.request_index)
            if (
                step is None
                or key in seen
                or receipt.run_id != execution.snapshot.run_id
                or receipt.routing_snapshot_hash
                != execution.snapshot.canonical_content_hash
                or receipt.plan_id != plan.plan_id
                or receipt.tool != step.tool
                or receipt.resource != step.resource
                or canonical_hash(body) != receipt.receipt_id
                or evidence is None
                or canonical_hash(evidence) != receipt.evidence_digest
                or any(not _official_source(url) for url in receipt.source_urls)
                or (
                    receipt.produced_by_attempt_id != execution.attempt_id
                    and receipt.receipt_id not in execution.adopted_receipt_ids
                )
            ):
                return False, "receipt_invalid"
            seen.add(key)
        unique_count = execution.unique_result_count()
        if execution.store is not None:
            ledger = ReceiptLedgerV1.model_validate(
                execution.store.get_receipt_ledger(execution.snapshot.run_id)
            )
            receipt_refs = {
                (ref.plan_step_id, ref.request_index, ref.receipt_id)
                for ref in ledger.receipt_refs
            }
            expected_refs = {
                (receipt.plan_step_id, receipt.request_index, receipt.receipt_id)
                for receipt in execution.receipts
            }
            if receipt_refs != expected_refs:
                return False, "receipt_invalid"
            if any(
                ledger.budget.next_request_index_by_step.get(
                    receipt.plan_step_id, 1
                )
                <= receipt.request_index
                for receipt in execution.receipts
            ):
                return False, "receipt_invalid"
            adopted_refs = {
                (ref.receipt_id, ref.adopted_by_attempt_number)
                for ref in ledger.adoption_refs
            }
            if any(
                receipt.receipt_id in execution.adopted_receipt_ids
                and (receipt.receipt_id, execution.attempt_number)
                not in adopted_refs
                for receipt in execution.receipts
            ):
                return False, "receipt_invalid"
            budget = ledger.budget
        else:
            budget = execution._ephemeral_budget  # noqa: SLF001
        if (
            budget is None
            or budget.run_id != execution.snapshot.run_id
            or budget.routing_snapshot_hash
            != execution.snapshot.canonical_content_hash
            or budget.items_committed != unique_count
            or budget.requests_reserved < len(execution.receipts)
            or budget.pages_reserved < len(execution.receipts)
            or budget.requests_reserved > plan.budget.max_requests
            or budget.pages_reserved > plan.budget.max_pages
            or budget.items_committed > plan.budget.max_items
            or budget.tool_elapsed_committed_ms
            + budget.tool_elapsed_reserved_ms
            > plan.budget.max_elapsed_ms
        ):
            return False, "receipt_invalid"
    except Exception:  # noqa: BLE001 - deterministic guard exposes stable codes only.
        return False, "receipt_invalid"
    return True, ""


def _official_source(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.netloc == "www.aihehuo.com"
        and parsed.path.startswith("/ai/")
    )


def _public_result_count(reads: list[PublicReadGround] | None) -> int:
    return sum(len(read.records) for read in reads or [])


def _top_level_records(evidence: list[Any]) -> list[dict[str, Any]]:
    """Flatten receipt evidence to top-level row dicts (users or group members)."""
    records: list[dict[str, Any]] = []
    for value in evidence:
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, dict):
                continue
            members = item.get("members")
            if isinstance(members, list):
                for member in members:
                    if isinstance(member, dict):
                        records.append(member)
                continue
            records.append(item)
    return records


def _evidence_person_names(evidence: list[Any]) -> set[str]:
    """Person identifiers only: top-level name/display_name/nickname (never title)."""
    names: set[str] = set()
    for item in _top_level_records(evidence):
        for key in ("name", "display_name", "nickname"):
            candidate = item.get(key)
            if isinstance(candidate, str) and len(candidate.strip()) >= 2:
                names.add(candidate.strip())
    return names


def _evidence_resource_labels(evidence: list[Any]) -> set[str]:
    """Non-person result labels (FAQ/weekly/project titles) for completeness checks."""
    labels: set[str] = set()
    for item in _top_level_records(evidence):
        for key in ("title", "question", "name"):
            candidate = item.get(key)
            if isinstance(candidate, str) and len(candidate.strip()) >= 2:
                labels.add(candidate.strip())
    return labels


def _evidence_result_summaries(
    evidence: list[Any], *, limit: int = 3
) -> tuple[tuple[str, str], ...]:
    """Return a bounded, receipt-ordered view of non-person public results.

    The public API owns relevance.  This projection deliberately does not
    re-rank or re-filter its records; it only selects user-visible fields for a
    deterministic fallback when the composing model omits the results.
    """
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in _top_level_records(evidence):
        label = next(
            (
                str(item[key]).strip()
                for key in ("title", "question", "name")
                if isinstance(item.get(key), str) and str(item[key]).strip()
            ),
            "",
        )
        if len(label) < 2 or label in seen:
            continue
        summary = next(
            (
                str(item[key]).strip()
                for key in ("summary", "description", "answer", "bio")
                if isinstance(item.get(key), str) and str(item[key]).strip()
            ),
            "",
        )
        # Keep untrusted public text on one bounded line in the fallback.
        clean_label = re.sub(r"\s+", " ", label)[:100]
        clean_summary = re.sub(r"\s+", " ", summary)[:240]
        seen.add(label)
        results.append((clean_label, clean_summary))
        if len(results) >= max(1, limit):
            break
    return tuple(results)


def _name_appears_in_reply(name: str, reply: str) -> bool:
    text = reply or ""
    token = (name or "").strip()
    if len(token) < 2:
        return False
    return token in text


def _explicit_entity_names(reply: str) -> set[str]:
    names: set[str] = set()
    for pattern in _EXPLICIT_ENTITY:
        names.update(match.group(1).strip() for match in pattern.finditer(reply))
    return names


def _scope(receipts: list[Any], item_count: int) -> str:
    """User-visible scope footer without source URLs (URLs stay in receipts/debug)."""
    pages = len(receipts)
    return f"查询范围：本次查询覆盖 {pages} 页，检查 {item_count} 条去重公开记录。"


def _strip_user_visible_source_urls(reply: str) -> str:
    """Remove official evidence links and orphan「来源：」labels from user-facing text."""
    text = _USER_VISIBLE_SOURCE_URL.sub("", reply or "")
    text = _STANDALONE_SOURCE_LINE.sub("\n", text)
    text = _ORPHAN_SOURCE_LABEL.sub("", text)
    # Drop empty compound labels left after URL removal (e.g. 「项目来源：。」).
    text = re.sub(r"[\u4e00-\u9fff]{0,8}来源：\s*(?=[，。；;\n]|$)", "", text)
    text = re.sub(r"[，,]\s*[。．.]", "。", text)
    text = re.sub(r"[，,]\s*(?=[，。；;\n]|$)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n{3,}", "\n\n", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


def _append_scope(reply: str, receipts: list[Any], item_count: int) -> str:
    scope = _scope(receipts, item_count)
    cleaned = _strip_user_visible_source_urls(reply)
    if not cleaned:
        return scope
    if _SCOPE.search(cleaned):
        return cleaned
    return f"{cleaned}\n\n{scope}"


def _projected_candidate_names(projection: Any) -> tuple[str, ...]:
    """Names from constraint-filtered public records, not actionable cards."""
    names: list[str] = []
    seen: set[str] = set()
    for name in getattr(projection, "public_result_names", None) or []:
        if not isinstance(name, str) or len(name.strip()) < 2 or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= 3:
            return tuple(names)
    for candidate in getattr(projection, "candidates", None) or []:
        if not isinstance(candidate, dict):
            continue
        name = str(
            candidate.get("display_name") or candidate.get("name") or ""
        ).strip()
        if len(name) < 2 or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= 3:
            break
    return tuple(names)


def _safe_reply(
    *,
    violations: list[str],
    execute: bool,
    receipts: list[Any],
    result_count: int,
    projected_names: tuple[str, ...] = (),
    projected_candidates: tuple[dict[str, Any], ...] = (),
    projected_results: tuple[tuple[str, str], ...] = (),
    raw_result_count: int = 0,
    is_people_search: bool = False,
) -> str:
    if "unverified_external_action" in violations:
        return "我无法确认消息、通知或联系已经完成；本次没有可验证的外部操作凭证。"
    if "unverified_profile_claim" in violations:
        return "本次没有可验证的画像保存凭证，因此不能确认画像已经保存或更新。"
    invalid = {"receipt_invalid", "query_plan_unsatisfied"}
    if invalid.intersection(violations):
        return "本次公开查询的执行凭证未通过校验，无法据此提供结果，请重新发起查询。"
    if execute and receipts:
        card_mismatch = {
            "invented_public_entity",
            "missing_recommendation_cards",
        }.intersection(violations)
        if is_people_search and projected_candidates and card_mismatch:
            lines = ["本次公开查询已完成，以下候选人与页面展示的卡片一致："]
            for candidate in projected_candidates:
                name = str(candidate.get("display_name") or "").strip()
                intro = _candidate_intro(candidate)
                lines.append(f"- {name}：{intro}" if intro else f"- {name}")
            lines.append("你可以点击对应卡片查看公开资料或进入联系页面。")
            body = "\n".join(lines)
            return f"{body}\n\n{_scope(receipts, raw_result_count)}"
        if (
            is_people_search
            and "execution_result_omitted" in violations
            and not projected_names
            and raw_result_count > 0
        ):
            lead = (
                "本次公开查询已完成，但没有找到完全满足当前条件的人选。"
                "你可以调整地点或能力条件后重试。"
            )
        elif result_count <= 0 or "empty_result_misrepresented" in violations:
            lead = "本次公开查询已完成，但没有返回符合条件的公开记录。"
        elif "execution_result_omitted" in violations:
            if is_people_search and projected_names:
                lead = (
                    "本次公开查询已完成。返回的公开记录包括："
                    f"{'、'.join(projected_names)}。"
                    "这些记录是查询候选，是否符合全部条件仍需依公开资料逐项核对。"
                )
            elif projected_results:
                lines = ["本次公开查询已完成，返回的公开记录包括："]
                for label, summary in projected_results:
                    lines.append(f"- {label}：{summary}" if summary else f"- {label}")
                lines.append("以上为本次 API 返回的公开记录摘要。")
                lead = "\n".join(lines)
            else:
                lead = "本次公开查询已完成，但返回记录缺少可展示的标题或名称。"
        else:
            lead = "本次公开查询已完成；具体结果仅以已返回的公开记录为准。"
        return f"{lead}\n\n{_scope(receipts, raw_result_count)}"
    if "future_execution_promise" in violations:
        return "当前回复不能以“正在处理”或“稍后返回”作为终态；本次没有可继续执行的后台任务。"
    return "本次没有可验证的查询凭证，无法确认已经查询或找到结果，请重新发起查询。"


def _candidate_intro(candidate: dict[str, Any]) -> str:
    """Return one bounded public introduction from a frozen card."""
    for field_name in ("doing", "need", "offer"):
        raw = candidate.get(field_name)
        value = raw.get("value") if isinstance(raw, dict) else raw
        text = str(value or "").strip()
        if text:
            return text[:200]
    evidence = candidate.get("match_evidence")
    if isinstance(evidence, list) and evidence:
        first = evidence[0]
        summary = first.get("summary") if isinstance(first, dict) else first
        return str(summary or "").strip()[:200]
    return ""


def _log(violations: list[str], *, blocked: bool) -> None:
    _logger.info(
        "action=intent_final_guard check_id=%s blocked=%s violations=%s",
        CHECK_ID,
        blocked,
        ",".join(violations),
    )
