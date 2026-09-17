"""Shared Phase 2 planner fixtures for group-agent intent routing tests."""

from __future__ import annotations

import asyncio
from typing import Any


def proposal(
    *,
    action: str = "query",
    intent: str | None = "people_search",
    users: str | None = None,
    ideas: str | None = None,
    projects: str | None = None,
    groups: str | None = None,
    faqs: str | None = None,
    clarification: str | None = None,
    time_range: str | None = None,
) -> dict[str, Any]:
    """Build a closed PlannerProposalV2 payload."""
    return {
        "version": 2,
        "action": action,
        "intent": intent,
        "queries": {
            "users": users,
            "ideas": ideas,
            "projects": projects,
            "groups": groups,
            "faqs": faqs,
        },
        "clarification": clarification,
        "time_range": time_range,
        "planner_version": "intent-planner-v2",
    }


def people_query(query: str, *, time_range: str | None = None) -> dict[str, Any]:
    return proposal(intent="people_search", users=query, time_range=time_range)


def opportunity_query(
    ideas: str,
    *,
    projects: str | None = None,
    time_range: str | None = None,
) -> dict[str, Any]:
    return proposal(
        intent="opportunity_search",
        ideas=ideas,
        projects=projects,
        time_range=time_range,
    )


def community_query(groups: str) -> dict[str, Any]:
    return proposal(intent="community_search", groups=groups)


def platform_help_query(faqs: str) -> dict[str, Any]:
    return proposal(intent="platform_help", faqs=faqs)


def reply(*, intent: str = "conversation") -> dict[str, Any]:
    return proposal(action="reply", intent=intent)


def clarify(text: str | None = None) -> dict[str, Any]:
    return proposal(
        action="clarify",
        intent="clarification",
        clarification=text,
    )


# Canned proposals for well-known QueryPlan / guard messages.
MESSAGE_PROPOSALS: dict[str, dict[str, Any]] = {
    "有没有上海餐饮合伙人": people_query("上海 餐饮 合伙人"),
    "帮我找上海做餐饮的创业者，优先有供应链经验，最近的": people_query(
        "上海 餐饮 供应链经验 创业者", time_range="最近"
    ),
    "请帮我搜索上海餐饮行业的运营合伙人": people_query("上海 餐饮 运营合伙人"),
    "请帮我搜索上海餐饮行业的全职运营合伙人": people_query(
        "上海 餐饮 运营合伙人 全职"
    ),
    "看看上海有哪些餐饮创业项目": opportunity_query(
        "上海 餐饮 创业项目", projects="上海 餐饮"
    ),
    "看看本月的创业机会": opportunity_query("创业机会", time_range="本月"),
    "看看本周的创业机会": opportunity_query("创业机会", time_range="本周"),
    "查看上海创业社群和第一个社群的报名名单": community_query(
        "上海 创业社群 第一个社群 报名名单"
    ),
    "查看社群 ID 763 的报名名单": community_query("社群 ID 763 报名名单"),
    "查看社群 ID 763 的详情": community_query("社群 ID 763 详情"),
    "帮我找技术合伙人": people_query("技术合伙人"),
    "帮我找合伙人": people_query("合伙人"),
    "帮我找上海的技术合伙人": people_query("上海 技术合伙人"),
    "有没有上海餐饮合伙人": people_query("上海 餐饮 合伙人"),
    "找上海餐饮运营合伙人": people_query("上海 餐饮 运营合伙人"),
    "找上海餐饮创业者，优先有供应链经验": people_query(
        "上海 餐饮 创业者 供应链经验"
    ),
    "看看本周创业周报和文章": proposal(intent="startup_digest"),
    "如何发布餐饮项目招合伙人": platform_help_query("发布 餐饮项目 招合伙人"),
    "北京的呢": people_query("北京 餐饮 创业者 供应链经验"),
    "不要北京，改成上海的": people_query("上海 餐饮 创业者 供应链经验"),
    "把我的所在地改成上海": reply(intent="profile_update"),
    "先别查，只是聊聊怎么找合伙人": reply(),
    "找合伙人是不是很难": reply(),
}


def proposal_for_message(message: str) -> dict[str, Any]:
    """Return a canned proposal, defaulting to people search on the raw message."""
    if message in MESSAGE_PROPOSALS:
        return MESSAGE_PROPOSALS[message]
    if "社群" in message or "社区" in message:
        return community_query(message[:160])
    if "机会" in message or "项目" in message or "idea" in message.lower():
        return opportunity_query(message[:160], projects=message[:160])
    if "faq" in message.lower() or "如何" in message or "怎么" in message:
        return platform_help_query(message[:160])
    return people_query(message[:160])


class StructuredPlanner:
    """Queue of structured planner outputs with optional delay."""

    def __init__(
        self,
        payloads: list[Any],
        *,
        delay: float = 0.0,
        raise_on_call: Exception | None = None,
    ) -> None:
        self.payloads = list(payloads)
        self.delay = delay
        self.raise_on_call = raise_on_call
        self.calls = 0
        self.prompts: list[Any] = []

    def invoke(self, prompts: Any) -> Any:
        self.calls += 1
        self.prompts.append(prompts)
        if self.raise_on_call is not None:
            raise self.raise_on_call
        if self.delay:
            # Sync path uses blocking sleep only in timeout tests via async.
            pass
        if not self.payloads:
            raise ValueError("no planner payload left")
        return self.payloads.pop(0)

    async def ainvoke(self, prompts: Any) -> Any:
        self.calls += 1
        self.prompts.append(prompts)
        if self.raise_on_call is not None:
            raise self.raise_on_call
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.payloads:
            raise ValueError("no planner payload left")
        return self.payloads.pop(0)


class PlannerModel:
    """Bare chat model stub that only exposes with_structured_output."""

    def __init__(self, structured: StructuredPlanner) -> None:
        self._structured = structured

    def with_structured_output(self, _: Any) -> StructuredPlanner:
        return self._structured


def planner_for_message(message: str) -> PlannerModel:
    """Return a one-shot planner model for a known test message."""
    return PlannerModel(StructuredPlanner([proposal_for_message(message)]))


def planner_returning(*payloads: dict[str, Any]) -> PlannerModel:
    """Return a planner model that yields the given proposals in order."""
    return PlannerModel(StructuredPlanner(list(payloads)))


class AdaptivePlannerModel:
    """Planner that derives canned proposals from planner_context.current_message."""

    def with_structured_output(self, schema: Any) -> Any:
        class _Adapter:
            def invoke(self, prompts: Any) -> Any:
                content = str(getattr(prompts[1], "content", "") or "")
                prefix = "planner_context="
                assert content.startswith(prefix)
                import json

                context = json.loads(content[len(prefix) :])
                return schema.model_validate(
                    proposal_for_message(str(context.get("current_message") or ""))
                )

            async def ainvoke(self, prompts: Any) -> Any:
                return self.invoke(prompts)

        return _Adapter()

