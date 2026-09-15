"""Bounded, tool-free prompt for ambiguous L2 intent classification."""

CLASSIFIER_SYSTEM_PROMPT = """你是群智能体的只读意图分类器，不执行任务、不调用工具、不判断权限。
只从给定九类中返回 IntentClassificationV1：people_search、opportunity_search、
project_partnership_search、community_search、startup_digest、platform_help、
profile_update、conversation、clarification。否定搜索和方法讨论优先 conversation；
对象不明确时用 clarification。evidence_spans 只能复制短语，最多 16 项。"""


def classifier_user_prompt(message: str, prior_intent: str | None) -> str:
    """Build a bounded prompt without passing conversation history or credentials."""
    text = (message or "").strip()[:2000]
    prior = prior_intent or "none"
    return f"prior_intent={prior}\nmessage={text}"
