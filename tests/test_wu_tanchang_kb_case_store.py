from __future__ import annotations

import pytest

from apps.wu_tanchang_api.agent_factory.kb_case_store import (
    StoredKbHit,
    clear_kb_hits,
    ground_meeting_prep_kb_cases,
    last_kb_hits_for_config,
    remember_kb_hits,
)

HIT_A = StoredKbHit(
    note_id="wu-jiushiye",
    title="九十葉凭什么让上海人排队1小时？拆解抹茶赛道的“浓度”生意经",
    brand="九十葉",
    xhs_url="https://xhslink.cn/o/jiushiye",
    insight="用浓度分级做招牌心智",
)
HIT_B = StoredKbHit(
    note_id="wu-chagee",
    title="上海科技馆霸王茶姬，为何拿了一手好牌却打得稀烂？",
    brand="霸王茶姬",
    xhs_url="https://xhslink.cn/o/chagee",
    insight="写字楼点位不等于进店转化",
)

PLACEHOLDER_BODY = """【会议准备 · 供面谈参考】

一、交谈要点
成都高新区茶小馆，客单15元，进店少复购低。

二、初步建议（八维锚点）
菜单工程、门头截流、模型测算。

三、建议讨论话题
1. 【菜单工程】柠檬茶引流 vs 奶茶利润
2. 【门头截流】写字楼路过心智

四、可参考案例
（来自 kb_analyst，1–2 个品牌；有探店笔记网页 URL 时写成 `[品牌 · 看探店笔记](完整URL)`）
"""


def setup_function() -> None:
    clear_kb_hits()


def test_remember_hits_are_visible_on_parent_conversation_key() -> None:
    sub_config = {
        "configurable": {"thread_id": "wt::default::1::wu_agent_1:subgraph"},
        "metadata": {"user_id": "1", "conversation_id": "wu_agent_1"},
    }
    parent_config = {
        "configurable": {"thread_id": "wt::default::1::wu_agent_1"},
        "metadata": {"user_id": "1", "conversation_id": "wu_agent_1"},
    }
    remember_kb_hits(sub_config, [HIT_A, HIT_B])
    found = last_kb_hits_for_config(parent_config)
    assert [h.brand for h in found] == ["九十葉", "霸王茶姬"]


def test_placeholder_section_four_is_replaced_with_search_links() -> None:
    filled, chat = ground_meeting_prep_kb_cases(PLACEHOLDER_BODY, (HIT_A, HIT_B))
    assert "来自 kb_analyst" not in filled
    assert "[九十葉 · 看探店笔记](https://xhslink.cn/o/jiushiye)" in filled
    assert "[霸王茶姬 · 看探店笔记](https://xhslink.cn/o/chagee)" in filled
    assert chat is not None
    assert "九十葉" in chat
    assert "xhslink.cn/o/jiushiye" in chat


def test_already_cited_section_four_is_left_alone() -> None:
    body = PLACEHOLDER_BODY.replace(
        "（来自 kb_analyst，1–2 个品牌；有探店笔记网页 URL 时写成 `[品牌 · 看探店笔记](完整URL)`）",
        "- [九十葉 · 看探店笔记](https://xhslink.cn/o/jiushiye)\n  浓度分级做招牌",
    )
    filled, chat = ground_meeting_prep_kb_cases(body, (HIT_A, HIT_B))
    assert filled == body.strip()
    assert chat is None


def test_new_section_four_template_is_treated_as_placeholder() -> None:
    body = PLACEHOLDER_BODY.replace(
        "（来自 kb_analyst，1–2 个品牌；有探店笔记网页 URL 时写成 `[品牌 · 看探店笔记](完整URL)`）",
        "- [品牌 · 看探店笔记](完整短链)\n↑ 换成 kb_analyst 本轮返回的真实品牌与 URL；禁止把上一行模板原样留给用户。",
    )
    filled, chat = ground_meeting_prep_kb_cases(body, (HIT_A,))
    assert "完整短链" not in filled
    assert "九十葉" in filled
    assert chat is not None


def test_missing_section_four_is_appended_from_hits() -> None:
    body = """【会议准备】
一、交谈要点 成都茶饮
二、初步建议 门头
三、建议讨论话题 进店率
"""
    filled, chat = ground_meeting_prep_kb_cases(body, (HIT_A,))
    assert "四、可参考案例" in filled
    assert "九十葉" in filled
    assert chat is not None


def test_save_meeting_prep_splices_hits_into_body_and_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.wu_tanchang_api.agent_factory.agent import save_meeting_prep
    from apps.wu_tanchang_api.agent_factory.prompt_modules import register_prompt_modules
    from apps.wu_tanchang_api.config import WuPromptModules

    register_prompt_modules("default", WuPromptModules())
    config = {
        "configurable": {"thread_id": "wt::default::1::wu_agent_1"},
        "metadata": {
            "user_id": "1",
            "conversation_id": "wu_agent_1",
            "callback_url": (
                "http://host.docker.internal:3001/wu_tanchang_callbacks/1/wu_agent_1"
            ),
        },
    }
    remember_kb_hits(config, [HIT_A, HIT_B])
    posted: dict = {}
    chats: list[dict] = []

    class Resp:
        status_code = 201
        text = ""

    monkeypatch.setattr(
        "requests.post", lambda _url, **kwargs: posted.update(kwargs) or Resp()
    )
    monkeypatch.setattr(
        "apps.wu_tanchang_api.app.callbacks.invoke_callback",
        lambda _url, payload: chats.append(payload) or False,
    )

    padded = PLACEHOLDER_BODY.replace(
        "成都高新区茶小馆，客单15元，进店少复购低。",
        "成都高新区茶小馆，客单15元，进店少复购低。" + ("补充说明。" * 80),
    )
    res = save_meeting_prep.invoke({"body": padded}, config=config)
    assert res == "meeting_prep_saved"
    body = posted["json"]["body"]
    assert "[九十葉 · 看探店笔记](https://xhslink.cn/o/jiushiye)" in body
    assert any("九十葉" in (item.get("message") or "") for item in chats)
