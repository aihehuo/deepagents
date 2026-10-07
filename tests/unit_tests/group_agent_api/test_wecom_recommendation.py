"""One-record selection, pagination, stable identities and source-owned links."""
import json
from unittest.mock import patch

import pytest

from apps.group_agent_api.app.wecom_recommendation import MAX_PAGES, find_one, public_record, public_url, recommendation_text


def data(*ids, more=False):
    return json.dumps({"data": [{"id": str(i), "number": i, "name": f"候选{i}", "summary": "技术经验", "url": f"https://www.aihehuo.com/users/{i}"} for i in ids], "meta": {"has_more": more}})


async def test_only_one_result_is_returned_and_next_search_skips_previous_result():
    with patch("skillpacks.aihehuo.client.fetch", return_value=data(1, 2, 3)) as fetch:
        first, scope = await find_one(resource="users", parameters={"q": "技术"}, excluded=[])
        second, _ = await find_one(resource="users", parameters={"keyword": "换过的词"}, excluded=[first["key"]])
    assert first["id"] == "1"
    assert second["id"] == "2"
    assert scope["returned"] == 1
    assert fetch.call_count == 2


async def test_seen_page_is_skipped_and_pagination_finds_one_fresh_record():
    with patch("skillpacks.aihehuo.client.fetch", side_effect=[data(1, more=True), data(2)]) as fetch:
        card, scope = await find_one(resource="users", parameters={"q": "上海"}, excluded=["users:1"])
    assert card["id"] == "2"
    assert scope["pages"] == 2
    assert "page=2" in fetch.call_args.args[0]


async def test_all_seen_returns_no_repeat_and_stops_at_budget():
    with patch("skillpacks.aihehuo.client.fetch", return_value=data(1, more=True)) as fetch:
        card, scope = await find_one(resource="users", parameters={}, excluded=["users:1"])
    assert card is None
    assert scope["returned"] == 0
    assert fetch.call_count == MAX_PAGES


async def test_empty_data_does_not_generate_a_fake_candidate():
    with patch("skillpacks.aihehuo.client.fetch", return_value=data()) as fetch:
        card, _ = await find_one(resource="projects", parameters={}, excluded=[])
    assert card is None
    fetch.assert_called_once()


def test_identity_uses_public_number_and_resource_not_private_user_id():
    card = public_record("users", {"id": "internal123", "number": 99, "name": "甲", "recommended_members": [{"name": "乙"}]})
    assert card["key"] == "users:99"
    assert "recommended_members" not in card
    assert public_record("projects", {"id": "99", "title": "项目"})["key"] == "projects:99"


def test_faq_without_id_has_stable_identity_across_answer_changes():
    first = public_record("faqs", {"question": "如何注册", "answer": "旧答案", "category": "注册", "source": "FAQ"})
    second = public_record("faqs", {"question": "如何注册", "answer": "新答案", "category": "注册", "source": "FAQ"})
    assert first["key"] == second["key"]


@pytest.mark.parametrize("url", ["javascript:alert(1)", "https://evil.example/a", "https://www.aihehuo.com@evil.example/a", "https://u:p@www.aihehuo.com/a", "https://www.aihehuo.com:123/a"])
def test_untrusted_links_are_rejected(url):
    assert public_url(url) == ""


def test_model_cannot_replace_the_source_link():
    card = public_record("users", {"id": "1", "title": "甲", "url": "https://www.aihehuo.com/users/1"})
    text = recommendation_text("请访问 https://evil.example", card)
    assert "evil.example" not in text
    assert text.endswith(card["url"])
    assert recommendation_text("这位技术伙伴可以先了解一下。", card).startswith("这位技术伙伴")


async def test_bad_response_is_not_treated_as_no_results():
    with patch("skillpacks.aihehuo.client.fetch", return_value='{"error":"unavailable"}'):
        with pytest.raises(ValueError, match="invalid_public_search_response"):
            await find_one(resource="users", parameters={}, excluded=[])


async def test_customer_self_is_skipped():
    with patch("skillpacks.aihehuo.client.fetch", return_value=data(1, 2)):
        card, _ = await find_one(resource="users", parameters={}, excluded=[], customer_number="1")
    assert card["id"] == "2"


async def test_customer_projects_and_unknown_publishers_are_skipped():
    rows = [{"id": "1", "title": "自己的项目", "user": {"number": "123"}},
            {"id": "2", "title": "发布者未知"},
            {"id": "3", "title": "别人的项目", "user": {"number": "456"}}]
    with patch("skillpacks.aihehuo.client.fetch", return_value=json.dumps({"data": rows})):
        card, _ = await find_one(resource="projects", parameters={}, excluded=[], customer_number="123")
    assert card["id"] == "3"
