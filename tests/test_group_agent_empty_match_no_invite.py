"""Empty match must not auto-emit invite/topic artifacts."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.invite_copy import should_emit_invite_artifact


def test_should_emit_invite_skips_empty_match() -> None:
    assert (
        should_emit_invite_artifact(
            match_status="empty",
            match_reason="sc05_no_suitable_match",
            candidate_count=0,
        )
        is False
    )


def test_should_emit_invite_skips_zero_candidates_even_if_matched_label() -> None:
    assert (
        should_emit_invite_artifact(
            match_status="matched",
            match_reason="matched_3",
            candidate_count=0,
        )
        is False
    )


def test_should_emit_invite_allows_matched_with_candidates() -> None:
    assert (
        should_emit_invite_artifact(
            match_status="matched",
            match_reason="matched_2",
            candidate_count=2,
        )
        is True
    )


def test_should_emit_invite_allows_weak_with_candidates() -> None:
    assert (
        should_emit_invite_artifact(
            match_status="weak",
            match_reason="sc06_weak_match",
            candidate_count=1,
        )
        is True
    )


def test_should_emit_invite_skips_thin_profile() -> None:
    assert (
        should_emit_invite_artifact(
            match_status="skipped",
            match_reason="profile_too_thin",
            candidate_count=0,
        )
        is False
    )
