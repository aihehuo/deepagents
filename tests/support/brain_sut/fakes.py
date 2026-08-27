"""Contract-shaped fakes for ear / hand / mouth around the production brain."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.integrations.match_client import MatchHttpError
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.profile_client import (
    ProfileHttpError,
    canonical_profile_digest,
)
from apps.group_agent_api.agent_factory.match_stub import MatchResult
from apps.group_agent_api.agent_factory.profile_schema import GroupProfile

from tests.support.brain_sut.contracts import (
    HandProfileScript,
    HandSearchScript,
    MouthCallbackRecord,
    MouthDisposition,
    MouthScript,
    empty_result,
    matched_result,
    production_shaped_candidate,
)


class FakeHand:
    """Stand-in for hand.write (Micro profile) + hand.search (new_api match).

    Call signatures match production ``persist_group_profile`` / ``run_match``.
    """

    def __init__(self) -> None:
        self.search_scripts: deque[HandSearchScript] = deque()
        self.profile_scripts: deque[HandProfileScript] = deque()
        self.search_calls: list[dict[str, Any]] = []
        self.profile_calls: list[dict[str, Any]] = []
        self.default_profile_ok = True
        self._sticky_search: HandSearchScript | None = None

    def enqueue_search(self, script: HandSearchScript) -> None:
        self.search_scripts.append(script)

    def enqueue_profile(self, script: HandProfileScript) -> None:
        self.profile_scripts.append(script)

    def enqueue_matched(
        self,
        *,
        group_id: str,
        query: str = "poc",
        candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        cands = candidates or [
            production_shaped_candidate(
                user_id="cand_poc_1",
                group_id=group_id,
                display_name="PoC候选人",
                doing_value="Python LLM Agent 后端",
            )
        ]
        self.enqueue_search(
            HandSearchScript(
                kind="ok_matched",
                result=matched_result(query=query, group_id=group_id, candidates=cands),
            )
        )

    def stick_matched(
        self,
        *,
        group_id: str,
        query: str = "poc",
        candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        """Reuse one matched result for every subsequent search (multi-turn)."""
        cands = candidates or [
            production_shaped_candidate(
                user_id="cand_poc_1",
                group_id=group_id,
                display_name="PoC候选人",
                doing_value="Python LLM Agent 后端",
            )
        ]
        self._sticky_search = HandSearchScript(
            kind="ok_matched",
            result=matched_result(query=query, group_id=group_id, candidates=cands),
        )

    def enqueue_empty(self, *, group_id: str, query: str = "poc") -> None:
        self.enqueue_search(
            HandSearchScript(
                kind="ok_empty",
                result=empty_result(query=query, group_id=group_id),
            )
        )

    def enqueue_rejected(
        self, *, message: str = "v2_not_configured", status_code: int = 422
    ) -> None:
        self.enqueue_search(
            HandSearchScript(
                kind="rejected",
                http_error=MatchHttpError(message, status_code=status_code),
            )
        )

    def run_match(self, **kwargs: Any) -> MatchResult:
        """Production signature of ``match_backend.run_match``."""
        self.search_calls.append(dict(kwargs))
        query = str(kwargs.get("query") or "")
        group_id = str(kwargs.get("group_id") or "")
        if self.search_scripts:
            script = self.search_scripts.popleft()
        elif self._sticky_search is not None:
            script = self._sticky_search
        else:
            script = HandSearchScript(
                kind="ok_empty",
                result=empty_result(
                    query=query,
                    group_id=group_id,
                    reason="fake_hand_default_empty",
                ),
            )
        # BSD-01 P0: mirror production match_backend — never collapse HTTP to empty.
        try:
            return script.materialize()
        except MatchHttpError as exc:
            from apps.group_agent_api.agent_factory.integrations.match_backend import (
                disposition_for_http_error,
            )

            status, reason = disposition_for_http_error(exc)
            return MatchResult(
                status=status,
                candidates=[],
                query=query,
                group_id=group_id,
                reason=reason,
            )

    def persist_group_profile(
        self,
        *,
        profile: GroupProfile,
        run_id: str,
        attempt_id: str | None = None,
        fencing_token: int | None = None,
        base_url: str | None = None,
        secret: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Production signature of ``profile_client.persist_group_profile``."""
        self.profile_calls.append(
            {
                "user_id": profile.user_id,
                "group_id": profile.group_id,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "fencing_token": fencing_token,
            }
        )
        if self.profile_scripts:
            return self.profile_scripts.popleft().materialize()

        if not self.default_profile_ok:
            raise ProfileHttpError("fake_hand_default_profile_reject", status_code=422)

        digest = canonical_profile_digest(profile)
        return {
            "user_id": profile.user_id,
            "group_id": profile.group_id,
            "status": "created",
            "profile_version": 1,
            "schema_version": int(getattr(profile, "schema_version", 1) or 1),
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "profile_digest": digest,
        }


class FakeEar:
    """Stand-in for ear.in_group_probe / membership HTTP."""

    def __init__(self, tier: CapabilityTier = CapabilityTier.in_group) -> None:
        self.tier: CapabilityTier = tier
        self.calls: list[dict[str, Any]] = []

    def fetch_membership(self, **kwargs: Any) -> MembershipResult:
        self.calls.append(dict(kwargs))
        return MembershipResult(
            tier=self.tier,
            event_id="evt_poc",
            reason="fake_ear",
            source="fake",
        )


class FakeMouth:
    """Stand-in for mouth ingress callback HTTP.

    Records every envelope the brain emits. Matches production
    ``send_callback_event``: True on accept, False on transport fail,
    raises ``MouthIngressRejected`` on disposition=rejected.
    """

    def __init__(self) -> None:
        self.scripts: deque[MouthScript] = deque()
        self.records: list[MouthCallbackRecord] = []
        self.default_disposition: MouthDisposition = "accepted"

    def enqueue(self, script: MouthScript) -> None:
        self.scripts.append(script)

    async def send_callback_event(
        self,
        *,
        callback_url: str,
        envelope_dict: dict[str, Any],
        secret: str | None = None,
        max_retries: int | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        from apps.group_agent_api.agent_factory.integrations.callback_client import (
            MouthIngressRejected,
        )

        script = (
            self.scripts.popleft()
            if self.scripts
            else MouthScript(disposition=self.default_disposition)
        )
        self.records.append(
            MouthCallbackRecord(
                callback_url=callback_url,
                envelope=dict(envelope_dict),
                disposition=script.disposition,
            )
        )
        if script.disposition == "transport_fail":
            return False
        if script.disposition == "rejected":
            raise MouthIngressRejected(
                script.reason_code or "unverified_fact_source",
                disposition="rejected",
                repairable_by=script.repairable_by or "orchestrator",
                message=script.message,
                hint=script.hint,
            )
        return True
