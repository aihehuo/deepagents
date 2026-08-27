"""Build check.reply_grounding.v1 ground from this-turn hand facts."""

from __future__ import annotations

from typing import Any

from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import (
    CandidateGround,
    FactItem,
    GroundBlock,
    MatchEvidenceItem,
)
from apps.group_agent_api.agent_factory.profile_schema import GroupProfile


def build_ground_from_turn(
    *,
    candidates: list[dict[str, Any]] | None,
    profile: GroupProfile | None = None,
    receipts: list[dict[str, Any]] | None = None,
    candidate_count: int | None = None,
) -> GroundBlock:
    """Map orchestrator turn state into the module ground block.

    Only confirmed-public / already-visible candidate fields are copied.
    Does not read Micro / new_api.
    """
    raw_candidates = [c for c in (candidates or []) if isinstance(c, dict)]
    mapped: list[CandidateGround] = []
    for raw in raw_candidates[:20]:
        user_id = str(raw.get("user_id") or raw.get("id") or "").strip()
        if not user_id:
            continue
        display = str(raw.get("display_name") or raw.get("name") or "").strip()[:64]
        facts = _extract_facts(raw)
        evidence = _extract_evidence(raw)
        mapped.append(
            CandidateGround(
                user_id=user_id[:64],
                display_name=display,
                facts=facts,
                match_evidence=evidence,
            )
        )

    count = candidate_count if candidate_count is not None else len(mapped)
    count = max(0, min(100, int(count)))

    initiator: dict[str, str] = {}
    if profile is not None:
        for key in ("doing", "need", "offer"):
            field_obj = getattr(profile, key, None)
            value = str(getattr(field_obj, "value", "") or "").strip()
            if value:
                initiator[key] = value[:2000]

    safe_receipts: list[dict[str, Any]] = []
    for item in receipts or []:
        if isinstance(item, dict):
            # Keep only compact receipt ids / kinds — no PII blobs.
            safe_receipts.append(
                {
                    k: item.get(k)
                    for k in ("id", "kind", "status", "action")
                    if k in item
                }
            )
        if len(safe_receipts) >= 20:
            break

    return GroundBlock(
        candidates=mapped,
        initiator_profile=initiator,
        receipts=safe_receipts,
        candidate_count=count,
    )


def _extract_facts(raw: dict[str, Any]) -> list[FactItem]:
    facts: list[FactItem] = []
    for field in ("doing", "need", "offer"):
        value = _field_text(raw.get(field))
        if value:
            facts.append(FactItem(field=field, value=value[:2000]))
    # Optional flat facts[] from match v2 payloads
    for item in raw.get("facts") or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or item.get("name") or "").strip()
        value = str(item.get("value") or "").strip()
        if field and value:
            facts.append(FactItem(field=field[:64], value=value[:2000]))
        if len(facts) >= 32:
            break
    return facts


def _extract_evidence(raw: dict[str, Any]) -> list[MatchEvidenceItem]:
    out: list[MatchEvidenceItem] = []
    for item in raw.get("match_evidence") or raw.get("evidence") or []:
        if isinstance(item, dict):
            summary = str(item.get("summary") or item.get("text") or "").strip()
        else:
            summary = str(item or "").strip()
        if summary:
            out.append(MatchEvidenceItem(summary=summary[:1000]))
        if len(out) >= 16:
            break
    return out


def _field_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return str(raw.get("value") or "").strip()
    return str(raw).strip()
