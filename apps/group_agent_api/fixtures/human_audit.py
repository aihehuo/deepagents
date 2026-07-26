"""REQ-013 human-readable audit reports for the opt-in REQ-012 scenario.

This module is deliberately deterministic. It never calls a model, never
summarizes hidden model state, and only accepts the user-visible fields passed
by the REQ-012 test after its machine oracles have succeeded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from apps.group_agent_api.agent_factory.disclosure import (
    public_match_basis,
    stable_candidate_user_id,
    stable_user_id_value,
)
from apps.group_agent_api.agent_factory.guard import extract_at_identities

AUDIT_SCHEMA_VERSION = "GA-HUMAN-AUDIT-V1"
AUDIT_ENV = "GROUP_AGENT_HUMAN_AUDIT_REPORT"
AUDIT_OUTPUT_ENV = "GROUP_AGENT_HUMAN_AUDIT_OUTPUT_DIR"
DEFAULT_AUDIT_RELATIVE_DIR = Path(".local-artifacts/group-agent-audit")
MAX_REPLY_EXCERPT_CHARS = 600
MAX_INVITE_EXCERPT_CHARS = 400
MAX_EXCERPT_UNITS = 4
MAX_EXCERPT_COVERAGE = 0.50
READY_FILENAME = "READY.json"

_FORBIDDEN_KEYS = frozenset(
    {
        "phone",
        "wechat",
        "email",
        "private_notes",
        "authorization",
        "api_key",
        "access_token",
        "user_token",
        "group_token",
        "hmac_secret",
    }
)
_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("AUTHORIZATION", re.compile(r"(?i)\bauthorization\s*[:=]")),
    ("BEARER", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}")),
    ("API_KEY", re.compile(r"(?i)\b(?:api[_ -]?key|dashscope[_ -]?api[_ -]?key)\s*[:=]")),
    ("SK_KEY", re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")),
    ("CN_MOBILE", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")),
    ("EMAIL", re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")),
    ("CHAIN_OF_THOUGHT", re.compile(r"(?i)(?:chain[- ]of[- ]thought|system prompt|<think>|隐藏思维链)")),
)
_HIGH_VALUE_TERMS = (
    "doing",
    "need",
    "offer",
    "python",
    "langchain",
    "fastapi",
    "pytorch",
    "agent",
    "技术",
    "合作",
    "架构",
    "推荐",
    "匹配",
    "邀请",
    "@",
)
_LOW_VALUE_SENTENCES = (
    "你好",
    "您好",
    "收到",
    "明白了",
    "好的",
    "感谢",
)
_OVERPROMISE_TERMS = ("保证成功", "绝对适合", "必然成为合伙人")
_SENTENCE_RE = re.compile(r".+?(?:[。！？!?；;]\s*|\n+|$)", re.DOTALL)


class HumanAuditError(RuntimeError):
    """Base class for deterministic audit-generation failures."""


class HumanAuditRedactionError(HumanAuditError):
    """Raised when report-bound data contains a forbidden key or value."""

    def __init__(self, *, rule_id: str, field_path: str, content_hash: str) -> None:
        self.rule_id = rule_id
        self.field_path = field_path
        self.content_hash = content_hash
        super().__init__(
            f"FAILED:HUMAN_AUDIT_REDACTION rule={rule_id} "
            f"path={field_path} sha256={content_hash}"
        )


@dataclass(frozen=True)
class AuditWriteResult:
    markdown_path: Path
    json_path: Path
    markdown_size: int
    json_size: int
    markdown_sha256: str
    json_sha256: str
    ready_path: Path
    ready_sha256: str

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "markdown": {
                "path": str(self.markdown_path.resolve()),
                "size": self.markdown_size,
                "sha256": self.markdown_sha256,
            },
            "json": {
                "path": str(self.json_path.resolve()),
                "size": self.json_size,
                "sha256": self.json_sha256,
            },
            "ready": {
                "path": str(self.ready_path.resolve()),
                "sha256": self.ready_sha256,
            },
        }


def audit_enabled(env: dict[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return source.get(AUDIT_ENV, "").strip() == "1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _sentences(source: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(source or "") if match.group(0).strip()]


def extract_high_value_sentences(source: str, *, max_chars: int) -> list[str]:
    """Select complete source sentences deterministically and preserve order.

    Returned items are byte-for-byte substrings of ``source``. They are kept
    as a list so rendering separators cannot be mistaken for model-authored
    prose.
    """
    if max_chars <= 0:
        return []
    sentences = _sentences(source)
    if not sentences:
        return []

    ranked: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        score = sum(1 for term in _HIGH_VALUE_TERMS if term in lowered)
        if any(
            lowered.rstrip("。！？!?；;，,").strip() == term.lower()
            for term in _LOW_VALUE_SENTENCES
        ):
            score -= 10
        ranked.append((score, index, sentence))

    selected_indexes: set[int] = set()
    used = 0
    for _, index, sentence in sorted(ranked, key=lambda item: (-item[0], item[1])):
        if len(selected_indexes) >= MAX_EXCERPT_UNITS:
            break
        if len(sentence) > max_chars:
            continue
        separator_cost = 1 if selected_indexes else 0
        if used + separator_cost + len(sentence) > max_chars:
            continue
        selected_chars = sum(len(sentences[i]) for i in selected_indexes) + len(sentence)
        if source and selected_chars / len(source) > MAX_EXCERPT_COVERAGE:
            continue
        selected_indexes.add(index)
        used += separator_cost + len(sentence)

    # Never persist the complete response. If all source units fit, discard
    # the lowest-value unit (stable tie-break: the later unit). A one-unit
    # response therefore yields no excerpt rather than a complete response.
    if selected_indexes and len(selected_indexes) == len(sentences):
        discard = min(
            (item for item in ranked if item[1] in selected_indexes),
            key=lambda item: (item[0], -item[1]),
        )
        selected_indexes.remove(discard[1])

    # A very long single unit cannot be safely truncated because that would
    # cease to be an original complete unit. Fail closed to no excerpt.
    return [sentences[index] for index in sorted(selected_indexes)]


def excerpt_coverage(source: str, excerpts: Iterable[str]) -> float:
    if not source:
        return 0.0
    return round(sum(len(excerpt) for excerpt in excerpts) / len(source), 6)


def assert_excerpt_from_source(source: str, excerpts: Iterable[str]) -> None:
    cursor = 0
    for excerpt in excerpts:
        index = source.find(excerpt, cursor)
        if index < 0:
            raise HumanAuditError("excerpt is not an ordered source substring")
        cursor = index + len(excerpt)


def _profile_dimensions(profile: dict[str, Any] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    source = profile or {}
    for name in ("doing", "need", "offer"):
        raw = source.get(name, {})
        value = raw.get("value", "") if isinstance(raw, dict) else raw
        result[name] = str(value or "").strip()
    return result


def profile_diff(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    before_dims = _profile_dimensions(before)
    after_dims = _profile_dimensions(after)
    fields = {
        name: {
            "before": before_dims[name],
            "after": after_dims[name],
            "changed": before_dims[name] != after_dims[name],
        }
        for name in ("doing", "need", "offer")
    }
    before_updated = str((before or {}).get("updated_at", "") or "")
    after_updated = str((after or {}).get("updated_at", "") or "")
    return {
        "fields": fields,
        "updated_at_changed": before_updated != after_updated,
    }


def _public_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for candidate in candidates:
        user_id = stable_candidate_user_id(candidate)
        if user_id is None:
            raise HumanAuditError("candidate missing stable user_id")
        basis = public_match_basis(candidate)
        if not basis:
            raise HumanAuditError("candidate missing auditable public match basis")
        public.append(
            {
                "user_id": user_id,
                "display_name": str(candidate.get("display_name", "") or ""),
                "source_group_id": str(
                    candidate.get("source_group_id") or candidate.get("group_id") or ""
                ),
                "public_match_basis": basis,
                "match_confidence": str(candidate.get("match_confidence", "") or ""),
            }
        )
    return public


def assert_auditable_candidate_evidence(report: dict[str, Any]) -> None:
    """Fail closed unless every candidate and mention has the same public basis."""
    for turn in report.get("rounds", []):
        if turn.get("round") != 3:
            continue
        candidate_rows = list(turn.get("candidates", []))
        candidates: dict[str, Any] = {}
        for row in candidate_rows:
            if not isinstance(row, dict):
                raise HumanAuditError("candidate identity must be canonical string")
            user_id = stable_user_id_value(row.get("user_id"))
            if user_id is None:
                raise HumanAuditError("candidate identity must be canonical string")
            candidates[user_id] = row.get("public_match_basis")
        if (
            len(candidates) != len(candidate_rows)
            or any(not basis for basis in candidates.values())
        ):
            raise HumanAuditError("candidate missing auditable public match basis")
        evidence_rows = list(turn.get("mentioned_evidence", []))
        evidence: dict[str, Any] = {}
        for row in evidence_rows:
            if not isinstance(row, dict):
                raise HumanAuditError(
                    "mentioned evidence identity must be canonical string"
                )
            user_id = stable_user_id_value(row.get("user_id"))
            if user_id is None:
                raise HumanAuditError(
                    "mentioned evidence identity must be canonical string"
                )
            evidence[user_id] = row.get("public_match_basis")
        mentioned = list(turn.get("mentioned_user_ids", []))
        if any(stable_user_id_value(uid) is None for uid in mentioned):
            raise HumanAuditError("mentioned identity must be canonical string")
        actual_mentions = list(turn.get("invite_actual_at_user_ids", []))
        if any(stable_user_id_value(uid) is None for uid in actual_mentions):
            raise HumanAuditError(
                "invite text identity must be canonical string"
            )
        if (
            len(actual_mentions) != len(set(actual_mentions))
            or set(actual_mentions) != set(mentioned)
            or len(actual_mentions) != len(mentioned)
        ):
            raise HumanAuditError(
                "invite text mentions inconsistent with mentioned metadata"
            )
        if (
            len(evidence_rows) != len(mentioned)
            or len(evidence) != len(evidence_rows)
        ):
            raise HumanAuditError(
                "mentioned candidate evidence missing or inconsistent"
            )
        for user_id in mentioned:
            if (
                not evidence.get(user_id)
                or evidence[user_id] != candidates.get(user_id)
            ):
                raise HumanAuditError(
                    "mentioned candidate evidence missing or inconsistent"
                )


def _coverage_checks(message: str, profile: dict[str, str]) -> dict[str, bool]:
    lowered = message.lower()
    combined_profile = " ".join(profile.values()).lower()
    return {
        "reply_non_empty": bool(message.strip()),
        "doing_present": bool(profile["doing"]),
        "need_present": bool(profile["need"]),
        "offer_present": bool(profile["offer"]),
        "mentions_technical_or_collaboration_fact": any(
            term in lowered for term in ("python", "langchain", "fastapi", "pytorch", "agent", "合作", "技术")
        ),
        "profile_covers_user_facts": any(
            term in combined_profile
            for term in ("python", "langchain", "fastapi", "pytorch", "agent", "合作", "技术")
        ),
    }


def _nested_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, dict):
        result: set[str] = set()
        for child in value.values():
            result.update(_nested_strings(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_nested_strings(child))
        return result
    return set()


def collect_l1_forbidden_values(
    scenario_group_id: str = "group_l1_alpha",
) -> set[str]:
    """Build forbidden values from disclosure and trusted-group relationships."""
    members_path = Path(__file__).parent / "group_agent" / "l1" / "members.json"
    raw = json.loads(members_path.read_text(encoding="utf-8"))
    values: set[str] = set()
    members = raw.get("members", [])
    trusted_user_ids = {
        str(member.get("user_id"))
        for member in members
        if member.get("group_id") == scenario_group_id and member.get("user_id")
    }
    for member in members:
        # Top-level declared sensitive values are always forbidden.
        for key in _FORBIDDEN_KEYS:
            value = member.get(key)
            if isinstance(value, str) and value:
                values.add(value)
        profile = member.get("profile", {})
        # Non-public profiles are forbidden even for a member of the trusted
        # group. They may be used for internal matching but never audit output.
        if member.get("disclosure_level") != "confirmed_public":
            values.update(_nested_strings(profile))
        # Every identity and profile fact from another group is forbidden.
        if member.get("group_id") != scenario_group_id:
            for key in ("group_id", "display_name"):
                value = member.get(key)
                if isinstance(value, str) and value:
                    values.add(value)
            foreign_user_id = member.get("user_id")
            if (
                isinstance(foreign_user_id, str)
                and foreign_user_id
                and foreign_user_id not in trusted_user_ids
            ):
                values.add(foreign_user_id)
            values.update(_nested_strings(profile))
    return values


def collect_l1_sensitive_values() -> set[str]:
    """Compatibility alias for the Alpha REQ-012 scenario."""
    return collect_l1_forbidden_values("group_l1_alpha")


def _redaction_failure(rule_id: str, path: str, value: str) -> HumanAuditRedactionError:
    return HumanAuditRedactionError(
        rule_id=rule_id,
        field_path=path,
        content_hash=_sha256_text(value),
    )


def assert_report_safe(data: Any, *, sensitive_values: Iterable[str] = ()) -> None:
    values = tuple(value for value in sensitive_values if value)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                if normalized in _FORBIDDEN_KEYS:
                    raise _redaction_failure("FORBIDDEN_FIELD", f"{path}.{key}", str(child))
                visit(child, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        if not isinstance(value, str):
            return
        folded = value.casefold()
        for actual in values:
            if actual.casefold() in folded:
                raise _redaction_failure("FIXTURE_SENSITIVE_VALUE", path, value)
        for rule_id, pattern in _SENSITIVE_PATTERNS:
            if pattern.search(value):
                raise _redaction_failure(rule_id, path, value)

    visit(data, "$")


class HumanAuditCollector:
    """Collect three user-visible turns and atomically write MD + JSON."""

    def __init__(
        self,
        *,
        enabled: bool,
        run_id: str,
        provider: str,
        model: str,
        base_url_configured: bool,
        fixture_level: str,
        group_id: str,
        caller_id: str,
        output_dir: Path | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id):
            raise HumanAuditError("audit run_id must be 1-128 safe filename characters")
        self.enabled = enabled
        self.run_id = run_id
        self.provider = provider
        self.model = model
        self.base_url_configured = base_url_configured
        self.fixture_level = fixture_level
        self.group_id = group_id
        self.caller_id = caller_id
        self.output_dir = output_dir
        self._rounds: list[dict[str, Any]] = []

    @property
    def captured_rounds(self) -> int:
        return len(self._rounds)

    def capture_round(
        self,
        *,
        number: int,
        user_input: str,
        reply: str,
        llm_delta: dict[str, Any],
        latency_s: float,
        profile_before: dict[str, Any] | None,
        profile_after: dict[str, Any] | None,
        candidates: list[dict[str, Any]] | None = None,
        invite_text: str = "",
        mentioned_user_ids: list[str] | None = None,
        invite_ok: bool | None = None,
        guard_blocked: bool | None = None,
    ) -> None:
        if not self.enabled:
            return
        # Scan the complete user-visible output before excerpt selection. This
        # prevents a sensitive or overpromising sentence from disappearing
        # merely because the deterministic selector did not retain it.
        assert_report_safe(
            {"reply": reply, "invite": invite_text},
            sensitive_values=collect_l1_forbidden_values(self.group_id),
        )
        reply_excerpt = extract_high_value_sentences(reply, max_chars=MAX_REPLY_EXCERPT_CHARS)
        invite_excerpt = extract_high_value_sentences(
            invite_text, max_chars=MAX_INVITE_EXCERPT_CHARS
        )
        assert_excerpt_from_source(reply, reply_excerpt)
        assert_excerpt_from_source(invite_text, invite_excerpt)

        after_dims = _profile_dimensions(profile_after)
        candidate_rows = _public_candidates(candidates or [])
        candidate_ids = [row["user_id"] for row in candidate_rows]
        mentioned = list(mentioned_user_ids or [])
        if any(stable_user_id_value(value) is None for value in mentioned):
            raise HumanAuditError("mentioned identity must be canonical string")
        actual_invite_mentions = extract_at_identities(invite_text)
        if any(
            stable_user_id_value(value) is None
            for value in actual_invite_mentions
        ):
            raise HumanAuditError(
                "invite text identity must be canonical string"
            )
        if (
            len(actual_invite_mentions) != len(set(actual_invite_mentions))
            or set(actual_invite_mentions) != set(mentioned)
            or len(actual_invite_mentions) != len(mentioned)
        ):
            raise HumanAuditError(
                "invite text mentions inconsistent with mentioned metadata"
            )
        candidate_by_id = {row["user_id"]: row for row in candidate_rows}
        mentioned_evidence: list[dict[str, Any]] = []
        for mentioned_id in mentioned:
            candidate = candidate_by_id.get(mentioned_id)
            if candidate is None or not candidate["public_match_basis"]:
                raise HumanAuditError(
                    "mentioned candidate missing auditable public match basis"
                )
            mentioned_evidence.append(
                {
                    "user_id": mentioned_id,
                    "public_match_basis": dict(candidate["public_match_basis"]),
                }
            )
        complete_visible = "\n".join((reply, invite_text))
        checks: dict[str, Any] = {
            **_coverage_checks(reply, after_dims),
            "save_group_profile_called": (
                int((llm_delta.get("tool_calls") or {}).get("save_group_profile", 0)) >= 1
                if number in (1, 2)
                else None
            ),
            "profile_updated": profile_diff(profile_before, profile_after)["updated_at_changed"],
            "candidate_count_lte_3": len(candidate_rows) <= 3,
            "all_candidates_have_public_basis": all(
                bool(row["public_match_basis"]) for row in candidate_rows
            ),
            "all_candidates_current_group": all(
                row["source_group_id"] == self.group_id for row in candidate_rows
            ),
            "caller_not_self_matched": self.caller_id not in candidate_ids,
            "u101_present": "u101" in candidate_ids if number == 3 else None,
            "known_foreign_candidate_count": sum(
                uid in {"u201", "u202"} for uid in candidate_ids
            ),
            "mentioned_subset_of_candidates": set(mentioned).issubset(candidate_ids),
            "invite_text_mentions_match_metadata": (
                actual_invite_mentions == mentioned
                or (
                    len(actual_invite_mentions) == len(mentioned)
                    and set(actual_invite_mentions) == set(mentioned)
                )
            ),
            "all_mentioned_have_public_basis": (
                len(mentioned_evidence) == len(mentioned)
            ),
            "invite_ok": invite_ok,
            "guard_blocked": guard_blocked,
            "sensitive_leak_count": 0,
            "overpromise_term_count": sum(
                complete_visible.count(term) for term in _OVERPROMISE_TERMS
            ),
        }
        if number == 2:
            profile_text = " ".join(after_dims.values()).lower()
            checks["r2_absorbs_new_stack"] = all(
                term in profile_text for term in ("python", "langchain", "fastapi", "pytorch")
            )
            checks["r2_absorbs_collaboration_constraint"] = any(
                term in profile_text for term in ("全职", "架构", "合作", "投入")
            )
        self._rounds.append(
            {
                "round": number,
                "user_input": user_input,
                "assistant_excerpt": reply_excerpt,
                "reply_sha256": _sha256_text(reply),
                "reply_original_chars": len(reply),
                "reply_coverage_ratio": excerpt_coverage(reply, reply_excerpt),
                "invite_excerpt": invite_excerpt,
                "invite_sha256": _sha256_text(invite_text) if invite_text else "",
                "invite_original_chars": len(invite_text),
                "invite_coverage_ratio": excerpt_coverage(invite_text, invite_excerpt),
                "llm_delta": {
                    "llm_starts": int(llm_delta.get("llm_starts", 0)),
                    "llm_ends": int(llm_delta.get("llm_ends", 0)),
                    "tool_calls": dict(llm_delta.get("tool_calls") or {}),
                    "tokens": int(llm_delta.get("tokens", 0)),
                },
                "latency_s": round(float(latency_s), 2),
                "saved_profile": after_dims,
                "profile_diff": profile_diff(profile_before, profile_after),
                "candidates": candidate_rows,
                "mentioned_user_ids": mentioned,
                "invite_actual_at_user_ids": actual_invite_mentions,
                "mentioned_evidence": mentioned_evidence,
                "invite_ok": invite_ok,
                "guard_blocked": guard_blocked,
                "automatic_checks": checks,
                "human_review": {
                    "easy_to_understand": None,
                    "accurate": None,
                    "helpful": None,
                    "natural": None,
                    "notes": "",
                },
            }
        )

    def build_report(
        self,
        *,
        total_llm_invocations: int,
        total_tokens: int,
        total_time_s: float,
        machine_oracles: dict[str, Any],
        generated_at: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if len(self._rounds) != 3:
            raise HumanAuditError("audit requires exactly three captured rounds")
        for turn in self._rounds:
            if not turn["assistant_excerpt"]:
                raise HumanAuditError(
                    f"round {turn['round']} has no safe non-complete assistant excerpt"
                )
        if not self._rounds[2]["invite_excerpt"]:
            raise HumanAuditError("round 3 has no safe non-complete invite excerpt")
        report = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "generated_at": generated_at or datetime.now(UTC).isoformat(),
            "metadata": {
                "provider": self.provider,
                "model": self.model,
                "base_url_configured": self.base_url_configured,
                "fixture_level": self.fixture_level,
                "group_id": self.group_id,
                "mock_caller_id": self.caller_id,
                "llm_invocations": int(total_llm_invocations),
                "token_usage": int(total_tokens),
                "total_time_s": round(float(total_time_s), 2),
            },
            "machine_oracles": machine_oracles,
            "rounds": list(self._rounds),
            "human_audit_notice": (
                "人工评分由审核者填写；自动检查不评价内容质量。"
                "报告不包含隐藏推理、系统指令或完整模型响应。"
            ),
        }
        assert_auditable_candidate_evidence(report)
        assert_report_safe(
            report,
            sensitive_values=collect_l1_forbidden_values(self.group_id),
        )
        return report

    def write_report(self, report: dict[str, Any]) -> AuditWriteResult:
        if not self.enabled:
            raise HumanAuditError("audit report is disabled")
        forbidden_values = collect_l1_forbidden_values(self.group_id)
        assert_auditable_candidate_evidence(report)
        assert_report_safe(report, sensitive_values=forbidden_values)
        output_dir = _resolve_output_dir(self.output_dir)
        output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        published_dir = output_dir / f"req013-audit-{self.run_id}"
        if published_dir.exists():
            raise HumanAuditError("audit run_id collision; existing report preserved")
        markdown = render_markdown(report)
        json_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        assert_report_safe(markdown, sensitive_values=forbidden_values)

        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".req013-audit-{self.run_id}-",
                suffix=".tmp",
                dir=output_dir,
            )
        )
        try:
            md_path = staging_dir / f"req013-audit-{self.run_id}.md"
            json_path = staging_dir / f"req013-audit-{self.run_id}.json"
            ready_path = staging_dir / READY_FILENAME
            _write_final_file(md_path, markdown)
            _write_final_file(json_path, json_text)
            os.chmod(md_path, 0o600)
            os.chmod(json_path, 0o600)
            ready = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "files": {
                    "markdown": {
                        "name": md_path.name,
                        "size": md_path.stat().st_size,
                        "sha256": _sha256_file(md_path),
                    },
                    "json": {
                        "name": json_path.name,
                        "size": json_path.stat().st_size,
                        "sha256": _sha256_file(json_path),
                    },
                },
            }
            _write_final_file(
                ready_path,
                json.dumps(ready, ensure_ascii=True, sort_keys=True) + "\n",
            )
            os.chmod(ready_path, 0o600)
            _fsync_directory(staging_dir)
            # The non-empty, complete directory (including READY) becomes
            # visible in one rename. Existing run IDs are never replaced.
            if published_dir.exists():
                raise HumanAuditError("audit run_id collision; existing report preserved")
            os.rename(staging_dir, published_dir)
            _fsync_directory(output_dir)
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise

        md_path = published_dir / f"req013-audit-{self.run_id}.md"
        json_path = published_dir / f"req013-audit-{self.run_id}.json"
        ready_path = published_dir / READY_FILENAME
        return AuditWriteResult(
            markdown_path=md_path,
            json_path=json_path,
            markdown_size=md_path.stat().st_size,
            json_size=json_path.stat().st_size,
            markdown_sha256=_sha256_file(md_path),
            json_sha256=_sha256_file(json_path),
            ready_path=ready_path,
            ready_sha256=_sha256_file(ready_path),
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_output_dir(explicit: Path | None) -> Path:
    configured = explicit
    if configured is None:
        raw = os.environ.get(AUDIT_OUTPUT_ENV, "").strip()
        configured = Path(raw).expanduser() if raw else _repo_root() / DEFAULT_AUDIT_RELATIVE_DIR
    resolved = configured.resolve()
    repo = _repo_root().resolve()
    try:
        relative = resolved.relative_to(repo)
    except ValueError:
        return resolved
    if not relative.parts or relative.parts[0] != ".local-artifacts":
        raise HumanAuditError(
            "audit output inside repository must be under Git-ignored .local-artifacts"
        )
    return resolved


def _write_final_file(path: Path, content: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def markdown_safe_text(value: Any) -> str:
    """Render untrusted model/fixture text without active Markdown constructs."""
    text = str(value if value is not None else "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\", "\\\\")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("\n", " ↵ ")
    text = re.sub(r"[ \t\f\v]+", " ", text).strip()
    for char in "`*_{}[]()#+-.!|~:":
        text = text.replace(char, "\\" + char)
    return text


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "# Group Agent Real-LLM 内容审计报告",
        "",
        "## 基本信息",
        "",
        f"- Run ID：{markdown_safe_text(report['run_id'])}",
        f"- 生成时间：{markdown_safe_text(report['generated_at'])}",
        f"- Provider / Model：{markdown_safe_text(metadata['provider'])} / "
        f"{markdown_safe_text(metadata['model'])}",
        f"- Base URL configured：`{metadata['base_url_configured']}`",
        f"- Fixture / Group / Caller：{markdown_safe_text(metadata['fixture_level'])} / "
        f"{markdown_safe_text(metadata['group_id'])} / "
        f"{markdown_safe_text(metadata['mock_caller_id'])}",
        f"- LLM 调用 / Token / 总耗时：`{metadata['llm_invocations']}` / "
        f"`{metadata['token_usage']}` / `{metadata['total_time_s']}s`",
        "",
        "## Machine Oracle 摘要",
        "",
    ]
    for key, value in report["machine_oracles"].items():
        lines.append(f"- {markdown_safe_text(key)}：{markdown_safe_text(value)}")

    for turn in report["rounds"]:
        lines.extend(
            [
                "",
                f"## Round {turn['round']}",
                "",
                "### 用户输入",
                "",
                markdown_safe_text(turn["user_input"]),
                "",
                "### Assistant 实际回复高价值片段",
                "",
            ]
        )
        if turn["assistant_excerpt"]:
            lines.extend(
                f"> {markdown_safe_text(sentence)}"
                for sentence in turn["assistant_excerpt"]
            )
        else:
            lines.append("> （无可安全保留的完整句）")
        lines.extend(
            [
                "",
                f"- 原始 reply：`chars={turn['reply_original_chars']}` / "
                f"`sha256={turn['reply_sha256']}` / "
                f"`coverage={turn['reply_coverage_ratio']}`",
                "- 本轮 LLM/tool delta："
                f"{markdown_safe_text(json.dumps(turn['llm_delta'], ensure_ascii=False, sort_keys=True))}",
                "",
                "### 画像与状态变化",
                "",
                "| 字段 | Before | After | Changed |",
                "|---|---|---|---|",
            ]
        )
        for name, change in turn["profile_diff"]["fields"].items():
            lines.append(
                f"| {markdown_safe_text(name)} | "
                f"{markdown_safe_text(change['before'])} | "
                f"{markdown_safe_text(change['after'])} | "
                f"{markdown_safe_text(change['changed'])} |"
            )
        if turn["round"] == 3:
            lines.extend(["", "### 推荐与邀请", ""])
            for candidate in turn["candidates"]:
                basis = markdown_safe_text(
                    json.dumps(
                        candidate["public_match_basis"],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                lines.append(
                    f"- {markdown_safe_text(candidate['user_id'])} / "
                    f"{markdown_safe_text(candidate['source_group_id'])}：{basis}"
                )
            lines.extend(["", "#### Mentioned 候选逐一公开依据", ""])
            for evidence in turn["mentioned_evidence"]:
                basis = markdown_safe_text(
                    json.dumps(
                        evidence["public_match_basis"],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                lines.append(
                    f"- @{markdown_safe_text(evidence['user_id'])}：{basis}"
                )
            lines.extend(["", "#### Invite 实际高价值片段", ""])
            if turn["invite_excerpt"]:
                lines.extend(
                    f"> {markdown_safe_text(sentence)}"
                    for sentence in turn["invite_excerpt"]
                )
            else:
                lines.append("> （无邀请片段）")
            lines.append(
                f"- 原始 invite：`chars={turn['invite_original_chars']}` / "
                f"`sha256={turn['invite_sha256']}` / "
                f"`coverage={turn['invite_coverage_ratio']}`"
            )
        lines.extend(["", "### 自动内容检查", ""])
        for key, value in turn["automatic_checks"].items():
            lines.append(f"- {markdown_safe_text(key)}：{markdown_safe_text(value)}")
        lines.extend(
            [
                "",
                "### 人工评分",
                "",
                "- 易理解：__/5",
                "- 准确：__/5",
                "- 有帮助：__/5",
                "- 自然：__/5",
                "- 审核备注：",
            ]
        )
    lines.extend(
        [
            "",
            "## 审计边界",
            "",
            markdown_safe_text(report["human_audit_notice"]),
            "",
        ]
    )
    return "\n".join(lines)
