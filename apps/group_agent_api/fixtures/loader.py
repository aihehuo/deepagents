"""REQ-010 Three-Level Mock Fixture Loader for group_agent_api."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "GA-FIXTURE-V1"
SENSITIVE_FIELDS = {"phone", "mobile", "wechat", "wx_id", "email", "private_notes", "secret", "password"}
VALID_DISCLOSURE_LEVELS = {"confirmed_public", "match_only", "inferred_unconfirmed"}
VALID_MEMBERSHIPS = {"in_group", "not_in_group"}
VALID_ROLES = {"owner", "admin", "member", "guest"}

LevelType = Literal["L1", "L2", "L3"]


class FixtureSecurityError(RuntimeError):
    """Raised when fixture security assertions fail or unsafe access is attempted."""


class FixtureValidationError(ValueError):
    """Raised when fixture schema, ID uniqueness, or group membership validation fails."""


@dataclass
class GroupFixture:
    group_id: str
    name: str
    description: str = ""
    topic_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemberFixture:
    user_id: str
    group_id: str
    display_name: str
    bound: bool = True
    membership: str = "in_group"  # in_group, not_in_group
    reachable: bool = True
    unionid: str | None = None
    role: str = "member"
    profile: dict[str, Any] = field(default_factory=dict)
    disclosure_level: str = "confirmed_public"  # confirmed_public, match_only, inferred_unconfirmed
    phone: str | None = None
    wechat: str | None = None
    email: str | None = None
    private_notes: str | None = None

    def safe_profile_for_llm(self) -> dict[str, Any]:
        """Return profile stripping out all sensitive fields before prompt/callback assembly."""
        safe = {k: v for k, v in self.profile.items() if k.lower() not in SENSITIVE_FIELDS}
        return safe


@dataclass
class ScenarioFixture:
    scenario_id: str
    title: str
    caller_user_id: str
    group_id: str
    conversation_id: str = "default"
    messages: list[str] = field(default_factory=list)
    expected_matches: list[str] = field(default_factory=list)
    forbidden_matches: list[str] = field(default_factory=list)
    expected_at_users: list[str] = field(default_factory=list)
    expected_disclosure_max: str = "confirmed_public"
    multi_round: bool = False


@dataclass
class FixtureDataSet:
    schema_version: str
    level: LevelType
    seed: int
    groups: dict[str, GroupFixture]
    members: dict[str, MemberFixture]  # key: f"{group_id}:{user_id}"
    scenarios: list[ScenarioFixture]
    raw_data: dict[str, Any] = field(default_factory=dict)

    def filter_candidates_for_group(
        self,
        group_id: str,
        raw_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Authoritative Membership Guard: candidate must exist in members table for group_id and be bound/in_group."""
        filtered = []
        for cand in raw_candidates:
            cand_id = cand.get("user_id") or cand.get("id")
            if not cand_id:
                continue
            m_key = f"{group_id}:{cand_id}"
            member = self.members.get(m_key)
            if member and member.bound and member.membership == "in_group" and member.group_id == group_id:
                filtered.append(cand)
        return filtered


def assert_fixture_environment_allowed() -> None:
    """Fail-closed check: Fixtures are ONLY allowed when GROUP_AGENT_INTEGRATION=stub and in explicit test/dev env."""
    raw_env = os.environ.get("GROUP_AGENT_ENV") or os.environ.get("APP_ENV")
    raw_integration = os.environ.get("GROUP_AGENT_INTEGRATION")

    if not raw_env or not raw_integration:
        raise FixtureSecurityError(
            "Fixture loading requires explicit GROUP_AGENT_ENV and GROUP_AGENT_INTEGRATION environment variables."
        )

    env = raw_env.strip().lower()
    integration = raw_integration.strip().lower()

    if integration != "stub" or env in {"production", "prod"}:
        raise FixtureSecurityError(
            f"Fixture loading forbidden in integration={integration}, env={env}. "
            "Must be integration=stub and env!=production."
        )


def _base_fixtures_dir() -> Path:
    env_dir = os.environ.get("GROUP_AGENT_MOCK_FIXTURE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).parent / "group_agent"


def load_fixture(
    level: LevelType = "L1",
    fixture_dir: Path | str | None = None,
    seed: int = 20260725,
) -> FixtureDataSet:
    """Load and validate fixture dataset for the given level with full schema/ID/reference checking."""
    assert_fixture_environment_allowed()

    base_dir = Path(fixture_dir) if fixture_dir else _base_fixtures_dir()
    level_dir = base_dir / level.lower()

    if level == "L3":
        from apps.group_agent_api.fixtures.group_agent.l3.generator import generate_l3_fixture
        return generate_l3_fixture(level_dir, seed=seed)

    if not level_dir.exists():
        raise FixtureValidationError(f"Fixture directory not found for level {level}: {level_dir}")

    groups_file = level_dir / "groups.json"
    members_file = level_dir / "members.json"
    scenarios_file = level_dir / "scenarios.json"

    for fpath in (groups_file, members_file, scenarios_file):
        if not fpath.exists():
            raise FixtureValidationError(f"Required fixture file missing: {fpath}")

    groups_raw = json.loads(groups_file.read_text(encoding="utf-8"))
    members_raw = json.loads(members_file.read_text(encoding="utf-8"))
    scenarios_raw = json.loads(scenarios_file.read_text(encoding="utf-8"))

    # Multi-file Schema & Level validation
    for fname, fcontent in [("groups.json", groups_raw), ("members.json", members_raw), ("scenarios.json", scenarios_raw)]:
        ver = fcontent.get("schema_version")
        if ver != SCHEMA_VERSION:
            raise FixtureValidationError(f"File '{fname}' has invalid schema_version '{ver}', expected '{SCHEMA_VERSION}'")
        file_lvl = fcontent.get("level")
        if file_lvl and file_lvl.upper() != level.upper():
            raise FixtureValidationError(f"File '{fname}' has mismatched level '{file_lvl}', expected '{level}'")

    if not isinstance(groups_raw.get("groups"), list) or not groups_raw.get("groups"):
        raise FixtureValidationError("Top-level 'groups' must be a non-empty list")
    if not isinstance(members_raw.get("members"), list) or not members_raw.get("members"):
        raise FixtureValidationError("Top-level 'members' must be a non-empty list")
    if not isinstance(scenarios_raw.get("scenarios"), list) or not scenarios_raw.get("scenarios"):
        raise FixtureValidationError("Top-level 'scenarios' must be a non-empty list")

    # Load groups
    groups: dict[str, GroupFixture] = {}
    for g in groups_raw.get("groups", []):
        gid = g.get("group_id")
        if not gid:
            raise FixtureValidationError("Group record missing 'group_id'")
        if gid in groups:
            raise FixtureValidationError(f"Duplicate group_id: {gid}")
        groups[gid] = GroupFixture(
            group_id=gid,
            name=g.get("name", gid),
            description=g.get("description", ""),
            topic_tags=g.get("topic_tags", []),
            metadata=g.get("metadata", {}),
        )

    # Load members & validate relationship
    members: dict[str, MemberFixture] = {}
    for m in members_raw.get("members", []):
        uid = m.get("user_id")
        gid = m.get("group_id")
        if not uid or not gid:
            raise FixtureValidationError(f"Member record missing user_id or group_id: {m}")
        if gid not in groups:
            raise FixtureValidationError(f"Member '{uid}' references non-existent group '{gid}'")
        m_key = f"{gid}:{uid}"
        if m_key in members:
            raise FixtureValidationError(f"Duplicate member key: {m_key}")

        disc_level = m.get("disclosure_level", "confirmed_public")
        if disc_level not in VALID_DISCLOSURE_LEVELS:
            raise FixtureValidationError(f"Member '{uid}' has invalid disclosure_level '{disc_level}'")

        bound_raw = m.get("bound", True)
        if not isinstance(bound_raw, bool):
            raise FixtureValidationError(f"Member '{uid}' bound field must be a JSON boolean, got {type(bound_raw)}")
        reachable_raw = m.get("reachable", True)
        if not isinstance(reachable_raw, bool):
            raise FixtureValidationError(f"Member '{uid}' reachable field must be a JSON boolean, got {type(reachable_raw)}")

        membership_val = str(m.get("membership", "in_group"))
        if membership_val not in VALID_MEMBERSHIPS:
            raise FixtureValidationError(f"Member '{uid}' has invalid membership '{membership_val}'")

        role_val = str(m.get("role", "member"))
        if role_val not in VALID_ROLES:
            raise FixtureValidationError(f"Member '{uid}' has invalid role '{role_val}'")

        profile_raw = m.get("profile", {})
        if not isinstance(profile_raw, dict):
            raise FixtureValidationError(f"Member '{uid}' profile must be a dict, got {type(profile_raw)}")

        members[m_key] = MemberFixture(
            user_id=uid,
            group_id=gid,
            display_name=m.get("display_name", uid),
            bound=bound_raw,
            membership=membership_val,
            reachable=reachable_raw,
            unionid=m.get("unionid"),
            role=role_val,
            profile=profile_raw,
            disclosure_level=disc_level,
            phone=m.get("phone"),
            wechat=m.get("wechat"),
            email=m.get("email"),
            private_notes=m.get("private_notes"),
        )

    # Load scenarios & validate references
    scenarios: list[ScenarioFixture] = []
    seen_scenario_ids = set()
    for sc in scenarios_raw.get("scenarios", []):
        sc_id = sc.get("scenario_id")
        caller = sc.get("caller_user_id")
        gid = sc.get("group_id")
        if not sc_id or not caller or not gid:
            raise FixtureValidationError(f"Scenario record missing required fields: {sc}")
        if sc_id in seen_scenario_ids:
            raise FixtureValidationError(f"Duplicate scenario_id: {sc_id}")
        seen_scenario_ids.add(sc_id)

        if gid not in groups:
            raise FixtureValidationError(f"Scenario '{sc_id}' references non-existent group '{gid}'")

        if not any(mem.user_id == caller for mem in members.values()):
            raise FixtureValidationError(f"Scenario '{sc_id}' references unknown caller_user_id '{caller}'")

        # Validate reference existence in members
        for match_user_id in (sc.get("expected_matches", []) + sc.get("forbidden_matches", []) + sc.get("expected_at_users", [])):
            if not any(mem.user_id == match_user_id for mem in members.values()):
                raise FixtureValidationError(f"Scenario '{sc_id}' references unknown user '{match_user_id}'")

        scenarios.append(
            ScenarioFixture(
                scenario_id=sc_id,
                title=sc.get("title", sc_id),
                caller_user_id=caller,
                group_id=gid,
                conversation_id=sc.get("conversation_id", "default"),
                messages=sc.get("messages", []),
                expected_matches=sc.get("expected_matches", []),
                forbidden_matches=sc.get("forbidden_matches", []),
                expected_at_users=sc.get("expected_at_users", []),
                expected_disclosure_max=sc.get("expected_disclosure_max", "confirmed_public"),
                multi_round=sc.get("multi_round", False),
            )
        )

    return FixtureDataSet(
        schema_version=SCHEMA_VERSION,
        level=level,
        seed=seed,
        groups=groups,
        members=members,
        scenarios=scenarios,
        raw_data={"groups": groups_raw, "members": members_raw, "scenarios": scenarios_raw},
    )
