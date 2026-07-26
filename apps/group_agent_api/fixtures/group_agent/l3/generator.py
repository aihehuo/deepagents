"""L3 Deterministic Fixture Generator for REQ-010 with full spec loading & validation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.group_agent_api.fixtures.loader import FixtureDataSet


def generate_l3_fixture(l3_dir: Path, seed: int = 20260725) -> FixtureDataSet:
    from apps.group_agent_api.fixtures.loader import (
        SCHEMA_VERSION,
        FixtureDataSet,
        FixtureValidationError,
        GroupFixture,
        MemberFixture,
        ScenarioFixture,
    )

    gen_spec_file = l3_dir / "generation_spec.json"
    workload_file = l3_dir / "workload.json"
    adv_file = l3_dir / "adversarial_cases.json"

    for fpath in (gen_spec_file, workload_file, adv_file):
        if not fpath.exists():
            raise FixtureValidationError(f"Required L3 spec file missing: {fpath}")

    gen_spec_data = json.loads(gen_spec_file.read_text(encoding="utf-8"))
    workload_data = json.loads(workload_file.read_text(encoding="utf-8"))
    adv_data = json.loads(adv_file.read_text(encoding="utf-8"))

    # Validate schema_version & level for all 3 L3 spec files
    for fname, data in [
        ("generation_spec.json", gen_spec_data),
        ("workload.json", workload_data),
        ("adversarial_cases.json", adv_data),
    ]:
        ver = data.get("schema_version")
        if ver != SCHEMA_VERSION:
            raise FixtureValidationError(f"File '{fname}' has invalid schema_version '{ver}', expected '{SCHEMA_VERSION}'")
        lvl = data.get("level")
        if lvl and lvl.upper() != "L3":
            raise FixtureValidationError(f"File '{fname}' has invalid level '{lvl}', expected 'L3'")

    profile_spec = gen_spec_data.get("profile", {})
    num_groups = profile_spec.get("num_groups", 30)
    num_members = profile_spec.get("num_members", 10000)

    workload_spec = workload_data.get("workload_spec", {})
    num_scenarios = workload_spec.get("concurrency_limit", 300)
    adversarial_cases = adv_data.get("adversarial_cases", [])

    rng = random.Random(seed)

    groups: dict[str, GroupFixture] = {}
    for g_idx in range(1, num_groups + 1):
        gid = f"group_l3_{g_idx:02d}"
        groups[gid] = GroupFixture(
            group_id=gid,
            name=f"L3 Large Group {g_idx:02d}",
            description=f"Synthetic large scale group {g_idx}",
            topic_tags=[f"tag_{g_idx % 10}", f"domain_{g_idx % 5}"],
        )

    members: dict[str, MemberFixture] = {}
    # Distribute members across groups deterministically
    group_ids = list(groups.keys())
    for m_idx in range(1, num_members + 1):
        uid = f"user_l3_{m_idx:05d}"
        gid = group_ids[(m_idx - 1) % num_groups]
        m_key = f"{gid}:{uid}"

        skill_tag = f"skill_{m_idx % 20}"
        members[m_key] = MemberFixture(
            user_id=uid,
            group_id=gid,
            display_name=f"L3 User {m_idx}",
            role="owner" if m_idx <= num_groups else "member",
            profile={
                "doing": f"Specialist in {skill_tag}",
                "need": f"Need partner for skill_{(m_idx + 1) % 20}",
                "offer": f"Offering expertise in {skill_tag}",
            },
            disclosure_level="confirmed_public" if m_idx % 4 != 0 else "match_only",
            phone=f"+86188{m_idx:08d}" if m_idx % 10 == 0 else None,
        )

    scenarios: list[ScenarioFixture] = []
    for sc_idx in range(1, num_scenarios + 1):
        gid = group_ids[(sc_idx - 1) % num_groups]
        caller_id = f"user_l3_{sc_idx:05d}"
        scenarios.append(
            ScenarioFixture(
                scenario_id=f"l3_sc_{sc_idx:03d}",
                title=f"L3 Concurrent Scenario {sc_idx}",
                caller_user_id=caller_id,
                group_id=gid,
                conversation_id=f"conv_l3_{sc_idx:03d}",
                messages=[f"L3 benchmark query for skill_{sc_idx % 20}"],
                expected_disclosure_max="confirmed_public",
            )
        )

    return FixtureDataSet(
        schema_version=SCHEMA_VERSION,
        level="L3",
        seed=seed,
        groups=groups,
        members=members,
        scenarios=scenarios,
        raw_data={
            "generation_spec": gen_spec_data,
            "workload": workload_spec,
            "adversarial_cases": adversarial_cases,
            "seed": seed,
        },
    )
