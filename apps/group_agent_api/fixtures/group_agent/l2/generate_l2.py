"""Generate deterministic L2 fixture data for REQ-010."""

import json
from pathlib import Path

SCHEMA_VERSION = "GA-FIXTURE-V1"

TOPIC_CLUSTERS = [
    ("ai_llm", "AI & LLM Technology", ["python", "pytorch", "langchain", "agent", "rag"]),
    ("ecommerce", "Cross-Border E-Commerce", ["shopify", "amazon", "tiktok", "logistics", "ad_placement"]),
    ("fintech", "FinTech & Web3", ["blockchain", "payment_gateway", "compliance", "defi"]),
    ("saas", "Enterprise B2B SaaS", ["crm", "erp", "devops", "kubernetes", "sales"]),
    ("hardware", "IoT & Robotics", ["embedded", "circuit", "robotics", "ros", "manufacturing"]),
    ("healthcare", "BioTech & Digital Health", ["medical_ai", "fda", "clinical", "genomics"]),
    ("education", "EdTech & Content", ["online_course", "k12", "learning_lms", "video"]),
    ("consumer", "Consumer Brands", ["d2c", "branding", "supply_chain", "packaging"]),
]


def build_l2():
    l2_dir = Path(__file__).parent

    groups = []
    for idx, (code, name, tags) in enumerate(TOPIC_CLUSTERS):
        gid = f"group_l2_{idx + 1:02d}_{code}"
        groups.append({
            "group_id": gid,
            "name": f"L2 Group: {name}",
            "description": f"Dedicated group for {name}",
            "topic_tags": tags,
        })

    members = []
    member_count = 0
    for g_idx, g in enumerate(groups):
        gid = g["group_id"]
        tags = g["topic_tags"]
        for m_idx in range(1, 26):
            member_count += 1
            uid = f"user_l2_{member_count:03d}"
            role = "owner" if m_idx == 1 else ("admin" if m_idx <= 3 else "member")
            skill = tags[(m_idx - 1) % len(tags)]
            disclosure = "confirmed_public" if m_idx % 3 != 0 else ("match_only" if m_idx % 3 == 1 else "inferred_unconfirmed")

            profile = {
                "doing": f"Working on {skill} for group {g['name']}",
                "need": f"Need partner for {tags[(m_idx) % len(tags)]}",
                "offer": f"Expertise in {skill} and {tags[(m_idx + 1) % len(tags)]}",
            }

            member = {
                "user_id": uid,
                "group_id": gid,
                "display_name": f"User {member_count} ({skill})",
                "bound": True,
                "membership": "in_group",
                "reachable": True,
                "role": role,
                "profile": profile,
                "disclosure_level": disclosure,
            }

            if m_idx % 5 == 0:
                member["phone"] = f"+86139{member_count:08d}"
                member["wechat"] = f"wx_{uid}"

            members.append(member)

    # Add dual identity natural persons: 5 users present in 2 groups with different profiles
    dual_users = ["user_l2_005", "user_l2_010", "user_l2_015", "user_l2_020", "user_l2_025"]
    for idx, du in enumerate(dual_users):
        target_group = groups[(idx + 1) % len(groups)]["group_id"]
        members.append({
            "user_id": du,
            "group_id": target_group,
            "display_name": f"Dual Identity {du} in {target_group}",
            "bound": True,
            "membership": "in_group",
            "reachable": True,
            "role": "member",
            "profile": {
                "doing": f"Cross-domain consulting in {target_group}",
                "need": "Co-founder",
                "offer": "Secondary domain skills",
            },
            "disclosure_level": "confirmed_public",
        })

    # 24 scenarios with 3-8 distinct evolving multi-round dialogue messages each
    scenarios = []
    for sc_idx in range(1, 25):
        g_target = groups[(sc_idx - 1) % len(groups)]
        gid = g_target["group_id"]
        caller_id = f"user_l2_{(sc_idx * 5) % 200 + 1:03d}"
        tags = g_target["topic_tags"]
        topic = tags[sc_idx % len(tags)]
        num_rounds = 3 + (sc_idx % 6)  # 3 to 8 rounds

        messages = [
            f"[Round 1] 在 {g_target['name']} 中，我们需要找到关于 {topic} 的技术合伙人或顾问。",
            f"[Round 2] 我们主要专注在 {tags[(sc_idx + 1) % len(tags)]} 的落地上，需要有实际实操经验。",
            f"[Round 3] 项目已经完成 MVP，现在寻找能够全职投入的联合创始人。",
        ]
        if num_rounds >= 4:
            messages.append(f"[Round 4] 最好具备 3 年以上 {topic} 相关项目架构经验，能独立带团队。")
        if num_rounds >= 5:
            messages.append(f"[Round 5] 我们提供股权和基础薪资，群里有合适的人选推荐吗？")
        if num_rounds >= 6:
            messages.append(f"[Round 6] 另外我们也在考察海外市场的推广渠道。")
        if num_rounds >= 7:
            messages.append(f"[Round 7] 有意向的朋友可以随时在群里回复。")
        if num_rounds >= 8:
            messages.append(f"[Round 8] 感谢大家，本轮需求确认完毕。")

        scenarios.append({
            "scenario_id": f"l2_sc_{sc_idx:02d}",
            "title": f"Multi-round scenario {sc_idx} ({num_rounds} rounds) for {topic} in {gid}",
            "caller_user_id": caller_id,
            "group_id": gid,
            "conversation_id": f"conv_l2_{sc_idx:02d}",
            "messages": messages,
            "expected_matches": [],
            "forbidden_matches": [],
            "multi_round": True,
        })

    actors = [
        {"actor_id": f"actor_{idx:02d}", "name": f"Test Actor {idx}"} for idx in range(1, 11)
    ]

    expected_rules = {
        "cross_group_leakage_max": 0,
        "sensitive_field_leakage_max": 0,
        "total_groups": len(groups),
        "total_members": len(members),
        "total_scenarios": len(scenarios),
    }

    (l2_dir / "groups.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "level": "L2", "groups": groups}, indent=2, ensure_ascii=False))
    (l2_dir / "members.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "level": "L2", "members": members}, indent=2, ensure_ascii=False))
    (l2_dir / "actors.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "level": "L2", "level": "L2", "actors": actors}, indent=2, ensure_ascii=False))
    (l2_dir / "scenarios.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "level": "L2", "scenarios": scenarios}, indent=2, ensure_ascii=False))
    (l2_dir / "expected_rules.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "level": "L2", "rules": expected_rules}, indent=2, ensure_ascii=False))
    print(f"Generated L2 fixtures: {len(groups)} groups, {len(members)} members, {len(scenarios)} multi-round scenarios.")


if __name__ == "__main__":
    build_l2()
