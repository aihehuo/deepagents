"""Wu Tanchang brain-layer eval: live LLM per node, rubric, written report.

Opt-in::

    WU_E2E_REAL_LLM=1 PYTHONPATH=. .venv/bin/pytest \\
      tests/test_wu_tanchang_brain_eval_real_llm.py -v -s --tb=short
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.support.wu_tanchang_brain_eval import (
    PERSONAS,
    NodeResult,
    PersonaRun,
    evaluate_anchors,
    evaluate_cases,
    evaluate_profile,
    evaluate_profile_first_turn,
    write_reports,
)
from tests.test_wu_tanchang_prompt_modules_real_llm import (
    _ai_text,
    _invoke,
    _last_ai,
    _make_agent,
    _tool_names,
    module_agent_env,
)

pytestmark = [
    pytest.mark.real_llm,
    pytest.mark.integration,
    pytest.mark.brain_sut,
]


@pytest.mark.timeout(900)
def test_wu_tanchang_brain_eval_writes_report(module_agent_env: Path) -> None:
    """Four customer backgrounds × three nodes; write Markdown/JSON under docs/test_reports."""
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    model = "deepseek-v4-flash"
    runs: list[PersonaRun] = []

    for spec in PERSONAS:
        run = PersonaRun(spec=spec)
        print(f"\n===== {spec.title} =====")

        t0 = time.time()
        agent = _make_agent(module_agent_env, "profile", f"eval_{spec.id}_profile")
        tid = f"wt::eval_{spec.id}::profile"
        messages = []
        first_reply = ""
        for turn, user in enumerate(spec.profile_script, start=1):
            result = _invoke(
                agent,
                thread_id=tid,
                text=user,
                agent_name=f"{spec.id}-profile-t{turn}",
            )
            messages = result.get("messages") or []
            if turn == 1:
                first_reply = _last_ai(messages)
        profile_text = _ai_text(messages)
        profile_tools = _tool_names(messages)
        profile_checks = evaluate_profile_first_turn(first_reply) + evaluate_profile(
            spec, tools=profile_tools, text=profile_text
        )
        # First-turn city/category recap
        recap_ok = spec.city in first_reply or spec.category in first_reply
        profile_checks = [
            c if c.id != "recap_or_city" else type(c)(
                id="recap_or_city",
                ok=recap_ok,
                detail=f"首轮是否点到 {spec.city}/{spec.category}",
                critical=True,
            )
            for c in profile_checks
        ]
        run.nodes["profile"] = NodeResult(
            node="profile",
            tools=profile_tools,
            output=profile_text,
            checks=profile_checks,
            elapsed_s=time.time() - t0,
        )

        t0 = time.time()
        agent = _make_agent(module_agent_env, "eight_dim_anchors", f"eval_{spec.id}_anchors")
        result = _invoke(
            agent,
            thread_id=f"wt::eval_{spec.id}::anchors",
            text=spec.anchor_brief,
            agent_name=f"{spec.id}-anchors",
        )
        messages = result.get("messages") or []
        anchor_text = _ai_text(messages)
        anchor_tools = _tool_names(messages)
        run.nodes["eight_dim_anchors"] = NodeResult(
            node="eight_dim_anchors",
            tools=anchor_tools,
            output=anchor_text,
            checks=evaluate_anchors(spec, tools=anchor_tools, text=anchor_text),
            elapsed_s=time.time() - t0,
        )

        t0 = time.time()
        kb_db = (
            Path(__file__).resolve().parents[1]
            / "apps"
            / "wu_tanchang_api"
            / "workspace"
            / "kb"
        )
        if not (kb_db / "kb.db").exists() or not (kb_db / "chroma").exists():
            pytest.skip("local kb.db / chroma missing")
        agent = _make_agent(module_agent_env, "kb_cases", f"eval_{spec.id}_cases")
        result = _invoke(
            agent,
            thread_id=f"wt::eval_{spec.id}::cases",
            text=spec.cases_prompt,
            agent_name=f"{spec.id}-cases",
        )
        messages = result.get("messages") or []
        cases_text = _ai_text(messages)
        cases_tools = _tool_names(messages)
        run.nodes["kb_cases"] = NodeResult(
            node="kb_cases",
            tools=cases_tools,
            output=cases_text,
            checks=evaluate_cases(spec, tools=cases_tools, text=cases_text),
            elapsed_s=time.time() - t0,
        )

        runs.append(run)

    md_path, json_path = write_reports(runs, model=model, started=started)
    print(f"\n[report] {md_path}")
    print(f"[report] {json_path}")
    assert md_path.exists() and json_path.exists()
    failed = [
        f"{run.spec.id}/{node}"
        for run in runs
        for node, result in run.nodes.items()
        if not result.passed
    ]
    if failed:
        print(f"[eval] quality failures recorded in report: {failed}")
