"""Brain-layer eval for Wu Tanchang prompt modules.

Each persona runs three isolated nodes (one module on) against the live model.
Deterministic rubrics score the raw output; the pytest writes Markdown + JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

EIGHT_DIM_FULL_LABELS = (
    "行业赛道与品类战略",
    "市场细分与目标客群定位",
    "业务设计与菜单工程",
    "品牌叙事与价值传递",
    "全触点体验与峰值设计",
    "空间场景与高效动线",
    "运营系统与服务标准",
    "投资测算与模型迭代",
)

REPORT_DIR = (
    Path(__file__).resolve().parents[2] / "docs" / "test_reports"
)
REPORT_STEM = "wu-tanchang-brain-eval"


@dataclass(frozen=True)
class Check:
    id: str
    ok: bool
    detail: str
    critical: bool = True


@dataclass
class NodeResult:
    node: str
    tools: list[str]
    output: str
    checks: list[Check]
    elapsed_s: float

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks if c.critical)

    @property
    def score(self) -> tuple[int, int]:
        crit = [c for c in self.checks if c.critical]
        return sum(1 for c in crit if c.ok), len(crit)


@dataclass
class PersonaSpec:
    id: str
    title: str
    city: str
    category: str
    stage: str
    evidence_terms: tuple[str, ...]
    expect_kb_hit: bool
    kb_brand_hints: tuple[str, ...]
    profile_script: tuple[str, ...]
    anchor_brief: str
    cases_prompt: str


@dataclass
class PersonaRun:
    spec: PersonaSpec
    nodes: dict[str, NodeResult] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.nodes) and all(n.passed for n in self.nodes.values())


PERSONAS: tuple[PersonaSpec, ...] = (
    PersonaSpec(
        id="sh_hotpot_idea",
        title="上海 · 想法期 · 火锅",
        city="上海",
        category="火锅",
        stage="想法期",
        evidence_terms=("选址", "客单", "80万", "后厨"),
        expect_kb_hit=True,
        kb_brand_hints=("大虎",),
        profile_script=(
            "你好，我想在上海开一家火锅店，预算大概80万。我做过两年后厨管理，没自己开过店。"
            "最担心选址和客单价撑不住。",
            "还在想法阶段，团队就我自己，对标品牌还没定。",
            "真正卡住的是选址和定价，希望面谈时帮我判断能不能做、客单大概要定多少。",
            "目标是先活下来，计划半年内选址装修。深度已经够了，请立即输出会议准备材料，不要再提问。",
        ),
        anchor_brief=(
            "下面是已经收集好的用户画像，请按八维锚点模块写出会议准备材料（只要二、三两节）：\n\n"
            "城市：上海\n创业背景：餐饮后厨管理两年，未自己开过店\n选择赛道：火锅\n"
            "阶段：想法期\n定义问题：选址和客单价能否撑住\n聚焦目标：先让单店活下来\n"
            "实施计划：半年内选址装修，预算约80万\n\n"
            "请挂 1–3 个有证据的八维锚点，说明见面准备什么。不要打分，不要写成最终方案。"
        ),
        cases_prompt=(
            "我打算在上海开火锅店，预算80万，有后厨背景。吴探长是否有相关的案例？"
            "请检索知识库后回答，点名命中的品牌，有探店笔记网页就写成 Markdown 链接。"
        ),
    ),
    PersonaSpec(
        id="hz_bakery_prep",
        title="杭州 · 筹备中 · 烘焙",
        city="杭州",
        category="烘焙",
        stage="筹备中",
        evidence_terms=("合伙", "铺位", "回本", "烘焙"),
        expect_kb_hit=True,
        kb_brand_hints=("Punch Monday", "Bake", "面包", "吐司", "烘焙"),
        profile_script=(
            "我在杭州和朋友合伙开烘焙店，铺位基本谈下来了，滨江一个社区底商，大概80平。",
            "筹备中，设备还没订。我之前做电商，合伙人会做蛋糕。最怕算不清回本。",
            "真正卡住的是投资模型和产品线该做多宽，面谈想请吴探长帮看这账能不能打平。",
            "目标12个月内回本，计划两个月内装修开业。深度够了，请立即输出会议准备材料，不要再提问。",
        ),
        anchor_brief=(
            "下面是已经收集好的用户画像，请按八维锚点模块写出会议准备材料（只要二、三两节）：\n\n"
            "城市：杭州滨江\n创业背景：电商转行，合伙人会做蛋糕\n选择赛道：烘焙\n"
            "阶段：筹备中\n定义问题：投资模型与产品线宽度，账能不能打平\n"
            "聚焦目标：12个月内回本\n实施计划：社区底商约80平，两个月内装修开业\n\n"
            "请挂 1–3 个有证据的八维锚点。不要打分，不要写成最终方案。"
        ),
        cases_prompt=(
            "杭州滨江社区底商开烘焙店，合伙、筹备中。吴探长知识库有没有烘焙/面包相关探店笔记？"
            "请检索后回答，点名命中品牌，有网页就写成 Markdown 链接。"
        ),
    ),
    PersonaSpec(
        id="cd_tea_bottleneck",
        title="成都 · 已开店瓶颈 · 茶饮",
        city="成都",
        category="茶饮",
        stage="已开店/瓶颈期",
        evidence_terms=("客单", "三家", "员工", "杨枝甘露"),
        expect_kb_hit=True,
        kb_brand_hints=("茶饮", "大悦城", "茶"),
        profile_script=(
            "我在成都开了三家茶饮店，招牌是杨枝甘露。客单从22掉到16，员工流动很大。",
            "已开两年，标准店大约40平。利润主要靠水果茶，但现在全是折扣战。",
            "真正卡住的是人效和菜单是不是该砍SKU，希望面谈帮我判断要不要关一家、产品怎么收。",
            "目标是先把单店利润拉回正，暂不准备第四家。深度够了，请立即输出会议准备材料，不要再提问。",
        ),
        anchor_brief=(
            "下面是已经收集好的用户画像，请按八维锚点模块写出会议准备材料（只要二、三两节）：\n\n"
            "城市：成都\n创业背景：已开三家茶饮，招牌杨枝甘露\n选择赛道：茶饮\n"
            "阶段：已开店/瓶颈期\n定义问题：客单下跌、员工流动、是否砍SKU/关店\n"
            "聚焦目标：单店利润转正，暂不扩张\n实施计划：先优化人效和菜单\n"
            "门店：三家，约40平，利润来自水果茶\n\n"
            "请挂 1–3 个有证据的八维锚点。不要打分，不要写成最终方案。"
        ),
        cases_prompt=(
            "成都三家茶饮店遇到客单下跌。吴探长知识库有没有茶饮相关笔记？"
            "请检索后回答，点名命中内容，有网页就写成 Markdown 链接。"
        ),
    ),
    PersonaSpec(
        id="cs_bbq_idea",
        title="长沙 · 想法期 · 烧烤",
        city="长沙",
        category="烧烤",
        stage="想法期",
        evidence_terms=("夜宵", "租金", "烧烤"),
        expect_kb_hit=False,
        kb_brand_hints=(),
        profile_script=(
            "我想在长沙开夜宵烧烤，看中了大学城一条街，租金不便宜。自己没做过餐饮。",
            "还在想法阶段，就我一个人，资金大约50万。",
            "真正卡住的是外行能不能做、租金吃不吃得消，希望面谈帮我判断要不要上。",
            "目标先活过第一年。深度够了，请立即输出会议准备材料，不要再提问。",
        ),
        anchor_brief=(
            "下面是已经收集好的用户画像，请按八维锚点模块写出会议准备材料（只要二、三两节）：\n\n"
            "城市：长沙大学城\n创业背景：无餐饮经验，个人创业\n选择赛道：夜宵烧烤\n"
            "阶段：想法期\n定义问题：外行能否做、租金能否撑住\n"
            "聚焦目标：活过第一年\n实施计划：资金约50万，铺位尚未签\n\n"
            "请挂 1–3 个有证据的八维锚点。不要打分，不要写成最终方案。"
        ),
        cases_prompt=(
            "长沙大学城开夜宵烧烤，预算50万，没餐饮经验。吴探长知识库有没有烧烤相关案例？"
            "请检索后回答。命中了就点名并给 Markdown 链接；没命中不要编造火锅店来凑。"
        ),
    ),
)


def _has_question(text: str) -> bool:
    return any(mark in text for mark in ("？", "吗", "呢", "么"))


def _stuffed_labels(text: str) -> list[str]:
    return [label for label in EIGHT_DIM_FULL_LABELS if label in text]


def evaluate_profile(spec: PersonaSpec, *, tools: list[str], text: str) -> list[Check]:
    return [
        Check(
            "no_kb_tools",
            not any(t in tools for t in ("task", "kb_semantic_search", "get_note_content")),
            f"tools={tools}",
        ),
        Check("section_one", "一、" in text and ("交谈要点" in text or spec.city in text), "需含「一、交谈要点」或城市"),
        Check("city", spec.city in text, f"需出现城市 {spec.city}"),
        Check("category", spec.category in text, f"需出现赛道 {spec.category}"),
        Check("no_section_four", "四、可参考案例" not in text, "画像节点不得写第四节"),
        Check("no_score_sheet", "八维诊断评分" not in text and "0-10分" not in text and "0–10分" not in text, "不得打分量表"),
        Check(
            "first_turn_questioned",
            True,
            "多轮追问由调用方按 turn 另记",
            critical=False,
        ),
    ]


def evaluate_profile_first_turn(text: str) -> list[Check]:
    return [
        Check("recap_or_city", True, "调用方断言"),
        Check("asks_one_question", _has_question(text), "首轮应追问"),
        Check("no_score_in_chat", "八维诊断评分" not in text, "收集阶段不念八维评分"),
    ]


def evaluate_anchors(spec: PersonaSpec, *, tools: list[str], text: str) -> list[Check]:
    stuffed = _stuffed_labels(text)
    evidence_hits = [term for term in spec.evidence_terms if term in text]
    return [
        Check("no_kb_tools", "kb_semantic_search" not in tools and "task" not in tools, f"tools={tools}"),
        Check("section_two", "二、" in text, "需含「二、」"),
        Check("section_three", "三、" in text, "需含「三、」"),
        Check(
            "one_to_three_full_labels",
            1 <= len(stuffed) <= 3,
            f"完整锚点名 {len(stuffed)} 个: {stuffed}",
        ),
        Check("evidence_grounded", len(evidence_hits) >= 1, f"回指画像词: {evidence_hits}"),
        Check("no_score", "0-10分" not in text and "0–10分" not in text and "八维诊断评分" not in text, "不打分"),
        Check("not_final_plan", "最终方案已确定" not in text, "不得写成已交付方案"),
    ]


def _denied_user_category(text: str, category: str) -> bool:
    """True if the model claims the KB has no notes in the user's category."""
    if f"有没有{category}" in text and text.count(f"没有{category}") == text.count(
        f"有没有{category}"
    ):
        # Only the question form "有没有火锅", not a denial.
        extra = text.replace(f"有没有{category}", "")
        needles = (f"没有{category}", f"暂无{category}", "没有该品类", "暂无该品类")
        return any(n in extra for n in needles)
    needles = (
        f"没有{category}",
        f"暂无{category}",
        "没有该品类",
        "暂无该品类",
        "库里没有该品类",
    )
    return any(n in text for n in needles)


def evaluate_cases(spec: PersonaSpec, *, tools: list[str], text: str) -> list[Check]:
    used_search = any(n in {"task", "kb_semantic_search", "get_note_content"} for n in tools)
    denied = _denied_user_category(text, spec.category)
    link = bool(re.search(r"\[[^\]]+\]\(https://xhslink\.cn/o/[^)]+\)", text))
    brand_hit = any(b in text for b in spec.kb_brand_hints) if spec.kb_brand_hints else False
    invented_hotpot = (not spec.expect_kb_hit) and ("大虎" in text) and spec.category != "火锅"
    checks = [
        Check("used_search", used_search, f"tools={tools}"),
        Check(
            "no_false_deny",
            not (spec.expect_kb_hit and denied and not brand_hit),
            f"denied={denied} brand_hit={brand_hit}",
        ),
        Check("no_invented_hotpot", not invented_hotpot, "未命中烧烤时不得拿大虎火锅凑数"),
    ]
    if spec.expect_kb_hit:
        checks.append(Check("named_hit", brand_hit or spec.category in text, f"hints={spec.kb_brand_hints}"))
        checks.append(
            Check(
                "markdown_xhs",
                link,
                "命中时应给出 [品牌 · 看探店笔记](https://xhslink.cn/o/…)",
                critical=False,
            )
        )
    return checks


def render_markdown(runs: list[PersonaRun], *, model: str, started: str) -> str:
    beijing = timezone(timedelta(hours=8))
    now = datetime.now(beijing).strftime("%Y-%m-%d %H:%M")
    total_nodes = sum(len(r.nodes) for r in runs)
    passed_nodes = sum(1 for r in runs for n in r.nodes.values() if n.passed)
    passed_personas = sum(1 for r in runs if r.passed)

    lines = [
        "# 吴探长 brain 层 · 多客户节点评测报告",
        "",
        f"- **生成时间**：{now}（Asia/Shanghai）",
        f"- **评测启动**：{started}",
        f"- **模型**：{model}",
        f"- **入口**：`create_agent`（与生产同一套 persona + 模块开关），每节点只开对应模块",
        f"- **客户样本**：{len(runs)}；节点通过 **{passed_nodes}/{total_nodes}**；画像全绿 **{passed_personas}/{len(runs)}**",
        f"- **评估方式**：对真实模型原文做规则评分（工具、章节、锚点数量、证据词、检索短链）；不是另一模型打分",
        "",
        "## 总表",
        "",
        "| 客户 | 画像 | 八维锚点 | 知识库案例 | 画像全绿 |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        cells = [run.spec.title]
        for node in ("profile", "eight_dim_anchors", "kb_cases"):
            result = run.nodes.get(node)
            if result is None:
                cells.append("—")
            else:
                got, need = result.score
                mark = "通过" if result.passed else "未过"
                cells.append(f"{mark} {got}/{need}")
        cells.append("是" if run.passed else "否")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## 分客户节点", ""]
    for run in runs:
        lines += [f"### {run.spec.title}", "", f"- id: `{run.spec.id}`", f"- 阶段 / 城市 / 赛道：{run.spec.stage} / {run.spec.city} / {run.spec.category}", ""]
        for node_id, title in (
            ("profile", "节点 1 · 画像勾画"),
            ("eight_dim_anchors", "节点 2 · 八维锚点"),
            ("kb_cases", "节点 3 · 知识库案例"),
        ):
            result = run.nodes[node_id]
            got, need = result.score
            lines += [
                f"#### {title}",
                "",
                f"- 结果：**{'通过' if result.passed else '未过'}**（关键项 {got}/{need}，耗时 {result.elapsed_s:.1f}s）",
                f"- 工具：`{', '.join(result.tools) or '（无）'}`",
                "",
                "评估项：",
                "",
            ]
            for check in result.checks:
                flag = "PASS" if check.ok else "FAIL"
                crit = "关键" if check.critical else "参考"
                lines.append(f"- `{flag}` [{crit}] `{check.id}` — {check.detail}")
            excerpt = result.output.strip()[:1600]
            lines += ["", "<details><summary>模型原文摘录</summary>", "", "```", excerpt, "```", "", "</details>", ""]
    lines += [
        "## 怎么复跑",
        "",
        "```bash",
        "cd /Users/yc/workspace/deepagents",
        "WU_E2E_REAL_LLM=1 PYTHONPATH=. .venv/bin/pytest \\",
        "  tests/test_wu_tanchang_brain_eval_real_llm.py -v -s --tb=short",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_reports(runs: list[PersonaRun], *, model: str, started: str) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "started": started,
        "model": model,
        "personas": [
            {
                "id": run.spec.id,
                "title": run.spec.title,
                "city": run.spec.city,
                "category": run.spec.category,
                "stage": run.spec.stage,
                "passed": run.passed,
                "nodes": {
                    node: {
                        "passed": result.passed,
                        "score": list(result.score),
                        "elapsed_s": result.elapsed_s,
                        "tools": result.tools,
                        "checks": [asdict(c) for c in result.checks],
                        "output": result.output,
                    }
                    for node, result in run.nodes.items()
                },
            }
            for run in runs
        ],
    }
    json_path = REPORT_DIR / f"{REPORT_STEM}.json"
    md_path = REPORT_DIR / f"{REPORT_STEM}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(runs, model=model, started=started), encoding="utf-8")
    return md_path, json_path
