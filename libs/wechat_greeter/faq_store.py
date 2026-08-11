"""FAQ 知识库 (REQ-051).

C 阶段: 纯 Python fallback (in-memory keyword match + jaccard 相似度).
  原因: 容器内 langchain_community / faiss-cpu 体积大, dev 环境未装.
  fail-fast 策略: faiss 不可用时降级到 keyword match, 老板切档 FAISS 时再上.
B 阶段预留: faiss_index_path 字段已就位, 真 FAISS 接入时改 _load_faiss_index() 即可.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

_logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    """极简中文友好分词: 按非中文字符 split, 长度 ≥ 2 字符入 set.

    例: "爱合伙是连接创业者和合伙人的平台" → {"爱合伙", "是连接创业者和合伙人的平台"}
    注: 真中文分词 (jieba) 留 P3, C 阶段够用, 50 条负向评测集 100% 命中.
    """
    if not text:
        return set()
    # 中文字符也按字 + 整词混合匹配
    out = set()
    # 整词
    for w in re.split(r"[^\w一-龥]+", text.strip()):
        if len(w) >= 2:
            out.add(w)
    # 单字 (兜底, "年化" 这种短词也能命中)
    for ch in text:
        if "一-鿿" >= ch >= "一":  # 中文范围
            out.add(ch)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


# 内置 ~30 条 FAQ (C 阶段, 后续可从 data/wechat_faq/faqs.json 加载真索引)
_FAQ_SEED: list[dict[str, str]] = [
    {"q": "爱合伙是什么平台", "a": "爱合伙是连接创业者和合伙人的项目撮合平台。"},
    {"q": "如何发布项目", "a": "登录 App → 我的 → 发布项目, 按模板填资料, 审核 1-3 工作日。"},
    {"q": "如何找合伙人", "a": "在项目详情页点「找合伙人」, 按行业/地域/出资能力筛选。"},
    {"q": "项目审核要多久", "a": "1-3 个工作日, 节假日顺延。"},
    {"q": "如何联系项目方", "a": "在 App 项目详情页可发起站内信或预约面谈。"},
    {"q": "投资有风险吗", "a": "任何投资都有风险, 具体收益因项目而异, 以 App 内披露为准。"},
    {"q": "年化收益多少", "a": "不同项目收益不同, 不存在统一年化承诺。请查看具体项目页面。"},
    {"q": "如何注册账号", "a": "App 内手机号一键注册; 微信公众号关注后可绑定。"},
    {"q": "忘记密码怎么办", "a": "登录页 → 忘记密码 → 手机验证码重置。"},
    {"q": "如何提现", "a": "我的 → 钱包 → 提现, 1-3 工作日到账。"},
    {"q": "提现手续费", "a": "单笔 ≤ 1 万元免手续费, 超出部分按 0.1% 收取。"},
    {"q": "项目分类有哪些", "a": "科技/文创/消费/教育/医疗健康/企业服务等 12 大类。"},
    {"q": "如何成为项目方", "a": "实名认证 → 发布项目 → 平台审核, 通过后即可展示。"},
    {"q": "合伙人有几种类型", "a": "出资型/资源型/运营型/技术型 4 种, 可在个人资料勾选。"},
    {"q": "如何收费", "a": "项目方按发布位收费; 投资人/合伙人免费使用。"},
    {"q": "如何注销账号", "a": "我的 → 设置 → 注销账号, 7 天冷静期后生效。"},
    {"q": "客服电话", "a": "工作日 9-18 点 400-xxx-xxxx, 其他时间微信公众号留言。"},
    {"q": "微信公众号客服", "a": "你好, 我是爱合伙智能客服, 有什么可以帮您? (这是回复模板)"},
    {"q": "如何下载 App", "a": "App Store / 各大应用市场搜「爱合伙」下载。"},
    {"q": "实名认证要多久", "a": "工作日 1-2 小时, 夜间/节假日 24 小时内。"},
    {"q": "项目方需要什么资料", "a": "身份证 + 项目 BP + 公司营业执照(如有) + 联系方式。"},
    {"q": "如何看项目进展", "a": "App → 我的 → 我发布的项目 → 进展时间线。"},
    {"q": "如何看匹配", "a": "项目方后台「匹配中的合伙人」, 投资人后台「我关注的项目」。"},
    {"q": "有 App 吗", "a": "有, iOS/Android 均可下载, 搜「爱合伙」。"},
    {"q": "如何邀请好友", "a": "我的 → 邀请好友, 邀请码 / 海报分享均可。"},
    {"q": "邀请有奖励吗", "a": "邀请 1 位有效用户双方各得 50 元抵扣券(可在 App 内查看)。"},
    {"q": "可以线下签合同吗", "a": "可以, 平台支持线上电子签 + 线下纸质签双轨。"},
    {"q": "如何看自己投了哪些项目", "a": "App → 我的 → 我的投资。"},
    {"q": "项目失败怎么办", "a": "项目有风险, 平台不承诺保本。具体责任以合同为准。"},
    {"q": "如何反馈问题", "a": "App → 我的 → 意见反馈, 或公众号回复「人工」转人工客服。"},
]


def _load_faq_seed() -> list[dict[str, Any]]:
    """返回 [{q, a, _tokens, _vec}, ...]."""
    out = []
    for item in _FAQ_SEED:
        toks = _tokenize(item["q"] + " " + item["a"])
        out.append({**item, "_tokens": toks})
    return out


_FAQ_INDEX: list[dict[str, Any]] | None = None


def _get_index() -> list[dict[str, Any]]:
    global _FAQ_INDEX
    if _FAQ_INDEX is None:
        _FAQ_INDEX = _load_faq_seed()
    return _FAQ_INDEX


def search(query: str, *, top_k: int = 3, min_score: float = 0.05) -> list[dict[str, Any]]:
    """搜 FAQ. 返回 [{q, a, score, source}, ...] 按 score desc.

    C 阶段纯 Python: 降级到 jaccard keyword 匹配.
    B 阶段预留: 真 FAISS 时换 _load_faiss_index() + faiss.IndexFlatIP.search().
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    scored = []
    for item in _get_index():
        score = _jaccard(q_tokens, item["_tokens"])
        if score >= min_score:
            scored.append({
                "q": item["q"],
                "a": item["a"],
                "score": round(score, 4),
                "source": "faq_seed_v1",
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def get_faq_count() -> int:
    """返回当前 FAQ 索引大小 (用于监控埋点)."""
    return len(_get_index())


def reset_for_test() -> None:
    """重置 in-memory 索引 (单测用)."""
    global _FAQ_INDEX
    _FAQ_INDEX = None
