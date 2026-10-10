"""Optional upward signal from the existing validated customer assessment, no new model call."""
import os
from typing import Any


def build_report(assessment: dict | None, context: dict | None) -> dict | None:
    """Report a material assessment change without modifying customer reply or execution.

    Args:
        assessment: Already validated per-customer model assessment.
        context: Host-authored personal evidence context.
    """
    if os.environ.get('WECOM_SERVICE_MANAGER_REPORT_ENABLED') != '1' or not assessment or not context:
        return None
    previous = context.get('previous') or {}
    old = {n['id']: n for n in previous.get('needs', [])}
    changed = [n for n in assessment.get('needs', []) if n.get('clarity') == 'clear' and old.get(n['id'], {}).get('clarity') != 'clear']
    if not changed:
        return None
    keys = list(dict.fromkeys(k for n in changed for k in n.get('evidence_keys', [])))[:10]
    if not keys or any(k not in context['sources'] for k in keys):
        return None
    return {'type': 'new_clear_need', 'summary': '客服评估发现新的明确找人需求，请经理核验当前资料、已发送推荐与服务安排。',
            'evidence_keys': keys}
