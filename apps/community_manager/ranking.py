"""Current-run chronological receipts, independent of assistant claims/history."""
import re
from datetime import datetime, timezone, timedelta

BEIJING = timezone(timedelta(hours=8))
FAILURE = '本轮未取得符合要求的实时排序查询回执，无法确认最新用户名单和活跃时间。历史名单不代表本轮查询结果。'


def intent(started: dict) -> dict | None:
    text = started.get('text', '')
    # Inherit only an explicit filter adjustment to the last human ranking request.
    if re.search(r'去掉|取消|不限|不限制', text) and re.search(r'地区|地域|城市|上海', text):
        previous = next((r['text'] for r in reversed(started.get('history', [])) if r.get('role') == 'staff'), '')
        text = previous + '\n' + text
    if not (re.search(r'活跃|最新注册|最近注册|注册时间', text) and re.search(r'排序|倒序|最近|最新|当天|今日|今天|前.{0,3}[位名个]|top', text, re.I)):
        return None
    recent = bool(re.search(r'活跃', text))
    value = {'sort': 'last_active_desc' if recent else 'registered_desc',
             'contact_scope': 'unbound' if re.search(r'未.{0,10}(?:企微|企业微信)|没.{0,8}(?:企微|企业微信)', text) else 'reachable'}
    count = re.search(r'(?:前|列出|推荐|找出|返回)?\s*([0-9]{1,3}|[一二三四五六七八九十]{1,3})\s*[位名个]', text)
    if count:
        numerals = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
        raw = count.group(1)
        if raw.isdigit():
            requested = int(raw)
        elif raw in numerals:
            requested = numerals[raw]
        elif '十' in raw:
            left, right = raw.split('十', 1)
            requested = numerals.get(left, 1) * 10 + numerals.get(right, 0)
        else:
            requested = 0
        if requested > 0:
            value['limit'] = min(requested, 20)
    if '上海' in text and not re.search(r'去掉|取消|不限|不限制', started.get('text', '')):
        value['city'] = '上海'
    if re.search(r'去掉|取消|不限|不限制', started.get('text', '')):
        value['no_city'] = True
    if recent and re.search(r'当天|今日|今天', text):
        value['today'] = True
    return value


def valid_receipt(receipt: dict, args: dict, expected: dict) -> bool:
    if not isinstance(receipt, dict) or receipt.get('tool_error') or receipt.get('error'):
        return False
    field = 'users.last_accessed_at' if expected['sort'] == 'last_active_desc' else 'users.created_at'
    if receipt.get('sort_field') != field or receipt.get('sort') != expected['sort'] or receipt.get('contact_scope') != expected['contact_scope']:
        return False
    if expected.get('city') and str(receipt.get('city', '')).removesuffix('市') != expected['city']:
        return False
    if expected.get('no_city') and receipt.get('city'):
        return False
    if expected.get('today'):
        floor = datetime.now(BEIJING).replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            since = datetime.fromisoformat(receipt['active_since'].replace('Z', '+00:00'))
            if since != floor:
                return False
        except (KeyError, ValueError, TypeError):
            return False
    if expected.get('limit') and args.get('limit', 10) != expected['limit']:
        return False
    rows = receipt.get('users')
    if not isinstance(rows, list) or len(rows) > args.get('limit', 10) or not receipt.get('observed_at'):
        return False
    key = 'last_active_at' if expected['sort'] == 'last_active_desc' else 'registered_at'
    dates = []
    for row in rows:
        if not isinstance(row, dict) or not row.get('user_number') or row.get('bound') != ('not_bound' if expected['contact_scope'] == 'unbound' else 'bound_snapshot'):
            return False
        try:
            stamp = datetime.fromisoformat(row[key].replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                return False
            dates.append(stamp)
        except (KeyError, ValueError, TypeError):
            return False
    return dates == sorted(dates, reverse=True)


def render(receipt: dict | None, contacts: dict, limit: int = 10) -> str:
    if receipt is None:
        return FAILURE
    active = receipt['sort'] == 'last_active_desc'
    label = '最近活跃' if active else '最近注册'
    key = 'last_active_at' if active else 'registered_at'
    rows = receipt['users']
    lines = [f'本轮实时查询：{receipt.get("city") or "配置范围全地区"}，{"未加企微" if receipt["contact_scope"] == "unbound" else "已加企微可触达"}，按{label}时间倒序。']
    if not rows:
        lines.append('本次筛选未返回符合条件的用户。')
    for i, row in enumerate(rows, 1):
        number = str(row['user_number'])
        stamp = datetime.fromisoformat(row[key].replace('Z', '+00:00')).astimezone(BEIJING).strftime('%Y-%m-%d %H:%M:%S')
        lines.append(f'{i}. {str(row.get("display_name") or "未填写姓名")[:30]}（创业号 {number}）\n{label}时间：{stamp}（北京时间）')
        profile = row.get('profile') or {}
        description = '；'.join(str(profile.get(k) or '')[:30] for k in ('headline', 'goal')).strip('；')
        if description:
            lines.append('资料摘要：' + description)
        lead = (contacts.get(number) or {}).get('preferred_contact') or {}
        if lead.get('value') and lead.get('kind') in {'wechat', 'phone'}:
            lines.append(('微信：' if lead['kind'] == 'wechat' else '手机号：') + str(lead['value'])[:128] + '（联系线索，请人工核验）')
        elif number in contacts:
            lines.append('联系方式未取得。')
        else:
            lines.append('联系方式本轮尚未核验。')
    if len(rows) < limit:
        lines.append(f'本轮请求最多{limit}位，实际返回{len(rows)}位；不使用旧名单补数。')
    if receipt.get('has_more'):
        lines.append('筛选范围仍有后续候选；本轮只展示当前返回人数。')
    lines.append('以上时间来自本轮数据库回执；平台活跃不代表微信在线，资料摘要不等于已核验联系价值。')
    return '\n'.join(lines)
