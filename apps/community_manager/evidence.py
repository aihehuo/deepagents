"""Resolve opaque model selections to backend-issued evidence references."""
from copy import deepcopy


def catalogue(rows: list) -> dict[str, dict]:
    """Keep server-issued refs intact; models only select bounded opaque keys."""
    return {f'E{i + 1}': deepcopy(row) for i, row in enumerate(rows) if isinstance(row, dict) and isinstance(row.get('ref'), dict)}


def public_rows(entries: dict[str, dict]) -> list[dict]:
    """Expose text and attribution without asking models to reconstruct IDs."""
    return [{'key': key, 'text': row.get('text', ''), 'role': row.get('role')} for key, row in entries.items()]


def resolve(values: list, entries: dict[str, dict], maximum: int = 20) -> list[dict]:
    """Reject unknown selections; never replace them with unrelated evidence."""
    if not isinstance(values, list) or not 1 <= len(values) <= maximum:
        raise ValueError('invalid_evidence_selection')
    refs = []
    for value in values:
        row = entries.get(value) if isinstance(value, str) else next((r for r in entries.values() if r['ref'] == value), None)
        if row is None:
            raise ValueError('unknown_evidence_selection')
        refs.append(deepcopy(row['ref']))
    return refs


def resolve_draft(output: dict, entries: dict[str, dict]) -> dict:
    """Convert every summary/body evidence selection, preserving model prose."""
    result = deepcopy(output)
    for items in result['summary'].values():
        for item in items:
            item['evidence'] = resolve(item['evidence'], entries)
    result['body']['evidence'] = resolve(result['body']['evidence'], entries)
    return result
