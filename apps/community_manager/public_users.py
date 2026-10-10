"""Read the existing public semantic/keyword API without exposing out-of-scope hits."""
import json
from urllib.parse import urlencode
from urllib.request import Request, build_opener
from apps.community_manager.protocol import NoRedirect


def candidate_numbers(arguments: dict) -> list[str]:
    """Return only candidate numbers for Micro scope validation, never raw public hits.

    Args:
        arguments: One keyword or semantic query plus optional public filters.
    """
    if set(arguments) - {'keyword', 'q', 'location', 'vertical', 'role'}:
        raise ValueError('invalid_public_query')
    if bool(arguments.get('keyword')) == bool(arguments.get('q')):
        raise ValueError('invalid_public_query')
    if any(not isinstance(v, str) or len(v) > 200 for v in arguments.values()):
        raise ValueError('invalid_public_query')
    query = {**arguments, 'page': 1, 'per': 10, 'current_page': 1, 'per_page': 10}
    request = Request('https://www.aihehuo.com/ai/users?' + urlencode(query), headers={'Accept': 'application/json'})
    with build_opener(NoRedirect()).open(request, timeout=15) as response:
        raw = response.read(300_001)
    if len(raw) > 300_000:
        raise ValueError('public_search_too_large')
    data = json.loads(raw)
    rows = data.get('data') if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError('invalid_public_response')
    numbers = []
    for row in rows[:20]:
        value = row.get('number') if isinstance(row, dict) else None
        number = str(value)
        if number.isascii() and number.isdecimal() and not number.startswith('0') and len(number) <= 15 and number not in numbers:
            numbers.append(number)
    return numbers
