"""Opaque selections must never change backend evidence identity or attribution."""
import pytest
from apps.community_manager.evidence import catalogue, public_rows, resolve, resolve_draft


def test_exact_refs_are_code_resolved_and_not_exposed_as_model_fields() -> None:
    ref = {'source': 'wecom_turn', 'id': 12, 'field': 'input', 'message_index': 0}
    entries = catalogue([{'ref': ref, 'text': 'fictional', 'role': 'user'}])
    assert 'ref' not in public_rows(entries)[0]
    assert resolve(['E1'], entries) == [ref]
    assert resolve([ref], entries) == [ref]
    with pytest.raises(ValueError):resolve([dict(ref, id=99)], entries)
    with pytest.raises(ValueError):resolve(['E2'], entries)
    with pytest.raises(ValueError):resolve([], entries)


def test_draft_every_evidence_path_is_mapped_without_changing_prose() -> None:
    entries = catalogue([{'ref': {'source': 'profile', 'id': 7, 'field': 'bio'}, 'text': 'fictional'}])
    output = {'summary': {'current_needs': [{'text': 'need', 'classification': 'user_stated', 'evidence': ['E1']}]}, 'body': {'current_need': 'need', 'evidence': ['E1']}}
    result = resolve_draft(output, entries)
    assert result['body']['evidence'] == [{'source': 'profile', 'id': 7, 'field': 'bio'}]
    assert result['summary']['current_needs'][0]['evidence'] == result['body']['evidence']
    assert result['body']['current_need'] == 'need'
    assert output['body']['evidence'] == ['E1']
