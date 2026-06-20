# Integration Tests

These tests require additional dependencies to be installed.

## Required Dependencies

For async tests (like `test_facilitator_guidance_response.py`), you need to install `pytest-asyncio`:

```bash
pip install pytest-asyncio
```

Or if using the project's development dependencies:

```bash
pip install -e ".[dev]"
```

## Running Tests

Run all integration tests:
```bash
pytest tests/integration/ -v
```

Run a specific test:
```bash
pytest tests/integration/test_facilitator_guidance_response.py -v -s
```

The `-s` flag shows print output for debugging.

## Test: Facilitator Guidance Response

This test verifies that the facilitator agent can pick up and follow expert guidance prompts, even when they contradict the base system prompt.

**Test scenario:**
- Facilitator agent starts with business co-founder system prompt
- Expert agent provides guidance to ask about prime numbers instead
- We verify the facilitator agent follows the guidance and asks about prime numbers

**Requirements:**
- Real LLM API credentials (Qwen provider)
- pytest-asyncio installed
- Network access for LLM API calls
