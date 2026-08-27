"""Module identity for mod.brain.reply_grounding."""

MODULE_ID = "mod.brain.reply_grounding"
CHECK_ID = "chk.reply_fact_grounding_llm"
PROTOCOL_NAME = "check.reply_grounding.v1"

# Same-turn repair attempts including the first check (TSD-14 §4.6.4).
DEFAULT_MAX_ATTEMPTS = 2
