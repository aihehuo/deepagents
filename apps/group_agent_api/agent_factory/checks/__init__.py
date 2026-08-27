"""Optional brain Check modules (switchable; not Bare Backbone).

Packaged (YAML-gated via ``module_config.is_check_enabled`` / module switches):

- ``checks.reply_grounding`` — ``mod.brain.reply_grounding`` / ``chk.reply_fact_grounding_llm``
- ``checks.action_claim`` — ``chk.action_claim``
- ``checks.invented_candidate`` — ``chk.invented_candidate``
- ``checks.finalize_templates`` — ``chk.finalize_templates`` (off = keep-DA-reply)
"""
