"""Optional brain Check modules (switchable; not Bare Backbone).

Packaged (YAML-gated via ``module_config.is_check_enabled`` / module switches):

- ``checks.reply_grounding`` — ``mod.brain.reply_grounding`` / ``chk.reply_fact_grounding_llm``
- ``checks.action_claim`` — ``chk.action_claim`` (skipped when RG on; L0 owns)
- ``checks.invented_candidate`` — ``chk.invented_candidate``
- ``checks.finalize_templates`` — ``chk.finalize_templates`` (off = keep-DA-reply)

Also YAML-gated (live hang, not under ``checks/`` package yet):

- ``guard.enforce_capability_guard`` — ``chk.capability_guard`` (off = passthrough)
- ``integrations.config.llm_polish_enabled`` — ``chk.invite_llm_polish``
  (ENV ``GROUP_AGENT_LLM_POLISH`` overrides when explicitly set)
"""
