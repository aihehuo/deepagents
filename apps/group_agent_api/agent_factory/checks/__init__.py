"""Optional brain Check modules (switchable; not Bare Backbone).

Packaged (YAML-gated via ``module_config.is_check_enabled`` / module switches):

- ``checks.reply_grounding`` — ``mod.brain.reply_grounding`` / ``chk.reply_fact_grounding_llm``
- ``checks.action_claim`` — ``chk.action_claim`` (soft under ``mod.brain.check``)
- ``checks.invented_candidate`` — ``chk.invented_candidate`` (soft)
- ``checks.finalize_templates`` — ``chk.finalize_templates`` (soft; off = keep-DA-reply)
- ``checks.profile_quality`` — ``chk.profile_quality_llm`` (soft; off = Layer1 rules only)
- ``checks.force_save_retry`` — ``chk.force_save_retry`` (hard; missing → on)
- ``checks.deterministic_profile_save`` — ``chk.deterministic_profile_save`` (hard)
- ``checks.match_v2_schema`` — ``chk.match_v2_schema`` (hard)

Also YAML-gated (live hang, not under ``checks/`` package yet):

- ``guard.enforce_capability_guard`` — ``chk.capability_guard`` (off = passthrough;
  **not** soft under ``mod.brain.check``)
- ``integrations.config.llm_polish_enabled`` — ``chk.invite_llm_polish``
  (ENV ``GROUP_AGENT_LLM_POLISH`` overrides when explicitly set)

``mod.brain.check`` soft-master off skips only soft checks listed in
``module_config._SOFT_BRAIN_CHECK_IDS`` — never capability_guard / force_save /
deterministic / match_v2 / invite / reply_grounding.
"""
