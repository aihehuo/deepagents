"""Agent factory - backward compatibility re-exports.

This module re-exports functions from the agent_factory package to maintain
backward compatibility. The code has been refactored into separate modules
for better organization and readability.

New code should import directly from the package:
    from apps.business_cofounder_api.agent_factory import create_business_cofounder_agent
"""

# Re-export all public functions from the package
from apps.business_cofounder_api.agent_factory import (
    create_business_cofounder_agent,
    create_code_agent,
    create_expert_agent,
    create_facilitator_agent,
    create_search_agent,
)

__all__ = [
    "create_business_cofounder_agent",
    "create_facilitator_agent",
    "create_expert_agent",
    "create_code_agent",
    "create_search_agent",
]
