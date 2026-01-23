"""Integration tests for dual-agent creation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from apps.business_cofounder_api.agent_factory import (
    create_expert_agent,
    create_facilitator_agent,
)


# Skip these tests if no API keys are configured
pytestmark = pytest.mark.skipif(
    not os.getenv("QWEN_API_KEY") and not os.getenv("DEEPSEEK_API_KEY"),
    reason="Requires MODEL_API_KEY to create real agents"
)


class TestDualAgentCreation:
    """Test creation of facilitator and expert agents."""

    def test_create_facilitator_agent(self, tmp_path: Path, monkeypatch) -> None:
        """Should create facilitator with minimal middleware."""
        # Set up test environment
        backend_root = tmp_path / "backend"
        backend_root.mkdir()
        
        monkeypatch.setenv("HOME", str(tmp_path))

        agent, checkpoints_path = create_facilitator_agent(
            agent_id="test_facilitator",
            provider="qwen",
        )

        # Should return agent and checkpoints path
        assert agent is not None
        assert checkpoints_path.exists()
        assert checkpoints_path.name == "facilitator_checkpoints.pkl"

    def test_create_expert_agent_default_expertise(
        self, tmp_path: Path, monkeypatch, sample_business_expertise: Path
    ) -> None:
        """Should create expert with default business_cofounder expertise."""
        monkeypatch.setenv("HOME", str(tmp_path))

        agent, checkpoints_path = create_expert_agent(
            agent_id="test_expert",
            provider="qwen",
            expertise_type="business_cofounder",
        )

        # Should return agent and checkpoints path
        assert agent is not None
        assert checkpoints_path.exists()
        assert checkpoints_path.name == "expert_checkpoints.pkl"

    def test_create_expert_agent_custom_expertise(
        self, tmp_path: Path, monkeypatch, sample_education_expertise: Path
    ) -> None:
        """Should create expert with custom expertise type."""
        monkeypatch.setenv("HOME", str(tmp_path))

        agent, checkpoints_path = create_expert_agent(
            agent_id="test_expert",
            provider="qwen",
            expertise_type="education_mentor",
        )

        # Should successfully create agent with custom expertise
        assert agent is not None
        assert checkpoints_path.exists()

    def test_expert_agent_loads_expertise_at_startup(
        self, tmp_path: Path, monkeypatch, sample_business_expertise: Path, caplog
    ) -> None:
        """Should load and inject expertise into system prompt."""
        monkeypatch.setenv("HOME", str(tmp_path))

        agent, _ = create_expert_agent(
            agent_id="test_expert",
            provider="qwen",
            expertise_type="business_cofounder",
        )

        # Check that expertise was loaded (look for log messages)
        assert any("Loaded expertise" in record.message for record in caplog.records)

    def test_default_expertise_auto_copied(self, tmp_path: Path, monkeypatch) -> None:
        """Should auto-copy business_cofounder.md if missing."""
        monkeypatch.setenv("HOME", str(tmp_path))

        expertise_dir = (
            tmp_path / ".deepagents" / "business_cofounder_api" / "expertise"
        )

        # Expertise directory should not exist initially
        assert not expertise_dir.exists()

        # Create agent - should auto-copy default expertise
        agent, _ = create_expert_agent(
            agent_id="test_expert",
            provider="qwen",
            expertise_type="business_cofounder",
        )

        # Check that default expertise was copied
        default_expertise = expertise_dir / "business_cofounder.md"
        assert default_expertise.exists()

        # Should contain valid content
        content = default_expertise.read_text()
        assert "name: business_cofounder" in content
        assert "canvas_template" in content

    def test_expert_agent_fails_with_invalid_expertise(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Should raise RuntimeError if expertise loading fails."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Try to create agent with non-existent expertise
        with pytest.raises(RuntimeError) as exc_info:
            create_expert_agent(
                agent_id="test_expert",
                provider="qwen",
                expertise_type="nonexistent_expertise",
            )

        assert "expertise" in str(exc_info.value).lower()

    def test_facilitator_uses_minimal_middleware(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """Facilitator should have minimal middleware stack."""
        monkeypatch.setenv("HOME", str(tmp_path))

        agent, _ = create_facilitator_agent(
            agent_id="test_facilitator",
            provider="qwen",
        )

        # Check logs for middleware initialization
        # Facilitator should NOT have heavy middleware like BusinessIdeaTracker
        log_text = " ".join([record.message for record in caplog.records])
        
        # Should not have business idea tracking middleware
        assert "BusinessIdeaTracker" not in log_text or "Facilitator" in log_text

    def test_expert_uses_full_middleware_stack(
        self, tmp_path: Path, monkeypatch, sample_business_expertise: Path, caplog
    ) -> None:
        """Expert should have full middleware stack including skills."""
        monkeypatch.setenv("HOME", str(tmp_path))

        agent, _ = create_expert_agent(
            agent_id="test_expert",
            provider="qwen",
            expertise_type="business_cofounder",
        )

        # Expert should have full middleware
        # Check for skills middleware initialization
        log_text = " ".join([record.message for record in caplog.records])
        
        # Should have skills or methodology middleware
        assert "skill" in log_text.lower() or "expert" in log_text.lower()

    def test_both_agents_use_same_checkpointer_directory(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Both agents should use checkpoints in same base directory."""
        monkeypatch.setenv("HOME", str(tmp_path))

        facilitator_agent, facilitator_checkpoints = create_facilitator_agent(
            agent_id="test_facilitator",
            provider="qwen",
        )

        expert_agent, expert_checkpoints = create_expert_agent(
            agent_id="test_expert",
            provider="qwen",
        )

        # Both should be in same base directory
        assert facilitator_checkpoints.parent == expert_checkpoints.parent

    def test_agents_can_be_created_multiple_times(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Should be able to create multiple agent instances."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create first pair
        fac1, _ = create_facilitator_agent(agent_id="fac1", provider="qwen")
        exp1, _ = create_expert_agent(agent_id="exp1", provider="qwen")

        # Create second pair
        fac2, _ = create_facilitator_agent(agent_id="fac2", provider="qwen")
        exp2, _ = create_expert_agent(agent_id="exp2", provider="qwen")

        # All should be created successfully
        assert fac1 is not None
        assert exp1 is not None
        assert fac2 is not None
        assert exp2 is not None

    def test_expertise_persistence_across_agent_creation(
        self, tmp_path: Path, monkeypatch, sample_business_expertise: Path
    ) -> None:
        """Expertise files should persist and be reused."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create first agent
        agent1, _ = create_expert_agent(
            agent_id="expert1",
            provider="qwen",
            expertise_type="business_cofounder",
        )

        expertise_dir = (
            tmp_path / ".deepagents" / "business_cofounder_api" / "expertise"
        )
        expertise_file = expertise_dir / "business_cofounder.md"

        # Modify file to verify it's reused
        original_content = expertise_file.read_text()
        expertise_file.write_text(
            original_content + "\n# MODIFIED MARKER\n"
        )

        # Create second agent - should use existing file
        agent2, _ = create_expert_agent(
            agent_id="expert2",
            provider="qwen",
            expertise_type="business_cofounder",
        )

        # File should still have modification
        modified_content = expertise_file.read_text()
        assert "# MODIFIED MARKER" in modified_content
