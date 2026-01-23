"""Pytest configuration for tests."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add libs/deepagents to Python path so we can import deepagents modules
# This is needed because the package might not be installed in editable mode
repo_root = Path(__file__).parent.parent
deepagents_lib = repo_root / "libs" / "deepagents"
if deepagents_lib.exists() and str(deepagents_lib) not in sys.path:
    sys.path.insert(0, str(deepagents_lib))

# Try to configure pytest-asyncio if available
try:
    import pytest_asyncio
    # Configure asyncio mode if not already set
    if not hasattr(pytest_asyncio, '_asyncio_event_loop_policy'):
        pytest_asyncio.plugin._asyncio_event_loop_policy = None
except ImportError:
    # pytest-asyncio not installed - tests using @pytest.mark.asyncio will fail
    # Install with: pip install pytest-asyncio
    pass


# ========== Dual-Agent Test Fixtures ==========


@pytest.fixture
def expertise_dir(tmp_path: Path) -> Path:
    """Create temporary expertise directory."""
    dir_path = tmp_path / "expertise"
    dir_path.mkdir()
    return dir_path


@pytest.fixture
def sample_business_expertise(expertise_dir: Path) -> Path:
    """Create sample business_cofounder.md expertise."""
    from tests.mocks.dual_agent_mocks import create_business_cofounder_expertise

    return create_business_cofounder_expertise(expertise_dir)


@pytest.fixture
def sample_education_expertise(expertise_dir: Path) -> Path:
    """Create sample education_mentor.md expertise."""
    from tests.mocks.dual_agent_mocks import create_education_mentor_expertise

    return create_education_mentor_expertise(expertise_dir)


@pytest.fixture
def fake_expert_agent():
    """Create fake expert agent with canned responses."""
    from tests.mocks.dual_agent_mocks import MockExpertAgent

    return MockExpertAgent()


@pytest.fixture
def fake_facilitator_agent():
    """Create fake facilitator agent."""
    from tests.mocks.dual_agent_mocks import MockFacilitatorAgent

    return MockFacilitatorAgent()


@pytest.fixture
def fake_checkpointer():
    """Create fake checkpointer for testing."""
    from tests.mocks.dual_agent_mocks import MockCheckpointer

    return MockCheckpointer()


@pytest.fixture
def dual_agent_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """TestClient with dual-agent mode enabled."""
    import apps.business_cofounder_api.app as app_module
    from tests.mocks.dual_agent_mocks import MockCheckpointer, MockExpertAgent, MockFacilitatorAgent

    # Ensure a clean module-level state per test
    app_module._state = None

    # Create fake agents
    facilitator_checkpointer = MockCheckpointer()
    facilitator_agent = MockFacilitatorAgent()
    facilitator_agent.checkpointer = facilitator_checkpointer  # type: ignore[attr-defined]

    expert_checkpointer = MockCheckpointer()
    expert_agent = MockExpertAgent()
    expert_agent.checkpointer = expert_checkpointer  # type: ignore[attr-defined]

    facilitator_checkpoints = tmp_path / "facilitator_checkpoints.pkl"
    expert_checkpoints = tmp_path / "expert_checkpoints.pkl"
    expertise_dir_path = tmp_path / "expertise"
    expertise_dir_path.mkdir()

    def _fake_create_facilitator_agent(*, agent_id: str, provider: str = "qwen", **kwargs) -> tuple[object, Path]:
        return facilitator_agent, facilitator_checkpoints

    def _fake_create_expert_agent(
        *, agent_id: str, provider: str = "qwen", expertise_type: str = "business_cofounder", **kwargs
    ) -> tuple[object, Path]:
        return expert_agent, expert_checkpoints

    monkeypatch.setattr(app_module, "create_facilitator_agent", _fake_create_facilitator_agent)
    monkeypatch.setattr(app_module, "create_expert_agent", _fake_create_expert_agent)
    monkeypatch.setenv("BC_API_USE_DUAL_AGENT", "1")

    with TestClient(app_module.app) as c:
        # Expose fakes for assertions
        c._fake_facilitator_agent = facilitator_agent  # type: ignore[attr-defined]
        c._fake_expert_agent = expert_agent  # type: ignore[attr-defined]
        c._fake_facilitator_checkpointer = facilitator_checkpointer  # type: ignore[attr-defined]
        c._fake_expert_checkpointer = expert_checkpointer  # type: ignore[attr-defined]
        yield c

