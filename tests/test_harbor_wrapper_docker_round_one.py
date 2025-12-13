"""Integration test: DeepAgents Harbor wrapper running in a Docker-backed Harbor environment.

This is a *production-shaped* test:
- Uses `DeepAgentsWrapper` (Harbor wrapper) to run a single instruction ("round one").
- Uses a Docker container as the execution environment (via the Docker Python SDK).
- Verifies Harbor-style trajectory logging is produced (trajectory.json).

The test is skipped unless:
- `harbor` package is installed
- `docker` Python package is installed
- Docker daemon is reachable
- Model credentials are provided via env vars
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


def _require_env(var_name: str) -> str:
    val = os.environ.get(var_name, "").strip()
    if not val:
        pytest.skip(f"{var_name} is not set; skipping Harbor wrapper integration test.")
    return val


@dataclass
class _ExecResult:
    stdout: str
    stderr: str
    return_code: int


@dataclass
class _TrialPaths:
    config_path: Path


class _DockerHarborEnvironment:
    """Minimal Harbor BaseEnvironment-compatible object backed by a Docker container.

    We only implement what `deepagents_harbor` uses:
    - session_id
    - trial_paths.config_path (JSON dict)
    - async exec(command) -> {stdout, stderr, return_code}
    """

    def __init__(self, *, docker_client: Any, root_dir: Path, image: str = "ubuntu:22.04") -> None:
        self.session_id = f"test-{uuid.uuid4().hex[:12]}"
        self._docker = docker_client
        self._image = image
        self._root_dir = root_dir
        self._container = None

        config_path = root_dir / "config.json"
        config_path.write_text(json.dumps({"task_name": "round-one-business-idea"}, indent=2))
        self.trial_paths = _TrialPaths(config_path=config_path)

    async def start(self) -> None:
        # Ensure image exists (pull if needed)
        try:
            self._docker.images.get(self._image)
        except Exception:
            # Pull can take time; if it fails we skip (common in CI without docker registry access)
            try:
                self._docker.images.pull(self._image)
            except Exception as e:
                pytest.skip(f"Could not pull Docker image {self._image}: {e}")

        # Create a long-running container with /app as working directory
        self._container = self._docker.containers.run(
            image=self._image,
            name=f"deepagents-harbor-{self.session_id}",
            detach=True,
            tty=False,
            stdin_open=False,
            remove=False,
            working_dir="/app",
            command="bash -lc 'mkdir -p /app && sleep infinity'",
        )

    async def exec(self, command: str) -> _ExecResult:
        if self._container is None:
            raise RuntimeError("Container not started")

        # Run command inside container
        exec_result = self._container.exec_run(
            cmd=f"bash -lc {json.dumps(command)}",
            stdout=True,
            stderr=True,
            demux=True,
        )
        stdout_b, stderr_b = exec_result.output
        stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
        stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
        return _ExecResult(stdout=stdout, stderr=stderr, return_code=exec_result.exit_code)

    async def stop(self) -> None:
        if self._container is None:
            return
        try:
            self._container.stop(timeout=5)
        finally:
            try:
                self._container.remove(force=True)
            finally:
                self._container = None


@pytest.mark.timeout(900)
def test_round_one_harbor_wrapper_in_docker_environment(tmp_path: Path) -> None:
    """Runs one complete instruction through DeepAgentsWrapper in a Docker environment and checks logs."""

    try:
        import harbor  # noqa: F401
    except ModuleNotFoundError:
        # Make the skip reason obvious even without `-rs`
        print("\n[SKIP] Missing dependency: `harbor` (HarborAI evaluation framework).")
        print("       Install it in your active venv, e.g.:")
        print("         - pip install harbor")
        print("       Or, if you use uv, from repo root:")
        print("         - cd libs/harbor && uv sync")
        pytest.skip("Missing dependency: harbor")

    try:
        import docker  # type: ignore
    except ModuleNotFoundError:
        print("\n[SKIP] Missing dependency: `docker` (Python Docker SDK).")
        print("       Install it in your active venv, e.g.:")
        print("         - pip install docker")
        pytest.skip("Missing dependency: docker")

    # Model creds must be provided by env in production-like runs
    _require_env("ANTHROPIC_API_KEY")
    _require_env("ANTHROPIC_BASE_URL")

    # Make sure Docker daemon is reachable
    client = docker.from_env()
    try:
        client.ping()
    except Exception as e:
        pytest.skip(f"Docker daemon not reachable: {e}")

    # Import the Harbor wrapper from our repo (without requiring installation)
    import sys

    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "libs" / "harbor"))
    from deepagents_harbor.deepagents_wrapper import DeepAgentsWrapper  # noqa: E402
    from harbor.models.agent.context import AgentContext  # noqa: E402

    logs_dir = tmp_path / "harbor_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = _DockerHarborEnvironment(docker_client=client, root_dir=tmp_path)
    instruction = """I have a complete business idea:

Product: BalanceAI – a mobile app for busy mid-career professionals to prevent burnout using AI task prioritization,
automatic calendar blocking, and boundary recommendations. Integrates with Google Calendar and Slack.

Target users: 30–45 year-old professionals working 50+ hours/week earning $80K–$150K.
Value: saves 5–10 hours/week, worth ~$2,000–$4,000/month in time value.

Please respond in English. For round one, just summarize the idea clearly and list 3 clarifying questions to validate assumptions.
"""

    wrapper = DeepAgentsWrapper(
        logs_dir=logs_dir,
        model_name=os.environ.get("ANTHROPIC_MODEL") or "deepseek-chat",
        temperature=0.4,
        verbose=False,
        use_cli_agent=True,
    )

    async def _run() -> None:
        await env.start()
        try:
            await wrapper.run(instruction=instruction, environment=env, context=AgentContext())
        finally:
            await env.stop()

    asyncio.run(_run())

    trajectory_path = logs_dir / "trajectory.json"
    assert trajectory_path.exists(), f"Expected Harbor trajectory at {trajectory_path}"

    data = json.loads(trajectory_path.read_text())
    assert isinstance(data, dict)
    assert data.get("schema_version", "").startswith("ATIF"), "Expected ATIF trajectory schema"

    steps = data.get("steps", [])
    assert steps and steps[0].get("source") == "user"
    assert instruction.strip() in steps[0].get("message", "")

    # Print the final agent response for visibility when running with `-s`
    last_agent_msg = None
    for step in reversed(steps):
        if step.get("source") == "agent" and step.get("message"):
            last_agent_msg = step.get("message")
            break

    print("\n" + "=" * 80)
    print("HARBOR WRAPPER OUTPUT (from trajectory.json)")
    print("=" * 80)
    print(f"Trajectory path: {trajectory_path}")
    if last_agent_msg:
        print("\nFinal agent message:\n")
        print(last_agent_msg)
    else:
        print("\n(No agent message found in trajectory steps.)")
    print("\n" + "=" * 80)


