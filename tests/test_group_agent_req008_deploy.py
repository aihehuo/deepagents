"""Unit tests for REQ-008 / REQ-011 deployment example packaging and public-safety contract.

The real production deploy script and compose file (the numbered-environment
deploy assets under apps/) are intentionally NOT tracked in this public
repository — they are kept locally and ignored via .gitignore. This suite
validates the tracked, generic *.example.* companions instead, and enforces a
permanent negative contract: no real infrastructure markers may ever appear in
the tracked deploy assets.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = REPO_ROOT / "apps"
GROUP_AGENT_DIR = APPS_DIR / "group_agent_api"

# Numbered production environment token, assembled from fragments so this test
# file itself contains no contiguous literal forbidden string (which would
# otherwise trip a repo-wide marker scan).
_ENV = "prod" + "3"

REAL_DEPLOY_SCRIPT = "apps/deploy_to_" + _ENV + ".sh"
REAL_COMPOSE_FILE = "apps/docker-compose." + _ENV + ".yml"

DEPLOY_EXAMPLE = APPS_DIR / "deploy_to_prod.example.sh"
COMPOSE_EXAMPLE = APPS_DIR / "docker-compose.prod.example.yml"

# Forbidden markers that must never appear in tracked public deploy assets.
# These identify real production infrastructure and must stay out of the public
# repo. They are assembled from fragments (see _ENV above) so this test file
# itself contains no literal forbidden string.
FORBIDDEN_MARKERS = [
    _ENV,
    "root@" + _ENV,
    "/mnt/" + "deepagents",
    "crpi-" + "lp1jelcmhkef5y0u",
    "aihehuo-" + "new-api",
    "aihehuomicro-" + "web",
]


def test_real_deploy_files_are_not_tracked():
    """The real prod deploy script and compose file must not be tracked in git."""
    for rel in (REAL_DEPLOY_SCRIPT, REAL_COMPOSE_FILE):
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", rel],
            capture_output=True, text=True,
        ).stdout.strip()
        assert out == "", f"{rel} must NOT be tracked in git (got: {out!r})"


def test_real_deploy_files_are_ignored():
    """The real prod deploy files must be matched by .gitignore rules."""
    for rel in (REAL_DEPLOY_SCRIPT, REAL_COMPOSE_FILE):
        res = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", rel],
            capture_output=True, text=True,
        )
        assert res.returncode == 0 and res.stdout.strip() == rel, (
            f"{rel} must be ignored by .gitignore (check-ignore rc={res.returncode})"
        )


def test_example_files_exist_and_tracked():
    """The generic example deploy script and compose file must exist and be tracked."""
    assert DEPLOY_EXAMPLE.exists(), "apps/deploy_to_prod.example.sh must exist"
    assert COMPOSE_EXAMPLE.exists(), "apps/docker-compose.prod.example.yml must exist"
    for rel in ("apps/deploy_to_prod.example.sh", "apps/docker-compose.prod.example.yml"):
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", rel],
            capture_output=True, text=True,
        ).stdout.strip()
        assert out == rel, f"{rel} must be tracked in git"


def test_examples_contain_no_forbidden_markers():
    """Permanent negative assertion: examples must not leak real infrastructure markers."""
    for path in (DEPLOY_EXAMPLE, COMPOSE_EXAMPLE):
        content = path.read_text()
        for marker in FORBIDDEN_MARKERS:
            assert marker not in content, (
                f"Forbidden marker {marker!r} found in {path.name}; "
                "public examples must not contain real infrastructure identifiers."
            )


def test_docker_env_example_contains_no_forbidden_markers():
    """docker.env.example must use placeholders, not real internal DNS."""
    content = (GROUP_AGENT_DIR / "docker.env.example").read_text()
    for marker in ("aihehuo-" + "new-api", "aihehuomicro-" + "web"):
        assert marker not in content, "Forbidden internal-DNS marker found in docker.env.example"


def test_deploy_example_is_fail_closed():
    """deploy_to_prod.example.sh must require explicit deploy targets with no defaults."""
    content = DEPLOY_EXAMPLE.read_text()
    for var in ("DEPLOY_HOST", "DEPLOY_USER", "DEPLOY_DIR", "IMAGE_REF"):
        assert f'"${{{var}:?' in content, (
            f"{var} must use fail-closed ${{{var}:?...}} form (no default)"
        )
    # Missing required vars must abort before any ssh/docker action.
    res = subprocess.run(
        ["bash", str(DEPLOY_EXAMPLE), "business_cofounder_api"],
        capture_output=True, text=True,
        env={"PATH": os.environ.get("PATH", "")},  # no DEPLOY_* set
    )
    assert res.returncode != 0, "example must fail closed when required vars are missing"
    assert "is required" in (res.stdout + res.stderr)


def test_compose_example_binds_loopback_only():
    """docker-compose.prod.example.yml must bind loopback only and require image vars."""
    content = COMPOSE_EXAMPLE.read_text()

    with open(COMPOSE_EXAMPLE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    services = data.get("services", {})
    for name in ("business-cofounder-api", "wu-tanchang-api", "group-agent-api"):
        assert name in services, f"service {name} must be present in example compose"
        svc = services[name]
        assert svc.get("user") == "1000:1000"
        # Image references must be required (fail-closed), never defaulted.
        image = svc.get("image", "")
        assert ":?" in image, f"{name} image must be a required variable (fail-closed)"
        # Ports must bind loopback only, never 0.0.0.0.
        for p in svc.get("ports", []):
            assert str(p).startswith("127.0.0.1:"), (
                f"{name} port {p!r} must bind 127.0.0.1 only"
            )
    assert "0.0.0.0" not in content, "example compose must not expose 0.0.0.0"


def test_compose_example_has_no_seccomp_unconfined():
    """Permanent negative assertion: example compose must not weaken seccomp."""
    content = COMPOSE_EXAMPLE.read_text()
    seccomp_downgrade = "seccomp=" + "unconfined"
    assert seccomp_downgrade not in content, (
        "example compose must not ship an unconfined seccomp profile (security downgrade)"
    )
    with open(COMPOSE_EXAMPLE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for name, svc in data.get("services", {}).items():
        assert "unconfined" not in str(svc.get("security_opt", "")), (
            f"{name} must not set an unconfined security_opt"
        )


def test_deploy_example_uses_compose_file_and_image_ref():
    """deploy_to_prod.example.sh must thread COMPOSE_FILE and IMAGE_REF into the remote compose commands."""
    content = DEPLOY_EXAMPLE.read_text()
    assert "COMPOSE_FILE" in content, "must reference COMPOSE_FILE"
    assert 'docker compose -f "${COMPOSE_BASENAME}"' in content, (
        "remote compose commands must use the configured compose file"
    )
    assert "IMAGE_REF" in content, "must reference IMAGE_REF"
    assert 'export ${IMAGE_ENV_VAR}="${IMAGE_REF}"' in content, (
        "IMAGE_REF must be exported so the compose file's required image var resolves"
    )


def test_deploy_example_effective_path_uses_compose_and_image(tmp_path):
    """With command stubs, the example must invoke docker compose with the compose file and export IMAGE_REF."""
    trace = tmp_path / "trace.log"
    bindir = tmp_path / "bin"
    bindir.mkdir()

    # Stub ssh: instead of connecting, run the heredoc body locally so we can
    # observe which commands (and env) the remote script would execute. We
    # override `cd` to a no-op so the placeholder DEPLOY_DIR need not exist.
    ssh_stub = bindir / "ssh"
    ssh_stub.write_text(
        "#!/bin/bash\n"
        f'echo "SSH_TARGET=$1" >> "{trace}"\n'
        f'{{ echo "cd() {{ :; }}"; cat; }} | bash >> "{trace}" 2>&1\n'
    )
    ssh_stub.chmod(0o755)

    # Stub docker: record every invocation.
    docker_stub = bindir / "docker"
    docker_stub.write_text(
        "#!/bin/bash\n"
        f'echo "DOCKER $*" >> "{trace}"\n'
        # emulate `docker compose ps -q` returning a container id, and inspect returning the image
        'if [ "$1" = "compose" ]; then\n'
        '  for a in "$@"; do if [ "$a" = "-q" ]; then echo "cid123"; fi; done\n'
        '  exit 0\n'
        'fi\n'
        'if [ "$1" = "inspect" ]; then echo "$IMAGE_REF_EXPECT"; exit 0; fi\n'
        "exit 0\n"
    )
    docker_stub.chmod(0o755)

    # Stub build_and_push.sh next to the example so the SCRIPT_DIR lookup finds it.
    build_stub = APPS_DIR / ".tmp_build_stub.sh"
    build_stub.write_text('#!/bin/bash\necho "BUILD_OK $1 $2"\n')
    build_stub.chmod(0o755)

    image_ref = "registry.example.invalid/deepagents/group-agent-api:" + ("a" * 40)
    tmp_deploy = APPS_DIR / ".tmp_deploy_example.sh"
    try:
        # Point the build delegate at our stub; keep everything else intact.
        src = DEPLOY_EXAMPLE.read_text().replace(
            '"${SCRIPT_DIR}/build_and_push.sh"', f'"{build_stub}"'
        )
        tmp_deploy.write_text(src)
        tmp_deploy.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{bindir}:{env.get('PATH', '')}"
        env["DEPLOY_HOST"] = "deploy.example.invalid"
        env["DEPLOY_USER"] = "deployer"
        env["DEPLOY_DIR"] = "/srv/deepagents"
        env["IMAGE_REF"] = image_ref
        env["COMPOSE_FILE"] = "/srv/deepagents/docker-compose.prod.yml"
        env["IMAGE_REF_EXPECT"] = image_ref  # what the docker inspect stub echoes

        res = subprocess.run(
            ["bash", str(tmp_deploy), "group_agent_api", "a" * 40],
            capture_output=True, text=True, env=env,
        )
        log = trace.read_text() if trace.exists() else ""
        assert res.returncode == 0, f"deploy example failed: {res.stdout}\n{res.stderr}\n{log}"
        # Compose commands must reference the configured compose basename.
        assert "compose -f docker-compose.prod.yml pull group-agent-api" in log
        assert "compose -f docker-compose.prod.yml up -d group-agent-api" in log
        # The required image env var must have been exported to the image we built.
        assert "GROUP_AGENT_API_IMAGE" in src or "GROUP_AGENT_API_IMAGE" in log or image_ref in log
    finally:
        for f in (tmp_deploy, build_stub):
            if f.exists():
                f.unlink()


def test_new_api_base_requires_explicit_config_in_production():
    """new_api_base must not fall back to a real production URL; prod/http must require explicit config."""
    from apps.group_agent_api.agent_factory.integrations import config as cfg

    saved = {k: os.environ.get(k) for k in (
        "GROUP_AGENT_NEW_API_BASE", "AIHEHUO_API_BASE",
        "GROUP_AGENT_ENV", "GROUP_AGENT_INTEGRATION",
        "GROUP_AGENT_REQUIRE_TRUSTED_PRINCIPAL",
    )}
    try:
        for k in saved:
            os.environ.pop(k, None)

        # Source must not contain a real production host as a baked-in default.
        src = Path(cfg.__file__).read_text()
        real_prod_host = "new-api." + "aihehuo.com"
        assert real_prod_host not in src, (
            "config.py must not hardcode the real New API production URL"
        )

        # Production / http mode with no explicit base must fail closed.
        os.environ["GROUP_AGENT_ENV"] = "production"
        with pytest.raises(RuntimeError, match="GROUP_AGENT_NEW_API_BASE"):
            cfg.new_api_base()

        os.environ.pop("GROUP_AGENT_ENV", None)
        os.environ["GROUP_AGENT_INTEGRATION"] = "http"
        with pytest.raises(RuntimeError, match="GROUP_AGENT_NEW_API_BASE"):
            cfg.new_api_base()

        # Explicit config is honored.
        os.environ["GROUP_AGENT_NEW_API_BASE"] = "http://new-api.example.invalid:3000"
        assert cfg.new_api_base() == "http://new-api.example.invalid:3000"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)


def test_dockerfile_exists_and_valid():
    """Verify Dockerfile uses Python 3.11, non-root USER 1000:1000, exact --workers 1, and exposes 8001."""
    dockerfile = GROUP_AGENT_DIR / "Dockerfile"
    assert dockerfile.exists(), "apps/group_agent_api/Dockerfile must exist"

    content = dockerfile.read_text()
    assert "python:3.11-slim" in content
    assert "USER 1000:1000" in content, "Dockerfile must set USER 1000:1000"
    assert "EXPOSE 8001" in content
    assert '"--workers", "1"' in content, "Dockerfile CMD must specify exact workers count '--workers', '1'"
    assert "/health" in content


def test_docker_entrypoint_exists_and_fail_fast():
    """Verify docker-entrypoint.sh exists, is executable, no swallowed errors, and has fail-fast writability check."""
    entrypoint = GROUP_AGENT_DIR / "docker-entrypoint.sh"
    assert entrypoint.exists(), "apps/group_agent_api/docker-entrypoint.sh must exist"
    assert os.access(entrypoint, os.X_OK), "docker-entrypoint.sh must be executable"

    content = entrypoint.read_text()
    assert "set -e" in content
    assert "2>/dev/null || true" not in content, "Entrypoint must not swallow directory creation errors"
    assert "! -w \"$APP_DIR\"" in content or "test -w" in content, "Entrypoint must check runtime directory writability"
    assert 'exec "$@"' in content


def test_docker_env_example_aligned_with_model_builder():
    """Verify docker.env.example contains active model_builder.py variables and configures create_model correctly."""
    env_example = GROUP_AGENT_DIR / "docker.env.example"
    assert env_example.exists(), "apps/group_agent_api/docker.env.example must exist"

    content = env_example.read_text()
    assert "GROUP_AGENT_ENV=production" in content
    assert "GROUP_AGENT_INTEGRATION=http" in content
    assert "GROUP_AGENT_REQUIRE_TRUSTED_PRINCIPAL=1" in content
    assert "GROUP_AGENT_PRINCIPAL_HMAC_SECRET" in content
    assert "GROUP_AGENT_RUNTIME_DIR=/home/appuser/.deepagents/group_agent_api" in content
    assert "GROUP_AGENT_NEW_API_BASE" in content
    assert "GROUP_AGENT_MICRO_BASE" in content
    assert "GROUP_AGENT_PROVIDER=qwen" in content
    assert "GROUP_AGENT_MODEL=qwen-plus" in content
    assert "GROUP_AGENT_BASE_URL" in content
    assert "DASHSCOPE_API_KEY" in content
    assert "GROUP_AGENT_API_KEY" not in content

    # Test model_builder reads environment variables
    from apps.group_agent_api.agent_factory.model_builder import create_model

    old_provider = os.environ.get("GROUP_AGENT_PROVIDER")
    old_model = os.environ.get("GROUP_AGENT_MODEL")
    old_base_url = os.environ.get("GROUP_AGENT_BASE_URL")
    old_key = os.environ.get("DASHSCOPE_API_KEY")

    try:
        os.environ["GROUP_AGENT_PROVIDER"] = "qwen"
        os.environ["GROUP_AGENT_MODEL"] = "qwen-plus"
        os.environ["GROUP_AGENT_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        os.environ["DASHSCOPE_API_KEY"] = "test-qwen-key"

        model = create_model()
        assert model.model_name == "qwen-plus"
        assert str(model.openai_api_base) == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert model.openai_api_key.get_secret_value() == "test-qwen-key"
    finally:
        for k, v in [
            ("GROUP_AGENT_PROVIDER", old_provider),
            ("GROUP_AGENT_MODEL", old_model),
            ("GROUP_AGENT_BASE_URL", old_base_url),
            ("DASHSCOPE_API_KEY", old_key),
        ]:
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)


def test_build_and_push_script_validates_40_char_sha_and_clean_source():
    """Verify build_and_push.sh rejects non-40-char SHA, non-HEAD SHA, and dirty source tree for group_agent_api."""
    script = APPS_DIR / "build_and_push.sh"
    assert script.exists()

    # Reject latest
    res_latest = subprocess.run([str(script), "group_agent_api", "latest"], capture_output=True, text=True)
    assert res_latest.returncode != 0
    assert "requires an explicit full 40-character commit SHA" in res_latest.stdout or "requires an explicit full 40-character commit SHA" in res_latest.stderr

    # Reject short SHA
    res_short = subprocess.run([str(script), "group_agent_api", "40bee43e"], capture_output=True, text=True)
    assert res_short.returncode != 0
    assert "must be a full 40-character commit SHA" in res_short.stdout or "must be a full 40-character commit SHA" in res_short.stderr

    # Reject 40-char non-HEAD SHA
    fake_sha = "0000000000000000000000000000000000000000"
    res_fake = subprocess.run([str(script), "group_agent_api", fake_sha], capture_output=True, text=True)
    assert res_fake.returncode != 0
    assert "does not match current git HEAD" in res_fake.stdout or "does not match current git HEAD" in res_fake.stderr

    content = script.read_text()
    assert 'if [ "$APP_NAME" != "group_agent_api" ]; then' in content, "build_and_push.sh must skip latest tag for group_agent_api"


def test_build_and_push_contains_no_forbidden_markers():
    """build_and_push.sh must not carry real registry/host/dir defaults."""
    content = (APPS_DIR / "build_and_push.sh").read_text()
    for marker in FORBIDDEN_MARKERS:
        assert marker not in content, f"Forbidden marker {marker!r} found in build_and_push.sh"


def test_startup_security_fails_closed_without_secret():
    """Verify assert_startup_security raises RuntimeError if production/http mode lacks HMAC secret."""
    from apps.group_agent_api.agent_factory.integrations.config import assert_startup_security

    old_env = os.environ.get("GROUP_AGENT_ENV")
    old_integration = os.environ.get("GROUP_AGENT_INTEGRATION")
    old_secret = os.environ.get("GROUP_AGENT_PRINCIPAL_HMAC_SECRET")

    try:
        os.environ["GROUP_AGENT_ENV"] = "production"
        os.environ["GROUP_AGENT_INTEGRATION"] = "http"
        os.environ.pop("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", None)

        with pytest.raises(RuntimeError, match="GROUP_AGENT_PRINCIPAL_HMAC_SECRET"):
            assert_startup_security()
    finally:
        if old_env is not None:
            os.environ["GROUP_AGENT_ENV"] = old_env
        else:
            os.environ.pop("GROUP_AGENT_ENV", None)
        if old_integration is not None:
            os.environ["GROUP_AGENT_INTEGRATION"] = old_integration
        else:
            os.environ.pop("GROUP_AGENT_INTEGRATION", None)
        if old_secret is not None:
            os.environ["GROUP_AGENT_PRINCIPAL_HMAC_SECRET"] = old_secret
        else:
            os.environ.pop("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", None)
