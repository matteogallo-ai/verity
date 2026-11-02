"""Static checks on the container setup. The running stack is exercised by
``scripts/docker_smoke.sh`` in the CI ``docker`` job (build, compose up, /health,
/status, 401 without a key, one stub /ask)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_COMPOSE = yaml.safe_load(Path("docker-compose.yml").read_text())
_DOCKERFILE = Path("Dockerfile").read_text()


def test_compose_runs_db_then_migrate_then_ingest_then_app() -> None:
    services = _COMPOSE["services"]
    assert services["migrate"]["depends_on"] == {"db": {"condition": "service_healthy"}}
    assert services["ingest"]["depends_on"] == {
        "migrate": {"condition": "service_completed_successfully"}
    }
    assert services["app"]["depends_on"] == {
        "ingest": {"condition": "service_completed_successfully"}
    }
    assert services["migrate"]["command"] == ["db", "upgrade"]
    assert services["ingest"]["command"] == ["ingest", "datasets/corpus"]


def test_published_ports_bind_loopback_only() -> None:
    for name, service in _COMPOSE["services"].items():
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:"), f"{name} publishes {port} on every interface"


def test_compose_never_opens_the_api_by_default() -> None:
    env = _COMPOSE["services"]["app"]["environment"]
    assert env["VERITY_API_AUTH_DISABLED"] == "${VERITY_API_AUTH_DISABLED:-false}"


def test_runtime_stage_runs_as_an_unprivileged_user() -> None:
    runtime = _DOCKERFILE.split("\nFROM ")[-1]
    users = re.findall(r"^USER (\S+)$", runtime, re.MULTILINE)
    assert users and users[-1] not in {"root", "0"}
    assert "HF_HUB_OFFLINE=1" in runtime
