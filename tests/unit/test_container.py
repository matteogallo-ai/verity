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


def test_runtime_image_carries_the_docling_models_offline() -> None:
    """Without them the first PDF upload tries to download ~700 MB and fails, since
    the runtime image sets HF_HUB_OFFLINE. scripts/docker_smoke.sh uploads a PDF."""
    runtime = _DOCKERFILE.rsplit("FROM ", 1)[1]
    assert "docling-tools models download" in _DOCKERFILE
    assert "COPY --from=builder /opt/docling /opt/docling" in runtime
    assert "DOCLING_ARTIFACTS_PATH=/opt/docling" in runtime
    assert "HF_HUB_OFFLINE=1" in runtime


_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def _external_images() -> list[str]:
    """Every image pulled from a registry: Dockerfile FROM and COPY --from, Compose and
    the CI service containers. ``builder`` and ``verity:local`` are built here."""
    images = re.findall(r"^FROM (\S+)", _DOCKERFILE, re.M)
    images += re.findall(r"^COPY --from=(\S+)", _DOCKERFILE, re.M)
    images += [s["image"] for s in _COMPOSE["services"].values() if "image" in s]
    ci = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    images += [s["image"] for job in ci["jobs"].values() for s in job.get("services", {}).values()]
    return [i for i in images if i not in {"builder", "verity:local"}]


def test_every_external_image_is_pinned_by_digest() -> None:
    images = _external_images()
    assert len(images) >= 5
    assert [i for i in images if not _DIGEST.search(i)] == []


def test_no_workflow_checkout_keeps_the_token_in_git_config() -> None:
    for path in sorted(Path(".github/workflows").glob("*.yml")):
        for job in yaml.safe_load(path.read_text())["jobs"].values():
            for step in job.get("steps", []):
                if str(step.get("uses", "")).startswith("actions/checkout@"):
                    assert step.get("with", {}).get("persist-credentials") is False, path


def test_ci_tests_the_pgvector_image_compose_ships() -> None:
    """Dependabot moves the Compose digest; no ecosystem moves a workflow service image,
    so this keeps CI from testing an older pgvector than the one shipped."""
    ci = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    services = [s["image"] for j in ci["jobs"].values() for s in j.get("services", {}).values()]
    assert services and set(services) == {_COMPOSE["services"]["db"]["image"]}
