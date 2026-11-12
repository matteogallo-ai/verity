"""The CCT demo profile (profiles/demo-cct-fr.env, used by ``make demo-cct``).

It holds the settings the prospection bench measured on French collective labour
agreements. The Makefile sources it in a shell, so the test reads it the same way, then
ingests an excerpt of a real agreement under it: every passage must carry its article.
No model is loaded and no LLM is called.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.unit.api_app import FakeEmbeddingBackend
from verity.config import get_settings
from verity.ingestion import LocalEmbedder
from verity.retrieval.memory_store import InMemoryVectorStore
from verity.runtime import ingest_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPO_ROOT / "profiles" / "demo-cct-fr.env"
EXCERPT = REPO_ROOT / "tests" / "fixtures" / "articles" / "paysagistes_vd_art8-10.txt"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")


def _sourced(profile: Path) -> dict[str, str]:
    """The VERITY_ variables a shell exports after ``set -a; . profile``."""
    out = subprocess.run(
        ["bash", "-c", 'set -a; . "$1"; env -0', "_", str(profile)],
        capture_output=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    ).stdout.decode()
    pairs = (item.split("=", 1) for item in out.split("\0") if item.startswith("VERITY_"))
    return dict(pairs)


@pytest.fixture
def profile(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    values = _sourced(PROFILE)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    yield values
    get_settings.cache_clear()


def test_the_profile_holds_the_settings_the_bench_measured(profile: dict[str, str]) -> None:
    settings = get_settings()
    assert settings.language == "fr"
    assert settings.chunking == "articles"
    assert settings.index_tables is True
    assert settings.name_articles is True
    assert settings.retrieve_original_question is True
    assert settings.complete_answers is False
    assert settings.llm_temperature == 0.0
    assert settings.synthesis_max_tokens == 1024
    assert settings.min_citation_coverage == 0.5
    assert settings.store == "memory"
    assert settings.refusal_text == (
        "Les documents fournis ne permettent pas de répondre. "
        "Contactez le secrétariat de l'association."
    )


def test_the_profile_holds_no_secret() -> None:
    names = re.findall(r"^\s*(?:export\s+)?([A-Z0-9_]+)=", PROFILE.read_text(), re.MULTILINE)
    assert names
    secret = re.compile(r"(?:_KEYS?|_TOKEN|_SECRET|_PASSWORD|DATABASE_URL)$")
    assert not [n for n in names if secret.search(n)]
    assert secret.search("VERITY_ANTHROPIC_API_KEY")
    assert not secret.search("VERITY_SYNTHESIS_MAX_TOKENS")


@pytest.mark.asyncio
async def test_a_cct_excerpt_ingested_under_the_profile_is_cut_by_article(
    profile: dict[str, str], tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy(EXCERPT, corpus / EXCERPT.name)
    store = InMemoryVectorStore()
    embedder = LocalEmbedder(backend=FakeEmbeddingBackend(384))

    await ingest_corpus(corpus_dir=corpus, store=store, embedder=embedder)

    hits = await store.sparse_search("travail salaire dimanche vacances", k=50)
    assert hits
    assert {hit.chunk.metadata["article"] for hit in hits} == {"8", "9", "10"}
    sunday = [hit.chunk for hit in hits if "supplément de 50%" in hit.chunk.text]
    assert [c.section for c in sunday] == [
        "Article 9 \N{EN DASH} Travail du dimanche ou jours fériés"
    ]
