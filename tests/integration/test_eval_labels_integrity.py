"""The committed eval labels and the dev/test split are exactly what the scripts produce.

Guards two regressions: chunk ids that stop resolving (``document_id_for_uri`` deriving
ids from machine paths, a chunker change), and a dataset edited by hand without
re-running ``scripts/build_eval_labels.py`` or ``scripts/split_dataset.py``. Every
answerable question's evidence quotes must each be found in a chunk.

Integration lane: parsing the corpus loads docling (PDF and DOCX documents).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_committed_labels_are_the_evidence_resolved_against_the_corpus() -> None:
    labels = _script("build_eval_labels")
    committed = _records(labels.DATASET_PATH)
    relabelled, problems = labels.label(committed, asyncio.run(labels._load_chunks()))
    assert problems == []
    assert relabelled == committed, "run `uv run python scripts/build_eval_labels.py`"


def test_committed_split_is_the_seeded_split() -> None:
    splitter = _script("split_dataset")
    dev, test = splitter.split(_records(splitter.SOURCE))
    assert _records(splitter.DEV) == dev
    assert _records(splitter.TEST) == test
    assert {r["id"] for r in dev}.isdisjoint({r["id"] for r in test})
