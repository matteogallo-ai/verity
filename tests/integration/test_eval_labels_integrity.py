"""Every evidence quote is in a chunk of this machine's parse, and the dev/test split is
the seeded one.

The committed chunk ids are the labelling machine's parse and are pinned against the
committed docling fixture in the fast lane (``tests/unit/test_eval_labels.py``); a live
parse on other hardware may chunk the PDFs differently, and the eval resolves its gold
from the quotes, so what must hold here is that every quote is still found.

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


def test_every_evidence_quote_is_in_a_chunk_of_the_live_parse() -> None:
    labels = _script("build_eval_labels")
    _, problems = labels.label(_records(labels.DATASET_PATH), asyncio.run(labels._load_chunks()))
    assert problems == []


def test_committed_split_is_the_seeded_split() -> None:
    splitter = _script("split_dataset")
    dev, test = splitter.split(_records(splitter.SOURCE))
    assert _records(splitter.DEV) == dev
    assert _records(splitter.TEST) == test
    assert {r["id"] for r in dev}.isdisjoint({r["id"] for r in test})
