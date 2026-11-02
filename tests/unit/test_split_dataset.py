"""The eval dataset and its frozen dev/test split (``scripts/split_dataset.py``)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "split_dataset.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("split_dataset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ids(path: Path) -> list[str]:
    return [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]


def test_dataset_has_the_published_size_and_mix() -> None:
    mod = _module()
    records = [json.loads(line) for line in mod.SOURCE.read_text().splitlines() if line.strip()]
    strata = [mod._stratum(r) for r in records]
    assert len(records) >= 80
    assert sum(1 for r in records if r["answerable"]) >= 50
    assert sum(1 for r in records if not r["answerable"]) >= 25
    assert strata.count("adversarial") >= 5


def test_committed_split_is_the_seeded_one() -> None:
    assert _module().main(["--check"]) == 0


def test_dev_and_test_partition_the_dataset() -> None:
    mod = _module()
    dev, test, source = _ids(mod.DEV), _ids(mod.TEST), _ids(mod.SOURCE)
    assert not set(dev) & set(test)
    assert sorted(dev + test) == sorted(source)


def test_every_stratum_is_on_both_sides() -> None:
    mod = _module()
    records = [json.loads(line) for line in mod.SOURCE.read_text().splitlines() if line.strip()]
    dev, test = mod.split(records)
    for stratum in ("answerable", "unanswerable", "adversarial"):
        assert any(mod._stratum(r) == stratum for r in dev)
        assert any(mod._stratum(r) == stratum for r in test)
