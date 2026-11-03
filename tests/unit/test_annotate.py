"""scripts/annotate.py: saves after every answer, resumes, goes back, and stays blind."""

from __future__ import annotations

import builtins
import csv
import importlib.util
import io
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "annotate.py"
COLUMNS = ["item_id", "claim", "cited_passage", "extracts", "human_verdict"]


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("annotate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["annotate"] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


def _csv(tmp_path: Path, verdicts: list[str]) -> Path:
    path = tmp_path / "annotate_morning_1.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for i, verdict in enumerate(verdicts, start=1):
            writer.writerow(
                [
                    f"A{i:02d}",
                    f"Claim {i} costs 4{i} dollars.",
                    f"Passage {i}.",
                    f"[1] Ex {i}",
                    verdict,
                ]
            )
    return path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _plain(screen: str) -> str:
    return re.sub(r"\033\[[0-9;]*[A-Za-z]", "", screen)


def _no_input(prompt: str) -> str:
    raise AssertionError(f"unexpected prompt {prompt!r}")


def _run(mod: ModuleType, path: Path, keys: str, ask: Any = _no_input) -> list[str]:
    screens: list[str] = []
    mod.run(mod.load(path), keys, ask, screens.append)
    return screens


def test_every_answer_is_saved_before_the_next_claim(tmp_path: Path) -> None:
    mod = _module()
    path = _csv(tmp_path, ["", "", ""])
    session = mod.load(path)
    mod.step(session, "s", _no_input)
    assert [r["human_verdict"] for r in _rows(path)] == ["supported", "", ""]
    mod.step(session, "u", _no_input)
    assert [r["human_verdict"] for r in _rows(path)] == ["supported", "unsupported", ""]
    # Columns the kappa step reads are kept; the comment column is added.
    assert list(_rows(path)[0]) == [*COLUMNS, "comment"]


def test_a_new_session_resumes_at_the_first_claim_without_a_verdict(tmp_path: Path) -> None:
    mod = _module()
    path = _csv(tmp_path, ["supported", "unsupported", "", ""])
    session = mod.load(path)
    assert session.index == 2
    assert "3/4" in mod.render(session) and "(2 annotated)" in mod.render(session)


def test_quitting_then_relaunching_continues_where_it_stopped(tmp_path: Path) -> None:
    mod = _module()
    path = _csv(tmp_path, ["", "", ""])
    _run(mod, path, "sq")
    assert mod.load(path).index == 1
    _run(mod, path, "uu")
    assert [r["human_verdict"] for r in _rows(path)] == ["supported", "unsupported", "unsupported"]
    assert mod.load(path).index == 3  # all done


def test_back_returns_to_the_previous_claim_and_lets_it_be_changed(tmp_path: Path) -> None:
    mod = _module()
    path = _csv(tmp_path, ["", "", ""])
    session = mod.load(path)
    for key in "ss":
        mod.step(session, key, _no_input)
    mod.step(session, "b", _no_input)
    assert session.index == 1
    mod.step(session, "u", _no_input)
    assert [r["human_verdict"] for r in _rows(path)] == ["supported", "unsupported", ""]
    assert session.index == 2


def test_back_works_after_a_resume_and_stops_at_the_first_claim(tmp_path: Path) -> None:
    mod = _module()
    path = _csv(tmp_path, ["supported", ""])
    session = mod.load(path)
    mod.step(session, "b", _no_input)
    assert session.index == 0
    mod.step(session, "b", _no_input)
    assert session.index == 0


def test_comment_is_saved_without_changing_the_verdict(tmp_path: Path) -> None:
    mod = _module()
    path = _csv(tmp_path, ["", ""])
    _run(mod, path, "c", ask=lambda _prompt: "  number off by one  ")
    row = _rows(path)[0]
    assert row["comment"] == "number off by one" and row["human_verdict"] == ""


def test_extracts_show_only_on_request(tmp_path: Path) -> None:
    mod = _module()
    screens = [_plain(s) for s in _run(mod, _csv(tmp_path, [""]), "ee")]
    assert "[1] Ex 1" not in screens[0]
    assert "[1] Ex 1" in screens[1]
    assert "[1] Ex 1" not in screens[2]


def test_unknown_keys_change_nothing(tmp_path: Path) -> None:
    mod = _module()
    path = _csv(tmp_path, ["", ""])
    before = path.read_bytes()
    _run(mod, path, "xz1 ")
    assert path.read_bytes() == before


def test_numbers_months_negations_and_names_are_highlighted() -> None:
    mod = _module()
    out = mod.highlight("Fees of US$42.00 are not due in March to Northwind Logistics.")
    for span in ("US$42.00", "not", "March", "Northwind Logistics"):
        assert f"{mod.YELLOW}{span}{mod.RESET}" in out
    assert f"{mod.YELLOW}Fees" not in out  # first word of the sentence


# --- blindness -----------------------------------------------------------------------


def test_the_session_opens_only_the_csv_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _module()
    path = _csv(tmp_path, ["", "", ""])
    # The answer key and the judge's verdicts sit next to the CSV, as in the real folder.
    (tmp_path / "judge_key.json").write_text('{"A01": {"origin": "perturbed"}}')
    (tmp_path / "perturbed_verdicts.json").write_text("{}")
    opened: list[str] = []
    real_open, real_os_open = builtins.open, os.open

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(file, int):  # os.fdopen re-opens a descriptor already counted
            opened.append(os.path.basename(str(file)))
        return real_open(file, *args, **kwargs)

    def spy_os_open(file: Any, *args: Any, **kwargs: Any) -> int:
        opened.append(os.path.basename(str(file)))
        return real_os_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_open)  # Path.open goes through io.open
    monkeypatch.setattr(os, "open", spy_os_open)
    monkeypatch.setattr(Path, "read_text", lambda *_a, **_k: pytest.fail("read_text called"))
    monkeypatch.setattr(Path, "read_bytes", lambda *_a, **_k: pytest.fail("read_bytes called"))
    _run(mod, path, "suebsc", ask=lambda _p: "note")

    assert opened, "the spy saw no file access"
    assert set(opened) <= {path.name} | {n for n in opened if n.endswith(".tmp")}
    assert not {"judge_key.json", "perturbed_verdicts.json"} & set(opened)


def test_the_source_never_names_a_key_file_except_to_refuse_it() -> None:
    source = SCRIPT.read_text()
    forbidden_line = next(line for line in source.splitlines() if "FORBIDDEN_NAMES = " in line)
    for name in ("judge_key", "perturbed_verdicts", "agreement.json"):
        mentions = [line for line in source.splitlines() if name in line]
        assert mentions == [forbidden_line], name


@pytest.mark.parametrize("name", ["judge_key.json", "perturbed_verdicts.json", "notes.txt"])
def test_it_refuses_to_open_anything_but_an_annotation_csv(tmp_path: Path, name: str) -> None:
    mod = _module()
    path = tmp_path / name
    path.write_text("{}")
    with pytest.raises(SystemExit):
        mod.load(path)


def test_screens_show_no_origin_id_or_judge_information(tmp_path: Path) -> None:
    mod = _module()
    path = _csv(tmp_path, ["", "", ""])
    text = _plain("".join(_run(mod, path, "esubeq"))).lower()
    for word in ("perturb", "origin", "real", "judge", "key", "kind", "a01", "a02"):
        assert word not in text, word


def test_main_rejects_a_missing_argument(capsys: pytest.CaptureFixture[str]) -> None:
    assert _module().main([]) == 2
    assert "annotate.py" in capsys.readouterr().err


def test_render_on_completion_reports_the_count(tmp_path: Path) -> None:
    mod = _module()
    session = mod.load(_csv(tmp_path, ["supported", "unsupported"]))
    assert "2/2 annotated" in mod.render(session)
