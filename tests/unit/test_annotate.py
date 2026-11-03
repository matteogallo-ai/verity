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


def _spy_files(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the name of every file opened; fail on any directory listing or text read."""
    opened: list[str] = []
    real_open, real_os_open, real_read_bytes = builtins.open, os.open, Path.read_bytes

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(file, int):  # os.fdopen re-opens a descriptor already counted
            opened.append(os.path.basename(str(file)))
        return real_open(file, *args, **kwargs)

    def spy_os_open(file: Any, *args: Any, **kwargs: Any) -> int:
        opened.append(os.path.basename(str(file)))
        return real_os_open(file, *args, **kwargs)

    def spy_read_bytes(self: Path) -> bytes:
        opened.append(self.name)
        return real_read_bytes(self)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_open)  # Path.open goes through io.open
    monkeypatch.setattr(os, "open", spy_os_open)
    monkeypatch.setattr(Path, "read_bytes", spy_read_bytes)
    monkeypatch.setattr(Path, "read_text", lambda *_a, **_k: pytest.fail("read_text called"))
    monkeypatch.setattr(os, "listdir", lambda *_a, **_k: pytest.fail("listdir called"))
    monkeypatch.setattr(Path, "iterdir", lambda *_a, **_k: pytest.fail("iterdir called"))
    monkeypatch.setattr(Path, "glob", lambda *_a, **_k: pytest.fail("glob called"))
    return opened


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
    opened = _spy_files(monkeypatch)
    _run(mod, path, "suebsc", ask=lambda _p: "note")

    assert path.name in opened
    temporaries = [n for n in opened if n != path.name]
    assert all(n.endswith(".tmp") and n.startswith("tmp") for n in temporaries), temporaries


@pytest.mark.parametrize("name", ["judge_key.json", "perturbed_verdicts.json", "notes.txt"])
def test_it_refuses_to_open_anything_but_an_annotation_csv(tmp_path: Path, name: str) -> None:
    mod = _module()
    path = tmp_path / name
    path.write_text("{}")
    with pytest.raises(SystemExit):
        mod.load(path)


def test_screens_show_no_origin_id_or_any_extra_column(tmp_path: Path) -> None:
    mod = _module()
    path = tmp_path / "leaky.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([*COLUMNS, "origin", "kind", "judge_verdict"])
        writer.writerow(["A01", "Claim one.", "Passage.", "[1] Ex", "", "perturbed", "date", "x"])
        writer.writerow(["A02", "Claim two.", "Passage.", "[1] Ex", "", "real", "", "y"])
    text = _plain("".join(_run(mod, path, "esubeq"))).lower()
    for word in ("perturb", "origin", "real", "judge", "date", "kind", "a01", "a02"):
        assert word not in text, word


# --- failure modes -------------------------------------------------------------------


def test_arrow_keys_never_act_as_commands(tmp_path: Path) -> None:
    mod = _module()
    path = _csv(tmp_path, ["supported", "", ""])
    session = mod.load(path)
    stream = iter("\x1b[B\x1b[C\x1b[A\x1b[D\x1bOB\x1b[3~")
    keys = list(mod.keys_from(lambda: next(stream, "")))
    assert keys == []  # Down ends in "B", Right in "C": swallowed before step()
    mod.run(session, keys, _no_input, lambda _s: None)
    assert session.index == 1


def test_keys_from_passes_plain_keys_and_stops_at_eof_or_ctrl_d() -> None:
    mod = _module()
    stream = iter("su\x04q")
    assert list(mod.keys_from(lambda: next(stream, ""))) == ["s", "u"]


def test_ctrl_c_quits_cleanly_with_answers_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _module()
    path = _csv(tmp_path, ["", ""])

    def keys() -> Any:
        yield "s"
        raise KeyboardInterrupt

    monkeypatch.setattr(mod, "_terminal_keys", keys)
    assert mod.main([str(path)]) == 0
    assert "Saved" in capsys.readouterr().out
    assert [r["human_verdict"] for r in _rows(path)] == ["supported", ""]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("A01,claim,passage,extracts\n", "wrong number of fields"),
        ("A01,claim,passage,extracts,,,extra\n", "wrong number of fields"),
        ("A01,claim,passage,extracts,yes\n", "not one of"),
    ],
)
def test_a_malformed_csv_is_refused_with_its_line(tmp_path: Path, body: str, message: str) -> None:
    mod = _module()
    path = tmp_path / "bad.csv"
    path.write_text(",".join([*COLUMNS, "comment"]) + "\n" + body)
    with pytest.raises(SystemExit, match=message) as err:
        mod.load(path)
    assert "line 2" in str(err.value)


def test_a_bom_from_a_spreadsheet_is_accepted(tmp_path: Path) -> None:
    mod = _module()
    path = tmp_path / "bom.csv"
    path.write_bytes(("\ufeff" + ",".join(COLUMNS) + "\nA01,c,p,e,Supported \n").encode())
    session = mod.load(path)
    assert session.rows[0]["human_verdict"] == "supported" and session.index == 1


def test_saving_keeps_permissions_line_endings_and_awkward_fields(tmp_path: Path) -> None:
    mod = _module()
    path = tmp_path / "f.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerow(["A01", 'He said "no, never"', "line one\nline two", "a,b", ""])
    path.chmod(0o644)
    _run(mod, path, "u")
    assert path.stat().st_mode & 0o777 == 0o644
    assert b"\r\n" not in path.read_bytes()
    row = _rows(path)[0]
    assert row["claim"] == 'He said "no, never"' and row["cited_passage"] == "line one\nline two"


def test_escape_sequences_in_document_text_never_reach_the_terminal(tmp_path: Path) -> None:
    mod = _module()
    path = tmp_path / "esc.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        writer.writerow(["A01", "Claim\x1b[2J hidden", "Pass\x1b]0;x\x07age", "Ex\x9b2J", ""])
    screen = "".join(_run(mod, path, "e"))
    payload = screen.replace(mod.CLEAR, "").replace(mod.BOLD, "").replace(mod.RESET, "")
    payload = payload.replace(mod.YELLOW, "")
    assert "\x1b" not in payload and "\x07" not in payload and "\x9b" not in payload
    assert "Claim[2J hidden" in payload


def test_a_comment_can_be_cleared(tmp_path: Path) -> None:
    mod = _module()
    path = _csv(tmp_path, [""])
    answers = iter(["first", "-"])
    _run(mod, path, "cc", ask=lambda _p: next(answers))
    assert _rows(path)[0]["comment"] == ""


def test_back_from_the_completion_screen_reopens_the_last_claim(tmp_path: Path) -> None:
    mod = _module()
    session = mod.load(_csv(tmp_path, ["supported", "unsupported"]))
    mod.step(session, "b", _no_input)
    assert session.index == 1 and "2/2" in mod.render(session)


def test_main_rejects_a_missing_argument(capsys: pytest.CaptureFixture[str]) -> None:
    assert _module().main([]) == 2
    assert "annotate.py" in capsys.readouterr().err


def test_render_on_completion_reports_the_count(tmp_path: Path) -> None:
    mod = _module()
    session = mod.load(_csv(tmp_path, ["supported", "unsupported"]))
    assert "2/2 annotated" in mod.render(session)


def test_typographic_apostrophe_negations_are_highlighted() -> None:
    mod = _module()
    assert f"{mod.YELLOW}doesn\u2019t{mod.RESET}" in mod.highlight("It doesn\u2019t apply.")


# --- back on a finished file, review mode --------------------------------------------


@pytest.mark.parametrize("key", ["b", "B"])
def test_back_works_on_a_fully_annotated_file_whatever_the_case(tmp_path: Path, key: str) -> None:
    # Caps Lock or Shift sends "B": on the finished screen it used to do nothing.
    mod = _module()
    path = _csv(tmp_path, ["supported", "unsupported", "supported"])
    session = mod.load(path)
    assert session.index == 3
    mod.step(session, key, _no_input)
    assert session.index == 2
    mod.step(session, key.swapcase(), _no_input)
    assert session.index == 1


def _main(mod: ModuleType, monkeypatch: pytest.MonkeyPatch, args: list[str], keys: str) -> int:
    monkeypatch.setattr(mod, "_terminal_keys", lambda: iter(keys))
    code: int = mod.main(args)
    return code


def test_review_starts_at_claim_one_and_shows_the_current_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _module()
    path = _csv(tmp_path, ["supported", "unsupported", "supported"])
    assert _main(mod, monkeypatch, ["--review", str(path)], "q") == 0
    screen = _plain(capsys.readouterr().out)
    assert "REVIEW   1/3" in screen and "Current verdict: supported" in screen


def test_review_enter_keeps_and_s_u_replace_each_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _module()
    path = _csv(tmp_path, ["supported", "unsupported", "supported"])
    session = mod.load(path)
    session.review, session.index = True, 0
    mod.step(session, "\n", _no_input)  # keep claim 1
    assert session.index == 1 and _rows(path)[0]["human_verdict"] == "supported"
    mod.step(session, "s", _no_input)  # replace claim 2, saved at once
    assert [r["human_verdict"] for r in _rows(path)] == ["supported", "supported", "supported"]
    mod.step(session, "\r", _no_input)
    assert session.index == 3


def test_enter_never_skips_an_unanswered_claim(tmp_path: Path) -> None:
    mod = _module()
    session = mod.load(_csv(tmp_path, ["", ""]))
    mod.step(session, "\n", _no_input)
    assert session.index == 0


def test_from_jumps_to_claim_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _module()
    path = _csv(tmp_path, ["supported", "unsupported", "supported", "supported"])
    assert _main(mod, monkeypatch, ["--review", "--from", "3", str(path)], "u") == 0
    assert [r["human_verdict"] for r in _rows(path)][2] == "unsupported"
    assert "REVIEW   3/4" in _plain(capsys.readouterr().out)


@pytest.mark.parametrize("start", ["0", "5"])
def test_from_out_of_range_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: str
) -> None:
    mod = _module()
    path = _csv(tmp_path, ["", "", "", ""])
    before = path.read_bytes()
    assert _main(mod, monkeypatch, ["--from", start, str(path)], "s") == 2
    assert path.read_bytes() == before


@pytest.mark.parametrize("args", [["--review"], ["--review", "--from", "2"], []])
def test_every_mode_opens_only_the_csv_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    mod = _module()
    path = _csv(tmp_path, ["supported", "unsupported", ""])
    (tmp_path / "judge_key.json").write_text('{"A01": {"origin": "perturbed"}}')
    (tmp_path / "perturbed_verdicts.json").write_text("{}")
    opened = _spy_files(monkeypatch)
    assert _main(mod, monkeypatch, [*args, str(path)], "\nsubeu\nq") == 0

    assert path.name in opened
    temporaries = [n for n in opened if n != path.name]
    assert all(n.endswith(".tmp") and n.startswith("tmp") for n in temporaries), temporaries


def test_review_screens_show_no_origin_id_or_extra_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _module()
    path = tmp_path / "leaky.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([*COLUMNS, "origin", "kind", "judge_verdict"])
        writer.writerow(
            ["A01", "Claim one.", "Passage.", "[1] Ex", "supported", "perturbed", "date", "x"]
        )
        writer.writerow(["A02", "Claim two.", "Passage.", "[1] Ex", "unsupported", "real", "", "y"])
    assert _main(mod, monkeypatch, ["--review", str(path)], "e\nbsq") == 0
    text = _plain(capsys.readouterr().out).lower()
    for word in ("perturb", "origin", "real", "judge", "date", "kind", "a01", "a02"):
        assert word not in text, word


# --- one session over two CSVs ---------------------------------------------------------


def _two(tmp_path: Path, first: list[str], second: list[str]) -> tuple[Path, Path]:
    one = _csv(tmp_path, first)
    two = one.with_name("annotate_morning_2.csv")
    with two.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for i, verdict in enumerate(second, start=1):
            writer.writerow([f"B{i:02d}", f"Other {i}.", f"Text {i}.", f"[1] Y {i}", verdict])
    return one, two


def test_two_files_form_one_session_numbered_across_them(tmp_path: Path) -> None:
    mod = _module()
    one, two = _two(tmp_path, ["", ""], ["", "", ""])
    session = mod.load([one, two])
    assert len(session.rows) == 5 and session.index == 0
    assert "1/5   (0 annotated)" in _plain(mod.render(session))
    session.index = 2
    assert "Other 1." in _plain(mod.render(session))  # first claim of the second file


def test_each_answer_is_saved_to_the_file_its_claim_came_from(tmp_path: Path) -> None:
    mod = _module()
    one, two = _two(tmp_path, ["", ""], ["", ""])
    second_before = two.read_bytes()
    session = mod.load([one, two])
    for key in "su":
        mod.step(session, key, _no_input)
    assert [r["human_verdict"] for r in _rows(one)] == ["supported", "unsupported"]
    assert two.read_bytes() == second_before  # untouched until one of its claims changes
    mod.step(session, "s", _no_input)
    assert [r["human_verdict"] for r in _rows(two)] == ["supported", ""]
    assert [r["item_id"] for r in _rows(two)] == ["B01", "B02"]  # order kept


def test_two_file_session_resumes_across_the_boundary(tmp_path: Path) -> None:
    mod = _module()
    one, two = _two(tmp_path, ["supported", "supported"], ["unsupported", ""])
    session = mod.load([one, two])
    assert session.index == 3 and session.done == 3


@pytest.mark.parametrize("key", ["b", "B"])
def test_back_from_the_end_of_two_finished_files_crosses_back_into_the_first(
    tmp_path: Path, key: str
) -> None:
    mod = _module()
    one, two = _two(tmp_path, ["supported", "supported"], ["unsupported", "supported"])
    session = mod.load([one, two])
    assert session.index == 4  # finished screen
    for _ in range(3):
        mod.step(session, key, _no_input)
    assert session.index == 1  # second claim of the first file
    mod.step(session, "u", _no_input)
    assert [r["human_verdict"] for r in _rows(one)] == ["supported", "unsupported"]


def test_review_walks_both_files_and_from_counts_across_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _module()
    one, two = _two(tmp_path, ["supported", "supported"], ["unsupported", "supported"])
    assert _main(mod, monkeypatch, ["--review", "--from", "3", str(one), str(two)], "s\nq") == 0
    assert [r["human_verdict"] for r in _rows(two)] == ["supported", "supported"]
    assert [r["human_verdict"] for r in _rows(one)] == ["supported", "supported"]
    out = _plain(capsys.readouterr().out)
    assert "REVIEW   3/4" in out and "REVIEW   4/4" in out


def test_the_same_file_twice_is_refused(tmp_path: Path) -> None:
    mod = _module()
    one = _csv(tmp_path, [""])
    with pytest.raises(SystemExit, match="twice"):
        mod.load([one, one])


@pytest.mark.parametrize("args", [[], ["--review"], ["--review", "--from", "3"]])
def test_a_two_file_session_opens_only_its_two_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    mod = _module()
    one, two = _two(tmp_path, ["supported", ""], ["", ""])
    (tmp_path / "judge_key.json").write_text('{"A01": {"origin": "perturbed"}}')
    (tmp_path / "perturbed_verdicts.json").write_text("{}")
    opened = _spy_files(monkeypatch)
    assert _main(mod, monkeypatch, [*args, str(one), str(two)], "\nsusubq") == 0

    assert {one.name, two.name} <= set(opened)
    others = [n for n in opened if n not in (one.name, two.name)]
    assert all(n.endswith(".tmp") and n.startswith("tmp") for n in others), others


ADJ_COLUMNS = [
    "item_id",
    "claim",
    "cited_passage",
    "extracts",
    "blind_verdict",
    "judge_verdict",
    "adjudicated_verdict",
    "reason",
]


def _adj(tmp_path: Path, decided: list[tuple[str, str]]) -> Path:
    path = tmp_path / "adjudication.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(ADJ_COLUMNS)
        for i, (verdict, reason) in enumerate(decided, start=1):
            writer.writerow(
                [
                    f"A{i:02d}",
                    f"Claim {i}.",
                    f"Passage {i}.",
                    f"[1] Ex {i}",
                    "supported",
                    "unsupported",
                    verdict,
                    reason,
                ]
            )
    return path


def _adj_session(mod: ModuleType, path: Path) -> Any:
    return mod.load(path, mod.ADJUDICATE)


def test_adjudication_shows_both_verdicts_next_to_the_claim(tmp_path: Path) -> None:
    mod = _module()
    screen = _plain(mod.render(_adj_session(mod, _adj(tmp_path, [("", "")]))))
    assert "ADJUDICATE   1/1   (0 adjudicated)" in screen
    assert "Your blind verdict: supported" in screen
    assert "Judge's verdict: unsupported" in screen
    assert "c reason" in screen


def test_a_decision_is_saved_with_its_reason_and_blind_columns_untouched(tmp_path: Path) -> None:
    mod = _module()
    path = _adj(tmp_path, [("", ""), ("", "")])
    session = _adj_session(mod, path)
    mod.step(session, "u", lambda _p: "the passage says 70.9%")
    rows = _rows(path)
    assert (rows[0]["adjudicated_verdict"], rows[0]["reason"]) == (
        "unsupported",
        "the passage says 70.9%",
    )
    assert rows[1]["adjudicated_verdict"] == ""
    assert [(r["blind_verdict"], r["judge_verdict"]) for r in rows] == [
        ("supported", "unsupported")
    ] * 2
    assert list(rows[0]) == ADJ_COLUMNS
    assert session.index == 1


@pytest.mark.parametrize("typed", ["", "   "])
def test_no_decision_is_recorded_without_a_reason(tmp_path: Path, typed: str) -> None:
    mod = _module()
    path = _adj(tmp_path, [("", "")])
    before = path.read_bytes()
    session = _adj_session(mod, path)
    mod.step(session, "s", lambda _p: typed)
    assert path.read_bytes() == before
    assert session.index == 0
    assert "A reason is required" in mod.render(session)


def test_reconfirming_a_decision_may_keep_its_reason(tmp_path: Path) -> None:
    mod = _module()
    path = _adj(tmp_path, [("supported", "first reading")])
    session = _adj_session(mod, path)
    assert session.index == 1  # resumes past the decided claim
    mod.step(session, "b", _no_input)
    mod.step(session, "s", lambda _p: "")
    assert (_rows(path)[0]["adjudicated_verdict"], _rows(path)[0]["reason"]) == (
        "supported",
        "first reading",
    )


def test_flipping_a_decision_requires_a_new_reason(tmp_path: Path) -> None:
    mod = _module()
    path = _adj(tmp_path, [("supported", "passage states X")])
    before = path.read_bytes()
    session = _adj_session(mod, path)
    session.index = 0
    mod.step(session, "u", lambda _p: "")  # the old reason argued for the other verdict
    assert path.read_bytes() == before
    mod.step(session, "u", lambda _p: "X is about another quarter")
    assert (_rows(path)[0]["adjudicated_verdict"], _rows(path)[0]["reason"]) == (
        "unsupported",
        "X is about another quarter",
    )


def test_a_dash_is_not_a_reason(tmp_path: Path) -> None:
    mod = _module()
    path = _adj(tmp_path, [("", "")])
    before = path.read_bytes()
    session = _adj_session(mod, path)
    mod.step(session, "s", lambda _p: " - ")
    assert path.read_bytes() == before
    assert "A reason is required" in mod.render(session)


def test_adjudication_review_is_labelled_as_both(tmp_path: Path) -> None:
    mod = _module()
    session = _adj_session(mod, _adj(tmp_path, [("supported", "r")]))
    session.review, session.index = True, 0
    assert "ADJUDICATE REVIEW   1/1" in _plain(mod.render(session))


@pytest.mark.parametrize(("adjudicate", "word"), [(False, "annotated"), (True, "adjudicated")])
def test_the_exit_line_names_what_was_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    adjudicate: bool,
    word: str,
) -> None:
    mod = _module()
    path = _adj(tmp_path, [("", "")]) if adjudicate else _csv(tmp_path, [""])
    args = ["--adjudicate", str(path)] if adjudicate else [str(path)]
    assert _main(mod, monkeypatch, args, "q") == 0
    assert f"(0/1 {word})" in capsys.readouterr().out


def test_the_reason_cannot_be_cleared(tmp_path: Path) -> None:
    mod = _module()
    path = _adj(tmp_path, [("supported", "first reading")])
    session = _adj_session(mod, path)
    session.index = 0
    mod.step(session, "c", lambda _p: "-")
    assert _rows(path)[0]["reason"] == "first reading"
    mod.step(session, "c", lambda _p: "better reason")
    assert _rows(path)[0]["reason"] == "better reason"


def test_adjudication_resumes_at_the_first_undecided_claim(tmp_path: Path) -> None:
    mod = _module()
    path = _adj(tmp_path, [("supported", "r1"), ("", ""), ("", "")])
    session = _adj_session(mod, path)
    assert session.index == 1
    assert "(1 adjudicated)" in _plain(mod.render(session))


def test_an_annotation_csv_is_refused_in_adjudication_mode(tmp_path: Path) -> None:
    mod = _module()
    with pytest.raises(SystemExit, match="blind_verdict"):
        mod.load(_csv(tmp_path, [""]), mod.ADJUDICATE)


@pytest.mark.parametrize("extra", [[], ["--review"], ["--from", "2"]])
def test_adjudication_opens_only_the_file_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    mod = _module()
    path = _adj(tmp_path, [("", ""), ("", ""), ("", "")])
    # The blind labels and the judge's verdicts sit next to it, as in the real folder.
    for name in ("annotate_morning_1.csv", "judge_key.json", "agreement.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(mod, "_ask", lambda _p: "a reason")
    opened = _spy_files(monkeypatch)
    assert _main(mod, monkeypatch, ["--adjudicate", *extra, str(path)], "subcq") == 0

    assert path.name in opened
    temporaries = [n for n in opened if n != path.name]
    assert all(n.endswith(".tmp") and n.startswith("tmp") for n in temporaries), temporaries
