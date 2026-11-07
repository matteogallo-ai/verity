"""Check docs/CLAIMS.md against the test suite and docs/features.md (rule 1).

What it verifies:
1. Every row of the claims table has a unique ID (``C<nn>``) and a status of
   ``shipped`` or ``roadmap``.
2. Every ``shipped`` claim lists at least one test.
3. Every listed pytest test file exists and defines the listed test (read with
   ``ast``, also under ``--no-run``), and every listed node id passes when run
   explicitly: the whole set runs in one pytest call (markers are not filtered, so
   ``slow`` and ``integration`` tests run too, which needs Postgres), and the JUnit
   report is mapped back to each claim. A node id with no result is ``missing``
   (pytest aborts the whole call on an unknown node id, so all then read missing and
   the printed pytest output names the culprit). A skipped test counts as not passing.
4. Listed ``web/`` tests (vitest) exist as files.
5. In docs/features.md, every bullet of the ``## Features`` section cites at least one
   claim ID; every cited ID is ``shipped``; every ID cited in ``## Roadmap`` is
   ``roadmap``; every ``shipped`` claim is cited under Features and every
   ``roadmap`` claim under Roadmap.

What it does NOT verify:
- that a test exercises what its claim says (a human reviews the mapping);
- text outside the Features and Roadmap sections of docs/features.md, README included;
- vitest results (the CI ``web`` job runs them);
- the implementation column (informative only).

Usage: ``python scripts/check_claims.py [--claims PATH] [--features PATH] [--no-run]``.
``--no-run`` skips step 3's execution (structure and existence checks only).
Exit 0 when every check passes, 1 otherwise; each problem is printed on its own line.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO_ROOT / "docs" / "features.md"
STATUSES = frozenset({"shipped", "roadmap"})
_ROW_RE = re.compile(r"^\|\s*(C\d+)\s*\|")
_TICKED_RE = re.compile(r"`([^`]+)`")
_CLAIM_ID_RE = re.compile(r"\bC\d+\b")
_BULLET_RE = re.compile(r"^\s*[-*+]\s")


@dataclass
class Claim:
    claim_id: str
    status: str
    tests: list[str] = field(default_factory=list)

    @property
    def pytest_ids(self) -> list[str]:
        return [t for t in self.tests if t.startswith("tests/")]

    @property
    def web_tests(self) -> list[str]:
        return [t for t in self.tests if t.startswith("web/")]


def parse_claims(text: str) -> tuple[list[Claim], list[str]]:
    """Parse the claims table. Returns ``(claims, problems)``."""
    claims: list[Claim] = []
    problems: list[str] = []
    seen: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not _ROW_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            problems.append(f"CLAIMS.md:{lineno}: expected 5 columns, found {len(cells)}")
            continue
        claim_id, _claim, _impl, tests_cell, status = cells
        if claim_id in seen:
            problems.append(f"CLAIMS.md:{lineno}: duplicate claim id {claim_id}")
        seen.add(claim_id)
        if status not in STATUSES:
            problems.append(f"CLAIMS.md:{lineno}: {claim_id} has status {status!r}")
        tests = _TICKED_RE.findall(tests_cell)
        if status == "shipped" and not tests:
            problems.append(f"CLAIMS.md:{lineno}: shipped claim {claim_id} lists no test")
        claims.append(Claim(claim_id, status, tests))
    if not claims:
        problems.append("CLAIMS.md: no claims table rows found")
    return claims, problems


def readme_section(text: str, heading: str) -> list[str]:
    """Lines of the ``## <heading>`` section, up to the next ``## `` heading."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == f"## {heading}")
    except StopIteration:
        return []
    out: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        out.append(line)
    return out


def check_readme(readme: str, claims: list[Claim], name: str = "docs/features.md") -> list[str]:
    status = {c.claim_id: c.status for c in claims}
    problems: list[str] = []
    cited: dict[str, set[str]] = {"Features": set(), "Roadmap": set()}
    for section, expected in (("Features", "shipped"), ("Roadmap", "roadmap")):
        lines = readme_section(readme, section)
        if not lines:
            problems.append(f"{name}: missing '## {section}' section")
            continue
        for line in lines:
            if not _BULLET_RE.match(line):
                continue
            ids = _CLAIM_ID_RE.findall(line)
            if not ids:
                problems.append(f"{name} {section}: bullet without a claim id: {line.strip()}")
            for claim_id in ids:
                cited[section].add(claim_id)
                if claim_id not in status:
                    problems.append(f"{name} {section}: unknown claim {claim_id}")
                elif status[claim_id] != expected:
                    problems.append(
                        f"{name} {section}: {claim_id} is {status[claim_id]}, not {expected}"
                    )
    for claim_id, claim_status in status.items():
        section = "Features" if claim_status == "shipped" else "Roadmap"
        if claim_id not in cited[section]:
            problems.append(f"{name} {section}: {claim_status} claim {claim_id} not listed")
    return problems


def _junit_key(nodeid: str) -> tuple[str, str]:
    """Map a pytest node id to the (classname, name) JUnit uses, without params."""
    path, _, rest = nodeid.partition("::")
    module = path.removesuffix(".py").replace("/", ".")
    parts = rest.split("::") if rest else []
    name = parts[-1] if parts else ""
    classname = ".".join([module, *parts[:-1]])
    return classname, name.split("[", 1)[0]


def run_pytest(nodeids: list[str], root: Path) -> tuple[dict[str, str], str]:
    """Run ``nodeids`` once; return ``({nodeid: outcome}, output)``.

    Outcome is ``passed``, ``failed``, ``skipped`` or ``missing``. A node id that is
    a whole file or class aggregates its cases: any failure wins, then any skip.
    """
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                "-q",
                f"--junitxml={report}",
                *nodeids,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        cases: list[tuple[str, str, str]] = []
        if report.is_file():
            for case in ET.parse(report).iter("testcase"):
                outcome = "passed"
                if case.find("failure") is not None or case.find("error") is not None:
                    outcome = "failed"
                elif case.find("skipped") is not None:
                    outcome = "skipped"
                name = case.get("name", "").split("[", 1)[0]
                cases.append((case.get("classname", ""), name, outcome))

    outcomes: dict[str, str] = {}
    for nodeid in nodeids:
        classname, name = _junit_key(nodeid)
        matched = [
            o
            for c, n, o in cases
            if (n == name and c == classname)
            or (not name and (c == classname or c.startswith(classname + ".")))
        ]
        if not matched:
            outcomes[nodeid] = "missing"
        elif "failed" in matched:
            outcomes[nodeid] = "failed"
        elif "skipped" in matched:
            outcomes[nodeid] = "skipped"
        else:
            outcomes[nodeid] = "passed"
    return outcomes, proc.stdout + proc.stderr


def _defined(path: Path, parts: list[str]) -> bool:
    """Whether ``path`` defines the class/function chain ``parts``, read with ``ast``:
    no import, so it needs neither Postgres nor the models the test itself needs. A
    test method a class inherits from a base class reads as not defined."""
    body: list[ast.stmt] = ast.parse(path.read_text(encoding="utf-8")).body
    for i, name in enumerate(parts):
        node = next(
            (
                n
                for n in body
                if isinstance(n, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
                and n.name == name.split("[", 1)[0]
            ),
            None,
        )
        if node is None:
            return False
        if i < len(parts) - 1:
            if not isinstance(node, ast.ClassDef):
                return False
            body = node.body
    return True


def check_tests(claims: list[Claim], root: Path, *, run: bool) -> list[str]:
    problems: list[str] = []
    for claim in claims:
        for test in claim.tests:
            file, _, rest = test.partition("::")
            path = root / file
            if not path.is_file():
                problems.append(f"{claim.claim_id}: test file not found: {test}")
            elif rest and file.endswith(".py") and not _defined(path, rest.split("::")):
                # pytest aborts the whole run on one unknown node id: catch it here,
                # where no database is needed.
                problems.append(f"{claim.claim_id}: test not defined in its file: {test}")
    if problems or not run:
        return problems

    nodeids = sorted({t for c in claims for t in c.pytest_ids})
    if not nodeids:
        return problems
    outcomes, output = run_pytest(nodeids, root)
    for claim in claims:
        for test in claim.pytest_ids:
            outcome = outcomes[test]
            if outcome != "passed":
                problems.append(f"{claim.claim_id} ({claim.status}): {test} {outcome}")
    if problems:
        problems.append("pytest output (tail):\n" + "\n".join(output.splitlines()[-30:]))
    return problems


def _display(path: Path) -> str:
    resolved = path.resolve()
    return (
        resolved.relative_to(REPO_ROOT).as_posix()
        if resolved.is_relative_to(REPO_ROOT)
        else str(path)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--claims", type=Path, default=REPO_ROOT / "docs" / "CLAIMS.md")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--no-run", action="store_true", help="Do not execute the tests.")
    args = parser.parse_args(argv)

    claims, problems = parse_claims(args.claims.read_text(encoding="utf-8"))
    problems += check_readme(
        args.features.read_text(encoding="utf-8"), claims, _display(args.features)
    )
    problems += check_tests(claims, args.root, run=not args.no_run)

    for problem in problems:
        print(problem)
    shipped = sum(c.status == "shipped" for c in claims)
    print(
        f"{len(claims)} claims ({shipped} shipped, {len(claims) - shipped} roadmap), "
        f"{len(problems)} problem(s)",
        file=sys.stderr,
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
