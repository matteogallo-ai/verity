"""Drive the web UI through the demo and record it, then cut the GIF and the MP4.

Run by scripts/demo/record_demo.sh once the stack is up; not on its own. The
questions are the UI's demo chips, which are rows of datasets/eval/questions.jsonl.
Writes into --out: verity-demo.mp4, verity-demo.gif and demo-run.json (provenance,
per-answer cost and latency, the edit's segments). Fails before encoding when the
UI's provenance is not the expected mode or model, or the session's spend, as the
API's /metrics reports it, exceeds the cap. The cap itself is enforced by the API's
spending budget (compose.demo.yml); this script only checks the total afterwards.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, Response, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from media import (
    AGENT_MODEL,
    CAP_USD,
    GIF_TAIL,
    MP4_TAIL,
    SERVER_BUDGET_USD,
    WAIT_SPEED_GIF,
    WAIT_SPEED_MP4,
    Wait,
    check_spend,
    duration,
    filter_graph,
    segments,
)

# Chosen from the live runs of record: both answerable questions were answered in each
# run that holds them, the out-of-corpus one refused. q-001 (the notice period) is not
# used: at commit e644a98 it is refused by the irrelevant-citation check.
ANSWERABLE = "What is the response time for a Priority 1 support request?"
ANSWERABLE_2 = "Where is personal data stored, and what mechanism covers EU-to-US transfers?"
OUT_OF_CORPUS = "What is the CEO's home address?"
# Settings.confidence_threshold, which the demo stack does not override; a test keeps
# the two equal so the caption cannot go stale.
CONFIDENCE_THRESHOLD = 0.0
# SupportCheck.rationale for RefusalReason.SUSPECTED_INJECTION, pinned by a test: the
# injection caption is shown only when the refusal came from that check.
INJECTION_RATIONALE = "A cited document contains text that tries to instruct the model."
VIEWPORT = {"width": 1280, "height": 800}
# The live answers of record took up to ~20 s; past this the stack is broken.
ANSWER_TIMEOUT_MS = 90_000

# A caption bar the recorder drives; it is part of the page, so part of the video.
CAPTION_JS = """
(() => {
  const install = () => {
    if (document.getElementById('demo-caption')) return;
    const bar = document.createElement('div');
    bar.id = 'demo-caption';
    bar.style.cssText = 'position:fixed;left:0;right:0;bottom:0;padding:14px 28px;'
      + 'background:rgba(10,12,16,.94);border-top:2px solid #f5c542;color:#f4f4f5;'
      + 'font:500 21px/1.35 system-ui,sans-serif;z-index:2147483647;text-align:center;'
      + 'pointer-events:none;display:none';
    document.body.appendChild(bar);
  };
  window.__caption = (text) => {
    install();
    const bar = document.getElementById('demo-caption');
    bar.textContent = text || '';
    bar.style.display = text ? 'block' : 'none';
  };
  document.addEventListener('DOMContentLoaded', install);
})();
"""


class Recorder:
    def __init__(self, page: Page, api: str) -> None:
        self.page = page
        self.api = api
        self.t0 = time.monotonic()
        self.waits: list[Wait] = []
        self.answers: list[dict[str, Any]] = []
        self.marks: dict[str, float] = {}
        self.errors: list[str] = []
        page.on("response", self._on_response)

    def now(self) -> float:
        return time.monotonic() - self.t0

    def mark(self, name: str) -> None:
        self.marks[name] = self.now()

    def _on_response(self, response: Response) -> None:
        if response.url == f"{self.api}/ask" and response.request.method == "POST":
            try:
                body = response.json()
                if response.status != 200:
                    self.errors.append(f"/ask returned {response.status}: {body}")
                    return
                answer, usage = body["answer"], body["answer"]["usage"]
                self.answers.append(
                    {
                        "question": response.request.post_data_json["question"],
                        "refused": answer["confidence"]["refused"],
                        "n_citations": len(answer["citations"]),
                        "rationale": answer["confidence"]["rationale"],
                        "cost_usd": usage["cost_usd"],
                        "latency_ms": usage["latency_ms"],
                        "trace_id": answer["trace_id"],
                        "agent_model": body["provenance"]["agent_model"],
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                # Raised in an event callback, it would never reach the recording
                # loop; ask() re-raises it instead.
                self.errors.append(f"unreadable /ask response: {exc!r}")

    def caption(self, text: str, hold_s: float = 0.0) -> None:
        self.page.evaluate("t => window.__caption(t)", text)
        self.hold(hold_s)

    def hold(self, seconds: float) -> None:
        if seconds > 0:
            self.page.wait_for_timeout(seconds * 1000)

    def ask(self, question: str, *, expect: str) -> None:
        """Type ``question``, send it, and log the wait until ``expect`` renders."""
        box = self.page.get_by_label("Question")
        box.fill("")
        box.press_sequentially(question, delay=28)
        self.hold(0.6)
        self.page.get_by_role("button", name="Ask").click()
        start = self.now()
        self.caption(
            "Model working… (waits are sped up here; the real latency is in the answer footer)"
        )
        # The previous answer stays on screen until the new one replaces it, so wait for
        # the /ask response, then for the UI to leave its pending state.
        n, deadline = len(self.answers), time.monotonic() + ANSWER_TIMEOUT_MS / 1000
        while len(self.answers) == n and not self.errors:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{question!r}: no /ask response")
            self.page.wait_for_timeout(100)
        if self.errors:
            raise RuntimeError(self.errors[-1])
        self.page.locator("p.pending").wait_for(state="hidden", timeout=10_000)
        if not self.page.locator(expect).first.is_visible():
            raise RuntimeError(f"{question!r}: the UI did not render {expect}")
        self.waits.append(Wait(start, self.now()))
        self.caption("")


def _get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return dict(json.load(resp))


def session_cost(api: str) -> float:
    """The API's own sum of every answer's cost, failed answers included."""
    return float(_get(f"{api}/metrics")["total_cost_usd"])


def run_demo(rec: Recorder, web: str, api: str) -> None:
    page = rec.page
    page.goto(web)
    page.get_by_text("corpus ·").wait_for(timeout=60_000)
    rec.hold(1.0)
    rec.caption(
        "Verity answers questions over a document corpus with verbatim citations, "
        "and refuses when its evidence is weak.",
        6.0,
    )
    rec.caption(
        "The demo corpus: 21 fictional enterprise documents (contracts, policies, "
        "reports), one of them adversarial.",
        5.0,
    )

    rec.mark("gif_a_start")
    rec.caption("An answerable question.", 1.2)
    rec.ask(ANSWERABLE, expect="article.answer")
    rec.caption("Each [n] is a quote copied verbatim from a source chunk; others are dropped.", 2.5)
    page.get_by_test_id("citation-1").click()
    rec.caption("Click a citation: its source chunk is highlighted on the right.", 2.5)
    rec.mark("gif_a_end")
    rec.hold(2.5)
    if page.get_by_test_id("citation-2").count():
        page.get_by_test_id("citation-2").click()
        rec.caption("A second citation.", 4.0)
    rec.caption(
        "Sources: the chunks the agent read, flagged cited or not, with their reranker scores.",
        6.0,
    )
    page.locator(".readout").first.scroll_into_view_if_needed()
    rec.caption(
        f"Confidence: the scorer model's self-report. At the default threshold of "
        f"{CONFIDENCE_THRESHOLD} it refuses nothing; the checks and the scorer's own "
        "refusal do.",
        6.0,
    )
    page.locator("footer.usage").scroll_into_view_if_needed()
    rec.caption("Per answer: latency, LLM calls, tokens, cost and trace id.", 6.0)

    page.mouse.wheel(0, -2000)
    rec.caption("A question with two parts.", 1.5)
    rec.ask(ANSWERABLE_2, expect="article.answer")
    n = rec.answers[-1]["n_citations"]
    rec.caption(f"Answered, citing {n} passage{'s' if n > 1 else ''}.", 5.0)
    page.get_by_test_id("citation-1").click()
    rec.hold(3.0)
    if page.get_by_test_id("citation-2").count():
        page.get_by_test_id("citation-2").click()
        rec.hold(3.0)

    page.mouse.wheel(0, -2000)
    rec.mark("gif_b_start")
    rec.caption("A question the corpus cannot answer.", 1.2)
    rec.ask(OUT_OF_CORPUS, expect="article.refusal")
    rec.caption("Refused: “I don't know” beats a confident guess.", 3.0)
    rec.mark("gif_b_end")
    rec.hold(1.5)
    rec.caption("The refusal shows the score, the reason and the evidence it refused against.", 6.0)
    # Claimed only when the deterministic injection check is what refused: other
    # refusals of this question never read the adversarial document's instructions.
    if rec.answers[-1]["rationale"] == INJECTION_RATIONALE:
        rec.caption(
            "Here the injection check refused first: a retrieved document (the adversarial "
            "one) contains instructions aimed at the model.",
            6.0,
        )

    page.goto(f"{api}/dashboard")
    page.wait_for_load_state("load")
    rec.caption("Traces: every stage of every answer is recorded as an OpenTelemetry span.", 6.0)
    page.get_by_text("Per-stage latency").scroll_into_view_if_needed()
    rec.caption("Per-stage latency of this session: decompose, retrieve, synthesize, score.", 7.0)
    page.goto(web)
    page.get_by_text("corpus ·").wait_for(timeout=30_000)
    rec.caption(
        "Published numbers come from committed live evaluation runs: docs/results.md · "
        "github.com/matteogallo-ai/verity",
        6.0,
    )


def ffmpeg(src: Path, dst: Path, graph: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
            "-filter_complex", graph, "-map", "[out]",
            *(["-c:v", "libx264", "-crf", "26", "-preset", "slow", "-movflags", "+faststart"]
              if dst.suffix == ".mp4" else []),
            str(dst),
        ],
        check=True,
    )  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--web", default="http://localhost:3000")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--expect", choices=["live", "stub"], required=True)
    args = parser.parse_args()

    prov = _get(f"{args.api}/status")["provenance"]
    if prov["is_stub"] != (args.expect == "stub"):
        raise SystemExit(f"expected a {args.expect} agent, the API reports {prov}")
    if args.expect == "live" and prov["agent_model"] != AGENT_MODEL:
        raise SystemExit(f"the spend bound is priced for {AGENT_MODEL}, not {prov}")
    args.out.mkdir(parents=True, exist_ok=True)
    # Outside --out, so a failed take leaves nothing next to the published media.
    raw_dir = Path(tempfile.mkdtemp(prefix="verity-demo-raw-"))

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport=VIEWPORT, record_video_dir=str(raw_dir), record_video_size=VIEWPORT
        )
        context.add_init_script(CAPTION_JS)
        page = context.new_page()
        rec = Recorder(page, args.api)
        try:
            run_demo(rec, args.web, args.api)
            end = rec.now()
        finally:
            # A failed take has paid for its answers too.
            print(f"spend: ${session_cost(args.api):.6f}", file=sys.stderr)
            video = page.video
            context.close()
            browser.close()
        assert video is not None
        raw = Path(video.path())

    total = check_spend([session_cost(args.api)])
    mp4_parts = segments((0.0, end), rec.waits, WAIT_SPEED_MP4)
    gif_parts = [
        *segments((rec.marks["gif_a_start"], rec.marks["gif_a_end"]), rec.waits, WAIT_SPEED_GIF),
        *segments((rec.marks["gif_b_start"], rec.marks["gif_b_end"]), rec.waits, WAIT_SPEED_GIF),
    ]
    ffmpeg(raw, args.out / "verity-demo.mp4", filter_graph(mp4_parts, MP4_TAIL))
    ffmpeg(raw, args.out / "verity-demo.gif", filter_graph(gif_parts, GIF_TAIL))
    shutil.rmtree(raw_dir)

    record = {
        "provenance": prov,
        "cap_usd": CAP_USD,
        "server_budget_usd": SERVER_BUDGET_USD,
        "total_cost_usd": round(total, 6),
        "answers": rec.answers,
        "mp4_seconds": round(duration(mp4_parts), 1),
        "gif_seconds": round(duration(gif_parts), 1),
        "wait_speed": {"mp4": WAIT_SPEED_MP4, "gif": WAIT_SPEED_GIF},
    }
    (args.out / "demo-run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("total_cost_usd", "mp4_seconds", "gif_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
