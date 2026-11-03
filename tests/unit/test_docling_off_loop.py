"""PDF/DOCX conversion is CPU-bound model inference: it must not run on the event loop,
or one upload stalls every other request (``/ask``, ``/health``, ``/ready``)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from verity.ingestion.parsers.docling_parser import DoclingParser


class _SlowConverter:
    def convert(self, _path: str) -> SimpleNamespace:
        time.sleep(0.5)
        return SimpleNamespace(document=SimpleNamespace(iterate_items=lambda: iter(())))


@pytest.mark.asyncio
async def test_other_tasks_run_while_a_document_converts() -> None:
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    try:
        await DoclingParser(converter=_SlowConverter()).parse("upload://x.pdf", raw=b"%PDF")  # type: ignore[arg-type]
    finally:
        task.cancel()
    # A blocked loop gives the ticker no turn during the 0.5 s conversion.
    assert ticks >= 5
