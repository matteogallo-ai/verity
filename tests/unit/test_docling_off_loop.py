"""PDF/DOCX conversion is CPU-bound model inference: it must not run on the event loop,
or one upload stalls every other request (``/ask``, ``/health``, ``/ready``)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from verity.ingestion.parsers.docling_parser import DoclingParser


class _Converter:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def convert(self, _path: str) -> SimpleNamespace:
        time.sleep(self._seconds)
        return SimpleNamespace(document=SimpleNamespace(iterate_items=lambda: iter(())))


def _slow_factory() -> _Converter:
    """Half a second of model loading, then an instant conversion."""
    time.sleep(0.5)
    return _Converter(0.0)


async def _ticks_during(parser: DoclingParser) -> int:
    """How often another task got a turn while ``parser`` parsed one document."""
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    try:
        await parser.parse("upload://x.pdf", raw=b"%PDF")
    finally:
        task.cancel()
    return ticks


@pytest.mark.asyncio
async def test_other_tasks_run_while_a_document_converts() -> None:
    # A blocked loop gives the ticker no turn during the 0.5 s conversion.
    assert await _ticks_during(DoclingParser(converter=_Converter(0.5))) >= 5  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_other_tasks_run_while_the_converter_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first PDF builds the converter, which loads docling's models."""
    monkeypatch.setattr("verity.ingestion.parsers.docling_parser._lazy_converter", _slow_factory)
    assert await _ticks_during(DoclingParser()) >= 5
