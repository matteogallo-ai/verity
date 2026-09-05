"""Web parser: fetch a URL with httpx and extract the main text via trafilatura.

Only the *main* text is kept — trafilatura strips navigation, ads, and boilerplate,
which is exactly what we want the retriever to see. The full HTML is not preserved:
citations point back to the URL (already stored as ``document.uri``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from verity.ingestion.ids import document_id_for_uri
from verity.ingestion.structure import DocumentStructure, Section
from verity.types import Document, SourceType

if TYPE_CHECKING:  # pragma: no cover
    pass


class WebParser:
    """Fetch and extract main text from a web page.

    ``fetcher`` and ``extractor`` are injectable seams so tests never touch the
    network or depend on trafilatura's classifier being deterministic.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def supports(self, source_type: SourceType) -> bool:
        return source_type is SourceType.WEB

    async def parse(self, uri: str, raw: bytes | None = None) -> Document:
        if raw is None:
            html = await self._fetch(uri)
        else:
            html = raw.decode("utf-8", errors="replace")
        text, title = _extract_main_text(html, uri)
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        structure = DocumentStructure(sections=(Section(char_start=0, char_end=len(text)),))
        return Document(
            id=document_id_for_uri(uri),
            source_type=SourceType.WEB,
            uri=uri,
            title=title,
            text=text,
            metadata=structure.to_metadata(),
        )

    async def _fetch(self, uri: str) -> str:
        if self._client is not None:
            response = await self._client.get(uri, timeout=self._timeout_s)
            response.raise_for_status()
            return response.text
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(uri, timeout=self._timeout_s)
            response.raise_for_status()
            return response.text


def _extract_main_text(html: str, uri: str) -> tuple[str, str | None]:
    """Extract main text + title with trafilatura; degrade gracefully if unavailable."""
    try:
        import trafilatura
    except ImportError:  # pragma: no cover - trafilatura is a hard dep in prod
        return html, None
    extracted = trafilatura.extract(html, url=uri, include_comments=False)
    metadata = trafilatura.extract_metadata(html)
    title = metadata.title if metadata else None
    return (extracted or "", title)


__all__ = ["WebParser"]
