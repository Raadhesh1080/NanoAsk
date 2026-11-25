"""
Document loading + chunking, adapted from askme/ingest/document_processor.py.

Supported: .txt, .md, .html/.htm, .pdf (pypdf), plus raw text. Chunking is askme's
recursive strategy (paragraph -> line -> sentence -> clause -> word separators) with
character overlap carried between consecutive chunks.
"""
from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkingConfig:
    chunk_size: int = 800        # characters
    chunk_overlap: int = 150
    min_chunk_size: int = 40


# ---------------------------------------------------------------------------- loaders
def _clean_html(raw: str) -> tuple[str, str | None]:
    title = None
    m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I)
    if m:
        title = html.unescape(m.group(1)).strip()
    raw = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<br\s*/?>|</p>|</div>|</h[1-6]>|</li>|</tr>", "\n\n", raw, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip(), title


def load_bytes(name: str, data: bytes) -> tuple[str, str]:
    """Returns (title, text) for an uploaded file."""
    suffix = Path(name).suffix.lower()
    title = Path(name).stem
    if suffix == ".pdf":
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        meta_title = reader.metadata.title if reader.metadata else None
        return (meta_title or title), "\n\n".join(pages)
    text = data.decode("utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        text, t = _clean_html(text)
        return (t or title), text
    if suffix in {".md", ".markdown"}:
        m = re.search(r"^#\s+(.+)$", text, re.M)
        if m:
            title = m.group(1).strip()
    return title, text


def load_path(path: str | Path) -> tuple[str, str]:
    p = Path(path)
    return load_bytes(p.name, p.read_bytes())


# ---------------------------------------------------------------------------- chunking
SEPARATORS = ["\n\n", "\n", "。", "! ", "? ", ". ", "；", "; ", "，", ", ", " "]


class RecursiveChunker:
    def __init__(self, config: ChunkingConfig | None = None):
        self.cfg = config or ChunkingConfig()

    def split(self, text: str) -> list[str]:
        text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
        pieces = self._split(text, 0)
        return self._add_overlap(pieces)

    def _split(self, text: str, sep_idx: int) -> list[str]:
        cs = self.cfg.chunk_size
        if len(text) <= cs:
            return [text] if text.strip() else []
        if sep_idx >= len(SEPARATORS):
            return [text[i:i + cs] for i in range(0, len(text), cs)]
        sep = SEPARATORS[sep_idx]
        out, cur = [], ""
        for part in text.split(sep):
            if not part.strip():
                continue
            cand = cur + sep + part if cur else part
            if len(cand) <= cs:
                cur = cand
                continue
            if cur:
                out.append(cur.strip())
            if len(part) > cs:
                out.extend(self._split(part, sep_idx + 1))
                cur = ""
            else:
                cur = part
        if cur.strip():
            out.append(cur.strip())
        return out

    def _add_overlap(self, pieces: list[str]) -> list[str]:
        ov = self.cfg.chunk_overlap
        out = []
        for i, p in enumerate(pieces):
            if i > 0 and ov > 0:
                tail = pieces[i - 1][-ov:]
                cut = tail.find(" ")  # start the overlap on a word boundary
                tail = tail[cut + 1:] if 0 <= cut < len(tail) - 1 else tail
                p = tail + " " + p
            if len(p) >= self.cfg.min_chunk_size or len(pieces) == 1:
                out.append(p)
        return out


def make_chunks(title: str, text: str, chunker: RecursiveChunker, source: str = "") -> list[Chunk]:
    doc_id = hashlib.sha1((title + "\0" + text).encode()).hexdigest()[:12]
    return [
        Chunk(id=f"{doc_id}-{i}", doc_id=doc_id, title=title, text=t, chunk_index=i,
              metadata={"source": source or title})
        for i, t in enumerate(chunker.split(text))
    ]
