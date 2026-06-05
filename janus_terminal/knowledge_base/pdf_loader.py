"""PDF + markdown loader.

We support two inputs for the knowledge base:
  * PDF (PyMuPDF / fitz) — typical annual reports.
  * Markdown / TXT — used by tests and the synthetic sample dataset.

The loader returns plain text chunks plus metadata so downstream chunking
+ embedding can be tested without PyMuPDF installed (lazy import).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class LoadedDocument:
    source: str           # absolute path or relative identifier
    page: int             # 0-based; 0 for non-paginated formats
    text: str
    metadata: dict


def load_pdf(path: str | Path) -> list[LoadedDocument]:
    """Read a PDF and return one ``LoadedDocument`` per page."""
    import fitz  # PyMuPDF, lazy

    path = Path(path)
    out: list[LoadedDocument] = []
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text("text")
            if not text.strip():
                continue
            out.append(
                LoadedDocument(
                    source=str(path),
                    page=i,
                    text=text,
                    metadata={"file": path.name, "page": i, "format": "pdf"},
                )
            )
    return out


def load_markdown(path: str | Path) -> list[LoadedDocument]:
    """Read a markdown or txt file as a single document (page=0)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return [
        LoadedDocument(
            source=str(path),
            page=0,
            text=text,
            metadata={"file": path.name, "page": 0, "format": path.suffix.lstrip(".") or "md"},
        )
    ]


def load_any(path: str | Path) -> list[LoadedDocument]:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".pdf":
        return load_pdf(p)
    if suf in (".md", ".txt", ".markdown"):
        return load_markdown(p)
    raise ValueError(f"unsupported document format: {suf}")


def iter_directory(root: str | Path) -> Iterator[LoadedDocument]:
    """Yield every supported document under ``root`` recursively."""
    root = Path(root)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".md", ".txt", ".markdown"}:
            try:
                yield from load_any(p)
            except Exception as exc:  # noqa: BLE001 - corrupt PDFs are common
                # callers may want to log; we just skip
                continue
