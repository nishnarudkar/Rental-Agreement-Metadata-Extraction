"""Turn a rental agreement (DOCX or scanned PNG) into plain text.

DOCX files are read straight from their XML in document order, including
tables and text boxes. python-docx's ``Document.paragraphs`` skips text boxes,
and several agreements in this dataset keep real content (dates, names) inside
them. PNG scans are sent to OCR (see ``src.ocr``).

Anything the reader sees but cannot turn into text (embedded images, header
and footer text, footnotes) is reported in ``warnings`` rather than silently
dropped. See ``docs/docx_reader.md`` for the limitations.

This module only converts documents to text; it extracts no metadata.
"""

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import docx
from docx.oxml.ns import qn
from lxml import etree

MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
VML_IMAGEDATA = "{urn:schemas-microsoft-com:vml}imagedata"

# Below this many non-space characters the document is probably an image of
# text (e.g. a scan pasted into Word) rather than real text.
MIN_TEXT_CHARS = 100


class DocumentReadError(ValueError):
    """A document could not be read; ``reason`` says why in plain words."""

    def __init__(self, path: Path, reason: str):
        super().__init__(f"Cannot read {path.name}: {reason}")
        self.path = path
        self.reason = reason


@dataclass
class ExtractedText:
    filename: str
    document_type: str  # "docx" or "png"
    text: str
    warnings: list[str] = field(default_factory=list)
    mean_confidence: float | None = None  # OCR only
    diagnostics: dict | None = None  # OCR only: per-line text, confidence, boxes


def _in_fallback(el) -> bool:
    """True for the legacy duplicate copy Word stores for old readers."""
    return any(a.tag == MC_FALLBACK for a in el.iterancestors())


def _run_text(paragraph) -> str:
    """Text that belongs to this paragraph itself (not to nested text boxes)."""
    parts = []
    for el in paragraph.iter(qn("w:t"), qn("w:tab"), qn("w:br"), qn("w:cr")):
        # Skip runs that live inside a text box nested in this paragraph.
        owner = next(el.iterancestors(qn("w:p")), None)
        if owner is not paragraph:
            continue
        if el.tag == qn("w:t"):
            parts.append(el.text or "")
        elif el.tag == qn("w:tab"):
            parts.append(" ")
        else:
            parts.append("\n")
    return "".join(parts).strip()


def _text_boxes(paragraph) -> list:
    """Text-box contents anchored directly in this paragraph (fallback copies skipped)."""
    boxes = []
    for box in paragraph.iter(qn("w:txbxContent")):
        if _in_fallback(box):
            continue
        if next(box.iterancestors(qn("w:p")), None) is paragraph:
            boxes.append(box)
    return boxes


def _block_lines(container) -> list[str]:
    """Lines for every paragraph/table directly inside ``container``."""
    lines = []
    for child in container.iterchildren():
        if child.tag == qn("w:p"):
            text = _run_text(child)
            if text:
                lines.append(text)
            for box in _text_boxes(child):
                lines.extend(_block_lines(box))
        elif child.tag == qn("w:tbl"):
            for row in child.iterchildren(qn("w:tr")):
                cells = [" ".join(_block_lines(cell)) for cell in row.iterchildren(qn("w:tc"))]
                cells = [c for c in cells if c]
                if cells:
                    lines.append(" | ".join(cells))
        elif child.tag == qn("w:sdt"):  # content controls wrap ordinary blocks
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                lines.extend(_block_lines(content))
    return lines


def _count_images(body) -> int:
    """Pictures/drawings in the body that are not text boxes."""
    count = 0
    for el in body.iter(qn("w:drawing"), qn("w:pict")):
        if _in_fallback(el) or next(el.iter(qn("w:txbxContent")), None) is not None:
            continue
        if next(el.iter(qn("a:blip"), VML_IMAGEDATA), None) is not None:
            count += 1
    return count


def _parts_with_text(document, prefixes: tuple[str, ...]) -> list[str]:
    """Names of package parts (e.g. headers) under ``prefixes`` that contain text."""
    names = []
    for part in document.part.package.iter_parts():
        name = str(part.partname)
        if not name.startswith(prefixes):
            continue
        root = etree.fromstring(part.blob)
        if any((t.text or "").strip() for t in root.iter(qn("w:t"))):
            names.append(name.rsplit("/", 1)[-1])
    return sorted(names)


def _docx_warnings(document, text: str) -> list[str]:
    body = document.element.body
    warnings = []
    if not text.strip():
        warnings.append("no text found in the document body")
    elif len("".join(text.split())) < MIN_TEXT_CHARS:
        warnings.append(f"very little text found ({len(text.strip())} chars); content may be in images")
    images = _count_images(body)
    if images:
        warnings.append(f"{images} embedded image(s): any text inside them is not extracted")
    boxes = sum(1 for b in body.iter(qn("w:txbxContent")) if not _in_fallback(b))
    if boxes:
        warnings.append(f"{boxes} text box(es) read; their position in the reading order is approximate")
    header_footer = _parts_with_text(document, ("/word/header", "/word/footer"))
    if header_footer:
        warnings.append(f"header/footer text not included: {', '.join(header_footer)}")
    notes = _parts_with_text(document, ("/word/footnotes", "/word/endnotes", "/word/comments"))
    if notes:
        warnings.append(f"footnote/endnote/comment text not included: {', '.join(notes)}")
    return warnings


def _open_docx(path: Path):
    """Open a DOCX, turning library errors into a DocumentReadError with a clear reason."""
    if not path.exists():
        raise DocumentReadError(path, "file not found")
    if path.stat().st_size == 0:
        raise DocumentReadError(path, "file is empty (0 bytes)")
    if not zipfile.is_zipfile(path):
        raise DocumentReadError(
            path, "not a DOCX (ZIP) package; it may be a legacy .doc, a password-protected file, or corrupted"
        )
    try:
        return docx.Document(path)
    except Exception as exc:  # python-docx raises KeyError, ValueError, lxml errors, ...
        raise DocumentReadError(path, f"malformed DOCX package ({type(exc).__name__}: {exc})") from exc


def read_docx(path: Path) -> ExtractedText:
    document = _open_docx(path)
    text = "\n".join(_block_lines(document.element.body))
    return ExtractedText(
        filename=path.name,
        document_type="docx",
        text=text,
        warnings=_docx_warnings(document, text),
    )


def read_document(path: Path) -> ExtractedText:
    """Dispatch on file type. OCR is imported lazily because it is slow to load."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".png":
        from src.ocr import ocr_image

        ocr = ocr_image(path)
        return ExtractedText(
            filename=path.name,
            document_type="png",
            text=ocr.text,
            warnings=ocr.warnings,
            mean_confidence=ocr.mean_confidence,
            diagnostics=ocr.diagnostics(),
        )
    raise DocumentReadError(path, f"unsupported file type '{suffix}' (expected .docx or .png)")
