"""Turn a rental agreement (DOCX or scanned PNG) into plain text.

DOCX files are read straight from their XML in document order, including
tables and text boxes. python-docx's ``Document.paragraphs`` skips text boxes,
and several agreements in this dataset keep real content (dates, names) inside
them. PNG scans are sent to OCR (see ``src.ocr``).

This module only converts documents to text; it extracts no metadata.
"""

from dataclasses import dataclass
from pathlib import Path

import docx
from docx.oxml.ns import qn

MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


@dataclass
class ExtractedText:
    text: str
    method: str  # "docx" or "ocr"
    mean_confidence: float | None = None  # OCR only


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
    """Text-box contents anchored directly in this paragraph.

    Word often stores each text box twice (a modern version plus an
    ``mc:Fallback`` copy for old readers); the fallback copy is skipped.
    """
    boxes = []
    for box in paragraph.iter(qn("w:txbxContent")):
        ancestors = list(box.iterancestors())
        if any(a.tag == MC_FALLBACK for a in ancestors):
            continue
        if next((a for a in ancestors if a.tag == qn("w:p")), None) is paragraph:
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


def read_docx(path: Path) -> ExtractedText:
    document = docx.Document(path)
    lines = _block_lines(document.element.body)
    return ExtractedText(text="\n".join(lines), method="docx")


def read_document(path: Path) -> ExtractedText:
    """Dispatch on file type. OCR is imported lazily because it is slow to load."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".png":
        from src.ocr import ocr_image

        text, confidence = ocr_image(path)
        return ExtractedText(text=text, method="ocr", mean_confidence=confidence)
    raise ValueError(f"Unsupported file type: {path.name}")
