from pathlib import Path

import docx
import pytest
from docx.oxml import parse_xml

from src.config import TRAIN_DIR
from src.document_reader import read_docx, read_document
from src.ocr import _group_lines

NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
)


def _textbox_paragraph(anchor: str, box: str) -> str:
    """A paragraph with a text box stored twice (Choice + Fallback), as Word does."""
    content = f"<w:txbxContent><w:p><w:r><w:t>{box}</w:t></w:r></w:p></w:txbxContent>"
    return (
        f"<w:p {NS}><w:r><w:t>{anchor}</w:t></w:r><w:r><mc:AlternateContent>"
        f"<mc:Choice Requires='wps'>{content}</mc:Choice>"
        f"<mc:Fallback>{content}</mc:Fallback>"
        "</mc:AlternateContent></w:r></w:p>"
    )


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    document = docx.Document()
    document.add_paragraph("First paragraph")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Rent"
    table.cell(0, 1).text = "5000"
    document.element.body.append(parse_xml(_textbox_paragraph("Anchor", "Boxed text")))
    # add_table/append put elements after sectPr; move sectPr back to the end.
    body = document.element.body
    body.append(body.find(docx.oxml.ns.qn("w:sectPr")))
    path = tmp_path / "sample.docx"
    document.save(path)
    return path


def test_docx_reads_paragraphs_tables_and_text_boxes_in_order(sample_docx):
    lines = read_docx(sample_docx).text.splitlines()
    assert lines == ["First paragraph", "Rent | 5000", "Anchor", "Boxed text"]


def test_docx_text_box_fallback_copy_is_not_duplicated(sample_docx):
    assert read_docx(sample_docx).text.count("Boxed text") == 1


def test_real_docx_includes_text_box_content():
    # This phrase sits in a text box that python-docx's .paragraphs misses.
    text = read_docx(TRAIN_DIR / "18325926-Rental-Agreement-1.docx").text
    assert "In site No 8/20 AGS building" in text


def test_unsupported_extension_raises(tmp_path):
    with pytest.raises(ValueError):
        read_document(tmp_path / "agreement.pdf")


def test_ocr_fragments_on_one_line_are_joined_left_to_right():
    fragments = [("premise except the", [500, 102, 700, 120]), ("entire 2nd floor", [100, 100, 400, 120])]
    assert _group_lines(fragments) == "entire 2nd floor premise except the"


def test_ocr_large_vertical_gap_becomes_paragraph_break():
    fragments = [
        ("line one", [0, 0, 100, 20]),
        ("line two", [0, 25, 100, 45]),
        ("after gap", [0, 300, 100, 320]),
    ]
    assert _group_lines(fragments) == "line one\nline two\n\nafter gap"


def test_ocr_empty_input():
    assert _group_lines([]) == ""
