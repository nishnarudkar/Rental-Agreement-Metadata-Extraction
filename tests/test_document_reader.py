import io
import zipfile
from pathlib import Path

import docx
import pytest
from docx.oxml import parse_xml
from PIL import Image

from src.config import TRAIN_DIR
from src.document_reader import DocumentReadError, read_docx, read_document
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
    with pytest.raises(DocumentReadError, match="unsupported file type '.pdf'"):
        read_document(tmp_path / "agreement.pdf")


# --- structured result and warnings ------------------------------------------


def test_result_has_filename_type_text_and_warnings(sample_docx):
    result = read_docx(sample_docx)
    assert result.filename == "sample.docx"
    assert result.document_type == "docx"
    assert result.mean_confidence is None
    assert "1 text box(es) read; their position in the reading order is approximate" in result.warnings


def _save(document, tmp_path, name="doc.docx") -> Path:
    path = tmp_path / name
    document.save(path)
    return path


def test_empty_document_warns(tmp_path):
    result = read_docx(_save(docx.Document(), tmp_path))
    assert result.text == ""
    assert result.warnings == ["no text found in the document body"]


def test_empty_paragraphs_are_skipped(tmp_path):
    document = docx.Document()
    for text in ["", "   ", "Real line", ""]:
        document.add_paragraph(text)
    assert read_docx(_save(document, tmp_path)).text == "Real line"


def test_short_text_warns(tmp_path):
    document = docx.Document()
    document.add_paragraph("Signature page")
    (warning,) = read_docx(_save(document, tmp_path)).warnings
    assert warning.startswith("very little text found")


def test_embedded_image_warns(tmp_path):
    png = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(png, format="PNG")
    png.seek(0)
    document = docx.Document()
    document.add_paragraph("A rental agreement. " * 10)
    document.add_picture(png)
    assert "1 embedded image(s): any text inside them is not extracted" in read_docx(_save(document, tmp_path)).warnings


def test_header_text_is_excluded_but_reported(tmp_path):
    document = docx.Document()
    document.sections[0].header.paragraphs[0].text = "Page header"
    document.add_paragraph("Body text of the agreement. " * 5)
    result = read_docx(_save(document, tmp_path))
    assert "Page header" not in result.text
    assert any(w.startswith("header/footer text not included: header") for w in result.warnings)


# --- malformed files -----------------------------------------------------------


def test_missing_file(tmp_path):
    with pytest.raises(DocumentReadError, match="file not found"):
        read_docx(tmp_path / "nope.docx")


def test_zero_byte_file(tmp_path):
    path = tmp_path / "empty.docx"
    path.write_bytes(b"")
    with pytest.raises(DocumentReadError, match="empty"):
        read_docx(path)


def test_not_a_zip_file(tmp_path):
    path = tmp_path / "legacy.docx"
    path.write_bytes(b"\xd0\xcf\x11\xe0 legacy Word 97 header bytes")
    with pytest.raises(DocumentReadError, match="not a DOCX"):
        read_docx(path)


def test_zip_without_word_content(tmp_path):
    path = tmp_path / "broken.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("hello.txt", "not a word document")
    with pytest.raises(DocumentReadError, match="malformed DOCX package") as info:
        read_docx(path)
    assert info.value.path == path


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
