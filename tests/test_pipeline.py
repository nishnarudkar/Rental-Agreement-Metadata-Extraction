"""Unified ingestion: read_document routing and the extract_folder pipeline."""

import hashlib
import json
from pathlib import Path

import docx
import pytest
from PIL import Image

from src import ocr
from src.document_reader import DocumentReadError, read_document
from src.extract_text import extract_folder


def _docx(path: Path, *paragraphs: str) -> Path:
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    document.save(path)
    return path


def _png(path: Path) -> Path:
    Image.new("L", (200, 100), 255).save(path)
    return path


@pytest.fixture
def fake_ocr(monkeypatch):
    monkeypatch.setattr(ocr, "_ocr_tile", lambda image: [ocr.OcrLine("scanned text", 0.99, [0, 0, 80, 20])])


# --- read_document ---------------------------------------------------------------


def test_docx_routing_and_result_fields(tmp_path):
    result = read_document(_docx(tmp_path / "123-Rental-Agreement.pdf.docx", "Body of the agreement."))
    assert result.document_id == "123-Rental-Agreement"  # double extension stripped
    assert result.filename == "123-Rental-Agreement.pdf.docx"
    assert result.document_type == "docx"
    assert result.text == "Body of the agreement."
    assert isinstance(result.warnings, list)
    assert result.seconds >= 0


def test_png_routing(tmp_path, fake_ocr):
    result = read_document(_png(tmp_path / "456-Scan.png"))
    assert (result.document_id, result.document_type, result.text) == ("456-Scan", "png", "scanned text")
    assert result.diagnostics is not None


def test_extension_matching_is_case_insensitive(tmp_path):
    assert read_document(_docx(tmp_path / "UPPER.DOCX", "x")).document_type == "docx"


@pytest.mark.parametrize("name", ["agreement.pdf", "agreement.doc", "scan.jpg", "notes"])
def test_unsupported_formats(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"data")
    with pytest.raises(DocumentReadError, match="unsupported file type"):
        read_document(path)


@pytest.mark.parametrize("name", ["missing.docx", "missing.png"])
def test_missing_files(tmp_path, name):
    with pytest.raises(DocumentReadError, match="file not found"):
        read_document(tmp_path / name)


def test_empty_text_is_returned_with_warning(tmp_path):
    result = read_document(_docx(tmp_path / "blank.docx"))
    assert result.text == ""
    assert "no text found in the document body" in result.warnings


# --- extract_folder ---------------------------------------------------------------


@pytest.fixture
def mixed_folder(tmp_path, fake_ocr):
    folder = tmp_path / "docs"
    folder.mkdir()
    _docx(folder / "good.docx", "A rental agreement between two parties. " * 4)
    _docx(folder / "blank.docx")
    _png(folder / "scan.png")
    (folder / "broken.docx").write_bytes(b"not a zip")
    (folder / "notes.txt").write_text("stray file")
    (folder / ".hidden").write_text("ignored")
    return folder


def test_failures_do_not_stop_the_run_and_are_counted(mixed_folder, tmp_path):
    out = tmp_path / "out"
    summary = extract_folder(mixed_folder, out)
    assert summary["files"] == 5  # hidden file ignored
    assert (summary["succeeded"], summary["empty_text"], summary["failed"], summary["unsupported"]) == (3, 1, 1, 1)
    assert summary["by_type"] == {"docx": 3, "png": 1}
    assert set(summary["failed_documents"]) == {"broken", "notes.txt"}  # unsupported keep their full name
    assert "not a DOCX" in summary["failed_documents"]["broken"]

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert {d: e["status"] for d, e in manifest.items()} == {
        "good": "ok", "blank": "empty", "scan": "ok", "broken": "failed", "notes.txt": "unsupported",
    }
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary


def test_outputs_use_deterministic_names(mixed_folder, tmp_path):
    out = tmp_path / "out"
    extract_folder(mixed_folder, out)
    names = sorted(p.name for p in out.iterdir())
    assert names == ["blank.txt", "good.txt", "manifest.json", "scan.ocr.json", "scan.txt", "summary.json"]


def test_failed_document_leaves_no_stale_text(mixed_folder, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "broken.txt").write_text("text from an earlier, successful run")
    extract_folder(mixed_folder, out, force=True)
    assert not (out / "broken.txt").exists()


def test_cache_reuses_successes_and_retries_failures(mixed_folder, tmp_path):
    out = tmp_path / "out"
    extract_folder(mixed_folder, out)
    again = extract_folder(mixed_folder, out)
    assert (again["cached"], again["failed"], again["processed"]) == (3, 1, 1)
    forced = extract_folder(mixed_folder, out, force=True)
    assert (forced["cached"], forced["processed"]) == (0, 4)


def test_explicit_file_list_limits_the_run(mixed_folder, tmp_path):
    summary = extract_folder(mixed_folder, tmp_path / "out", files=[mixed_folder / "good.docx"])
    assert (summary["files"], summary["succeeded"]) == (1, 1)
    assert sorted(p.name for p in (tmp_path / "out").glob("*.txt")) == ["good.txt"]


def test_source_documents_are_not_modified(mixed_folder, tmp_path):
    def digests():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in mixed_folder.iterdir()}

    before = digests()
    extract_folder(mixed_folder, tmp_path / "out", force=True)
    assert digests() == before
