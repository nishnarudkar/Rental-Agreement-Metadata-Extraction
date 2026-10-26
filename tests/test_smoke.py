"""Smoke tests for the command-line prediction pipeline.

The first two always run (fake OCR and LLM, no model needed). The last one runs
the real local model on a synthetic DOCX and is opt-in:

    $env:RUN_LLM_SMOKE = "1"; python -m pytest tests/test_smoke.py
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import docx
import pytest
from PIL import Image

from src import ocr
from src.config import FILE_NAME_COLUMN, LABEL_COLUMNS, PROJECT_ROOT
from src.predict import predict_folder
from tests.test_extractor import FakeLLM, _answer

VALUE, START, END, NOTICE, PARTY_ONE, PARTY_TWO = LABEL_COLUMNS

SYNTHETIC_AGREEMENT = [
    "RENTAL AGREEMENT",
    "This rental agreement is made at Pune on 3rd January 2024 between Ms. Ananya Rao, residing at Kothrud, Pune "
    "(the Owner) and Mr. Kabir Mehta, residing at Baner, Pune (the Tenant).",
    "The tenancy commences on 15th January 2024 and ends on 14th December 2024.",
    "The Tenant shall pay a monthly rent of Rs. 18,500/- (Rupees Eighteen Thousand Five Hundred only) "
    "and has paid a refundable security deposit of Rs. 1,00,000/-.",
    "Either party may terminate this agreement by giving one month's written notice.",
]


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / "agreements"
    folder.mkdir()
    document = docx.Document()
    for paragraph in SYNTHETIC_AGREEMENT:
        document.add_paragraph(paragraph)
    document.save(folder / "A-001.docx")
    Image.new("L", (300, 200), 255).save(folder / "B-002.png")
    (folder / "notes.txt").write_text("not an agreement")
    return folder


class CheckedFakeLLM(FakeLLM):
    def check(self):
        pass


def test_predict_folder_end_to_end_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "_ocr_tile", lambda image: [ocr.OcrLine("scanned agreement text", 0.99, [0, 0, 90, 20])])
    folder, out_csv, work = _folder(tmp_path), tmp_path / "out" / "predictions.csv", tmp_path / "work"
    results = predict_folder(folder, out_csv, work, llm=CheckedFakeLLM(_answer(), _answer()))

    assert sorted(results) == ["A-001", "B-002"] and all(r.status == "ok" for r in results.values())
    with open(out_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == [FILE_NAME_COLUMN, *LABEL_COLUMNS]
    assert [r[FILE_NAME_COLUMN] for r in rows] == ["A-001", "B-002"]
    assert rows[0][VALUE] == "18500" and rows[0][NOTICE] == "30"
    details = json.loads((work / "A-001.prediction.json").read_text(encoding="utf-8"))
    assert details["evidence"][VALUE]["evidence"] == "Monthly rent of Rs. 18,500/-"
    summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
    assert summary["unsupported"] == 1  # notes.txt reported, not silently skipped


def test_predict_single_file(tmp_path):
    folder, out_csv, work = _folder(tmp_path), tmp_path / "one.csv", tmp_path / "work"
    results = predict_folder(folder / "A-001.docx", out_csv, work, llm=CheckedFakeLLM(_answer()))

    assert list(results) == ["A-001"]  # the PNG next to it is not touched
    with open(out_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r[FILE_NAME_COLUMN] for r in rows] == ["A-001"] and rows[0][VALUE] == "18500"
    assert not (work / "B-002.txt").exists()
    assert json.loads((work / "summary.json").read_text(encoding="utf-8"))["files"] == 1


@pytest.mark.parametrize("name, message", [("missing.docx", "input not found"),
                                           ("notes.txt", "unsupported file type '.txt'")])
def test_cli_rejects_bad_input(tmp_path, name, message):
    _folder(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-m", "src.predict", "--input", str(tmp_path / "agreements" / name),
         "--output", str(tmp_path / "x.csv")],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 2 and message in completed.stderr


def test_cli_runs_outside_jupyter():
    completed = subprocess.run([sys.executable, "-m", "src.predict", "--help"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0 and "--input" in completed.stdout


@pytest.mark.skipif(os.getenv("RUN_LLM_SMOKE") != "1", reason="set RUN_LLM_SMOKE=1 to run against the local model")
def test_real_model_on_synthetic_docx(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    document = docx.Document()
    for paragraph in SYNTHETIC_AGREEMENT:
        document.add_paragraph(paragraph)
    document.save(folder / "synthetic.docx")
    results = predict_folder(folder, tmp_path / "p.csv", tmp_path / "work")
    m = results["synthetic"].metadata
    assert (m.agreement_value, m.agreement_start_date, m.agreement_end_date, m.renewal_notice_days) == (
        18500, "15.01.2024", "14.12.2024", 30)
    assert "Ananya Rao" in m.party_one and "Kabir Mehta" in m.party_two
