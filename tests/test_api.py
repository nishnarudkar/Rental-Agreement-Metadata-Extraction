"""REST API tests with FastAPI's test client, a fake LLM and fake OCR (no model or server needed)."""

import io

import docx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import api, ocr
from src.config import LABEL_COLUMNS
from src.llm import LLMError
from tests.test_extractor import FakeLLM, _answer
from tests.test_smoke import SYNTHETIC_AGREEMENT

VALUE, START, END, NOTICE, PARTY_ONE, PARTY_TWO = LABEL_COLUMNS
DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class ApiFakeLLM(FakeLLM):
    def __init__(self, *outcomes, available=True):
        super().__init__(*outcomes)
        self.available = available
        self.config = api.LocalLLM().config  # real defaults for /health

    def check(self):
        if not self.available:
            raise LLMError("Ollama server not reachable at http://localhost:11434")


@pytest.fixture
def client_with():
    def make(llm):
        api.app.dependency_overrides[api.get_llm] = lambda: llm
        return TestClient(api.app)

    yield make
    api.app.dependency_overrides.clear()


def _docx_bytes() -> bytes:
    document = docx.Document()
    for paragraph in SYNTHETIC_AGREEMENT:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (300, 200), 255).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_ok(client_with):
    body = client_with(ApiFakeLLM()).get("/health").json()
    assert body["status"] == "ok" and body["llm"] == "ok"
    assert body["configuration"] == "v3" and body["supported_types"] == [".docx", ".png"]


def test_health_reports_missing_model_server(client_with):
    body = client_with(ApiFakeLLM(available=False)).get("/health").json()
    assert body["status"] == "degraded" and "not reachable" in body["llm"]


def test_extract_docx(client_with):
    client = client_with(ApiFakeLLM(_answer()))
    response = client.post("/extract", files={"file": ("123-Rental-Agreement.pdf.docx", _docx_bytes(), DOCX_TYPE)})
    assert response.status_code == 200
    body = response.json()
    assert (body["document_id"], body["document_type"], body["status"]) == ("123-Rental-Agreement", "docx", "ok")
    assert body["prediction"] == {VALUE: 18500, START: "15.01.2024", END: "14.12.2024", NOTICE: 30,
                                  PARTY_ONE: "Ms. Ananya Rao", PARTY_TWO: "Mr. Kabir Mehta"}
    assert body["evidence"][VALUE]["evidence"] == "Monthly rent of Rs. 18,500/-"
    assert body["configuration"] == "v3"


def test_extract_png_goes_through_ocr(client_with, monkeypatch):
    monkeypatch.setattr(ocr, "_ocr_tile", lambda image: [ocr.OcrLine("scanned agreement", 0.99, [0, 0, 90, 20])])
    llm = ApiFakeLLM(_answer())
    response = client_with(llm).post("/extract", files={"file": ("scan.png", _png_bytes(), "image/png")})
    assert response.status_code == 200 and response.json()["document_type"] == "png"
    assert "scanned agreement" in llm.prompts[0][1]  # the OCR text reached the model


@pytest.mark.parametrize("name", ["..\\..\\evil\\A-1.docx", "../../evil/A-1.docx"])
def test_uploaded_path_is_not_trusted(client_with, name):
    response = client_with(ApiFakeLLM(_answer())).post("/extract", files={"file": (name, _docx_bytes(), DOCX_TYPE)})
    assert response.status_code == 200 and response.json()["filename"] == "A-1.docx"


@pytest.mark.parametrize("name", ["agreement.pdf", "notes.txt", "noextension"])
def test_unsupported_type(client_with, name):
    response = client_with(ApiFakeLLM()).post("/extract", files={"file": (name, b"data", "application/octet-stream")})
    assert response.status_code == 415 and "unsupported file type" in response.json()["detail"]


def test_corrupt_docx_is_422(client_with):
    response = client_with(ApiFakeLLM()).post("/extract", files={"file": ("bad.docx", b"not a zip", DOCX_TYPE)})
    assert response.status_code == 422 and "not a DOCX" in response.json()["detail"]


def test_empty_upload_is_422(client_with):
    response = client_with(ApiFakeLLM()).post("/extract", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 422 and "empty" in response.json()["detail"]


def test_too_large_is_413(client_with, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    response = client_with(ApiFakeLLM()).post("/extract", files={"file": ("big.docx", b"x" * 11, DOCX_TYPE)})
    assert response.status_code == 413


def test_model_unavailable_is_503(client_with):
    response = client_with(ApiFakeLLM(available=False)).post(
        "/extract", files={"file": ("a.docx", _docx_bytes(), DOCX_TYPE)})
    assert response.status_code == 503 and "not reachable" in response.json()["detail"]


def test_failed_extraction_returns_status_and_no_invented_values(client_with):
    from src.llm import LLMOutputError

    llm = ApiFakeLLM(*[LLMOutputError("bad answer") for _ in range(3)])
    body = client_with(llm).post("/extract", files={"file": ("a.docx", _docx_bytes(), DOCX_TYPE)}).json()
    assert body["status"] == "failed" and "gave up after 3 attempts" in body["error"]
    assert set(body["prediction"].values()) == {None}


def test_openapi_docs_available(client_with):
    client = client_with(ApiFakeLLM())
    assert client.get("/docs").status_code == 200
    assert "/extract" in client.get("/openapi.json").json()["paths"]
