"""REST API around the frozen extraction pipeline.

    uvicorn src.api:app --host 127.0.0.1 --port 8000

Endpoints (interactive documentation at http://127.0.0.1:8000/docs):

- ``GET  /health``  server, local-model and configuration status
- ``POST /extract`` upload one agreement (.docx or .png, multipart field ``file``);
  returns the six fields, an evidence quote per field and any warnings

The API uses exactly the same code and frozen configuration as the command line
(``src.predict``): DOCX reader / OCR, then the local LLM. Uploaded files are
written only to a temporary folder that is deleted after the request, and
requests are processed one at a time because the OCR engine and the GPU model
are shared. A scanned page can take several minutes to OCR on CPU.
"""

import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import LABEL_COLUMNS
from src.dataset import document_id
from src.document_reader import READERS, DocumentReadError, read_document
from src.extractor import DEFAULT_CONFIG, FINAL_CONFIG_NAME, extract
from src.llm import LLMError, LocalLLM

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(
    title="Rental Agreement Metadata Extraction",
    version="1.0.0",
    description="Extracts agreement value, start/end dates, notice period and both parties from a rental "
    "agreement (DOCX or scanned PNG) with a local LLM. Each value comes with an evidence quote.",
)

_processing = threading.Lock()


@lru_cache(maxsize=1)
def get_llm() -> LocalLLM:
    return LocalLLM()


class FieldEvidence(BaseModel):
    value: str | int | None
    evidence: str | None
    evidence_found: bool


class ExtractResponse(BaseModel):
    document_id: str
    filename: str
    document_type: str
    status: str  # "ok" or "failed"
    prediction: dict[str, str | int | None]  # the six fields, keyed by the dataset's CSV column names
    evidence: dict[str, FieldEvidence]
    warnings: list[str]
    error: str | None
    ingestion_seconds: float
    extraction_seconds: float
    configuration: str


@app.get("/health")
def health(llm: LocalLLM = Depends(get_llm)) -> dict:
    try:
        llm.check()
        llm_status = "ok"
    except LLMError as exc:
        llm_status = str(exc)
    return {
        "status": "ok" if llm_status == "ok" else "degraded",
        "llm": llm_status,
        "model": llm.config.model,
        "configuration": FINAL_CONFIG_NAME,
        "supported_types": sorted(READERS),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


@app.post("/extract", response_model=ExtractResponse)
def extract_document(file: UploadFile = File(...), llm: LocalLLM = Depends(get_llm)) -> ExtractResponse:
    # Never trust a client-supplied path; strip both separator styles on every OS.
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in READERS:
        raise HTTPException(415, f"unsupported file type '{suffix}' (expected .docx or .png)")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        llm.check()
    except LLMError as exc:
        raise HTTPException(503, f"local model unavailable: {exc}") from exc

    with _processing, tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"  # safe name; the original is only used for the ID
        path.write_bytes(data)
        start = time.perf_counter()
        try:
            text = read_document(path)
        except DocumentReadError as exc:
            raise HTTPException(422, f"cannot read {filename}: {exc.reason}") from exc
        ingestion_seconds = round(time.perf_counter() - start, 2)
        doc_id = document_id(Path(filename))
        result = extract(text.text, llm, doc_id, DEFAULT_CONFIG)

    empty = {"value": None, "evidence": None, "evidence_found": False}
    return ExtractResponse(
        document_id=doc_id,
        filename=filename,
        document_type=text.document_type,
        status=result.status,
        prediction=result.metadata.model_dump(by_alias=True),
        evidence={c: FieldEvidence(**result.evidence.get(c, empty)) for c in LABEL_COLUMNS},
        warnings=text.warnings + result.warnings,
        error=result.error,
        ingestion_seconds=ingestion_seconds,
        extraction_seconds=result.seconds,
        configuration=FINAL_CONFIG_NAME,
    )
