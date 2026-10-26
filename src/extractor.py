"""AI extraction of rental agreement metadata with a local LLM.

The model reads the extracted document text and fills ``AgreementExtraction``
(see ``src/schemas.py``): every value with a verbatim evidence quote, null when
the document does not state it. Output is constrained to the JSON schema by
Ollama and validated by Pydantic; invalid answers are retried a bounded number
of times with the validation error fed back to the model.

No regex or keyword rules extract any field. The only text matching here is
the *evidence check*, which verifies after the fact that each quoted excerpt
really occurs in the document, and only adds warnings.

    python -m src.extractor --split train --documents 47854715-RENTAL-AGREEMENT
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field

from src.config import EXTRACTED_TEXT_DIR, LABEL_COLUMNS, OUTPUTS_DIR
from src.llm import LLMError, LLMOutputError, LocalLLM
from src.schemas import AgreementExtraction, AgreementMetadata

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

INSTRUCTIONS = """You extract metadata from rental and lease agreements.

Rules:
1. Use only the agreement text you are given. Do not guess or use outside knowledge.
2. If a value is not stated in the text, return null for the value and for its evidence.
3. For every non-null value, "evidence" is a short quote (at most about 25 words) copied exactly from the text that supports it.
4. The text may come from OCR and contain spelling mistakes. Copy names and quotes as they appear.

Field definitions:
- party_one: the lessor / landlord / owner who lets the property. party_two: the lessee / tenant.
  "name" is the name as written where the party is introduced, without address, age or relation (S/o, D/o, W/o).
  If a party is a company or firm, "name" is the company and "representative" is the person signing or acting for it.
- monthly_rent: the recurring rent per month. Never the security deposit, advance, maintenance charges or a one-time payment.
  If the amount in figures and the amount in words disagree, use the figures.
- security_deposit, advance_rent: report separately when stated.
- execution_date: the date the agreement was made, executed or signed.
- tenancy_start_date: the date the tenancy or lease period begins, when stated. It can differ from the execution date.
  If the period is said to run from the date of the agreement, use the execution date.
- tenancy_end_date: the end date when stated; otherwise, when a start date and a term are stated, the last day of the term
  (start date plus the term, minus one day).
- term_months: the length of the tenancy in months, when stated.
- termination_notice: the notice either party must give to terminate the agreement or vacate the premises.
  Give the number and the unit as written, e.g. "three months" -> value 3, unit "months".
- renewal_notice: notice required specifically to renew or extend the agreement, only if such a clause exists.
- Dates are DD.MM.YYYY. Amounts are whole numbers without currency symbols, separators or decimals.

Return one JSON object that matches this JSON schema:
"""


def system_prompt() -> str:
    return INSTRUCTIONS + json.dumps(AgreementExtraction.model_json_schema(), separators=(",", ":"))


def user_prompt(text: str) -> str:
    return f"Rental agreement text:\n<<<\n{text}\n>>>\nReturn the JSON object."


def _comparable(text: str) -> str:
    return " ".join(text.split()).casefold().strip(" .,;:'\"")


def evidence_report(extraction: AgreementExtraction, text: str) -> dict[str, dict]:
    """Per submission column: value, evidence quote, and whether the quote occurs in the text."""
    document = _comparable(text)
    metadata = AgreementMetadata.from_extraction(extraction).model_dump(by_alias=True)
    report = {}
    for column, source in extraction.submission_sources().items():
        quote = source.evidence
        report[column] = {
            "value": metadata[column],
            "evidence": quote,
            "evidence_found": bool(quote) and _comparable(quote) in document,
        }
    return report


def _evidence_warnings(report: dict[str, dict]) -> list[str]:
    warnings = []
    for column, item in report.items():
        if item["value"] is None:
            continue
        if not item["evidence"]:
            warnings.append(f"{column}: value given without evidence")
        elif not item["evidence_found"]:
            warnings.append(f"{column}: evidence quote not found verbatim in the document")
    return warnings


@dataclass
class ExtractionResult:
    document_id: str
    status: str  # "ok" or "failed"
    metadata: AgreementMetadata = field(default_factory=AgreementMetadata)
    extraction: AgreementExtraction | None = None
    evidence: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    attempts: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "status": self.status,
            "prediction": self.metadata.model_dump(by_alias=True),
            "evidence": self.evidence,
            "warnings": self.warnings,
            "error": self.error,
            "attempts": self.attempts,
            "seconds": self.seconds,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "extraction": self.extraction.model_dump() if self.extraction else None,
        }


def extract(text: str, llm: LocalLLM, document_id: str = "") -> ExtractionResult:
    """Extract metadata from one document's text. Never raises; failures are reported."""
    start = time.perf_counter()
    result = ExtractionResult(document_id=document_id, status="failed")
    if not text.strip():
        result.error = "document text is empty; nothing to extract"
        return result

    system, user = system_prompt(), user_prompt(text)
    feedback = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result.attempts = attempt
        try:
            answer = llm.generate_json(system, user + feedback, AgreementExtraction)
        except LLMOutputError as exc:
            result.error = str(exc)
            log.warning("%s attempt %d/%d rejected: %s", document_id, attempt, MAX_ATTEMPTS, exc)
            feedback = (
                f"\n\nYour previous answer was rejected: {exc}\n"
                "Return a corrected JSON object that follows the schema and the rules."
            )
            continue
        except LLMError as exc:  # server down, input too long: retrying cannot help
            result.error = str(exc)
            log.error("%s failed: %s", document_id, exc)
            break
        result.status, result.error = "ok", None
        result.extraction = answer.parsed
        result.metadata = AgreementMetadata.from_extraction(answer.parsed)
        result.evidence = evidence_report(answer.parsed, text)
        result.warnings = _evidence_warnings(result.evidence)
        result.prompt_tokens, result.output_tokens = answer.prompt_tokens, answer.output_tokens
        break
    else:
        result.error = f"gave up after {MAX_ATTEMPTS} attempts; last error: {result.error}"

    result.seconds = round(time.perf_counter() - start, 1)
    return result


def print_result(result: ExtractionResult) -> None:
    print(f"\n=== {result.document_id}: {result.status} "
          f"({result.attempts} attempt(s), {result.seconds}s, {result.prompt_tokens}+{result.output_tokens} tokens)")
    if result.error:
        print(f"  error: {result.error}")
    for column in LABEL_COLUMNS:
        item = result.evidence.get(column, {"value": None, "evidence": None, "evidence_found": False})
        mark = "✓" if item["evidence_found"] else ("-" if item["value"] is None else "✗")
        print(f"  {column:<22} {str(item['value']):<40} [{mark}] {item['evidence']}")
    for warning in result.warnings:
        print(f"  warning: {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--documents", nargs="*", help="document IDs (default: every cached document)")
    parser.add_argument("--final", action="store_true",
                        help="required for --split test, so test data is only run deliberately")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("running on test requires --final (test data is reserved for final evaluation)")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    text_dir = EXTRACTED_TEXT_DIR / args.split
    ids = args.documents or sorted(p.stem for p in text_dir.glob("*.txt"))
    out_dir = PREDICTIONS_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LocalLLM()
    llm.check()
    failed = 0
    for doc_id in ids:
        text_path = text_dir / f"{doc_id}.txt"
        if not text_path.exists():
            print(f"[missing text] {doc_id}: run `python -m src.extract_text --split {args.split}` first")
            failed += 1
            continue
        result = extract(text_path.read_text(encoding="utf-8"), llm, doc_id)
        (out_dir / f"{doc_id}.json").write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                                               encoding="utf-8")
        print_result(result)
        failed += result.status != "ok"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
