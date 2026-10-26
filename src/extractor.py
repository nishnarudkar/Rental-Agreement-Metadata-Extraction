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

from pydantic import BaseModel, ConfigDict

from src.config import EXTRACTED_TEXT_DIR, LABEL_COLUMNS, OUTPUTS_DIR
from src.llm import LLMError, LLMOutputError, LocalLLM
from src.schemas import AgreementExtraction, AgreementMetadata

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

PROMPT_V1 = """You extract metadata from rental and lease agreements.

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


# v2: changes driven by the v1 error analysis on training documents (docs/experiments.md).
PROMPT_V2 = """You extract metadata from rental and lease agreements.

Rules:
1. Use only the agreement text you are given. Do not guess or use outside knowledge.
2. If a value is not stated in the text, or is illegible or incomplete, return null for the value and for its evidence.
3. For every non-null value, "evidence" is a short quote (at most about 25 words) copied exactly from the text, and the quote must state that value.
4. The text may come from OCR and contain spelling mistakes. Copy names and quotes as they appear.

Field definitions:
- party_one: the lessor / landlord / owner who lets the property. party_two: the lessee / tenant.
  Take the names from the part of the agreement that introduces the parties (usually near the start, e.g. "between A ... and B"),
  not from signature blocks or witness lists.
  "name" is the name only: leave out courtesy titles (Mr., Mrs., Ms., Dr., Prof., Sri, Smt.), relations (S/o, D/o, W/o and the
  relative's name), ages and addresses. Keep initials and spelling exactly as written.
  If one party consists of several people, give all their names, joined as in the text.
  If a party is a company or firm, "name" is the company and "representative" is the person signing or acting for it.
- monthly_rent: the recurring rent per month. Never the security deposit, advance, maintenance charges or a one-time payment.
  If the amount in figures and the amount in words disagree, use the figures.
- security_deposit, advance_rent: report separately when stated.
- execution_date: the date the agreement was made, executed or signed.
- tenancy_start_date: the date the tenancy or lease period begins, when stated. It can differ from the execution date.
  If the period is said to run from the date of the agreement, use the execution date.
- tenancy_end_date: the end date when stated; otherwise, when a start date and a term are stated, the last day of the term
  (start date plus the term, minus one day).
- term: the length of the tenancy in months when stated, e.g. "eleven months" -> 11, "one year" -> 12.
- termination_notice: the notice period a party must give before terminating the agreement or vacating the premises.
  Grace periods, payment deadlines, penalties and interest are not notice periods. If there is no such clause, return null.
  Give the number and the unit as written, e.g. "three months" -> value 3, unit "months".
- renewal_notice: notice required specifically to renew or extend the agreement, only if such a clause exists.
- Dates are DD.MM.YYYY and must be complete calendar dates. Never put a duration or a partial date in a date field.
- Amounts are whole numbers without currency symbols, separators or decimals.

Return one JSON object that matches this JSON schema:
"""

# v3: v2 error analysis. Titles and relations get their own schema slots instead of
# an instruction to delete them; the rule that clashed with it ("copy names as they
# appear") is reworded; the model no longer does date arithmetic (the end date is
# derived in code from start + term when it is not stated).
PROMPT_V3 = """You extract metadata from rental and lease agreements.

Rules:
1. Use only the agreement text you are given. Do not guess or use outside knowledge.
2. If a value is not stated in the text, or is illegible or incomplete, return null for the value and for its evidence.
   Never reconstruct a garbled number or date.
3. For every non-null value, "evidence" is a short quote (at most about 25 words) copied exactly from the text, and the quote must state that value.
4. The text may come from OCR and contain spelling mistakes. Copy evidence quotes exactly; keep the spelling of names as written.

Field definitions:
- party_one: the lessor / landlord / owner who lets the property. party_two: the lessee / tenant.
  Take them from the part of the agreement that introduces the parties (usually near the start, e.g. "between A ... and B"),
  not from signature blocks or witness lists. Split what is written there into:
  "title": a courtesy title before the name (Mr., Mrs., Ms., Dr., Prof., Sri, Smt.), or null;
  "name": the name itself, with its initials, without title, relation, list number, age or address;
  "relation": a relation after the name (S/o, D/o, W/o and the relative's name), or null.
  If one party consists of several people, put all their names in "name", joined as in the text.
  If a party is a company or firm, "name" is the company and "representative" is the person signing or acting for it.
- monthly_rent: the recurring rent per month. Never the security deposit, advance, maintenance charges or a one-time payment.
  If the amount in figures and the amount in words disagree, use the figures.
- security_deposit, advance_rent: report separately when stated.
- execution_date: the date the agreement was made, executed or signed.
- tenancy_start_date: the date the tenancy or lease period begins, when stated. It can differ from the execution date.
  If the period is said to run from the date of the agreement, use the execution date.
- tenancy_end_date: only when the agreement states the end date itself. Do not calculate it; otherwise return null.
- term: the length of the tenancy in months when stated, e.g. "eleven months" -> 11, "one year" -> 12.
- termination_notice: the notice period a party must give before terminating the agreement or vacating the premises.
  Grace periods, payment deadlines, penalties and interest are not notice periods. If there is no such clause, return null.
  Give the number and the unit as written, e.g. "three months" -> value 3, unit "months".
- renewal_notice: notice required specifically to renew or extend the agreement, only if such a clause exists.
- Dates are DD.MM.YYYY and must be complete calendar dates. Never put a duration or a partial date in a date field.
- Amounts are whole numbers without currency symbols, separators or decimals.

Return one JSON object that matches this JSON schema:
"""

# Every prompt version used in development stays here so experiments can be reproduced.
# (v1 and v2 were run with earlier schema versions; their exact prompts are saved
# with each experiment in outputs/experiments/<name>/<subset>/system_prompt.txt.)
PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3}

# Second pass (v4): the model checks each submitted value against its own quote.
VERIFY_PROMPT = """You check extracted values against the quotes they were taken from.
For each item, answer true only if the quote itself explicitly states that value for that field:
- an amount must appear in the quote as that amount of monthly rent;
- a date must be written in the quote (in any format), with its day, month and year;
- a term must be stated in the quote as that number of months or years;
- a notice period must be a period of notice for terminating the agreement or vacating, of that length;
- a party name must appear in the quote as that party.
Answer false if the value is guessed, reconstructed from unreadable text, or not in the quote.
Return one JSON object with a true/false answer for exactly the items given."""


class Verdicts(BaseModel):
    """Verification answers keyed by submission column (null when not asked)."""

    model_config = ConfigDict(extra="forbid")
    agreement_value: bool | None
    start_date: bool | None
    end_date_or_term: bool | None
    notice: bool | None
    party_one: bool | None
    party_two: bool | None


VERDICT_KEYS = dict(zip(LABEL_COLUMNS, Verdicts.model_fields))


@dataclass(frozen=True)
class ExtractorConfig:
    prompt_version: str = "v1"
    derive_end_date: bool = False  # compute a missing end date from start + term in code
    salvage: bool = False  # after the last failed attempt, null only the invalid fields
    verify: bool = False  # second pass: drop values the model cannot support with its own quote


CONFIGS = {
    "v1": ExtractorConfig("v1"),
    "v2": ExtractorConfig("v2", salvage=True),
    "v3": ExtractorConfig("v3", derive_end_date=True, salvage=True),
    "v4": ExtractorConfig("v3", derive_end_date=True, salvage=True, verify=True),
}
# Selected on training documents and frozen before test evaluation (docs/experiments.md,
# config/frozen_config.json). Do not change without re-running development.
FINAL_CONFIG_NAME = "v3"
DEFAULT_CONFIG = CONFIGS[FINAL_CONFIG_NAME]


def system_prompt(version: str = DEFAULT_CONFIG.prompt_version) -> str:
    return PROMPTS[version] + json.dumps(AgreementExtraction.model_json_schema(), separators=(",", ":"))


def user_prompt(text: str) -> str:
    return f"Rental agreement text:\n<<<\n{text}\n>>>\nReturn the JSON object."


def _comparable(text: str) -> str:
    return " ".join(text.split()).casefold().strip(" .,;:'\"")


def evidence_report(extraction: AgreementExtraction, text: str, derive_end_date: bool = False) -> dict[str, dict]:
    """Per submission column: value, evidence quote, and whether the quote occurs in the text."""
    document = _comparable(text)
    metadata = AgreementMetadata.from_extraction(extraction, derive_end_date).model_dump(by_alias=True)
    report = {}
    for column, source in extraction.submission_sources(derive_end_date).items():
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
    verification: dict | None = None  # v4: verdict per checked value
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
            "verification": self.verification,
            "error": self.error,
            "attempts": self.attempts,
            "seconds": self.seconds,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "extraction": self.extraction.model_dump() if self.extraction else None,
        }


def salvage(error: LLMOutputError) -> tuple[AgreementExtraction | None, list[str]]:
    """Keep a rejected answer's valid fields, setting only the invalid top-level fields to null.

    Each invalid field is nulled as a whole (value and evidence together), so no
    partial value survives. Returns (None, []) if the answer cannot be rescued.
    """
    if not error.raw or not error.errors:
        return None, []
    try:
        data = json.loads(error.raw)
    except ValueError:
        return None, []
    fields = AgreementExtraction.model_fields
    dropped = sorted({e["loc"][0] for e in error.errors if e["loc"]})
    if not dropped or any(name not in fields for name in dropped):
        return None, []
    for name in dropped:
        data[name] = dict.fromkeys(fields[name].annotation.model_fields)
    try:
        return AgreementExtraction.model_validate(data), dropped
    except ValueError:
        return None, []


FIELD_LABELS = dict(zip(LABEL_COLUMNS, ["monthly rent", "agreement start date", "agreement end date",
                                         "notice period", "lessor / owner (party one)", "lessee / tenant (party two)"]))


def verify(result: "ExtractionResult", llm: LocalLLM) -> None:
    """Ask the model whether each submitted value is stated by its own quote; null the ones that are not.

    A derived end date is checked through the term it was computed from.
    Values are only ever removed, never changed or added.
    """
    x = result.extraction
    items = {}
    for column, item in result.evidence.items():
        if item["value"] is None:
            continue
        label, value, quote = FIELD_LABELS[column], item["value"], item["evidence"]
        if column == LABEL_COLUMNS[2] and x.tenancy_end_date.value is None:  # derived from start + term
            label, value, quote = "tenancy term", f"{x.term.months} months", x.term.evidence
        items[VERDICT_KEYS[column]] = {"field": label, "value": value, "quote": quote}
    if not items:
        return
    user = ("Items to check (answer null for keys not listed):\n" + json.dumps(items, indent=1, ensure_ascii=False))
    try:
        verdicts = llm.generate_json(VERIFY_PROMPT, user, Verdicts).parsed.model_dump()
    except LLMError as exc:
        result.warnings.append(f"verification failed, values kept unverified: {exc}")
        return
    result.verification = {key: verdicts[key] for key in items}
    attributes = dict(zip(LABEL_COLUMNS, AgreementMetadata.model_fields))
    for column, key in VERDICT_KEYS.items():
        if key in items and verdicts[key] is False:
            result.warnings.append(f"{column}: removed, its quote does not state the value {items[key]['value']!r}")
            result.metadata = result.metadata.model_copy(update={attributes[column]: None})
            result.evidence[column]["value"] = None


def _finish(result: ExtractionResult, extraction: AgreementExtraction, text: str, config: ExtractorConfig) -> None:
    result.status, result.error = "ok", None
    result.extraction = extraction
    result.metadata = AgreementMetadata.from_extraction(extraction, config.derive_end_date)
    result.evidence = evidence_report(extraction, text, config.derive_end_date)
    result.warnings = _evidence_warnings(result.evidence) + result.warnings


def extract(text: str, llm: LocalLLM, document_id: str = "", config: ExtractorConfig = DEFAULT_CONFIG) -> ExtractionResult:
    """Extract metadata from one document's text. Never raises; failures are reported."""
    start = time.perf_counter()
    result = ExtractionResult(document_id=document_id, status="failed")
    if not text.strip():
        result.error = "document text is empty; nothing to extract"
        return result

    system, user = system_prompt(config.prompt_version), user_prompt(text)
    feedback, last_output_error = "", None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result.attempts = attempt
        try:
            answer = llm.generate_json(system, user + feedback, AgreementExtraction)
        except LLMOutputError as exc:
            result.error, last_output_error = str(exc), exc
            log.warning("%s attempt %d/%d rejected: %s", document_id, attempt, MAX_ATTEMPTS, exc)
            feedback = (
                f"\n\nYour previous answer was rejected: {exc}\n"
                "Return a corrected JSON object that follows the schema and the rules. "
                "If you cannot give a valid value for a field, use null."
            )
            continue
        except LLMError as exc:  # server down, input too long: retrying cannot help
            result.error = str(exc)
            log.error("%s failed: %s", document_id, exc)
            break
        result.prompt_tokens, result.output_tokens = answer.prompt_tokens, answer.output_tokens
        _finish(result, answer.parsed, text, config)
        break
    else:
        rescued, dropped = salvage(last_output_error) if config.salvage and last_output_error else (None, [])
        if rescued is not None:
            result.warnings = [f"{name}: set to null after {MAX_ATTEMPTS} invalid answers ({last_output_error})"
                               for name in dropped]
            _finish(result, rescued, text, config)
            log.warning("%s: salvaged answer, nulled %s", document_id, dropped)
        else:
            result.error = f"gave up after {MAX_ATTEMPTS} attempts; last error: {result.error}"

    if config.verify and result.status == "ok":
        verify(result, llm)
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
