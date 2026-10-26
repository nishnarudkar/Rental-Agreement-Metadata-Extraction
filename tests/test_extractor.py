"""Extractor tests with a scripted fake LLM (no model needed)."""

import json
from types import SimpleNamespace

import pytest

from src.config import LABEL_COLUMNS, TRAIN_CSV
from src.dataset import load_labels
from src.extractor import MAX_ATTEMPTS, evidence_report, extract, system_prompt, user_prompt
from src.llm import LLMError, LLMInputTooLongError, LLMOutputError
from src.schemas import AgreementExtraction

VALUE, START, END, NOTICE, PARTY_ONE, PARTY_TWO = LABEL_COLUMNS

DOCUMENT = (
    "This lease is made on 3rd January 2024 between Ms. Ananya Rao (the Owner) and Mr. Kabir Mehta (the Tenant). "
    "The tenancy commences on 15th January 2024 and ends on 14th December 2024. "
    "Monthly rent of Rs. 18,500/-. Either party may end this lease by giving one month's written notice."
)

NULL_AMOUNT = {"value": None, "evidence": None}
NULL_NOTICE = {"value": None, "unit": None, "evidence": None}


def _answer(**overrides) -> AgreementExtraction:
    data = {
        "party_one": {"name": "Ms. Ananya Rao", "representative": None, "evidence": "between Ms. Ananya Rao (the Owner)"},
        "party_two": {"name": "Mr. Kabir Mehta", "representative": None, "evidence": "Mr. Kabir Mehta (the Tenant)"},
        "monthly_rent": {"value": 18500, "evidence": "Monthly rent of Rs. 18,500/-"},
        "security_deposit": NULL_AMOUNT,
        "advance_rent": NULL_AMOUNT,
        "execution_date": {"value": "03.01.2024", "evidence": "made on 3rd January 2024"},
        "tenancy_start_date": {"value": "15.01.2024", "evidence": "commences on 15th January 2024"},
        "tenancy_end_date": {"value": "14.12.2024", "evidence": "ends on 14th December 2024"},
        "term_months": None,
        "termination_notice": {"value": 1, "unit": "months", "evidence": "giving one month's written notice"},
        "renewal_notice": NULL_NOTICE,
    }
    data.update(overrides)
    return AgreementExtraction.model_validate(data)


class FakeLLM:
    """Returns (or raises) scripted outcomes in order and records each user prompt."""

    def __init__(self, *outcomes):
        self.outcomes, self.prompts = list(outcomes), []

    def generate_json(self, system, user, schema):
        self.prompts.append((system, user))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(parsed=outcome, prompt_tokens=900, output_tokens=400)


def test_successful_extraction_with_evidence():
    result = extract(DOCUMENT, FakeLLM(_answer()), "doc-1")
    assert (result.status, result.attempts, result.error, result.warnings) == ("ok", 1, None, [])
    assert result.metadata.to_csv_row() == {
        VALUE: "18500", START: "15.01.2024", END: "14.12.2024", NOTICE: "30",
        PARTY_ONE: "Ms. Ananya Rao", PARTY_TWO: "Mr. Kabir Mehta",
    }
    assert result.evidence[NOTICE] == {
        "value": 30, "evidence": "giving one month's written notice", "evidence_found": True,
    }
    json.dumps(result.to_dict())  # internal output is JSON-serialisable


def test_retry_feeds_back_the_error_then_succeeds():
    llm = FakeLLM(LLMOutputError("tenancy_end_date.value: not a valid date"), _answer())
    result = extract(DOCUMENT, llm, "doc-1")
    assert (result.status, result.attempts) == ("ok", 2)
    assert "rejected" not in llm.prompts[0][1]
    assert "Your previous answer was rejected: tenancy_end_date.value: not a valid date" in llm.prompts[1][1]


def test_gives_up_after_bounded_attempts():
    llm = FakeLLM(*[LLMOutputError(f"bad {i}") for i in range(MAX_ATTEMPTS)])
    result = extract(DOCUMENT, llm, "doc-1")
    assert (result.status, result.attempts) == ("failed", MAX_ATTEMPTS)
    assert result.error == f"gave up after {MAX_ATTEMPTS} attempts; last error: bad {MAX_ATTEMPTS - 1}"
    assert result.metadata.to_csv_row() == dict.fromkeys(LABEL_COLUMNS, "")  # nothing invented


@pytest.mark.parametrize("error", [LLMInputTooLongError("too long"), LLMError("server not reachable")])
def test_unrecoverable_errors_are_not_retried(error):
    llm = FakeLLM(error, _answer())
    result = extract(DOCUMENT, llm, "doc-1")
    assert (result.status, result.attempts, result.error) == ("failed", 1, str(error))
    assert len(llm.prompts) == 1


def test_empty_text_is_not_sent_to_the_model():
    llm = FakeLLM()
    result = extract("  \n ", llm, "blank")
    assert result.status == "failed" and "empty" in result.error
    assert llm.prompts == []


def test_evidence_not_in_document_is_flagged_but_value_kept():
    invented = _answer(monthly_rent={"value": 20000, "evidence": "rent of Rs. 20,000 per month"})
    result = extract(DOCUMENT, FakeLLM(invented), "doc-1")
    assert result.metadata.agreement_value == 20000  # not silently changed
    assert result.warnings == [f"{VALUE}: evidence quote not found verbatim in the document"]


def test_value_without_evidence_is_flagged():
    result = extract(DOCUMENT, FakeLLM(_answer(monthly_rent={"value": 18500, "evidence": None})), "doc-1")
    assert result.warnings == [f"{VALUE}: value given without evidence"]


def test_evidence_matching_ignores_case_whitespace_and_line_breaks():
    quote = "MONTHLY   rent of\nRs. 18,500/-."
    report = evidence_report(_answer(monthly_rent={"value": 18500, "evidence": quote}), DOCUMENT)
    assert report[VALUE]["evidence_found"] is True


def test_null_values_need_no_evidence():
    result = extract(DOCUMENT, FakeLLM(_answer(termination_notice=NULL_NOTICE)), "doc-1")
    assert result.metadata.renewal_notice_days is None
    assert result.warnings == []


def test_prompt_contains_rules_schema_and_document():
    system, user = system_prompt(), user_prompt(DOCUMENT)
    assert "Return one JSON object that matches this JSON schema" in system
    assert json.dumps(AgreementExtraction.model_json_schema(), separators=(",", ":")) in system
    assert DOCUMENT in user


def test_prompt_contains_no_dataset_specific_values():
    """Guard against hard-coding: no file name, party name or amount from the labels."""
    prompt = system_prompt().casefold()
    labels = load_labels(TRAIN_CSV)
    for _, row in labels.iterrows():
        assert row["File Name"].strip().casefold() not in prompt
        for column in (PARTY_ONE, PARTY_TWO):
            assert row[column].strip().casefold() not in prompt
        assert f" {row[VALUE].strip()} " not in prompt
