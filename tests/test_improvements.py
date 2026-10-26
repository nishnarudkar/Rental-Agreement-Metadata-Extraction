"""Prompt-10 additions: end-date derivation, salvage of partly invalid answers, prompt versions."""

import json

import pytest

from src.config import LABEL_COLUMNS
from src.extractor import CONFIGS, MAX_ATTEMPTS, PROMPTS, extract, salvage, system_prompt
from src.llm import LLMOutputError
from src.schemas import AgreementExtraction, AgreementMetadata, add_months, term_end_date
from tests.test_extractor import DOCUMENT, FakeLLM, _answer

VALUE, START, END, NOTICE, PARTY_ONE, PARTY_TWO = LABEL_COLUMNS
NULL_DATE = {"value": None, "evidence": None}


@pytest.mark.parametrize(
    "start, months, end",
    [
        ("01.04.2010", 11, "28.02.2011"),  # lands in a short February
        ("01.04.2008", 12, "31.03.2009"),
        ("05.12.2008", 11, "04.11.2009"),  # crosses a year end
        ("31.01.2023", 1, "27.02.2023"),  # day clamped to month length, then minus one day
        ("01.03.2024", 11, "31.01.2025"),
        ("29.02.2024", 12, "27.02.2025"),
    ],
)
def test_term_end_date(start, months, end):
    assert term_end_date(start, months) == end


def test_add_months_clamps_to_leap_february():
    from datetime import date

    assert add_months(date(2023, 1, 31), 13) == date(2024, 2, 29)


def _with_term(months, **overrides):
    return _answer(tenancy_end_date=NULL_DATE, term={"months": months, "evidence": "for eleven months"}, **overrides)


def test_end_date_derived_only_when_enabled():
    x = _with_term(11)
    assert AgreementMetadata.from_extraction(x).agreement_end_date is None
    assert AgreementMetadata.from_extraction(x, derive_end_date=True).agreement_end_date == "14.12.2024"


def test_stated_end_date_is_never_overridden():
    x = _answer(term={"months": 6, "evidence": "six months"})  # stated end 14.12.2024
    assert AgreementMetadata.from_extraction(x, derive_end_date=True).agreement_end_date == "14.12.2024"


def test_no_derivation_without_term():
    x = _with_term(None)
    assert AgreementMetadata.from_extraction(x, derive_end_date=True).agreement_end_date is None


def test_derived_end_date_uses_term_evidence():
    result = extract(DOCUMENT, FakeLLM(_with_term(11)), "doc-1", CONFIGS["v3"])
    assert result.evidence[END] == {"value": "14.12.2024", "evidence": "for eleven months", "evidence_found": False}


# --- salvage ------------------------------------------------------------------------------


def _rejected(**broken) -> LLMOutputError:
    """An LLMOutputError as the client raises it, built from a real validation failure."""
    data = _answer().model_dump()
    for path, value in broken.items():
        group, key = path.split("__")
        data[group][key] = value
    raw = json.dumps(data)
    try:
        AgreementExtraction.model_validate_json(raw)
    except ValueError as exc:
        return LLMOutputError("invalid", raw=raw, errors=exc.errors())
    raise AssertionError("expected the data to be invalid")


def test_salvage_nulls_only_the_invalid_fields():
    rescued, dropped = salvage(_rejected(execution_date__value="?.??& day of 2010", tenancy_start_date__value="11 months"))
    assert dropped == ["execution_date", "tenancy_start_date"]
    assert rescued.execution_date.value is None and rescued.execution_date.evidence is None
    assert rescued.monthly_rent.value == 18500  # valid fields kept
    assert rescued.party_one.name == "Ms. Ananya Rao"


def test_salvage_handles_cross_field_errors():
    rescued, dropped = salvage(_rejected(renewal_notice__unit="months"))  # unit without value
    assert dropped == ["renewal_notice"]
    assert rescued.renewal_notice.unit is None


def test_salvage_gives_up_on_unparseable_output():
    assert salvage(LLMOutputError("cut off", raw='{"party_one": ', errors=[])) == (None, [])
    assert salvage(LLMOutputError("cut off")) == (None, [])


def test_extract_salvages_after_last_attempt_when_enabled():
    errors = [_rejected(execution_date__value="??") for _ in range(MAX_ATTEMPTS)]
    result = extract(DOCUMENT, FakeLLM(*errors), "doc-1", CONFIGS["v2"])
    assert (result.status, result.attempts) == ("ok", MAX_ATTEMPTS)
    assert result.metadata.agreement_value == 18500
    assert result.warnings[0].startswith(f"execution_date: set to null after {MAX_ATTEMPTS} invalid answers")


def test_extract_without_salvage_still_fails():
    errors = [_rejected(execution_date__value="??") for _ in range(MAX_ATTEMPTS)]
    assert extract(DOCUMENT, FakeLLM(*errors), "doc-1", CONFIGS["v1"]).status == "failed"


def test_retry_feedback_allows_null():
    llm = FakeLLM(_rejected(execution_date__value="??"), _answer())
    extract(DOCUMENT, llm, "doc-1", CONFIGS["v2"])
    assert "If you cannot give a valid value for a field, use null." in llm.prompts[1][1]


# --- prompt versions ----------------------------------------------------------------------


def test_v3_prompt_does_not_ask_the_model_for_date_arithmetic():
    assert "minus one day" in PROMPTS["v2"]
    assert "minus one day" not in PROMPTS["v3"]
    assert "Do not calculate it" in PROMPTS["v3"]


def test_every_config_has_a_prompt():
    for config in CONFIGS.values():
        assert system_prompt(config.prompt_version).startswith("You extract metadata")
