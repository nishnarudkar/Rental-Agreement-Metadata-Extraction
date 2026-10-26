"""v4 verification pass, with a fake LLM that answers both calls."""

from types import SimpleNamespace

from src.config import LABEL_COLUMNS
from src.extractor import CONFIGS, Verdicts, extract
from src.llm import LLMError
from tests.test_extractor import DOCUMENT, _answer

VALUE, START, END, NOTICE, PARTY_ONE, PARTY_TWO = LABEL_COLUMNS
ALL_TRUE = dict.fromkeys(Verdicts.model_fields, True)


class TwoPassLLM:
    """First call returns the extraction; second call returns the verdicts (or raises)."""

    def __init__(self, extraction, verdicts):
        self.extraction, self.verdicts, self.calls = extraction, verdicts, []

    def generate_json(self, system, user, schema):
        self.calls.append((schema.__name__, user))
        if schema is Verdicts:
            if isinstance(self.verdicts, Exception):
                raise self.verdicts
            return SimpleNamespace(parsed=Verdicts(**self.verdicts), prompt_tokens=200, output_tokens=40)
        return SimpleNamespace(parsed=self.extraction, prompt_tokens=900, output_tokens=400)


def test_supported_values_are_kept():
    result = extract(DOCUMENT, TwoPassLLM(_answer(), ALL_TRUE), "doc-1", CONFIGS["v4"])
    assert result.metadata.agreement_value == 18500 and result.warnings == []
    assert result.verification == ALL_TRUE


def test_unsupported_value_is_removed_with_warning():
    llm = TwoPassLLM(_answer(), {**ALL_TRUE, "notice": False})
    result = extract(DOCUMENT, llm, "doc-1", CONFIGS["v4"])
    assert result.metadata.renewal_notice_days is None
    assert result.evidence[NOTICE]["value"] is None
    assert result.warnings == [f"{NOTICE}: removed, its quote does not state the value 30"]
    assert result.extraction.termination_notice.value == 1  # raw extraction kept for inspection


def test_only_present_values_are_sent_for_checking():
    llm = TwoPassLLM(_answer(monthly_rent={"value": None, "evidence": None}), ALL_TRUE)
    extract(DOCUMENT, llm, "doc-1", CONFIGS["v4"])
    schema, user = llm.calls[1]
    assert schema == "Verdicts" and '"agreement_value"' not in user and '"party_one"' in user


def test_derived_end_date_is_checked_through_the_term():
    x = _answer(tenancy_end_date={"value": None, "evidence": None}, term={"months": 11, "evidence": "for eleven months"})
    llm = TwoPassLLM(x, {**ALL_TRUE, "end_date_or_term": False})
    result = extract(DOCUMENT, llm, "doc-1", CONFIGS["v4"])
    assert '"value": "11 months"' in llm.calls[1][1]
    assert result.metadata.agreement_end_date is None


def test_verification_failure_keeps_values_and_warns():
    result = extract(DOCUMENT, TwoPassLLM(_answer(), LLMError("server gone")), "doc-1", CONFIGS["v4"])
    assert result.metadata.agreement_value == 18500
    assert result.warnings == ["verification failed, values kept unverified: server gone"]


def test_no_verification_call_without_flag():
    llm = TwoPassLLM(_answer(), ALL_TRUE)
    extract(DOCUMENT, llm, "doc-1", CONFIGS["v3"])
    assert [schema for schema, _ in llm.calls] == ["AgreementExtraction"]
