import json

import pytest
from pydantic import ValidationError

from src.config import LABEL_COLUMNS
from src.schemas import AgreementExtraction, AgreementMetadata, DateValue, NoticePeriod, Party

NULL_AMOUNT = {"value": None, "evidence": None}
NULL_DATE = {"value": None, "evidence": None}
NULL_NOTICE = {"value": None, "unit": None, "evidence": None}


def _extraction(**overrides) -> dict:
    """A complete, synthetic model answer (fictional parties)."""
    data = {
        "party_one": {"name": "Ms. Ananya Rao", "representative": None, "evidence": "between Ms. Ananya Rao (the Owner)"},
        "party_two": {"name": "Mr. Kabir Mehta", "representative": None, "evidence": "and Mr. Kabir Mehta (the Tenant)"},
        "monthly_rent": {"value": 18500, "evidence": "monthly rent of Rs. 18,500/-"},
        "security_deposit": {"value": 100000, "evidence": "security deposit of Rs. 1,00,000/-"},
        "advance_rent": NULL_AMOUNT,
        "execution_date": {"value": "03.01.2024", "evidence": "made at Pune on 3rd January 2024"},
        "tenancy_start_date": {"value": "15.01.2024", "evidence": "commencing 15th January 2024"},
        "tenancy_end_date": {"value": "14.12.2024", "evidence": "ending 14th December 2024"},
        "term_months": 11,
        "termination_notice": {"value": 1, "unit": "months", "evidence": "one month's written notice"},
        "renewal_notice": NULL_NOTICE,
    }
    data.update(overrides)
    return data


# --- AgreementExtraction: validation ------------------------------------------------


def test_valid_extraction_parses():
    x = AgreementExtraction.model_validate(_extraction())
    assert x.monthly_rent.value == 18500
    assert x.termination_notice.days == 30


@pytest.mark.parametrize("bad", ["31.02.2011", "31.11.2009", "2024-01-15", "15/01/2024", "January 15, 2024"])
def test_invalid_dates_are_rejected(bad):
    with pytest.raises(ValidationError, match="not a valid DD.MM.YYYY calendar date"):
        DateValue(value=bad, evidence="x")


def test_dates_are_zero_padded():
    assert DateValue(value="1.4.2010", evidence=None).value == "01.04.2010"


def test_blank_strings_mean_missing_and_whitespace_is_collapsed():
    party = Party(name="  ", representative="", evidence="  between   A  and B ")
    assert (party.name, party.representative, party.evidence) == (None, None, "between A and B")


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgreementExtraction.model_validate(_extraction(confidence=0.9))


def test_missing_keys_are_rejected():
    data = _extraction()
    del data["renewal_notice"]
    with pytest.raises(ValidationError, match="renewal_notice"):
        AgreementExtraction.model_validate(data)


@pytest.mark.parametrize("rent", [-500, "18,500", "eighteen thousand", 18500.5])
def test_invalid_amounts_are_rejected(rent):
    with pytest.raises(ValidationError):
        AgreementExtraction.model_validate(_extraction(monthly_rent={"value": rent, "evidence": "x"}))


@pytest.mark.parametrize(
    "notice",
    [{"value": 2, "unit": None, "evidence": "x"}, {"value": None, "unit": "months", "evidence": "x"},
     {"value": 0, "unit": "days", "evidence": "x"}, {"value": 2, "unit": "years", "evidence": "x"}],
)
def test_invalid_notice_periods_are_rejected(notice):
    with pytest.raises(ValidationError):
        NoticePeriod.model_validate(notice)


@pytest.mark.parametrize("value, unit, days", [(15, "days", 15), (2, "weeks", 14), (3, "months", 90), (None, None, None)])
def test_notice_days(value, unit, days):
    assert NoticePeriod(value=value, unit=unit, evidence=None).days == days


def test_json_schema_is_closed_and_fully_required():
    schema = AgreementExtraction.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(AgreementExtraction.model_fields)
    for definition in schema["$defs"].values():
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"])


def test_extraction_json_round_trip():
    x = AgreementExtraction.model_validate(_extraction())
    assert AgreementExtraction.model_validate_json(x.model_dump_json()) == x


# --- mapping to the six fields ---------------------------------------------------------


def test_mapping_uses_rent_commencement_and_termination_notice():
    m = AgreementMetadata.from_extraction(AgreementExtraction.model_validate(_extraction()))
    assert m.agreement_value == 18500  # rent, not the 100000 deposit
    assert m.agreement_start_date == "15.01.2024"  # commencement, not the 03.01.2024 execution date
    assert m.agreement_end_date == "14.12.2024"
    assert m.renewal_notice_days == 30
    assert (m.party_one, m.party_two) == ("Ms. Ananya Rao", "Mr. Kabir Mehta")


def test_start_date_falls_back_to_execution_date():
    x = AgreementExtraction.model_validate(_extraction(tenancy_start_date=NULL_DATE))
    assert AgreementMetadata.from_extraction(x).agreement_start_date == "03.01.2024"


def test_termination_notice_preferred_over_renewal_notice():
    x = AgreementExtraction.model_validate(
        _extraction(
            termination_notice={"value": 2, "unit": "months", "evidence": "two months notice to terminate"},
            renewal_notice={"value": 1, "unit": "months", "evidence": "one month notice for renewal"},
        )
    )
    assert AgreementMetadata.from_extraction(x).renewal_notice_days == 60


def test_renewal_notice_used_when_no_termination_notice():
    x = AgreementExtraction.model_validate(
        _extraction(termination_notice=NULL_NOTICE, renewal_notice={"value": 45, "unit": "days", "evidence": "x"})
    )
    assert AgreementMetadata.from_extraction(x).renewal_notice_days == 45


def test_company_party_keeps_company_name_not_representative():
    company = {"name": "Acme Dairy Products Pvt Ltd", "representative": "Mr. R. Sharma, Director", "evidence": "x"}
    x = AgreementExtraction.model_validate(_extraction(party_two=company))
    assert AgreementMetadata.from_extraction(x).party_two == "Acme Dairy Products Pvt Ltd"


def test_missing_values_map_to_none():
    x = AgreementExtraction.model_validate(
        _extraction(monthly_rent=NULL_AMOUNT, tenancy_end_date=NULL_DATE, termination_notice=NULL_NOTICE)
    )
    m = AgreementMetadata.from_extraction(x)
    assert (m.agreement_value, m.agreement_end_date, m.renewal_notice_days) == (None, None, None)


# --- AgreementMetadata serialisation ------------------------------------------------------


def test_csv_row_uses_exact_column_names_and_empty_for_missing():
    row = AgreementMetadata(agreement_value=4000, party_one="A. Owner").to_csv_row()
    assert list(row) == LABEL_COLUMNS
    assert row == {
        "Aggrement Value": "4000", "Aggrement Start Date": "", "Aggrement End Date": "",
        "Renewal Notice (Days)": "", "Party One": "A. Owner", "Party Two": "",
    }


def test_json_uses_column_names_and_null():
    data = json.loads(AgreementMetadata(renewal_notice_days=60).to_json())
    assert data["Renewal Notice (Days)"] == 60
    assert data["Aggrement Value"] is None


def test_metadata_round_trip_and_alias_population():
    m = AgreementMetadata(agreement_value=9000, agreement_start_date="01.04.2010", party_two="B. Tenant")
    assert AgreementMetadata.model_validate_json(m.to_json()) == m
    assert AgreementMetadata.model_validate({"Aggrement Value": "9000"}).agreement_value == 9000


def test_metadata_never_holds_an_impossible_date():
    with pytest.raises(ValidationError):
        AgreementMetadata(agreement_end_date="31.02.2011")
