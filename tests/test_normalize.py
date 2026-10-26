import pytest

from src.config import LABEL_COLUMNS
from src.normalize import normalize, normalize_date, normalize_number, normalize_party

VALUE, START, END, NOTICE, PARTY_ONE, PARTY_TWO = LABEL_COLUMNS


@pytest.mark.parametrize("raw", ["6500", " 6500 ", "6,500", "6500.00", "6500.0", 6500])
def test_numbers(raw):
    assert normalize_number(raw) == "6500"


def test_number_fallback_keeps_text():
    assert normalize_number(" Sixty ") == "sixty"


@pytest.mark.parametrize("raw", ["01.04.2010", "1.4.2010", " 01.04.2010 ", "1/4/2010", "01-04-2010"])
def test_dates(raw):
    assert normalize_date(raw) == "01.04.2010"


def test_impossible_reference_date_is_not_corrected():
    assert normalize_date("31.02.2011") == "31.02.2011"


@pytest.mark.parametrize(
    "a, b",
    [
        ("MR.K.Kuttan ", "K. Kuttan"),
        (" P C MATHEW", "p.c. mathew"),
        ("Asha Ramesh & Ramesh K.N", "Mrs.Asha Ramesh and Mr.Ramesh.K.N."),
        ("Prof. K. Parthasarathy", "K Parthasarathy"),
    ],
)
def test_party_variants_match(a, b):
    assert normalize_party(a) == normalize_party(b)


@pytest.mark.parametrize("a, b", [("Deepthi", "Deepti"), ("A. Rao", "B. Rao"), ("Rao Kumar", "Kumar Rao")])
def test_party_spelling_and_order_still_matter(a, b):
    assert normalize_party(a) != normalize_party(b)


def test_missing_values_normalise_to_empty():
    assert {normalize(column, None) for column in LABEL_COLUMNS} == {""}
    assert normalize(VALUE, "  ") == ""


def test_dispatch_by_column():
    assert normalize(VALUE, "4,000") == "4000"
    assert normalize(START, "5.12.2008") == "05.12.2008"
    assert normalize(NOTICE, "90 ") == "90"
    assert normalize(PARTY_TWO, "L GOPINATH ") == "l gopinath"
    with pytest.raises(KeyError):
        normalize("Unknown Column", "x")
