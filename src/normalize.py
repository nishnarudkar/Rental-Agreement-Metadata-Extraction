"""Evaluation-only normalisation of reference labels and predictions.

Used **only** to compute the optional normalised metric next to the raw
exact-match metric. It never changes stored predictions or the reference CSVs,
and it never repairs invalid values: an impossible reference date such as
31.02.2011 stays 31.02.2011 after normalisation.

Rules per field:

- Amounts and day counts: the whole number they represent
  ("6500", " 6500 ", "6,500", "6500.00" all become "6500").
- Dates: day/month zero-padded, "." as separator ("1/4/2010" becomes "01.04.2010").
  Calendar validity is not checked.
- Party names: case-insensitive. Punctuation becomes spaces, "&" becomes "and",
  courtesy titles (Mr, Mrs, Prof, Sri, ...) are dropped and spaces are collapsed
  ("MR.K.Kuttan " becomes "k kuttan").

Anything that does not fit its field's pattern falls back to trimmed,
lower-cased text, so it can still match an identical value.
"""

from src.config import LABEL_COLUMNS

VALUE_COLUMN, START_COLUMN, END_COLUMN, NOTICE_COLUMN, PARTY_ONE_COLUMN, PARTY_TWO_COLUMN = LABEL_COLUMNS

TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "sri", "shri", "smt", "kum", "mx"}


def _text(value) -> str:
    return "" if value is None else " ".join(str(value).split())


def normalize_number(value) -> str:
    text = _text(value).replace(",", "")
    if text.endswith(".00") or text.endswith(".0"):
        text = text.rsplit(".", 1)[0]
    return str(int(text)) if text.isdigit() else text.casefold()


def normalize_date(value) -> str:
    text = _text(value)
    for sep in "/-":
        text = text.replace(sep, ".")
    parts = text.split(".")
    if len(parts) == 3 and all(p.isdigit() for p in parts) and len(parts[2]) == 4:
        day, month, year = parts
        return f"{int(day):02d}.{int(month):02d}.{year}"
    return text.casefold()


def normalize_party(value) -> str:
    text = _text(value).casefold().replace("&", " and ")
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(token for token in text.split() if token not in TITLES)


NORMALIZERS = {
    VALUE_COLUMN: normalize_number,
    START_COLUMN: normalize_date,
    END_COLUMN: normalize_date,
    NOTICE_COLUMN: normalize_number,
    PARTY_ONE_COLUMN: normalize_party,
    PARTY_TWO_COLUMN: normalize_party,
}


def normalize(column: str, value) -> str:
    """Normalised form of ``value`` for comparison in ``column``."""
    return NORMALIZERS[column](value)
