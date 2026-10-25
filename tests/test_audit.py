import pandas as pd
import pytest

from src.audit import (
    check_date,
    file_matching,
    identical_label_rows,
    issue_counts,
    label_issues,
    label_support,
    near_duplicates,
    text_similarity,
    value_contexts,
)
from src.config import FILE_NAME_COLUMN, LABEL_COLUMNS


def _labels(rows):
    return pd.DataFrame(rows, columns=[FILE_NAME_COLUMN, *LABEL_COLUMNS], dtype=str)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("01.04.2010", None),
        ("31.02.2011", "impossible calendar date"),
        ("2010-04-01", "not in DD.MM.YYYY format"),
        ("", None),
        ("01.04.2010 ", None),
    ],
)
def test_check_date(value, expected):
    assert check_date(value) == expected


def test_label_issues_flags_without_correcting():
    labels = _labels([["doc-a", "5000 ", "31.11.2009", "01.12.2010", "", " Alice", ".Bob"]])
    found = {(i["column"], i["issue"]): i["value"] for i in label_issues(labels)}
    assert found[(LABEL_COLUMNS[0], "leading/trailing whitespace")] == "5000 "
    assert found[(LABEL_COLUMNS[1], "impossible calendar date")] == "31.11.2009"
    assert found[(LABEL_COLUMNS[3], "missing")] == ""
    assert found[(LABEL_COLUMNS[4], "leading/trailing whitespace")] == " Alice"
    assert found[(LABEL_COLUMNS[5], "starts with punctuation")] == ".Bob"
    # The source frame is untouched.
    assert labels.iloc[0, 1] == "5000 "


def test_issue_counts_contain_no_values():
    labels = _labels([["doc-a", "x", "31.11.2009", "", "", "A", "B"]])
    counts = issue_counts(label_issues(labels))
    assert counts[LABEL_COLUMNS[1]] == {"impossible calendar date": 1}
    assert "31.11.2009" not in str(counts)


def test_file_matching(tmp_path):
    (tmp_path / "doc-a.docx").write_bytes(b"")
    (tmp_path / "doc-c.pdf.docx").write_bytes(b"")
    labels = _labels([["doc-a", *[""] * 6], ["doc-b", *[""] * 6], ["doc-a", *[""] * 6]])
    result = file_matching(labels, tmp_path)
    assert result["matched"] == ["doc-a"]
    assert result["label_without_file"] == ["doc-b"]
    assert result["file_without_label"] == ["doc-c"]
    assert result["duplicate_ids_in_csv"] == ["doc-a"]


def test_identical_label_rows():
    labels = _labels([["a", "1", "", "", "", "X", "Y"], ["b", "1", "", "", "", "X ", "Y"], ["c", "2", "", "", "", "X", "Y"]])
    assert identical_label_rows(labels) == [["a", "b"]]


def test_text_similarity_and_near_duplicates():
    a = "the tenant shall pay the monthly rent on time every month"
    assert text_similarity(a, a) == 1.0
    assert text_similarity(a, "completely different words appear in this other sentence here") == 0.0
    pairs = near_duplicates({"x": a, "y": a + " extra", "z": "unrelated text about something else entirely"})
    assert [(p["a"], p["b"]) for p in pairs] == [("x", "y")]


def test_label_support_detects_swapped_labels():
    texts = {
        "doc-1": "Agreement between Asha Ramesh and Deepthi Kiran. Rent Rs.5,500 per month.",
        "doc-2": "Agreement between Parthasarathy and Veerabrahmam Bathini. Rent Rs.8,000 per month.",
    }
    labels = _labels([
        ["doc-1", "8000", "", "", "", "Parthasarathy", "Veerabrahmam Bathini"],
        ["doc-2", "8000", "", "", "", "Parthasarathy", "Veerabrahmam Bathini"],
    ])
    support = {r["document"]: r for r in label_support(labels, texts)}
    assert support["doc-1"]["fits_other_document_better"] is True
    assert support["doc-1"][LABEL_COLUMNS[0]] == {"found_in_own_text": False, "found_in": ["doc-2"]}
    assert support["doc-2"]["fits_other_document_better"] is False


def test_value_contexts_ignores_longer_numbers():
    texts = {"doc-1": "deposit of Rs.40,000 and a monthly rent of Rs.4,000.00 only"}
    labels = _labels([["doc-1", "4000", "", "", "", "", ""]])
    (ctx,) = value_contexts(labels, texts, window=3)
    assert ctx["snippets"] == ["rent of rs 4000 00 only"]
