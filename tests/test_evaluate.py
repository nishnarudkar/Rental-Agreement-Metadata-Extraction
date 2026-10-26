"""Metric tests on synthetic labels and predictions (no dataset values)."""

import csv
import json

import pandas as pd
import pytest

from src.config import FILE_NAME_COLUMN, LABEL_COLUMNS
from src.evaluate import evaluate, load_predictions, load_predictions_csv

VALUE, START, END, NOTICE, PARTY_ONE, PARTY_TWO = LABEL_COLUMNS


def _labels(*rows):
    return pd.DataFrame(rows, columns=[FILE_NAME_COLUMN, *LABEL_COLUMNS], dtype=str)


def _pred(value="", start="", end="", notice="", one="", two=""):
    return dict(zip(LABEL_COLUMNS, [value, start, end, notice, one, two]))


A = ["doc-a", "5000", "01.01.2020", "30.11.2020", "30", "Alice Owner", "Bob Tenant"]
B = ["doc-b", "7000", "01.02.2021", "31.12.2021", "60", "Carol Lessor", "Dan Lessee"]


def test_perfect_predictions():
    preds = {"doc-a": _pred(*A[1:]), "doc-b": _pred(*B[1:])}
    m = evaluate(_labels(A, B), preds).metrics
    assert all(f["raw_recall"] == 1.0 and f["denominator"] == 2 for f in m["fields"].values())
    assert m["macro_raw_recall"] == m["micro_raw_recall"] == 1.0


def test_partial_recall_per_field_and_macro():
    preds = {
        "doc-a": _pred("5000", "01.01.2020", "01.12.2020", "30", "Alice Owner", "Bob Tenant"),  # end wrong
        "doc-b": _pred("7000", "", "31.12.2021", "90", "Carol Lessor", "Dan Lessee"),  # start missing, notice wrong
    }
    m = evaluate(_labels(A, B), preds).metrics
    recalls = {f: v["raw_recall"] for f, v in m["fields"].items()}
    assert recalls == {VALUE: 1.0, START: 0.5, END: 0.5, NOTICE: 0.5, PARTY_ONE: 1.0, PARTY_TWO: 1.0}
    assert m["fields"][START]["not_extracted"] == 1
    assert m["macro_raw_recall"] == pytest.approx((1 + 0.5 + 0.5 + 0.5 + 1 + 1) / 6, abs=1e-4)
    assert m["micro_raw_recall"] == pytest.approx(9 / 12, abs=1e-4)


def test_document_without_prediction_counts_as_false():
    m = evaluate(_labels(A, B), {"doc-a": _pred(*A[1:])}).metrics
    assert m["documents_without_prediction"] == ["doc-b"]
    assert all(f["raw_true"] == 1 and f["denominator"] == 2 for f in m["fields"].values())
    assert all(f["not_extracted"] == 1 for f in m["fields"].values())


def test_blank_reference_is_excluded_from_denominator():
    a_blank_notice = A[:4] + ["  "] + A[5:]
    preds = {"doc-a": _pred(*A[1:]), "doc-b": _pred(*B[1:])}
    result = evaluate(_labels(a_blank_notice, B), preds)
    notice = result.metrics["fields"][NOTICE]
    assert (notice["denominator"], notice["raw_true"], notice["blank_reference"]) == (1, 1, 1)
    row = next(r for r in result.per_document if r["document"] == "doc-a" and r["field"] == NOTICE)
    assert row["outcome"] == "blank reference (not counted)"


def test_raw_is_verbatim_and_normalized_is_separate():
    padded = ["doc-a", "5000 ", "1.1.2020", "30.11.2020", "30", " MR. Alice Owner", "Bob Tenant "]
    preds = {"doc-a": _pred("5000", "01.01.2020", "30.11.2020", "30", "Alice Owner", "Bob Tenant")}
    fields = evaluate(_labels(padded), preds).metrics["fields"]
    assert [fields[c]["raw_true"] for c in LABEL_COLUMNS] == [0, 0, 1, 1, 0, 0]
    assert [fields[c]["normalized_true"] for c in LABEL_COLUMNS] == [1, 1, 1, 1, 1, 1]


def test_outcome_labels():
    padded = ["doc-a", "5000", "01.01.2020", "30.11.2020", "30", " Alice Owner", "Bob Tenant"]
    preds = {"doc-a": _pred("5000", "", "01.12.2020", "30", "Alice Owner", "Bob Tenant")}
    outcomes = {r["field"]: r["outcome"] for r in evaluate(_labels(padded), preds).per_document}
    assert outcomes == {
        VALUE: "exact match", START: "not extracted", END: "mismatch", NOTICE: "exact match",
        PARTY_ONE: "match after normalisation only", PARTY_TWO: "exact match",
    }


def test_invalid_reference_date_is_scored_as_is_and_flagged():
    impossible = A[:3] + ["31.11.2020"] + A[4:]
    preds = {"doc-a": _pred("5000", "01.01.2020", "30.11.2020", "30", "Alice Owner", "Bob Tenant")}
    result = evaluate(_labels(impossible), preds)
    end = result.metrics["fields"][END]
    assert (end["raw_true"], end["normalized_true"], end["unattainable_reference"]) == (0, 0, 1)
    row = next(r for r in result.per_document if r["field"] == END)
    assert row["reference"] == "31.11.2020"  # never corrected


def test_empty_prediction_never_matches_after_normalization():
    preds = {"doc-a": _pred(value="  ")}
    assert evaluate(_labels(A), preds).metrics["fields"][VALUE]["normalized_true"] == 0


def test_exclusions_are_reported_and_not_scored():
    preds = {"doc-a": _pred(*A[1:]), "doc-x": _pred()}
    m = evaluate(_labels(A, B), preds, exclude={"doc-b": "document belongs to another split"}).metrics
    assert m["documents_scored"] == 1
    assert m["excluded"] == {"doc-b": "document belongs to another split"}
    assert m["predictions_without_label"] == ["doc-x"]
    assert m["macro_raw_recall"] == 1.0


def test_field_with_only_blank_references_is_left_out_of_macro():
    a = A[:4] + [""] + A[5:]
    m = evaluate(_labels(a), {"doc-a": _pred(*A[1:])}).metrics
    assert m["fields"][NOTICE]["raw_recall"] is None
    assert m["macro_raw_recall"] == 1.0  # averaged over the 5 fields that have a denominator


def test_duplicate_label_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(_labels(A, A), {})


def test_label_ids_are_matched_after_trimming():
    m = evaluate(_labels([" doc-a "] + A[1:]), {"doc-a": _pred(*A[1:])}).metrics
    assert m["macro_raw_recall"] == 1.0 and m["documents_without_prediction"] == []


# --- loading and reports -----------------------------------------------------------------


def _write_prediction(folder, doc, prediction, document_id=None):
    data = {"document_id": document_id or doc, "status": "ok", "prediction": prediction}
    (folder / f"{doc}.json").write_text(json.dumps(data), encoding="utf-8")


def test_load_predictions_from_extractor_files(tmp_path):
    _write_prediction(tmp_path, "doc-a", {VALUE: 5000, START: "01.01.2020", END: None, NOTICE: 30,
                                          PARTY_ONE: "Alice", PARTY_TWO: None})
    assert load_predictions(tmp_path) == {"doc-a": _pred("5000", "01.01.2020", "", "30", "Alice", "")}


def test_mismatched_document_id_is_rejected(tmp_path):
    _write_prediction(tmp_path, "doc-a", {}, document_id="doc-b")
    with pytest.raises(ValueError, match="does not match the file name"):
        load_predictions(tmp_path)


def test_load_predictions_csv(tmp_path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame([A, B], columns=[FILE_NAME_COLUMN, *LABEL_COLUMNS]).to_csv(path, index=False)
    assert load_predictions_csv(path)["doc-b"] == _pred(*B[1:])


def test_reports_are_written(tmp_path):
    result = evaluate(_labels(A, B), {"doc-a": _pred(*A[1:])})
    result.save(tmp_path)
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")) == result.metrics
    with open(tmp_path / "metrics.csv", encoding="utf-8") as f:
        assert [r["field"] for r in csv.DictReader(f)] == LABEL_COLUMNS
    with open(tmp_path / "per_document.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2 * len(LABEL_COLUMNS)
