"""Per-field recall of predictions against reference labels.

Definition (assignment brief, per field):

    Recall = True / (True + False)
    True   = prediction exactly matches the reference value
    False  = prediction differs, or nothing was extracted

Scoring rules:

- **Denominator**: documents under evaluation whose reference cell for the
  field is not blank. A blank reference has nothing to recall; it is excluded
  and counted in ``blank_reference``.
- **Missing predictions** (no prediction file, failed extraction, null value)
  count as False. Documents are never skipped silently.
- **Raw exact match**: prediction cell == reference cell, character for
  character. The reference is used exactly as stored (no trimming).
- **Normalised match** (reported separately): both sides passed through
  ``src.normalize`` (whitespace, number formatting, date padding, name titles
  and punctuation). An empty prediction never matches.
- Reference values are never corrected. Impossible reference dates are scored
  as they are and counted in ``unattainable_reference``.
- Label rows can be excluded only explicitly (``exclude``), and every
  exclusion is listed in the report with its reason.

    python -m src.evaluate --split train
    python -m src.evaluate --split train --documents 47854715-RENTAL-AGREEMENT 36199312-Rental-Agreement
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.audit import DATE_COLUMNS, check_date
from src.config import FILE_NAME_COLUMN, LABEL_COLUMNS, OUTPUTS_DIR, TEST_CSV, TEST_DIR, TRAIN_CSV, TRAIN_DIR
from src.dataset import list_documents, load_labels
from src.normalize import normalize

EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"


# --------------------------------------------------------------------------- loading


def _cell(value) -> str:
    return "" if value is None else str(value)


def load_predictions(folder: Path) -> dict[str, dict[str, str]]:
    """Read ``<id>.json`` files written by ``src.extractor`` as CSV-style cells.

    The ID inside each file must match its file name; failed extractions keep
    their (empty) prediction so they are scored as "not extracted".
    """
    predictions = {}
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("document_id") != path.stem:
            raise ValueError(f"{path.name}: document_id {data.get('document_id')!r} does not match the file name")
        predictions[path.stem] = {c: _cell(data["prediction"].get(c)) for c in LABEL_COLUMNS}
    return predictions


def load_predictions_csv(path: Path) -> dict[str, dict[str, str]]:
    """Read a predictions CSV with the same columns as the label files."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    ids = frame[FILE_NAME_COLUMN].str.strip()
    if ids.duplicated().any():
        raise ValueError(f"{path.name}: duplicate document IDs {sorted(set(ids[ids.duplicated()]))}")
    return {doc: {c: row[c] for c in LABEL_COLUMNS} for doc, (_, row) in zip(ids, frame.iterrows())}


# --------------------------------------------------------------------------- scoring


@dataclass
class Evaluation:
    per_document: list[dict]
    metrics: dict

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            rows = [{"field": field, **{k: v for k, v in m.items() if k != "definition"}}
                    for field, m in self.metrics["fields"].items()]
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with open(out_dir / "per_document.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.per_document[0]))
            writer.writeheader()
            writer.writerows(self.per_document)


def _recall(true: int, total: int) -> float | None:
    return round(true / total, 4) if total else None


def evaluate(
    labels: pd.DataFrame,
    predictions: dict[str, dict[str, str]],
    exclude: dict[str, str] | None = None,
) -> Evaluation:
    """Score ``predictions`` against ``labels`` (raw strings, as loaded by ``load_labels``)."""
    exclude = exclude or {}
    ids = labels[FILE_NAME_COLUMN].str.strip()
    if ids.duplicated().any():
        raise ValueError(f"duplicate document IDs in labels: {sorted(set(ids[ids.duplicated()]))}")

    rows, scored_ids = [], []
    for doc, (_, label) in zip(ids, labels.iterrows()):
        if doc in exclude:
            continue
        scored_ids.append(doc)
        predicted = predictions.get(doc)
        for column in LABEL_COLUMNS:
            reference = label[column]
            prediction = predicted[column] if predicted else ""
            counted = bool(reference.strip())
            raw = counted and prediction == reference
            norm = counted and prediction.strip() != "" and normalize(column, prediction) == normalize(column, reference)
            if not counted:
                outcome = "blank reference (not counted)"
            elif predicted is None:
                outcome = "no prediction for document"
            elif not prediction.strip():
                outcome = "not extracted"
            elif raw:
                outcome = "exact match"
            elif norm:
                outcome = "match after normalisation only"
            else:
                outcome = "mismatch"
            unattainable = counted and column in DATE_COLUMNS and check_date(reference) is not None
            rows.append({
                "document": doc, "field": column, "reference": reference, "prediction": prediction,
                "counted": counted, "raw_match": raw, "normalized_match": norm, "outcome": outcome,
                "unattainable_reference": unattainable,
            })

    fields = {}
    for column in LABEL_COLUMNS:
        cells = [r for r in rows if r["field"] == column]
        counted = [r for r in cells if r["counted"]]
        raw_true = sum(r["raw_match"] for r in counted)
        norm_true = sum(r["normalized_match"] for r in counted)
        fields[column] = {
            "denominator": len(counted),
            "raw_true": raw_true,
            "raw_recall": _recall(raw_true, len(counted)),
            "normalized_true": norm_true,
            "normalized_recall": _recall(norm_true, len(counted)),
            "not_extracted": sum(r["outcome"] in ("not extracted", "no prediction for document") for r in counted),
            "blank_reference": len(cells) - len(counted),
            "unattainable_reference": sum(r["unattainable_reference"] for r in counted),
            "definition": f"{len(counted)} documents with a non-blank reference for {column!r}; "
                          f"{len(cells) - len(counted)} blank reference(s) excluded",
        }

    def macro(key):
        values = [m[key] for m in fields.values() if m[key] is not None]
        return round(sum(values) / len(values), 4) if values else None

    def micro(key):
        total = sum(m["denominator"] for m in fields.values())
        return _recall(sum(m[key] for m in fields.values()), total)

    metrics = {
        "documents_scored": len(scored_ids),
        "documents_without_prediction": sorted(d for d in scored_ids if d not in predictions),
        "predictions_without_label": sorted(set(predictions) - set(ids)),
        "excluded": dict(sorted(exclude.items())),
        "fields": fields,
        "macro_raw_recall": macro("raw_recall"),
        "macro_normalized_recall": macro("normalized_recall"),
        "micro_raw_recall": micro("raw_true"),
        "micro_normalized_recall": micro("normalized_true"),
    }
    return Evaluation(per_document=rows, metrics=metrics)


# --------------------------------------------------------------------------- CLI


def print_metrics(metrics: dict) -> None:
    print(f"documents scored: {metrics['documents_scored']}")
    for key in ("excluded", "documents_without_prediction", "predictions_without_label"):
        if metrics[key]:
            print(f"{key}: {metrics[key]}")
    print(f"\n{'Field':<24}{'n':>3}{'raw':>8}{'norm':>8}  not extracted / blank ref / unattainable")
    for field, m in metrics["fields"].items():
        fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"
        print(f"{field:<24}{m['denominator']:>3}{fmt(m['raw_recall']):>8}{fmt(m['normalized_recall']):>8}"
              f"  {m['not_extracted']} / {m['blank_reference']} / {m['unattainable_reference']}")
    # Three decimals: 0.625 is exactly representable and ".2f" would round it down to 0.62.
    print(f"{'Macro average':<27}{metrics['macro_raw_recall'] or 0:>8.3f}{metrics['macro_normalized_recall'] or 0:>8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--predictions", type=Path, help="predictions folder or CSV (default: outputs/predictions/<split>)")
    parser.add_argument("--documents", nargs="*", help="score only these document IDs")
    parser.add_argument("--exclude", nargs="*", default=[], help="label rows to exclude (reported in the output)")
    parser.add_argument("--out", type=Path, help="report folder (default: outputs/evaluation/<split>)")
    parser.add_argument("--final", action="store_true", help="required for --split test")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("evaluating test requires --final (test labels are reserved for final evaluation)")

    csv_path, folder = {"train": (TRAIN_CSV, TRAIN_DIR), "test": (TEST_CSV, TEST_DIR)}[args.split]
    labels = load_labels(csv_path)
    source = args.predictions or PREDICTIONS_DIR / args.split
    predictions = load_predictions_csv(source) if source.suffix == ".csv" else load_predictions(source)

    available = set(list_documents(folder))
    exclude = {doc: f"document file is not in {folder.name}/" for doc in labels[FILE_NAME_COLUMN].str.strip()
               if doc not in available}
    exclude.update({doc: "excluded on the command line" for doc in args.exclude})
    if args.documents:
        exclude.update({doc: "not selected (--documents)" for doc in labels[FILE_NAME_COLUMN].str.strip()
                        if doc not in args.documents and doc not in exclude})

    result = evaluate(labels, predictions, exclude)
    out_dir = args.out or EVALUATION_DIR / args.split
    result.save(out_dir)
    print_metrics(result.metrics)
    print(f"\nreports: {out_dir}/metrics.json, metrics.csv, per_document.csv")


if __name__ == "__main__":
    main()
