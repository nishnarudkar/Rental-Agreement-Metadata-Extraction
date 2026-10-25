"""Dataset audit: label quality, file matching, duplicates and label support.

These checks inspect the *dataset* (CSV labels and document inventory) before
any model is built. They are never used to extract metadata from a document.

Test labels are audited only structurally (counts, never values) so they stay
unseen until final evaluation.
"""

import json
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.config import EXTRACTED_TEXT_DIR, FILE_NAME_COLUMN, LABEL_COLUMNS
from src.dataset import list_documents, load_labels

VALUE_COLUMN, START_COLUMN, END_COLUMN, NOTICE_COLUMN, PARTY_ONE_COLUMN, PARTY_TWO_COLUMN = LABEL_COLUMNS
DATE_COLUMNS = [START_COLUMN, END_COLUMN]
INTEGER_COLUMNS = [VALUE_COLUMN, NOTICE_COLUMN]
PARTY_COLUMNS = [PARTY_ONE_COLUMN, PARTY_TWO_COLUMN]
LABEL_DATE_FORMAT = "%d.%m.%Y"


# --------------------------------------------------------------------------- CSV structure


def csv_overview(csv_path: Path) -> dict:
    """Shape, exact column names and the dtypes pandas would infer by default."""
    inferred = pd.read_csv(csv_path)
    return {
        "file": csv_path.name,
        "rows": int(inferred.shape[0]),
        "columns": list(inferred.columns),
        "inferred_dtypes": {c: str(t) for c, t in inferred.dtypes.items()},
    }


def file_matching(labels: pd.DataFrame, folder: Path) -> dict:
    """Pair CSV rows with files in ``folder`` by document ID."""
    ids = [name.strip() for name in labels[FILE_NAME_COLUMN]]
    docs = list_documents(folder)
    return {
        "matched": sorted(i for i in set(ids) if i in docs),
        "label_without_file": sorted(i for i in set(ids) if i not in docs),
        "file_without_label": sorted(set(docs) - set(ids)),
        "duplicate_ids_in_csv": sorted(i for i, n in Counter(ids).items() if n > 1),
        "file_formats": dict(Counter(p.suffix.lower() for p in docs.values())),
    }


# --------------------------------------------------------------------------- label quality


def check_date(value: str) -> str | None:
    """Return a problem description for a label date, or None if it is valid."""
    stripped = value.strip()
    if not stripped:
        return None  # missing values are reported separately
    try:
        datetime.strptime(stripped, LABEL_DATE_FORMAT)
    except ValueError:
        parts = stripped.split(".")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return "impossible calendar date"
        return "not in DD.MM.YYYY format"
    return None


def label_issues(labels: pd.DataFrame) -> list[dict]:
    """One record per problematic label cell (includes the value)."""
    issues = []
    for _, row in labels.iterrows():
        doc = row[FILE_NAME_COLUMN].strip()
        for column in LABEL_COLUMNS:
            value = row[column]
            if not value.strip():
                issues.append({"document": doc, "column": column, "issue": "missing", "value": value})
                continue
            if value != value.strip():
                issues.append({"document": doc, "column": column, "issue": "leading/trailing whitespace", "value": value})
            if column in DATE_COLUMNS and (problem := check_date(value)):
                issues.append({"document": doc, "column": column, "issue": problem, "value": value})
            if column in INTEGER_COLUMNS and not value.strip().isdigit():
                issues.append({"document": doc, "column": column, "issue": "not a whole number", "value": value})
            if column in PARTY_COLUMNS and not value.strip()[0].isalnum():
                issues.append({"document": doc, "column": column, "issue": "starts with punctuation", "value": value})
    return issues


def issue_counts(issues: list[dict]) -> dict:
    """Value-free summary: {column: {issue: count}}. Safe to compute on test labels."""
    counts: dict[str, Counter] = {}
    for item in issues:
        counts.setdefault(item["column"], Counter())[item["issue"]] += 1
    return {column: dict(c) for column, c in counts.items()}


def identical_label_rows(labels: pd.DataFrame) -> list[list[str]]:
    """Groups of documents whose six labels are identical (likely duplicates)."""
    keyed = labels.assign(_key=labels[LABEL_COLUMNS].apply(lambda r: tuple(v.strip() for v in r), axis=1))
    groups = keyed.groupby("_key")[FILE_NAME_COLUMN].apply(lambda s: sorted(n.strip() for n in s))
    return [g for g in groups if len(g) > 1]


# --------------------------------------------------------------------------- document text


def load_cached_texts(split: str) -> dict[str, str]:
    """Text produced by ``python -m src.extract_text`` (empty if not run yet)."""
    folder = EXTRACTED_TEXT_DIR / split
    if not folder.exists():
        return {}
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.txt"))}


def _tokens(text: str) -> list[str]:
    """Lower-case alphanumeric tokens; thousands separators are removed first."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.replace(",", "").casefold())
    return cleaned.split()


def text_similarity(a: str, b: str, n: int = 3) -> float:
    """Share of the shorter document's word n-grams that also occur in the other.

    Containment rather than Jaccard, so a single page extracted from a longer
    contract still scores high (e.g. 6683127 is page 1 of 6683129).
    """
    def shingles(text):
        toks = _tokens(text)
        return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}

    sa, sb = shingles(a), shingles(b)
    return len(sa & sb) / min(len(sa), len(sb)) if sa and sb else 0.0


def near_duplicates(texts: dict[str, str], threshold: float = 0.5) -> list[dict]:
    """Document pairs whose wording overlaps above ``threshold``."""
    pairs = []
    for (a, ta), (b, tb) in combinations(sorted(texts.items()), 2):
        score = text_similarity(ta, tb)
        if score >= threshold:
            pairs.append({"a": a, "b": b, "similarity": round(score, 3)})
    return sorted(pairs, key=lambda p: -p["similarity"])


# --------------------------------------------------------------------------- label support (train only)


def _party_score(label: str, text_tokens: set[str]) -> float:
    """Share of the label's name words (3+ chars) that occur in the document."""
    words = [t for t in _tokens(label) if len(t) >= 3]
    return sum(w in text_tokens for w in words) / len(words) if words else 0.0


def label_support(labels: pd.DataFrame, texts: dict[str, str]) -> list[dict]:
    """Does each label actually appear in its own document, or in another one?

    For every row this scores the party names (word overlap) and the agreement
    value (whole-number token present) against every document, and flags rows
    whose labels fit a *different* document better. Train labels only.
    """
    token_sets = {doc: set(_tokens(t)) for doc, t in texts.items()}
    results = []
    for _, row in labels.iterrows():
        doc = row[FILE_NAME_COLUMN].strip()
        if doc not in token_sets:
            continue
        record = {"document": doc}
        for column in PARTY_COLUMNS:
            scores = {d: _party_score(row[column], toks) for d, toks in token_sets.items()}
            best = max(scores, key=scores.get)
            record[column] = {
                "own_score": round(scores[doc], 2),
                "best_document": best,
                "best_score": round(scores[best], 2),
            }
        value = row[VALUE_COLUMN].strip()
        record[VALUE_COLUMN] = {
            "found_in_own_text": bool(value) and value in token_sets[doc],
            "found_in": sorted(d for d, toks in token_sets.items() if value and value in toks),
        }
        record["fits_other_document_better"] = all(
            record[c]["best_document"] != doc and record[c]["best_score"] > record[c]["own_score"]
            for c in PARTY_COLUMNS
        )
        results.append(record)
    return results


def value_contexts(labels: pd.DataFrame, texts: dict[str, str], window: int = 12) -> list[dict]:
    """Words surrounding each occurrence of the labelled Agreement Value.

    Used to judge by eye whether the label means monthly rent, deposit, etc.
    """
    contexts = []
    for _, row in labels.iterrows():
        doc, value = row[FILE_NAME_COLUMN].strip(), row[VALUE_COLUMN].strip()
        if doc not in texts or not value:
            continue
        toks = _tokens(texts[doc])
        snippets = [
            " ".join(toks[max(0, i - window) : i + window + 1]) for i, tok in enumerate(toks) if tok == value
        ]
        contexts.append({"document": doc, "value": value, "snippets": snippets})
    return contexts


# --------------------------------------------------------------------------- report


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def build_report(train_csv: Path, test_csv: Path, train_dir: Path, test_dir: Path) -> dict:
    train, test = load_labels(train_csv), load_labels(test_csv)
    train_texts, test_texts = load_cached_texts("train"), load_cached_texts("test")

    train_files, test_files = file_matching(train, train_dir), file_matching(test, test_dir)
    train_ids = {n.strip() for n in train[FILE_NAME_COLUMN]}
    test_ids = {n.strip() for n in test[FILE_NAME_COLUMN]} | set(list_documents(test_dir))
    leaked = sorted(train_ids & test_ids)
    # Rows in train.csv that describe a test document are test labels in
    # disguise: audit them by counts only, like the rest of the test labels.
    is_leaked = train[FILE_NAME_COLUMN].str.strip().isin(leaked)
    dev, leaked_rows = train[~is_leaked], train[is_leaked]

    support = label_support(dev, train_texts)
    cross_split = [
        p for p in near_duplicates({f"train/{k}": v for k, v in train_texts.items()}
                                   | {f"test/{k}": v for k, v in test_texts.items()})
        if p["a"].split("/")[0] != p["b"].split("/")[0]
    ]

    return {
        "policy": "Test labels are summarised by counts only; their values are not read here.",
        "train": {
            "overview": csv_overview(train_csv),
            "files": train_files,
            "label_issues": label_issues(dev),
            "leaked_row_issue_counts": issue_counts(label_issues(leaked_rows)),
            "identical_label_rows": identical_label_rows(dev),
            "near_duplicate_documents": near_duplicates(train_texts),
            "label_support": support,
            "agreement_value_contexts": value_contexts(dev, train_texts),
        },
        "test": {
            "overview": csv_overview(test_csv),
            "files": test_files,
            "label_issue_counts": issue_counts(label_issues(test)),
        },
        "cross_split": {
            "train_label_rows_for_test_documents": leaked,
            "near_duplicate_documents": cross_split,
        },
        "development_set": {
            "usable": sorted(
                set(train_files["matched"]) - set(leaked)
            ),
            "excluded": {doc: "document belongs to the test split" for doc in leaked},
            "labels_fit_other_document": sorted(r["document"] for r in support if r["fits_other_document_better"]),
        },
    }
