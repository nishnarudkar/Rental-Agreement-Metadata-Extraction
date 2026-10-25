"""Load label CSVs and pair each row with its document file.

This module only handles file bookkeeping (which document belongs to which
label row). It does not look inside documents and plays no part in metadata
extraction.
"""

from pathlib import Path

import pandas as pd

from src.config import (
    FILE_NAME_COLUMN,
    LABEL_COLUMNS,
    SUPPORTED_EXTENSIONS,
    TEST_CSV,
    TEST_DIR,
    TRAIN_CSV,
    TRAIN_DIR,
)


def document_id(path: Path) -> str:
    """Return the ID used in the CSV "File Name" column for a document path.

    Some test files carry a double extension such as ``.pdf.docx`` while the
    CSV stores the name without either, so both suffixes are dropped.
    """
    name = path.name
    for suffix in SUPPORTED_EXTENSIONS:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.lower().endswith(".pdf"):
        name = name[: -len(".pdf")]
    return name


def list_documents(folder: Path) -> dict[str, Path]:
    """Map document ID -> file path for every supported file in ``folder``."""
    return {
        document_id(p): p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    }


def load_labels(csv_path: Path) -> pd.DataFrame:
    """Read a label CSV as raw strings, exactly as stored (no cleaning)."""
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False)


def load_split(split: str) -> pd.DataFrame:
    """Return labels for ``"train"`` or ``"test"`` with a ``path`` column.

    Rows whose document is not in the split's own folder get ``path = None``
    so callers can see (and decide how to handle) the mismatch.
    """
    csv_path, folder = {"train": (TRAIN_CSV, TRAIN_DIR), "test": (TEST_CSV, TEST_DIR)}[split]
    labels = load_labels(csv_path)
    docs = list_documents(folder)
    labels["path"] = labels[FILE_NAME_COLUMN].map(lambda name: docs.get(name.strip()))
    return labels[[FILE_NAME_COLUMN, "path", *LABEL_COLUMNS]]


def leaked_ids() -> set[str]:
    """Document IDs that appear in both the train labels and the test folder.

    These must never be used as few-shot examples or for tuning prompts,
    otherwise test scores would be inflated.
    """
    train_ids = {n.strip() for n in load_labels(TRAIN_CSV)[FILE_NAME_COLUMN]}
    test_ids = set(list_documents(TEST_DIR)) | {
        n.strip() for n in load_labels(TEST_CSV)[FILE_NAME_COLUMN]
    }
    return train_ids & test_ids
