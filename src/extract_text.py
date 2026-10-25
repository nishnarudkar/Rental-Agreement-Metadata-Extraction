"""Convert every document in a split to text and cache it on disk.

    python -m src.extract_text --split train
    python -m src.extract_text --split all --force

Writes ``outputs/extracted_text/<split>/<document id>.txt`` plus a
``manifest.json`` per split (method, length, OCR confidence, timing). Cached
files are reused unless ``--force`` is given, because OCR is slow on CPU.
"""

import argparse
import json
import time

from src.config import EXTRACTED_TEXT_DIR, TEST_DIR, TRAIN_DIR
from src.dataset import list_documents
from src.document_reader import read_document

SPLIT_DIRS = {"train": TRAIN_DIR, "test": TEST_DIR}


def extract_split(split: str, force: bool = False) -> dict:
    out_dir = EXTRACTED_TEXT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    for doc_id, path in list_documents(SPLIT_DIRS[split]).items():
        txt_path = out_dir / f"{doc_id}.txt"
        if txt_path.exists() and doc_id in manifest and not force:
            print(f"[cached] {split}/{doc_id}")
            continue
        start = time.perf_counter()
        result = read_document(path)
        seconds = round(time.perf_counter() - start, 1)
        txt_path.write_text(result.text, encoding="utf-8")
        manifest[doc_id] = {
            "source": result.filename,
            "document_type": result.document_type,
            "chars": len(result.text),
            "lines": result.text.count("\n") + 1,
            "mean_ocr_confidence": result.mean_confidence,
            "warnings": result.warnings,
            "seconds": seconds,
        }
        print(f"[{result.document_type}] {split}/{doc_id}: {len(result.text)} chars in {seconds}s")
        for warning in result.warnings:
            print(f"    warning: {warning}")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = parser.parse_args()
    for split in ["train", "test"] if args.split == "all" else [args.split]:
        extract_split(split, force=args.force)


if __name__ == "__main__":
    main()
