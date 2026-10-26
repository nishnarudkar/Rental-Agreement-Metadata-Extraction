"""Predict the six metadata fields for one agreement or a folder of agreements.

    python -m src.predict --input data/test --output outputs/predictions.csv
    python -m src.predict --input data/test/24158401-Rental-Agreement.png --output outputs/one.csv

Reads ``--input`` (a .docx / .png file, or every such file in a folder; other
files are reported), extracts
text (DOCX reader or OCR), runs the frozen AI extractor, and writes a CSV with
the same columns as the dataset's label files ("File Name" + six fields). A
field the system cannot support from the document is left empty.

Per-document details (evidence quotes, warnings, errors) are written next to
the extracted text in ``--work-dir`` (default: outputs/extracted_text/<folder
name>, using the file's folder for a single file); extracted text found there
is reused, so OCR runs only once.
The command exits with status 1 if any document failed.

No labels are read: this works on any folder of agreements.
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

from src.config import EXTRACTED_TEXT_DIR, FILE_NAME_COLUMN, LABEL_COLUMNS
from src.dataset import document_id
from src.document_reader import READERS
from src.extract_text import extract_folder, print_summary
from src.extractor import DEFAULT_CONFIG, FINAL_CONFIG_NAME, ExtractionResult, extract
from src.llm import LocalLLM


def predict_folder(input_path: Path, output_csv: Path, work_dir: Path, llm: LocalLLM | None = None,
                   force: bool = False) -> dict[str, ExtractionResult]:
    """Run ingestion + extraction on a folder or a single file and write ``output_csv``."""
    folder, files = (input_path.parent, [input_path]) if input_path.is_file() else (input_path, None)
    summary = extract_folder(folder, work_dir, force=force, files=files)
    print_summary(input_path.name, summary)

    llm = llm or LocalLLM()
    llm.check()
    results: dict[str, ExtractionResult] = {}
    candidates = files if files is not None else [p for p in folder.iterdir() if p.is_file()]
    documents = sorted(p for p in candidates if p.suffix.lower() in READERS)
    for path in documents:
        doc_id = document_id(path)
        text_path = work_dir / f"{doc_id}.txt"
        if not text_path.exists():
            error = summary["failed_documents"].get(doc_id, "no extracted text")
            results[doc_id] = ExtractionResult(document_id=doc_id, status="failed", error=f"ingestion failed: {error}")
        else:
            results[doc_id] = extract(text_path.read_text(encoding="utf-8"), llm, doc_id, DEFAULT_CONFIG)
        r = results[doc_id]
        (work_dir / f"{doc_id}.prediction.json").write_text(
            json.dumps(r.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{r.status}] {doc_id} ({r.seconds}s){': ' + r.error if r.error else ''}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[FILE_NAME_COLUMN, *LABEL_COLUMNS])
        writer.writeheader()
        for doc_id, r in results.items():
            writer.writerow({FILE_NAME_COLUMN: doc_id, **r.metadata.to_csv_row()})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="a .docx / .png agreement, or a folder of them")
    parser.add_argument("--output", type=Path, required=True, help="predictions CSV to write")
    parser.add_argument("--work-dir", type=Path, help="where extracted text and per-document JSON go")
    parser.add_argument("--force", action="store_true", help="re-extract text even if cached")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"input not found: {args.input}")
    if args.input.is_file() and args.input.suffix.lower() not in READERS:
        parser.error(f"unsupported file type '{args.input.suffix}' (expected .docx or .png)")

    folder = args.input.parent if args.input.is_file() else args.input
    work_dir = args.work_dir or EXTRACTED_TEXT_DIR / folder.name
    work_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(work_dir / "predict.log", encoding="utf-8")])

    print(f"configuration: {FINAL_CONFIG_NAME} (frozen), model {LocalLLM().config.model}")
    start = time.perf_counter()
    results = predict_folder(args.input, args.output, work_dir, force=args.force)
    failed = [d for d, r in results.items() if r.status != "ok"]
    print(f"\n{len(results)} document(s), {len(failed)} failed, {time.perf_counter() - start:.0f}s -> {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
