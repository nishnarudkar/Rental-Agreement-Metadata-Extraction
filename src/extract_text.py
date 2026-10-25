"""Document ingestion pipeline: convert every document in a split to text.

    python -m src.extract_text --split train
    python -m src.extract_text --split all --force

For each split, writes to ``outputs/extracted_text/<split>/``:

- ``<document id>.txt``: extracted text (deterministic name from the CSV ID)
- ``<document id>.ocr.json``: OCR'd images only; every line, confidence, box
- ``manifest.json``: one entry per file with status, type, length, warnings, timing, error
- ``summary.json``: success/failure counts for the latest run

A run log goes to ``outputs/extracted_text/extraction.log``. A failing document
never stops the run: it is recorded with its error and the run continues; the
command exits with status 1 if anything failed. Cached text is reused unless
``--force`` is given, because OCR is slow on CPU.
"""

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

from src.config import EXTRACTED_TEXT_DIR, TEST_DIR, TRAIN_DIR
from src.dataset import document_id
from src.document_reader import READERS, read_document

SPLIT_DIRS = {"train": TRAIN_DIR, "test": TEST_DIR}

log = logging.getLogger(__name__)


def _outputs_for(out_dir: Path, doc_id: str) -> list[Path]:
    return [out_dir / f"{doc_id}.txt", out_dir / f"{doc_id}.ocr.json"]


def _process(path: Path, out_dir: Path) -> dict:
    """Extract one document and save its outputs; never raises."""
    doc_id = document_id(path)
    try:
        result = read_document(path)
    except Exception as exc:  # DocumentReadError or anything unexpected
        log.exception("failed: %s", path.name)
        for stale in _outputs_for(out_dir, doc_id):  # never leave old text looking valid
            stale.unlink(missing_ok=True)
        return {
            "source": path.name,
            "status": "failed",
            "document_type": path.suffix.lower().lstrip("."),
            "error": getattr(exc, "reason", None) or f"{type(exc).__name__}: {exc}",
        }

    txt_path, ocr_path = _outputs_for(out_dir, doc_id)
    txt_path.write_text(result.text, encoding="utf-8")
    if result.diagnostics is not None:
        ocr_path.write_text(json.dumps(result.diagnostics, indent=2), encoding="utf-8")
    else:
        ocr_path.unlink(missing_ok=True)
    log.info("%s: %s, %d chars, %.2fs", path.name, result.document_type, len(result.text), result.seconds)
    return {
        "source": result.filename,
        "status": "ok" if result.text.strip() else "empty",
        "document_type": result.document_type,
        "chars": len(result.text),
        "lines": result.text.count("\n") + 1 if result.text else 0,
        "mean_ocr_confidence": result.mean_confidence,
        "warnings": result.warnings,
        "seconds": result.seconds,
    }


def extract_folder(folder: Path, out_dir: Path, force: bool = False) -> dict:
    """Process every file in ``folder``; return a run summary.

    Unsupported files (anything but .docx/.png) are reported, not skipped
    silently. Hidden files (names starting with ".") are ignored.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    run: dict[str, str] = {}  # document id -> how this run handled it
    start = time.perf_counter()

    for path in sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")):
        doc_id = document_id(path)
        if path.suffix.lower() not in READERS:
            manifest[doc_id] = {"source": path.name, "status": "unsupported", "error": f"unsupported file type '{path.suffix}'"}
            run[doc_id] = "unsupported"
            log.warning("unsupported file: %s", path.name)
            print(f"[unsupported] {path.name}")
            continue
        cached = manifest.get(doc_id, {})
        if not force and cached.get("status") in ("ok", "empty") and (out_dir / f"{doc_id}.txt").exists():
            run[doc_id] = "cached"
            print(f"[cached] {doc_id}")
            continue

        entry = _process(path, out_dir)
        manifest[doc_id] = entry
        run[doc_id] = entry["status"]
        if entry["status"] == "failed":
            print(f"[FAILED] {path.name}: {entry['error']}")
        else:
            print(f"[{entry['document_type']}] {doc_id}: {entry['chars']} chars in {entry['seconds']}s")
            for warning in entry["warnings"]:
                print(f"    warning: {warning}")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    counts = Counter(run.values())
    summary = {
        "folder": folder.name,
        "files": len(run),
        "processed": counts["ok"] + counts["empty"] + counts["failed"],
        "succeeded": counts["ok"] + counts["empty"],
        "cached": counts["cached"],
        "empty_text": counts["empty"],
        "failed": counts["failed"],
        "unsupported": counts["unsupported"],
        "by_type": dict(Counter(manifest[d]["document_type"] for d in run if run[d] != "unsupported")),
        "with_warnings": sum(bool(manifest[d].get("warnings")) for d in run),
        "failed_documents": {d: manifest[d]["error"] for d in run if run[d] in ("failed", "unsupported")},
        "seconds": round(time.perf_counter() - start, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(split: str, summary: dict) -> None:
    print(f"\n=== {split}: {summary['files']} files in {summary['seconds']}s ===")
    print(f"  succeeded {summary['succeeded']} (empty text: {summary['empty_text']}), cached {summary['cached']}, "
          f"failed {summary['failed']}, unsupported {summary['unsupported']}")
    print(f"  by type {summary['by_type']}, documents with warnings: {summary['with_warnings']}")
    for doc, error in summary["failed_documents"].items():
        print(f"  ! {doc}: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = parser.parse_args()
    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(EXTRACTED_TEXT_DIR / "extraction.log", encoding="utf-8")],
    )
    problems = 0
    for split in ["train", "test"] if args.split == "all" else [args.split]:
        summary = extract_folder(SPLIT_DIRS[split], EXTRACTED_TEXT_DIR / split, force=args.force)
        print_summary(split, summary)
        problems += summary["failed"] + summary["unsupported"]
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
