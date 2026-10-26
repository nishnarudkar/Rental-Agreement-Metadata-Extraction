"""One-shot final evaluation of the frozen pipeline on the test documents.

    python -m src.final_evaluation

Steps, in order:
1. Refuse to run if the frozen configuration changed (``src.freeze``).
2. Ingestion: re-extract every test document's text (DOCX reader / OCR), fresh.
3. Re-check train/test near-duplicates on the fresh text.
4. LLM extraction with the frozen config; schema validation inside the extractor.
5. Write ``outputs/predictions.csv`` (test.csv columns, empty cell = not extracted).
6. Score against ``data/test.csv`` and write ``outputs/test_metrics.json`` and
   ``outputs/test_error_analysis.csv`` (plus full reports in outputs/evaluation/test/).

Nothing here feeds back into the extraction system.
"""

import csv
import json
import time
from dataclasses import asdict

from src.audit import load_cached_texts, near_duplicates
from src.config import EXTRACTED_TEXT_DIR, FILE_NAME_COLUMN, LABEL_COLUMNS, OUTPUTS_DIR, TEST_CSV, TEST_DIR
from src.dataset import list_documents, load_labels
from src.evaluate import EVALUATION_DIR, PREDICTIONS_DIR, evaluate, load_predictions_csv, print_metrics
from src.extract_text import extract_folder, print_summary
from src.extractor import DEFAULT_CONFIG, FINAL_CONFIG_NAME, ExtractionResult, extract
from src.freeze import FROZEN_PATH, fingerprint
from src.llm import LocalLLM

PREDICTIONS_CSV = OUTPUTS_DIR / "predictions.csv"
METRICS_JSON = OUTPUTS_DIR / "test_metrics.json"
ERROR_ANALYSIS_CSV = OUTPUTS_DIR / "test_error_analysis.csv"


def main() -> int:
    timings = {}

    # 1. frozen configuration
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    current = fingerprint()
    changed = sorted(k for k in frozen.keys() | current.keys() if frozen.get(k) != current.get(k))
    if changed:
        print(f"ABORT: configuration changed since freeze: {changed}")
        return 2
    print(f"frozen configuration verified ({FINAL_CONFIG_NAME}: {asdict(DEFAULT_CONFIG)})")

    # 2. ingestion (fresh)
    start = time.perf_counter()
    ingestion = extract_folder(TEST_DIR, EXTRACTED_TEXT_DIR / "test", force=True)
    timings["ingestion_seconds"] = round(time.perf_counter() - start, 1)
    print_summary("test", ingestion)

    # 3. train/test near-duplicates on the fresh text
    texts = {**{f"train/{k}": v for k, v in load_cached_texts("train").items()},
             **{f"test/{k}": v for k, v in load_cached_texts("test").items()}}
    cross = [p for p in near_duplicates(texts, threshold=0.0)
             if p["a"].split("/")[0] != p["b"].split("/")[0]][:3]
    print(f"highest train/test text overlap: {cross[0] if cross else 'n/a'}")

    # 4. extraction
    llm = LocalLLM()
    llm.check()
    out_dir = PREDICTIONS_DIR / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, ExtractionResult] = {}
    start = time.perf_counter()
    for doc_id in list_documents(TEST_DIR):
        text_path = EXTRACTED_TEXT_DIR / "test" / f"{doc_id}.txt"
        if not text_path.exists():  # ingestion failed: report, never skip silently
            results[doc_id] = ExtractionResult(document_id=doc_id, status="failed", error="no extracted text")
        else:
            results[doc_id] = extract(text_path.read_text(encoding="utf-8"), llm, doc_id, DEFAULT_CONFIG)
        r = results[doc_id]
        (out_dir / f"{doc_id}.json").write_text(json.dumps(r.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{r.status}] {doc_id} ({r.attempts} attempt(s), {r.seconds}s){' ' + r.error if r.error else ''}")
    timings["extraction_seconds"] = round(time.perf_counter() - start, 1)

    # 5. predictions.csv
    with open(PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[FILE_NAME_COLUMN, *LABEL_COLUMNS])
        writer.writeheader()
        for doc_id, r in results.items():
            writer.writerow({FILE_NAME_COLUMN: doc_id, **r.metadata.to_csv_row()})

    # 6. scoring
    evaluation = evaluate(load_labels(TEST_CSV), load_predictions_csv(PREDICTIONS_CSV))
    evaluation.save(EVALUATION_DIR / "test")
    print()
    print_metrics(evaluation.metrics)

    with open(ERROR_ANALYSIS_CSV, "w", newline="", encoding="utf-8") as f:
        fields = [*evaluation.per_document[0], "evidence", "evidence_found", "document_status", "document_warnings"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in evaluation.per_document:
            r = results.get(row["document"])
            ev = r.evidence.get(row["field"], {}) if r else {}
            writer.writerow({**row, "evidence": ev.get("evidence"), "evidence_found": ev.get("evidence_found"),
                             "document_status": r.status if r else "missing",
                             "document_warnings": " | ".join(r.warnings) if r else ""})

    report = {
        "split": "test",
        "frozen_config": current,
        "documents": len(results),
        "failed_documents": {d: r.error for d, r in results.items() if r.status != "ok"},
        "ingestion": ingestion,
        "highest_train_test_text_overlap": cross,
        "timings": {**timings,
                    "per_document_extraction_seconds": {d: r.seconds for d, r in results.items()},
                    "per_document_ingestion_seconds": {d: json.loads(
                        (EXTRACTED_TEXT_DIR / "test" / "manifest.json").read_text(encoding="utf-8"))[d].get("seconds")
                        for d in results}},
        "tokens": {d: [r.prompt_tokens, r.output_tokens] for d, r in results.items()},
        "warnings": {d: r.warnings for d, r in results.items() if r.warnings},
        "metrics": evaluation.metrics,
    }
    METRICS_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {PREDICTIONS_CSV}, {METRICS_JSON}, {ERROR_ANALYSIS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
