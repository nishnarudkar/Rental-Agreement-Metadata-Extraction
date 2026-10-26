"""Development experiments on training documents, with a persistent log.

    python -m src.experiment --name v1-baseline --prompt v1
    python -m src.experiment --name v3-final --prompt v3 --holdout

Protocol (fixed before the baseline; see docs/experiments.md):

- Only training documents and training labels are used.
- HOLDOUT documents are not extracted or inspected while iterating. They are
  scored once, for the selected configuration (``--holdout``).
- Two views are scored: "all" (every tuning document) and "clean" (without the
  two train rows whose labels belong to each other's document).

Each run writes predictions and evaluation reports to
``outputs/experiments/<name>/<subset>/`` and appends one line to
``outputs/experiments/log.jsonl``.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime

from src.config import EXTRACTED_TEXT_DIR, FILE_NAME_COLUMN, OUTPUTS_DIR, TRAIN_CSV
from src.dataset import leaked_ids, load_labels
from src.evaluate import evaluate, load_predictions, print_metrics
from src.extractor import CONFIGS, ExtractorConfig, extract, system_prompt
from src.llm import LocalLLM

EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
LOG_PATH = EXPERIMENTS_DIR / "log.jsonl"

HOLDOUT = ["18325926-Rental-Agreement-1", "50070534-RENTAL-AGREEMENT"]
SWAPPED_LABELS = ["54770958-Rental-Agreement", "54945838-Rental-Agreement"]


def subsets() -> dict[str, list[str]]:
    labelled = sorted(n.strip() for n in load_labels(TRAIN_CSV)[FILE_NAME_COLUMN])
    usable = [d for d in labelled if d not in leaked_ids()]
    return {"tuning": [d for d in usable if d not in HOLDOUT], "holdout": list(HOLDOUT)}


def run(name: str, config: ExtractorConfig, subset: str) -> dict:
    documents = subsets()[subset]
    out_dir = EXPERIMENTS_DIR / name / subset
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    prompt = system_prompt(config.prompt_version)
    (out_dir / "system_prompt.txt").write_text(prompt, encoding="utf-8")  # exact prompt, for reproducibility

    llm = LocalLLM()
    llm.check()
    start = time.perf_counter()
    statuses, attempts = {}, 0
    for doc in documents:
        text = (EXTRACTED_TEXT_DIR / "train" / f"{doc}.txt").read_text(encoding="utf-8")
        result = extract(text, llm, doc, config)
        (pred_dir / f"{doc}.json").write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                                              encoding="utf-8")
        statuses[doc] = result.status
        attempts += result.attempts
        print(f"[{result.status}] {doc} ({result.attempts} attempt(s), {result.seconds}s)")
    seconds = round(time.perf_counter() - start, 1)

    labels = load_labels(TRAIN_CSV)
    predictions = load_predictions(pred_dir)
    outside = {d: f"not in {subset} set" for d in labels[FILE_NAME_COLUMN].str.strip() if d not in documents}
    views = {"all": outside, "clean": {**outside, **{d: "labels belong to the other document" for d in SWAPPED_LABELS}}}
    metrics = {}
    for view, exclude in views.items():
        evaluation = evaluate(labels, predictions, exclude)
        evaluation.save(out_dir / f"eval_{view}")
        metrics[view] = evaluation.metrics
        print(f"\n--- {subset} / {view} ---")
        print_metrics(evaluation.metrics)

    record = {
        "name": name,
        "time": datetime.now().isoformat(timespec="seconds"),
        "subset": subset,
        "config": asdict(config),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:12],
        "model": llm.config.model,
        "llm_options": llm.config.options(),
        "documents": len(documents),
        "failed": sorted(d for d, s in statuses.items() if s != "ok"),
        "attempts": attempts,
        "seconds": seconds,
        **{f"macro_{kind}_{view}": metrics[view][f"macro_{kind}_recall"]
           for view in views for kind in ("raw", "normalized")},
        "fields_clean": {f: [m["raw_recall"], m["normalized_recall"]] for f, m in metrics["clean"]["fields"].items()},
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--holdout", action="store_true", help="score the held-out documents (selected config only)")
    args = parser.parse_args()
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    record = run(args.name, CONFIGS[args.config], "holdout" if args.holdout else "tuning")
    print(f"\nlogged to {LOG_PATH}: macro raw/normalised (clean) = "
          f"{record['macro_raw_clean']} / {record['macro_normalized_clean']}")


if __name__ == "__main__":
    main()
