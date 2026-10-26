"""Fingerprint of the frozen extraction configuration.

    python -m src.freeze --write   # record the current configuration (end of development)
    python -m src.freeze           # check that nothing changed since it was recorded

Everything that affects predictions is hashed: the system prompt, the JSON
schema, the evaluation normaliser, the LLM options, the OCR settings and the
extractor config. ``tests/test_frozen.py`` fails if any of them drifts.
"""

import argparse
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path

from src import normalize, ocr
from src.config import PROJECT_ROOT
from src.extractor import CONFIGS, FINAL_CONFIG_NAME, MAX_ATTEMPTS, system_prompt
from src.llm import LLMConfig
from src.schemas import AgreementExtraction

FROZEN_PATH = PROJECT_ROOT / "config" / "frozen_config.json"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint() -> dict:
    config = CONFIGS[FINAL_CONFIG_NAME]
    llm = LLMConfig()
    return {
        "extractor_config_name": FINAL_CONFIG_NAME,
        "extractor_config": asdict(config),
        "max_attempts": MAX_ATTEMPTS,
        "system_prompt_sha256": _sha(system_prompt(config.prompt_version)),
        "schema_sha256": _sha(json.dumps(AgreementExtraction.model_json_schema(), sort_keys=True)),
        "normalizer_sha256": _sha(inspect.getsource(normalize)),
        "model": llm.model,
        "llm_options": llm.options(),
        "ocr": {
            "tile_height": ocr.TILE_HEIGHT,
            "tile_overlap": ocr.TILE_OVERLAP,
            "paragraph_gap": ocr.PARAGRAPH_GAP,
            "engine_settings_sha256": _sha(inspect.getsource(ocr._engine)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="record the current configuration as frozen")
    args = parser.parse_args()
    current = fingerprint()
    if args.write:
        FROZEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        FROZEN_PATH.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {FROZEN_PATH}")
        return 0
    frozen = json.loads(Path(FROZEN_PATH).read_text(encoding="utf-8"))
    changed = sorted(k for k in frozen.keys() | current.keys() if frozen.get(k) != current.get(k))
    print("frozen configuration unchanged" if not changed else f"CHANGED since freeze: {changed}")
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
