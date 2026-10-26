"""Check the local LLM: JSON output, speed, GPU memory and context length.

    python -m src.llm_benchmark

Uses a synthetic contract for the JSON test and only *training* documents for
the context-length check (their text, never their labels). Results are saved
to ``outputs/llm_benchmark.json``.
"""

import json
import subprocess
from dataclasses import replace

from pydantic import BaseModel

from src.audit import load_cached_texts
from src.config import OUTPUTS_DIR
from src.llm import LLMConfig, LocalLLM

CONTEXT_SIZES = [4096, 8192, 16384, 32768]

# Fictional contract written for this test; it is not from the dataset.
SYNTHETIC_CONTRACT = """LEASE AGREEMENT
This lease is made at Pune on 3rd January 2024 between Ms. Ananya Rao (the Owner)
and Mr. Kabir Mehta (the Tenant). The Owner lets Flat 12B, Lake View Apartments,
to the Tenant for eleven months commencing 15th January 2024 and ending
14th December 2024. The Tenant shall pay a monthly rent of Rs. 18,500/- (Rupees
Eighteen Thousand Five Hundred only) and has paid a refundable security deposit of
Rs. 1,00,000/-. Either party may end this lease by giving one month's written notice."""


class DemoFields(BaseModel):
    """Throw-away schema for the smoke test; the real schema is designed separately."""

    owner: str | None
    tenant: str | None
    monthly_rent: int | None
    security_deposit: int | None
    start_date: str | None
    end_date: str | None
    renewal_notice_days: int | None


SYSTEM = (
    "You read rental agreements and return JSON only. Use only facts stated in the text. "
    "Write dates as DD.MM.YYYY and amounts as whole numbers without currency symbols or "
    "separators. If a value is not stated, return null."
)


def gpu_memory_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def loaded_model_memory(llm: LocalLLM) -> dict:
    for m in llm.client.ps().models:
        if m.model.startswith(llm.config.model.split(":")[0]):
            size, vram = m.size or 0, m.size_vram or 0
            return {
                "total_mib": round(size / 2**20),
                "vram_mib": round(vram / 2**20),
                "on_gpu_pct": round(100 * vram / size) if size else None,
                "context_length": getattr(m, "context_length", None),
            }
    return {}


def unload(llm: LocalLLM) -> None:
    llm.client.generate(model=llm.config.model, prompt="", keep_alive=0)


def count_tokens(llm: LocalLLM, text: str, num_ctx: int) -> int:
    """Tokens the server actually ingested for ``text`` at a given context size."""
    response = llm.client.generate(
        model=llm.config.model, prompt=text, keep_alive=llm.config.keep_alive,
        options={"num_ctx": num_ctx, "num_predict": 1, "temperature": 0},
    )
    return response.prompt_eval_count or 0


def main() -> None:
    base = LLMConfig()
    llm = LocalLLM(base)
    llm.check()
    report: dict = {"model": base.model, "options": base.options()}

    unload(llm)
    report["gpu_used_before_mib"] = gpu_memory_used_mib()

    # 1. JSON test on the synthetic contract (first call also loads the model).
    first = llm.generate_json(SYSTEM, SYNTHETIC_CONTRACT, DemoFields)
    second = llm.generate_json(SYSTEM, SYNTHETIC_CONTRACT, DemoFields)
    report["json_test"] = {
        "parsed": first.parsed.model_dump(),
        "identical_on_repeat": first.raw == second.raw,
        "cold_seconds": first.seconds,
        "load_seconds": first.load_seconds,
        "warm_seconds": second.seconds,
        "prompt_tokens": second.prompt_tokens,
        "output_tokens": second.output_tokens,
        "output_tokens_per_second": second.tokens_per_second,
    }
    print(json.dumps(report["json_test"], indent=2))

    # 2. Memory and speed at each context size (changing num_ctx reloads the model).
    report["by_context"] = []
    for num_ctx in CONTEXT_SIZES:
        unload(llm)
        sized = LocalLLM(replace(base, num_ctx=num_ctx), client=llm.client)
        result = sized.generate_json(SYSTEM, SYNTHETIC_CONTRACT, DemoFields)
        row = {
            "num_ctx": num_ctx,
            "gpu_used_mib": gpu_memory_used_mib(),
            **loaded_model_memory(sized),
            "load_seconds": result.load_seconds,
            "output_tokens_per_second": result.tokens_per_second,
        }
        report["by_context"].append(row)
        print(row)

    # 3. Token counts of every training document at the full 32k window, plus
    #    what the 4k default would have ingested for the longest one.
    unload(llm)
    texts = load_cached_texts("train")
    tokens = {doc: count_tokens(llm, text, 32768) for doc, text in texts.items()}
    longest = max(tokens, key=tokens.get)
    report["train_document_tokens"] = {
        "per_document": {d: {"chars": len(texts[d]), "tokens": t} for d, t in sorted(tokens.items(), key=lambda x: -x[1])},
        "longest": longest,
        "longest_tokens": tokens[longest],
        "longest_tokens_at_default_4096": count_tokens(llm, texts[longest], 4096),
    }
    print(json.dumps(report["train_document_tokens"], indent=2))

    out = OUTPUTS_DIR / "llm_benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
