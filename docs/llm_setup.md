# Local LLM setup (`src/llm.py`)

Everything runs locally through [Ollama](https://ollama.com). No paid API is
used and no document text leaves the machine.

## Choice: `qwen2.5:7b` (Qwen2.5-7B-Instruct, Q4_K_M)

Checked on 2026-10-25 (ollama.com library and docs.ollama.com):

| Tag | Quantisation | Download | Fits 8 GB VRAM? |
|---|---|---|---|
| `qwen2.5:7b` (= `7b-instruct-q4_K_M`) | 4-bit K-quant (medium) | 4.7 GB | **yes, with room for context** |
| `qwen2.5:7b-instruct-q5_K_M` | 5-bit | 5.4 GB | tight at 16k context |
| `qwen2.5:7b-instruct-q8_0` | 8-bit | 8.1 GB | no |

The model supports up to 32K tokens of context. Q4_K_M is the usual quality/size
balance for 7B models. Because the 7B model fits entirely on the GPU with good speed (below),
**a smaller model was not needed**. `qwen2.5:3b` remains a fallback and can be selected
with `LLM_MODEL=qwen2.5:3b`.

## Installation (Windows)

1. Ollama **0.40.11** official installer (`OllamaSetup.exe`, 1.66 GB, Authenticode
   signature by *Ollama Inc.* verified). It installs per-user to
   `%LOCALAPPDATA%\Programs\Ollama` and needs an NVIDIA driver ≥ 551.61 (installed: 596.21).
2. Model weights go to **D:** because C: is nearly full. This is set with the user environment variable
   `OLLAMA_MODELS=D:\ollama\models`.
3. `ollama pull qwen2.5:7b`

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\ollama\models", "User")
ollama pull qwen2.5:7b
ollama list
```

After changing `OLLAMA_MODELS`, the Ollama server has to be restarted (quit the tray app
and start it again, or sign out and back in). If `ollama list` does not show the model, the server was
started without the variable. Start it manually in a terminal:

```powershell
$env:OLLAMA_MODELS = "D:\ollama\models"; ollama serve
```

## Generation settings

| Setting | Value | Why |
|---|---|---|
| `temperature` | 0 | Deterministic decoding |
| `seed` | 42 | Reproducible |
| `num_ctx` | **16384** | Ollama's default is only **4096** on GPUs with less than 24 GB (confirmed in its log: `vram-based default context ... default_num_ctx=4096`) |
| `num_predict` | 1024 | Cap on answer length |
| `format` | Pydantic JSON schema | Output is constrained to the schema, then validated with Pydantic |
| `keep_alive` | 10 min | Avoids reloading the model between documents |

All settings are overridable with `LLM_MODEL`, `LLM_NUM_CTX` and `OLLAMA_HOST`
(environment variables or a `.env` file).

The client raises `LLMError` when:
- the server is unreachable (with a hint to start it),
- the model is not pulled (with the exact `ollama pull` command),
- the prompt might have been truncated (`prompt tokens + num_predict > num_ctx`),
- the answer was cut off,
- the JSON does not match the schema.

## Weights vs context length vs KV cache

- **Model weights** are the fixed, learned parameters. 7.6 B parameters at about 4.5 bits each
  (Q4_K_M) take about **4.4 GB**, regardless of the input.
- **Context length** (`num_ctx`) is how many tokens the model can see at once: instructions,
  document and answer together. Anything beyond it is dropped.
- **KV cache** is working memory that stores attention keys and values for every token in
  the context window, and it is reserved up front for the full `num_ctx`. For Qwen2.5-7B
  (28 layers × 4 KV heads × 128 dimensions × K and V × 2 bytes) that is **about 56 KB per token**.
  So 4K of context costs about 0.23 GB, 16K about 0.9 GB and 32K about 1.8 GB.
- Total VRAM is roughly weights + KV cache + compute buffers. A bigger window costs memory
  even for short documents.

The measurements match the theory: from 4K to 16K context, VRAM rose by 684 MiB for 12,288
tokens, which is **57 KB per token**.

## Benchmark (`python -m src.llm_benchmark` → `outputs/llm_benchmark.json`)

RTX 4070 Laptop GPU, 8 GB:

| `num_ctx` | Model memory | On GPU | Output speed |
|---|---|---|---|
| 4,096 | 4,535 MiB | 100% | 44.8 tok/s |
| 8,192 | 4,763 MiB | 100% | 44.6 tok/s |
| **16,384** | **5,219 MiB** | **100%** | **44.3 tok/s** |
| 32,768 | 6,525 MiB | **91%** (9% on CPU) | 40.1 tok/s |

At 32K context, part of the model spills onto the CPU and it slows down. **16K is the largest window that
stays fully on the GPU**, and it leaves headroom for instructions and examples.

Timing:
- Cold start (loading the weights from disk): 34 s, plus 28 s for the first answer.
- Warm requests: **about 2.2 s** for a 209-token prompt and a 98-token answer.
- Reloading the model at a new `num_ctx`: about 5 s.

## Structured JSON test (synthetic contract)

The test contract was written for this purpose and is not from the dataset. Output:

```json
{"owner": "Ms. Ananya Rao", "tenant": "Mr. Kabir Mehta", "monthly_rent": 18500,
 "security_deposit": 100000, "start_date": "15.01.2024", "end_date": "14.12.2024",
 "renewal_notice_days": 30}
```

- All values are correct, including the rent being distinguished from the deposit,
  the Indian-format `1,00,000` being read as 100000, and dates normalised to DD.MM.YYYY.
- Repeating the request gave **byte-identical** output.
- The demo schema is throw-away. The real schema is designed in `src/schemas.py`.

## Will the longest agreement fit?

Token counts of the extracted training text (text only, labels not used):

| Document | Characters | Tokens |
|---|---|---|
| `6683129` (longest) | 16,898 | **4,783** |
| `46239065` (unlabelled) | 19,792 | 3,945 |
| the other 8 | 1,117–4,562 | 390–1,063 |

- With 16K context, the longest document plus about 1K of instructions plus a 1K answer
  uses less than half the window. **No truncation.**
- **At Ollama's default 4,096, the same document is silently truncated to 2,050 tokens**,
  so more than half of it would be lost.

## Housekeeping notes

- An older model, `gpt-oss:20b` (about 13 GB, from August 2025), still sits in
  `C:\Users\Asus\.ollama\models`. It was left untouched. With `OLLAMA_MODELS` pointing to
  D:, Ollama no longer lists it.
- The installer remains at `D:\ollama\installer\OllamaSetup.exe` (1.66 GB) and can be deleted.
