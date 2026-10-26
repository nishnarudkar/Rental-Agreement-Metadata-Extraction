# Extraction engine (`src/extractor.py`)

## Flow

```
extracted text ──► prompt (rules + field definitions + JSON schema) + document
                └► local LLM (Ollama, JSON-schema-constrained, temperature 0)
                └► Pydantic validation (AgreementExtraction)
                     ├─ invalid / cut off ──► retry with the error fed back (max 3 attempts)
                     └─ valid ──► AgreementMetadata (six fields) + evidence check
```

- **Prompt**: general rules (use only the text, null if absent, verbatim evidence of
  at most about 25 words, OCR may contain typos), field definitions derived from the
  training documents (see `docs/schema.md`), and the JSON schema itself. Ollama's
  `format` does not show field descriptions to the model, so the schema must be in the prompt.
- **No dataset-specific content**: a test checks that the prompt contains no file name,
  party name or amount from `train.csv`.
- **Retries**: only for malformed or cut-off answers (`LLMOutputError`). At temperature 0
  the same prompt gives the same answer, so each retry appends the validation error
  ("Your previous answer was rejected: …"). Unrecoverable errors (server down,
  input longer than the context) fail immediately.
- **Failure reporting**: `extract()` never raises. It returns `status="failed"` with the
  error and the number of attempts, and the six fields stay empty. Nothing is invented.
- **Evidence check**: each submitted value's quote is matched against the document
  (ignoring case, line breaks and spacing). A missing or non-verbatim quote produces a
  **warning only**. Values are never changed by this check.
- **Test-data guard**: `--split test` refuses to run without `--final`.

## Output

`python -m src.extractor --split train --documents <id> ...` writes
`outputs/predictions/<split>/<id>.json`:

| Key | Content |
|---|---|
| `prediction` | the six fields under the CSV column names (submission format) |
| `evidence` | per column: value, quote, `evidence_found` |
| `extraction` | the full `AgreementExtraction` (deposit, execution date, both notices, representatives, …) |
| `status`, `error`, `warnings`, `attempts`, `seconds`, `prompt_tokens`, `output_tokens` | diagnostics |

## First run on two training documents

The documents were chosen because they are hard cases and their labels were verified in the audit.

| Field | `47854715` (DOCX) | label | `36199312` (scan, OCR) | label |
|---|---|---|---|---|
| Value | 9000 | 9000 | 3800 | 3800 |
| Start | 01.04.2010 | 01.04.2010 | 01.05.2010 | 01.05.2010 |
| End | 30.04.2011 | 31.02.2011 | 31.03.2011 | 31.04.2011 |
| Notice | 60 | 60 | 30 | 30 |
| Party One | `Mr. P C MATHEW, S/o K JOSEPH CHACKO` | ` P C MATHEW` | `Mr. Balaji.R` | `Balaji.R ` |
| Party Two | `Mr. L GOPINATH S/o G LAKSHMI NARISIMHAN` | ` L GOPINATH ` | `Mr.Kartheek R` | `Kartheek R` |
| Attempts / time | 2 / 40.1 s | | 1 / 22.1 s | |

What worked:
- **Rent vs deposit, and figures vs words.** `47854715` writes "Rs.9000/- (Rupees Seven
  thousand)"; `36199312` writes "Rs.3800/- (Rupees Thirty Eight Thousand)". Both were taken
  from the figures, and both deposits were ignored.
- **Execution vs commencement.** `47854715` was signed on 21 March 2010; the model returned
  the lease start, 01.04.2010.
- **Termination vs renewal notice.** `47854715` has both clauses; the model used the 2-month
  termination clause (60), as the labels do.
- **Retry.** On the first attempt the model produced `30.02.2011`. Validation rejected it,
  and the retry with feedback returned a valid date.
- **Evidence check.** It flagged one paraphrased quote ("The agreement…" where the text says
  "This agreement…").

Problems to address in Prompt 10 (from training documents only):
1. **End dates computed by the model are unreliable.** For 01.04.2010 + 11 months the
   correct last day is 28.02.2011; the model said 30.04.2011. For `36199312` its
   31.03.2011 is arithmetically correct, but the label is the impossible 31.04.2011, so it
   cannot be matched exactly. A possible fix is to have the model report the start
   date and term (which it read correctly) and do the date arithmetic in code.
2. **Party names include titles and relations** (`Mr.`, `S/o …`) although the rules ask
   to drop relations. Labels mostly omit both, so the raw exact match fails. Only titles are
   handled by the normalised metric.
3. **Speed**: 22–40 s per document, because the evidence quotes add about 575 output tokens.
