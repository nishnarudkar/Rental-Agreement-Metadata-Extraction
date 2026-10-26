# Development experiments (training data only)

> **Caveat.** There are 9 usable labelled training documents (8 distinct contracts). Two
> have swapped labels, and several labels are unattainable or unsupported by their own text. One
> document more or less right moves a field's recall by 20 points on the 5-document clean
> view. **None of the differences below are statistically meaningful.** They are used only to
> pick between changes that also make sense on their own merits.

## Protocol (fixed before the baseline)

| Item | Decision |
|---|---|
| Data | `train/` documents and `train.csv` only. `test.csv` and `test/` are not used. |
| Usable documents | 9: the `train.csv` row for `24158401` is excluded because its file is a test document |
| Holdout | `18325926`, `50070534`: DOCX files with sound labels, not run or inspected during iteration, scored **once** for the selected config |
| Tuning set | the other 7. The near-duplicate pair `6683127` (page 1) / `6683129` (full contract) is kept together on the tuning side. |
| Views | **clean** = tuning without the swapped-label pair (5 docs); **all** = 7 docs |
| Selection metric | Macro **raw** recall, tuning-clean; ties broken by normalised recall |
| Budget | At most 4 configurations |
| Leave-one-out | Not needed. The prompts contain no training examples (zero-shot), so no document's labels enter the prompt; the holdout guards against tuning the wording to specific documents. |
| Log | `outputs/experiments/log.jsonl` (one line per run), with each run's exact system prompt, predictions and evaluation reports in `outputs/experiments/<name>/<subset>/` |

## Results

| Run | Changes | Tuning-clean macro raw / norm | Tuning-all raw / norm | Failed docs | Time (7 docs) |
|---|---|---|---|---|---|
| `v1-baseline` | Prompt 8 extractor | 0.49 / 0.59 | 0.35 / 0.42 | 1 (`44737744`) | 145 s |
| `v2-names-nulls-salvage` | Instruction to drop titles/relations; null for illegible values; stricter notice definition; salvage | 0.46 / 0.56 | 0.32 / 0.39 | 0 | 191 s |
| **`v3-slots-derived-end`** | `title`/`relation` schema slots; contradictory rule fixed; **end date derived in code** from start + term | **0.53 / 0.59** | 0.37 / 0.42 | 0 | 274 s |
| `v4-verified` | v3 + LLM verification pass that nulls unsupported values | 0.42 / 0.49 | 0.30 / 0.35 | 0 | 142 s |
| **`v3` on holdout** | selected config, scored once | **0.50 / 0.75** | (same) | 0 | 28 s (2 docs) |

Per field, tuning-clean, raw (normalised):

| Field | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Value | 0.80 | 0.80 | 0.80 | 0.80 |
| Start date | 0.80 | 0.80 | 0.80 | 0.40 |
| End date | 0.40 | 0.40 | 0.40 | 0.20 |
| Notice | 0.75 | 0.75 | 0.75 | 0.75 |
| Party One | 0.20 (0.60) | 0.00 (0.60) | 0.20 (0.60) | 0.20 (0.60) |
| Party Two | 0.00 (0.20) | 0.00 (0.00) | 0.20 (0.20) | 0.20 (0.20) |

## Error analysis (v1 baseline, tuning set)

| Source | Cases | Fixable by the extractor? |
|---|---|---|
| **Label problems** | swapped pair `54770958`/`54945838` (12 cells; the predictions match the *other* row's labels); impossible end dates (`31.04.2011`, `31.02.2011`); `6683127` notice 15 not in its text; `44737744` rent 3000 not readable in its text | No |
| **Source-text OCR** (PDF→DOCX conversion) | `GERALDINE O. GALINATO` in the text vs `Q.` in the label; `Ingles. Jr.` vs `Ingles, Jr.` | No |
| **Our OCR / layout** | `54770958`: the two-column signature block was merged into one line and the model took names from it | Partly (prompt: use the introduction, not signatures) |
| **LLM reasoning** | End-date arithmetic wrong 3/5 (`30.04.2011` for 1.4.2010 + 11 months); notice "3 days/months" from a grace-period clause; names from the signature block | Yes, partly |
| **LLM instruction-following / format** | Titles and relations kept in names (`Mr. P C MATHEW, S/o K JOSEPH CHACKO`) | Partly |
| **Robustness** | `44737744`: garbled date in a date field, retried identically 3 times, so the whole document was lost | Yes (salvage) |

## What each change did

- **Salvage** (v2+): `44737744` is no longer lost. After 3 invalid answers only the invalid fields are
  nulled, so its parties etc. survive.
- **"Drop titles/relations" as an instruction** (v2): ignored. A 7B model keeps copying the
  introduction verbatim, especially next to the old rule "copy names as they appear".
- **Separate `title` and `relation` slots** (v3): titles now leave the name (`Kartheek R` exact,
  `Balaji.R` differs only by the label's trailing space). Relations still leak sometimes.
- **End date in code** (v3): the model reports start date and term with evidence; code computes
  the last day of the term. Every derived end date is now arithmetically correct (`28.02.2011`,
  `04.11.2009`, `31.03.2011`). The remaining misses are references that are impossible dates.
- **Verification pass** (v4): it removed 4 genuine fabrications (rent `9997` from
  `Rs 9.99.7.9°`, a date invented from `.?.??& day of 2010`, notice "3" from a grace-period clause, an
  unsupported start date). It also removed 7 correct values, mostly dates written in words
  ("1st day April two thousand eleven"). **Under a recall-only metric, nulling a wrong value
  cannot raise the score**, so a verifier can only lower it. It would be worth it if precision
  (no hallucinations) mattered, but not for this metric.

## Selected configuration: v3 (frozen)

`ExtractorConfig(prompt_version="v3", derive_end_date=True, salvage=True, verify=False)`,
`qwen2.5:7b` (Ollama model ID `845dbda0ea48`), temperature 0, seed 42, `num_ctx` 16384, at most 3
attempts, PaddleOCR strips of 1600 px with 200 px overlap. Fingerprint: `config/frozen_config.json`.
`python -m src.freeze` and `tests/test_frozen.py` fail if the prompt, schema, normaliser, LLM or OCR
settings change.

Known weaknesses carried into the test evaluation:
- Fabrication on garbled text (`44737744` rent/date) and on missing notice clauses (`6683127`).
- Relations or nicknames sometimes stay in names (`K.Kuttan S/o Kelu Aehari (Late)`); extra
  occupants are sometimes listed as tenants (`6683129`).
- About 20–40 s per document.
