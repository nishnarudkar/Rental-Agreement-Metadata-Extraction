# Extraction schema (`src/schemas.py`) and evaluation normalisation (`src/normalize.py`)

## 1. Reference columns and formats (train.csv)

| Column (exact) | Raw format in the CSV | Output type | Missing value |
|---|---|---|---|
| `Aggrement Value` | digits, e.g. `4000` | `int` | `None` → empty cell |
| `Aggrement Start Date` | `DD.MM.YYYY` (one has a trailing space) | `str` (validated calendar date) | `None` → empty cell |
| `Aggrement End Date` | `DD.MM.YYYY` (3 impossible dates in train) | `str` (validated calendar date) | `None` → empty cell |
| `Renewal Notice (Days)` | digits; one blank | `int` (days) | `None` → empty cell |
| `Party One` / `Party Two` | free text, inconsistent titles and spaces | `str` | `None` → empty cell |

`None` is JSON `null` and an empty CSV cell. Values are never invented to fill a gap.

## 2. Two schema layers

- **`AgreementExtraction`** is what the model fills in. It is richer than the six fields, so
  the model never has to merge concepts that the documents keep separate. Every value
  carries a verbatim `evidence` quote (null when the value is null). Unknown keys are
  forbidden and every key is required, so the JSON schema is closed.
- **`AgreementMetadata`** holds the six assignment fields with the exact CSV column names.
  `to_csv_row()` produces empty strings for missing values; `to_json()` produces `null`.
  `AgreementMetadata.from_extraction()` maps one layer to the other, as defined below.

## 3. Field definitions and the distinctions Prompt 7 asks for

These were derived from the **training** documents and labels only (see `docs/data_notes.md`).

| Field | Extraction fields | Mapping to the submission field |
|---|---|---|
| Agreement Value | `monthly_rent`, `security_deposit`, `advance_rent` (kept separate) | **`monthly_rent`**. Train labels are the monthly rent, never the deposit or advance. |
| Start Date | `execution_date` and `tenancy_start_date` (kept separate) | **`tenancy_start_date`, otherwise `execution_date`**. In train, `47854715` was signed 21.03.2010 but the lease runs from 01.04.2010 (label 01.04.2010); `18325926` states only the execution date (label = that date). |
| End Date | `tenancy_end_date` (only if stated), `term` (months + evidence) | `tenancy_end_date` if stated; otherwise **computed in code** as the last day of the term: start + term months − 1 day (e.g. 01.04.2010 + 11 months → 28.02.2011). The model does no date arithmetic (changed in Prompt 10, config v3). |
| Renewal Notice (Days) | `termination_notice` and `renewal_notice` (kept separate, each stored as value + unit) | **`termination_notice`, otherwise `renewal_notice`**, converted to days (1 month = 30, 1 week = 7). See the note below. |
| Party One / Two | `party_one` / `party_two` with `title`, `name`, `relation`, `representative` | `name`. Party One = lessor/owner, Party Two = lessee/tenant. Titles (Mr., Prof.) and relations (S/o …) have their own slots so they stay out of `name` (Prompt 10, v3). |

**Renewal vs termination notice.** The column is called "Renewal Notice", but the training
labels follow the **termination/vacating** notice: `47854715` contains both
"Two months notice … to terminate" and "one month advance notice … for negotiating
renewal", and its label is 60. The schema therefore extracts both notices separately
(nothing is merged at extraction time). The mapping uses the termination notice, and falls
back to a renewal notice only when no termination notice exists. That fallback is an
assumption with no training example either way.

**Company vs representative.** If a party is a company, `name` holds the company (the
legal party) and `representative` holds the person signing or acting for it. Only `name`
is submitted. No training document has a company party, so this rule could not be
checked against labels.

**Notice as value + unit.** The model reports the number and the unit as written
(`2`, `months`). Converting to days happens in code, so the model does no arithmetic.

*This document describes the frozen schema (config v3). Prompt 7's first version had a single
`term_months` integer and no `title`/`relation` slots; see `docs/experiments.md` for why they changed.*

## 4. Validation (Pydantic)

| Check | Behaviour |
|---|---|
| Dates | Must be real `DD.MM.YYYY` dates. `31.02.2011`, ISO dates and month names are **rejected**. `1.4.2010` is re-emitted as `01.04.2010`. |
| Amounts | Whole numbers ≥ 0. `"18,500"`, words, decimals and negatives are rejected. |
| Notice | Value > 0 and unit ∈ {days, weeks, months}. Value and unit must both be present or both be null. |
| Text | Whitespace collapsed; empty or blank strings become `None`. |
| Structure | Unknown keys rejected; every key required (closed JSON schema). |

Rejected output is **never repaired**. The extractor (Prompt 8) retries or reports the failure.

## 5. Evaluation-only normalisation (`src/normalize.py`)

Kept separate from the schema. It is used only for the optional *normalised* metric
next to the raw exact-match metric, and it never changes predictions or reference files.

| Field | Normalisation | Example |
|---|---|---|
| Amount, notice | Whole number; commas and `.00` removed | `"6,500.00"` → `6500` |
| Dates | Zero-padded, `.` separator. **Not** calendar-checked, so `31.02.2011` stays as it is. | `1/4/2010` → `01.04.2010` |
| Parties | Lower-cased; punctuation becomes spaces; `&` becomes `and`; titles (Mr, Mrs, Ms, Prof, Dr, Sri, Smt, …) dropped | `MR.K.Kuttan ` → `k kuttan` |

Spelling and word order still matter after normalisation (`Deepthi` ≠ `Deepti`).

## 6. Check with the local model (synthetic contract)

Ollama's `format` constrains only the **shape** of the JSON; the model does not see the
field descriptions. With the schema given only as `format`, the model filled `evidence`
with `"True"`/`"False"` and produced dates such as `strconv.3.1.2024`. The validators
rejected that output, as intended. With the JSON schema also included in the system
prompt, every field was correct, with verbatim evidence and correct nulls (16 s:
1,367 prompt tokens and 464 output tokens). **The extraction prompt must therefore include the
schema and the field definitions** (Prompt 8).
