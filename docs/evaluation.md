# Evaluation (`src/evaluate.py`)

## Metric

From the assignment brief, computed **per field**:

```
Recall = True / (True + False)
True   = predicted value exactly matches the reference value
False  = predicted value differs, or nothing was extracted
```

Two separate numbers are reported per field:

| Metric | Comparison |
|---|---|
| **Raw exact match** (the assignment metric) | Prediction cell == reference cell, **character for character**. The reference is used exactly as stored: no trimming, no correction. |
| **Normalised match** (diagnostic) | Both sides passed through `src/normalize.py`: whitespace; number format (`6,500.00` = `6500`); date padding (`1.4.2010` = `01.04.2010`); names (case, punctuation, `&` = `and`, titles such as Mr/Prof dropped). An empty prediction never matches. |

**Macro average** is the mean of the per-field recalls (fields without a denominator are left
out). **Micro average** (all counted cells pooled) is also reported.

## Denominator per field

For each field, the denominator is the **number of documents under evaluation whose
reference value for that field is not blank**.

| Case | Treatment |
|---|---|
| Reference present, prediction matches | True |
| Reference present, prediction differs | False (`mismatch`) |
| Reference present, prediction empty or null | False (`not extracted`) |
| Reference present, document has no prediction file or extraction failed | False (`no prediction for document`). Documents are never skipped silently. |
| **Reference blank** | **Not counted**: there is nothing to recall. Reported as `blank_reference`. |
| Reference is an impossible date (e.g. `31.02.2011`) | Counted and scored as stored, so a correct calendar date cannot match it. Reported as `unattainable_reference`. |

Example (train): `Renewal Notice (Days)` is blank for `44737744`, so over the 9 usable
train documents its denominator is 8, while the other five fields have 9.

Label rows are excluded **only explicitly**, and every exclusion appears in `metrics.json`
with its reason:

- automatically, when the document file is not in that split's folder
  (`24158401` in `train.csv` lives in `test/`);
- `--exclude ID ...` for documented decisions (e.g. the two train rows whose labels are swapped);
- `--documents ID ...` to score a subset during development.

Identifiers: label IDs are matched after trimming spaces. Each prediction file's
`document_id` must equal its file name. Duplicate IDs are an error. Predictions with no
label row are listed in `predictions_without_label`.

## Outputs (`outputs/evaluation/<split>/`)

| File | Content |
|---|---|
| `metrics.json` | Per field: denominator, raw/normalised true and recall, not extracted, blank and unattainable counts, definition text. Plus macro/micro averages, exclusions, documents without predictions. |
| `metrics.csv` | The per-field table |
| `per_document.csv` | One row per document × field: reference, prediction, counted, raw/normalised match, outcome, unattainable flag |

## Usage

```powershell
python -m src.evaluate --split train
python -m src.evaluate --split train --documents 47854715-RENTAL-AGREEMENT 36199312-Rental-Agreement
python -m src.evaluate --split train --predictions outputs\predictions.csv
python -m src.evaluate --split test --final     # only for the final evaluation
```

## Check on two training predictions (from Prompt 8)

| Field | n | Raw | Normalised | Note |
|---|---|---|---|---|
| Aggrement Value | 2 | 1.00 | 1.00 | |
| Aggrement Start Date | 2 | 1.00 | 1.00 | |
| Aggrement End Date | 2 | 0.00 | 0.00 | both references are impossible dates (unattainable) |
| Renewal Notice (Days) | 2 | 1.00 | 1.00 | |
| Party One | 2 | 0.00 | 0.50 | `Mr. Balaji.R` vs `Balaji.R ` matches only after normalisation |
| Party Two | 2 | 0.00 | 0.50 | likewise `Mr.Kartheek R` vs `Kartheek R` |
| **Macro** | | **0.50** | **0.67** | |

These figures match a manual comparison. Two documents are far too few to judge the system;
this run only verifies the framework.
