# Final test evaluation

Run once on 2026-10-26 with `python -m src.final_evaluation`, after development was frozen.
The extraction system was **not** changed after seeing these results.

## Pre-run checks

| Check | Result |
|---|---|
| Frozen configuration (`src.freeze`, `tests/test_frozen.py`) | Unchanged: config `v3`, prompt `22ff42ae…`, schema `0e689147…`, normaliser `cbfcad79…`, temperature 0, seed 42, `num_ctx` 16384; OCR strips 1600/200 px |
| Model | `qwen2.5:7b`, Ollama ID `845dbda0ea48` (same as during development); PaddlePaddle 3.3.1 / PaddleOCR 3.7.0 |
| Test labels unused in development | Searched `src/`, `tests/`, `config/`, `docs/`, `notebooks/` and all development outputs for the 12 distinctive values that occur only in test labels. The only hit, `31.03.2011`, is the model's own end-date prediction for *train* document `36199312`, which coincides with a test value. No test-only document was extracted before this run. *Disclosure:* `test.csv` was printed once during project setup (Prompt 1); it was not used for any decision. |
| Train/test overlap | `24158401` (a test document) also has a label row in `train.csv`. That row was excluded from development and the document was never extracted before this run. No near-duplicates: the highest train/test text overlap is 0.10 (`95980236` vs `50070534`), recomputed on fresh OCR. |

## Pipeline run

Document ingestion → DOCX parsing / OCR → local LLM extraction → schema validation → predictions.

| Stage | Result | Time |
|---|---|---|
| Ingestion | 4/4 succeeded (2 DOCX, 2 PNG); 0 empty, 0 failed | 917.6 s (OCR 327 s and 590 s; DOCX < 0.1 s) |
| Extraction | 4/4 succeeded at the first attempt; 0 salvaged; 0 failed | 127.7 s (12–53 s per document) |
| Tokens per document | 2,635–3,105 prompt + 593–744 output | |

OCR took noticeably longer than on the training scans (163–292 s). The settings were identical;
CPU load on the machine is the likely cause.

## Test performance (4 documents)

| Field | n | Raw exact recall | Normalised recall |
|---|---|---|---|
| Aggrement Value | 4 | **1.00** | 1.00 |
| Aggrement Start Date | 4 | **1.00** | 1.00 |
| Aggrement End Date | 4 | **0.75** | 0.75 |
| Renewal Notice (Days) | 4 | **1.00** | 1.00 |
| Party One | 4 | **0.00** | 0.75 |
| Party Two | 4 | **0.00** | 0.25 |
| **Macro average** | | **0.62** | **0.79** |
| Micro average (24 cells) | | 0.625 | 0.792 |

**Missing values:** no test reference cell is blank, so every denominator is 4. No prediction
cell is empty, so nothing counts as "not extracted". **Failed documents: 0.**

Predictions (`outputs/predictions.csv`):

| File Name | Value | Start | End | Notice | Party One | Party Two |
|---|---|---|---|---|---|---|
| 156155545-Rental-Agreement-Kns-Home | 12000 | 15.12.2012 | 14.11.2013 | 30 | V.K.NATARAJ, son of V.KANDASWAMI CHETTIAR | VYSHNAVI DAIRY SPECIALITIES Private Ltd. |
| 228094620-Rental-Agreement | 15000 | 07.07.2013 | 06.06.2014 | 30 | KAPIL MEHROTRA | B.Kishore , S/o. Mr. B. Pampaiah |
| 24158401-Rental-Agreement | 12000 | 01.04.2008 | 31.03.2009 | 60 | Hanumaiah | Vishal Bhardwaj S/O Charnel Singh |
| 95980236-Rental-Agreement | 9000 | 01.04.2010 | 28.02.2011 | 30 | S.Sakunthala | V.V.Ravi Kian, S/o V.R.G.Sastry |

## Error categories (9 of 24 cells not an exact match)

| Category | Cells | Examples |
|---|---|---|
| **Reference formatting only**: the prediction is right, but the label has extra spaces or punctuation that differ (all match after normalisation) | 4 | `Hanumaiah` vs `Hanumaiah `; `KAPIL MEHROTRA` vs ` KAPIL MEHROTRA `; `S.Sakunthala` vs ` S.Sakunthala`; `…Private Ltd.` (as written in the document) vs `…Private Ltd` |
| **Relation kept in the party name** (LLM formatting; known weakness from development) | 4 | `V.K.NATARAJ, son of V.KANDASWAMI CHETTIAR`; `Vishal Bhardwaj S/O Charnel Singh`; `V.V.Ravi Kian, S/o V.R.G.Sastry`; `B.Kishore , S/o. Mr. B. Pampaiah` (the reference `.B.Kishore ` also has a stray leading dot) |
| **Reference disagrees with the document** | 1 | `95980236` end date: the document states only "11 (Eleven) months" from 01.04.2010, so the system returns 28.02.2011; the reference 31.03.2011 corresponds to 12 months |
| OCR errors in target fields | 0 | Both scans were read correctly for all six fields |
| Wrong value chosen (rent vs deposit, execution vs start, termination vs renewal) | 0 | |
| Fabricated values | 0 observed | 3 evidence quotes were slightly paraphrased (warnings in `test_metrics.json`), but their values are correct |

## Test vs training (do not compare too closely)

| Set | Docs | Macro raw | Macro normalised | Notes |
|---|---|---|---|---|
| Train tuning, all | 7 | 0.37 | 0.42 | includes 2 swapped-label rows and impossible reference dates |
| Train tuning, clean | 5 | 0.53 | 0.59 | used for selection; slightly optimistic for that reason |
| Train holdout | 2 | 0.50 | 0.75 | scored once before freezing |
| **Test** | **4** | **0.62** | **0.79** | scored once after freezing |

The test score is higher than the training scores mainly because the test labels are cleaner:
no swapped rows and no impossible dates. With 4 documents (24 cells), one cell changes the
macro recall by about 4 points, so **this is not evidence that the system generalises better than
the training figures suggest.** The main real weakness on unseen data is party-name formatting:
relations ("S/o …", "son of …") are kept in the name, so both party fields score 0 on raw exact match.
