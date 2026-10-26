# Rental Agreement Metadata Extraction

Extracts six metadata fields from rental agreements, given as **DOCX files or scanned PNG
images**, with a **locally hosted language model**. No regular expressions or keyword rules
are used to find the fields: a local LLM reads the agreement and fills a validated JSON
schema, quoting the evidence for every value.

| Field (CSV column) | Meaning in this dataset |
|---|---|
| `Aggrement Value` | monthly rent (not deposit / advance) |
| `Aggrement Start Date` | tenancy start date, otherwise execution date (`DD.MM.YYYY`) |
| `Aggrement End Date` | stated end date, otherwise last day of the stated term |
| `Renewal Notice (Days)` | notice period for termination / vacating, in days |
| `Party One` | lessor / owner |
| `Party Two` | lessee / tenant |

The CSV column names keep the dataset's spelling ("Aggrement").

## Results

Per-field **exact-match recall** on the 4 test documents. The pipeline was frozen before
this single run (`docs/test_results.md`):

| Field | Raw exact recall | Normalised recall* |
|---|---|---|
| Aggrement Value | **1.00** | 1.00 |
| Aggrement Start Date | **1.00** | 1.00 |
| Aggrement End Date | **0.75** | 0.75 |
| Renewal Notice (Days) | **1.00** | 1.00 |
| Party One | **0.00** | 0.75 |
| Party Two | **0.00** | 0.25 |
| **Macro average** | **0.625** | **0.792** |

\* *Normalised* ignores formatting only (spaces, `6,500` vs `6500`, `1.4.2010` vs `01.04.2010`,
case, punctuation, titles such as "Mr."). The **raw** column is the assignment metric
(Recall = True / (True + False), per field).

Of the 9 test cells that are not exact matches:
- 4 are correct values whose reference has extra spaces or punctuation (` KAPIL MEHROTRA `);
- 4 are party names that keep a relation (`Vishal Bhardwaj S/O Charnel Singh`);
- 1 is an end date where the reference (12 months) contradicts the document ("11 months").

No document failed and no OCR error affected a field.

### Test predictions (`outputs/predictions.csv`)

| File Name | Value | Start | End | Notice | Party One | Party Two |
|---|---|---|---|---|---|---|
| 156155545-Rental-Agreement-Kns-Home | 12000 | 15.12.2012 | 14.11.2013 | 30 | V.K.NATARAJ, son of V.KANDASWAMI CHETTIAR | VYSHNAVI DAIRY SPECIALITIES Private Ltd. |
| 228094620-Rental-Agreement | 15000 | 07.07.2013 | 06.06.2014 | 30 | KAPIL MEHROTRA | B.Kishore , S/o. Mr. B. Pampaiah |
| 24158401-Rental-Agreement | 12000 | 01.04.2008 | 31.03.2009 | 60 | Hanumaiah | Vishal Bhardwaj S/O Charnel Singh |
| 95980236-Rental-Agreement | 9000 | 01.04.2010 | 28.02.2011 | 30 | S.Sakunthala | V.V.Ravi Kian, S/o V.R.G.Sastry |

### Per-field recall on the training set

Same frozen configuration (v3), on the 9 usable training documents. The `train.csv` row for
`24158401` is excluded because that file is a test document. These predictions come from the
development runs: 7 tuning documents, used to choose the configuration (so slightly
optimistic), and 2 held-out documents. Reports: `outputs/evaluation/train_v3_all/` and
`outputs/evaluation/train_v3_clean/`.

| Field | All 9 docs: raw | normalised | Without the swapped-label pair (7): raw | normalised |
|---|---|---|---|---|
| Aggrement Value | 0.667 (6/9) | 0.667 | 0.857 (6/7) | 0.857 |
| Aggrement Start Date | 0.556 (5/9) | 0.667 | 0.714 (5/7) | 0.857 |
| Aggrement End Date | 0.333 (3/9)† | 0.333 | 0.429 (3/7)† | 0.429 |
| Renewal Notice (Days) | 0.625 (5/8)‡ | 0.625 | 0.833 (5/6)‡ | 0.833 |
| Party One | 0.111 (1/9) | 0.444 | 0.143 (1/7) | 0.571 |
| Party Two | 0.111 (1/9) | 0.222 | 0.143 (1/7) | 0.286 |
| **Macro average** | **0.401** | **0.493** | **0.520** | **0.639** |

† 3 reference end dates do not exist on the calendar (`31.11.2009`, `31.04.2011`, `31.02.2011`),
so at most 6/9 (4/7) can match exactly. ‡ One reference is blank, so the denominator is 8 (6).
The two rows whose labels are swapped (`54770958`, `54945838`) cannot match whatever the system
outputs; the second view leaves them out and says so.

### Summary across sets

| Set | Docs | Macro raw | Macro normalised |
|---|---|---|---|
| Train, all usable | 9 | 0.401 | 0.493 |
| Train, without swapped labels | 7 | 0.520 | 0.639 |
| Train, holdout only | 2 | 0.500 | 0.750 |
| **Test** | 4 | **0.625** | **0.792** |

The dataset is tiny (9 usable training documents, 4 test documents): one cell moves a
field's recall by 11–25 points. The test score is higher mainly because the test labels are
cleaner, not because the system generalises better. See `docs/experiments.md`.

## How it works

```
 DOCX ──► XML reader (paragraphs, tables, text boxes) ─┐
                                                       ├─► plain text ─► local LLM (Ollama, Qwen2.5-7B)
 PNG ───► PaddleOCR (full-resolution strips) ──────────┘        │  prompt = rules + field definitions + JSON schema
                                                                ▼  output constrained to the schema, temperature 0
                                         Pydantic validation (real dates, whole numbers, closed schema)
                                           │ invalid → retry with the error fed back (max 3), then null only invalid fields
                                           ▼
                           six fields + evidence quote per field ─► predictions.csv ─► per-field recall
```

1. **Ingestion** (`src/document_reader.py`, `src/ocr.py`, `src/extract_text.py`)
   - DOCX files are read from their XML in reading order, including **text boxes**, which
     `python-docx` normally skips and which hold real content in this dataset.
   - PNG scans go through **PaddleOCR** on the CPU, cut into overlapping 1600 px strips, so
     the tall 6600 px pages are not downscaled.
   - Every document gets warnings and timing; failures are reported, never skipped.
2. **Extraction** (`src/extractor.py`, `src/schemas.py`, `src/llm.py`)
   - The model fills a detailed schema that keeps confusable concepts apart:
     - rent vs deposit vs advance;
     - execution date vs tenancy start;
     - termination vs renewal notice;
     - company vs its representative;
     - titles and relations vs the name.
   - Each value comes with a verbatim evidence quote, or `null` if it isn't stated.
   - The end date is computed **in code** from start date + term when it isn't stated, because the
     7B model's date arithmetic was unreliable.
   - A check flags evidence quotes that don't occur in the document.
3. **Evaluation** (`src/evaluate.py`, `src/normalize.py`)
   - Raw exact-match recall and a separate normalised recall, with explicit denominators.
   - Blank references are excluded and impossible reference dates are flagged, never corrected.

Why local: the agreements contain personal data. Everything runs on the machine, with no paid
API and no upload.

## Is this rule-based?

No. The assignment asks for no rule-based approach (regex, static conditions). This is how
each part of the system relates to that requirement:

| Step | Who does it | Any regex or static condition on the document text? |
|---|---|---|
| Reading DOCX / OCR on PNG | `python-docx` XML walk / PaddleOCR (a deep-learning model) | No. They turn the document into text and look for no fields. |
| **Finding all six fields in the text** | **The LLM**, from field *definitions* (e.g. "the recurring rent per month, never the deposit") | **No.** There are no keyword lists, patterns, positions or per-template rules. The same prompt is used for every document, and a test checks that it contains no file name, party name or amount from the dataset. |
| Validating the answer | Pydantic | No. It only checks the *answer's* format (a real calendar date, a whole number). An invalid answer is sent back to the LLM with the error; nothing is repaired by hand. |
| Turning the model's answer into the six CSV columns | Small, documented post-processing **of values the LLM already extracted** | Never reads the document. Details below. |
| Evaluation normalisation | `src/normalize.py` | Used only to report the secondary "normalised" score, never to produce predictions. |

The post-processing in `AgreementExtraction.submission_sources()` (`src/schemas.py`) consists of:

1. **Units** – the model reports a notice period as written ("2", "months"); code converts it to
   days (×30, ×7) because the column is in days. This is unit conversion, not extraction.
2. **Calendar arithmetic** – when the agreement states a term but no end date, the model
   reports the start date and the term (each with its evidence quote), and code computes the last
   day of the term. In configs v1 and v2 the model did this arithmetic itself, and in the baseline
   it got 3 of 5 wrong (e.g. `30.04.2011` for 1 April 2010 + 11 months). Doing it in code fixed every
   case (`docs/experiments.md`).
3. **Choosing between two extracted values** – the model extracts the execution date and the
   tenancy start date separately, and the termination notice and the renewal notice separately.
   The CSV column takes the tenancy start date (or the execution date if no separate start is
   stated), and the termination notice (or the renewal notice if there is no termination
   clause). These choices follow what the training labels mean (`docs/schema.md`). They pick
   between the model's own outputs and never look at the document.

## Installation (Windows 10/11, PowerShell)

Requirements:
- Conda (Anaconda or Miniconda)
- About 8 GB of free disk (environment about 2 GB; model 4.7 GB; OCR weights about 0.1 GB)
- An NVIDIA GPU with ≥ 8 GB VRAM is recommended for the LLM (a CPU works, much more slowly)

**1. Python environment**

```bash
conda create -n rental-extraction python=3.11 -y
```
```bash
conda activate rental-extraction
```
Exact tested versions (recommended):
```bash
python -m pip install -r requirements-lock.txt
```
Or compatible ranges:
```bash
python -m pip install -r requirements.txt
```

**2. Ollama and the model**

Install Ollama from https://ollama.com/download (tested with 0.40.11). Make sure the Ollama
app is running, then:

```bash
ollama pull qwen2.5:7b
```

Optional: to store the 4.7 GB model somewhere else (e.g. if C: is full), set this before
pulling, then restart Ollama:
`[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\ollama\models", "User")`.

**3. Check the setup**

```bash
python -m src.check_setup
```
```bash
python -m pytest
```

`check_setup` verifies Python 3.11, the packages, the GPU, that the Ollama server is reachable
with the model pulled, and the dataset files. `pytest` runs the unit and offline smoke tests
without the model. To also run one end-to-end smoke test against the real local model, set
`$env:RUN_LLM_SMOKE = "1"` first. The first OCR run downloads PaddleOCR's weights
(about 100 MB) to `%USERPROFILE%\.paddlex`.

## Usage

**Predict a folder of agreements, or a single DOCX / PNG file** (writes a CSV with the dataset's columns):

```bash
python -m src.predict --input data/test --output outputs/predictions.csv
```
```bash
python -m src.predict --input data/test/24158401-Rental-Agreement.png --output outputs/one_prediction.csv
```

- Per-document evidence, warnings and errors go to `outputs/extracted_text/<folder>/`.
- Extracted text is cached, so OCR (about 3–10 minutes per scan on CPU) runs once; add
  `--force` to redo it.
- The command exits with code 1 if any document failed.

**Score predictions against the test labels:**

```bash
python -m src.evaluate --split test --final --predictions outputs/predictions.csv
```

`--final` is required because test labels are reserved for final evaluation. The reports go to
`outputs/evaluation/test/`.

**Reproduce the whole final evaluation in one command.** It checks the frozen configuration,
re-runs OCR, extracts and scores, and writes `outputs/predictions.csv`,
`outputs/test_metrics.json` and `outputs/test_error_analysis.csv`:

```bash
python -m src.final_evaluation
```

Other entry points:

| Command | Purpose |
|---|---|
| `python -m src.extract_text --split all` | text extraction only (DOCX + OCR) |
| `python -m src.extractor --split train --documents <id>` | extract one training document and print the evidence |
| `python -m src.evaluate --split train` | score training predictions |
| `python -m src.experiment --name <run> --config v3` | development experiment on training data (logged) |
| `python -m src.llm_benchmark` | JSON test, speed and VRAM benchmark for the local LLM |
| `python -m src.freeze` | confirm the frozen configuration is unchanged |
| `jupyter lab notebooks/data_exploration.ipynb` | dataset audit |

## REST API

The same frozen pipeline is also available as a web service (`src/api.py`, FastAPI). Start it
with Ollama running:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Interactive documentation, where you can upload a file from the browser: http://127.0.0.1:8000/docs

| Endpoint | Description |
|---|---|
| `GET /health` | Server status, whether the local model is reachable, model name, configuration |
| `POST /extract` | Upload **one** agreement as multipart field `file` (`.docx` or `.png`, max 20 MB). Returns the six fields, an evidence quote per field, and any warnings |

Example (PowerShell, using the curl that ships with Windows):

```bash
curl.exe -X POST http://127.0.0.1:8000/extract -F "file=@data/test/228094620-Rental-Agreement.pdf.docx"
```

Response (shortened):

```json
{
  "document_id": "228094620-Rental-Agreement",
  "document_type": "docx",
  "status": "ok",
  "prediction": {
    "Aggrement Value": 15000,
    "Aggrement Start Date": "07.07.2013",
    "Aggrement End Date": "06.06.2014",
    "Renewal Notice (Days)": 30,
    "Party One": "KAPIL MEHROTRA",
    "Party Two": "B.Kishore , S/o. Mr. B. Pampaiah"
  },
  "evidence": {
    "Aggrement Value": {"value": 15000, "evidence": "Rs. 15,000.00 (Rupees Fifteen Thousand Only)", "evidence_found": true}
  },
  "warnings": [],
  "error": null,
  "ingestion_seconds": 0.01,
  "extraction_seconds": 15.9,
  "configuration": "v3"
}
```

| HTTP status | When |
|---|---|
| 200 | Processed. `status` is `"ok"`, or `"failed"` with `error` and all fields `null` (nothing is invented) |
| 413 | File larger than 20 MB |
| 415 | Not a `.docx` or `.png` |
| 422 | The file cannot be read (corrupt DOCX, not an image, empty file); `detail` says why |
| 503 | The local model is not reachable (Ollama not running or model not pulled) |

Notes:
- The server binds to `127.0.0.1`, so it is reachable only from this computer.
- Uploaded files are written to a temporary folder that is deleted after each request; nothing is stored.
- Requests are processed one at a time, because the OCR engine and the GPU model are shared.
- A DOCX takes about 15–55 s. A tall scanned PNG needs OCR first, which takes several minutes on CPU.

Optional settings (environment variables or a `.env` file): `LLM_MODEL` (default
`qwen2.5:7b`), `LLM_NUM_CTX` (default 16384), `OLLAMA_HOST` (default `http://localhost:11434`).
Changing them changes the frozen configuration.

## Evaluation method

- **Recall per field** = exact matches ÷ documents with a non-blank reference for that field.
- A missing prediction, a failed extraction or an empty value counts as **not matched**.
- Blank references are excluded from the denominator and reported.
- Raw matching compares cells **character for character**; normalised matching is reported separately.
- **Macro average** = mean of the six field recalls.

Full definition and the denominator per field: `docs/evaluation.md`.

## Dataset notes

Found in the audit (`notebooks/data_exploration.ipynb`, `outputs/dataset_audit.json`), with
the original files left untouched:

- The `train.csv` row for `24158401` describes a document in `test/`. It was excluded from
  development.
- The labels of `54770958` and `54945838` are **swapped** relative to the documents.
- Three reference end dates do not exist (`31.11.2009`, `31.04.2011`, `31.02.2011`).
- Many reference cells have leading or trailing spaces.
- "Renewal Notice" labels follow the **termination** notice: `47854715` has both clauses, and its label is the
  termination one.
- `6683127` is page 1 of `6683129`.
- One training DOCX has an unreadable rent amount in its text.

## Development and leakage control

- Only training documents were used for development: 7 for tuning and 2 held out, which were scored once.
- Four configurations were tried, and each run's exact prompt and scores are logged
  (`outputs/experiments/`).
- The selected configuration (v3) was frozen before the test run. Its fingerprint
  (`config/frozen_config.json`) covers the prompt, schema, normaliser, LLM and OCR settings and is
  checked by a test.
- No test-label value appears in the code, prompts or development outputs (checked before the
  test run). `test.csv` was printed once during project setup and not used afterwards.

## Limitations

- **Party names**: relations ("S/o …", "son of …") often stay in the name, so raw exact match
  on party fields is low. Spelling follows the document, which may differ from the label.
- **Fabrication on garbled text**: on badly OCR'd source text the 7B model can invent a value
  (e.g. rent `9997` from `Rs 9.99.7.9°`). A verification pass that removes such values was
  tried; it also removed correct values, so it was not selected for this recall-only metric.
- **Speed**: OCR takes about 3–10 minutes per tall scan on CPU (oneDNN acceleration crashes on Windows),
  and the LLM about 15–55 s per document.
- **Labels**: label conventions (12- vs 11-month terms, impossible dates, spacing) limit exact
  match. With 13 labelled documents in total, all metrics have very wide uncertainty.
- Single-column layouts are assumed; handwriting, stamps and rotated scans are not handled.
- The REST API processes one request at a time and has no authentication; it is meant for local
  use, not for exposure on a network.

## Repository structure

```
├── data/                      provided dataset (unchanged): train/, test/, train.csv, test.csv
├── src/
│   ├── config.py              paths and CSV column names
│   ├── dataset.py             label loading, document IDs, train/test overlap
│   ├── audit.py               dataset audit functions
│   ├── document_reader.py     DOCX reader + unified read_document()
│   ├── ocr.py                 PaddleOCR with tiling, diagnostics, errors
│   ├── extract_text.py        ingestion pipeline with per-document status
│   ├── llm.py                 Ollama client (schema-constrained JSON, truncation guard)
│   ├── schemas.py             Pydantic extraction schema + submission record
│   ├── extractor.py           prompts, retries, salvage, evidence check
│   ├── normalize.py           evaluation-only normalisation
│   ├── evaluate.py            per-field recall, reports
│   ├── predict.py             CLI: file or folder → predictions CSV
│   ├── api.py                 REST API (FastAPI): /health, /extract
│   ├── final_evaluation.py    one-shot frozen test evaluation
│   ├── experiment.py          development experiments with log
│   ├── freeze.py              frozen-configuration fingerprint
│   ├── llm_benchmark.py       LLM speed / memory / context check
│   └── check_setup.py         environment check
├── tests/                     unit tests + smoke tests (pytest)
├── notebooks/                 dataset audit notebook
├── config/frozen_config.json  fingerprint of the frozen configuration
├── outputs/                   predictions.csv, test_metrics.json, test_error_analysis.csv,
│                              evaluation/, experiments/, dataset_audit.json, llm_benchmark.json
├── docs/                      design notes per component (see below)
├── requirements.txt           direct dependencies (ranges)
├── requirements-lock.txt      exact tested versions
└── environment.yml            conda environment
```

Documentation: `docs/data_notes.md` (dataset), `docs/docx_reader.md`, `docs/ocr.md`,
`docs/llm_setup.md`, `docs/schema.md`, `docs/extractor.md`, `docs/evaluation.md`,
`docs/experiments.md`, `docs/test_results.md`.

## Disclaimer

Built for an AI/ML internship assignment. Extracted values should be checked by a person
before being used for legal or financial decisions. No license has been specified, so all rights are
reserved.
