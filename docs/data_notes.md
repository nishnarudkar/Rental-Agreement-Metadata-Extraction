# Data notes

Issues found while inspecting the provided dataset. The original files in
`data/` are left untouched; these notes record how the pipeline deals with them.

## Labels (`train.csv`, `test.csv`)

| Issue | Rows | Handling |
|---|---|---|
| Column names misspelled `Aggrement ...` | all | Kept verbatim (`src/config.py`) so predictions match the reference files. |
| `24158401-Rental-Agreement` is labelled in `train.csv` but its file is in `test/` (and in `test.csv`) | 1 | Treated as test-only. Never used as a few-shot example or for prompt tuning (`src.dataset.leaked_ids`). |
| `46239065-Standard-Rental-Agreement-...` has a file in `train/` but no label row | 1 | Unlabelled; can only be used for qualitative checks. |
| Labels for `54770958` and `54945838` are swapped relative to file content: `54770958.png` is the Asha Ramesh / Sadasivuni agreement (Rs. 5,500) and `54945838.png` is the K. Parthasarathy / Veerabrahmam Bathini agreement (Rs. 8,000) | 2 | Verified visually. Decision on how to handle in train evaluation is pending. |
| `6683127` and `6683129` are two versions of the same contract | 2 | Count as near-duplicates when reasoning about train scores. |
| Impossible calendar dates: `31.11.2009`, `31.04.2011`, `31.02.2011` | 3 (train) | Exact-match scoring against these is impossible for a correct reader; to be documented in evaluation. |
| Leading/trailing whitespace (`' Vishal Bhardwaj '`, `'01.04.2010 '`) and a stray leading dot (`.B.Kishore`) | several | Whitespace stripping planned for evaluation; to be documented. |
| Blank `Renewal Notice (Days)` for `44737744` | 1 | Treated as "no value expected". |

## Documents

- Test DOCX files have a double extension (`.pdf.docx`); the CSV name has neither.
  `src.dataset.document_id` strips both.
- The DOCX files are converted from PDFs. Several keep content inside **text boxes**
  (e.g. `"day of May,"` and `"2007"` in `6683129`), which python-docx's
  `Document.paragraphs` silently skips. `src.document_reader` reads them.
- Some DOCX text is itself OCR output with noise (e.g. `rEST"° LANDLORD` in `44737744`).
- PNG scans are single tall pages (up to 2550x6600) with large blank stamp-paper areas.
  PaddleOCR downsizes the longest side to 4000 px before detection.
