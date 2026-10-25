# DOCX reader (`src/document_reader.py`)

## What it does

`read_docx(path)` returns an `ExtractedText`. `read_document(path)` returns the
same type for both DOCX and PNG and also fills in the processing time:

| Field | Meaning |
|---|---|
| `document_id` | ID used in the CSV `File Name` column (`.pdf.docx` → both extensions dropped) |
| `filename` | Source file name, e.g. `47854715-RENTAL-AGREEMENT.docx` |
| `document_type` | `"docx"` or `"png"` |
| `text` | Plain text, one line per paragraph or table row |
| `warnings` | Things the reader saw but could not turn into text |
| `seconds` | Processing time (set by `read_document`) |
| `mean_confidence` | OCR only; `None` for DOCX |
| `diagnostics` | OCR only: per-line text, confidence and box (see `docs/ocr.md`) |

The document body is walked in XML order, not through python-docx's
`Document.paragraphs`, so these are all kept in reading order:

- **Paragraphs**: one line each; empty and whitespace-only paragraphs are skipped.
  Tabs become spaces, manual line breaks stay as line breaks.
- **Tables**: one line per row, cells joined with ` | `. Nested tables are read
  inside their cell.
- **Text boxes**: emitted right after the paragraph they are anchored in. Word
  often stores each box twice (a modern copy plus an `mc:Fallback` copy for old
  readers); the fallback copy is skipped so text is not duplicated.
- **Content controls** (`w:sdt`): their contents are read like normal blocks.

The reader extracts no metadata and uses no pattern matching on the text.

## Errors

Unreadable files raise `DocumentReadError` (a `ValueError`) with `.path` and a
plain-language `.reason`:

| Situation | Reason |
|---|---|
| Path does not exist | `file not found` |
| 0-byte file | `file is empty (0 bytes)` |
| Not a ZIP package (legacy `.doc` renamed, password-protected, corrupted) | `not a DOCX (ZIP) package; ...` |
| ZIP but not a valid Word package | `malformed DOCX package (<library error>)` |
| Unsupported extension in `read_document` | `unsupported file type '.pdf' (expected .docx or .png)` |

## Warnings

| Warning | Why it matters |
|---|---|
| `no text found in the document body` | Nothing for the model to read. |
| `very little text found (N chars)` | Fewer than 100 non-space characters: the content is probably a picture of text. |
| `N embedded image(s): any text inside them is not extracted` | Stamps, signatures or scanned pages pasted in as pictures are invisible to the reader. |
| `N text box(es) read; their position in the reading order is approximate` | Text boxes float on the page; their anchor paragraph is not always where they appear visually. |
| `header/footer text not included: ...` | Headers/footers repeat on every page and here contain page numbers and template boilerplate, so they are excluded to avoid noise. |
| `footnote/endnote/comment text not included: ...` | These parts are not read. |

## Limitations

1. **Text inside images is not extracted.** The reader does not run OCR on
   embedded pictures. In this dataset only `44737744` has images (two revenue /
   notary stamps with no contract terms), so nothing relevant is lost, but a DOCX
   that is just a scanned page pasted into Word would produce no text. Such files
   get the "no text" or "very little text" warning. Running OCR on embedded images
   is a possible extension.
2. **Shapes and text boxes have approximate positions.** Text-box content is placed
   after its anchor paragraph, which may differ from where it is drawn. For example,
   in `6683129` the date fragments `"day of May,"` and `"2007"` are in separate text
   boxes, so the date is not contiguous in the output.
3. **Headers, footers, footnotes, endnotes and comments are not included.** They are
   only reported in the warnings.
4. **Tracked changes:** inserted text is included and deleted text is not, which is
   the final state of the document. Pending revisions are not flagged.
5. **No layout information:** columns, indentation, fonts and bold/underline
   (often used for party names) are lost. Multi-column layouts are read in XML
   order, which is column by column.
6. **Charts, SmartArt and WordArt** are not read. Drawings without a picture or
   text box are not counted as images.
7. **The source text quality is inherited.** Several DOCX files in this dataset were
   produced from PDFs by OCR (`44737744`: `Rs 9.99.7.9°`, `rEST"° LANDLORD`). The
   reader returns that text as-is and does not correct it.
8. **Legacy `.doc` files and password-protected files are not supported.** They are
   rejected with a clear `DocumentReadError`.

## Dataset check (all 12 DOCX files)

- 0 files with no text.
- 1 file with embedded images: `44737744`, which contains 2 stamps.
- 6 files with text boxes, all in train (21 text boxes in total; none have fallback copies).
- 5 files with header/footer text: page numbers, plus a template header and footer in `46239065`.
- 0 tracked changes.
