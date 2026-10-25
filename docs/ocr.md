# OCR (`src/ocr.py`)

## Compatibility check (2026-10-25)

| Item | Found | Verdict |
|---|---|---|
| Python | 3.11.17 (conda env `rental-extraction`) | Supported by PaddlePaddle 3.x |
| PaddlePaddle | 3.3.1, **CPU build** (`is_compiled_with_cuda() == False`, 0 CUDA devices) | `paddle.utils.run_check()`: "works well on 1 CPU" |
| PaddleOCR / PaddleX | 3.7.0 / 3.7.2, models PP-OCRv6 medium det + rec | Works |
| GPU | RTX 4070 Laptop, 8 GB, driver 596.21 (supports CUDA ≤ 13.2) | Present, but not used by the installed Paddle |
| CUDA toolkit | not installed (`nvcc` absent, `CUDA_PATH` unset) | |
| CPU | i9-13980HX, 32 logical threads | |
| oneDNN (MKL-DNN) | **crashes** on Windows: `ConvertPirAttribute2RuntimeAttribute not support` | Disabled with `enable_mkldnn=False` |

**Decision: CPU inference.** GPU acceleration would need the `paddlepaddle-gpu`
wheel from Paddle's own package index, possibly a matching CUDA/cuDNN setup,
and it would share the 8 GB of VRAM with the local LLM. There are only 8 scans
and OCR output is cached after the first run, so CPU speed is acceptable and the
setup stays simple to reproduce. The code does not assume a GPU.

## How it works

1. **Load** (`load_image`): Pillow decodes the full file, so truncated files fail
   here. Grayscale is copied into 3 identical channels (BGR for PaddleOCR).
   **No other preprocessing**: no binarisation, deskewing, denoising or resizing.
2. **Tile tall pages** (`plan_tiles`): strips are 1600 px tall with a 200 px
   overlap, at full resolution. PaddleOCR would otherwise shrink any image whose
   long side exceeds 4000 px; `36199312` (2550x6600) would be read at 61% scale.
3. **De-duplicate overlaps**: each fragment is kept only by the strip that "owns"
   its vertical centre. Owned ranges cover the page exactly once, and the 100 px
   half-overlap is larger than the tallest text line (~60 px), so a line cut by
   one strip's edge is complete and owned in the next strip.
4. **Reading order** (`_group_lines`): fragments whose centres are within half a
   median line height are joined left-to-right into one line. A vertical gap
   larger than 1.5 line heights inserts a blank line (paragraph break).
5. **Warnings**: `no text detected`, `low mean OCR confidence (<0.90)`,
   `N OCR line(s) below 0.8 confidence`.

Blank-strip skipping was considered and **rejected**: even the emptiest-looking
strips contain text (the last strip of `54945838` has 0.08% dark pixels but holds
the "LESSEE" signature lines).

## Errors and logging

`OcrError` (a `DocumentReadError`) is raised with a plain-language reason for:
missing file, 0-byte file, not an image, truncated/corrupt image, image with no
pixels, and OCR engine start-up or inference failure (the message includes the
failing row range). Each image logs size, number of strips, number of lines,
mean confidence and time at INFO level, and any warnings at WARNING level.
`python -m src.extract_text` writes the log to `outputs/extracted_text/extraction.log`.

## Saved outputs

| File | Content |
|---|---|
| `<id>.txt` | Reconstructed text |
| `<id>.ocr.json` | Image size, number of strips, time, line count, mean/min confidence, low-confidence lines, warnings, and **every line with its confidence and bounding box** |
| `manifest.json` | Per-document summary including warnings |

## Results on training scans

| Scan | Size | Strips | Lines | Mean conf. | Time (tiled) | Time (old, downscaled) |
|---|---|---|---|---|---|---|
| `36199312` | 2550x6600 | 5 | 46 | 0.996 | 204 s | 111–121 s |
| `54770958` | 1275x4950 | 4 | 42 | 0.995 | 174 s | 120–134 s |
| `54945838` | 1275x4950 | 4 | 63 | 0.989 | 167 s | 126–138 s |

Compared with the earlier whole-image (downscaled) run, tiling changed 5 lines
and none got worse. The termination-notice sentence in `54770958` lost a spurious
split (`Agreem ent` → `Agreement`), and two words in `54945838` were corrected.
The cost is about 45% more time, paid once because the text is cached.

## Manual quality check (train scans; OCR compared with the image pixels)

| Scan | Kind | Image says | OCR output | OK |
|---|---|---|---|---|
| `36199312` | date | Executed today the lst^th of May 2010 | `...the lstth of May` / `2010` | ✓ (the source itself prints "lst") |
| `36199312` | name | 1. Mr. Balaji.R | `1. Mr. Balaji.R` | ✓ |
| `36199312` | name | 1 Mr.Kartheek R | `1 Mr.Kartheek R` | ✓ |
| `36199312` | rent | Rs.3800/- (Rupees Thirty | `Rs.3800/- (Rupees Thirty` | ✓ |
| `36199312` | deposit | Rs.40,000/- | `Rs.40,000/-` | ✓ |
| `36199312` | notice | informed within one month prior notice., | identical | ✓ |
| `54770958` | date + names | executed on 20 April 2011 between: Mrs.Asha Ramesh &Mr.Ramesh.K.N. | identical | ✓ |
| `54770958` | names | Mrs.Sadasivuni Deepti & Mr.Sadasivuni Kiran | identical | ✓ |
| `54770958` | rent | Rs.5500/-(Five thousand five hundred only} | identical | ✓ |
| `54770958` | deposit | Rs. 55,000/- | `Rs. 55,O00/-` | ✗ letter O for zero |
| `54770958` | notice | 2{Two} months' notice ... termination | identical | ✓ |
| `54770958` | term | 11 (months) from the date of Tenancy | `rent agreem ent ... 11 (months)` | ✓ (the source has wide letter spacing) |

11 of 12 key lines are exact. The one error is a non-target amount (the deposit).
Spellings in the scans can differ from the reference labels: `54770958` prints
"Deepti", while the corresponding (swapped) label row spells it "Deepthi".

## Limitations

- **Speed**: about 2.5–3.5 minutes per scan on CPU, because oneDNN acceleration is unusable on
  this platform. Cached text avoids repeating this.
- **Overlapping print**: where two text blocks are printed on top of each other
  (the bottom of `54945838`), the output is garbled (`XANITNEXSRE`). No target fields are there.
- **Character confusions** remain possible (`O`/`0`, `l`/`I`: `lyengar` for "Iyengar").
- Handwriting and stamps are not targeted. Rotated or skewed pages would need the
  orientation and unwarping models, which are disabled.
- Line grouping assumes a single text column; a two-column layout would be interleaved.
- The test-split text in `outputs/extracted_text/test/` was produced by the earlier
  downscaled OCR and has no `.ocr.json`. It is regenerated when the pipeline is
  rerun on test data.
