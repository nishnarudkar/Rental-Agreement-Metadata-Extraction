"""OCR for scanned agreements using PaddleOCR (PP-OCR, CPU).

PaddleOCR returns one entry per detected text fragment with a bounding box.
We rebuild reading order from those boxes: fragments on the same visual line
are joined left-to-right, and a blank line is inserted where the vertical gap
is large, so paragraph breaks survive for the language model.
"""

import os
from functools import lru_cache
from pathlib import Path
from statistics import median

# Skip PaddleX's network probe of model hosts on every start-up.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# A gap larger than this many median line heights starts a new paragraph.
PARAGRAPH_GAP = 1.5


@lru_cache(maxsize=1)
def _engine():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="en",
        # Scans in this dataset are upright, flat pages; these extra models
        # only add time.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # oneDNN crashes on Windows with PaddlePaddle 3.x
        # ("ConvertPirAttribute2RuntimeAttribute not support").
        enable_mkldnn=False,
    )


def _group_lines(fragments: list[tuple[str, list[float]]]) -> str:
    """Order (text, [x1, y1, x2, y2]) fragments into lines and paragraphs."""
    if not fragments:
        return ""
    fragments = sorted(fragments, key=lambda f: ((f[1][1] + f[1][3]) / 2, f[1][0]))
    line_height = median(b[3] - b[1] for _, b in fragments)

    lines: list[list[tuple[str, list[float]]]] = []
    for frag in fragments:
        center = (frag[1][1] + frag[1][3]) / 2
        if lines:
            last = lines[-1]
            last_center = sum((b[1] + b[3]) / 2 for _, b in last) / len(last)
            if abs(center - last_center) < line_height / 2:
                last.append(frag)
                continue
        lines.append([frag])

    out: list[str] = []
    prev_bottom = None
    for line in lines:
        top = min(b[1] for _, b in line)
        if prev_bottom is not None and top - prev_bottom > PARAGRAPH_GAP * line_height:
            out.append("")
        out.append(" ".join(t for t, _ in sorted(line, key=lambda f: f[1][0])))
        prev_bottom = max(b[3] for _, b in line)
    return "\n".join(out)


def ocr_image(path: Path) -> tuple[str, float]:
    """Return (text, mean recognition confidence) for one image."""
    result = _engine().predict(str(path))[0]
    texts, scores, boxes = result["rec_texts"], result["rec_scores"], result["rec_boxes"]
    fragments = [(t, [float(v) for v in b]) for t, b in zip(texts, boxes) if t.strip()]
    confidence = float(sum(scores) / len(scores)) if len(scores) else 0.0
    return _group_lines(fragments), confidence
