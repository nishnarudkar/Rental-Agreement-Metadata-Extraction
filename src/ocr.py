"""OCR for scanned agreements using PaddleOCR (PP-OCR, CPU).

Tall pages are split into overlapping horizontal strips that are OCR'd at full
resolution. Without this, PaddleOCR shrinks any image whose long side exceeds
4000 px (one scan here is 2550x6600 and would be read at 61% scale). Strips
are only cropped, never altered, so no preprocessing touches the pixels.

Each detected fragment is kept by exactly one strip (the one that "owns" its
vertical centre), so text in the overlap is not duplicated. Fragments are then
ordered into lines and paragraphs for the language model.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.document_reader import DocumentReadError

# Skip PaddleX's network probe of model hosts on every start-up.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

log = logging.getLogger(__name__)

# Strip height stays well below PaddleOCR's 4000 px resize limit; the overlap
# is larger than twice the tallest text line in these scans (~60 px), so a line
# cut by one strip's edge is whole, and owned, in the next strip.
TILE_HEIGHT = 1600
TILE_OVERLAP = 200

# A gap larger than this many median line heights starts a new paragraph.
PARAGRAPH_GAP = 1.5

LOW_LINE_CONFIDENCE = 0.80
LOW_MEAN_CONFIDENCE = 0.90


class OcrError(DocumentReadError):
    """An image could not be OCR'd."""


@dataclass
class OcrLine:
    text: str
    score: float
    box: list[int]  # x1, y1, x2, y2 in full-image pixels


@dataclass
class OcrResult:
    text: str
    lines: list[OcrLine]
    image_size: tuple[int, int]  # width, height
    tiles: int
    seconds: float
    warnings: list[str] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float | None:
        return sum(l.score for l in self.lines) / len(self.lines) if self.lines else None

    def diagnostics(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "tiles": self.tiles,
            "seconds": self.seconds,
            "line_count": len(self.lines),
            "mean_confidence": self.mean_confidence,
            "min_confidence": min((l.score for l in self.lines), default=None),
            "low_confidence_lines": [l.__dict__ for l in self.lines if l.score < LOW_LINE_CONFIDENCE],
            "warnings": self.warnings,
            "lines": [l.__dict__ for l in self.lines],
        }


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
        cpu_threads=min(os.cpu_count() or 8, 16),
    )


def plan_tiles(height: int, tile: int = TILE_HEIGHT, overlap: int = TILE_OVERLAP) -> list[tuple[int, int, int, int]]:
    """Strips as (start, end, own_start, own_end) row ranges covering ``height``.

    A fragment belongs to the strip whose [own_start, own_end) contains its
    vertical centre; owned ranges tile the image exactly once.
    """
    if height <= tile:
        return [(0, height, 0, height)]
    tiles, start = [], 0
    while True:
        end = min(start + tile, height)
        tiles.append([start, end])
        if end == height:
            break
        start = end - overlap
    planned = []
    for i, (start, end) in enumerate(tiles):
        own_start = 0 if i == 0 else start + overlap // 2
        own_end = height if i == len(tiles) - 1 else end - overlap // 2
        planned.append((start, end, own_start, own_end))
    return planned


def load_image(path: Path) -> np.ndarray:
    """Read an image as a 3-channel BGR array (PaddleOCR's input format)."""
    if not path.exists():
        raise OcrError(path, "file not found")
    if path.stat().st_size == 0:
        raise OcrError(path, "file is empty (0 bytes)")
    try:
        with Image.open(path) as image:
            image.load()  # force full decode so truncated files fail here
            rgb = image.convert("RGB")  # grayscale -> 3 identical channels
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise OcrError(path, f"not a readable image ({type(exc).__name__}: {exc})") from exc
    if rgb.width == 0 or rgb.height == 0:
        raise OcrError(path, "image has no pixels")
    return np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])


def _ocr_tile(image: np.ndarray) -> list[OcrLine]:
    result = _engine().predict(image)[0]
    return [
        OcrLine(text=t, score=float(s), box=[int(v) for v in b])
        for t, s, b in zip(result["rec_texts"], result["rec_scores"], result["rec_boxes"])
        if t.strip()
    ]


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


def _warnings(lines: list[OcrLine]) -> list[str]:
    if not lines:
        return ["no text detected in the image"]
    warnings = []
    mean = sum(l.score for l in lines) / len(lines)
    if mean < LOW_MEAN_CONFIDENCE:
        warnings.append(f"low mean OCR confidence ({mean:.2f})")
    low = sum(l.score < LOW_LINE_CONFIDENCE for l in lines)
    if low:
        warnings.append(f"{low} OCR line(s) below {LOW_LINE_CONFIDENCE} confidence")
    return warnings


def ocr_image(path: Path) -> OcrResult:
    """OCR one image; raises OcrError with a clear reason on failure."""
    image = load_image(path)
    height, width = image.shape[:2]
    tiles = plan_tiles(height)
    start_time = time.perf_counter()
    lines: list[OcrLine] = []
    for start, end, own_start, own_end in tiles:
        try:
            fragments = _ocr_tile(image[start:end])
        except Exception as exc:  # engine start-up or inference failure
            log.error("OCR failed for %s rows %d-%d: %s", path.name, start, end, exc)
            raise OcrError(path, f"OCR engine failed on rows {start}-{end} ({type(exc).__name__}: {exc})") from exc
        for frag in fragments:
            frag.box[1] += start
            frag.box[3] += start
            if own_start <= (frag.box[1] + frag.box[3]) / 2 < own_end:
                lines.append(frag)

    lines.sort(key=lambda l: ((l.box[1] + l.box[3]) / 2, l.box[0]))
    result = OcrResult(
        text=_group_lines([(l.text, l.box) for l in lines]),
        lines=lines,
        image_size=(width, height),
        tiles=len(tiles),
        seconds=round(time.perf_counter() - start_time, 1),
        warnings=_warnings(lines),
    )
    log.info(
        "OCR %s: %dx%d, %d tile(s), %d lines, mean confidence %s, %.1fs",
        path.name, width, height, result.tiles, len(lines),
        f"{result.mean_confidence:.3f}" if lines else "n/a", result.seconds,
    )
    for warning in result.warnings:
        log.warning("OCR %s: %s", path.name, warning)
    return result
