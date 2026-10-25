"""OCR tests use a fake engine so they run in milliseconds without PaddleOCR."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import ocr
from src.document_reader import read_document
from src.ocr import OcrError, load_image, ocr_image, plan_tiles


def _png(tmp_path: Path, height: int, width: int = 300, name: str = "scan.png") -> Path:
    path = tmp_path / name
    Image.new("L", (width, height), 255).save(path)
    return path


def _fake_engine(monkeypatch, height: int, page_lines: list[tuple[str, float, list[int]]]):
    """Replace the tile OCR with one that 'sees' lines lying fully inside each strip."""
    strips = iter(plan_tiles(height))
    seen_heights = []

    def fake_ocr_tile(image):
        start, end, _, _ = next(strips)
        seen_heights.append(image.shape[0])
        assert image.shape[0] == end - start
        return [
            ocr.OcrLine(text, score, [box[0], box[1] - start, box[2], box[3] - start])
            for text, score, box in page_lines
            if box[1] >= start and box[3] <= end
        ]

    monkeypatch.setattr(ocr, "_ocr_tile", fake_ocr_tile)
    return seen_heights


# --- tiling -------------------------------------------------------------------


def test_short_image_is_one_tile():
    assert plan_tiles(1000) == [(0, 1000, 0, 1000)]


@pytest.mark.parametrize("height", [1601, 3000, 4950, 6600])
def test_tiles_cover_image_and_owned_ranges_partition_it(height):
    tiles = plan_tiles(height)
    assert tiles[0][0] == 0 and tiles[-1][1] == height
    assert all(end - start <= ocr.TILE_HEIGHT for start, end, _, _ in tiles)
    owned = [(own_start, own_end) for _, _, own_start, own_end in tiles]
    assert owned[0][0] == 0 and owned[-1][1] == height
    assert all(a[1] == b[0] for a, b in zip(owned, owned[1:]))  # no gaps, no overlap
    for (start, end, own_start, own_end) in tiles:
        assert start <= own_start < own_end <= end


def test_line_in_overlap_is_kept_once(monkeypatch, tmp_path):
    height = 3000
    first_end = plan_tiles(height)[0][1]
    boundary_line = ("in overlap", 0.99, [10, first_end - 60, 200, first_end - 30])
    page = [("top", 0.99, [10, 10, 100, 40]), boundary_line, ("bottom", 0.98, [10, 2900, 100, 2930])]
    _fake_engine(monkeypatch, height, page)
    result = ocr_image(_png(tmp_path, height))
    assert [line.text for line in result.lines] == ["top", "in overlap", "bottom"]
    assert result.lines[1].box == boundary_line[2]  # mapped back to page coordinates
    assert result.tiles == len(plan_tiles(height))


def test_tiles_are_full_resolution_crops(monkeypatch, tmp_path):
    heights = _fake_engine(monkeypatch, 4950, [])
    ocr_image(_png(tmp_path, 4950, width=1275))
    assert max(heights) == ocr.TILE_HEIGHT


# --- output structure -----------------------------------------------------------


def test_output_structure_and_diagnostics(monkeypatch, tmp_path):
    page = [
        ("RENTAL AGREEMENT", 0.99, [100, 10, 300, 30]),
        ("monthly rent", 0.95, [10, 50, 120, 70]),
        ("Rs.5000", 0.60, [130, 52, 200, 70]),
    ]
    _fake_engine(monkeypatch, 500, page)
    result = ocr_image(_png(tmp_path, 500))
    assert result.text == "RENTAL AGREEMENT\nmonthly rent Rs.5000"
    assert result.image_size == (300, 500)
    assert result.mean_confidence == pytest.approx((0.99 + 0.95 + 0.60) / 3)
    assert result.warnings == ["low mean OCR confidence (0.85)", "1 OCR line(s) below 0.8 confidence"]
    diag = json.loads(json.dumps(result.diagnostics()))  # must be JSON-serialisable
    assert diag["line_count"] == 3
    assert diag["min_confidence"] == 0.60
    assert [l["text"] for l in diag["low_confidence_lines"]] == ["Rs.5000"]
    assert set(diag["lines"][0]) == {"text", "score", "box"}


def test_low_mean_confidence_warns(monkeypatch, tmp_path):
    _fake_engine(monkeypatch, 200, [("blurry", 0.85, [0, 0, 50, 20])])
    assert ocr_image(_png(tmp_path, 200)).warnings == ["low mean OCR confidence (0.85)"]


def test_blank_image_warns(monkeypatch, tmp_path):
    _fake_engine(monkeypatch, 200, [])
    result = ocr_image(_png(tmp_path, 200))
    assert result.text == "" and result.mean_confidence is None
    assert result.warnings == ["no text detected in the image"]


def test_read_document_routes_png_to_ocr(monkeypatch, tmp_path):
    _fake_engine(monkeypatch, 200, [("hello", 0.97, [0, 0, 50, 20])])
    result = read_document(_png(tmp_path, 200))
    assert (result.filename, result.document_type, result.text) == ("scan.png", "png", "hello")
    assert result.mean_confidence == pytest.approx(0.97)
    assert result.diagnostics["line_count"] == 1


# --- image loading and errors ----------------------------------------------------


def test_grayscale_loads_as_unchanged_three_channel(tmp_path):
    path = tmp_path / "gray.png"
    Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)).save(path)  # 2-D uint8 -> mode "L"
    array = load_image(path)
    assert array.shape == (1, 3, 3)
    assert array[0, :, 0].tolist() == [0, 128, 255]  # pixel values untouched


def test_missing_image(tmp_path):
    with pytest.raises(OcrError, match="file not found"):
        load_image(tmp_path / "missing.png")


def test_zero_byte_image(tmp_path):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    with pytest.raises(OcrError, match="empty"):
        load_image(path)


def test_non_image_file(tmp_path):
    path = tmp_path / "notes.png"
    path.write_text("this is not an image")
    with pytest.raises(OcrError, match="not a readable image"):
        load_image(path)


def test_truncated_image(tmp_path):
    full = _png(tmp_path, 400, name="full.png")
    path = tmp_path / "truncated.png"
    path.write_bytes(full.read_bytes()[: full.stat().st_size // 2])
    with pytest.raises(OcrError, match="not a readable image"):
        load_image(path)


def test_engine_failure_is_reported(monkeypatch, tmp_path):
    def broken(image):
        raise RuntimeError("inference crashed")

    monkeypatch.setattr(ocr, "_ocr_tile", broken)
    with pytest.raises(OcrError, match=r"OCR engine failed on rows 0-200 \(RuntimeError: inference crashed\)"):
        ocr_image(_png(tmp_path, 200))
