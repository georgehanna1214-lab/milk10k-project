"""Tests for milk10k.preprocessing (run with: pytest)."""
import numpy as np
import pytest
from PIL import Image

from milk10k.preprocessing import preprocess_batch, preprocess_image


@pytest.fixture
def raw_image():
    """A random 450 x 600 RGB image, the size of the MILK10k images."""
    return np.random.default_rng(0).integers(0, 256, size=(450, 600, 3), dtype=np.uint8)


def test_default_output_shape_and_range(raw_image):
    arr, info = preprocess_image(raw_image, size=224)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0          # "scale" = divide by 255
    assert info["final_shape"] == (224, 224, 3)
    assert info["original_size_wh"] == (600, 450)


def test_gray_has_one_channel(raw_image):
    arr, _ = preprocess_image(raw_image, size=64, color_mode="gray")
    assert arr.shape == (64, 64, 1)


def test_minmax_spans_zero_to_one(raw_image):
    arr, _ = preprocess_image(raw_image, size=64, normalize="minmax")
    assert arr.min() == pytest.approx(0.0) and arr.max() == pytest.approx(1.0)


def test_zscore_with_given_stats():
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)     # every pixel = 128/255 = 0.502
    arr, info = preprocess_image(flat, size=32, normalize="zscore", mean=[0.502] * 3, std=[0.25] * 3)
    assert np.abs(arr).max() < 0.01                        # (0.502 - 0.502) / 0.25 = 0
    assert info["stats_from"].startswith("given")


def test_keep_aspect_crops_instead_of_stretching():
    # left half black, right half white: a centre crop keeps both halves, in the middle
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, 100:] = 255
    arr, _ = preprocess_image(img, size=50, keep_aspect=True)
    assert arr.shape == (50, 50, 3)
    assert arr[:, :20].mean() < 0.1 and arr[:, 30:].mean() > 0.9


def test_batch_skips_bad_files_and_reports_them(tmp_path, raw_image):
    good = tmp_path / "good.jpg"
    Image.fromarray(raw_image).save(good)
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"this is not a jpeg")
    missing = tmp_path / "missing.jpg"

    result = preprocess_batch([good, corrupt, missing], size=32)
    assert result.images.shape == (1, 32, 32, 3)
    assert result.ids == ["good"]
    assert {ident for ident, _ in result.skipped} == {"corrupt", "missing"}


def test_invalid_option_fails_loudly(raw_image):
    with pytest.raises(ValueError):
        preprocess_batch([raw_image], color_mode="purple")
