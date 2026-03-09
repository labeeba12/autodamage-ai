import pytest
import numpy as np

from src.logic.overlap_engine import (
    calculate_iou,
    calculate_overlap_ratio,
    find_damage_on_parts
)


# Fixtures 
@pytest.fixture
def full_mask():
    """A mask covering the entire 100x100 image."""
    return np.ones((100, 100), dtype=bool)


@pytest.fixture
def half_mask():
    """A mask covering the left half of a 100x100 image."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, :50] = True
    return mask


@pytest.fixture
def small_mask():
    """A small mask in the top-left corner."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[:10, :10] = True
    return mask


@pytest.fixture
def empty_mask():
    """An empty mask with no pixels."""
    return np.zeros((100, 100), dtype=bool)


@pytest.fixture
def sample_parts():
    """Sample car parts detections."""
    hood_mask = np.zeros((100, 100), dtype=bool)
    hood_mask[10:60, 10:90] = True

    bumper_mask = np.zeros((100, 100), dtype=bool)
    bumper_mask[70:95, 10:90] = True

    return [
        {
            "class_id":   0,
            "class_name": "Hood",
            "confidence": 0.92,
            "mask":       hood_mask,
            "pixel_area": int(hood_mask.sum()),
            "area_ratio": float(hood_mask.sum()) / 10000
        },
        {
            "class_id":   1,
            "class_name": "Front-bumper",
            "confidence": 0.88,
            "mask":       bumper_mask,
            "pixel_area": int(bumper_mask.sum()),
            "area_ratio": float(bumper_mask.sum()) / 10000
        }
    ]


@pytest.fixture
def sample_damages():
    """Sample damage detections on the hood."""
    scratch_mask = np.zeros((100, 100), dtype=bool)
    scratch_mask[20:30, 20:40] = True  # clearly on hood

    return [
        {
            "class_id":   0,
            "class_name": "scratch",
            "confidence": 0.85,
            "mask":       scratch_mask,
            "pixel_area": int(scratch_mask.sum()),
            "area_ratio": float(scratch_mask.sum()) / 10000
        }
    ]


# IoU tests
class TestIoU:

    def test_identical_masks_iou_is_one(self, full_mask):
        """Two identical masks should have IoU of 1.0."""
        iou = calculate_iou(full_mask, full_mask)
        assert iou == pytest.approx(1.0)

    def test_no_overlap_iou_is_zero(self, half_mask):
        """Non-overlapping masks should have IoU of 0.0."""
        other = np.zeros((100, 100), dtype=bool)
        other[:, 50:] = True
        iou = calculate_iou(half_mask, other)
        assert iou == pytest.approx(0.0)

    def test_empty_masks_iou_is_zero(self, empty_mask):
        """Two empty masks should return IoU of 0.0."""
        iou = calculate_iou(empty_mask, empty_mask)
        assert iou == 0.0

    def test_iou_between_zero_and_one(self, half_mask, full_mask):
        """IoU should always be between 0 and 1."""
        iou = calculate_iou(half_mask, full_mask)
        assert 0.0 <= iou <= 1.0


#   Overlap ratio tests
class TestOverlapRatio:

    def test_full_overlap_ratio_is_one(self, small_mask, full_mask):
        """Small mask fully inside large mask = overlap ratio 1.0."""
        ratio = calculate_overlap_ratio(small_mask, full_mask)
        assert ratio == pytest.approx(1.0)

    def test_no_overlap_ratio_is_zero(self, half_mask):
        """Non-overlapping masks = overlap ratio 0.0."""
        other = np.zeros((100, 100), dtype=bool)
        other[:, 50:] = True
        ratio = calculate_overlap_ratio(half_mask, other)
        assert ratio == pytest.approx(0.0)

    def test_empty_damage_mask_returns_zero(self, empty_mask, full_mask):
        """Empty damage mask should return 0.0."""
        ratio = calculate_overlap_ratio(empty_mask, full_mask)
        assert ratio == 0.0


# Find damage on parts tests
class TestFindDamageOnParts:

    def test_damage_matched_to_correct_part(self, sample_parts, sample_damages):
        """Scratch on hood should be matched to Hood part."""
        results = find_damage_on_parts(sample_parts, sample_damages)
        assert len(results) == 1
        assert results[0]["damage_type"] == "scratch"
        assert results[0]["part_name"]   == "Hood"

    def test_empty_damage_returns_empty(self, sample_parts):
        """No damage detections should return empty list."""
        results = find_damage_on_parts(sample_parts, [])
        assert results == []

    def test_empty_parts_returns_unknown(self, sample_damages):
        """No parts detected should label damage as Unknown Part."""
        results = find_damage_on_parts([], sample_damages)
        assert results[0]["part_name"] == "Unknown Part"

    def test_result_has_required_keys(self, sample_parts, sample_damages):
        """Each result should contain all required fields."""
        results = find_damage_on_parts(sample_parts, sample_damages)
        required = [
            "damage_type", "part_name", "confidence",
            "overlap_ratio", "pixel_area", "area_ratio"
        ]
        for key in required:
            assert key in results[0]