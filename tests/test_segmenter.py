import pytest
import numpy as np
import os
import cv2

from src.detection.validator import validate_image, load_image
from config import MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT


# Fixtures
@pytest.fixture
def valid_image_path(tmp_path):
    """Creates a valid dummy image for testing."""
    img_path = tmp_path / "test_car.jpg"
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some variation so it's not too blurry
    cv2.rectangle(img, (100, 100), (500, 400), (255, 255, 255), -1)
    cv2.rectangle(img, (200, 200), (400, 350), (128, 128, 128), -1)
    cv2.imwrite(str(img_path), img)
    return str(img_path)


@pytest.fixture
def small_image_path(tmp_path):
    """Creates an image that is too small."""
    img_path = tmp_path / "small.jpg"
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return str(img_path)


@pytest.fixture
def blurry_image_path(tmp_path):
    """Creates a completely blurry (solid color) image."""
    img_path = tmp_path / "blurry.jpg"
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return str(img_path)


# Validator tests
class TestValidator:

    def test_valid_image_passes(self, valid_image_path):
        """A proper image should pass all validation checks."""
        is_valid, message = validate_image(valid_image_path)
        assert is_valid is True
        assert "success" in message.lower() or "validated" in message.lower()

    def test_missing_file_fails(self):
        """A non-existent file should fail validation."""
        is_valid, message = validate_image("non_existent_image.jpg")
        assert is_valid is False
        assert "not found" in message.lower()

    def test_small_image_fails(self, small_image_path):
        """An image below minimum resolution should fail."""
        is_valid, message = validate_image(small_image_path)
        assert is_valid is False
        assert "small" in message.lower() or "minimum" in message.lower()

    def test_blurry_image_fails(self, blurry_image_path):
        """A completely blurry image should fail validation."""
        is_valid, message = validate_image(blurry_image_path)
        assert is_valid is False
        assert "blurry" in message.lower()

    def test_load_image_returns_array(self, valid_image_path):
        """load_image should return a numpy array."""
        img = load_image(valid_image_path)
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3  # BGR channels

    def test_image_meets_minimum_size(self, valid_image_path):
        """Loaded image should meet minimum dimensions."""
        img = load_image(valid_image_path)
        h, w = img.shape[:2]
        assert w >= MIN_IMAGE_WIDTH
        assert h >= MIN_IMAGE_HEIGHT