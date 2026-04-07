import cv2
import numpy as np
from src.utils.logger import get_logger
from config import (
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
    BLUR_THRESHOLD
)

logger = get_logger(__name__)


def validate_image(image_path: str) -> tuple[bool, str]:
    """
    Validates an image before running inference.
    Returns (True, "success message") or (False, "reason for failure")

    Checks:
        1. File exists
        2. File is a valid image
        3. Image meets minimum resolution
        4. Image is not too blurry
    """

    # CHECK 1: Does the file exist?
    import os
    if not os.path.exists(image_path):
        msg = f"File not found: {image_path}"
        logger.error(msg)
        return False, msg


    # CHECK 2: Is it a valid image?
    img = cv2.imread(image_path)
    if img is None:
        msg = "Invalid or corrupt image file. Please upload a valid JPG or PNG."
        logger.error(msg)
        return False, msg

    # CHECK 3: Is resolution acceptable?
    h, w = img.shape[:2]
    if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
        msg = (
            f"Image too small ({w}x{h}px). "
            f"Minimum required: {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}px."
        )
        logger.warning(msg)
        return False, msg

    # CHECK 4: Is it too blurry?
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < BLUR_THRESHOLD:
        msg = (
            f"Image too blurry (score: {blur_score:.2f}). "
            f"Please upload a clearer photo."
        )
        logger.warning(msg)
        return False, msg

    # All checks passed
    logger.info(f"Image validated — {w}x{h}px | Blur score: {blur_score:.2f}")
    return True, "Image validated successfully"


def load_image(image_path: str) -> np.ndarray:
    """
    Loads and returns image as numpy array (BGR format).
    Only call this after validate_image() passes.
    """
    img = cv2.imread(image_path)
    logger.info(f"Image loaded: {image_path}")
    return img