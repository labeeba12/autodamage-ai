import cv2
import numpy as np
from ultralytics import YOLO
from src.utils.logger import get_logger
from config import (
    PARTS_MODEL_PATH,
    DAMAGE_MODEL_PATH,
    CONFIDENCE_THRESH,
    IMAGE_SIZE,
    MIN_SEGMENT_AREA
)

logger = get_logger(__name__)


class CarSegmenter:
    """
    Loads both YOLO models once and reuses them for every request.
    Loading models is expensive — we do it once at startup.
    """

    def __init__(self):
        logger.info("Loading car parts segmentation model...")
        self.parts_model  = YOLO(PARTS_MODEL_PATH)
        logger.info("Loading car damage segmentation model...")
        self.damage_model = YOLO(DAMAGE_MODEL_PATH)
        logger.info("Both models loaded successfully!")

    # INTERNAL HELPER
    def _run_model(self, model: YOLO, image_path: str) -> list[dict]:
        """
        Runs a single YOLO model on an image.
        Returns a list of detections, each containing:
            - class_id
            - class_name
            - confidence
            - mask (binary numpy array, same size as image)
            - pixel_area (number of pixels in mask)
            - area_ratio (mask area / total image area)
        """
        results = model.predict(
            source=image_path,
            conf=CONFIDENCE_THRESH,
            imgsz=IMAGE_SIZE,
            save=False,
            verbose=False
        )

        detections = []
        result = results[0]

        # No detections found
        if result.masks is None:
            return detections

        # Get image dimensions
        img_h, img_w = result.orig_shape
        total_pixels = img_h * img_w

        for i, mask in enumerate(result.masks):
            box        = result.boxes[i]
            class_id   = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            # Convert mask to binary numpy array
            mask_binary = mask.data[0].cpu().numpy()
            mask_scaled = cv2.resize(
                mask_binary,
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            )
            mask_bool = mask_scaled > 0.5

            # Calculate area
            pixel_area = int(np.sum(mask_bool))
            area_ratio = pixel_area / total_pixels

            # Filter out tiny noise masks
            if area_ratio < MIN_SEGMENT_AREA:
                continue

            detections.append({
                "class_id":   class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "mask":       mask_bool,
                "pixel_area": pixel_area,
                "area_ratio": round(area_ratio, 6)
            })

        logger.info(f"{len(detections)} detections found by {model.names[0]} model")
        return detections

    # PUBLIC METHODS
    def detect_parts(self, image_path: str) -> list[dict]:
        """Runs car parts segmentation model."""
        logger.info("Running car parts detection...")
        return self._run_model(self.parts_model, image_path)

    def detect_damage(self, image_path: str) -> list[dict]:
        """Runs car damage segmentation model."""
        logger.info("Running car damage detection...")
        return self._run_model(self.damage_model, image_path)

    def detect_all(self, image_path: str) -> tuple[list[dict], list[dict]]:
        """
        Runs both models on the same image.
        Returns (parts_detections, damage_detections)
        """
        parts  = self.detect_parts(image_path)
        damage = self.detect_damage(image_path)
        return parts, damage