import cv2
import numpy as np
from ultralytics import YOLO
from src.utils.logger import get_logger
from config import (
    PARTS_MODEL_PATH,
    DAMAGE_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    CONFIDENCE_THRESH,
    IMAGE_SIZE,
    MIN_SEGMENT_AREA
)

logger = get_logger(__name__)


class CarSegmenter:
    """
    Loads all three YOLO models once and reuses them for every request.
    Loading models is expensive — we do it once at startup.
    """

    def __init__(self):
        logger.info("Loading car parts segmentation model...")
        self.parts_model      = YOLO(PARTS_MODEL_PATH)
        logger.info("Loading car damage segmentation model...")
        self.damage_model     = YOLO(DAMAGE_MODEL_PATH)
        logger.info("Loading car classifier model...")
        self.classifier_model = YOLO(CLASSIFIER_MODEL_PATH)
        logger.info("All three models loaded successfully!")

    # Internal helper
    def _run_model(self, model: YOLO, image_path: str) -> list[dict]:
        """
        Runs a single YOLO segmentation model on an image.
        Returns a list of detections, each containing:
            - class_id
            - class_name
            - confidence
            - mask (binary numpy array, same size as image)
            - pixel_area (number of pixels in mask)
            - area_ratio (mask area / total image area)
        """
        results = model.predict(
            source  = image_path,
            conf    = CONFIDENCE_THRESH,
            imgsz   = IMAGE_SIZE,
            save    = False,
            verbose = False
        )

        detections = []
        result     = results[0]

        # No detections found
        if result.masks is None:
            return detections

        # Get image dimensions
        img_h, img_w = result.orig_shape
        total_pixels = img_h * img_w
        logger.info(f"Model: {list(model.names.values())[0]} | orig_shape: {img_h}x{img_w}")

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

        logger.info(f"{len(detections)} detections found")
        return detections

    # Public methods
    def detect_parts(self, image_path: str) -> list[dict]:
        """Runs car parts segmentation model."""
        logger.info("Running car parts detection...")
        return self._run_model(self.parts_model, image_path)

    def detect_damage(self, image_path: str) -> list[dict]:
        """Runs car damage segmentation model."""
        logger.info("Running car damage detection...")
        return self._run_model(self.damage_model, image_path)

    def classify_car(self, image_path: str) -> dict:
        """
        Runs car classifier model.
        Returns top 3 predictions with confidence scores.
        """
        logger.info("Running car classifier...")

        results = self.classifier_model.predict(
            source  = image_path,
            imgsz   = 224,
            save    = False,
            verbose = False
        )

        probs       = results[0].probs
        class_names = results[0].names

        top3_indices = probs.top5[:3]
        top3_confs   = probs.top5conf[:3].tolist()
        
        predicted_label = class_names[top3_indices[0]]
        main_confidence = round(float(top3_confs[0]), 4)
        
        # Default status
        is_car = True

        # List of high-priority models from our Pakistani Cars Dataset
        # We allow these even with lower confidence to improve recall
        pakistani_cars = ["aqua", "mehran", "corolla", "civic", "vitz", "city", "alto"] 
        
        # Check if the current prediction matches any known Pakistani model
        is_known_model = any(car in predicted_label.lower() for car in pakistani_cars)

        if main_confidence < 0.5:
            if is_known_model and main_confidence > 0.2:
                # Bypass strict threshold for recognized car models
                is_car = True
                logger.info(f"Low confidence ({main_confidence:.2f}) but matched priority model: {predicted_label}")
            else:
                # Maintain strict 0.5 threshold for unknown objects/labels
                is_car = False

        # Immediate rejection for non-vehicle classes regardless of confidence
        if predicted_label.lower() in ["background", "not_a_car", "person"]:
            is_car = False

        predictions = [
            {
                "rank":       i + 1,
                "model":      class_names[idx],
                "confidence": round(float(conf), 4)
            }
            for i, (idx, conf) in enumerate(zip(top3_indices, top3_confs))
        ]

        result = {
            "predicted_model": predicted_label,
            "confidence":      main_confidence,
            "is_car":          is_car, 
            "top3":            predictions
        }

        logger.info(f"Car classified as: {result['predicted_model']} | Is Car: {is_car}")
        return result

    def detect_all(self, image_path: str) -> tuple[list[dict], list[dict], dict]:
        """
        Runs all three models on the same image.
        Returns (parts_detections, damage_detections, car_classification)
        """
        parts          = self.detect_parts(image_path)
        damage         = self.detect_damage(image_path)
        classification = self.classify_car(image_path)
        return parts, damage, classification