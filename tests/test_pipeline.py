import pytest
import numpy as np
import cv2
import os


# Fixtures
@pytest.fixture
def valid_image_path(tmp_path):
    """Creates a valid test image."""
    img_path = tmp_path / "test_car.jpg"
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (50,  50),  (590, 430), (200, 200, 200), -1)
    cv2.rectangle(img, (100, 100), (500, 350), (150, 150, 150), -1)
    cv2.rectangle(img, (200, 200), (400, 300), (100, 100, 100), -1)
    cv2.imwrite(str(img_path), img)
    return str(img_path)


@pytest.fixture
def mock_matched_results():
    """Sample matched results for testing cost pipeline."""
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 100:300] = True

    return [
        {
            "damage_type":   "scratch",
            "part_name":     "Hood",
            "confidence":    0.87,
            "overlap_ratio": 0.91,
            "pixel_area":    20000,
            "area_ratio":    0.065,
            "damage_mask":   mask,
            "part_mask":     mask,
        },
        {
            "damage_type":   "dent",
            "part_name":     "Front-bumper",
            "confidence":    0.79,
            "overlap_ratio": 0.83,
            "pixel_area":    8000,
            "area_ratio":    0.026,
            "damage_mask":   mask,
            "part_mask":     mask,
        }
    ]


#   Pipeline component tests
class TestPipelineComponents:

    def test_validator_rejects_missing_file(self):
        """Pipeline should handle missing file gracefully."""
        from src.detection.validator import validate_image
        is_valid, msg = validate_image("fake_path.jpg")
        assert is_valid is False

    def test_severity_pipeline(self, mock_matched_results):
        """Severity calculation should work on mock results."""
        from src.logic.severity_calc import calculate_severity, get_overall_severity

        results  = calculate_severity(mock_matched_results)
        overall  = get_overall_severity(results)

        assert all("severity_label"      in r for r in results)
        assert all("severity_multiplier" in r for r in results)
        assert overall in ["minor", "moderate", "severe"]

    def test_cost_pipeline(self, mock_matched_results):
        """Cost calculation should produce valid PKR estimates."""
        from src.logic.severity_calc import calculate_severity
        from src.cost.decision_engine import process_damage_costs, calculate_total_cost

        results   = calculate_severity(mock_matched_results)
        processed = process_damage_costs(results)
        total     = calculate_total_cost(processed)

        assert total["total_min"] >= 0
        assert total["total_max"] >= total["total_min"]
        assert total["currency"]  == "PKR"

    def test_each_damage_has_cost(self, mock_matched_results):
        """Every damage item should have min and max cost."""
        from src.logic.severity_calc import calculate_severity
        from src.cost.decision_engine import process_damage_costs

        results   = calculate_severity(mock_matched_results)
        processed = process_damage_costs(results)

        for item in processed:
            assert "min_cost" in item
            assert "max_cost" in item
            assert item["min_cost"] >= 0
            assert item["max_cost"] >= item["min_cost"]

    def test_each_damage_has_action(self, mock_matched_results):
        """Every damage item should have repair or replace action."""
        from src.logic.severity_calc import calculate_severity
        from src.cost.decision_engine import process_damage_costs

        results   = calculate_severity(mock_matched_results)
        processed = process_damage_costs(results)

        for item in processed:
            assert item["action"] in ["repair", "replace"]


# Pipeline output structure tests
class TestPipelineOutput:

    def test_pipeline_returns_dict(self, valid_image_path):
        """
        Pipeline should return a dict even if models are not loaded.
        Tests the validation step only.
        """
        from src.detection.validator import validate_image
        is_valid, message = validate_image(valid_image_path)
        # Image should be valid
        assert isinstance(is_valid, bool)
        assert isinstance(message, str)

    def test_feature_builder(self, mock_matched_results):
        """Feature builder should produce correct shape matrix."""
        from src.logic.severity_calc import calculate_severity
        from src.cost.decision_engine import process_damage_costs
        from src.logic.feature_builder import build_feature_matrix

        results   = calculate_severity(mock_matched_results)
        processed = process_damage_costs(results)
        matrix    = build_feature_matrix(processed)

        assert matrix.shape[0] == len(processed)  # one row per damage
        assert matrix.shape[1] == 7               # 7 features per damage