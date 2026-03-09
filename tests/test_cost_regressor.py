import pytest
from src.logic.severity_calc import (
    get_severity_label,
    get_severity_multiplier,
    calculate_severity,
    get_overall_severity
)
from src.cost.decision_engine import (
    decide_action,
    calculate_final_cost
)


# Severity tests
class TestSeverityCalc:

    def test_minor_severity(self):
        """Area ratio below 2% should be minor."""
        assert get_severity_label(0.01) == "minor"

    def test_moderate_severity(self):
        """Area ratio between 2% and 8% should be moderate."""
        assert get_severity_label(0.05) == "moderate"

    def test_severe_severity(self):
        """Area ratio above 8% should be severe."""
        assert get_severity_label(0.15) == "severe"

    def test_boundary_minor_to_moderate(self):
        """Exactly 2% should be moderate."""
        assert get_severity_label(0.02) == "moderate"

    def test_boundary_moderate_to_severe(self):
        """Exactly 8% should be severe."""
        assert get_severity_label(0.08) == "severe"

    def test_multiplier_minor(self):
        """Minor multiplier should be 1.0."""
        assert get_severity_multiplier("minor") == 1.0

    def test_multiplier_moderate(self):
        """Moderate multiplier should be 1.5."""
        assert get_severity_multiplier("moderate") == 1.5

    def test_multiplier_severe(self):
        """Severe multiplier should be 2.2."""
        assert get_severity_multiplier("severe") == 2.2

    def test_calculate_severity_adds_fields(self):
        """calculate_severity should add severity fields to each item."""
        items = [
            {"damage_type": "scratch", "part_name": "Hood",
             "area_ratio": 0.01, "confidence": 0.85,
             "pixel_area": 100, "overlap_ratio": 0.9}
        ]
        result = calculate_severity(items)
        assert "severity_label"      in result[0]
        assert "severity_multiplier" in result[0]
        assert result[0]["severity_label"] == "minor"

    def test_overall_severity_worst_case(self):
        """Overall severity should reflect the worst damage."""
        items = [
            {"severity_label": "minor"},
            {"severity_label": "severe"},
            {"severity_label": "moderate"}
        ]
        assert get_overall_severity(items) == "severe"

    def test_overall_severity_empty(self):
        """Empty list should return none."""
        assert get_overall_severity([]) == "none"


# Decision engine tests
class TestDecisionEngine:

    def test_replace_action_stays_replace(self):
        """If CSV says replace, always replace."""
        action = decide_action("replace", "minor", 0.01)
        assert action == "replace"

    def test_severe_damage_becomes_replace(self):
        """Severe damage should trigger replace."""
        action = decide_action("repair", "severe", 0.05)
        assert action == "replace"

    def test_high_area_ratio_becomes_replace(self):
        """Area ratio above threshold should trigger replace."""
        action = decide_action("repair", "moderate", 0.90)
        assert action == "replace"

    def test_minor_damage_stays_repair(self):
        """Minor damage with low area ratio should stay repair."""
        action = decide_action("repair", "minor", 0.01)
        assert action == "repair"

    def test_cost_calculation_minor(self):
        """Minor severity should not increase base price."""
        base = {"min_price": 5000, "max_price": 15000, "action": "repair"}
        cost = calculate_final_cost(base, 1.0, "repair")
        assert cost["min_cost"] == 5000
        assert cost["max_cost"] == 15000

    def test_cost_calculation_moderate(self):
        """Moderate severity should apply 1.5x multiplier."""
        base = {"min_price": 5000, "max_price": 15000, "action": "repair"}
        cost = calculate_final_cost(base, 1.5, "repair")
        assert cost["min_cost"] == 7500
        assert cost["max_cost"] == 22500

    def test_replace_adds_premium(self):
        """Replace action should add 30% premium."""
        base = {"min_price": 10000, "max_price": 20000, "action": "repair"}
        cost = calculate_final_cost(base, 1.0, "replace")
        assert cost["min_cost"] == 13000
        assert cost["max_cost"] == 26000

    def test_cost_never_negative(self):
        """Cost should never be negative."""
        base = {"min_price": 0, "max_price": 0, "action": "repair"}
        cost = calculate_final_cost(base, 1.0, "repair")
        assert cost["min_cost"] >= 0
        assert cost["max_cost"] >= 0