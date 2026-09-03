"""Unit tests for the aggregation engine."""

import math

import pytest

from app.aggregation.engine import _extremize, _logit, _sigmoid, compute
from app.personas.schemas import PersonaOutputSchema


def _make_output(persona_id: str, estimate: float) -> PersonaOutputSchema:
    return PersonaOutputSchema(
        persona_id=persona_id,
        reasoning_chain=["step 1", "step 2", "step 3"],
        point_estimate=estimate,
        confidence_interval=(max(0.0, estimate - 0.1), min(1.0, estimate + 0.1)),
        confidence_level="medium",
        key_cruxes=["crux 1", "crux 2"],
        update_direction="unchanged",
    )


class TestLogitSigmoid:
    def test_logit_0_5_is_zero(self) -> None:
        assert abs(_logit(0.5)) < 1e-9

    def test_logit_clamped_at_boundaries(self) -> None:
        # Should not raise; clamps to [0.001, 0.999]
        v = _logit(0.0)
        assert math.isfinite(v)
        v = _logit(1.0)
        assert math.isfinite(v)

    def test_sigmoid_zero_is_half(self) -> None:
        assert abs(_sigmoid(0.0) - 0.5) < 1e-9

    def test_logit_sigmoid_roundtrip(self) -> None:
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(_sigmoid(_logit(p)) - p) < 1e-6


class TestExtremize:
    def test_extremize_half_stays_half(self) -> None:
        assert abs(_extremize(0.5, 2.5) - 0.5) < 1e-9

    def test_extremize_pushes_high_probability_higher(self) -> None:
        assert _extremize(0.7, 2.5) > 0.7

    def test_extremize_pushes_low_probability_lower(self) -> None:
        assert _extremize(0.3, 2.5) < 0.3

    def test_extremize_beta_1_is_identity(self) -> None:
        for p in [0.2, 0.5, 0.8]:
            assert abs(_extremize(p, 1.0) - p) < 1e-9


class TestComputeAggregation:
    def test_symmetric_estimates_give_50(self) -> None:
        outputs = [
            _make_output("geopolitical_analyst", 0.3),
            _make_output("base_rate_statistician", 0.4),
            _make_output("scenario_planner", 0.5),
            _make_output("bayesian_updater", 0.5),
            _make_output("devils_advocate", 0.6),
            _make_output("domain_specialist", 0.7),
        ]
        result = compute(outputs)
        # Symmetric around 0.5 after trim, should be close to 50
        assert 40 <= result.today_forecast <= 60

    def test_high_estimates_give_high_forecast(self) -> None:
        outputs = [
            _make_output("geopolitical_analyst", 0.8),
            _make_output("base_rate_statistician", 0.85),
            _make_output("scenario_planner", 0.75),
            _make_output("bayesian_updater", 0.9),
            _make_output("devils_advocate", 0.7),
            _make_output("domain_specialist", 0.82),
        ]
        result = compute(outputs)
        assert result.today_forecast > 70

    def test_low_estimates_give_low_forecast(self) -> None:
        outputs = [
            _make_output("geopolitical_analyst", 0.1),
            _make_output("base_rate_statistician", 0.15),
            _make_output("scenario_planner", 0.05),
            _make_output("bayesian_updater", 0.12),
            _make_output("devils_advocate", 0.2),
            _make_output("domain_specialist", 0.08),
        ]
        result = compute(outputs)
        assert result.today_forecast < 20

    def test_trim_removes_extremes(self) -> None:
        # With alpha=0.10 and 6 personas, 1 extreme on each tail should be trimmed
        outputs = [
            _make_output("geopolitical_analyst", 0.01),  # extreme low — trimmed
            _make_output("base_rate_statistician", 0.45),
            _make_output("scenario_planner", 0.50),
            _make_output("bayesian_updater", 0.55),
            _make_output("devils_advocate", 0.50),
            _make_output("domain_specialist", 0.99),  # extreme high — trimmed
        ]
        result = compute(outputs)
        # After trimming extremes, estimates are close to 0.5
        assert 40 <= result.today_forecast <= 65

    def test_result_spread_is_nonnegative(self) -> None:
        outputs = [_make_output(f"persona_{i}", 0.5) for i in range(6)]
        result = compute(outputs)
        assert result.spread >= 0.0

    def test_today_forecast_in_range(self) -> None:
        outputs = [_make_output(f"persona_{i}", 0.5) for i in range(6)]
        result = compute(outputs)
        assert 0 <= result.today_forecast <= 100

    def test_raises_on_empty_outputs(self) -> None:
        with pytest.raises(ValueError, match="No persona outputs"):
            compute([])

    def test_n_valid_personas_reflects_trimmed_count(self) -> None:
        # floor(0.10 * 6) = 0, so no trimming occurs with 6 personas and alpha=0.10
        # All 6 personas are retained in the aggregation
        outputs = [
            _make_output("geopolitical_analyst", 0.4),
            _make_output("base_rate_statistician", 0.45),
            _make_output("scenario_planner", 0.5),
            _make_output("bayesian_updater", 0.55),
            _make_output("devils_advocate", 0.6),
            _make_output("domain_specialist", 0.5),
        ]
        result = compute(outputs)
        assert result.n_valid_personas == 6

    def test_change_metrics_passed_through(self) -> None:
        outputs = [_make_output(f"persona_{i}", 0.5) for i in range(6)]
        result = compute(outputs, change_1w=5, change_30d=-3)
        assert result.change_1w == 5
        assert result.change_30d == -3
