"""Unit tests for persona orchestration."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.personas.orchestrator import run_personas
from app.personas.schemas import PersonaOutputSchema


def _valid_output(persona_id: str) -> dict:
    return {
        "persona_id": persona_id,
        "reasoning_chain": ["I identified the key actors.", "I consulted historical precedent.", "I arrived at a calibrated estimate."],
        "point_estimate": 0.35,
        "confidence_interval": [0.2, 0.5],
        "confidence_level": "medium",
        "key_cruxes": ["Actor incentives", "Historical base rate"],
        "update_direction": "unchanged",
    }


def _make_mock_response(persona_id: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(_valid_output(persona_id))
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    response.usage = MagicMock()
    response.usage.output_tokens = 100
    return response


class TestPersonaOutputSchema:
    def test_valid_schema(self) -> None:
        data = _valid_output("test_persona")
        output = PersonaOutputSchema.model_validate(data)
        assert output.persona_id == "test_persona"
        assert output.point_estimate == 0.35

    def test_rejects_point_estimate_above_1(self) -> None:
        data = _valid_output("test_persona")
        data["point_estimate"] = 1.5
        with pytest.raises(Exception):
            PersonaOutputSchema.model_validate(data)

    def test_rejects_point_estimate_below_0(self) -> None:
        data = _valid_output("test_persona")
        data["point_estimate"] = -0.1
        with pytest.raises(Exception):
            PersonaOutputSchema.model_validate(data)

    def test_rejects_too_few_reasoning_steps(self) -> None:
        data = _valid_output("test_persona")
        data["reasoning_chain"] = ["only one step"]
        with pytest.raises(Exception):
            PersonaOutputSchema.model_validate(data)

    def test_rejects_too_few_cruxes(self) -> None:
        data = _valid_output("test_persona")
        data["key_cruxes"] = ["only one"]
        with pytest.raises(Exception):
            PersonaOutputSchema.model_validate(data)

    def test_rejects_invalid_confidence_level(self) -> None:
        data = _valid_output("test_persona")
        data["confidence_level"] = "extreme"
        with pytest.raises(Exception):
            PersonaOutputSchema.model_validate(data)

    def test_rejects_invalid_update_direction(self) -> None:
        data = _valid_output("test_persona")
        data["update_direction"] = "sideways"
        with pytest.raises(Exception):
            PersonaOutputSchema.model_validate(data)


class TestRunPersonas:
    @pytest.mark.asyncio
    async def test_all_valid_personas_returned(self) -> None:
        from app.personas.definitions import PERSONA_CONFIGS

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=lambda **kwargs: _make_mock_response(
                # Extract persona_id from the system prompt context — use geopolitical_analyst as default
                "geopolitical_analyst"
            )
        )

        with patch("app.personas.orchestrator.AsyncAnthropic", return_value=mock_client):
            outputs = await run_personas(
                question_text="Will X happen before Y?",
                resolution_criteria="X happens if Z is true.",
                context_digest="Some context here.",
                prior_reasoning_summaries=None,
                run_id="test-run-001",
            )

        assert len(outputs) == len(PERSONA_CONFIGS)
        for output in outputs:
            assert isinstance(output, PersonaOutputSchema)

    @pytest.mark.asyncio
    async def test_invalid_json_is_discarded(self) -> None:
        call_count = 0

        async def bad_response(**kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            block = MagicMock()
            block.type = "text"
            block.text = "This is not JSON at all."
            response = MagicMock()
            response.content = [block]
            response.stop_reason = "end_turn"
            response.usage = MagicMock()
            response.usage.output_tokens = 10
            return response

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=bad_response)

        with patch("app.personas.orchestrator.AsyncAnthropic", return_value=mock_client):
            outputs = await run_personas(
                question_text="Will X happen?",
                resolution_criteria="X happens if Z.",
                context_digest="",
                prior_reasoning_summaries=None,
                run_id="test-run-002",
            )

        # All outputs should be discarded (each persona retries once = 2 calls per persona)
        assert len(outputs) == 0

    @pytest.mark.asyncio
    async def test_quorum_check_triggers_insufficient_quorum(self) -> None:
        """When fewer than MIN_QUORUM personas return valid output, run_forecast marks insufficient_quorum."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from app.config import settings
        from app.services.forecast import run_forecast

        # Mock: only 1 valid persona (below MIN_QUORUM=3)
        single_valid = [
            PersonaOutputSchema(
                persona_id="geopolitical_analyst",
                reasoning_chain=["s1", "s2", "s3"],
                point_estimate=0.5,
                confidence_interval=(0.3, 0.7),
                confidence_level="medium",
                key_cruxes=["crux1", "crux2"],
                update_direction="unchanged",
            )
        ]

        mock_db = AsyncMock()

        with (
            patch("app.services.forecast.fetch_context", return_value=("digest", [])),
            patch("app.services.forecast.run_personas", return_value=single_valid),
            patch("app.services.forecast.crud.get_question") as mock_get_q,
            patch("app.services.forecast.crud.get_latest_snapshot", return_value=None),
            patch("app.services.forecast.crud.create_snapshot") as mock_create_snap,
        ):
            question = MagicMock()
            question.id = "abc123"
            question.text = "Will X happen?"
            question.resolution_criteria = "X happens if Z."
            mock_get_q.return_value = question

            await run_forecast("abc123", mock_db)

        # Should have written a snapshot with run_status=insufficient_quorum
        mock_create_snap.assert_called_once()
        call_kwargs = mock_create_snap.call_args.kwargs
        assert call_kwargs["run_status"] == "insufficient_quorum"
