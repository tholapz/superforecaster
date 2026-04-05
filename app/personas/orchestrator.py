import asyncio
import json

import structlog
from anthropic import AsyncAnthropic
from pydantic import ValidationError

from app.config import settings
from app.personas.definitions import PersonaConfig, get_persona_configs
from app.personas.schemas import PersonaOutputSchema

logger = structlog.get_logger(__name__)

_CORRECTION_PROMPT = (
    "Your previous response was not valid JSON matching the required schema. "
    "Please respond with ONLY the raw JSON object — no prose, no markdown code fences. "
    "The required fields are: persona_id, reasoning_chain (list of strings, min 3), "
    "point_estimate (float 0–1), confidence_interval ([lo, hi]), "
    "confidence_level (low/medium/high), key_cruxes (2–4 strings), "
    "update_direction (up/down/unchanged)."
)


async def _call_persona(
    client: AsyncAnthropic,
    persona: PersonaConfig,
    question_text: str,
    resolution_criteria: str,
    context_digest: str,
    prior_reasoning_summaries: str | None,
    run_id: str,
) -> PersonaOutputSchema | None:
    """Call a single persona and return a validated PersonaOutputSchema or None."""

    context_block = f"\n\n## Web Research Digest\n{context_digest}" if context_digest else ""
    prior_block = (
        f"\n\n## Prior Round Reasoning Summaries\n{prior_reasoning_summaries}"
        if prior_reasoning_summaries
        else ""
    )

    user_message = (
        f"## Forecasting Question\n{question_text}\n\n"
        f"## Resolution Criteria\n{resolution_criteria}"
        f"{context_block}"
        f"{prior_block}\n\n"
        "Now provide your forecast as a JSON object."
    )

    messages: list[dict] = [{"role": "user", "content": user_message}]

    for attempt in range(2):
        try:
            response = await client.messages.create(
                model=settings.llm_model,
                max_tokens=2048,
                system=persona.system_prompt,
                messages=messages,
            )
            raw_text = ""
            tokens_used = response.usage.output_tokens if response.usage else None

            for block in response.content:
                if hasattr(block, "text"):
                    raw_text = block.text.strip()
                    break

            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            data = json.loads(raw_text)
            # Ensure persona_id matches
            data["persona_id"] = persona.id
            output = PersonaOutputSchema.model_validate(data)
            logger.info(
                "persona_output_valid",
                run_id=run_id,
                persona_id=persona.id,
                point_estimate=output.point_estimate,
            )
            return output

        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "persona_output_invalid",
                run_id=run_id,
                persona_id=persona.id,
                attempt=attempt,
                error=str(exc),
            )
            if attempt == 0:
                # Retry with correction prompt
                messages.append({"role": "assistant", "content": raw_text if "raw_text" in dir() else ""})
                messages.append({"role": "user", "content": _CORRECTION_PROMPT})
            else:
                return None

        except Exception as exc:
            logger.warning(
                "persona_llm_error",
                run_id=run_id,
                persona_id=persona.id,
                error=str(exc),
            )
            return None

    return None


async def run_personas(
    question_text: str,
    resolution_criteria: str,
    context_digest: str,
    prior_reasoning_summaries: str | None,
    run_id: str,
) -> list[PersonaOutputSchema]:
    """Fan out to all personas in parallel and return validated outputs."""
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    personas = get_persona_configs()

    tasks = [
        _call_persona(
            client=client,
            persona=persona,
            question_text=question_text,
            resolution_criteria=resolution_criteria,
            context_digest=context_digest,
            prior_reasoning_summaries=prior_reasoning_summaries,
            run_id=run_id,
        )
        for persona in personas
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_outputs: list[PersonaOutputSchema] = []
    for result in results:
        if isinstance(result, PersonaOutputSchema):
            valid_outputs.append(result)
        elif isinstance(result, Exception):
            logger.warning("persona_task_exception", run_id=run_id, error=str(result))

    logger.info(
        "personas_complete",
        run_id=run_id,
        total=len(personas),
        valid=len(valid_outputs),
    )
    return valid_outputs
