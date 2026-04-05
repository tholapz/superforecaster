from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import verify_api_key
from app.personas.definitions import PERSONA_CONFIGS, get_persona_configs, update_persona_weight

router = APIRouter(prefix="/v1/admin", tags=["admin"])


class PersonaInfo(BaseModel):
    persona_id: str
    name: str
    weight: float
    description: str


class WeightUpdate(BaseModel):
    weight: float


@router.get("/personas", response_model=list[PersonaInfo], dependencies=[Depends(verify_api_key)])
async def list_personas() -> list[PersonaInfo]:
    configs = get_persona_configs()
    # Build description from persona config name/id
    descriptions = {
        "geopolitical_analyst": "Actor incentives, historical precedent",
        "base_rate_statistician": "Reference class frequency, outside view",
        "scenario_planner": "Structural drivers, contingencies",
        "bayesian_updater": "Prior probability + evidence strength",
        "devils_advocate": "Best case for opposite outcome",
        "domain_specialist": "Technical domain facts",
    }
    return [
        PersonaInfo(
            persona_id=p.id,
            name=p.name,
            weight=p.weight,
            description=descriptions.get(p.id, ""),
        )
        for p in configs
    ]


@router.post(
    "/personas/{persona_id}/weight",
    dependencies=[Depends(verify_api_key)],
)
async def set_persona_weight(persona_id: str, body: WeightUpdate) -> dict:
    try:
        update_persona_weight(persona_id, body.weight)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")
    return {"persona_id": persona_id, "weight": body.weight}
