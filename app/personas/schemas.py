from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PersonaOutputSchema(BaseModel):
    persona_id: str
    reasoning_chain: list[str] = Field(..., min_length=3)
    point_estimate: float
    confidence_interval: tuple[float, float]
    confidence_level: Literal["low", "medium", "high"]
    key_cruxes: list[str] = Field(..., min_length=2, max_length=4)
    update_direction: Literal["up", "down", "unchanged"]

    @field_validator("point_estimate")
    @classmethod
    def validate_point_estimate(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"point_estimate must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("confidence_interval")
    @classmethod
    def validate_confidence_interval(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"confidence_interval must satisfy 0 <= lo <= hi <= 1, got {v}")
        return v
