from dataclasses import dataclass

import numpy as np
import structlog

from app.config import settings
from app.personas.definitions import get_persona_configs
from app.personas.schemas import PersonaOutputSchema

logger = structlog.get_logger(__name__)


@dataclass
class AggregationResult:
    today_forecast: int
    raw_mean: float
    extremized: float
    n_valid_personas: int
    spread: float
    persona_estimates: list[float]
    change_1w: int | None = None
    change_30d: int | None = None


def _logit(p: float) -> float:
    p = float(np.clip(p, 0.001, 0.999))
    return float(np.log(p / (1 - p)))


def _sigmoid(l: float) -> float:
    return float(1 / (1 + np.exp(-l)))


def _extremize(p: float, beta: float) -> float:
    pb = p**beta
    return float(pb / (pb + (1 - p) ** beta))


def compute(
    outputs: list[PersonaOutputSchema],
    change_1w: int | None = None,
    change_30d: int | None = None,
) -> AggregationResult:
    """Trim → weighted log-odds mean → extremize."""
    if not outputs:
        raise ValueError("No persona outputs to aggregate")

    # Build weights map from current persona configs
    weight_map = {p.id: p.weight for p in get_persona_configs()}

    estimates = np.array([o.point_estimate for o in outputs], dtype=float)
    weights = np.array([weight_map.get(o.persona_id, 1.0) for o in outputs], dtype=float)

    # Step 1 — Outlier trim
    alpha = settings.trim_alpha
    n = len(estimates)
    n_trim = max(0, int(np.floor(alpha * n)))

    if n_trim > 0:
        sorted_idx = np.argsort(estimates)
        keep_idx = sorted_idx[n_trim : n - n_trim] if n_trim < n // 2 else sorted_idx
        estimates_trimmed = estimates[keep_idx]
        weights_trimmed = weights[keep_idx]
    else:
        estimates_trimmed = estimates
        weights_trimmed = weights

    logger.info(
        "aggregation_trim",
        original_n=n,
        trimmed_n=len(estimates_trimmed),
        alpha=alpha,
    )

    # Step 2 — Weighted mean on log-odds scale
    logits = np.array([_logit(p) for p in estimates_trimmed])
    weighted_logit = float(np.average(logits, weights=weights_trimmed))
    raw_mean = _sigmoid(weighted_logit)

    # Step 3 — Extremization
    beta = settings.extremize_beta
    extremized = _extremize(raw_mean, beta)

    spread = float(np.std(estimates_trimmed))
    today_forecast = int(round(extremized * 100))

    logger.info(
        "aggregation_result",
        raw_mean=raw_mean,
        extremized=extremized,
        today_forecast=today_forecast,
        spread=spread,
    )

    return AggregationResult(
        today_forecast=today_forecast,
        raw_mean=raw_mean,
        extremized=extremized,
        n_valid_personas=len(estimates_trimmed),
        spread=spread,
        persona_estimates=estimates_trimmed.tolist(),
        change_1w=change_1w,
        change_30d=change_30d,
    )
