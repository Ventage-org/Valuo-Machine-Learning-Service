"""
app/routers/mercado.py
──────────────────────
Router público para estadísticas y tendencias de mercado por distrito:
  GET /api/v1/mercado/tendencias-distritales
  GET /api/v1/distritos/tendencias
"""

from __future__ import annotations

from typing import Literal
from fastapi import APIRouter, Query

from app.core.model_loader import model_state
from app.schemas.mercado import TendenciasMercadoResponse
from app.services.mercado_service import obtener_tendencias_distritales

router = APIRouter(
    tags=["Mercado y Tendencias Distritales"],
)


@router.get(
    "/api/v1/mercado/tendencias-distritales",
    response_model=TendenciasMercadoResponse,
    summary="Tendencias de Valor m² y Variación Trimestral por Distrito (Público)",
    description=(
        "Devuelve el precio mediano por m² (en USD y PEN), el ticket promedio y la variación "
        "porcentual trimestral para cada uno de los distritos de Lima Metropolitana.\n\n"
        "Calculado a partir de la mediana real de observaciones históricas ajustada por el factor IPC vigente."
    ),
)
def get_tendencias_distritales(
    tipo_operacion: Literal["venta", "alquiler"] = Query(
        default="venta",
        description="Tipo de operación a consultar ('venta' o 'alquiler').",
    ),
    tipo_cambio: float = Query(
        default=3.75,
        ge=1.0,
        le=10.0,
        description="Tipo de cambio PEN/USD para convertir las métricas monetarias.",
    ),
) -> TendenciasMercadoResponse:
    return obtener_tendencias_distritales(
        state=model_state,
        tipo_operacion=tipo_operacion,
        tipo_cambio=tipo_cambio,
    )


@router.get(
    "/api/v1/distritos/tendencias",
    response_model=TendenciasMercadoResponse,
    summary="Alias de tendencias distritales para el módulo de distritos",
    include_in_schema=True,
)
def get_distritos_tendencias(
    tipo_operacion: Literal["venta", "alquiler"] = Query(
        default="venta",
        description="Tipo de operación a consultar ('venta' o 'alquiler').",
    ),
    tipo_cambio: float = Query(
        default=3.75,
        ge=1.0,
        le=10.0,
        description="Tipo de cambio PEN/USD para convertir las métricas monetarias.",
    ),
) -> TendenciasMercadoResponse:
    return obtener_tendencias_distritales(
        state=model_state,
        tipo_operacion=tipo_operacion,
        tipo_cambio=tipo_cambio,
    )

