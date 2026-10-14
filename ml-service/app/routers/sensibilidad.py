"""
app/routers/sensibilidad.py
───────────────────────────
Router público para la pantalla principal (Landing Page) y módulos de información:
  GET /api/v1/modelos/sensibilidad-factores
  GET /api/v1/modelos/sensibilidad-factores/{tipo_operacion}
"""

from __future__ import annotations

from typing import Literal
from fastapi import APIRouter, Query

from app.core.model_loader import model_state
from app.schemas.sensibilidad import SensibilidadFactoresResponse
from app.services.sensibilidad_service import obtener_sensibilidad_factores

router = APIRouter(
    prefix="/api/v1/modelos",
    tags=["Explicabilidad y Sensibilidad de Modelos"],
)


@router.get(
    "/sensibilidad-factores",
    response_model=SensibilidadFactoresResponse,
    summary="Análisis de Sensibilidad de Factores Inmobiliarios (Público)",
    description=(
        "Descompone el peso relativo de los macro-factores que determinan el precio inmobiliario en Lima Metropolitana.\n\n"
        "Diseñado para la pantalla principal / landing page institucional (no requiere token de administrador)."
    ),
)
def get_sensibilidad_factores(
    tipo_operacion: Literal["venta", "alquiler"] = Query(
        default="venta",
        description="Tipo de operación a consultar ('venta' o 'alquiler'). Por defecto es 'venta'.",
    ),
) -> SensibilidadFactoresResponse:
    return obtener_sensibilidad_factores(model_state, tipo_operacion=tipo_operacion)


@router.get(
    "/sensibilidad-factores/{tipo_operacion}",
    response_model=SensibilidadFactoresResponse,
    summary="Análisis de Sensibilidad de Factores por Operación (Público)",
    description="Variante RESTful con parámetro de ruta ('venta' o 'alquiler').",
)
def get_sensibilidad_factores_path(
    tipo_operacion: Literal["venta", "alquiler"],
) -> SensibilidadFactoresResponse:
    return obtener_sensibilidad_factores(model_state, tipo_operacion=tipo_operacion)
