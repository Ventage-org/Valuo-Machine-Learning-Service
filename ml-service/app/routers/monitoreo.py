"""
app/routers/monitoreo.py
────────────────────────
Router administrativo para las vistas de monitoreo y observabilidad del modelo:
  - Recursos de infraestructura (CPU y Memoria RAM)
  - Data Drift (Population Stability Index - PSI)
  - Fluctuación temporal del ciclo inmobiliario (MAPE/MAE por periodo)
  - Latencia P95 y validación de artefactos serializados
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.core.model_loader import model_state
from app.schemas.monitoreo import (
    DriftAnalisisResponse,
    FluctuacionTemporalResponse,
    LatenciaP95Response,
    RecursosSistemaResponse,
)
from app.services.monitoreo_service import (
    calcular_data_drift_psi,
    medir_latencia_p95,
    obtener_fluctuacion_temporal,
    obtener_recursos_sistema,
)

router = APIRouter(
    prefix="/api/v1/admin/monitoreo",
    tags=["Administración — Monitoreo y Observabilidad"],
)


def _verificar_admin_token(x_admin_token: str | None = Header(None)) -> None:
    """Valida el token de administración si está configurado en .env."""
    expected = os.getenv("ADMIN_TOKEN", "").strip()
    if not expected:
        return
    if x_admin_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "token_invalido",
                "message": "El token de administrador es inválido.",
            },
        )


@router.get(
    "/recursos",
    response_model=RecursosSistemaResponse,
    summary="Métricas de Infraestructura (CPU y RAM)",
    description="Retorna el uso actual de CPU y memoria RAM del host/contenedor FastAPI en tiempo real.",
)
def get_recursos_sistema(
    _: None = Depends(_verificar_admin_token),
) -> RecursosSistemaResponse:
    return obtener_recursos_sistema()


@router.get(
    "/drift/{tipo_operacion}",
    response_model=DriftAnalisisResponse,
    summary="Detección de Data Drift (Population Stability Index - PSI)",
    description=(
        "Calcula el Population Stability Index (PSI) de las variables clave del mercado "
        "comparando la distribución histórica de entrenamiento (2016-2023) contra el ciclo reciente (2024-2025)."
    ),
)
def get_data_drift(
    tipo_operacion: Literal["venta", "alquiler"],
    _: None = Depends(_verificar_admin_token),
) -> DriftAnalisisResponse:
    return calcular_data_drift_psi(tipo_operacion=tipo_operacion)


@router.get(
    "/fluctuacion-temporal/{tipo_operacion}",
    response_model=FluctuacionTemporalResponse,
    summary="Fluctuación Temporal del Ciclo de Mercado (MAPE/MAE por periodo)",
    description=(
        "Genera los puntos de serie temporal (por trimestre/mes) evaluando el MAPE y MAE "
        "frente al umbral regulatorio del 15.0% para la gráfica de la pantalla de monitoreo."
    ),
)
def get_fluctuacion_temporal(
    tipo_operacion: Literal["venta", "alquiler"],
    _: None = Depends(_verificar_admin_token),
) -> FluctuacionTemporalResponse:
    return obtener_fluctuacion_temporal(tipo_operacion=tipo_operacion, state=model_state)


@router.get(
    "/latencia-p95/{tipo_operacion}",
    response_model=LatenciaP95Response,
    summary="Latencia P95 y Verificación de Artefactos",
    description=(
        "Ejecuta un micro-benchmark para medir la latencia mediana (P50), percentil 95 (P95) "
        "y verificar que todos los artefactos requeridos (model.pkl, params, metricas, config) estén disponibles."
    ),
)
def get_latencia_p95(
    tipo_operacion: Literal["venta", "alquiler"],
    _: None = Depends(_verificar_admin_token),
) -> LatenciaP95Response:
    return medir_latencia_p95(tipo_operacion=tipo_operacion, state=model_state)
