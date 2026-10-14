"""
app/routers/prediccion.py
─────────────────────────
Endpoints principales de predicción:
  - POST /api/v1/prediccion/venta     → Predicción de precio de venta
  - POST /api/v1/prediccion/alquiler  → Predicción de canon de arrendamiento mensual
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.model_loader import model_state
from app.schemas.prediccion import PrediccionVentaRequest, PrediccionVentaResponse
from app.services.prediccion_service import predecir_alquiler, predecir_venta

router = APIRouter(prefix="/api/v1/prediccion", tags=["Predicción"])


@router.post(
    "/venta",
    response_model=PrediccionVentaResponse,
    summary="Predecir precio de venta",
    description=(
        "Predice el precio de venta de un inmueble en Lima Metropolitana.\n\n"
        "**Variables del inmueble requeridas**: distrito, superficie, habitaciones, "
        "baños, piso y antigüedad.\n\n"
        "Las variables contextuales del distrito (NSE, criminalidad, distancias geoespaciales "
        "y datos demográficos) son resueltas automáticamente por el servicio.\n\n"
        "**Respuesta incluye**:\n"
        "- Precio en Soles Constantes (base dic 2009) y Soles Nominales (IPC vigente)\n"
        "- Precio por m² en ambas unidades\n"
        "- Intervalo de confianza ± MAPE del modelo\n"
        "- Top 10 variables SHAP con su contribución individual al precio predicho\n"
        "- Metadata del modelo activo (versión, IPC, métricas)"
    ),
    responses={
        200: {"description": "Predicción de venta exitosa con valores SHAP."},
        422: {"description": "Datos de entrada inválidos o distrito no disponible."},
        500: {"description": "Error interno durante inferencia o cálculo SHAP."},
    },
)
def predecir_precio_venta(request: PrediccionVentaRequest) -> PrediccionVentaResponse:
    """Endpoint de predicción de precio de venta."""
    return predecir_venta(req=request, state=model_state)


@router.post(
    "/alquiler",
    response_model=PrediccionVentaResponse,
    summary="Predecir precio de alquiler mensual",
    description=(
        "Predice el canon de arrendamiento mensual de un inmueble en Lima Metropolitana.\n\n"
        "**Variables del inmueble requeridas**: distrito, superficie, habitaciones, "
        "baños, piso y antigüedad.\n\n"
        "**Respuesta incluye**:\n"
        "- Alquiler en Soles Constantes y Soles Nominales (ajustados por IPC)\n"
        "- Alquiler por m² en ambas unidades\n"
        "- Intervalo de confianza basado en el MAPE del modelo de alquiler (13.85%)\n"
        "- Top 10 variables SHAP (explicabilidad en tiempo real)\n"
        "- Metadata del modelo de alquiler activo"
    ),
    responses={
        200: {"description": "Predicción de alquiler exitosa con valores SHAP."},
        422: {"description": "Datos de entrada inválidos o distrito no disponible."},
        500: {"description": "Error interno durante inferencia o cálculo SHAP."},
    },
)
def predecir_precio_alquiler(request: PrediccionVentaRequest) -> PrediccionVentaResponse:
    """Endpoint de predicción de alquiler mensual."""
    return predecir_alquiler(req=request, state=model_state)
