"""
app/routers/health.py
─────────────────────
Endpoints de estado y metadata del servicio.

GET /health                → estado básico del servicio (para load balancer / k8s probe)
GET /api/v1/modelo/info    → metadata completa del modelo activo
GET /api/v1/distritos      → lista de distritos disponibles para predicción
"""

from __future__ import annotations

from typing import Literal
from fastapi import APIRouter, Query

from app.core.model_loader import model_state
from app.schemas.modelo_info import (
    DistritosResponse,
    IpcInfo,
    MetricasModelo,
    ModeloInfoResponse,
)

router = APIRouter(tags=["Estado y Metadata"])


@router.get(
    "/health",
    summary="Health check",
    description="Verifica que el servicio y los modelos (venta y alquiler) están operativos.",
)
def health_check() -> dict:
    modelo_venta_ok = model_state.modelo is not None
    modelo_alquiler_ok = model_state.modelo_alquiler is not None
    return {
        "status": "ok" if (modelo_venta_ok and modelo_alquiler_ok) else "degraded",
        "modelo_venta": model_state.config.get("modelo_version", "desconocido") if modelo_venta_ok else None,
        "modelo_alquiler": model_state.config_alquiler.get("modelo_version", "desconocido") if modelo_alquiler_ok else None,
        "ipc_periodo": (
            model_state.config.get("ipc_actual", {}).get("periodo")
            if modelo_venta_ok else None
        ),
    }


@router.get(
    "/api/v1/modelo/info",
    response_model=ModeloInfoResponse,
    summary="Información del modelo activo (Venta o Alquiler)",
    description=(
        "Devuelve la metadata completa del modelo cargado según el tipo de operación: "
        "versión, métricas, IPC vigente, fuente de datos activa y lista de distritos disponibles."
    ),
)
def modelo_info(
    tipo_operacion: Literal["venta", "alquiler"] = Query(
        default="venta",
        description="Tipo de modelo a consultar ('venta' o 'alquiler')."
    ),
) -> ModeloInfoResponse:
    if tipo_operacion == "alquiler":
        cfg = model_state.config_alquiler
        encoding_map = model_state.encoding_map_alquiler
        default_version = "v1"
        default_file = "xgboost_alquiler_v1.pkl"
    else:
        cfg = model_state.config
        encoding_map = model_state.encoding_map
        default_version = "v2"
        default_file = "xgboost_venta_v2.pkl"

    ipc_cfg = cfg.get("ipc_actual", {})

    distritos = (
        model_state.data_provider.listar_distritos()  # type: ignore[union-attr]
        if model_state.data_provider
        else sorted(encoding_map.keys())
    )

    return ModeloInfoResponse(
        version=cfg.get("modelo_version", default_version),
        archivo=cfg.get("modelo_archivo", default_file),
        metricas=MetricasModelo(
            mape_pct=cfg.get("mape_test", 0.0),
            r2=cfg.get("r2_test", 0.0),
            mae_soles=cfg.get("mae_test", 0.0),
            rmse_soles=cfg.get("rmse_test", 0.0),
            train_periodo=cfg.get("train_periodo", "2016-2023"),
            test_periodo=cfg.get("test_periodo", "2024-2025"),
        ),
        ipc_actual=IpcInfo(
            valor=ipc_cfg.get("valor", 0.0),
            periodo=ipc_cfg.get("periodo", ""),
            fuente=ipc_cfg.get("fuente", ""),
            nota=ipc_cfg.get("nota", ""),
        ),
        n_features=cfg.get("n_features", 33),
        n_distritos=cfg.get("n_distritos", 22),
        data_provider="postgresql",
        distritos_disponibles=distritos,
    )


@router.get(
    "/api/v1/distritos",
    response_model=DistritosResponse,
    summary="Distritos disponibles",
    description=(
        "Lista de los distritos de Lima Metropolitana disponibles en el modelo. "
        "El campo 'distrito' del request de predicción debe usar exactamente "
        "uno de estos nombres (case-sensitive)."
    ),
)
def listar_distritos() -> DistritosResponse:
    distritos = (
        model_state.data_provider.listar_distritos()  # type: ignore[union-attr]
        if model_state.data_provider
        else sorted(model_state.encoding_map.keys())
    )
    return DistritosResponse(distritos=distritos, total=len(distritos))

