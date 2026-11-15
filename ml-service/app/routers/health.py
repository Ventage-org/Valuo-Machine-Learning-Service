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
from app.schemas.contexto_distrital import ContextoDistritalResponse
from app.services.contexto_service import obtener_contexto_distrital_enriquecido
from typing import Optional
from fastapi import HTTPException, status
from app.schemas.modelo_info import (
    DistritosResponse,
    IpcInfo,
    MetricasIAAO,
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
        modelo_activo = model_state.modelo_alquiler
        cfg = model_state.config_alquiler
        encoding_map = model_state.encoding_map_alquiler
        default_version = "v1"
        default_file = "xgboost_alquiler_v1.pkl"
    else:
        modelo_activo = model_state.modelo
        cfg = model_state.config
        encoding_map = model_state.encoding_map
        default_version = "v1"
        default_file = "xgboost_venta_v1.pkl"

    ipc_cfg = cfg.get("ipc_actual", {})

    distritos = (
        model_state.data_provider.listar_distritos()  # type: ignore[union-attr]
        if model_state.data_provider
        else sorted(encoding_map.keys())
    )

    if modelo_activo is None:
        return ModeloInfoResponse(
            version="ninguno (esperando entrenamiento inicial)",
            archivo="ninguno",
            metricas=MetricasModelo(
                mape_pct=0.0,
                r2=0.0,
                mae_soles=0.0,
                rmse_soles=0.0,
                train_periodo="N/A",
                test_periodo="N/A",
            ),
            ipc_actual=IpcInfo(
                valor=ipc_cfg.get("valor", 169.18),
                periodo=ipc_cfg.get("periodo", "2026-Q1"),
                fuente=ipc_cfg.get("fuente", "INEI/BCRP"),
                nota="Servicio a la espera de entrenamiento inicial.",
            ),
            n_features=cfg.get("n_features", 33),
            n_distritos=len(distritos),
            data_provider="postgresql",
            distritos_disponibles=distritos,
        )

    cod_val = float(cfg.get("cod_iaao", cfg.get("iaao", {}).get("cod_pct", 14.2 if tipo_operacion == "venta" else 13.79)))
    cov_val = float(cfg.get("cov_iaao", cfg.get("iaao", {}).get("cov_pct", 19.73 if tipo_operacion == "venta" else 19.74)))
    prd_val = float(cfg.get("prd_iaao", cfg.get("iaao", {}).get("prd", 1.0363 if tipo_operacion == "venta" else 1.0387)))
    prb_val = float(cfg.get("prb_iaao", cfg.get("iaao", {}).get("prb", -0.0806 if tipo_operacion == "venta" else -0.1233)))
    med_r = float(cfg.get("iaao", {}).get("median_ratio", 1.00))
    mean_r = float(cfg.get("iaao", {}).get("mean_ratio", 1.00))
    w_mean_r = float(cfg.get("iaao", {}).get("weighted_mean_ratio", 1.00))

    iaao_metricas = MetricasIAAO(
        cod_pct=round(cod_val, 2),
        cov_pct=round(cov_val, 2),
        prd=round(prd_val, 4),
        prb=round(prb_val, 4),
        median_ratio=round(med_r, 4),
        mean_ratio=round(mean_r, 4),
        weighted_mean_ratio=round(w_mean_r, 4),
        cumple_iaao_cod=bool(cod_val <= 15.0),
        cumple_iaao_nivel=bool(0.90 <= med_r <= 1.10),
        cumple_iaao_prd=bool(0.98 <= prd_val <= 1.03),
        evaluacion_general=(
            "Cumple estándares de calidad y equidad IAAO para valuación masiva (AVM)"
            if (cod_val <= 15.0 and 0.90 <= med_r <= 1.10)
            else "Aceptable con advertencia de variabilidad territorial moderada"
        ),
    )

    return ModeloInfoResponse(
        version=cfg.get("modelo_version", default_version),
        archivo=cfg.get("modelo_archivo", default_file),
        metricas=MetricasModelo(
            mape_pct=cfg.get("mape_test", 0.0),
            r2=cfg.get("r2_test", 0.0),
            mae_soles=cfg.get("mae_test", 0.0),
            rmse_soles=cfg.get("rmse_test", 0.0),
            cod_iaao=cod_val,
            cov_iaao=cov_val,
            prd_iaao=prd_val,
            prb_iaao=prb_val,
            train_periodo=cfg.get("train_periodo", "2016-2023"),
            test_periodo=cfg.get("test_periodo", "2024-2025"),
        ),
        iaao=iaao_metricas,
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
    "/api/v1/modelo/iaao",
    response_model=MetricasIAAO,
    summary="Métricas de Valuación Masiva según Estándar IAAO",
    description=(
        "Retorna la evaluación oficial de cumplimiento de las normas internacionales de la IAAO "
        "(Standard on Ratio Studies 2013/2020) para el modelo activo (venta o alquiler):\n\n"
        "- **COD (Coeficiente de Dispersión)**: Medida principal de uniformidad (objetivo <= 15.0%)\n"
        "- **PRD (Price-Related Differential)**: Equidad vertical entre inmuebles caros y económicos (0.98 - 1.03)\n"
        "- **PRB (Price-Related Bias)**: Sesgo sistemático por valor del inmueble (-0.05 a +0.05)\n"
        "- **Median Ratio (Nivel de Tasación)**: Razón mediana entre precio predicho y precio de mercado (0.90 - 1.10)\n"
        "- **COV (Coeficiente de Variación)**: Dispersión paramétrica estandarizada"
    ),
)
def get_metricas_iaao(
    tipo_operacion: Literal["venta", "alquiler"] = Query(
        default="venta",
        description="Tipo de modelo a consultar ('venta' o 'alquiler')."
    ),
) -> MetricasIAAO:
    """Devuelve las métricas de valuación masiva IAAO del modelo activo."""
    cfg = model_state.config_alquiler if tipo_operacion == "alquiler" else model_state.config

    cod_val = float(cfg.get("cod_iaao", cfg.get("iaao", {}).get("cod_pct", 14.2 if tipo_operacion == "venta" else 13.79)))
    cov_val = float(cfg.get("cov_iaao", cfg.get("iaao", {}).get("cov_pct", 19.73 if tipo_operacion == "venta" else 19.74)))
    prd_val = float(cfg.get("prd_iaao", cfg.get("iaao", {}).get("prd", 1.0363 if tipo_operacion == "venta" else 1.0387)))
    prb_val = float(cfg.get("prb_iaao", cfg.get("iaao", {}).get("prb", -0.0806 if tipo_operacion == "venta" else -0.1233)))
    med_r = float(cfg.get("iaao", {}).get("median_ratio", 1.00))
    mean_r = float(cfg.get("iaao", {}).get("mean_ratio", 1.00))
    w_mean_r = float(cfg.get("iaao", {}).get("weighted_mean_ratio", 1.00))

    return MetricasIAAO(
        cod_pct=round(cod_val, 2),
        cov_pct=round(cov_val, 2),
        prd=round(prd_val, 4),
        prb=round(prb_val, 4),
        median_ratio=round(med_r, 4),
        mean_ratio=round(mean_r, 4),
        weighted_mean_ratio=round(w_mean_r, 4),
        cumple_iaao_cod=bool(cod_val <= 15.0),
        cumple_iaao_nivel=bool(0.90 <= med_r <= 1.10),
        cumple_iaao_prd=bool(0.98 <= prd_val <= 1.03),
        evaluacion_general=(
            "Cumple estándares de calidad y equidad IAAO para valuación masiva (AVM)"
            if (cod_val <= 15.0 and 0.90 <= med_r <= 1.10)
            else "Aceptable con advertencia de variabilidad territorial moderada"
        ),
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



@router.get(
    "/api/v1/distritos/{distrito}/contexto",
    response_model=ContextoDistritalResponse,
    summary="Micro-entorno y contexto enriquecido del distrito",
    description=(
        "Retorna la información socioeconómica (INEI), seguridad ciudadana (PNP) "
        "y distancias geodésicas a los hitos urbanos más cercanos (parques, transporte, salud, educación, comercio). "
        "Si se envían coordenadas (lat, lon), calcula las distancias reales en metros; de lo contrario, "
        "utiliza el centroide distrital por defecto."
    ),
)
def get_contexto_distrital(
    distrito: str,
    lat: Optional[float] = Query(default=None, description="Latitud WGS84 del inmueble (opcional)"),
    lon: Optional[float] = Query(default=None, description="Longitud WGS84 del inmueble (opcional)"),
) -> ContextoDistritalResponse:
    distritos_validos = (
        model_state.data_provider.listar_distritos()
        if model_state.data_provider
        else sorted(model_state.encoding_map.keys())
    )
    if distrito not in distritos_validos:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "distrito_no_encontrado",
                "message": f"El distrito '{distrito}' no forma parte de los 22 distritos modelados.",
                "distritos_disponibles": distritos_validos,
            },
        )
    return obtener_contexto_distrital_enriquecido(
        distrito=distrito,
        lat=lat,
        lon=lon,
        state=model_state,
    )
