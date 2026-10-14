"""
app/routers/entrenamiento.py
────────────────────────────
Router administrativo para reentrenamiento de modelos (Venta y Alquiler).

Endpoints:
  POST /api/v1/admin/entrenamiento/venta     → Lanza reentrenamiento del modelo de venta (HTTP 202)
  POST /api/v1/admin/entrenamiento/alquiler  → Lanza reentrenamiento del modelo de alquiler (HTTP 202)
  GET  /api/v1/admin/entrenamiento/estado/{id} → Consulta progreso y métricas de un job
  GET  /api/v1/admin/entrenamiento/estado     → Consulta estado del último job
  GET  /api/v1/admin/entrenamiento/jobs       → Lista historial de jobs en la sesión

Seguridad:
  Autenticación mediante header X-Admin-Token (configurable en .env).
  Si no se define ADMIN_TOKEN en .env, permite el acceso en modo desarrollo.
"""

from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.core.model_loader import model_state
from app.schemas.entrenamiento import (
    ArtifactosGenerados,
    EntrenamientoAlquilerRequest,
    EntrenamientoIniciadoResponse,
    EntrenamientoVentaRequest,
    EstadoEntrenamientoResponse,
    EstadoJob,
    MetricasEntrenamiento,
)
from app.services.entrenamiento_service import (
    get_job_status,
    get_latest_job,
    iniciar_entrenamiento_alquiler,
    iniciar_entrenamiento_venta,
    list_all_jobs,
)

router = APIRouter(
    prefix="/api/v1/admin/entrenamiento",
    tags=["Administración — Entrenamiento"],
)


# ─── Dependencia de autenticación admin ───────────────────────────────────────

def _verificar_admin_token(x_admin_token: str | None = Header(None)) -> None:
    """
    Valida el header X-Admin-Token contra la variable de entorno ADMIN_TOKEN si está definida.
    Si ADMIN_TOKEN no está configurado, permite el acceso (entorno de desarrollo/pruebas).
    """
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


# ─── Endpoints de Inicio de Entrenamiento ─────────────────────────────────────

@router.post(
    "/venta",
    response_model=EntrenamientoIniciadoResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Iniciar reentrenamiento del modelo de venta",
    description=(
        "Lanza el pipeline de reentrenamiento del modelo XGBoost de venta "
        "utilizando los datos actuales de `dataset_inmuebles_venta` en PostgreSQL.\n\n"
        "Retorna inmediatamente un `job_id` para consultar el progreso en background."
    ),
    responses={
        202: {"description": "Job de venta iniciado exitosamente."},
        401: {"description": "Token de administrador inválido."},
        409: {"description": "Ya hay un job de entrenamiento en curso."},
    },
)
def iniciar_entrenamiento_modelo_venta(
    request: EntrenamientoVentaRequest,
    _: None = Depends(_verificar_admin_token),
) -> EntrenamientoIniciadoResponse:
    nombre = request.nombre_modelo or "xgboost_venta_v2"
    job_id = iniciar_entrenamiento_venta(
        nombre_modelo=nombre,
        hiperparametros=request.hiperparametros,
        shap_top_n=request.shap_top_n,
        state=model_state,
        guardar_como_activo=request.guardar_como_activo,
        max_depth=request.max_depth,
        learning_rate=request.learning_rate,
        n_estimators=request.n_estimators,
        min_child_weight=request.min_child_weight,
        usar_target_m2=request.usar_target_m2,
        factor_calibracion=request.factor_calibracion,
    )

    return EntrenamientoIniciadoResponse(
        job_id=job_id,
        tipo_operacion="venta",
        estado=EstadoJob.EN_PROGRESO,
        nombre_modelo=nombre,
        iniciado_en=datetime.utcnow(),
        mensaje=(
            f"Job de reentrenamiento de venta '{nombre}' iniciado. "
            "El pipeline corre en background y registrará el modelo en MLflow."
        ),
        consultar_estado_en=f"/api/v1/admin/entrenamiento/estado/{job_id}",
    )


@router.post(
    "/alquiler",
    response_model=EntrenamientoIniciadoResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Iniciar reentrenamiento del modelo de alquiler",
    description=(
        "Lanza el pipeline de reentrenamiento del modelo XGBoost de alquiler "
        "utilizando los datos actuales de `dataset_inmuebles_alquiler` en PostgreSQL.\n\n"
        "Aplica ponderación temporal exponencial (E1 con decay=0.85) dando prioridad "
        "a los años más recientes (2021-2023)."
    ),
    responses={
        202: {"description": "Job de alquiler iniciado exitosamente."},
        401: {"description": "Token de administrador inválido."},
        409: {"description": "Ya hay un job de entrenamiento en curso."},
    },
)
def iniciar_entrenamiento_modelo_alquiler(
    request: EntrenamientoAlquilerRequest,
    _: None = Depends(_verificar_admin_token),
) -> EntrenamientoIniciadoResponse:
    nombre = request.nombre_modelo or "xgboost_alquiler_v1"
    job_id = iniciar_entrenamiento_alquiler(
        nombre_modelo=nombre,
        hiperparametros=request.hiperparametros,
        shap_top_n=request.shap_top_n,
        state=model_state,
        guardar_como_activo=request.guardar_como_activo,
        max_depth=request.max_depth,
        learning_rate=request.learning_rate,
        n_estimators=request.n_estimators,
        min_child_weight=request.min_child_weight,
        usar_target_m2=request.usar_target_m2,
        factor_calibracion=request.factor_calibracion,
    )

    return EntrenamientoIniciadoResponse(
        job_id=job_id,
        tipo_operacion="alquiler",
        estado=EstadoJob.EN_PROGRESO,
        nombre_modelo=nombre,
        iniciado_en=datetime.utcnow(),
        mensaje=(
            f"Job de reentrenamiento de alquiler '{nombre}' iniciado. "
            "El pipeline corre en background y puede tardar varios minutos."
        ),
        consultar_estado_en=f"/api/v1/admin/entrenamiento/estado/{job_id}",
    )


# ─── Endpoints de Consulta de Estado ──────────────────────────────────────────

@router.get(
    "/jobs",
    summary="Listar historial de jobs de entrenamiento",
    description="Retorna la lista de todos los jobs de entrenamiento ejecutados en la sesión actual.",
)
def listar_jobs(_: None = Depends(_verificar_admin_token)) -> list[dict]:
    return list_all_jobs()


@router.get(
    "/estado",
    summary="Consultar estado del último job o disponibilidad",
    description="Retorna el estado del último job de entrenamiento registrado, o un mensaje indicando que no hay jobs activos.",
)
def ultimo_estado_entrenamiento(_: None = Depends(_verificar_admin_token)) -> dict:
    job_data = get_latest_job()
    if not job_data:
        return {
            "mensaje": "No se ha ejecutado ningún job de reentrenamiento en esta sesión.",
            "disponible_para_entrenar": True,
        }
    return job_data


@router.get(
    "/estado/{job_id}",
    response_model=EstadoEntrenamientoResponse,
    summary="Consultar estado del job de entrenamiento",
    description=(
        "Retorna el estado actual, progreso y métricas de un job de reentrenamiento.\n\n"
        "**Estados posibles:**\n"
        "- `en_progreso`: El pipeline está corriendo (revisa `progreso` para el paso actual).\n"
        "- `completado`: El modelo fue reentrenado y recargado en memoria.\n"
        "- `fallido`: Ocurrió un error. Revisa `error` para el detalle."
    ),
    responses={
        200: {"description": "Estado del job retornado correctamente."},
        401: {"description": "Token de administrador inválido."},
        404: {"description": "job_id no encontrado."},
    },
)
def estado_entrenamiento(
    job_id: str,
    _: None = Depends(_verificar_admin_token),
) -> EstadoEntrenamientoResponse:
    """
    Consulta el estado de un job de reentrenamiento por su job_id.
    """
    job_data = get_job_status(job_id)

    metricas = job_data.get("metricas")
    if isinstance(metricas, dict):
        metricas = MetricasEntrenamiento(**metricas)

    artefactos = job_data.get("artefactos")
    if isinstance(artefactos, dict):
        artefactos = ArtifactosGenerados(**artefactos)

    return EstadoEntrenamientoResponse(
        job_id=job_data["job_id"],
        tipo_operacion=job_data.get("tipo_operacion", "venta"),
        estado=job_data["estado"],
        nombre_modelo=job_data["nombre_modelo"],
        iniciado_en=job_data["iniciado_en"],
        completado_en=job_data.get("completado_en"),
        duracion_segundos=job_data.get("duracion_segundos"),
        progreso=job_data.get("progreso", ""),
        metricas=metricas,
        artefactos=artefactos,
        error=job_data.get("error"),
    )
