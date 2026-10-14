"""
app/schemas/entrenamiento.py
────────────────────────────
Contratos Pydantic para los endpoints de reentrenamiento administrativo.

POST /api/v1/admin/entrenamiento/venta
  → Lanza un job de entrenamiento para venta en background

POST /api/v1/admin/entrenamiento/alquiler
  → Lanza un job de entrenamiento para alquiler en background

GET /api/v1/admin/entrenamiento/estado/{job_id}
  → Retorna el estado actual del job de entrenamiento

GET /api/v1/admin/entrenamiento/jobs
  → Lista todos los jobs ejecutados
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class EstadoJob(str, Enum):
    PENDIENTE   = "pendiente"
    EN_PROGRESO = "en_progreso"
    COMPLETADO  = "completado"
    FALLIDO     = "fallido"


class TipoOperacion(str, Enum):
    VENTA    = "venta"
    ALQUILER = "alquiler"


# ─── REQUESTS ─────────────────────────────────────────────────────────────────

class EntrenamientoBaseRequest(BaseModel):
    """Parámetros base para el job de reentrenamiento."""
    nombre_modelo: str | None = Field(
        default=None,
        description="Nombre base del artefacto a generar (sin extensión).",
    )
    max_depth: int | None = Field(
        default=None,
        ge=3, le=15,
        description="Profundidad máxima de los árboles. Si se omite, usa el valor óptimo validado (6 venta / 6 alquiler).",
        examples=[6],
    )
    learning_rate: float | None = Field(
        default=None,
        ge=0.001, le=1.0,
        description="Tasa de aprendizaje (shrinkage). Si se omite, usa el valor óptimo validado (0.040 venta / 0.035 alquiler).",
        examples=[0.040],
    )
    n_estimators: int | None = Field(
        default=None,
        ge=50, le=2000,
        description="Cantidad de árboles. Si se omite, usa el valor óptimo validado (650 venta / 600 alquiler).",
        examples=[650],
    )
    min_child_weight: int | None = Field(
        default=None,
        ge=1, le=50,
        description="Peso mínimo requerido en nodo hoja (Palanca 4: 8 venta / 4 alquiler).",
        examples=[8],
    )
    hiperparametros: dict[str, Any] | None = Field(
        default=None,
        description="Diccionario libre de hiperparámetros avanzados adicionales para XGBoost.",
        examples=[None],
    )
    shap_top_n: int = Field(
        default=10,
        ge=5, le=33,
        description="Número de contribuciones SHAP a incluir en la respuesta de inferencia.",
    )
    usar_target_m2: bool = Field(
        default=True,
        description="Si es True (recomendado CAMA), entrena sobre ln(Precio/m²) homogenizando la escala física.",
    )
    factor_calibracion: float | None = Field(
        default=None,
        ge=0.5, le=2.0,
        description="Factor de calibración post-hoc de ratio IAAO (ej: 1.0266 venta / 1.015 alquiler). Si es None, se calcula automáticamente.",
    )
    guardar_como_activo: bool = Field(
        default=False,
        description="Si es False (recomendado por gobernanza), el modelo se registra como Candidato en MLflow para validación humana. Si es True, pasa de inmediato a producción.",
    )


class EntrenamientoVentaRequest(EntrenamientoBaseRequest):
    """Parámetros para reentrenar el modelo de venta."""
    nombre_modelo: str = Field(
        default="xgboost_venta_v2",
        description="Nombre base del artefacto a generar (ej: 'xgboost_venta_v3').",
        examples=["xgboost_venta_v2"],
    )


class EntrenamientoAlquilerRequest(EntrenamientoBaseRequest):
    """Parámetros para reentrenar el modelo de alquiler."""
    nombre_modelo: str = Field(
        default="xgboost_alquiler_v1",
        description="Nombre base del artefacto a generar (ej: 'xgboost_alquiler_v2').",
        examples=["xgboost_alquiler_v1"],
    )


# ─── RESPONSE — Job iniciado ──────────────────────────────────────────────────

class EntrenamientoIniciadoResponse(BaseModel):
    """Respuesta inmediata al iniciar un job de reentrenamiento (HTTP 202)."""
    job_id: str = Field(
        description="Identificador único del job. Usar para consultar estado."
    )
    tipo_operacion: str = Field(
        default="venta",
        description="Tipo de modelo que se está entrenando ('venta' o 'alquiler')."
    )
    estado: EstadoJob = Field(
        default=EstadoJob.PENDIENTE,
        description="Estado inicial del job.",
    )
    nombre_modelo: str = Field(
        description="Nombre base del artefacto a generar."
    )
    iniciado_en: datetime = Field(
        description="Timestamp UTC de inicio del job."
    )
    mensaje: str = Field(
        description="Descripción del job iniciado."
    )
    consultar_estado_en: str = Field(
        description="URL relativa para consultar el estado del job."
    )


# ─── RESPONSE — Estado del Job ────────────────────────────────────────────────

class MetricasIAAO(BaseModel):
    """Métricas oficiales de valuación masiva IAAO (Standard on Ratio Studies 2013/2020)."""
    median_ratio: float = Field(description="Mediana del ratio de tasación (tilde{R}). Objetivo: 1.00 (0.90 - 1.10).")
    mean_ratio: float = Field(description="Media aritmética simple de los ratios (bar{R}).")
    weighted_mean_ratio: float = Field(description="Media ponderada por volumen monetario (bar{R}_w).")
    cod_pct: float = Field(description="Coeficiente de Dispersión en porcentaje (COD <= 15.0% para áreas heterogéneas).")
    cov_pct: float = Field(description="Coeficiente de Variación paramétrico en porcentaje (COV).")
    prd: float = Field(description="Diferencial Relacionado con el Precio (PRD). Rango admisible: 0.98 - 1.03.")
    prb: float = Field(description="Sesgo Relacionado con el Precio (PRB). Rango admisible: -0.05 a +0.05.")
    cumple_iaao_cod: bool = Field(description="True si COD <= 15.0%.")
    cumple_iaao_nivel: bool = Field(description="True si 0.90 <= Median Ratio <= 1.10.")
    cumple_iaao_prd: bool = Field(description="True si 0.98 <= PRD <= 1.03.")


class MetricasEntrenamiento(BaseModel):
    """Métricas calculadas sobre el test set tras el reentrenamiento."""
    model_config = {"extra": "ignore"}

    mae: float | None = Field(default=None, description="Error Absoluto Medio (Soles Constantes).")
    rmse: float | None = Field(default=None, description="Raíz del Error Cuadrático Medio.")
    mape_pct: float | None = Field(default=None, description="MAPE en el test set (porcentaje).")
    r2: float | None = Field(default=None, description="Coeficiente de Determinación R².")
    cod_pct: float | None = Field(default=None, description="Coeficiente de Dispersión IAAO (%).")
    cov_pct: float | None = Field(default=None, description="Coeficiente de Variación paramétrico (%).")
    prd: float | None = Field(default=None, description="Price-Related Differential IAAO.")
    prb: float | None = Field(default=None, description="Price-Related Bias IAAO.")
    median_ratio_calibrado: float | None = Field(default=None, description="Mediana del ratio tras calibración.")
    n_train: int | None = Field(default=None, description="Registros usados en entrenamiento.")
    n_test: int | None = Field(default=None, description="Registros usados en evaluación test.")
    supera_benchmark: bool | None = Field(
        default=None,
        description="True si el MAPE es menor al benchmark (15.0% para venta, 17.89% para alquiler).",
    )
    iaao: MetricasIAAO | None = Field(default=None, description="Métricas oficiales de valuación masiva IAAO.")


class ArtifactosGenerados(BaseModel):
    """Rutas relativas a los artefactos generados por el pipeline."""
    model_config = {"protected_namespaces": ()}

    modelo_pkl: str = Field(description="Ruta al modelo XGBoost serializado.")
    params_json: str = Field(description="Ruta al archivo con hiperparámetros usados.")
    metricas_json: str = Field(description="Ruta a las métricas oficiales calculadas.")
    model_config_json: str = Field(description="Ruta a la config del servicio actualizada.")
    modelo_recargado_en_memoria: bool = Field(
        description="True si el modelo fue cargado automáticamente en la app (hot-reload)."
    )
    mlflow_model_uri: str | None = Field(
        default=None,
        description="URI del modelo registrado en MLflow (ej: 'models:/xgboost_venta/2').",
    )


class EstadoEntrenamientoResponse(BaseModel):
    """Respuesta completa del estado de un job de reentrenamiento."""
    job_id: str
    tipo_operacion: str = "venta"
    estado: EstadoJob
    nombre_modelo: str
    iniciado_en: datetime
    completado_en: datetime | None = None
    duracion_segundos: float | None = None
    progreso: str = Field(default="", description="Paso actual del pipeline.")
    metricas: MetricasEntrenamiento | None = None
    artefactos: ArtifactosGenerados | None = None
    mlflow_run_id: str | None = Field(default=None, description="ID de corrida en MLflow Tracking.")
    mlflow_model_version: str | int | None = Field(default=None, description="Versión asignada en el Model Registry.")
    estado_gobernanza: str = Field(default="candidate", description="Estado del modelo ('candidate' o 'production').")
    error: str | None = None
