"""
app/schemas/modelos.py
──────────────────────
Contratos Pydantic para los endpoints de gobernanza, benchmark y promoción de modelos.

Endpoints:
  GET  /api/v1/admin/modelos/benchmark/{tipo_operacion}
  GET  /api/v1/admin/modelos/historial/{tipo_operacion}
  POST /api/v1/admin/modelos/promover
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class MetricasModelo(BaseModel):
    """Métricas de evaluación del modelo."""
    mape_pct: float | None = Field(default=None, description="MAPE en test (%)")
    r2: float | None = Field(default=None, description="Coeficiente de determinación R²")
    mae: float | None = Field(default=None, description="Error absoluto medio (Soles Constantes)")
    rmse: float | None = Field(default=None, description="Raíz del error cuadrático medio")
    n_train: int | None = Field(default=None, description="Filas de entrenamiento")
    n_test: int | None = Field(default=None, description="Filas de prueba")


class VersionModeloInfo(BaseModel):
    """Información detallada de una versión en el Model Registry."""
    model_config = {"protected_namespaces": ()}

    nombre_modelo: str = Field(description="Nombre registrado en MLflow (ej: 'xgboost_venta')")
    version: str = Field(description="Número de versión (ej: '1', '2')")
    stage: str = Field(description="Etapa del ciclo de vida ('Production', 'Candidate', 'Archived')")
    run_id: str | None = Field(default=None, description="ID de corrida en MLflow Tracking")
    metricas: MetricasModelo = Field(default_factory=MetricasModelo)
    tags: dict[str, str] = Field(default_factory=dict)
    creado_en: datetime | None = None
    fuente_artefacto: str | None = None


class DeltaMetricas(BaseModel):
    """Diferencias relativas entre un modelo candidato y el modelo en producción."""
    delta_mape_pct: float | None = Field(
        default=None,
        description="Diferencia de MAPE (candidato - producción). Valores negativos indican mejora.",
    )
    delta_r2: float | None = Field(
        default=None,
        description="Diferencia de R² (candidato - producción). Valores positivos indican mejora.",
    )
    delta_mae: float | None = Field(default=None, description="Diferencia de MAE")
    delta_rmse: float | None = Field(default=None, description="Diferencia de RMSE")
    mejora_mape: bool = Field(default=False, description="True si el candidato reduce el error MAPE")
    mejora_r2: bool = Field(default=False, description="True si el candidato aumenta el R²")


class ComparativaCandidato(BaseModel):
    """Evaluación comparativa de un modelo candidato frente al modelo oficial en producción."""
    candidato: VersionModeloInfo
    deltas: DeltaMetricas
    recomendacion: Literal["RECOMENDADO_PARA_PRODUCCION", "NO_RECOMENDADO", "REVISION_MANUAL"] = Field(
        description="Dictamen automático basado en benchmarks metodológicos de la tesis.",
    )
    motivo: str = Field(description="Explicación en lenguaje natural del dictamen.")


class BenchmarkResponse(BaseModel):
    """Respuesta completa del benchmark comparativo entre Producción y Candidatos."""
    tipo_operacion: Literal["venta", "alquiler"]
    benchmark_referencial_mape: float = Field(
        description="Benchmark mínimo exigido por la metodología de la tesis (15.0% venta / 17.89% alquiler)",
    )
    modelo_produccion: VersionModeloInfo | None = Field(
        default=None,
        description="Modelo actualmente activo en Producción y memoria RAM.",
    )
    candidatos: list[ComparativaCandidato] = Field(
        default_factory=list,
        description="Lista de modelos candidatos ordenados por desempeño reciente.",
    )
    total_candidatos: int = 0
    generado_en: datetime = Field(default_factory=datetime.utcnow)


class HistorialModelosResponse(BaseModel):
    """Historial completo de versiones registradas en el Model Registry."""
    tipo_operacion: Literal["venta", "alquiler"]
    nombre_modelo: str
    total_versiones: int
    versiones: list[VersionModeloInfo]
    generado_en: datetime = Field(default_factory=datetime.utcnow)


class PromoverModeloRequest(BaseModel):
    """Solicitud de aprobación humana para promover un modelo a Producción."""
    tipo_operacion: Literal["venta", "alquiler"] = Field(
        description="Tipo de modelo a promover ('venta' o 'alquiler')",
    )
    version: str | int | None = Field(
        default=None,
        description="Número de versión candidata a promover (opcional). Si se omite o es null, promueve automáticamente el último candidato.",
    )
    motivo: str = Field(
        default="Aprobación manual del administrador tras verificación de benchmark",
        description="Justificación registrada en tags de auditoría de MLflow",
    )


class PromoverModeloResponse(BaseModel):
    """Respuesta tras promover exitosamente un modelo a Producción o realizar rollback."""
    exito: bool = True
    mensaje: str
    tipo_operacion: str
    version_promovida: str
    version_anterior_archivada: str | None = None
    modelo_cargado_en_ram: bool = True
    metricas: MetricasModelo | None = Field(default=None, description="Métricas del modelo ahora activo en producción")
    metricas_anteriores: MetricasModelo | None = Field(default=None, description="Métricas de la versión previamente activa")
    deltas: DeltaMetricas | None = Field(default=None, description="Diferencias relativas (ΔMAPE, ΔR², mejora_mape)")
    promovido_en: datetime = Field(default_factory=datetime.utcnow)
