"""
app/schemas/monitoreo.py
────────────────────────
Contratos Pydantic para los endpoints auxiliares del módulo de monitoreo:
  - Recursos de infraestructura (CPU y Memoria RAM)
  - Data Drift (Population Stability Index - PSI)
  - Fluctuación temporal del ciclo inmobiliario (MAPE/MAE mensual/trimestral vs cota 15%)
  - Latencia P95 y validación de artefactos
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


# ─── 1. Recursos del Sistema ──────────────────────────────────────────────────

class RecursosSistemaResponse(BaseModel):
    """Métricas en tiempo real del contenedor y host de FastAPI."""
    cpu_usage_pct: float = Field(description="Porcentaje de uso actual de CPU (0 a 100%)")
    memory_used_mb: float = Field(description="Memoria RAM en uso en Megabytes")
    memory_total_mb: float = Field(description="Memoria RAM total del sistema en Megabytes")
    memory_usage_pct: float = Field(description="Porcentaje de memoria RAM ocupada")
    uptime_seconds: float = Field(description="Tiempo de actividad del microservicio en segundos")
    active_threads: int = Field(description="Cantidad de hilos concurrentes activos")
    timestamp: datetime = Field(description="Timestamp UTC de la lectura")


# ─── 2. Detección de Data Drift (PSI) ─────────────────────────────────────────

class FeatureDriftInfo(BaseModel):
    """Evaluación de Data Drift por variable mediante Population Stability Index (PSI)."""
    feature_name: str = Field(description="Nombre de la variable evaluada")
    psi_score: float = Field(description="Valor del Population Stability Index")
    status: Literal["STABLE", "MODERATE_DRIFT", "HIGH_DRIFT"] = Field(
        description="Estado del drift según umbrales IAAO/MLOps (<0.10 Estable, 0.10-0.25 Moderado, >0.25 Crítico)"
    )
    description: str = Field(description="Interpretación técnica del comportamiento de la variable")


class DriftAnalisisResponse(BaseModel):
    """Reporte integral de Population Stability Index (PSI) para el modelo."""
    operation_type: Literal["venta", "alquiler"]
    overall_psi: float = Field(description="Índice PSI ponderado global")
    drift_status: Literal["STABLE", "MODERATE_DRIFT", "HIGH_DRIFT"] = Field(
        description="Diagnóstico global de estabilidad poblacional"
    )
    baseline_period: str = Field(description="Periodo de referencia (entrenamiento)", default="2016-2023")
    target_period: str = Field(description="Periodo evaluado (reciente / test)", default="2024-2025")
    features: list[FeatureDriftInfo] = Field(description="Desglose por variables clave del modelo")
    message: str = Field(description="Resumen en lenguaje natural")
    evaluated_at: datetime = Field(description="Timestamp UTC de la evaluación")


# ─── 3. Fluctuación Temporal del Ciclo de Mercado ─────────────────────────────

class PuntoFluctuacionTemporal(BaseModel):
    """Punto en la serie de tiempo para la gráfica de error continuo."""
    period: str = Field(description="Etiqueta temporal (ej: 2024-Q1, 2024-Q2)")
    mape_pct: float = Field(description="MAPE alcanzado en dicho periodo")
    mae_soles: float = Field(description="MAE en Soles constantes para dicho periodo")
    upper_threshold_mape: float = Field(description="Cota máxima admisible del estándar", default=15.0)
    sample_count: int = Field(description="Número de transacciones en la ventana temporal")
    exceeds_threshold: bool = Field(description="True si el error superó el límite del 15%")


class FluctuacionTemporalResponse(BaseModel):
    """Serie de tiempo completa para graficar MAPE y MAE a lo largo de los periodos."""
    operation_type: Literal["venta", "alquiler"]
    threshold_mape: float = Field(description="Umbral regulatorio máximo (15.0% venta / 17.89% alquiler)")
    time_series: list[PuntoFluctuacionTemporal] = Field(description="Puntos de la serie temporal")
    average_mape: float = Field(description="MAPE promedio en toda la ventana evaluada")
    average_mae: float = Field(description="MAE promedio en toda la ventana evaluada")
    total_evaluated_samples: int = Field(description="Total de transacciones evaluadas en la serie")
    generated_at: datetime = Field(description="Timestamp UTC de generación")


# ─── 4. Latencia P95 y Verificación de Artefactos ─────────────────────────────

class LatenciaP95Response(BaseModel):
    """Evaluación de latencia de inferencia y completitud de artefactos serializados."""
    operation_type: Literal["venta", "alquiler"]
    model_version: str = Field(description="Versión del modelo en RAM evaluado")
    p50_latency_ms: float = Field(description="Latencia mediana (Percentil 50)")
    p95_latency_ms: float = Field(description="Latencia Percentil 95 (objetivo < 100ms SLA)")
    p99_latency_ms: float = Field(description="Latencia Percentil 99")
    sla_threshold_ms: float = Field(default=100.0, description="Límite máximo según SLA")
    meets_sla: bool = Field(description="True si p95_latency_ms <= sla_threshold_ms")
    sample_inferences: int = Field(description="Cantidad de inferencias evaluadas en el benchmark")
    artifacts_status: dict[str, bool] = Field(
        description="Verificación de artefactos esenciales (model.pkl, params.json, etc.)"
    )
    all_artifacts_ready: bool = Field(description="True si todos los artefactos requeridos están presentes")
    measured_at: datetime = Field(description="Timestamp UTC de la medición")
