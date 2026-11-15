"""
app/schemas/modelo_info.py
──────────────────────────
Schema Pydantic para GET /api/v1/modelo/info y GET /api/v1/distritos.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IpcInfo(BaseModel):
    valor: float = Field(description="Valor del IPC (Base Dic 2009 = 100).")
    periodo: str = Field(description="Periodo del IPC (ej: '2026-Q1').")
    fuente: str = Field(description="Fuente oficial del IPC.")
    nota: str = Field(description="Instrucciones para actualización.")


class MetricasIAAO(BaseModel):
    """Métricas oficiales según IAAO Standard on Ratio Studies (2013/2020) para AVM."""
    cod_pct: float = Field(description="Coeficiente de Dispersión (COD <= 15.0% según estándar IAAO)")
    cov_pct: float = Field(description="Coeficiente de Variación paramétrico (COV)")
    prd: float = Field(description="Price-Related Differential (objetivo 0.98 - 1.03)")
    prb: float = Field(description="Price-Related Bias (objetivo -0.05 a +0.05)")
    median_ratio: float = Field(default=1.00, description="Nivel de evaluación (0.90 - 1.10)")
    mean_ratio: float = Field(default=1.00, description="Media de ratios de valuación")
    weighted_mean_ratio: float = Field(default=1.00, description="Media ponderada por valor")
    cumple_iaao_cod: bool = Field(default=True, description="Cumple cota COD <= 15.0%")
    cumple_iaao_nivel: bool = Field(default=True, description="Cumple nivel de tasación 0.90 - 1.10")
    cumple_iaao_prd: bool = Field(default=True, description="Cumple equidad vertical 0.98 - 1.03")
    evaluacion_general: str = Field(
        default="Aceptable para Tasación Automatizada AVM según estándar IAAO",
        description="Dictamen metodológico formal",
    )


class MetricasModelo(BaseModel):
    mape_pct: float
    r2: float
    mae_soles: float
    rmse_soles: float
    cod_iaao: float | None = Field(default=None, description="Coeficiente de Dispersión COD (%)")
    cov_iaao: float | None = Field(default=None, description="Coeficiente de Variación COV (%)")
    prd_iaao: float | None = Field(default=None, description="Price-Related Differential PRD")
    prb_iaao: float | None = Field(default=None, description="Price-Related Bias PRB")
    train_periodo: str
    test_periodo: str


class ModeloInfoResponse(BaseModel):
    """Respuesta de GET /api/v1/modelo/info"""
    version: str
    archivo: str
    metricas: MetricasModelo
    iaao: MetricasIAAO | None = Field(default=None, description="Métricas de valuación masiva IAAO")
    ipc_actual: IpcInfo
    n_features: int
    n_distritos: int
    data_provider: str = Field(description="Fuente de datos contextual activa ('csv' o 'mongo').")
    distritos_disponibles: list[str] = Field(
        description="Lista completa de distritos disponibles para predicción."
    )


class DistritosResponse(BaseModel):
    """Respuesta de GET /api/v1/distritos"""
    distritos: list[str]
    total: int
    nota: str = Field(
        default="Usa el nombre exacto (case-sensitive) en el campo 'distrito' del request de predicción."
    )
