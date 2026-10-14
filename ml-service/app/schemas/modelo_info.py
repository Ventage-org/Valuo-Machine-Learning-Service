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


class MetricasModelo(BaseModel):
    mape_pct: float
    r2: float
    mae_soles: float
    rmse_soles: float
    train_periodo: str
    test_periodo: str


class ModeloInfoResponse(BaseModel):
    """Respuesta de GET /api/v1/modelo/info"""
    version: str
    archivo: str
    metricas: MetricasModelo
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
