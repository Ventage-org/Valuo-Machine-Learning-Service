"""
app/schemas/mercado.py
──────────────────────
Contratos Pydantic para el endpoint público de Tendencias de Mercado por Distrito:
  GET /api/v1/mercado/tendencias-distritales
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class TendenciaDistrito(BaseModel):
    """Estadísticas inmobiliarias de mercado para un distrito en el periodo actual."""
    distrito: str = Field(description="Nombre del distrito de Lima Metropolitana.")
    precio_m2_usd: float = Field(description="Precio mediano por m² en dólares americanos (USD).")
    precio_m2_pen: float = Field(description="Precio mediano por m² en soles nominales (PEN).")
    variacion_trimestral_pct: float = Field(
        description="Variación porcentual del precio m² respecto al trimestre inmediato anterior."
    )
    ticket_promedio_usd: float = Field(description="Precio total promedio o mediano del inmueble en USD.")
    ticket_promedio_pen: float = Field(description="Precio total promedio o mediano del inmueble en PEN.")
    superficie_promedio_m2: float = Field(description="Superficie promedio de las unidades en m².")
    total_muestras: int = Field(description="Cantidad de observaciones analizadas en el periodo.")


class TendenciasMercadoResponse(BaseModel):
    """Respuesta consolidada de tendencias distritales para el dashboard."""
    tipo_operacion: Literal["venta", "alquiler"] = Field(description="Tipo de operación inmobiliaria analizada.")
    periodo_actual: str = Field(description="Periodo de análisis actual (ej: '2025-Q4').")
    periodo_anterior: str = Field(description="Periodo de comparación anterior (ej: '2025-Q3').")
    tipo_cambio: float = Field(description="Tipo de cambio referencial PEN/USD aplicado.")
    ipc_factor: float = Field(description="Factor inflacionario IPC aplicado sobre moneda base constante.")
    total_distritos: int = Field(description="Número de distritos con datos válidos reportados.")
    distritos: list[TendenciaDistrito] = Field(
        description="Lista de distritos con métricas de precios m² y variación trimestral."
    )
    nota: str = Field(
        default="Métricas calculadas a partir de la mediana real de transacciones y ofertas históricas ajustadas por el IPC oficial.",
        description="Nota metodológica."
    )

