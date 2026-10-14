"""
app/schemas/sensibilidad.py
───────────────────────────
Contratos Pydantic para el endpoint público de Análisis de Sensibilidad de Factores:
  GET /api/v1/modelos/sensibilidad-factores
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class FactorSensibilidad(BaseModel):
    """Macro-factor que influye en el valor del inmueble."""
    id: str = Field(description="Identificador del macro-factor (ej: 'ubicacion_entorno').")
    factor: str = Field(description="Nombre descriptivo del macro-factor para la interfaz de usuario.")
    descripcion: str = Field(description="Explicación concisa de las características que conforman este factor.")
    peso_porcentaje: float = Field(
        description="Porcentaje relativo de ponderación sobre el 100% total de la estimación."
    )
    nivel_impacto: str = Field(
        description="Categoría cualitativa: 'Impacto Muy Alto' (≥30%), 'Impacto Alto' (20-29.9%), 'Impacto Moderado' (10-19.9%), 'Impacto Leve' (<10%)."
    )
    variables_incluidas: list[str] = Field(
        description="Lista de variables técnicas internas del modelo asociadas a este macro-factor."
    )


class SensibilidadFactoresResponse(BaseModel):
    """Respuesta canónica de GET /api/v1/modelos/sensibilidad-factores."""
    modelo: str = Field(description="Nombre del modelo activo en producción (ej: 'xgboost_venta_v2').")
    tipo_operacion: Literal["venta", "alquiler"] = Field(description="Tipo de operación inmobiliaria analizada.")
    version: str = Field(description="Versión del modelo.")
    metodo: str = Field(
        default="Análisis de Sensibilidad por Ganancia de Información (XGBoost Feature Attribution)",
        description="Metodología algorítmica utilizada para descomponer la sensibilidad de los factores."
    )
    periodo_referencia: str = Field(description="Periodo temporal de entrenamiento del modelo.")
    factores_principales: list[FactorSensibilidad] = Field(
        description="Lista de macro-factores ordenados por peso_porcentaje descendente. La suma total es 100.0%."
    )
    cobertura_macro_factores_pct: float = Field(
        default=100.0,
        description="Porcentaje acumulado de cobertura de los factores sobre el modelo."
    )
    nota: str = Field(
        default="Los porcentajes representan la sensibilidad promedio de mercado en Lima Metropolitana calculada a partir de los árboles de decisión del modelo.",
        description="Nota metodológica para el frontend y documentación."
    )
