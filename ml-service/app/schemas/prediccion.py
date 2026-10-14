"""
app/schemas/prediccion.py
─────────────────────────
Contratos Pydantic para el endpoint POST /api/v1/prediccion/venta.

Incluye el campo `explicabilidad` con los valores SHAP top-N
(impacto de cada variable en la predicción individual).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── REQUEST ──────────────────────────────────────────────────────────────────

class PrediccionVentaRequest(BaseModel):
    """
    Datos del inmueble para predecir su precio de venta.

    Las variables contextuales del distrito (NSE, criminalidad, distancias,
    demografía) son resueltas automáticamente por el servicio a partir
    del nombre del distrito.
    """

    distrito: str = Field(
        ...,
        description="Nombre del distrito de Lima Metropolitana. "
                    "Usar /api/v1/distritos para ver los disponibles.",
        examples=["San Miguel", "Miraflores", "La Molina"],
    )
    superficie: float = Field(
        ..., gt=10, lt=2000,
        description="Superficie total del inmueble en m².",
        examples=[75.0],
    )
    habitaciones: int = Field(
        ..., ge=1, le=20,
        description="Número de habitaciones/dormitorios.",
        examples=[3],
    )
    banios: float = Field(
        ..., ge=0.5, le=15.0,
        description="Número de baños (admite medios baños ej: 1.5, 2.0, 2.5).",
        examples=[2.0],
    )
    garajes: int = Field(
        default=0, ge=0, le=10,
        description="Número de cocheras/garajes. 0 si no tiene.",
        examples=[1],
    )
    piso: int = Field(
        ..., ge=1, le=50,
        description="Piso en el que se ubica el inmueble.",
        examples=[5],
    )
    antiguedad: int = Field(
        ..., ge=0, le=100,
        description="Antigüedad del inmueble en años.",
        examples=[8],
    )
    vista_exterior: bool = Field(
        default=True,
        description="True si el inmueble tiene vista hacia el exterior.",
    )
    anio: int | None = Field(
        default=None,
        ge=2016, le=2030,
        description="Año para la predicción (opcional). Si se omite, el backend lo calcula automáticamente con el año actual.",
        examples=[None],
    )
    trimestre: int | None = Field(
        default=None,
        ge=1, le=4,
        description="Trimestre para la predicción (opcional, 1–4). Si se omite, el backend lo calcula automáticamente con la fecha actual.",
        examples=[None],
    )
    @field_validator("distrito")
    @classmethod
    def distrito_no_vacio(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("El campo 'distrito' no puede estar vacío.")
        return v

    @field_validator("superficie")
    @classmethod
    def superficie_razonable(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("La superficie debe ser mayor a 0 m².")
        return round(v, 2)

    model_config = {"json_schema_extra": {
        "example": {
            "distrito": "San Miguel",
            "superficie": 75.0,
            "habitaciones": 3,
            "banios": 2,
            "garajes": 1,
            "piso": 5,
            "antiguedad": 8,
            "vista_exterior": True,
        }
    }}


# ─── RESPONSE — sub-schemas ───────────────────────────────────────────────────

class ValoresPrediccion(BaseModel):
    """Precios predichos en soles constantes y nominales."""
    soles_constantes: float = Field(
        description="Precio en Soles Constantes (Base Dic 2009 = 100). "
                    "Valor interno del modelo, aislado de inflación.",
    )
    soles_nominales: float = Field(
        description="Precio en Soles Nominales del periodo IPC vigente. "
                    "= soles_constantes × (IPC_actual / 100).",
    )
    precio_m2_constantes: float = Field(description="Precio por m² en Soles Constantes.")
    precio_m2_nominales: float = Field(description="Precio por m² en Soles Nominales.")


class IntervaloConfianza(BaseModel):
    """Intervalo de confianza basado en el MAPE del modelo en test set."""
    inferior_nominales: float = Field(
        description="Límite inferior del intervalo en Soles Nominales."
    )
    superior_nominales: float = Field(
        description="Límite superior del intervalo en Soles Nominales."
    )
    inferior_constantes: float = Field(
        description="Límite inferior del intervalo en Soles Constantes."
    )
    superior_constantes: float = Field(
        description="Límite superior del intervalo en Soles Constantes."
    )
    mape_pct: float = Field(
        description="MAPE del modelo en el test set 2024-2025. "
                    "El intervalo es ± este porcentaje sobre el precio predicho."
    )


class ContribucionFeature(BaseModel):
    """
    Contribución SHAP de una variable individual a la predicción.
    Representa cuánto subió o bajó el precio predicho por esta variable.
    """
    feature: str = Field(description="Nombre interno de la feature en el modelo.")
    label: str = Field(description="Etiqueta legible para mostrar al usuario final.")
    shap_value: float = Field(
        description="Valor SHAP en escala logarítmica (escala interna del modelo). "
                    "Positivo = empuja el precio hacia arriba; negativo = hacia abajo."
    )
    valor_feature: float = Field(description="Valor real de la feature para este inmueble.")
    impacto: Literal["positivo", "negativo", "neutro"] = Field(
        description="Dirección del impacto de esta variable en el precio."
    )
    peso_porcentaje: float = Field(
        description="Porcentaje relativo de peso sobre el 100% total del impacto del modelo. "
                    "La suma de peso_porcentaje de todas las variables es exactamente 100.0%."
    )
    nivel_impacto: str = Field(
        description="Categoría cualitativa: 'Impacto Muy Alto' (≥20%), 'Impacto Alto' (10-19.9%), 'Impacto Moderado' (5-9.9%), 'Impacto Leve' (<5%)."
    )


class Explicabilidad(BaseModel):
    """
    Explicación SHAP de la predicción individual.
    Retorna la totalidad de variables del modelo ordenadas por magnitud de impacto para almacenamiento en BD y consumo del backend.
    """
    valor_base_constantes: float = Field(
        description="Precio base del modelo (media global de entrenamiento en Soles Constantes). "
                    "Punto de partida antes de aplicar las contribuciones de cada variable."
    )
    contribuciones: list[ContribucionFeature] = Field(
        description="Lista completa de las features del modelo ordenadas por |shap_value| descendente "
                    "para almacenamiento en BD y decisión de visualización en el backend."
    )
    n_features_mostradas: int = Field(
        description="Número de features en contribuciones (todas las variables del modelo)."
    )
    n_features_total: int = Field(
        default=33,
        description="Número total de features del modelo (33)."
    )
    nota: str = Field(
        default="Los shap_values están en escala logarítmica (transformación log1p del target). "
                "'peso_porcentaje' suma exactamente 100.0% entre la totalidad de variables del modelo.",
        description="Nota metodológica sobre la escala de los valores SHAP."
    )


class InfoModelo(BaseModel):
    """Metadata del modelo incluida en cada response."""
    version: str
    mape_test: float
    r2_test: float
    ipc_factor: float
    periodo_ipc: str
    train_periodo: str
    test_periodo: str


# ─── RESPONSE COMPLETO ────────────────────────────────────────────────────────

class PrediccionVentaResponse(BaseModel):
    """
    Respuesta completa del endpoint de predicción de venta.
    Incluye precio, intervalo de confianza, explicabilidad SHAP y metadata del modelo.
    """
    distrito: str
    superficie_m2: float
    prediccion: ValoresPrediccion
    intervalo_confianza: IntervaloConfianza
    explicabilidad: Explicabilidad
    modelo: InfoModelo

    model_config = {"json_schema_extra": {
        "example": {
            "distrito": "San Miguel",
            "superficie_m2": 75.0,
            "prediccion": {
                "soles_constantes": 287000.0,
                "soles_nominales": 485416.0,
                "precio_m2_constantes": 3826.67,
                "precio_m2_nominales": 6472.21,
            },
            "intervalo_confianza": {
                "inferior_nominales": 412000.0,
                "superior_nominales": 558000.0,
                "inferior_constantes": 243780.0,
                "superior_constantes": 330220.0,
                "mape_pct": 15.04,
            },
            "explicabilidad": {
                "valor_base_constantes": 432295.69,
                "contribuciones": [
                    {
                        "feature": "Superficie",
                        "label": "Superficie (75.0 m²)",
                        "shap_value": 0.312,
                        "valor_feature": 75.0,
                        "impacto": "positivo",
                    },
                    {
                        "feature": "distrito_encoded",
                        "label": "Distrito (San Miguel)",
                        "shap_value": -0.198,
                        "valor_feature": 287982.23,
                        "impacto": "negativo",
                    },
                ],
                "n_features_mostradas": 10,
                "n_features_total": 33,
                "nota": "Los shap_values están en escala logarítmica.",
            },
            "modelo": {
                "version": "v2",
                "mape_test": 15.04,
                "r2_test": 0.7469,
                "ipc_factor": 169.18,
                "periodo_ipc": "2026-Q1",
                "train_periodo": "2016-2023",
                "test_periodo": "2024-2025",
            },
        }
    }}
