"""
app/schemas/contexto_distrital.py
──────────────────────────────────
Esquemas Pydantic para el endpoint unificado de Micro-entorno y Contexto Distrital:
  GET /api/v1/distritos/{distrito}/contexto
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class CoordenadasConsultadas(BaseModel):
    lat: float
    lon: float
    es_centroide: bool = Field(
        ...,
        description="True si las coordenadas corresponden al centroide distrital (fallback sin pin del usuario)."
    )


class EstratoSocioeconomico(BaseModel):
    titulo: str = "Estrato Socioeconómico"
    clasificacion: str = Field(..., description="Estrato predominante según clasificación ENAHO-INEI (ej: 'Estrato Alto')")
    decil_referencial: str = Field(..., description="Decil socioeconómico poblacional referencial (ej: 'Deciles 9-10')")
    concentracion_estrato_alto_pct: float = Field(..., description="Concentración de población en estratos altos (%)")
    distribucion_estratos: dict[str, float] = Field(..., description="Desglose porcentual por estrato ENAHO (alto, medio_alto, medio, medio_bajo, bajo)")
    es_imputado: bool = Field(default=False, description="True si los datos distritales de ENAHO provienen de estimación/imputación espacial.")
    alerta_contexto: Optional[str] = Field(default=None, description="Mensaje explicativo si el estrato es imputado por muestra reducida.")
    fuente: str = "INEI / ENAHO 2025"


class SeguridadCiudadana(BaseModel):
    titulo: str = "Seguridad Ciudadana"
    indice: float = Field(..., description="Índice de seguridad de 1.0 a 10.0 (normalización inversa de denuncias)")
    escala: str = "10.0"
    nivel: str = Field(..., description="Nivel cualitativo: 'Alto', 'Medio' o 'Moderado'")
    tasa_robos_10k_hab: float = Field(..., description="Tasa anual de robos por cada 10,000 habitantes (PNP)")
    tasa_denuncias_10k_hab: float = Field(..., description="Tasa global de denuncias por 10k hab (PNP)")
    comisaria_cercana: str = Field(..., description="Nombre de la comisaría o central policial de referencia")
    distancia_comisaria_m: int = Field(..., description="Distancia geodésica en metros a la comisaría")
    descripcion: str
    fuente: str = "Sistema Estadístico de la Policía Nacional del Perú (PNP)"


class MovilidadTransporte(BaseModel):
    titulo: str = "Transporte Público Masivo"
    calificacion: str = Field(..., description="Evaluación urbanística según estándar TOD (<500m Óptima)")
    estacion_cercana: str = Field(..., description="Nombre de la estación o paradero más cercano")
    sistema: str = Field(..., description="Metropolitano, Metro Línea 1/2 o Corredor")
    distancia_m: int = Field(..., description="Distancia geodésica en metros")
    badge_tiempo_a_pie: str = Field(..., description="Tiempo estimado de caminata a pie")
    fuente: str = "OpenStreetMap / ATU"


class AreasVerdes(BaseModel):
    titulo: str = "Parques y Espacios Públicos"
    calificacion: str = Field(..., description="Evaluación ambiental según estándar OMS (<300m Óptima)")
    parque_cercano: str = Field(..., description="Nombre del parque urbano más cercano")
    tipo_parque: str = Field(..., description="Parque Urbano, Malecón, Central o Histórico")
    distancia_m: int = Field(..., description="Distancia geodésica en metros")
    badge_tiempo_a_pie: str = Field(..., description="Tiempo estimado de caminata a pie")
    parques_en_radio_1km: int = Field(..., description="Cantidad de parques catalogados a menos de 1 km")
    fuente: str = "OpenStreetMap Cartography"


class SaludEducacion(BaseModel):
    titulo: str = "Equipamiento de Salud y Educación"
    hospital_clinica_cercana: str
    distancia_salud_m: int
    colegio_cercano: str
    distancia_colegio_m: int
    universidad_cercana: str
    distancia_universidad_km: float
    fuente: str = "OpenStreetMap / MINSA / MINEDU"


class ComercioRetail(BaseModel):
    titulo: str = "Comercio y Servicios"
    centro_comercial_cercano: str
    distancia_comercio_m: int
    fuente: str = "OpenStreetMap"


class ContextoDistritalResponse(BaseModel):
    """Respuesta unificada para pre-predicción en mapa y reporte de inmueble."""
    distrito: str
    coordenadas_consultadas: CoordenadasConsultadas
    estrato_socioeconomico: EstratoSocioeconomico
    seguridad_ciudadana: SeguridadCiudadana
    movilidad_transporte: MovilidadTransporte
    areas_verdes: AreasVerdes
    salud_educacion: SaludEducacion
    comercio_retail: ComercioRetail

