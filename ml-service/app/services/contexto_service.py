"""
app/services/contexto_service.py
─────────────────────────────────
Lógica de cálculo estadístico y geoespacial para el micro-entorno y contexto distrital.
Integra la tabla analítica (PostgreSQL/RAM) con el catálogo canónico de hitos urbanos.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from app.schemas.contexto_distrital import (
    AreasVerdes,
    ComercioRetail,
    ContextoDistritalResponse,
    CoordenadasConsultadas,
    EstratoSocioeconomico,
    MovilidadTransporte,
    SaludEducacion,
    SeguridadCiudadana,
)

_CATALOGO_HITOS: Dict[str, Any] = {}
# categoria -> lista plana de hitos de TODOS los distritos (con campo "distrito")
_INDICE_GLOBAL: Dict[str, List[Dict[str, Any]]] = {}
# distrito -> (lat, lon) mediana de sus hitos (aproximación al centro urbano del distrito)
_CENTROS: Dict[str, Tuple[float, float]] = {}


def _cargar_catalogo_hitos() -> Dict[str, Any]:
    """Carga el catálogo una sola vez y construye el índice global por categoría."""
    global _CATALOGO_HITOS
    if _CATALOGO_HITOS:
        return _CATALOGO_HITOS

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta1 = os.path.join(base_dir, "data", "catalogo_hitos_distritales.json")
    ruta2 = os.path.join(os.path.dirname(base_dir), "data", "catalogo_hitos_distritales.json")
    path_final = ruta1 if os.path.exists(ruta1) else ruta2
    if not os.path.exists(path_final):
        raise HTTPException(
            status_code=503,
            detail="Catálogo de hitos urbanos no disponible (catalogo_hitos_distritales.json).",
        )

    with open(path_final, "r", encoding="utf-8") as f:
        catalogo = json.load(f)

    for distrito, categorias in catalogo.items():
        lats: List[float] = []
        lons: List[float] = []
        for categoria, hitos in categorias.items():
            for h in hitos:
                _INDICE_GLOBAL.setdefault(categoria, []).append({**h, "distrito": distrito})
                lats.append(h["lat"])
                lons.append(h["lon"])
        if lats:
            _CENTROS[distrito] = (statistics.median(lats), statistics.median(lons))

    _CATALOGO_HITOS = catalogo
    return _CATALOGO_HITOS


def _rango_tasa_robo(state: Any) -> Tuple[float, float]:
    """Mínimo y máximo REALES de tasa_robo entre los distritos modelados (desde la BD)."""
    tasas: List[float] = []
    provider = getattr(state, "data_provider", None)
    if provider is not None:
        for nombre in provider.listar_distritos():
            try:
                tasas.append(float(provider.get_distrito_context(nombre)["tasa_robo"]))
            except Exception:
                continue
    if len(tasas) < 2:
        raise HTTPException(
            status_code=503,
            detail="Contexto distrital (tasas PNP) no disponible en el proveedor de datos.",
        )
    return min(tasas), max(tasas)


def _haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula la distancia geodésica en metros entre dos coordenadas WGS84."""
    R = 6371000.0  # Radio medio de la Tierra en metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def _calcular_tiempo_a_pie(distancia_m: float) -> str:
    """Estima el tiempo de caminata asumiendo velocidad media de 4.8 km/h (80 m/min)."""
    minutos = max(1, round(distancia_m / 80.0))
    if minutos == 1:
        return "1 min a pie"
    return f"{minutos} min a pie"


def _encontrar_hito_mas_cercano(
    lat: float, lon: float, categoria: str
) -> Tuple[Dict[str, Any], int]:
    """Hito más cercano de una categoría en TODO el catálogo (puede estar en un distrito vecino)."""
    hitos = _INDICE_GLOBAL.get(categoria, [])
    if not hitos:
        raise HTTPException(
            status_code=503,
            detail=f"El catálogo no contiene hitos de la categoría '{categoria}'.",
        )
    mejor = None
    min_dist = float("inf")
    for h in hitos:
        d = _haversine_distance_meters(lat, lon, h["lat"], h["lon"])
        if d < min_dist:
            min_dist = d
            mejor = h
    return mejor, int(round(min_dist))


def obtener_contexto_distrital_enriquecido(
    distrito: str,
    lat: Optional[float],
    lon: Optional[float],
    state: Any
) -> ContextoDistritalResponse:
    """
    Construye la respuesta completa combinando variables macro de PostgreSQL
    con el análisis geoespacial del catálogo de hitos urbanos.
    """
    _cargar_catalogo_hitos()

    # 1. Contexto distrital REAL desde el Data Provider (sin valores por defecto inventados)
    provider = getattr(state, "data_provider", None)
    if provider is None:
        raise HTTPException(status_code=503, detail="Proveedor de contexto distrital no disponible.")
    try:
        ctx_db = provider.get_distrito_context(distrito)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        pct_a = float(ctx_db["pct_NSE_A"])
        pct_b = float(ctx_db["pct_NSE_B"])
        pct_c = float(ctx_db["pct_NSE_C"])
        pct_d = float(ctx_db["pct_NSE_D"])
        pct_e = float(ctx_db["pct_NSE_E"])
        tasa_robo = float(ctx_db["tasa_robo"])
        tasa_denuncias = float(ctx_db["tasa_denuncias"])
    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"El contexto distrital de '{distrito}' está incompleto en el proveedor: {exc}",
        )

    # 2. Coordenadas: pin del usuario o centro urbano del distrito (mediana de sus hitos)
    es_centroide = False
    if lat is None or lon is None:
        es_centroide = True
        if distrito not in _CENTROS:
            raise HTTPException(
                status_code=404,
                detail=f"El catálogo no tiene hitos para '{distrito}'; envíe lat y lon.",
            )
        lat, lon = _CENTROS[distrito]

    # 3. Estrato Socioeconómico (INEI / APEIM)
    pct_ab = round(pct_a + pct_b, 1)
    if pct_ab >= 60.0:
        estrato = "Nivel A / B"
        decil = "Deciles 9-10 (Estrato Alto)"
    elif pct_ab >= 30.0:
        estrato = "Nivel B / C"
        decil = "Deciles 6-8 (Estrato Medio-Alto)"
    else:
        estrato = "Nivel C / D"
        decil = "Deciles 1-5 (Estrato Medio-Bajo)"

    # 4. Seguridad Ciudadana (PNP)
    # Normalización Min-Max inversa
    min_robo, max_robo = _rango_tasa_robo(state)
    denom = max(1.0, max_robo - min_robo)
    ind_calc = 10.0 - ((tasa_robo - min_robo) / denom * 9.0)
    indice_seguridad = round(max(1.0, min(10.0, ind_calc)), 1)
    
    if indice_seguridad >= 7.5:
        nivel_seg = "Alto"
        desc_seg = "Baja tasa relativa de delitos patrimoniales. Cobertura de patrullaje y videovigilancia."
    elif indice_seguridad >= 5.0:
        nivel_seg = "Medio"
        desc_seg = "Incidencia delictiva promedio de Lima Metropolitana. Cuadrantes urbanos con vigilancia activa."
    else:
        nivel_seg = "Moderado / Alerta"
        desc_seg = "Sector comercial con alta afluencia peatonal. Se recomienda verificar cuadrante de seguridad."

    comisaria, dist_comisaria = _encontrar_hito_mas_cercano(lat, lon, "seguridad")

    # 5. Movilidad y Transporte (TOD Standard)
    estacion, dist_estacion = _encontrar_hito_mas_cercano(lat, lon, "transporte")
    if dist_estacion < 500:
        calif_trans = "Accesibilidad Vial Óptima"
    elif dist_estacion < 1000:
        calif_trans = "Accesibilidad Vial Media"
    else:
        calif_trans = "Accesibilidad Intermedia / Requiere Conexión"

    # 6. Áreas Verdes (OMS Standard)
    parques_lista = _INDICE_GLOBAL.get("parques", [])
    parque_cercano, dist_parque = _encontrar_hito_mas_cercano(lat, lon, "parques")
    
    # Contar parques a menos de 1 km
    parques_1km = sum(
        1 for p in parques_lista if _haversine_distance_meters(lat, lon, p["lat"], p["lon"]) <= 1000.0
    )
    if dist_parque < 300:
        calif_parques = "Áreas verdes a corta distancia (< 300 m)"
    elif dist_parque < 700:
        calif_parques = "Parques y Espacios Públicos Accesibles"
    else:
        calif_parques = "Baja densidad de áreas verdes inmediatas"

    # 7. Salud y Educación
    salud_hito, dist_salud = _encontrar_hito_mas_cercano(lat, lon, "salud")
    colegio_hito, dist_colegio = _encontrar_hito_mas_cercano(lat, lon, "educacion_colegios")
    uni_hito, dist_uni = _encontrar_hito_mas_cercano(lat, lon, "universidades")

    # 8. Comercio y Retail
    mall_hito, dist_mall = _encontrar_hito_mas_cercano(lat, lon, "centros_comerciales")

    return ContextoDistritalResponse(
        distrito=distrito,
        coordenadas_consultadas=CoordenadasConsultadas(
            lat=round(lat, 6),
            lon=round(lon, 6),
            es_centroide=es_centroide,
        ),
        estrato_socioeconomico=EstratoSocioeconomico(
            estrato=estrato,
            decil=decil,
            porcentaje_a_b=pct_ab,
            distribucion_detallada={
                "A": round(pct_a, 1),
                "B": round(pct_b, 1),
                "C": round(pct_c, 1),
                "D": round(pct_d, 1),
                "E": round(pct_e, 1),
            },
        ),
        seguridad_ciudadana=SeguridadCiudadana(
            indice=indice_seguridad,
            nivel=nivel_seg,
            tasa_robos_10k_hab=round(tasa_robo, 2),
            tasa_denuncias_10k_hab=round(tasa_denuncias, 2),
            comisaria_cercana=comisaria.get("nombre", "Comisaría Central"),
            distancia_comisaria_m=dist_comisaria,
            descripcion=desc_seg,
        ),
        movilidad_transporte=MovilidadTransporte(
            calificacion=calif_trans,
            estacion_cercana=estacion.get("nombre", "Paradero Principal"),
            sistema=estacion.get("sistema", "Transporte Metropolitano"),
            distancia_m=dist_estacion,
            badge_tiempo_a_pie=_calcular_tiempo_a_pie(dist_estacion),
        ),
        areas_verdes=AreasVerdes(
            calificacion=calif_parques,
            parque_cercano=parque_cercano.get("nombre", "Parque del Distrito"),
            tipo_parque=parque_cercano.get("tipo", "Parque Urbano"),
            distancia_m=dist_parque,
            badge_tiempo_a_pie=_calcular_tiempo_a_pie(dist_parque),
            parques_en_radio_1km=parques_1km,
        ),
        salud_educacion=SaludEducacion(
            hospital_clinica_cercana=salud_hito.get("nombre", "Centro de Salud"),
            distancia_salud_m=dist_salud,
            colegio_cercano=colegio_hito.get("nombre", "Colegio Representativo"),
            distancia_colegio_m=dist_colegio,
            universidad_cercana=uni_hito.get("nombre", "Campus Universitario"),
            distancia_universidad_km=round(dist_uni / 1000.0, 1),
        ),
        comercio_retail=ComercioRetail(
            centro_comercial_cercano=mall_hito.get("nombre", "Centro Comercial"),
            distancia_comercio_m=dist_mall,
        ),
    )


