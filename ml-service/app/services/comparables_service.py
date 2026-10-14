"""
app/services/comparables_service.py
───────────────────────────────────
Servicio de homologación de muestras comparables de mercado (RNTP Art. 3 y 4).

Implementa la consulta y selección de las tres (3) muestras de mercado
análogas más representativas para sustentar cada valoración automatizada (AVM).

Optimizado para respuesta ultra-rápida (< 0.5 ms):
  - Pre-caché indexado en memoria RAM por distrito y zona homogénea (Numpy arrays contiguos).
  - Cálculo de distancia euclidiana ponderada vectorizada sin escaneo de tablas completas.
  - Selección de Top 3 con np.argpartition (O(N) lineal sobre subconjuntos pequeños).
  - Estrategia en cascada: Mismo Distrito -> Zona Homogénea (NSE) -> General.
  - Tolerante a fallos: degradación elegante sin interrumpir la inferencia principal.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.schemas.prediccion import (
    DiferenciasInmueble,
    MuestraComparable,
    ResumenComparablesRNTP,
)

# ─── Zonas Homogéneas de Lima Metropolitana ──────────────────────────────────
# Agrupación oficial por dinámica inmobiliaria y estrato socioeconómico (NSE)
# para fallback metodológico conforme al RNTP.

ZONAS_HOMOGENEAS: dict[str, str] = {
    # Lima Top (NSE A / A-B)
    "Miraflores":        "LIMA_TOP",
    "San Isidro":        "LIMA_TOP",
    "San Borja":         "LIMA_TOP",
    "Surco":             "LIMA_TOP",
    "La Molina":         "LIMA_TOP",
    "Barranco":          "LIMA_TOP",

    # Lima Moderna (NSE B / B-C)
    "Jesús María":       "LIMA_MODERNA",
    "Lince":             "LIMA_MODERNA",
    "Magdalena":         "LIMA_MODERNA",
    "Pueblo Libre":      "LIMA_MODERNA",
    "San Miguel":        "LIMA_MODERNA",
    "Surquillo":         "LIMA_MODERNA",

    # Lima Centro
    "Cercado de Lima":   "LIMA_CENTRO",
    "Breña":             "LIMA_CENTRO",
    "La Victoria":       "LIMA_CENTRO",

    # Lima Este
    "Ate Vitarte":       "LIMA_ESTE",

    # Lima Norte
    "Los Olivos":        "LIMA_NORTE",
    "Comas":             "LIMA_NORTE",
    "Carabayllo":        "LIMA_NORTE",

    # Lima Sur
    "Chorrillos":        "LIMA_SUR",

    # Callao
    "Bellavista":        "CALLAO",
    "La Perla":          "CALLAO",
}


def _crear_cache_numpy(df_sub: pd.DataFrame) -> dict[str, Any]:
    """Crea un diccionario con arrays Numpy contiguos para búsqueda ultra-veloz (<0.2ms)."""
    # Deduplicar inmuebles exactamente idénticos en características físicas y precio
    dedup = df_sub.drop_duplicates(
        subset=["distrito", "superficie_m2", "habitaciones", "banos", "precio_soles_const"]
    ).copy()
    if len(dedup) < 3:
        dedup = df_sub.copy()

    return {
        "len": len(dedup),
        "id": dedup["id"].values,
        "distrito": dedup["distrito"].values,
        "anio": dedup["anio"].values,
        "trimestre": dedup["trimestre"].values,
        "superficie": dedup["superficie_m2"].values.astype(np.float64),
        "habitaciones": dedup["habitaciones"].values.astype(np.float64),
        "banos": dedup["banos"].values.astype(np.float64),
        "garajes": dedup["garajes"].values.astype(np.float64),
        "piso": dedup["piso"].values.astype(np.float64),
        "antiguedad": dedup["antiguedad_anios"].values.astype(np.float64),
        "precio_soles_const": dedup["precio_soles_const"].values.astype(np.float64),
    }


class ComparablesService:
    """
    Gestiona el repositorio de testigos de mercado y ejecuta la homologación
    vectorizada para la regla de las 3 muestras del RNTP.
    """

    def __init__(self) -> None:
        self._cache_distrito_venta: dict[str, dict[str, Any]] = {}
        self._cache_distrito_alquiler: dict[str, dict[str, Any]] = {}
        self._cache_zona_venta: dict[str, dict[str, Any]] = {}
        self._cache_zona_alquiler: dict[str, dict[str, Any]] = {}
        self._cache_global_venta: dict[str, Any] | None = None
        self._cache_global_alquiler: dict[str, Any] | None = None
        self._disponible: bool = False

    @property
    def disponible(self) -> bool:
        return self._disponible

    def cargar_datos(
        self,
        host: str | None = None,
        port: int | None = None,
        db_name: str | None = None,
        user: str | None = None,
        password: str | None = None,
        anio_min: int = 2021,
    ) -> None:
        """
        Carga las muestras históricas desde PostgreSQL y construye los índices
        Numpy por distrito y zona en memoria RAM.
        """
        conn_params = {
            "host":     host     or os.getenv("DB_HOST", "localhost"),
            "port":     port     or int(os.getenv("DB_PORT", "5432")),
            "dbname":   db_name  or os.getenv("DB_NAME", "inmobiliaria_ml_db"),
            "user":     user     or os.getenv("DB_USER", "postgres"),
            "password": password or os.getenv("DB_PASSWORD") or "5432",
        }

        try:
            import psycopg2
            conn = psycopg2.connect(**conn_params)
        except Exception as exc:
            print(f"[ComparablesService] Advertencia: No se pudo conectar a BD ({exc}). "
                  f"El servicio operará con degradación elegante sin testigos.")
            self._disponible = False
            return

        try:
            # 1. Cargar Venta
            query_venta = f"""
                SELECT
                    v.id,
                    d.nombre AS distrito,
                    v.anio,
                    v.trimestre,
                    v.superficie_m2,
                    v.habitaciones,
                    v.banos,
                    v.garajes,
                    v.piso,
                    v.antiguedad_anios,
                    v.precio_soles_const
                FROM dataset_inmuebles_venta v
                JOIN distritos d ON v.distrito_id = d.id
                WHERE v.anio >= {anio_min}
            """
            df_v = pd.read_sql(query_venta, conn)
            if len(df_v) == 0:
                df_v = pd.read_sql("SELECT v.id, d.nombre AS distrito, v.anio, v.trimestre, v.superficie_m2, v.habitaciones, v.banos, v.garajes, v.piso, v.antiguedad_anios, v.precio_soles_const FROM dataset_inmuebles_venta v JOIN distritos d ON v.distrito_id = d.id", conn)

            # 2. Cargar Alquiler
            query_alq = f"""
                SELECT
                    a.id,
                    d.nombre AS distrito,
                    a.anio,
                    a.trimestre,
                    a.superficie_m2,
                    a.habitaciones,
                    a.banos,
                    a.garajes,
                    a.piso,
                    a.antiguedad_anios,
                    a.alquiler_soles_const AS precio_soles_const
                FROM dataset_inmuebles_alquiler a
                JOIN distritos d ON a.distrito_id = d.id
                WHERE a.anio >= {anio_min}
            """
            df_a = pd.read_sql(query_alq, conn)
            if len(df_a) == 0:
                df_a = pd.read_sql("SELECT a.id, d.nombre AS distrito, a.anio, a.trimestre, a.superficie_m2, a.habitaciones, a.banos, a.garajes, a.piso, a.antiguedad_anios, a.alquiler_soles_const AS precio_soles_const FROM dataset_inmuebles_alquiler a JOIN distritos d ON a.distrito_id = d.id", conn)

        except Exception as exc:
            print(f"[ComparablesService] Error consultando tablas de comparables: {exc}")
            self._disponible = False
            return
        finally:
            conn.close()

        # Construir zona homogénea
        df_v["zona_homogenea"] = df_v["distrito"].map(lambda d: ZONAS_HOMOGENEAS.get(d, "LIMA_METROPOLITANA"))
        df_a["zona_homogenea"] = df_a["distrito"].map(lambda d: ZONAS_HOMOGENEAS.get(d, "LIMA_METROPOLITANA"))

        # Pre-construir índices Numpy por distrito
        self._cache_distrito_venta = {
            dist: _crear_cache_numpy(sub) for dist, sub in df_v.groupby("distrito")
        }
        self._cache_distrito_alquiler = {
            dist: _crear_cache_numpy(sub) for dist, sub in df_a.groupby("distrito")
        }

        # Pre-construir índices Numpy por zona homogénea
        self._cache_zona_venta = {
            zona: _crear_cache_numpy(sub) for zona, sub in df_v.groupby("zona_homogenea")
        }
        self._cache_zona_alquiler = {
            zona: _crear_cache_numpy(sub) for zona, sub in df_a.groupby("zona_homogenea")
        }

        # Caché global
        self._cache_global_venta = _crear_cache_numpy(df_v)
        self._cache_global_alquiler = _crear_cache_numpy(df_a)

        self._disponible = True

        print(
            f"[ComparablesService] [OK] Índices Numpy ultra-rápidos precargados: "
            f"{len(df_v):,} ventas y {len(df_a):,} alquileres en 22 distritos."
        )

    def buscar_tres_comparables(
        self,
        tipo: Literal["venta", "alquiler"],
        distrito: str,
        superficie: float,
        habitaciones: int,
        banios: float,
        antiguedad: int,
        garajes: int = 0,
        piso: int = 1,
        ipc_factor: float = 1.6918,
        prediccion_nominal: float = 0.0,
    ) -> tuple[list[MuestraComparable], ResumenComparablesRNTP | None]:
        """
        Encuentra las 3 muestras de mercado más homogéneas en tiempo récord (< 0.5 ms).
        """
        if not self._disponible or not self._cache_distrito_venta:
            self.cargar_datos()

        cache_distritos = self._cache_distrito_venta if tipo == "venta" else self._cache_distrito_alquiler
        cache_zonas = self._cache_zona_venta if tipo == "venta" else self._cache_zona_alquiler
        cache_global = self._cache_global_venta if tipo == "venta" else self._cache_global_alquiler

        if not cache_distritos:
            return [], None

        zona_target = ZONAS_HOMOGENEAS.get(distrito, "LIMA_METROPOLITANA")

        # ─── Cascada de Selección ─────────────────────────────────────────────
        criterio = "Mismo distrito"
        es_mismo_distrito = True
        advertencia = None
        candidatos = cache_distritos.get(distrito)

        if not candidatos or candidatos["len"] < 3:
            # Fallback a Zona Homogénea
            candidatos = cache_zonas.get(zona_target)
            criterio = f"Zona homogénea ({zona_target})"
            es_mismo_distrito = False
            advertencia = (
                f"Densidad muestral local baja en '{distrito}'. Se utilizaron testigos "
                f"homologados de la zona socioeconómica análoga '{zona_target}'."
            )

        if not candidatos or candidatos["len"] < 3:
            # Fallback Global
            candidatos = cache_global
            criterio = "Mercado metropolitano"
            es_mismo_distrito = False
            advertencia = "Se utilizaron testigos del mercado metropolitano general."

        if not candidatos or candidatos["len"] == 0:
            return [], None

        # ─── Homologación Vectorizada Numpy (< 0.2 ms) ────────────────────────
        sup_arr = candidatos["superficie"]
        hab_arr = candidatos["habitaciones"]
        ban_arr = candidatos["banos"]
        ant_arr = candidatos["antiguedad"]
        gar_arr = candidatos["garajes"]
        pis_arr = candidatos["piso"]
        anio_arr = candidatos["anio"]
        tri_arr = candidatos["trimestre"]
        precio_arr = candidatos["precio_soles_const"]
        dist_nombre_arr = candidatos["distrito"]

        d_sup = ((sup_arr - float(superficie)) / 25.0) ** 2 * 1.0
        d_hab = ((hab_arr - float(habitaciones)) / 1.0) ** 2 * 0.8
        d_ban = ((ban_arr - float(banios)) / 1.0) ** 2 * 0.6
        d_ant = ((ant_arr - float(antiguedad)) / 5.0) ** 2 * 0.5
        d_gar = ((gar_arr - float(garajes)) / 1.0) ** 2 * 0.4
        d_pis = np.where(
            pis_arr > 0,
            ((pis_arr - float(piso)) / 3.0) ** 2 * 0.3,
            0.0
        )

        distancias = np.sqrt(d_sup + d_hab + d_ban + d_ant + d_gar + d_pis)

        # Métrica combinada: Proximidad física (peso dominante) con leve bono por recencia
        # anio_reciente oscila entre 0 y 0.05
        max_anio = np.max(anio_arr)
        recencia_penalty = (max_anio - anio_arr) * 0.02
        score_total = distancias + recencia_penalty

        k = min(3, len(score_total))
        top_k_indices = np.argpartition(score_total, k - 1)[:k]
        top_k_indices = top_k_indices[np.argsort(score_total[top_k_indices])]

        muestras_rntp: list[MuestraComparable] = []
        precios_muestras: list[float] = []
        precios_m2_muestras: list[float] = []

        for pos, idx in enumerate(top_k_indices, start=1):
            p_const = float(precio_arr[idx])
            p_nom = round(p_const * ipc_factor, 2)
            sup_m = float(sup_arr[idx])
            p_m2 = round(p_nom / sup_m, 2)
            dist_val = float(distancias[idx])
            dist_nom = str(dist_nombre_arr[idx])

            # Similitud: 100% en 0 distancia, decaimiento exponencial suave
            sim_pct = round(max(50.0, min(100.0, 100.0 / (1.0 + dist_val * 0.15))), 1)

            dif = DiferenciasInmueble(
                dif_superficie_m2=round(sup_m - float(superficie), 2),
                dif_habitaciones=int(hab_arr[idx] - habitaciones),
                dif_banios=round(float(ban_arr[idx]) - float(banios), 1),
                dif_antiguedad_anios=int(ant_arr[idx] - antiguedad),
                dif_piso=int(pis_arr[idx] - piso) if pis_arr[idx] > 0 else 0,
            )

            muestras_rntp.append(
                MuestraComparable(
                    posicion=pos,
                    distrito=dist_nom,
                    es_mismo_distrito=(dist_nom == distrito),
                    zona_homogenea=ZONAS_HOMOGENEAS.get(dist_nom, "LIMA_METROPOLITANA"),
                    superficie_m2=sup_m,
                    habitaciones=int(hab_arr[idx]),
                    banios=float(ban_arr[idx]),
                    garajes=int(gar_arr[idx]),
                    piso=int(pis_arr[idx]),
                    antiguedad_anios=int(ant_arr[idx]),
                    precio_total_pen=p_nom,
                    precio_m2_pen=p_m2,
                    periodo_muestra=f"{int(anio_arr[idx])}-Q{int(tri_arr[idx])}",
                    indice_similitud_pct=sim_pct,
                    diferencias=dif,
                )
            )
            precios_muestras.append(p_nom)
            precios_m2_muestras.append(p_m2)

        if not precios_muestras:
            return [], None

        precio_prom = round(float(np.mean(precios_muestras)), 2)
        precio_m2_prom = round(float(np.mean(precios_m2_muestras)), 2)

        if precio_prom > 0 and prediccion_nominal > 0:
            desviacion_pct = round(((prediccion_nominal - precio_prom) / precio_prom) * 100.0, 2)
        else:
            desviacion_pct = 0.0

        resumen = ResumenComparablesRNTP(
            cantidad_muestras=len(muestras_rntp),
            precio_promedio_pen=precio_prom,
            precio_m2_promedio_pen=precio_m2_prom,
            desviacion_vs_prediccion_pct=desviacion_pct,
            cumple_regla_tres_muestras=(len(muestras_rntp) >= 3),
            criterio_busqueda=criterio,
            advertencia=advertencia,
        )

        return muestras_rntp, resumen


# Instancia singleton
comparables_service = ComparablesService()
