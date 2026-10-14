"""
app/core/data_provider.py
─────────────────────────
Proveedor de contexto distrital para el servicio de inferencia.

Al startup del servicio, PostgreSQLContextProvider conecta a inmobiliaria_ml_db
y carga en memoria el contexto distrital más reciente (último año disponible
por distrito) desde la tabla distrito_anio_contexto.

Esto evita una consulta a BD en cada predicción individual:
  - El contexto distrital se precarga una sola vez al arrancar
  - Las predicciones leen del dict en memoria (<< 1ms)
  - El contexto se refresca automáticamente en cada reinicio o reentrenamiento

Tabla consumida: distrito_anio_contexto (JOIN con distritos)
Tablas NO tocadas: dataset_inmuebles_venta, auditoria_flags_imputacion, pipeline_ejecuciones
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import psycopg2
import pandas as pd


# ─── Interfaz abstracta ───────────────────────────────────────────────────────

class IContextDataProvider(ABC):
    """
    Interfaz para resolver el contexto distrital en tiempo de inferencia.

    El contexto distrital incluye:
      NSE (A-E), tasas de crimen, demografía y distancias a POIs.
    Estos datos se resuelven por nombre de distrito y corresponden
    al año más reciente disponible en la base de datos.
    """

    @abstractmethod
    def get_distrito_context(self, nombre_distrito: str) -> dict:
        """
        Retorna el contexto distrital (dict) para usar en la predicción.

        Args:
            nombre_distrito: Nombre exacto del distrito (ej: 'San Miguel').

        Returns:
            Dict con claves: pct_NSE_A..E, tasa_robo, tasa_hurto,
            poblacion_proyectada, area_distrito_km2, distancia_centro_km,
            dist_colegio_km, dist_hospital_km, dist_estacion_transporte_km,
            dist_centro_comercial_km, dist_parque_km, dist_universidad_km,
            densidad_hab_km2.

        Raises:
            KeyError: Si el distrito no existe en el contexto cargado.
        """
        ...

    @abstractmethod
    def listar_distritos(self) -> list[str]:
        """Retorna la lista de distritos disponibles en el contexto."""
        ...


# ─── Implementación PostgreSQL ────────────────────────────────────────────────

_QUERY_CONTEXTO = """
SELECT
    d.nombre                     AS distrito,
    c.pct_nse_a                  AS "pct_NSE_A",
    c.pct_nse_b                  AS "pct_NSE_B",
    c.pct_nse_c                  AS "pct_NSE_C",
    c.pct_nse_d                  AS "pct_NSE_D",
    c.pct_nse_e                  AS "pct_NSE_E",
    c.tasa_robo,
    c.tasa_hurto,
    c.poblacion_proyectada,
    c.densidad_hab_km2,
    c.distancia_centro_km,
    c.dist_colegio_km,
    c.dist_hospital_km,
    c.dist_estacion_transporte_km,
    c.dist_centro_comercial_km,
    c.dist_parque_km,
    c.dist_universidad_km,
    c.area_distrito_km2
FROM distrito_anio_contexto c
JOIN distritos d ON c.distrito_id = d.id
WHERE (c.distrito_id, c.anio) IN (
    SELECT distrito_id, MAX(anio)
    FROM distrito_anio_contexto
    GROUP BY distrito_id
)
ORDER BY d.nombre;
"""


class PostgreSQLContextProvider(IContextDataProvider):
    """
    Carga el contexto distrital desde PostgreSQL al iniciar el servicio.

    Precarga en memoria el contexto más reciente de cada distrito
    (último año disponible en distrito_anio_contexto).
    Las predicciones leen desde el dict en RAM — sin latencia de BD.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db_name: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self._conn_params = {
            "host":     host     or os.getenv("DB_HOST", "localhost"),
            "port":     port     or int(os.getenv("DB_PORT", "5432")),
            "dbname":   db_name  or os.getenv("DB_NAME", "inmobiliaria_ml_db"),
            "user":     user     or os.getenv("DB_USER", "postgres"),
            "password": password or os.getenv("DB_PASSWORD") or "5432",
        }
        self._contexto: dict[str, dict] = {}
        self._cargar()

    def _cargar(self) -> None:
        """Ejecuta la consulta y carga el contexto en memoria."""
        conn = psycopg2.connect(**self._conn_params)
        try:
            cur = conn.cursor()
            cur.execute(_QUERY_CONTEXTO)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            raise RuntimeError(
                "La tabla 'distrito_anio_contexto' no retornó datos. "
                "Verifica que la BD tenga datos de contexto cargados."
            )

        for row in rows:
            row_dict = dict(zip(cols, row))
            nombre = row_dict.pop("distrito")
            self._contexto[nombre] = row_dict

        print(
            f"[PostgreSQLContextProvider] Cargado: {len(self._contexto)} distritos "
            f"desde '{self._conn_params['dbname']}'."
        )

    def get_distrito_context(self, nombre_distrito: str) -> dict:
        if nombre_distrito not in self._contexto:
            raise KeyError(
                f"Distrito '{nombre_distrito}' no encontrado en el contexto distrital. "
                f"Disponibles: {sorted(self._contexto.keys())}"
            )
        return self._contexto[nombre_distrito]

    def listar_distritos(self) -> list[str]:
        return sorted(self._contexto.keys())
