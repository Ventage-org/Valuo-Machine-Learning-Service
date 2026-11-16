"""
app/services/mercado_service.py
───────────────────────────────
Servicio analítico de tendencias de mercado inmobiliario por distrito.
Calcula precios medianos por m² y variaciones trimestrales a partir del dataset histórico
almacenado en PostgreSQL, proyectado a valor nominal presente con el IPC vigente.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Literal

from fastapi import HTTPException, status
import psycopg2

from app.schemas.mercado import TendenciaDistrito, TendenciasMercadoResponse

if TYPE_CHECKING:
    from app.core.model_loader import ModelState

# Cache en memoria: (tipo_operacion, tipo_cambio) -> (timestamp, TendenciasMercadoResponse)
_CACHE_TENDENCIAS: dict[tuple[str, float], tuple[float, TendenciasMercadoResponse]] = {}
_CACHE_TTL_SECS = 3600  # 1 hora


def _get_db_connection() -> Any:
    host = os.getenv("DB_HOST", "127.0.0.1")
    if host == "localhost":
        host = "127.0.0.1"
    return psycopg2.connect(
        host=host,
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "inmobiliaria_ml_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "5432"),
        sslmode=os.getenv("DB_SSLMODE", "prefer"),
        connect_timeout=5,
    )


def limpiar_cache_tendencias() -> None:
    """Invalida el cache en memoria."""
    _CACHE_TENDENCIAS.clear()


def obtener_tendencias_distritales(
    state: "ModelState",
    tipo_operacion: Literal["venta", "alquiler"] = "venta",
    tipo_cambio: float = 3.75,
) -> TendenciasMercadoResponse:
    """Calcula precios medianos m² y variaciones porcentuales trimestrales por distrito."""
    cache_key = (tipo_operacion, round(tipo_cambio, 4))
    ahora = time.time()
    if cache_key in _CACHE_TENDENCIAS:
        ts, cached_resp = _CACHE_TENDENCIAS[cache_key]
        if ahora - ts < _CACHE_TTL_SECS:
            return cached_resp

    tabla = "dataset_inmuebles_venta" if tipo_operacion == "venta" else "dataset_inmuebles_alquiler"
    col_precio = "precio_soles_const" if tipo_operacion == "venta" else "alquiler_soles_const"

    config = state.config if tipo_operacion == "venta" else state.config_alquiler
    ipc_cfg = config.get("ipc_actual", {"valor": 169.18, "periodo": "2026-Q1"})
    ipc_val = float(ipc_cfg.get("valor", 169.18))
    ipc_factor = ipc_val / 100.0

    try:
        conn = _get_db_connection()
        cur = conn.cursor()

        # 1. Obtener los dos trimestres más recientes en la BD
        cur.execute(f"""
            SELECT DISTINCT anio, trimestre
            FROM {tabla}
            ORDER BY anio DESC, trimestre DESC
            LIMIT 2;
        """)
        periodos = cur.fetchall()
        if len(periodos) < 2:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No se encontraron al menos dos trimestres en el repositorio de datos.",
            )

        (a_act, t_act), (a_ant, t_ant) = periodos[0], periodos[1]

        # 2. Consultar métricas agregadas por distrito para ambos trimestres
        sql = f"""
            SELECT
                d.nombre AS distrito,
                v.anio,
                v.trimestre,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (v.{col_precio} / v.superficie_m2)) AS med_m2_const,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY v.{col_precio}) AS med_precio_const,
                AVG(v.superficie_m2) AS sup_prom,
                COUNT(*) AS total_muestras
            FROM {tabla} v
            JOIN distritos d ON v.distrito_id = d.id
            WHERE (v.anio = %s AND v.trimestre = %s)
               OR (v.anio = %s AND v.trimestre = %s)
            GROUP BY d.nombre, v.anio, v.trimestre
            ORDER BY d.nombre;
        """
        cur.execute(sql, (a_act, t_act, a_ant, t_ant))
        rows = cur.fetchall()
        conn.close()

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error consultando tendencias distritales en la base de datos: {exc}",
        ) from exc

    # 3. Procesar datos por distrito
    datos_por_dist: dict[str, dict[str, Any]] = {}
    for distrito, anio, trim, med_m2_c, med_p_c, sup_p, cant in rows:
        if distrito not in datos_por_dist:
            datos_por_dist[distrito] = {}
        datos_por_dist[distrito][(anio, trim)] = {
            "med_m2_const": float(med_m2_c) if med_m2_c is not None else 0.0,
            "med_p_const": float(med_p_c) if med_p_c is not None else 0.0,
            "sup_prom": float(sup_p) if sup_p is not None else 0.0,
            "cant": int(cant),
        }

    tc = max(0.1, tipo_cambio)
    lista_distritos: list[TendenciaDistrito] = []

    for dist, per_dict in datos_por_dist.items():
        data_act = per_dict.get((a_act, t_act))
        data_ant = per_dict.get((a_ant, t_ant))

        if not data_act:
            continue

        p_m2_pen = round(data_act["med_m2_const"] * ipc_factor, 2)
        p_m2_usd = round(p_m2_pen / tc, 2)

        ticket_pen = round(data_act["med_p_const"] * ipc_factor, 2)
        ticket_usd = round(ticket_pen / tc, 2)

        if data_ant and data_ant["med_m2_const"] > 0:
            p_m2_ant_pen = data_ant["med_m2_const"] * ipc_factor
            var_pct = round(((p_m2_pen - p_m2_ant_pen) / p_m2_ant_pen) * 100.0, 2)
        else:
            var_pct = 0.0

        lista_distritos.append(
            TendenciaDistrito(
                distrito=dist,
                precio_m2_usd=p_m2_usd,
                precio_m2_pen=p_m2_pen,
                variacion_trimestral_pct=var_pct,
                ticket_promedio_usd=ticket_usd,
                ticket_promedio_pen=ticket_pen,
                superficie_promedio_m2=round(data_act["sup_prom"], 1),
                total_muestras=data_act["cant"],
            )
        )

    # Ordenar por precio m² USD descendente
    lista_distritos.sort(key=lambda d: d.precio_m2_usd, reverse=True)

    # Periodos reales extraídos de la base de datos
    periodo_str_act = f"{a_act}-Q{t_act}"
    periodo_str_ant = f"{a_ant}-Q{t_ant}"

    respuesta = TendenciasMercadoResponse(
        tipo_operacion=tipo_operacion,
        periodo_actual=periodo_str_act,
        periodo_anterior=periodo_str_ant,
        tipo_cambio=tc,
        ipc_factor=round(ipc_factor, 4),
        total_distritos=len(lista_distritos),
        distritos=lista_distritos,
        nota=(
            "Valores calculados con la mediana de precios por m² del dataset de Lima Metropolitana "
            "ajustados con el factor de inflación IPC oficial a precios nominales vigentes."
        ),
    )

    _CACHE_TENDENCIAS[cache_key] = (ahora, respuesta)
    return respuesta
