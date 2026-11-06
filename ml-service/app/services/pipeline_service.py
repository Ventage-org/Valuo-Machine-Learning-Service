"""
app/services/pipeline_service.py
──────────────────────────────────
Servicio de auditoría y consulta del pipeline de ingesta/procesamiento ETL
almacenado en la base de datos PostgreSQL (db_dataset_valuo).

Permite al administrador supervisar en tiempo real:
  - Últimas corridas y estado (COMPLETADO, FALLIDO, etc.)
  - Volumen de datos procesados (leídos vs. guardados vs. filtrados)
  - Duración y excepciones capturadas
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status
from app.schemas.monitoreo import PipelineEjecucionItem, PipelineEjecucionesResponse

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


def obtener_historial_pipeline(limit: int = 50) -> PipelineEjecucionesResponse:
    """
    Consulta la tabla public.pipeline_ejecuciones en PostgreSQL
    y retorna un resumen estructurado para la vista administrativa.
    """
    if not PSYCOPG2_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="psycopg2 no está instalado en el entorno.",
        )

    # Variables de conexión a PostgreSQL
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_DATASET_NAME") or os.getenv("DB_NAME", "db_dataset_valuo")
    db_user = os.getenv("DB_USER", "postgres")
    db_pass = os.getenv("DB_PASSWORD", "5432")

    query = """
    SELECT
        id,
        tipo_ejecucion,
        ejecutado_por,
        iniciado_en,
        finalizado_en,
        duracion_segundos,
        registros_leidos,
        registros_guardados,
        registros_filtrados,
        errores,
        metricas_salida,
        observaciones,
        estado
    FROM public.pipeline_ejecuciones
    ORDER BY iniciado_en DESC NULLS LAST
    LIMIT %s;
    """

    conn = None
    items: list[PipelineEjecucionItem] = []
    exitosos = 0

    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_pass,
            connect_timeout=5,
        )
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(query, (max(1, min(limit, 200)),))
            rows = cur.fetchall()

            for r in rows:
                r_leidos = r["registros_leidos"] or 0
                r_guardados = r["registros_guardados"] or 0
                r_filtrados = r["registros_filtrados"] or 0

                # Calcular tasa de retención/efectividad del filtrado de datos
                tasa = None
                if r_leidos > 0:
                    tasa = round((r_guardados / r_leidos) * 100.0, 2)

                estado_norm = str(r["estado"]).strip().upper() if r["estado"] else "DESCONOCIDO"
                if estado_norm in ("EXITOSO", "COMPLETADO", "SUCCESS", "OK"):
                    exitosos += 1

                # Deserializar JSON si viene como texto
                errores_val = r["errores"]
                if isinstance(errores_val, str):
                    try:
                        errores_val = json.loads(errores_val)
                    except Exception:
                        pass

                metricas_val = r["metricas_salida"]
                if isinstance(metricas_val, str):
                    try:
                        metricas_val = json.loads(metricas_val)
                    except Exception:
                        pass

                items.append(
                    PipelineEjecucionItem(
                        id=r["id"],
                        tipo_ejecucion=r["tipo_ejecucion"],
                        ejecutado_por=r["ejecutado_por"],
                        iniciado_en=r["iniciado_en"],
                        finalizado_en=r["finalizado_en"],
                        duracion_segundos=float(r["duracion_segundos"]) if r["duracion_segundos"] is not None else None,
                        registros_leidos=r_leidos,
                        registros_guardados=r_guardados,
                        registros_filtrados=r_filtrados,
                        tasa_efectividad_pct=tasa,
                        estado=estado_norm,
                        observaciones=r["observaciones"],
                        errores=errores_val,
                        metricas_salida=metricas_val,
                    )
                )

    except psycopg2.errors.UndefinedTable:
        # Si la tabla aún no existe en este ambiente, retornar respuesta vacía sin quebrar la app
        print(f"[PipelineService] Advertencia: La tabla public.pipeline_ejecuciones no existe en {db_name}.")
        return PipelineEjecucionesResponse(
            total_ejecuciones=0,
            ultima_ejecucion=None,
            tasa_exito_pct=None,
            ejecuciones=[],
            consultado_en=datetime.now(timezone.utc),
        )
    except Exception as exc:
        print(f"[PipelineService] Error consultando pipeline_ejecuciones: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "db_query_failed",
                "message": f"No se pudo consultar el historial del pipeline en PostgreSQL: {exc}",
            },
        )
    finally:
        if conn is not None:
            conn.close()

    total = len(items)
    tasa_exito = round((exitosos / total) * 100.0, 2) if total > 0 else None
    ultima = items[0].iniciado_en if items else None

    return PipelineEjecucionesResponse(
        total_ejecuciones=total,
        ultima_ejecucion=ultima,
        tasa_exito_pct=tasa_exito,
        ejecuciones=items,
        consultado_en=datetime.now(timezone.utc),
    )
