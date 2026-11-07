"""
app/services/model_registry_service.py
──────────────────────────────────────
Servicio de gobernanza y benchmarking para modelos en MLflow Model Registry.

Funcionalidades:
  1. Comparativa de Benchmark: Modelo en Producción vs Modelos Candidatos
     - Cálculo de diferencias: ΔMAPE (%), ΔR², ΔMAE, ΔRMSE
     - Dictamen automático según estándares de la tesis (MAPE < 15% venta / 17.89% alquiler)
  2. Historial de versiones del Model Registry
  3. Promoción controlada a 'Production' con Hot-Reload seguro en memoria RAM
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

import mlflow
import mlflow.xgboost
from mlflow.entities.model_registry import ModelVersion
import shap
from fastapi import HTTPException, status

from app.core.mlflow_client import mlflow_manager
from app.schemas.modelos import (
    BenchmarkResponse,
    ComparativaCandidato,
    DeltaMetricas,
    DescartarModeloResponse,
    HistorialModelosResponse,
    MetricasModelo,
    PromoverModeloResponse,
    VersionModeloInfo,
)

if TYPE_CHECKING:
    from app.core.model_loader import ModelState


# Benchmarks metodológicos oficiales de la tesis
BENCHMARKS_MAPE = {
    "venta": 15.0,     # Oporto et al. (2024) / Metodología Venta
    "alquiler": 17.89, # Oporto et al. (2024) / Metodología Alquiler
}


def _get_model_name(tipo_operacion: Literal["venta", "alquiler"]) -> str:
    """Obtiene el nombre oficial registrado en MLflow según el tipo de operación."""
    return os.getenv(
        f"MLFLOW_MODEL_NAME_{tipo_operacion.upper()}",
        f"xgboost_{tipo_operacion}",
    )


def _build_version_info(
    model_name: str,
    version: str,
    stage: str,
    run_id: str | None = None,
    tags: dict[str, str] | None = None,
    created_at_timestamp: int | float | None = None,
    aliases: list[str] | None = None,
) -> VersionModeloInfo:
    """Construye un VersionModeloInfo consultando las métricas del run en MLflow."""
    metricas = MetricasModelo()
    tags_dict = tags or {}
    aliases_list = list(aliases) if aliases else []

    if run_id and mlflow_manager.check_connection():
        try:
            run_data = mlflow_manager.get_run_metrics_and_params(run_id)
            m = run_data.get("metrics", {})
            metricas = MetricasModelo(
                mape_pct=m.get("mape_pct") or m.get("mape"),
                r2=m.get("r2"),
                mae=m.get("mae"),
                rmse=m.get("rmse"),
                n_train=int(m.get("n_train")) if "n_train" in m else None,
                n_test=int(m.get("n_test")) if "n_test" in m else None,
            )
            # Combinar tags del run con los de la versión
            run_tags = run_data.get("tags", {})
            tags_dict = {**run_tags, **tags_dict}
        except Exception as e:
            print(f"[ModelRegistryService] Advertencia al leer métricas del run {run_id}: {e}")

    creado_en = None
    if created_at_timestamp:
        try:
            creado_en = datetime.fromtimestamp(created_at_timestamp / 1000.0, tz=timezone.utc)
        except Exception:
            pass

    # Normalización exhaustiva de Stage
    stage_str = str(stage).strip() if stage is not None else ""
    if "champion" in aliases_list or stage_str.lower() in ("production", "prod") or tags_dict.get("estado_gobernanza") == "production":
        stage_final = "Production"
    elif stage_str.lower() in ("archived", "archivado") or tags_dict.get("estado_gobernanza") in ("archived", "discarded"):
        stage_final = "Archived"
    else:
        stage_final = "Candidate"

    return VersionModeloInfo(
        nombre_modelo=model_name,
        version=str(version),
        stage=stage_final,
        run_id=run_id,
        metricas=metricas,
        tags=tags_dict,
        aliases=aliases_list,
        creado_en=creado_en,
        fuente_artefacto=f"models:/{model_name}/{version}",
    )


def obtener_benchmark(
    tipo_operacion: Literal["venta", "alquiler"],
    state: "ModelState",
    version_candidato: str | None = None,
    solo_ultimo: bool = True,
) -> BenchmarkResponse:
    """
    Construye la comparativa de benchmark entre la versión de producción
    activa en RAM y la versión candidata (o candidatas activas) registradas en MLflow.
    """
    model_name = _get_model_name(tipo_operacion)
    benchmark_mape = BENCHMARKS_MAPE.get(tipo_operacion, 15.0)

    # 1. Obtener información de la versión actualmente en Producción
    prod_info: VersionModeloInfo | None = None

    if mlflow_manager.check_connection():
        prod_ver = mlflow_manager.get_production_version(model_name)
        if prod_ver:
            prod_info = _build_version_info(
                model_name=model_name,
                version=prod_ver.version,
                stage="Production",
                run_id=prod_ver.run_id,
                tags=getattr(prod_ver, "tags", {}),
                created_at_timestamp=getattr(prod_ver, "creation_timestamp", None),
                aliases=getattr(prod_ver, "aliases", []),
            )

    # Si MLflow no tiene aún una versión en Production, usar la configuración activa en memoria
    if prod_info is None or prod_info.metricas.mape_pct is None:
        cfg = state.config if tipo_operacion == "venta" else state.config_alquiler
        prod_info = VersionModeloInfo(
            nombre_modelo=model_name,
            version=cfg.get("modelo_version", "v2"),
            stage="Production",
            run_id=None,
            metricas=MetricasModelo(
                mape_pct=cfg.get("mape_test"),
                r2=cfg.get("r2_test"),
                mae=cfg.get("mae_test"),
                rmse=cfg.get("rmse_test"),
            ),
            tags={"fuente": "config_local"},
            fuente_artefacto="memoria_activa",
        )

    # 2. Obtener versión(es) candidata(s) activas (excluye Production y Archived)
    candidatos_evaluados: list[ComparativaCandidato] = []

    if mlflow_manager.check_connection():
        candidates: list[ModelVersion] = []
        if version_candidato:
            specific_ver = mlflow_manager.get_model_version(model_name, version_candidato)
            if specific_ver:
                candidates = [specific_ver]
        else:
            limit = 1 if solo_ultimo else 5
            candidates = mlflow_manager.get_latest_candidate_versions(model_name, limit=limit)

        for cand_ver in candidates:
            cand_info = _build_version_info(
                model_name=model_name,
                version=cand_ver.version,
                stage=cand_ver.current_stage or "Candidate",
                run_id=cand_ver.run_id,
                tags=getattr(cand_ver, "tags", {}),
                created_at_timestamp=getattr(cand_ver, "creation_timestamp", None),
                aliases=getattr(cand_ver, "aliases", []),
            )

            # Calcular deltas frente a Producción
            delta_mape = None
            delta_r2 = None
            delta_mae = None
            delta_rmse = None
            mejora_mape = False
            mejora_r2 = False

            cand_m = cand_info.metricas
            prod_m = prod_info.metricas

            if cand_m.mape_pct is not None and prod_m.mape_pct is not None:
                delta_mape = round(cand_m.mape_pct - prod_m.mape_pct, 4)
                mejora_mape = delta_mape < 0

            if cand_m.r2 is not None and prod_m.r2 is not None:
                delta_r2 = round(cand_m.r2 - prod_m.r2, 4)
                mejora_r2 = delta_r2 > 0

            if cand_m.mae is not None and prod_m.mae is not None:
                delta_mae = round(cand_m.mae - prod_m.mae, 2)

            if cand_m.rmse is not None and prod_m.rmse is not None:
                delta_rmse = round(cand_m.rmse - prod_m.rmse, 2)

            # Dictamen automático
            if cand_m.mape_pct is not None and cand_m.mape_pct <= benchmark_mape and (mejora_mape or mejora_r2):
                recomendacion = "RECOMENDADO_PARA_PRODUCCION"
                motivo = (
                    f"El candidato mejora el desempeño frente a producción "
                    f"(ΔMAPE: {delta_mape:+.2f}%, ΔR²: {delta_r2:+.4f}) y supera el benchmark exigido ({benchmark_mape}%)."
                )
            elif cand_m.mape_pct is not None and cand_m.mape_pct > benchmark_mape:
                recomendacion = "NO_RECOMENDADO"
                motivo = (
                    f"El modelo candidato tiene un MAPE de {cand_m.mape_pct:.2f}%, "
                    f"el cual no alcanza el umbral de aprobación ({benchmark_mape}%)."
                )
            else:
                recomendacion = "REVISION_MANUAL"
                motivo = "Desempeño similar o métricas incompletas. Se recomienda inspección visual en panel de control."

            candidatos_evaluados.append(
                ComparativaCandidato(
                    candidato=cand_info,
                    deltas=DeltaMetricas(
                        delta_mape_pct=delta_mape,
                        delta_r2=delta_r2,
                        delta_mae=delta_mae,
                        delta_rmse=delta_rmse,
                        mejora_mape=mejora_mape,
                        mejora_r2=mejora_r2,
                    ),
                    recomendacion=recomendacion,
                    motivo=motivo,
                )
            )

    return BenchmarkResponse(
        tipo_operacion=tipo_operacion,
        benchmark_referencial_mape=benchmark_mape,
        modelo_produccion=prod_info,
        candidatos=candidatos_evaluados,
        total_candidatos=len(candidatos_evaluados),
    )


def obtener_historial_versiones(
    tipo_operacion: Literal["venta", "alquiler"],
) -> HistorialModelosResponse:
    """Lista todas las versiones registradas para el modelo en MLflow."""
    model_name = _get_model_name(tipo_operacion)

    if not mlflow_manager.check_connection():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "mlflow_no_disponible",
                "message": "El servidor MLflow no está accesible para consultar el historial.",
            },
        )

    filter_str = f"name = '{model_name}'"
    all_versions = mlflow_manager.client.search_model_versions(
        filter_string=filter_str,
        order_by=["version_number DESC"],
        max_results=50,
    )

    versiones_info = [
        _build_version_info(
            model_name=model_name,
            version=v.version,
            stage=v.current_stage,
            run_id=v.run_id,
            tags=getattr(v, "tags", {}),
            created_at_timestamp=getattr(v, "creation_timestamp", None),
            aliases=getattr(v, "aliases", []),
        )
        for v in all_versions
    ]

    return HistorialModelosResponse(
        tipo_operacion=tipo_operacion,
        nombre_modelo=model_name,
        total_versiones=len(versiones_info),
        versiones=versiones_info,
    )


def promover_modelo_a_produccion(
    tipo_operacion: Literal["venta", "alquiler"],
    version: str | int | None,
    motivo: str,
    state: "ModelState",
) -> PromoverModeloResponse:
    """
    1. Promueve la versión solicitada (o el último candidato activo si version es None)
       en MLflow Model Registry a stage 'Production'.
    2. Descarga el modelo directamente desde MLflow a la memoria RAM.
    3. Reconstruye el SHAP TreeExplainer en RAM.
    4. Actualiza atómicamente el ModelState global (Hot-Reload sin downtime).
    """
    model_name = _get_model_name(tipo_operacion)

    if not mlflow_manager.check_connection():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "mlflow_no_disponible",
                "message": "El servidor MLflow no está accesible para procesar la promoción.",
            },
        )

    # Sanitizar versión: extraer dígitos si vienen letras o prefijos
    version_digits = None
    if version is not None:
        v_raw = str(version).strip()
        m = re.search(r"\d+", v_raw)
        if m:
            version_digits = m.group(0)

    # Si no vino número de versión o vino un placeholder sin dígitos (ej: 'string', null)
    if not version_digits:
        candidates = mlflow_manager.get_latest_candidate_versions(model_name, limit=1)
        if not candidates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "sin_candidatos",
                    "message": f"No se encontró ninguna versión candidata activa pendiente de aprobación para '{model_name}'.",
                },
            )
        version_str = str(candidates[0].version)
    else:
        version_str = version_digits

    # Identificar versión anterior en producción para archivar y calcular deltas
    prod_anterior = mlflow_manager.get_production_version(model_name)
    ver_anterior_str = prod_anterior.version if prod_anterior else None
    info_anterior = _build_version_info(
        model_name=model_name,
        version=prod_anterior.version,
        stage="Production",
        run_id=prod_anterior.run_id,
    ) if prod_anterior else None

    # 1. Promover en MLflow
    try:
        mlflow_manager.promote_version_to_production(
            model_name=model_name,
            version=version_str,
            archive_existing=True,
            motivo=motivo,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "error_promocion_mlflow",
                "message": f"No se pudo promover la versión {version_str} en MLflow: {e}",
            },
        )

    # 2. Descargar y Cargar en RAM
    target_run = ver_nueva.run_id if 'ver_nueva' in locals() and ver_nueva else None
    if not target_run:
        try:
            v_obj = mlflow_manager.get_model_version(model_name, version_str)
            target_run = v_obj.run_id if v_obj else None
        except Exception:
            pass

    nuevo_modelo = None
    # Prioridad 1: Descarga ultrarrápida vía HTTP proxy de MLflow (sin depender de Azure Credentials en el cliente)
    if target_run:
        from app.core.model_loader import _descargar_modelo_mlflow_http
        print(f"[ModelRegistryService] Descargando modelo v{version_str} vía proxy HTTP de MLflow (run_id: {target_run})...")
        nuevo_modelo = _descargar_modelo_mlflow_http(mlflow_manager.tracking_uri, target_run)

    # Prioridad 2: Carga nativa mlflow.load_model si falla el proxy
    if nuevo_modelo is None:
        try:
            model_uri = f"models:/{model_name}/{version_str}"
            print(f"[ModelRegistryService] Intentando carga nativa desde {model_uri}...")
            nuevo_modelo = mlflow.xgboost.load_model(model_uri)
        except Exception as e_load:
            print(f"[ModelRegistryService] Fallo carga nativa: {e_load}")

    if nuevo_modelo is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "error_carga_ram",
                "message": f"El modelo se promovió en MLflow pero falló al cargarse en RAM.",
            },
        )

    try:
        print("[ModelRegistryService] Construyendo SHAP TreeExplainer en RAM...")
        nuevo_explainer = shap.TreeExplainer(nuevo_modelo)

        # 3. Hot-reload atómico en memoria
        import json
        from app.core.model_loader import _resolver_ruta
        config_path_env = os.getenv("CONFIG_PATH" if tipo_operacion == "venta" else "CONFIG_ALQUILER_PATH")
        config_path = _resolver_ruta(config_path_env or f"config/model_{'alquiler_' if tipo_operacion == 'alquiler' else ''}config.json")

        if tipo_operacion == "venta":
            state.modelo = nuevo_modelo
            state.explainer = nuevo_explainer
            if hasattr(nuevo_modelo, "feature_names_in_"):
                state.feature_cols = list(nuevo_modelo.feature_names_in_)
            state.config["modelo_version"] = f"v{version_str}_mlflow"
            state.config["promoted_at"] = datetime.now(timezone.utc).isoformat()
            cfg_to_persist = dict(state.config)
        else:
            state.modelo_alquiler = nuevo_modelo
            state.explainer_alquiler = nuevo_explainer
            if hasattr(nuevo_modelo, "feature_names_in_"):
                state.feature_cols_alquiler = list(nuevo_modelo.feature_names_in_)
            state.config_alquiler["modelo_version"] = f"v{version_str}_mlflow"
            state.config_alquiler["promoted_at"] = datetime.now(timezone.utc).isoformat()
            cfg_to_persist = dict(state.config_alquiler)

        try:
            if os.path.exists(config_path):
                with open(config_path, "w", encoding="utf-8") as fp:
                    json.dump(cfg_to_persist, fp, indent=2, ensure_ascii=False)
        except Exception as e_cfg:
            print(f"[ModelRegistryService] Advertencia al persistir config: {e_cfg}")

        # Sincronizar gobernanza en jobs de entrenamiento
        try:
            from app.services.entrenamiento_service import actualizar_estado_gobernanza_job
            actualizar_estado_gobernanza_job(
                tipo_operacion=tipo_operacion,
                version_promovida=version_str,
                version_anterior=ver_anterior_str,
            )
        except Exception as e_sync:
            print(f"[ModelRegistryService] Advertencia al sincronizar estado_gobernanza_job: {e_sync}")

        print(f"[ModelRegistryService] [OK] Hot-Reload completado en RAM para {tipo_operacion} (versión {version_str}).")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "error_carga_ram",
                "message": f"El modelo se promovió en MLflow pero falló al cargarse en RAM: {e}",
            },
        )

    # 4. Obtener métricas de la nueva versión y calcular deltas frente a la anterior
    ver_nueva = mlflow_manager.get_model_version(model_name, version_str)
    info_nueva = _build_version_info(
        model_name=model_name,
        version=version_str,
        stage="Production",
        run_id=ver_nueva.run_id if ver_nueva else None,
        aliases=["champion"],
    ) if ver_nueva else None

    deltas = None
    if info_nueva and info_anterior and ver_anterior_str != version_str:
        m_new = info_nueva.metricas
        m_old = info_anterior.metricas

        delta_mape = round(m_new.mape_pct - m_old.mape_pct, 4) if (m_new.mape_pct is not None and m_old.mape_pct is not None) else None
        delta_r2 = round(m_new.r2 - m_old.r2, 4) if (m_new.r2 is not None and m_old.r2 is not None) else None
        delta_mae = round(m_new.mae - m_old.mae, 2) if (m_new.mae is not None and m_old.mae is not None) else None
        delta_rmse = round(m_new.rmse - m_old.rmse, 2) if (m_new.rmse is not None and m_old.rmse is not None) else None

        deltas = DeltaMetricas(
            delta_mape_pct=delta_mape,
            delta_r2=delta_r2,
            delta_mae=delta_mae,
            delta_rmse=delta_rmse,
            mejora_mape=delta_mape is not None and delta_mape < 0,
            mejora_r2=delta_r2 is not None and delta_r2 > 0,
        )

    return PromoverModeloResponse(
        exito=True,
        mensaje=f"Modelo {model_name} v{version_str} promovido exitosamente a Producción y cargado en memoria RAM.",
        tipo_operacion=tipo_operacion,
        version_promovida=version_str,
        version_anterior_archivada=ver_anterior_str,
        modelo_cargado_en_ram=True,
        metricas=info_nueva.metricas if info_nueva else None,
        metricas_anteriores=info_anterior.metricas if info_anterior else None,
        deltas=deltas,
    )


def ejecutar_rollback(
    tipo_operacion: Literal["venta", "alquiler"],
    version: str | int | None,
    motivo: str | None,
    state: "ModelState",
) -> PromoverModeloResponse:
    """
    Restaura una versión archivada previa a Producción.
    Si 'version' no se especifica o no contiene dígitos (ej. el placeholder 'string' de Swagger),
    revierte automáticamente a la versión anterior inmediata en estado Archived.
    """
    model_name = _get_model_name(tipo_operacion)
    target_version = None

    if version is not None:
        v_str = str(version).strip()
        m = re.search(r"\d+", v_str)
        if m:
            target_version = m.group(0)

    # Si no se pasó un número de versión válido, buscar la versión archivada más reciente
    if not target_version:
        archived_ver = mlflow_manager.get_latest_archived_version(model_name)
        if not archived_ver:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "sin_version_archivada",
                    "message": f"No se encontró ninguna versión archivada previa de '{model_name}' a la cual revertir.",
                },
            )
        target_version = str(archived_ver.version)

    motivo_final = motivo or f"Rollback preventivo a versión {target_version}."
    return promover_modelo_a_produccion(
        tipo_operacion=tipo_operacion,
        version=target_version,
        motivo=motivo_final,
        state=state,
    )


def descartar_modelo_candidato(
    tipo_operacion: Literal["venta", "alquiler"],
    version: str | int | None,
    motivo: str,
) -> DescartarModeloResponse:
    """
    Descarta y archiva un modelo candidato en MLflow Model Registry:
    1. Verifica que la versión objetivo exista y no sea la versión activa en Producción.
    2. Transiciona su stage a 'Archived' en MLflow y estampa tags de auditoría.
    3. Sincroniza el estado en el registro en caliente de jobs si aplica.
    """
    model_name = _get_model_name(tipo_operacion)

    if not mlflow_manager.check_connection():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "mlflow_no_disponible",
                "message": "El servidor MLflow no está accesible para procesar el descarte.",
            },
        )

    # Sanitizar versión
    target_version: str | None = None
    if version is not None:
        v_raw = str(version).strip()
        m = re.search(r"\d+", v_raw)
        if m:
            target_version = m.group(0)

    # Si no se pasó versión específica, buscar el candidato activo más reciente
    if not target_version:
        candidates = mlflow_manager.get_latest_candidate_versions(model_name, limit=1)
        if not candidates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "sin_candidatos",
                    "message": f"No se encontró ninguna versión candidata activa para descartar en '{model_name}'.",
                },
            )
        target_version = str(candidates[0].version)

    # Verificar existencia de la versión en MLflow
    v_obj = mlflow_manager.get_model_version(model_name, target_version)
    if not v_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "version_no_encontrada",
                "message": f"La versión {target_version} de '{model_name}' no existe en el Model Registry.",
            },
        )

    # Proteger contra el descarte accidental del modelo en Producción
    is_prod = (
        v_obj.current_stage == "Production"
        or "champion" in getattr(v_obj, "aliases", [])
        or getattr(v_obj, "tags", {}).get("estado_gobernanza") == "production"
    )
    if is_prod:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "modelo_en_produccion",
                "message": (
                    f"La versión {target_version} de '{model_name}' está actualmente en PRODUCCIÓN. "
                    "No se puede descartar directamente. Promueva otra versión o ejecute un rollback primero."
                ),
            },
        )

    # 1. Transicionar y archivar en MLflow
    try:
        mlflow_manager.discard_candidate_version(
            model_name=model_name,
            version=target_version,
            motivo=motivo,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "error_descarte_mlflow",
                "message": f"No se pudo archivar la versión {target_version} en MLflow: {e}",
            },
        )

    # 2. Sincronizar en el registro de jobs
    try:
        from app.services.entrenamiento_service import descartar_estado_gobernanza_job
        descartar_estado_gobernanza_job(
            tipo_operacion=tipo_operacion,
            version_descartada=target_version,
            motivo=motivo,
        )
    except Exception as e_sync:
        print(f"[ModelRegistryService] Advertencia al sincronizar descarte en jobs: {e_sync}")

    # 3. Obtener métricas históricas de la versión descartada para trazabilidad
    info_descartada = _build_version_info(
        model_name=model_name,
        version=target_version,
        stage="Archived",
        run_id=v_obj.run_id,
        tags=getattr(v_obj, "tags", {}),
    )

    print(f"[ModelRegistryService] [OK] Modelo {model_name} v{target_version} descartado y archivado.")

    return DescartarModeloResponse(
        exito=True,
        mensaje=f"Modelo candidato {model_name} v{target_version} descartado y archivado exitosamente.",
        tipo_operacion=tipo_operacion,
        version_descartada=target_version,
        nuevo_stage="Archived",
        motivo=motivo,
        metricas=info_descartada.metricas if info_descartada else None,
    )
