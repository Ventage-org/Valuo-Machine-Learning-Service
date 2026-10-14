"""
app/core/model_loader.py
────────────────────────
Singleton de carga de los modelos XGBoost (Venta y Alquiler) + TreeExplainer SHAP + metadata.

Se ejecuta al arrancar la app (lifespan FastAPI).
Los TreeExplainer tardan ~500ms en inicializarse — por eso nunca deben
crearse por request.

Fuente de datos: exclusivamente PostgreSQL (inmobiliaria_ml_db).
  - Contexto distrital → tabla distrito_anio_contexto (via PostgreSQLContextProvider)
  - Dataset de entrenamiento → tablas de venta y alquiler (via db_provider)

Thread-safety:
  - Los modelos XGBoost y los TreeExplainer son seguros para lectura concurrente.
  - No hay escritura después del startup → sin locks necesarios en inferencia.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import joblib
import shap
from xgboost import XGBRegressor

from app.core.data_provider import (
    IContextDataProvider,
    PostgreSQLContextProvider,
)


# ─── Estado global de los modelos ─────────────────────────────────────────────

@dataclass
class ModelState:
    """Contenedor de todos los artefactos de modelos cargados al startup."""
    # ── Modelo Venta
    modelo: XGBRegressor | None = None
    explainer: shap.TreeExplainer | None = None
    feature_cols: list[str] = field(default_factory=list)
    encoding_map: dict[str, float] = field(default_factory=dict)
    media_global: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    # ── Modelo Alquiler
    modelo_alquiler: XGBRegressor | None = None
    explainer_alquiler: shap.TreeExplainer | None = None
    feature_cols_alquiler: list[str] = field(default_factory=list)
    encoding_map_alquiler: dict[str, float] = field(default_factory=dict)
    media_global_alquiler: float = 0.0
    config_alquiler: dict[str, Any] = field(default_factory=dict)

    # ── Contexto distrital compartido (PostgreSQL)
    data_provider: IContextDataProvider | None = None


# Instancia singleton — importada por routers y services
model_state = ModelState()


# ─── Funciones de carga ───────────────────────────────────────────────────────

from pathlib import Path


def _resolver_ruta(ruta: str) -> str:
    """
    Resuelve la ruta comprobando:
    1. Si existe en la ruta dada (CWD actual)
    2. Relativa a ml-service
    3. Relativa a la raíz del repositorio
    4. En data/processed/ o data/ si busca metadata
    5. En models/ si busca archivos de modelo
    """
    if os.path.exists(ruta):
        return ruta

    try:
        archivo_actual = Path(__file__).resolve()
        ml_service_dir = archivo_actual.parents[2]  # <repo>/ml-service
        repo_root = archivo_actual.parents[3]       # <repo>
    except (IndexError, ValueError):
        ml_service_dir = Path.cwd()
        repo_root = ml_service_dir.parent

    nombre_archivo = os.path.basename(ruta)

    candidatos = [
        ml_service_dir / ruta,
        repo_root / ruta,
        repo_root / "data" / "processed" / nombre_archivo,
        ml_service_dir / "data" / "processed" / nombre_archivo,
        ml_service_dir / "data" / nombre_archivo,
        repo_root / "models" / nombre_archivo,
        ml_service_dir / "models" / nombre_archivo,
    ]

    for c in candidatos:
        if c.exists():
            return str(c.resolve())

    return ruta


def _cargar_artefactos_venta(state: ModelState, model_path_override: str | None = None) -> None:
    """Carga el modelo de Venta, SHAP explainer y su configuración."""
    metadata_path = _resolver_ruta(os.getenv("METADATA_PATH", "data/features_metadata.json"))
    config_path   = _resolver_ruta(os.getenv("CONFIG_PATH",   "config/model_config.json"))
    models_dir    = _resolver_ruta(os.getenv("MODELS_DIR",    "models"))

    # 1. Configuración de venta
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            state.config = json.load(f)

    # 2. Intentar cargar versión Production desde MLflow Model Registry
    from app.core.mlflow_client import mlflow_manager
    model_name_mlflow = os.getenv("MLFLOW_MODEL_NAME_VENTA", "xgboost_venta")
    modelo_cargado = False

    if not model_path_override and mlflow_manager.check_connection():
        prod_ver = mlflow_manager.get_production_version(model_name_mlflow)
        if prod_ver:
            try:
                uri = f"models:/{model_name_mlflow}/Production"
                print(f"[ModelLoader][Venta] Cargando modelo en RAM desde MLflow: {uri} (v{prod_ver.version})...")
                import mlflow.xgboost
                state.modelo = mlflow.xgboost.load_model(uri)
                modelo_cargado = True
                state.config["modelo_version"] = f"v{prod_ver.version}_mlflow"
                try:
                    run_info = mlflow_manager.client.get_run(prod_ver.run_id)
                    m = run_info.data.metrics
                    if "mape_pct" in m:
                        state.config["mape_test"] = round(m["mape_pct"], 4)
                    if "r2" in m:
                        state.config["r2_test"] = round(m["r2"], 4)
                    if "mae" in m:
                        state.config["mae_test"] = round(m["mae"], 2)
                    if "rmse" in m:
                        state.config["rmse_test"] = round(m["rmse"], 2)
                except Exception as ex_m:
                    print(f"[ModelLoader][Venta] No se pudieron sincronizar métricas de MLflow: {ex_m}")
                print(f"[ModelLoader][Venta] [OK] Modelo cargado desde MLflow Registry a RAM.")
            except Exception as e:
                print(f"[ModelLoader][Venta] Advertencia: No se pudo cargar desde MLflow ({e}). Usando fallback local...")

    if not modelo_cargado:
        config_model_file = state.config.get("modelo_archivo", "xgboost_venta_v2.pkl")
        model_path = _resolver_ruta(
            model_path_override
            or os.getenv("MODEL_PATH")
            or os.path.join(models_dir, config_model_file)
            or "models/xgboost_venta_v2.pkl"
        )

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Modelo de venta no encontrado: {model_path}\n"
                f"Verifica que el archivo exista en {models_dir}/"
            )
        print(f"[ModelLoader][Venta] Cargando modelo: {model_path} ...")
        state.modelo = joblib.load(model_path)
        print(f"[ModelLoader][Venta] [OK] Modelo cargado ({os.path.getsize(model_path) / 1e6:.1f} MB)")

    print("[ModelLoader][Venta] Inicializando SHAP TreeExplainer ...")
    state.explainer = shap.TreeExplainer(state.modelo)
    print("[ModelLoader][Venta] [OK] SHAP TreeExplainer listo")

    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Metadata de venta no encontrada: {metadata_path}")
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    state.feature_cols = (
        list(state.modelo.feature_names_in_)
        if hasattr(state.modelo, "feature_names_in_")
        else metadata["features"]
    )
    state.encoding_map = metadata["encoding_map_distrito"]
    state.media_global = metadata["media_global_target"]
    print(f"[ModelLoader][Venta] [OK] Metadata: {len(state.feature_cols)} features, {len(state.encoding_map)} distritos")


def _cargar_artefactos_alquiler(state: ModelState, model_path_override: str | None = None) -> None:
    """Carga el modelo de Alquiler, SHAP explainer y su configuración."""
    metadata_path = _resolver_ruta(os.getenv("METADATA_ALQUILER_PATH", "data/features_metadata_alquiler.json"))
    config_path   = _resolver_ruta(os.getenv("CONFIG_ALQUILER_PATH",   "config/model_alquiler_config.json"))
    models_dir    = _resolver_ruta(os.getenv("MODELS_DIR",             "models"))

    # 1. Configuración de alquiler
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            state.config_alquiler = json.load(f)

    # 2. Intentar cargar versión Production desde MLflow Model Registry
    from app.core.mlflow_client import mlflow_manager
    model_name_mlflow = os.getenv("MLFLOW_MODEL_NAME_ALQUILER", "xgboost_alquiler")
    modelo_cargado = False

    if not model_path_override and mlflow_manager.check_connection():
        prod_ver = mlflow_manager.get_production_version(model_name_mlflow)
        if prod_ver:
            try:
                uri = f"models:/{model_name_mlflow}/Production"
                print(f"[ModelLoader][Alquiler] Cargando modelo en RAM desde MLflow: {uri} (v{prod_ver.version})...")
                import mlflow.xgboost
                state.modelo_alquiler = mlflow.xgboost.load_model(uri)
                modelo_cargado = True
                state.config_alquiler["modelo_version"] = f"v{prod_ver.version}_mlflow"
                try:
                    run_info = mlflow_manager.client.get_run(prod_ver.run_id)
                    m = run_info.data.metrics
                    if "mape_pct" in m:
                        state.config_alquiler["mape_test"] = round(m["mape_pct"], 4)
                    if "r2" in m:
                        state.config_alquiler["r2_test"] = round(m["r2"], 4)
                    if "mae" in m:
                        state.config_alquiler["mae_test"] = round(m["mae"], 2)
                    if "rmse" in m:
                        state.config_alquiler["rmse_test"] = round(m["rmse"], 2)
                except Exception as ex_m:
                    print(f"[ModelLoader][Alquiler] No se pudieron sincronizar métricas de MLflow: {ex_m}")
                print(f"[ModelLoader][Alquiler] [OK] Modelo cargado desde MLflow Registry a RAM.")
            except Exception as e:
                print(f"[ModelLoader][Alquiler] Advertencia: No se pudo cargar desde MLflow ({e}). Usando fallback local...")

    if not modelo_cargado:
        config_model_file = state.config_alquiler.get("modelo_archivo", "xgboost_alquiler_v1.pkl")
        model_path = _resolver_ruta(
            model_path_override
            or os.getenv("MODEL_ALQUILER_PATH")
            or os.path.join(models_dir, config_model_file)
            or "models/xgboost_alquiler_v1.pkl"
        )

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Modelo de alquiler no encontrado: {model_path}\n"
                f"Verifica que el archivo exista en {models_dir}/"
            )
        print(f"[ModelLoader][Alquiler] Cargando modelo: {model_path} ...")
        state.modelo_alquiler = joblib.load(model_path)
        print(f"[ModelLoader][Alquiler] [OK] Modelo cargado ({os.path.getsize(model_path) / 1e6:.1f} MB)")

    print("[ModelLoader][Alquiler] Inicializando SHAP TreeExplainer ...")
    state.explainer_alquiler = shap.TreeExplainer(state.modelo_alquiler)
    print("[ModelLoader][Alquiler] [OK] SHAP TreeExplainer listo")

    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Metadata de alquiler no encontrada: {metadata_path}")
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    state.feature_cols_alquiler = (
        list(state.modelo_alquiler.feature_names_in_)
        if hasattr(state.modelo_alquiler, "feature_names_in_")
        else metadata["features"]
    )
    state.encoding_map_alquiler = metadata["encoding_map_distrito"]
    state.media_global_alquiler = metadata["media_global_target"]
    print(f"[ModelLoader][Alquiler] [OK] Metadata: {len(state.feature_cols_alquiler)} features, {len(state.encoding_map_alquiler)} distritos")


def cargar_modelo(
    state: ModelState,
    tipo: Literal["all", "venta", "alquiler"] = "all",
    model_path_override: str | None = None,
) -> None:
    """
    Carga los modelos en el ModelState proporcionado.
    Lanza RuntimeError si algún archivo crítico no existe.
    """
    if tipo in ("all", "venta"):
        _cargar_artefactos_venta(state, model_path_override=model_path_override if tipo == "venta" else None)

    if tipo in ("all", "alquiler"):
        _cargar_artefactos_alquiler(state, model_path_override=model_path_override if tipo == "alquiler" else None)

    # Contexto distrital desde PostgreSQL (compartido para ambos modelos)
    if state.data_provider is None:
        print("[ModelLoader] Cargando contexto distrital desde PostgreSQL ...")
        state.data_provider = PostgreSQLContextProvider()
        n_distritos = len(state.data_provider.listar_distritos())
        print(f"[ModelLoader] [OK] Contexto distrital: {n_distritos} distritos cargados")

    print("[ModelLoader] ===== Servicio ML listo (Venta + Alquiler) =====")


def liberar_recursos(state: ModelState) -> None:
    """Limpieza al cerrar la app (opcional — Python GC lo maneja igual)."""
    state.modelo                 = None
    state.explainer              = None
    state.modelo_alquiler        = None
    state.explainer_alquiler     = None
    state.data_provider          = None
    print("[ModelLoader] Recursos liberados.")
