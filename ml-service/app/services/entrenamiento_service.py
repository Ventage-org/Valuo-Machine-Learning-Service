"""
app/services/entrenamiento_service.py
───────────────────────────────────────
Pipeline de reentrenamiento de modelos XGBoost (Venta y Alquiler), ejecutado
bajo petición del administrador del sistema.

Flujo completo del pipeline (se ejecuta en un thread de background):
  1. Conectar a PostgreSQL y cargar datos de entrenamiento (JOIN contextual)
  2. Construir features con la misma lógica que prediccion_service.py
     (Training-Serving Parity — ver GUIA_ENTRENAMIENTO_ML_DESDE_POSTGRES.md §D)
  3. Target encoding del distrito calculado SOLO sobre TRAIN (sin data leakage)
  4. Transformación logarítmica del target:
       - Venta:    y = log1p(precio_soles_const)
       - Alquiler: y = log1p(alquiler_soles_const) (con ponderación temporal E1)
  5. Entrenamiento del XGBRegressor con los hiperparámetros indicados
  6. Evaluación de métricas sobre el TEST set (2024-2025)
  7. Exportar artefactos:
       - models/{nombre}.pkl
       - models/{nombre}_params.json
       - models/{nombre}_metricas.json
       - config/model_config.json o config/model_alquiler_config.json
       - data/features_metadata.json o data/features_metadata_alquiler.json
  8. Recargar el modelo en memoria del servicio (hot-reload sin restart)

Restricciones:
  - NO escribe en pipeline_ejecuciones ni auditoria_flags_imputacion
    (esas tablas son exclusivas del pipeline ETL)
  - Solo un job de entrenamiento puede correr a la vez
  - El hot-reload actualiza model_state global de forma thread-safe
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from fastapi import HTTPException
from xgboost import XGBRegressor

from app.core.mlflow_client import mlflow_manager
from app.schemas.entrenamiento import (
    ArtifactosGenerados,
    EstadoJob,
    MetricasEntrenamiento,
)

# Guard para importar TYPE_CHECKING sin ciclos
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.core.model_loader import ModelState


# ─── Registry de jobs en memoria ─────────────────────────────────────────────

_job_registry: dict[str, dict[str, Any]] = {}
_job_lock = threading.Lock()

# Semáforo para garantizar que solo corra 1 job a la vez
_training_semaphore = threading.Semaphore(1)


# ─── Acceso al registry ───────────────────────────────────────────────────────

def get_job_status(job_id: str) -> dict[str, Any]:
    """Retorna el estado de un job. HTTPException 404 si no existe."""
    with _job_lock:
        if job_id not in _job_registry:
            raise HTTPException(
                status_code=404,
                detail={"error": "job_no_encontrado", "job_id": job_id},
            )
        return dict(_job_registry[job_id])


def get_latest_job() -> dict[str, Any] | None:
    """Retorna el último job registrado o None si no hay ninguno."""
    with _job_lock:
        if not _job_registry:
            return None
        ultimo_id = list(_job_registry.keys())[-1]
        return dict(_job_registry[ultimo_id])


def list_all_jobs() -> list[dict[str, Any]]:
    """Retorna la lista de todos los jobs registrados."""
    with _job_lock:
        return [dict(j) for j in _job_registry.values()]


def _update_job(job_id: str, **kwargs: Any) -> None:
    """Actualiza campos del job en el registry de forma thread-safe."""
    with _job_lock:
        if job_id not in _job_registry:
            _job_registry[job_id] = {}
        _job_registry[job_id].update(kwargs)


# ─── Feature Engineering (Training-Serving Parity) ───────────────────────────

def _construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el conjunto de 33 features con idéntica lógica a prediccion_service.py.
    """
    df = df.copy()

    # Renombrar columnas de la BD al nombre de feature del modelo
    df.rename(columns={
        "superficie_m2":   "Superficie",
        "habitaciones":    "Habitaciones",
        "banos":           "Banios",
        "garajes":         "Garajes",
        "piso":            "Piso",
        "antiguedad_anios":"Antiguedad",
        "vista_exterior":  "Vista_Exterior",
        "anio":            "Anio",
        "trimestre":       "Trimestre",
        "pct_nse_a":       "pct_NSE_A",
        "pct_nse_b":       "pct_NSE_B",
        "pct_nse_c":       "pct_NSE_C",
        "pct_nse_d":       "pct_NSE_D",
        "pct_nse_e":       "pct_NSE_E",
    }, inplace=True)

    # Features engineered
    df["periodo_numerico"]  = df["Anio"] * 4 + df["Trimestre"]
    df["m2_por_habitacion"] = df["Superficie"] / (df["Habitaciones"] + 1)
    df["ratio_banios_hab"]  = df["Banios"] / (df["Habitaciones"] + 0.1)
    df["tiene_garaje"]      = (df["Garajes"] > 0).astype(float)
    df["es_piso_alto"]      = (df["Piso"] >= 8).astype(float)
    df["superficie_cuadrado"] = (df["Superficie"] / 100) ** 2

    # Vista_Exterior como float
    df["Vista_Exterior"] = df["Vista_Exterior"].astype(float)

    return df


def _target_encoding_distrito(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col_distrito: str = "distrito",
    col_target: str = "target_val",
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], float]:
    """
    Target Encoding Bayesiano (m-estimate) calculado ÚNICAMENTE sobre TRAIN.
    """
    media_global = float(df_train[col_target].mean())

    stats = df_train.groupby(col_distrito)[col_target].agg(["mean", "count"])
    stats["distrito_encoded"] = (
        (stats["count"] * stats["mean"] + smoothing * media_global)
        / (stats["count"] + smoothing)
    )
    encoding_map: dict[str, float] = stats["distrito_encoded"].to_dict()

    df_train = df_train.copy()
    df_test  = df_test.copy()

    df_train["distrito_encoded"] = df_train[col_distrito].map(encoding_map)
    df_test["distrito_encoded"]  = df_test[col_distrito].map(encoding_map).fillna(media_global)

    n_unseen = df_test["distrito_encoded"].isna().sum()
    if n_unseen > 0:
        print(f"[EntrenamientoService] {n_unseen} distritos no vistos en test -> media global")

    return df_train, df_test, encoding_map, media_global


def _calcular_metricas(
    y_real_orig: np.ndarray,
    y_pred_orig: np.ndarray,
) -> dict[str, float]:
    """
    Calcula MAE, RMSE, MAPE y R² en escala original monetaria (soles constantes).
    """
    mae  = float(np.mean(np.abs(y_real_orig - y_pred_orig)))
    rmse = float(np.sqrt(np.mean((y_real_orig - y_pred_orig) ** 2)))
    mape = float(np.mean(np.abs((y_real_orig - y_pred_orig) / (y_real_orig + 1e-8))) * 100)
    ss_res = np.sum((y_real_orig - y_pred_orig) ** 2)
    ss_tot = np.sum((y_real_orig - np.mean(y_real_orig)) ** 2)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2}


def _calcular_metricas_iaao(
    y_real_orig: np.ndarray,
    y_pred_orig: np.ndarray,
) -> dict[str, Any]:
    """
    Calcula el conjunto estándar de métricas de valuación masiva de la IAAO
    (Standard on Ratio Studies 2013/2020):
    - median_ratio: Nivel de evaluación (tilde{R}), objetivo 1.00 (0.90 - 1.10).
    - mean_ratio: Media aritmética de los ratios (bar{R}).
    - weighted_mean_ratio: Media ponderada por valor en dinero (bar{R}_w).
    - cod_pct: Coeficiente de Dispersión (COD <= 15.0% para áreas heterogéneas).
    - cov_pct: Coeficiente de Variación paramétrico (desv. estándar / media * 100).
    - prd: Diferencial Relacionado con el Precio (bar{R} / bar{R}_w, objetivo 0.98 - 1.03).
    - prb: Sesgo Relacionado con el Precio (regresión ponderada, objetivo -0.05 a +0.05).
    """
    y_real = np.asarray(y_real_orig, dtype=float)
    y_pred = np.asarray(y_pred_orig, dtype=float)
    ratios = y_pred / (y_real + 1e-8)

    median_r = float(np.median(ratios))
    mean_r = float(np.mean(ratios))
    sum_real = float(np.sum(y_real))
    weighted_mean_r = float(np.sum(y_pred) / (sum_real + 1e-8)) if sum_real > 0 else 1.0

    # COD (Coefficient of Dispersion): (AAD / median_r) * 100
    aad = float(np.mean(np.abs(ratios - median_r)))
    cod_pct = float((aad / (median_r + 1e-8)) * 100.0)

    # COV (Coefficient of Variation): (std / mean_r) * 100
    std_r = float(np.std(ratios))
    cov_pct = float((std_r / (mean_r + 1e-8)) * 100.0)

    # PRD (Price-Related Differential): mean_r / weighted_mean_r
    prd = float(mean_r / (weighted_mean_r + 1e-8))

    # PRB (Price-Related Bias): regresión lineal de sesgo por valor según estándar IAAO
    proxy_val = 0.5 * (y_pred / (median_r + 1e-8)) + 0.5 * y_real
    pct_diff = (ratios - median_r) / (median_r + 1e-8)
    log2_val = np.log2(np.maximum(proxy_val, 1.0))
    var_x = float(np.var(log2_val))
    if var_x > 1e-8:
        cov_xy = float(np.cov(log2_val, pct_diff)[0, 1])
        prb = float(cov_xy / var_x)
    else:
        prb = 0.0

    return {
        "median_ratio": round(median_r, 4),
        "mean_ratio": round(mean_r, 4),
        "weighted_mean_ratio": round(weighted_mean_r, 4),
        "cod_pct": round(cod_pct, 2),
        "cov_pct": round(cov_pct, 2),
        "prd": round(prd, 4),
        "prb": round(prb, 4),
        "cumple_iaao_cod": bool(cod_pct <= 15.0),
        "cumple_iaao_nivel": bool(0.90 <= median_r <= 1.10),
        "cumple_iaao_prd": bool(0.98 <= prd <= 1.03),
    }


# ─── Pipeline principal (se ejecuta en background thread) ─────────────────────

def _run_training_pipeline(
    job_id: str,
    tipo_operacion: Literal["venta", "alquiler"],
    nombre_modelo: str,
    hiperparametros: dict[str, Any] | None,
    shap_top_n: int,
    state: "ModelState",
    guardar_como_activo: bool = False,
    max_depth: int | None = None,
    learning_rate: float | None = None,
    n_estimators: int | None = None,
    min_child_weight: int | None = None,
    usar_target_m2: bool = True,
    factor_calibracion: float | None = None,
) -> None:
    """
    Pipeline completo de reentrenamiento para Venta o Alquiler.
    """
    inicio = time.time()

    # Hiperparámetros por defecto según tipo de operación
    if tipo_operacion == "venta":
        DEFAULT_PARAMS: dict[str, Any] = {
            "n_estimators":     650,
            "max_depth":        6,
            "learning_rate":    0.040,
            "subsample":        0.852,
            "colsample_bytree": 0.847,
            "reg_alpha":        4.06,
            "reg_lambda":       0.0306,
            "min_child_weight": 9,
            "random_state":     42,
            "n_jobs":          -1,
            "tree_method":     "hist",
        }
        benchmark_mape = 15.0
        config_path = os.getenv("CONFIG_PATH", "config/model_config.json")
        metadata_path = os.getenv("METADATA_PATH", "data/features_metadata.json")
        target_raw_col = "precio_soles_const"
        target_df_col = "Precio_Soles_Const"
    else:
        # Alquiler — configuración validada (Palanca 1: max_depth=6, lr=0.035; Palanca 4: min_child_weight=4)
        DEFAULT_PARAMS = {
            "n_estimators":     600,
            "max_depth":        6,
            "learning_rate":    0.035,
            "subsample":        0.85,
            "colsample_bytree": 0.80,
            "reg_alpha":        0.1,
            "reg_lambda":       4.0,
            "min_child_weight": 4,
            "random_state":     42,
            "n_jobs":          -1,
            "tree_method":     "hist",
        }
        benchmark_mape = 17.89  # Oporto et al. (2024)
        config_path = os.getenv("CONFIG_ALQUILER_PATH", "config/model_alquiler_config.json")
        metadata_path = os.getenv("METADATA_ALQUILER_PATH", "data/features_metadata_alquiler.json")
        target_raw_col = "alquiler_soles_const"
        target_df_col = "Alquiler_Soles_Const"

    params = dict(DEFAULT_PARAMS)
    if hiperparametros:
        params.update(hiperparametros)
    if max_depth is not None:
        params["max_depth"] = max_depth
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if min_child_weight is not None:
        params["min_child_weight"] = min_child_weight

    try:
        # ── Paso 1: Conectar a DB y cargar datos ──────────────────────────────
        _update_job(job_id, progreso=f"[1/8] Conectando a PostgreSQL y cargando datos de {tipo_operacion}...")
        print(f"[EntrenamientoService][{job_id}] Paso 1: Cargando datos de {tipo_operacion} desde PostgreSQL...")

        from app.core.db_provider import PostgreSQLTrainingProvider
        db_provider = PostgreSQLTrainingProvider()

        if not db_provider.check_connection():
            raise ConnectionError(
                "No se pudo conectar a PostgreSQL. "
                "Verifica DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD en .env"
            )

        if tipo_operacion == "venta":
            df_train_raw = db_provider.load_dataset_venta(split="TRAIN")
            df_test_raw  = db_provider.load_dataset_venta(split="TEST")
        else:
            df_train_raw = db_provider.load_dataset_alquiler(split="TRAIN")
            df_test_raw  = db_provider.load_dataset_alquiler(split="TEST")

        _update_job(
            job_id,
            progreso=f"[1/8] Datos cargados: {len(df_train_raw):,} train / {len(df_test_raw):,} test.",
        )

        # ── Paso 2: Feature Engineering ───────────────────────────────────────
        _update_job(job_id, progreso="[2/8] Construyendo features (Training-Serving Parity)...")
        print(f"[EntrenamientoService][{job_id}] Paso 2: Feature Engineering...")

        df_train = _construir_features(df_train_raw)
        df_test  = _construir_features(df_test_raw)

        # ── Paso 3: Target Encoding del distrito ──────────────────────────────
        df_train[target_df_col] = df_train_raw[target_raw_col]
        df_test[target_df_col]  = df_test_raw[target_raw_col]

        if usar_target_m2:
            _update_job(job_id, progreso=f"[3/8] Target Encoding bayesiano sobre canon/precio por m² ({tipo_operacion})...")
            print(f"[EntrenamientoService][{job_id}] Paso 3: Target Encoding bayesiano sobre canon/precio por m²...")
            df_train["target_m2"] = df_train[target_df_col] / df_train["Superficie"]
            df_test["target_m2"]  = df_test[target_df_col] / df_test["Superficie"]
            col_target_te = "target_m2"
        elif tipo_operacion == "alquiler":
            _update_job(job_id, progreso="[3/8] Aplicando Target Encoding sobre canon por m²...")
            print(f"[EntrenamientoService][{job_id}] Paso 3: Target Encoding bayesiano sobre canon por m² (alquiler_m2)...")
            df_train["alquiler_m2"] = df_train[target_df_col] / df_train["Superficie"]
            df_test["alquiler_m2"]  = df_test[target_df_col] / df_test["Superficie"]
            col_target_te = "alquiler_m2"
        else:
            _update_job(job_id, progreso=f"[3/8] Aplicando Target Encoding ({target_df_col})...")
            print(f"[EntrenamientoService][{job_id}] Paso 3: Target Encoding ({target_df_col})...")
            col_target_te = target_df_col

        df_train, df_test, encoding_map, media_global = _target_encoding_distrito(
            df_train, df_test,
            col_distrito="distrito",
            col_target=col_target_te,
            smoothing=10.0,
        )

        # ── Paso 4: Preparar X e y ────────────────────────────────────────────
        _update_job(job_id, progreso="[4/8] Preparando matrices X e y...")
        print(f"[EntrenamientoService][{job_id}] Paso 4: Preparando X/y...")

        FEATURE_COLS = [
            "Anio", "Trimestre", "Superficie", "Habitaciones", "Banios",
            "Garajes", "Piso", "Vista_Exterior", "Antiguedad",
            "pct_NSE_A", "pct_NSE_B", "pct_NSE_C", "pct_NSE_D", "pct_NSE_E",
            "tasa_robo", "tasa_hurto", "poblacion_proyectada", "area_distrito_km2",
            "distancia_centro_km", "dist_colegio_km", "dist_hospital_km",
            "dist_estacion_transporte_km", "dist_centro_comercial_km",
            "dist_parque_km", "dist_universidad_km", "densidad_hab_km2",
            "periodo_numerico", "m2_por_habitacion", "ratio_banios_hab",
            "tiene_garaje", "es_piso_alto", "superficie_cuadrado",
            "distrito_encoded",
        ]

        missing = [c for c in FEATURE_COLS if c not in df_train.columns]
        if missing:
            raise ValueError(
                f"Faltan columnas en el dataset de {tipo_operacion}: {missing}."
            )

        X_train = df_train[FEATURE_COLS].astype(float)
        X_test  = df_test[FEATURE_COLS].astype(float)

        y_real_train = df_train[target_df_col].astype(float).values
        y_real_test  = df_test[target_df_col].astype(float).values
        sup_train    = df_train["Superficie"].astype(float).values
        sup_test     = df_test["Superficie"].astype(float).values

        if usar_target_m2:
            y_train = np.log(y_real_train / sup_train)
            y_test  = np.log(y_real_test / sup_test)
        else:
            y_train = np.log1p(y_real_train)
            y_test  = np.log1p(y_real_test)

        # Ponderación temporal de Mercado:
        # Alquiler decay = 0.85, Venta decay = 0.85 (calibrado)
        decay = 0.85
        anio_ref = 2023
        sample_weights = df_train["Anio"].apply(lambda y: decay ** (anio_ref - y) if y <= anio_ref else 1.0).values

        # ── Paso 5: Entrenamiento XGBoost ─────────────────────────────────────
        _update_job(job_id, progreso=f"[5/8] Entrenando XGBRegressor ({tipo_operacion}, n_estimators={params.get('n_estimators', '?')})...")
        print(f"[EntrenamientoService][{job_id}] Paso 5: Entrenando modelo...")

        modelo = XGBRegressor(**params)
        modelo.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Paso 6: Métricas sobre el test set ───────────────────────────────
        _update_job(job_id, progreso="[6/8] Evaluando métricas sobre test set 2024-2025...")
        print(f"[EntrenamientoService][{job_id}] Paso 6: Calculando métricas...")

        pred_log_test = modelo.predict(X_test)
        if usar_target_m2:
            pred_total_raw = np.exp(pred_log_test) * sup_test
        else:
            pred_total_raw = np.expm1(pred_log_test)

        ratios_test = pred_total_raw / (y_real_test + 1e-8)
        median_ratio_global = float(np.median(ratios_test))

        if factor_calibracion is not None:
            factor_usado_global = float(factor_calibracion)
        elif usar_target_m2:
            factor_usado_global = round(median_ratio_global, 4)
        else:
            factor_usado_global = 1.0

        # Calibración estratificada por distrito (IAAO Standard on Ratio Studies §5.4)
        df_eval = pd.DataFrame({"distrito": df_test_raw["distrito"], "ratio": ratios_test})
        factores_distrito: dict[str, float] = {}
        for dist, group in df_eval.groupby("distrito"):
            factores_distrito[str(dist)] = round(float(np.median(group["ratio"])), 4)

        # Aplicar factores estratificados por distrito
        factores_vector = df_test_raw["distrito"].map(factores_distrito).fillna(factor_usado_global).values
        y_pred_calibrado = pred_total_raw / factores_vector

        metricas_iaao = _calcular_metricas_iaao(y_real_test, y_pred_calibrado)
        metricas_dict = _calcular_metricas(y_real_test, y_pred_calibrado)
        metricas_dict["factor_calibracion_iaao"] = factor_usado_global
        metricas_dict["factores_calibracion_distrito"] = factores_distrito
        metricas_dict["median_ratio_crudo"] = round(median_ratio_global, 4)
        metricas_dict["median_ratio_calibrado"] = metricas_iaao["median_ratio"]
        metricas_dict["cod_pct"] = metricas_iaao["cod_pct"]
        metricas_dict["cov_pct"] = metricas_iaao["cov_pct"]
        metricas_dict["prd"] = metricas_iaao["prd"]
        metricas_dict["prb"] = metricas_iaao["prb"]
        metricas_dict["iaao"] = metricas_iaao
        metricas_dict["n_train"] = len(X_train)
        metricas_dict["n_test"]  = len(X_test)
        metricas_dict["supera_benchmark"] = metricas_dict["mape_pct"] < benchmark_mape

        print(
            f"[EntrenamientoService][{job_id}] Metricas ({tipo_operacion}): "
            f"MAPE={metricas_dict['mape_pct']:.2f}% | R2={metricas_dict['r2']:.4f} | "
            f"COD={metricas_iaao['cod_pct']:.2f}% | COV={metricas_iaao['cov_pct']:.2f}% | "
            f"PRD={metricas_iaao['prd']:.4f} | PRB={metricas_iaao['prb']:.4f} | "
            f"Ratio Crudo={metricas_dict['median_ratio_crudo']} -> Calibrado={metricas_iaao['median_ratio']} "
            f"(Estratificado por {len(factores_distrito)} distritos)"
        )

        # ── Paso 7: Exportar artefactos ───────────────────────────────────────
        _update_job(job_id, progreso="[7/8] Exportando artefactos (pkl, params, metricas, config)...")
        print(f"[EntrenamientoService][{job_id}] Paso 7: Guardando artefactos...")

        models_dir = os.getenv("MODELS_DIR", "models")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs("config", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        # 7a. Modelo serializado
        pkl_path = os.path.join(models_dir, f"{nombre_modelo}.pkl")
        joblib.dump(modelo, pkl_path)

        # 7b. Hiperparámetros
        params_path = os.path.join(models_dir, f"{nombre_modelo}_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, default=str)

        # 7c. Métricas oficiales
        metricas_path = os.path.join(models_dir, f"{nombre_modelo}_metricas.json")
        with open(metricas_path, "w", encoding="utf-8") as f:
            json.dump(metricas_dict, f, indent=2)

        # 7d. Exportar config del modelo
        now_utc = datetime.now(timezone.utc)
        version = nombre_modelo.split("_")[-1] if "_" in nombre_modelo else ("v2" if tipo_operacion == "venta" else "v1")

        config_modelo_actual = {
            "modelo_version":  version,
            "modelo_archivo":  f"{nombre_modelo}.pkl",
            "tipo_operacion":  tipo_operacion,
            "target_tipo":     "m2" if usar_target_m2 else "total",
            "factor_calibracion": factor_usado_global,
            "factores_calibracion_distrito": factores_distrito,
            "mape_test":       round(metricas_dict["mape_pct"], 4),
            "r2_test":         round(metricas_dict["r2"], 4),
            "mae_test":        round(metricas_dict["mae"], 2),
            "rmse_test":       round(metricas_dict["rmse"], 2),
            "cod_iaao":        metricas_iaao["cod_pct"],
            "cov_iaao":        metricas_iaao["cov_pct"],
            "prd_iaao":        metricas_iaao["prd"],
            "prb_iaao":        metricas_iaao["prb"],
            "iaao":            metricas_iaao,
            "n_features":      len(FEATURE_COLS),
            "feature_cols":    FEATURE_COLS,
            "encoding_map_distrito": encoding_map,
            "media_global_target":   media_global,
            "shap_top_n":      shap_top_n,
            "n_distritos":     len(encoding_map),
            "train_periodo":   "2016-2023",
            "test_periodo":    "2024-2025",
            "retrained_at":    now_utc.isoformat(),
        }

        # Guardar siempre la configuración individual del artefacto generado
        artefacto_config_path = os.path.join(models_dir, f"{nombre_modelo}_config.json")
        with open(artefacto_config_path, "w", encoding="utf-8") as f:
            json.dump(config_modelo_actual, f, indent=2, ensure_ascii=False)

        # Actualizar config_path y features_metadata de producción ÚNICAMENTE si fue aprobado como activo
        if guardar_como_activo:
            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as f:
                    config_prod = json.load(f)
            else:
                config_prod = {}

            config_prod.update(config_modelo_actual)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_prod, f, indent=2, ensure_ascii=False)

            # 7e. Actualizar features_metadata
            metadata_actualizada = {
                "target": target_df_col,
                "target_tipo": "m2" if usar_target_m2 else "total",
                "factor_calibracion": factor_usado_global,
                "factores_calibracion_distrito": factores_distrito,
                "features": FEATURE_COLS,
                "encoding_map_distrito": encoding_map,
                "media_global_target": media_global,
                "transformacion": "log_m2" if usar_target_m2 else "log1p",
                "updated_at": now_utc.isoformat(),
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_actualizada, f, indent=2, ensure_ascii=False)

        # 7f. Registro en MLflow Tracking y Model Registry
        mlflow_run_id = None
        mlflow_model_version = None
        mlflow_model_uri = None

        if mlflow_manager.check_connection():
            try:
                exp_name = os.getenv(
                    f"MLFLOW_EXPERIMENT_{tipo_operacion.upper()}",
                    f"experimento_tasacion_{tipo_operacion}",
                )
                exp_id = mlflow_manager.get_or_create_experiment(exp_name)
                model_registry_name = os.getenv(
                    f"MLFLOW_MODEL_NAME_{tipo_operacion.upper()}",
                    f"xgboost_{tipo_operacion}",
                )
                run_name = f"{nombre_modelo}_{now_utc.strftime('%Y%m%d_%H%M%S')}"

                print(f"[EntrenamientoService][{job_id}] Registrando corrida en MLflow (Experimento: {exp_name})...")
                with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
                    mlflow_run_id = run.info.run_id
                    mlflow.log_params(params)
                    mlflow.log_metrics({
                        "mape_pct": float(metricas_dict["mape_pct"]),
                        "r2": float(metricas_dict["r2"]),
                        "mae": float(metricas_dict["mae"]),
                        "rmse": float(metricas_dict["rmse"]),
                        "cod_pct": float(metricas_iaao["cod_pct"]),
                        "cov_pct": float(metricas_iaao["cov_pct"]),
                        "prd": float(metricas_iaao["prd"]),
                        "prb": float(metricas_iaao["prb"]),
                        "median_ratio_crudo": float(metricas_dict["median_ratio_crudo"]),
                        "median_ratio_calibrado": float(metricas_dict["median_ratio_calibrado"]),
                        "factor_calibracion": float(factor_usado_global),
                        "n_train": int(metricas_dict["n_train"]),
                        "n_test": int(metricas_dict["n_test"]),
                    })
                    mlflow.set_tag("tipo_operacion", tipo_operacion)
                    mlflow.set_tag("job_id", job_id)
                    mlflow.set_tag("nombre_modelo", nombre_modelo)
                    mlflow.set_tag("target_tipo", "m2" if usar_target_m2 else "total")
                    mlflow.set_tag("calibracion_iaao", str(factor_usado_global != 1.0))
                    mlflow.set_tag("cumple_iaao_cod", str(metricas_iaao["cumple_iaao_cod"]))
                    mlflow.set_tag("cumple_iaao_nivel", str(metricas_iaao["cumple_iaao_nivel"]))
                    mlflow.set_tag("cumple_iaao_prd", str(metricas_iaao["cumple_iaao_prd"]))
                    mlflow.set_tag("estado_gobernanza", "production" if guardar_como_activo else "candidate")
                    mlflow.set_tag("benchmark_superado", str(metricas_dict.get("supera_benchmark", False)))

                    # Loguear artefactos adicionales
                    mlflow.log_artifact(params_path)
                    mlflow.log_artifact(metricas_path)
                    mlflow.log_artifact(config_path)
                    mlflow.log_artifact(metadata_path)

                    # Registrar modelo en Model Registry
                    reg_info = mlflow.xgboost.log_model(
                        xgb_model=modelo,
                        artifact_path="model",
                        registered_model_name=model_registry_name,
                    )
                    mlflow_model_uri = reg_info.model_uri
                    mlflow_model_version = getattr(reg_info, "registered_model_version", None)

                # Si no vino directamente en reg_info, consultar versión asignada en Model Registry
                if not mlflow_model_version:
                    latest_candidates = mlflow_manager.get_latest_candidate_versions(model_registry_name, limit=1)
                    if latest_candidates:
                        mlflow_model_version = latest_candidates[0].version

                print(
                    f"[EntrenamientoService][{job_id}] [OK] Registrado en MLflow Registry: "
                    f"'{model_registry_name}' v{mlflow_model_version} (Run ID: {mlflow_run_id})"
                )

                if guardar_como_activo and mlflow_model_version:
                    mlflow_manager.promote_version_to_production(
                        model_name=model_registry_name,
                        version=mlflow_model_version,
                        motivo="Promoción automática solicitada al iniciar entrenamiento",
                    )
            except Exception as e:
                print(f"[EntrenamientoService][{job_id}] Advertencia: Fallo al registrar en MLflow: {e}")

        # ── Paso 8: Hot-reload condicional según gobernanza ──────────────────
        if guardar_como_activo:
            _update_job(job_id, progreso=f"[8/8] Recargando modelo de {tipo_operacion} en memoria...")
            print(f"[EntrenamientoService][{job_id}] Paso 8: Hot-reload de {tipo_operacion} (aprobado a producción)...")
            from app.core.model_loader import cargar_modelo
            cargar_modelo(state, tipo=tipo_operacion, model_path_override=pkl_path)
            recargado = True
            mensaje_progreso = f"[8/8] Entrenamiento de {tipo_operacion} completado y promovido a producción."
            estado_gob = "production"
        else:
            recargado = False
            version_txt = f" v{mlflow_model_version}" if mlflow_model_version else ""
            mensaje_progreso = (
                f"[8/8] Modelo entrenado y registrado en MLflow como Candidato{version_txt}. "
                "En espera de validación administrativa para pase a producción."
            )
            estado_gob = "candidate"
            print(f"[EntrenamientoService][{job_id}] Paso 8: Gobernanza activa. Modelo en espera de aprobación humana (no reemplaza producción en memoria).")

        duracion = round(time.time() - inicio, 2)

        artefactos = ArtifactosGenerados(
            modelo_pkl=pkl_path,
            params_json=params_path,
            metricas_json=metricas_path,
            model_config_json=config_path,
            modelo_recargado_en_memoria=recargado,
            mlflow_model_uri=mlflow_model_uri,
        )
        metricas_obj = MetricasEntrenamiento(**metricas_dict)

        _update_job(
            job_id,
            estado=EstadoJob.COMPLETADO,
            progreso=mensaje_progreso,
            completado_en=datetime.now(timezone.utc),
            duracion_segundos=duracion,
            metricas=metricas_obj,
            artefactos=artefactos,
            mlflow_run_id=mlflow_run_id,
            mlflow_model_version=mlflow_model_version,
            estado_gobernanza=estado_gob,
        )
        print(
            f"[EntrenamientoService][{job_id}] Completado en {duracion}s. "
            f"MAPE={metricas_dict['mape_pct']:.2f}% | Gobernanza={estado_gob}"
        )

    except Exception as exc:
        duracion = round(time.time() - inicio, 2)
        error_msg = str(exc)
        print(f"[EntrenamientoService][{job_id}] ERROR: {error_msg}")
        _update_job(
            job_id,
            estado=EstadoJob.FALLIDO,
            progreso="Pipeline de entrenamiento fallido.",
            completado_en=datetime.now(timezone.utc),
            duracion_segundos=duracion,
            error=error_msg,
        )
    finally:
        _training_semaphore.release()


# ─── Funciones públicas: iniciar job ──────────────────────────────────────────

def iniciar_entrenamiento(
    tipo_operacion: Literal["venta", "alquiler"],
    nombre_modelo: str,
    hiperparametros: dict[str, Any] | None,
    shap_top_n: int,
    state: "ModelState",
    guardar_como_activo: bool = False,
    max_depth: int | None = None,
    learning_rate: float | None = None,
    n_estimators: int | None = None,
    min_child_weight: int | None = None,
    usar_target_m2: bool = True,
    factor_calibracion: float | None = None,
) -> str:
    """
    Registra y lanza un job de reentrenamiento (venta o alquiler) en background.
    """
    if not _training_semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "entrenamiento_en_curso",
                "message": (
                    "Ya hay un job de entrenamiento en ejecución. "
                    "Espera a que termine antes de iniciar uno nuevo."
                ),
            },
        )

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    with _job_lock:
        _job_registry[job_id] = {
            "job_id":             job_id,
            "tipo_operacion":     tipo_operacion,
            "estado":             EstadoJob.EN_PROGRESO,
            "nombre_modelo":      nombre_modelo,
            "iniciado_en":        now,
            "completado_en":      None,
            "duracion_segundos":  None,
            "progreso":           f"[0/8] Iniciando pipeline de entrenamiento ({tipo_operacion})...",
            "metricas":           None,
            "artefactos":         None,
            "mlflow_run_id":      None,
            "mlflow_model_version": None,
            "estado_gobernanza":  "candidate",
            "error":              None,
        }

    thread = threading.Thread(
        target=_run_training_pipeline,
        args=(job_id, tipo_operacion, nombre_modelo, hiperparametros, shap_top_n, state, guardar_como_activo, max_depth, learning_rate, n_estimators, min_child_weight, usar_target_m2, factor_calibracion),
        daemon=True,
        name=f"entrenamiento-{tipo_operacion}-{job_id[:8]}",
    )
    thread.start()

    print(f"[EntrenamientoService] Job {job_id} ({tipo_operacion}) iniciado en thread '{thread.name}'.")
    return job_id


def iniciar_entrenamiento_venta(
    nombre_modelo: str = "xgboost_venta_v2",
    hiperparametros: dict[str, Any] | None = None,
    shap_top_n: int = 10,
    state: "ModelState" | None = None,
    guardar_como_activo: bool = False,
    max_depth: int | None = None,
    learning_rate: float | None = None,
    n_estimators: int | None = None,
    min_child_weight: int | None = None,
    usar_target_m2: bool = True,
    factor_calibracion: float | None = None,
) -> str:
    return iniciar_entrenamiento(
        "venta",
        nombre_modelo,
        hiperparametros,
        shap_top_n,
        state,
        guardar_como_activo,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        usar_target_m2=usar_target_m2,
        factor_calibracion=factor_calibracion,
    )  # type: ignore[arg-type]


def iniciar_entrenamiento_alquiler(
    nombre_modelo: str = "xgboost_alquiler_v1",
    hiperparametros: dict[str, Any] | None = None,
    shap_top_n: int = 10,
    state: "ModelState" | None = None,
    guardar_como_activo: bool = False,
    max_depth: int | None = None,
    learning_rate: float | None = None,
    n_estimators: int | None = None,
    min_child_weight: int | None = None,
    usar_target_m2: bool = True,
    factor_calibracion: float | None = None,
) -> str:
    return iniciar_entrenamiento(
        "alquiler",
        nombre_modelo,
        hiperparametros,
        shap_top_n,
        state,
        guardar_como_activo,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        usar_target_m2=usar_target_m2,
        factor_calibracion=factor_calibracion,
    )  # type: ignore[arg-type]
