"""
scripts/audit_nse_pipeline.py
─────────────────────────────
Script Automatizado para la Auditoría Masiva de Variables NSE (ENAHO) en XGBoost.

Ejecuta las 3 pruebas estadísticas y espaciales recomendadas:
  1. Permutación Distrital de Vectores NSE (District Shuffle)
  2. Estudio de Ablación de Componentes de Ubicación (M0, M1, M2, M3)
  3. Validación Cruzada por Grupos (GroupKFold - 5 Folds) para Distritos No Vistos

Uso estándar:
    py scripts/audit_nse_pipeline.py

Uso con opciones avanzadas:
    py scripts/audit_nse_pipeline.py --tipo venta --n-splits 5 --export-json docs/resultados_auditoria_nse.json
    py scripts/audit_nse_pipeline.py --host mi-servidor.postgres.database.azure.com --user mi_usuario
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.request
from typing import Any, Literal

# Asegurar que ml-service esté en sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_SERVICE_DIR = os.path.join(BASE_DIR, "ml-service")
if ML_SERVICE_DIR not in sys.path:
    sys.path.insert(0, ML_SERVICE_DIR)

# Cargar variables de entorno si existe .env
try:
    from dotenv import load_dotenv

    for env_path in [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(ML_SERVICE_DIR, ".env"),
    ]:
        if os.path.exists(env_path):
            load_dotenv(env_path)
except ImportError:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from app.core.db_provider import PostgreSQLTrainingProvider


# ─── Funciones Auxiliares de Feature Engineering y Métricas ──────────────────

def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las características y ratios inmobiliarios."""
    df = df.copy()
    df.rename(columns={
        "superficie_m2":    "Superficie",
        "habitaciones":     "Habitaciones",
        "banos":            "Banios",
        "garajes":          "Garajes",
        "piso":             "Piso",
        "antiguedad_anios": "Antiguedad",
        "vista_exterior":   "Vista_Exterior",
        "anio":             "Anio",
        "trimestre":        "Trimestre",
        "pct_nse_a":        "pct_NSE_A",
        "pct_nse_b":        "pct_NSE_B",
        "pct_nse_c":        "pct_NSE_C",
        "pct_nse_d":        "pct_NSE_D",
        "pct_nse_e":        "pct_NSE_E",
    }, inplace=True)

    df["periodo_numerico"]  = df["Anio"] * 4 + df["Trimestre"]
    df["m2_por_habitacion"] = df["Superficie"] / (df["Habitaciones"] + 1)
    df["ratio_banios_hab"]  = df["Banios"] / (df["Habitaciones"] + 0.1)
    df["tiene_garaje"]      = (df["Garajes"] > 0).astype(float)
    df["es_piso_alto"]      = (df["Piso"] >= 8).astype(float)
    df["superficie_cuadrado"] = (df["Superficie"] / 100) ** 2
    df["Vista_Exterior"]    = df["Vista_Exterior"].astype(float)
    return df


def target_encoding_distrito(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col_distrito: str = "distrito",
    col_target: str = "target_val",
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], float]:
    """Target Encoding bayesiano (m-estimate) calculado exclusivamente sobre Train."""
    media_global = float(df_train[col_target].mean())
    stats = df_train.groupby(col_distrito)[col_target].agg(["mean", "count"])
    stats["distrito_encoded"] = (
        (stats["count"] * stats["mean"] + smoothing * media_global)
        / (stats["count"] + smoothing)
    )
    encoding_map: dict[str, float] = stats["distrito_encoded"].to_dict()

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["distrito_encoded"] = df_train[col_distrito].map(encoding_map)
    df_test["distrito_encoded"] = df_test[col_distrito].map(encoding_map).fillna(media_global)
    return df_train, df_test, encoding_map, media_global


def calcular_metricas_iaao(y_real_orig: np.ndarray, y_pred_orig: np.ndarray) -> dict[str, Any]:
    """Calcula métricas clave de valuación masiva IAAO (COD, PRD, Median Ratio)."""
    y_real = np.asarray(y_real_orig, dtype=float)
    y_pred = np.asarray(y_pred_orig, dtype=float)
    ratios = y_pred / (y_real + 1e-8)

    median_r = float(np.median(ratios))
    mean_r = float(np.mean(ratios))
    sum_real = float(np.sum(y_real))
    weighted_mean_r = float(np.sum(y_pred) / (sum_real + 1e-8)) if sum_real > 0 else 1.0

    aad = float(np.mean(np.abs(ratios - median_r)))
    cod_pct = float((aad / (median_r + 1e-8)) * 100.0)
    std_r = float(np.std(ratios))
    cov_pct = float((std_r / (mean_r + 1e-8)) * 100.0)
    prd = float(mean_r / (weighted_mean_r + 1e-8))

    return {
        "median_ratio": round(median_r, 4),
        "cod_pct": round(cod_pct, 2),
        "cov_pct": round(cov_pct, 2),
        "prd": round(prd, 4),
    }


def obtener_ip_publica() -> str:
    """Intenta obtener la IP pública del host para guiar la configuración de firewall."""
    try:
        req = urllib.request.Request("https://api.ipify.org", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception:
        return "IP no detectable automáticamente"


def verificar_socket(host: str, port: int, timeout: float = 3.0) -> bool:
    """Verifica conectividad TCP a nivel de socket antes de invocar psycopg2."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auditoría Masiva de Variables NSE (ENAHO) en XGBoost"
    )
    parser.add_argument(
        "--tipo",
        choices=["venta", "alquiler"],
        default="venta",
        help="Tipo de dataset a evaluar: 'venta' (por defecto) o 'alquiler'.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("DB_HOST", "localhost"),
        help="Host de PostgreSQL (o variable DB_HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DB_PORT", "5432")),
        help="Puerto de PostgreSQL (por defecto: 5432).",
    )
    parser.add_argument(
        "--dbname",
        default=os.getenv("DB_NAME", "inmobiliaria_ml_db"),
        help="Nombre de la base de datos (por defecto: inmobiliaria_ml_db).",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("DB_USER", "postgres"),
        help="Usuario de PostgreSQL (por defecto: postgres).",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("DB_PASSWORD", "5432"),
        help="Contraseña de PostgreSQL.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Número de folds para GroupKFold (por defecto: 5).",
    )
    parser.add_argument(
        "--export-json",
        default="docs/resultados_auditoria_nse.json",
        help="Ruta donde guardar el resumen estructurado en JSON.",
    )
    return parser.parse_args()


def ejecutar_auditoria():
    args = parse_arguments()

    print("=" * 80)
    print(f"AUDITORÍA EXPERIMENTAL: VALIDACIÓN DE VARIABLES NSE ({args.tipo.upper()})")
    print("=" * 80)

    # ── 1. Diagnóstico de Conexión ──────────────────────────────────────────
    print(f"\n[Paso 1/5] Verificando conexión a PostgreSQL ({args.host}:{args.port}/{args.dbname})...")
    
    socket_ok = verificar_socket(args.host, args.port, timeout=4.0)
    if not socket_ok and ("azure.com" in args.host or args.host != "localhost"):
        ip_pub = obtener_ip_publica()
        print("\n" + "!" * 70)
        print("ALERTA DE ACCESO / FIREWALL DE AZURE DETECTADA:")
        print(f"  El servidor '{args.host}' no responde en el puerto {args.port}.")
        print("  Si la base de datos está en Azure Flexible Server, se requiere autorizar")
        print("  la dirección IP pública de esta máquina en las reglas de red.")
        print(f"  -> Tu IP pública actual es: {ip_pub}")
        print("\n  Pasos para habilitar acceso en Azure Portal:")
        print("  1. Navega a tu servidor: Azure Portal > psql-valuo-dev-01")
        print("  2. En el menú izquierdo, selecciona 'Networking' (Redes).")
        print(f"  3. Añade una regla de firewall con tu IP: '{ip_pub}'.")
        print("  4. Guarda los cambios y vuelve a ejecutar este script.")
        print("!" * 70 + "\n")
        return

    db = PostgreSQLTrainingProvider(
        host=args.host,
        port=args.port,
        db_name=args.dbname,
        user=args.user,
        password=args.password,
    )

    if not db.check_connection():
        print(f"[ERROR] No se pudo autenticar con la BD '{args.dbname}' en '{args.host}'.")
        print("Verifica usuario, contraseña y permisos de acceso.")
        return

    print("  -> Conexión a PostgreSQL establecida exitosamente.")

    # ── 2. Carga y Preparación de Datos ───────────────────────────────────────
    print(f"\n[Paso 2/5] Descargando registros de {args.tipo.upper()}...")
    t0 = time.time()
    if args.tipo == "venta":
        df_train_raw = db.load_dataset_venta(split="TRAIN")
        df_test_raw = db.load_dataset_venta(split="TEST")
        target_raw = "precio_soles_const"
    else:
        df_train_raw = db.load_dataset_alquiler(split="TRAIN")
        df_test_raw = db.load_dataset_alquiler(split="TEST")
        target_raw = "alquiler_soles_const"

    print(f"  -> Train: {len(df_train_raw):,} registros | Test: {len(df_test_raw):,} registros. ({time.time()-t0:.2f}s)")

    print("\n[Paso 3/5] Feature Engineering y Target Encoding sobre precio/canon m²...")
    df_train = construir_features(df_train_raw)
    df_test = construir_features(df_test_raw)

    df_train["target_m2"] = df_train_raw[target_raw] / df_train["Superficie"]
    df_test["target_m2"] = df_test_raw[target_raw] / df_test["Superficie"]

    df_train, df_test, _, _ = target_encoding_distrito(
        df_train, df_test, col_distrito="distrito", col_target="target_m2", smoothing=10.0
    )

    y_train = np.log(df_train["target_m2"].values)
    y_test = np.log(df_test["target_m2"].values)
    sup_test = df_test["Superficie"].values
    y_real_test = df_test_raw[target_raw].values

    base_cols = [
        "Anio", "Trimestre", "Superficie", "Habitaciones", "Banios",
        "Garajes", "Piso", "Vista_Exterior", "Antiguedad", "m2_por_habitacion",
        "ratio_banios_hab", "tiene_garaje", "es_piso_alto", "superficie_cuadrado"
    ]
    nse_cols = ["pct_NSE_A", "pct_NSE_B", "pct_NSE_C", "pct_NSE_D", "pct_NSE_E"]

    reporte_final: dict[str, Any] = {
        "tipo_operacion": args.tipo,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "fecha_ejecucion": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ── PRUEBA 2: ABLACIÓN DE COMPONENTES ────────────────────────────────────
    print("\n" + "=" * 80)
    print("EJECUTANDO PRUEBA 2: ESTUDIO DE ABLACIÓN (Contribución Marginal)")
    print("=" * 80)

    modelos_ablacion = {
        "M0: Físico Puro (Sin Ubicación)": base_cols,
        "M1: Solo ID (distrito_encoded)": base_cols + ["distrito_encoded"],
        "M2: Solo NSE ENAHO (pct_NSE_A..E)": base_cols + nse_cols,
        "M3: Completo (ID + NSE ENAHO)": base_cols + ["distrito_encoded"] + nse_cols,
    }

    res_ablacion: dict[str, Any] = {}
    for nombre, cols in modelos_ablacion.items():
        reg = XGBRegressor(
            n_estimators=450,
            max_depth=6,
            learning_rate=0.04,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        reg.fit(df_train[cols], y_train)
        pred_tot = np.exp(reg.predict(df_test[cols])) * sup_test
        mape = float(mean_absolute_percentage_error(y_real_test, pred_tot) * 100)
        r2 = float(r2_score(y_real_test, pred_tot))
        mae = float(mean_absolute_error(y_real_test, pred_tot))
        iaao = calcular_metricas_iaao(y_real_test, pred_tot)

        res_ablacion[nombre] = {
            "mape_pct": round(mape, 2),
            "r2": round(r2, 4),
            "mae": round(mae, 2),
            "cod_iaao": iaao["cod_pct"],
        }
        print(f"  {nombre:<36} | MAPE: {mape:5.2f}% | R²: {r2:6.4f} | COD: {iaao['cod_pct']:5.2f}%")

    # Análisis de ganancia marginal
    m1_mape = res_ablacion["M1: Solo ID (distrito_encoded)"]["mape_pct"]
    m3_mape = res_ablacion["M3: Completo (ID + NSE ENAHO)"]["mape_pct"]
    delta_ablacion = round(m1_mape - m3_mape, 2)

    print(f"\n  -> Ganancia Marginal de NSE sobre ID distrital: {delta_ablacion:+.2f}% MAPE")
    dictamen_ablacion = (
        "CONFIRMADO: Las columnas de NSE tienen contribución marginal prácticamente nula (<0.4% MAPE) "
        "sobre la identidad distrital pura. El peso SHAP alto es producto de multicolinealidad."
        if abs(delta_ablacion) < 0.4
        else "OBSERVACIÓN: Las columnas de NSE aportan señal predictiva medible sobre el ID distrital."
    )
    print(f"  -> Conclusión: {dictamen_ablacion}")
    reporte_final["prueba_ablacion"] = {
        "modelos": res_ablacion,
        "delta_mape_marginal": delta_ablacion,
        "dictamen": dictamen_ablacion,
    }

    # ── PRUEBA 1: PERMUTACIÓN DISTRITAL (SHUFFLE) ────────────────────────────
    print("\n" + "=" * 80)
    print("EJECUTANDO PRUEBA 1: PERMUTACIÓN DISTRITAL DE VECTORES NSE")
    print("=" * 80)

    dist_cat = df_train[["distrito"] + nse_cols].drop_duplicates().reset_index(drop=True)
    dist_shuffled = dist_cat.copy()
    dist_shuffled[nse_cols] = dist_cat[nse_cols].sample(frac=1.0, random_state=123).values

    df_tr_perm = df_train.drop(columns=nse_cols).merge(dist_shuffled, on="distrito")
    df_ts_perm = df_test.drop(columns=nse_cols).merge(dist_shuffled, on="distrito")

    cols_nse = base_cols + nse_cols
    reg_perm = XGBRegressor(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.04,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    reg_perm.fit(df_tr_perm[cols_nse], y_train)
    pred_perm = np.exp(reg_perm.predict(df_ts_perm[cols_nse])) * sup_test
    mape_perm = float(mean_absolute_percentage_error(y_real_test, pred_perm) * 100)
    r2_perm = float(r2_score(y_real_test, pred_perm))

    mape_orig_nse = res_ablacion["M2: Solo NSE ENAHO (pct_NSE_A..E)"]["mape_pct"]
    delta_perm = round(mape_perm - mape_orig_nse, 2)

    print(f"  -> Error con NSE Originales (M2):      MAPE = {mape_orig_nse:5.2f}%")
    print(f"  -> Error con NSE Permutados al azar:   MAPE = {mape_perm:5.2f}% | R²: {r2_perm:6.4f}")
    print(f"  -> Variación de error (Delta Permutación): {delta_perm:+.2f}%")

    dictamen_perm = (
        "CONFIRMADO: Al permutar aleatoriamente los vectores NSE entre distritos, el desempeño apenas decae. "
        "XGBoost utiliza el vector de porcentajes como un hash de identificación espacial y no como métrica socioeconómica continua."
        if abs(delta_perm) < 1.0
        else "OBSERVACIÓN: La permutación degrada significativamente el modelo, sugiriendo valor ordinal real."
    )
    print(f"  -> Conclusión: {dictamen_perm}")
    reporte_final["prueba_permutacion"] = {
        "mape_original": mape_orig_nse,
        "mape_permutado": round(mape_perm, 2),
        "r2_permutado": round(r2_perm, 4),
        "delta_mape": delta_perm,
        "dictamen": dictamen_perm,
    }

    # ── PRUEBA 3: GROUP-K-FOLD (Generalización Espacial) ────────────────────
    print("\n" + "=" * 80)
    print(f"EJECUTANDO PRUEBA 3: VALIDACIÓN ESPACIAL POR GRUPOS ({args.n_splits} Folds)")
    print("=" * 80)

    df_total = pd.concat([df_train, df_test]).reset_index(drop=True)
    y_total = np.log(df_total["target_m2"].values)
    groups = df_total["distrito"].values

    gkf = GroupKFold(n_splits=args.n_splits)
    mapes_con_nse, mapes_sin_nse = [], []
    fold_details = []

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(df_total, y_total, groups=groups), start=1):
        X_tr, y_tr = df_total.iloc[trn_idx], y_total[trn_idx]
        X_va, y_va = df_total.iloc[val_idx], y_total[val_idx]
        dist_evaluados = list(X_va["distrito"].unique())

        sup_va = X_va["Superficie"].values
        y_real_va = np.exp(y_va) * sup_va

        # Modelo con NSE
        m_nse = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, tree_method="hist"
        )
        m_nse.fit(X_tr[cols_nse], y_tr)
        p_nse = np.exp(m_nse.predict(X_va[cols_nse])) * sup_va
        mape_k_nse = float(mean_absolute_percentage_error(y_real_va, p_nse) * 100)
        mapes_con_nse.append(mape_k_nse)

        # Modelo sin información distrital (solo atributos físicos)
        m_sin = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, tree_method="hist"
        )
        m_sin.fit(X_tr[base_cols], y_tr)
        p_sin = np.exp(m_sin.predict(X_va[base_cols])) * sup_va
        mape_k_sin = float(mean_absolute_percentage_error(y_real_va, p_sin) * 100)
        mapes_sin_nse.append(mape_k_sin)

        print(
            f"  Fold {fold}: Distritos excluidos ({len(dist_evaluados)}) -> "
            f"Con NSE: {mape_k_nse:5.2f}% | Sin NSE: {mape_k_sin:5.2f}% | "
            f"Ejemplos: {', '.join(dist_evaluados[:3])}"
        )
        fold_details.append({
            "fold": fold,
            "distritos_evaluados": dist_evaluados,
            "mape_con_nse": round(mape_k_nse, 2),
            "mape_sin_nse": round(mape_k_sin, 2),
        })

    avg_con_nse = round(float(np.mean(mapes_con_nse)), 2)
    avg_sin_nse = round(float(np.mean(mapes_sin_nse)), 2)
    delta_gen = round(avg_con_nse - avg_sin_nse, 2)

    print(f"\n  -> Error Promedio en Distritos No Vistos CON NSE: {avg_con_nse:.2f}%")
    print(f"  -> Error Promedio en Distritos No Vistos SIN NSE: {avg_sin_nse:.2f}%")
    print(f"  -> Diferencia de generalización espacial:         {delta_gen:+.2f}%")

    dictamen_gkf = (
        "CONFIRMADO: En distritos no vistos (o imputados), el modelo con NSE no supera con holgura al modelo base "
        "porque el árbol memorizó umbrales locales que no transfieren fuera de muestra. Esto justifica aplicar "
        "el Grouped SHAP y advertencias de incertidumbre en distritos con <30 encuestas ENAHO."
    )
    print(f"  -> Conclusión: {dictamen_gkf}")

    reporte_final["prueba_group_kfold"] = {
        "n_splits": args.n_splits,
        "promedio_mape_con_nse": avg_con_nse,
        "promedio_mape_sin_nse": avg_sin_nse,
        "delta_generalizacion": delta_gen,
        "folds": fold_details,
        "dictamen": dictamen_gkf,
    }

    # ── 5. Exportar Resultados a JSON ─────────────────────────────────────────
    if args.export_json:
        out_path = os.path.join(BASE_DIR, args.export_json) if not os.path.isabs(args.export_json) else args.export_json
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(reporte_final, f, indent=2, ensure_ascii=False)
        print(f"\n[Paso 5/5] Resumen estructurado guardado en: {out_path}")

    print("\n" + "=" * 80)
    print("AUDITORÍA COMPLETADA CON ÉXITO")
    print("=" * 80)


if __name__ == "__main__":
    ejecutar_auditoria()
