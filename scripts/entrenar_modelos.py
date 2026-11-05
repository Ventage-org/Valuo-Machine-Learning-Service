"""
scripts/entrenar_modelos.py
───────────────────────────
Script para reentrenar los modelos XGBoost de Venta y Alquiler consumiendo
exclusivamente los datos desde Azure PostgreSQL (db_dataset_valuo),
omitiendo el servidor de MLflow (almacenamiento local de artefactos).

Uso:
    py scripts/entrenar_modelos.py --tipo all
    py scripts/entrenar_modelos.py --tipo venta
    py scripts/entrenar_modelos.py --tipo alquiler
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Fijar directorio base y ml-service
BASE_DIR = Path(__file__).resolve().parents[1]
ML_SERVICE_DIR = BASE_DIR / "ml-service"

# Cambiar CWD a ml-service para resolver models/, config/, data/ correctamente
os.chdir(ML_SERVICE_DIR)
sys.path.insert(0, str(ML_SERVICE_DIR))

# Cargar variables de entorno de Azure PostgreSQL
from dotenv import load_dotenv
load_dotenv(ML_SERVICE_DIR / ".env")

from app.core.model_loader import model_state, cargar_modelo
from app.services.entrenamiento_service import _run_training_pipeline, _job_registry


def parse_args():
    parser = argparse.ArgumentParser(description="Reentrenamiento de modelos XGBoost (Azure DB -> Local)")
    parser.add_argument(
        "--tipo",
        choices=["all", "venta", "alquiler"],
        default="all",
        help="Tipo de modelo a entrenar: 'all' (ambos), 'venta' o 'alquiler'.",
    )
    return parser.parse_args()


def entrenar_modelo(tipo: str):
    print("=" * 80)
    print(f"INICIANDO ENTRENAMIENTO DE {tipo.upper()} (Datos desde Azure PostgreSQL)")
    print("=" * 80)

    job_id = f"job-{tipo}-{int(time.time())}"
    nombre_modelo = f"xgboost_{tipo}_v2" if tipo == "venta" else f"xgboost_{tipo}_v1"

    t0 = time.time()
    _run_training_pipeline(
        job_id=job_id,
        tipo_operacion=tipo,
        nombre_modelo=nombre_modelo,
        hiperparametros=None,
        shap_top_n=10,
        state=model_state,
        guardar_como_activo=True,
        usar_target_m2=True,
    )
    duracion = time.time() - t0

    job_data = _job_registry.get(job_id, {})
    estado = job_data.get("estado")
    if estado and str(estado).lower() == "fallido":
        print(f"\n[ERROR] El entrenamiento de {tipo} falló:")
        print(job_data.get("error"))
        return False

    metricas = job_data.get("metricas")
    print(f"\n[OK] Entrenamiento de {tipo.upper()} finalizado en {duracion:.1f}s.")
    if metricas:
        print(f"  -> MAPE Test (2024-2025): {metricas.mape_pct:.2f}%")
        print(f"  -> R² Test:               {metricas.r2:.4f}")
        print(f"  -> MAE Test:              {metricas.mae:,.2f}")
        print(f"  -> RMSE Test:             {metricas.rmse:,.2f}")
        print(f"  -> COD IAAO:              {metricas.cod_pct:.2f}%")
        print(f"  -> PRD IAAO:              {metricas.prd:.4f}")
        print(f"  -> PRB IAAO:              {metricas.prb:.4f}")
        cumple_cod = getattr(metricas.iaao, "cumple_iaao_cod", None) if hasattr(metricas, "iaao") else None
        print(f"  -> Cumple IAAO COD:       {cumple_cod}")
    return True


def main():
    args = parse_args()

    tipos = ["venta", "alquiler"] if args.tipo == "all" else [args.tipo]

    for t in tipos:
        exito = entrenar_modelo(t)
        if not exito:
            print(f"\n[ABORTANDO] Falló el entrenamiento de {t}.")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("TODOS LOS ENTRENAMIENTOS HAN SIDO COMPLETADOS Y GUARDADOS COMO ACTIVOS")
    print("=" * 80)


if __name__ == "__main__":
    main()

