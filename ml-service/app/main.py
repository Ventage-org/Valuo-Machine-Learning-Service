"""
app/main.py
───────────
Entry point del ML Service.

Usa el patrón `lifespan` de FastAPI (recomendado desde v0.93) para:
  - Cargar el modelo + TreeExplainer SHAP al arrancar (startup)
  - Liberar recursos al cerrar (shutdown)

Endpoints registrados:
  GET  /health
  GET  /api/v1/modelo/info
  GET  /api/v1/distritos
  POST /api/v1/prediccion/venta
  POST /api/v1/admin/entrenamiento/venta   (requiere X-Admin-Token)
  GET  /api/v1/admin/entrenamiento/estado/{job_id}

Documentación automática:
  GET  /docs      → Swagger UI
  GET  /redoc     → ReDoc
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Cargar .env antes de cualquier otra importación que lea variables de entorno
load_dotenv(find_dotenv(usecwd=True))
load_dotenv()

from app.core.model_loader import cargar_modelo, liberar_recursos, model_state
from app.routers import configuracion as configuracion_router
from app.routers import entrenamiento as entrenamiento_router
from app.routers import health as health_router
from app.routers import modelos as modelos_router
from app.routers import monitoreo as monitoreo_router
from app.routers import prediccion as prediccion_router
from app.routers import sensibilidad as sensibilidad_router



# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: carga modelo, SHAP explainer y data provider.
    Shutdown: libera recursos.
    """
    print("\n" + "=" * 55)
    print("  ML Service - Tasacion Inmobiliaria Lima")
    print("  Iniciando carga de artefactos ...")
    print("=" * 55)

    cargar_modelo(model_state)

    print("=" * 55)
    print("  Servicio listo. Accede a /docs para la API.")
    print("=" * 55 + "\n")

    yield  # ← El servidor está activo aquí

    print("\n[ML Service] Cerrando servicio ...")
    liberar_recursos(model_state)


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Service — Tasación Inmobiliaria Lima",
    version="1.0.0",
    description=(
        "## Servicio de predicción de precio de venta de inmuebles\n\n"
        "Modelo: **XGBoost v2** entrenado con datos BCRP 2016–2023.\n"
        "Test set: 2024–2025 | **MAPE: 15.04%** | **R²: 0.7469**\n\n"
        "### Características\n"
        "- Predicción en **Soles Constantes** y **Soles Nominales** (IPC vigente)\n"
        "- Intervalo de confianza ± MAPE sobre el precio predicho\n"
        "- **Valores SHAP** por predicción: top 10 variables por impacto\n"
        "- 22 distritos de Lima Metropolitana disponibles\n\n"
        "### Fuente de datos\n"
        "Fuente única: **PostgreSQL** (`inmobiliaria_ml_db`).\n"
        "- Inferencia: contexto distrital precargado al startup desde `distrito_anio_contexto`.\n"
        "- Reentrenamiento: dataset histórico vía JOIN contextual.\n\n"
        "### Administración\n"
        "- `POST /api/v1/admin/entrenamiento/venta` — Reentrenamiento desde PostgreSQL (requiere `X-Admin-Token`)\n"
        "- `GET /api/v1/admin/entrenamiento/estado/{job_id}` — Progreso del job\n\n"
        "### Roadmap\n"
        "- Endpoint `POST /api/v1/prediccion/alquiler` con modelo de alquiler\n"
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─── CORS ─────────────────────────────────────────────────────────────────────

_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins: list[str] = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(health_router.router)
app.include_router(prediccion_router.router)
app.include_router(entrenamiento_router.router)
app.include_router(modelos_router.router)
app.include_router(monitoreo_router.router)
app.include_router(configuracion_router.router)
app.include_router(sensibilidad_router.router)
