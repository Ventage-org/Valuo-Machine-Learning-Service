"""
app/routers/configuracion.py
────────────────────────────
Router administrativo de configuración dinámica del sistema:
  - Consulta del factor macroeconómico IPC vigente
  - Actualización en caliente (Hot-Reload) del IPC en memoria RAM y persistencia en disco
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.core.model_loader import model_state
from app.schemas.ipc import (
    ActualizarIPCRequest,
    ActualizarIPCResponse,
    ConsultarIPCResponse,
)
from app.services.ipc_service import (
    actualizar_ipc_en_caliente,
    consultar_ipc_vigente,
)

router = APIRouter(
    prefix="/api/v1/admin/config",
    tags=["Administración — Configuración y Parámetros"],
)


def _verificar_admin_token(x_admin_token: str | None = Header(None)) -> None:
    """Valida el token de administración si está configurado en .env."""
    expected = os.getenv("ADMIN_TOKEN", "").strip()
    if not expected:
        return
    if x_admin_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "token_invalido",
                "message": "El token de administrador es inválido.",
            },
        )


@router.get(
    "/ipc",
    response_model=ConsultarIPCResponse,
    summary="Consultar factor IPC vigente",
    description=(
        "Retorna la configuración macroeconómica del factor IPC (Índice de Precios al Consumidor "
        "de Lima Metropolitana, Base Dic 2009 = 100) actualmente utilizado para reconvertir las "
        "predicciones en Soles Constantes a Soles Nominales vigentes."
    ),
)
def get_ipc_config(
    _: None = Depends(_verificar_admin_token),
) -> ConsultarIPCResponse:
    return consultar_ipc_vigente(state=model_state)


@router.put(
    "/ipc",
    response_model=ActualizarIPCResponse,
    summary="Actualizar factor IPC en caliente (Zero Downtime)",
    description=(
        "Actualiza el valor del IPC (ej. tras una nueva publicación trimestral del BCRP/INEI):\n\n"
        "1. Aplica el nuevo factor en **memoria RAM** de manera inmediata sin reiniciar el servidor.\n"
        "2. Persiste la configuración en los archivos `model_config.json` y `model_alquiler_config.json`.\n"
        "3. Registra una nueva entrada en la bitácora `config/ipc_changelog.md`.\n\n"
        "A partir de este instante, todas las nuevas predicciones nominales utilizarán el nuevo factor."
    ),
)
def update_ipc_config(
    request: ActualizarIPCRequest,
    _: None = Depends(_verificar_admin_token),
) -> ActualizarIPCResponse:
    return actualizar_ipc_en_caliente(req=request, state=model_state)
