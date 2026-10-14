"""
app/services/ipc_service.py
───────────────────────────
Lógica de negocio para consulta y actualización en caliente del factor IPC.
Actualiza el estado en memoria RAM (Zero Downtime) y persiste en disco.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.schemas.ipc import (
    ActualizarIPCRequest,
    ActualizarIPCResponse,
    ConsultarIPCResponse,
    IPCDetalle,
)

if TYPE_CHECKING:
    from app.core.model_loader import ModelState


def _obtener_rutas_configuracion() -> dict[str, list[Path]]:
    """Obtiene las rutas en ml-service y en la raíz del repositorio."""
    try:
        base_service = Path(__file__).resolve().parents[2]  # <repo>/ml-service
        base_repo = base_service.parent                    # <repo>
    except Exception:
        base_service = Path.cwd()
        base_repo = base_service.parent

    return {
        "venta": [
            base_service / "config" / "model_config.json",
            base_repo / "config" / "model_config.json",
        ],
        "alquiler": [
            base_service / "config" / "model_alquiler_config.json",
            base_repo / "config" / "model_alquiler_config.json",
        ],
        "changelog": [
            base_service / "config" / "ipc_changelog.md",
            base_repo / "config" / "ipc_changelog.md",
        ],
    }


def consultar_ipc_vigente(state: "ModelState") -> ConsultarIPCResponse:
    """Retorna el estado actual del factor IPC para venta y alquiler."""
    cfg_venta = state.config.get("ipc_actual", {})
    cfg_alquiler = state.config_alquiler.get("ipc_actual", {})

    # Priorizar venta o alquiler
    cfg_principal = cfg_venta or cfg_alquiler or {
        "valor": 169.18,
        "periodo": "2026-Q1",
        "fuente": "INEI/BCRP — serie PN01270PM, Base Dic 2009 = 100",
    }

    valor = float(cfg_principal.get("valor", 169.18))
    factor = round(valor / 100.0, 4)
    periodo = str(cfg_principal.get("periodo", "2026-Q1"))
    fuente = str(cfg_principal.get("fuente", "INEI/BCRP"))
    nota = cfg_principal.get("nota")
    fecha = cfg_principal.get("fecha_actualizacion", "2026-08-01T00:00:00Z")

    ipc_vigente = IPCDetalle(
        valor=valor,
        factor=factor,
        periodo=periodo,
        fuente=fuente,
        nota=nota,
        fecha_actualizacion=fecha,
    )

    return ConsultarIPCResponse(
        ipc_vigente=ipc_vigente,
        modelo_venta=cfg_venta,
        modelo_alquiler=cfg_alquiler,
        explicacion_macroeconomica=(
            "Los modelos predictivos (XGBoost) entrenan y evalúan sobre Soles Constantes (Base Dic-2009=100) "
            "para aislar la inflación de la plusvalía física y distrital. El factor IPC (valor/100) convierte "
            "dicho valor intrínseco en Soles Nominales vigentes para el usuario final."
        ),
    )


def actualizar_ipc_en_caliente(
    req: ActualizarIPCRequest,
    state: "ModelState",
) -> ActualizarIPCResponse:
    """
    Aplica el nuevo factor IPC en memoria RAM y persiste en los archivos JSON y changelog.
    """
    ahora_iso = datetime.now().isoformat()
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")

    rutas = _obtener_rutas_configuracion()
    modelos_actualizados: list[str] = []

    nuevo_bloque_ipc = {
        "valor": round(req.valor, 4),
        "periodo": req.periodo,
        "fuente": req.fuente,
        "nota": req.nota or "Actualizado vía endpoint administrativo /api/v1/admin/config/ipc",
        "fecha_actualizacion": ahora_iso,
        "actualizado_por": req.actualizado_por,
    }

    ipc_previo = dict(state.config.get("ipc_actual", {}))

    # 1. Hot-Reload en memoria RAM
    if req.aplicar_a in ("ambos", "venta"):
        state.config["ipc_actual"] = nuevo_bloque_ipc
        modelos_actualizados.append("venta")

    if req.aplicar_a in ("ambos", "alquiler"):
        state.config_alquiler["ipc_actual"] = nuevo_bloque_ipc
        modelos_actualizados.append("alquiler")

    # 2. Persistir en disco (model_config.json y model_alquiler_config.json)
    persistencia_ok = True
    try:
        if "venta" in modelos_actualizados:
            for ruta in rutas["venta"]:
                if ruta.exists():
                    try:
                        with open(ruta, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        data["ipc_actual"] = nuevo_bloque_ipc
                        with open(ruta, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    except Exception as ex_v:
                        print(f"[IPCService] Error persistiendo venta en {ruta}: {ex_v}")
                        persistencia_ok = False

        if "alquiler" in modelos_actualizados:
            for ruta in rutas["alquiler"]:
                if ruta.exists():
                    try:
                        with open(ruta, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        data["ipc_actual"] = nuevo_bloque_ipc
                        with open(ruta, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    except Exception as ex_a:
                        print(f"[IPCService] Error persistiendo alquiler en {ruta}: {ex_a}")
                        persistencia_ok = False

        # 3. Registrar en ipc_changelog.md
        factor_num = round(req.valor / 100.0, 4)
        valor_prev_str = str(ipc_previo.get("valor", "169.18"))
        linea_log = (
            f"| {fecha_hoy} | {req.periodo} | {valor_prev_str} → {req.valor} | "
            f"{factor_num} | {req.fuente} | {req.actualizado_por} |\n"
        )

        for ruta_cl in rutas["changelog"]:
            if ruta_cl.exists():
                try:
                    with open(ruta_cl, "a", encoding="utf-8") as f:
                        f.write(linea_log)
                except Exception as ex_cl:
                    print(f"[IPCService] Error registrando en changelog {ruta_cl}: {ex_cl}")

    except Exception as exc:
        print(f"[IPCService] Error general persistiendo IPC: {exc}")
        persistencia_ok = False

    ipc_nuevo_detalle = IPCDetalle(
        valor=req.valor,
        factor=round(req.valor / 100.0, 4),
        periodo=req.periodo,
        fuente=req.fuente,
        nota=nuevo_bloque_ipc["nota"],
        fecha_actualizacion=ahora_iso,
    )

    return ActualizarIPCResponse(
        exito=True,
        mensaje=(
            f"Factor IPC actualizado exitosamente a {req.valor} ({req.periodo}) para: "
            f"{', '.join(modelos_actualizados)}. Inferencia nominal actualizada en caliente."
        ),
        ipc_previo=ipc_previo,
        ipc_nuevo=ipc_nuevo_detalle,
        modelos_actualizados=modelos_actualizados,
        recargado_en_ram=True,
        persistencia_disco=persistencia_ok,
    )
