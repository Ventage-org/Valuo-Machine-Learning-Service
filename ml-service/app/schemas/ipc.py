"""
app/schemas/ipc.py
──────────────────
Esquemas Pydantic para consulta y actualización dinámica del factor IPC
(Índice de Precios al Consumidor de Lima Metropolitana - Base Dic 2009 = 100).
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ActualizarIPCRequest(BaseModel):
    valor: float = Field(
        ...,
        gt=50.0,
        lt=500.0,
        description="Nuevo valor del IPC base Dic-2009=100 (ej. 169.18 o 170.25).",
        examples=[169.18],
    )
    periodo: str = Field(
        ...,
        min_length=4,
        max_length=25,
        description="Periodo oficial correspondiente (ej. '2026-Q1', '2026-Q2', '2026-06').",
        examples=["2026-Q2"],
    )
    fuente: str = Field(
        default="INEI/BCRP — serie PN01270PM, Base Dic 2009 = 100",
        description="Fuente oficial del dato macroeconómico.",
    )
    actualizado_por: str = Field(
        default="Administrador",
        description="Usuario, rol o sistema que ejecuta la actualización.",
    )
    nota: str | None = Field(
        default=None,
        description="Comentario o justificación de la actualización.",
    )
    aplicar_a: Literal["ambos", "venta", "alquiler"] = Field(
        default="ambos",
        description="Modelos a los que se aplica el nuevo factor IPC ('ambos', 'venta' o 'alquiler').",
    )


class IPCDetalle(BaseModel):
    valor: float = Field(..., description="Valor del IPC base Dic-2009=100")
    factor: float = Field(..., description="Factor multiplicador aplicado a Soles Constantes (valor / 100)")
    periodo: str = Field(..., description="Periodo oficial vigente")
    fuente: str = Field(..., description="Fuente de los datos")
    nota: str | None = Field(default=None, description="Nota o comentario")
    fecha_actualizacion: str = Field(..., description="Marca de tiempo de la última actualización")


class ActualizarIPCResponse(BaseModel):
    exito: bool = Field(default=True, description="Indica si la actualización se aplicó con éxito")
    mensaje: str = Field(..., description="Resumen descriptivo del cambio aplicado")
    ipc_previo: dict[str, Any] = Field(..., description="Estado del IPC antes de la actualización")
    ipc_nuevo: IPCDetalle = Field(..., description="Nuevo estado del IPC en vigor")
    modelos_actualizados: list[str] = Field(..., description="Lista de modelos actualizados en RAM y disco")
    recargado_en_ram: bool = Field(default=True, description="Confirmación de actualización en caliente en RAM")
    persistencia_disco: bool = Field(default=True, description="Confirmación de persistencia en archivos JSON y changelog")


class ConsultarIPCResponse(BaseModel):
    ipc_vigente: IPCDetalle = Field(..., description="IPC actualmente activo para inferencias nominales")
    modelo_venta: dict[str, Any] = Field(..., description="Configuración IPC específica del modelo de venta")
    modelo_alquiler: dict[str, Any] = Field(..., description="Configuración IPC específica del modelo de alquiler")
    explicacion_macroeconomica: str = Field(..., description="Fundamentación econométrica del uso de Soles Constantes")
