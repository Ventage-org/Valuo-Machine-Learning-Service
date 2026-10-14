"""
app/services/prediccion_service.py
───────────────────────────────────
Lógica pura de predicción de precios para Venta y Alquiler.

Extrae y formaliza la función `predecir()` como función sin side effects:
  - Sin prints ni gráficos
  - Sin dependencia directa de disco (recibe ModelState)
  - Retorna PrediccionVentaResponse listo para serializar

Flujo por request (< 50ms):
  1. Validar distrito → HTTPException 422 si no está en encoding_map
  2. Obtener contexto distrital via data_provider (PostgreSQL en memoria)
  3. Construir X_fila con las 33 features
  4. modelo.predict(X_fila) → pred_const = exp(pred_log) - 1
  5. explainer.shap_values(X_fila) → array[33]
  6. Seleccionar top N por |shap_val| → lista de ContribucionFeature
  7. Convertir a nominales con IPC
  8. Calcular intervalo de confianza (± MAPE%)
  9. Armar y retornar PrediccionVentaResponse
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.schemas.prediccion import (
    ContribucionFeature,
    Explicabilidad,
    InfoModelo,
    IntervaloConfianza,
    PrediccionVentaRequest,
    PrediccionVentaResponse,
    ValoresPrediccion,
)

if TYPE_CHECKING:
    from app.core.model_loader import ModelState


# ─── Etiquetas legibles para las features ─────────────────────────────────────

_FEATURE_LABELS: dict[str, str] = {
    "Superficie":                  "Superficie ({v} m²)",
    "Habitaciones":                "Habitaciones ({v})",
    "Banios":                      "Baños ({v})",
    "Garajes":                     "Cocheras ({v})",
    "Piso":                        "Piso {v}",
    "Vista_Exterior":              "Vista exterior",
    "Antiguedad":                  "Antigüedad ({v} años)",
    "Anio":                        "Año {v}",
    "Trimestre":                   "Trimestre {v}",
    "pct_NSE_A":                   "NSE A en zona ({v:.1%})",
    "pct_NSE_B":                   "NSE B en zona ({v:.1%})",
    "pct_NSE_C":                   "NSE C en zona ({v:.1%})",
    "pct_NSE_D":                   "NSE D en zona ({v:.1%})",
    "pct_NSE_E":                   "NSE E en zona ({v:.1%})",
    "tasa_robo":                   "Tasa robos zona ({v:.2f})",
    "tasa_hurto":                  "Tasa hurtos zona ({v:.2f})",
    "poblacion_proyectada":        "Población distrito ({v:,.0f} hab)",
    "area_distrito_km2":           "Área distrito ({v:.1f} km²)",
    "densidad_hab_km2":            "Densidad poblacional ({v:.0f} hab/km²)",
    "distancia_centro_km":         "Dist. al centro ({v:.1f} km)",
    "dist_colegio_km":             "Dist. al colegio más cercano ({v:.1f} km)",
    "dist_hospital_km":            "Dist. al hospital más cercano ({v:.1f} km)",
    "dist_estacion_transporte_km": "Dist. a estación de transporte ({v:.1f} km)",
    "dist_centro_comercial_km":    "Dist. a centro comercial ({v:.1f} km)",
    "dist_parque_km":              "Dist. al parque más cercano ({v:.1f} km)",
    "dist_universidad_km":         "Dist. a universidad ({v:.1f} km)",
    "periodo_numerico":            "Periodo temporal ({v})",
    "m2_por_habitacion":           "m² por habitación ({v:.1f})",
    "ratio_banios_hab":            "Ratio baños/habitaciones ({v:.2f})",
    "tiene_garaje":                "Tiene cochera",
    "es_piso_alto":                "Piso alto (≥8)",
    "superficie_cuadrado":         "Superficie² ({v:.2f})",
    "distrito_encoded":            "Distrito ({distrito})",
}


def _build_label(feature: str, valor: float, distrito: str) -> str:
    """Genera la etiqueta legible de una feature para el response SHAP."""
    template = _FEATURE_LABELS.get(feature, feature)
    try:
        if "{v" in template:
            return template.format(v=valor, distrito=distrito)
        elif "{distrito}" in template:
            return template.format(distrito=distrito)
        return template
    except (ValueError, KeyError):
        return template


# ─── Función genérica de predicción ──────────────────────────────────────────

def _predecir_generico(
    req: PrediccionVentaRequest,
    state: "ModelState",
    tipo: Literal["venta", "alquiler"],
) -> PrediccionVentaResponse:
    """Ejecuta inferencia y cálculo SHAP para venta o alquiler."""
    if tipo == "venta":
        modelo = state.modelo
        explainer = state.explainer
        encoding_map = state.encoding_map
        feature_cols = state.feature_cols
        media_global = state.media_global
        config = state.config
    else:
        modelo = state.modelo_alquiler
        explainer = state.explainer_alquiler
        encoding_map = state.encoding_map_alquiler
        feature_cols = state.feature_cols_alquiler
        media_global = state.media_global_alquiler
        config = state.config_alquiler

    if modelo is None or explainer is None:
        raise HTTPException(
            status_code=503,
            detail=f"El modelo de {tipo} no está cargado en el servicio.",
        )

    # 1. Validar distrito
    if req.distrito not in encoding_map:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "distrito_no_disponible",
                "message": (
                    f"El distrito '{req.distrito}' no está disponible en el modelo de {tipo}. "
                    f"El modelo cubre {len(encoding_map)} distritos de Lima Metropolitana."
                ),
                "distritos_disponibles": sorted(encoding_map.keys()),
                "total_distritos": len(encoding_map),
            },
        )

    # 2. Contexto distrital desde PostgreSQL
    ctx = state.data_provider.get_distrito_context(req.distrito)  # type: ignore[union-attr]

    # 3. Construir fila de features
    from datetime import datetime
    ahora = datetime.now()
    anio: int = req.anio if req.anio is not None else ahora.year
    trimestre: int = req.trimestre if req.trimestre is not None else ((ahora.month - 1) // 3 + 1)

    sup = req.superficie
    hab = req.habitaciones
    ban = req.banios
    gar = req.garajes

    fila: dict[str, float] = {
        "Anio":                        float(anio),
        "Trimestre":                   float(trimestre),
        "Superficie":                  sup,
        "Habitaciones":                float(hab),
        "Banios":                      float(ban),
        "Garajes":                     float(gar),
        "Piso":                        float(req.piso),
        "Vista_Exterior":              1.0 if req.vista_exterior else 0.0,
        "Antiguedad":                  float(req.antiguedad),
        # NSE y contexto distrital
        "pct_NSE_A":                   float(ctx.get("pct_NSE_A") or 0),
        "pct_NSE_B":                   float(ctx.get("pct_NSE_B") or 0),
        "pct_NSE_C":                   float(ctx.get("pct_NSE_C") or 0),
        "pct_NSE_D":                   float(ctx.get("pct_NSE_D") or 0),
        "pct_NSE_E":                   float(ctx.get("pct_NSE_E") or 0),
        "tasa_robo":                   float(ctx.get("tasa_robo") or 0),
        "tasa_hurto":                  float(ctx.get("tasa_hurto") or 0),
        "poblacion_proyectada":        float(ctx.get("poblacion_proyectada") or 0),
        "area_distrito_km2":           float(ctx.get("area_distrito_km2") or 0),
        "distancia_centro_km":         float(ctx.get("distancia_centro_km") or 0),
        "dist_colegio_km":             float(ctx.get("dist_colegio_km") or 0),
        "dist_hospital_km":            float(ctx.get("dist_hospital_km") or 0),
        "dist_estacion_transporte_km": float(ctx.get("dist_estacion_transporte_km") or 0),
        "dist_centro_comercial_km":    float(ctx.get("dist_centro_comercial_km") or 0),
        "dist_parque_km":              float(ctx.get("dist_parque_km") or 0),
        "dist_universidad_km":         float(ctx.get("dist_universidad_km") or 0),
        "densidad_hab_km2":            float(ctx.get("densidad_hab_km2") or 0),
        # Features engineered
        "periodo_numerico":            float(anio * 4 + trimestre),
        "m2_por_habitacion":           sup / (hab + 1),
        "ratio_banios_hab":            ban / (hab + 0.1),
        "tiene_garaje":                1.0 if gar > 0 else 0.0,
        "es_piso_alto":                1.0 if req.piso >= 8 else 0.0,
        "superficie_cuadrado":         (sup / 100) ** 2,
        "distrito_encoded":            encoding_map[req.distrito],
    }

    X = pd.DataFrame([fila])[feature_cols]

    # 4. Predicción (soporte target_m2 y calibración estratificada IAAO)
    target_tipo = config.get("target_tipo", "total")
    factores_distrito = config.get("factores_calibracion_distrito", {})
    factor_calibracion = float(factores_distrito.get(req.distrito, config.get("factor_calibracion", 1.0)))

    try:
        pred_log = float(modelo.predict(X)[0])
        if target_tipo == "m2":
            pred_m2_const = float(np.exp(pred_log) / factor_calibracion)
            pred_const = float(pred_m2_const * sup)
        else:
            pred_const = float(np.expm1(pred_log) / factor_calibracion)
            pred_m2_const = float(pred_const / sup)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la inferencia de {tipo}: {exc}",
        ) from exc

    # 5. SHAP values
    try:
        shap_vals_raw = explainer.shap_values(X)
        shap_arr: np.ndarray = (
            shap_vals_raw[0] if shap_vals_raw.ndim == 2 else shap_vals_raw
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculando SHAP values de {tipo}: {exc}",
        ) from exc

    # 6. SHAP contribuciones: Todas las features del modelo ordenadas por magnitud
    feature_names = feature_cols
    fila_vals = X.iloc[0].to_dict()

    indexed = sorted(
        enumerate(shap_arr),
        key=lambda t: abs(t[1]),
        reverse=True,
    )

    total_abs_shap = sum(abs(float(sv)) for sv in shap_arr)
    if total_abs_shap < 1e-9:
        total_abs_shap = 1.0

    contribuciones: list[ContribucionFeature] = []
    for idx, sv in indexed:
        fname = feature_names[idx]
        fval  = float(fila_vals.get(fname, 0.0))
        abs_sv = abs(float(sv))
        p_pct = round((abs_sv / total_abs_shap) * 100.0, 1)

        if p_pct >= 20.0:
            nivel = "Impacto Muy Alto"
        elif p_pct >= 10.0:
            nivel = "Impacto Alto"
        elif p_pct >= 5.0:
            nivel = "Impacto Moderado"
        else:
            nivel = "Impacto Leve"

        contribuciones.append(
            ContribucionFeature(
                feature=fname,
                label=_build_label(fname, fval, req.distrito),
                shap_value=round(float(sv), 6),
                valor_feature=round(fval, 4),
                impacto=(
                    "positivo" if sv > 1e-6
                    else "negativo" if sv < -1e-6
                    else "neutro"
                ),
                peso_porcentaje=p_pct,
                nivel_impacto=nivel,
            )
        )

    # Ajuste de redondeo para que la suma total sea exactamente 100.0%
    if contribuciones:
        diff_total = round(100.0 - sum(c.peso_porcentaje for c in contribuciones), 1)
        if abs(diff_total) > 0:
            contribuciones[0].peso_porcentaje = round(contribuciones[0].peso_porcentaje + diff_total, 1)

    # 7. Conversión a soles nominales
    ipc_cfg = config.get("ipc_actual", {"valor": 169.18, "periodo": "2026-Q1"})
    ipc_val = float(ipc_cfg["valor"])
    ipc_factor = ipc_val / 100.0

    pred_nominal    = pred_const * ipc_factor
    pred_m2_const   = pred_const / sup
    pred_m2_nominal = pred_nominal / sup

    # 8. Intervalo de confianza (± MAPE)
    mape = float(config.get("mape_test", 15.0))
    ic_factor = mape / 100.0

    ic_inf_const   = pred_const   * (1 - ic_factor)
    ic_sup_const   = pred_const   * (1 + ic_factor)
    ic_inf_nominal = pred_nominal * (1 - ic_factor)
    ic_sup_nominal = pred_nominal * (1 + ic_factor)

    # 9. Valor base SHAP
    base_log = float(explainer.expected_value)
    if target_tipo == "m2":
        base_m2_const = float(math.exp(base_log) / factor_calibracion) if not math.isnan(base_log) else media_global
        base_const = float(base_m2_const * sup)
    else:
        base_const = float(math.expm1(base_log) / factor_calibracion) if not math.isnan(base_log) else media_global

    # 10. Armar response
    return PrediccionVentaResponse(
        distrito=req.distrito,
        superficie_m2=sup,
        prediccion=ValoresPrediccion(
            soles_constantes=round(pred_const, 2),
            soles_nominales=round(pred_nominal, 2),
            precio_m2_constantes=round(pred_m2_const, 2),
            precio_m2_nominales=round(pred_m2_nominal, 2),
        ),
        intervalo_confianza=IntervaloConfianza(
            inferior_nominales=round(ic_inf_nominal, 2),
            superior_nominales=round(ic_sup_nominal, 2),
            inferior_constantes=round(ic_inf_const, 2),
            superior_constantes=round(ic_sup_const, 2),
            mape_pct=mape,
        ),
        explicabilidad=Explicabilidad(
            valor_base_constantes=round(base_const, 2),
            contribuciones=contribuciones,
            n_features_mostradas=len(contribuciones),
            n_features_total=len(feature_names),
        ),
        modelo=InfoModelo(
            version=config.get("modelo_version", "v1"),
            mape_test=mape,
            r2_test=float(config.get("r2_test", 0.0)),
            ipc_factor=ipc_val,
            periodo_ipc=ipc_cfg.get("periodo", "2026-Q1"),
            train_periodo=config.get("train_periodo", "2016-2023"),
            test_periodo=config.get("test_periodo", "2024-2025"),
        ),
    )


def predecir_venta(
    req: PrediccionVentaRequest,
    state: "ModelState",
) -> PrediccionVentaResponse:
    """Predice el precio de venta de un inmueble."""
    return _predecir_generico(req, state, tipo="venta")


def predecir_alquiler(
    req: PrediccionVentaRequest,
    state: "ModelState",
) -> PrediccionVentaResponse:
    """Predice el precio de alquiler mensual de un inmueble."""
    return _predecir_generico(req, state, tipo="alquiler")
