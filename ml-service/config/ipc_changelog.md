# ipc_changelog.md
# Registro de actualizaciones del factor IPC para el sistema de inferencia
# Mantener este archivo actualizado mensualmente (ver protocolo en DOCUMENTACION_MODELO_TESIS.md Sec. 6.6.E)

| Fecha actualización | Período IPC | Valor IPC | Factor (IPC/100) | Fuente | Actualizado por |
|---|---|---|---|---|---|
| 2026-08-01 | Q1-2026 | 169.1771184 → 169.18 | 1.6918 | BCRP PN01270PM (Excel oficial) | Joaquín Cortez |

---
## Cómo actualizar

1. Consultar el nuevo valor en: https://estadisticas.bcrp.gob.pe/estadisticas/series/api/PN01270PM
2. Editar config/model_config.json:
   - Cambiar ipc_lima.valor y ipc_lima.valor_exacto
   - Recalcular ipc_lima.factor = valor / 100
   - Actualizar ipc_lima.periodo y ipc_lima.fecha_actualizacion_config
3. Agregar una fila a este changelog.
4. Ejecutar 
otebooks/07_validacion_prediccion.py para verificar coherencia de predicciones.
5. Hacer commit con mensaje: config: actualizar IPC [PERIODO] = [VALOR]

## Notas metodológicas

- El MAPE del modelo (14.88%) NO cambia al actualizar el IPC — fue calculado en soles constantes.
- El IPC actualizado solo afecta el precio mostrado al usuario final (soles nominales).
- Para reproducibilidad de la tesis: el valor canónico es 169.18 (Q1-2026). Ver Sec. 6.6.E.4 de DOCUMENTACION_MODELO_TESIS.md.
| 2026-10-13 | 2026-Q2 | 169.18 → 169.5 | 1.695 | INEI/BCRP — serie PN01270PM, Base Dic 2009 = 100 | Joaquin Cortez |
| 2026-10-13 | 2026-Q1 | 169.5 → 169.18 | 1.6918 | INEI/BCRP — serie PN01270PM, Base Dic 2009 = 100 | Joaquin Cortez |
