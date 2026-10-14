# Documentación Técnica: Reentrenamiento Asíncrono de Modelos de Venta y Alquiler

## 1. Resumen Ejecutivo

Esta funcionalidad permite que el **Administrador del Sistema** solicite el reentrenamiento bajo demanda de los modelos predictivos de Machine Learning (**Venta** y **Alquiler**) directamente a través de la API REST del microservicio (`ml-service`).

El proceso es **100% asíncrono y no bloqueante**:
- Extrae la información histórica directamente desde la base de datos **PostgreSQL** (`inmobiliaria_ml_db`).
- Se ejecuta en segundo plano (`BackgroundTasks` de FastAPI), devolviendo de inmediato un `job_id`.
- Permite monitorizar el avance paso a paso (1 a 8) mediante un endpoint de estado.
- Al culminar exitosamente, exporta los nuevos artefactos (`.pkl`, métricas, metadata y configuraciones) y realiza un **Hot-Reload en caliente en memoria**, asegurando disponibilidad continua (cero tiempo de inactividad) para las predicciones.

---

## 2. Arquitectura del Flujo de Reentrenamiento

```
  +---------------------------------------------------------------+
  |                     Administrador / Frontend                  |
  +-------------------------------+-------------------------------+
                                  |
            POST /api/v1/admin/entrenamiento/{venta|alquiler}
            (Header: X-Admin-Token)
                                  v
  +---------------------------------------------------------------+
  |                   FastAPI: Router Entrenamiento                |
  |  - Valida X-Admin-Token (si está configurado)                |
  |  - Valida hiperparámetros opcionales                          |
  |  - Registra Job con estado PENDING y retorna job_id de inmediato |
  |  - Despacha BackgroundTask                                    |
  +-------------------------------+-------------------------------+
                                  | (En segundo plano)
                                  v
  +---------------------------------------------------------------+
  |                 EntrenamientoService (Background)             |
  |                                                               |
  |  [Paso 1] db_provider.py -> Conexión y consulta a PostgreSQL  |
  |           (dataset_inmuebles_venta o alquiler)                |
  |                                                               |
  |  [Paso 2] Feature Engineering & Target Encoding               |
  |           - Reconstruye encoding de los 22 distritos          |
  |           - Tratamiento logarítmico (log1p)                   |
  |           - En alquiler: ponderación temporal E1              |
  |                                                               |
  |  [Paso 3] Split Temporal Train/Test                           |
  |           - Train: 2016 - 2023                                |
  |           - Test:  2024 - 2025                                |
  |                                                               |
  |  [Paso 4] Entrenamiento de XGBRegressor                       |
  |           - Algoritmo hist, regularizaciones L1/L2            |
  |                                                               |
  |  [Paso 5] Evaluación contra benchmarks metodológicos          |
  |           - MAPE, R², MAE, RMSE                               |
  |                                                               |
  |  [Paso 6] Validación de SHAP TreeExplainer                    |
  |                                                               |
  |  [Paso 7] Exportación de artefactos                           |
  |           - models/xgboost_{operacion}_{version}.pkl          |
  |           - models/*_params.json y *_metricas.json            |
  |           - config/model_config.json o alquiler_config.json   |
  |           - data/features_metadata.json                       |
  |                                                               |
  |  [Paso 8] Hot-Reload en memoria                               |
  |           - model_loader.cargar_modelo(state, tipo=...)       |
  |           - Actualiza singleton model_state sin reiniciar app |
  +---------------------------------------------------------------+
```

---

## 3. Endpoints Disponibles

Todos los endpoints administrativos están bajo el prefijo `/api/v1/admin/entrenamiento`.

### 3.1. Reentrenar Modelo de Venta

- **Método**: `POST`
- **Ruta**: `/api/v1/admin/entrenamiento/venta`
- **Headers**:
  - `Content-Type: application/json`
  - `X-Admin-Token: <token>` *(requerido si `ADMIN_TOKEN` está configurado en `.env`)*
- **Body (Opcional)**:
```json
{
  "nombre_modelo": "xgboost_venta_v2",
  "max_depth": 6,
  "learning_rate": 0.040,
  "n_estimators": 650,
  "min_child_weight": 9,
  "usar_target_m2": true,
  "factor_calibracion": 1.030,
  "hiperparametros": {
    "subsample": 0.852,
    "colsample_bytree": 0.847,
    "reg_alpha": 4.06,
    "reg_lambda": 0.0306
  },
  "guardar_como_activo": true
}
```
*Si no se envía body o se envían campos vacíos, se usan los hiperparámetros estándar validados para venta con Target $m^2$ y calibración IAAO (MAPE = 14.18%).*

- **Respuesta (202 Accepted)**:
```json
{
  "job_id": "7a30cf7f-bf83-4a11-8fcb-c12e84da63e6",
  "estado": "pendiente",
  "nombre_modelo": "xgboost_venta_v2",
  "tipo_operacion": "venta",
  "mensaje": "Job de reentrenamiento de venta 'xgboost_venta_v2' iniciado. El pipeline corre en background y registrará el modelo en MLflow.",
  "consultar_estado_en": "/api/v1/admin/entrenamiento/estado/7a30cf7f-bf83-4a11-8fcb-c12e84da63e6",
  "iniciado_en": "2026-10-06T14:41:10Z"
}
```

---

### 3.2. Reentrenar Modelo de Alquiler

- **Método**: `POST`
- **Ruta**: `/api/v1/admin/entrenamiento/alquiler`
- **Headers**:
  - `Content-Type: application/json`
  - `X-Admin-Token: <token>`
- **Body (Opcional)**:
```json
{
  "nombre_modelo": "xgboost_alquiler_v1",
  "max_depth": 6,
  "learning_rate": 0.035,
  "n_estimators": 600,
  "min_child_weight": 4,
  "usar_target_m2": true,
  "factor_calibracion": 1.015,
  "hiperparametros": {
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.1,
    "reg_lambda": 4.0
  },
  "guardar_como_activo": true
}
```
*Aplica Target $m^2$ y factor de calibración de 1.015 obteniendo MAPE = 13.59%.*

- **Respuesta (202 Accepted)**:
```json
{
  "job_id": "89fb6964-b0cf-46d5-a836-e8d9e7adbe09",
  "estado": "pendiente",
  "nombre_modelo": "xgboost_alquiler_v1",
  "tipo_operacion": "alquiler",
  "mensaje": "Job de reentrenamiento de alquiler 'xgboost_alquiler_v1' iniciado. El pipeline corre en background y registrará el modelo en MLflow.",
  "consultar_estado_en": "/api/v1/admin/entrenamiento/estado/89fb6964-b0cf-46d5-a836-e8d9e7adbe09",
  "iniciado_en": "2026-10-06T14:42:44Z"
}
```

---

### 3.3. Consultar Estado de un Job

- **Método**: `GET`
- **Ruta**: `/api/v1/admin/entrenamiento/estado/{job_id}`
- **Headers**:
  - `X-Admin-Token: <token>`

- **Respuesta mientras está en ejecución (200 OK)**:
```json
{
  "job_id": "89fb6964-b0cf-46d5-a836-e8d9e7adbe09",
  "estado": "running",
  "progreso": "[4/8] Entrenando modelo XGBoost para alquiler...",
  "iniciado_en": "2026-09-30T17:31:01Z",
  "completado_en": null,
  "duracion_segundos": null,
  "error": null,
  "metricas": null,
  "artefactos": null
}
```

- **Respuesta cuando culmina exitosamente (200 OK)**:
```json
{
  "job_id": "89fb6964-b0cf-46d5-a836-e8d9e7adbe09",
  "estado": "completed",
  "progreso": "Completado exitosamente en 18.42s",
  "iniciado_en": "2026-09-30T17:31:01Z",
  "completado_en": "2026-09-30T17:31:20Z",
  "duracion_segundos": 18.42,
  "error": null,
  "metricas": {
    "mape_pct": 13.85,
    "r2": 0.8124,
    "mae": 412.35,
    "rmse": 625.10,
    "n_train": 41921,
    "n_test": 19682,
    "benchmark_mape": 17.89,
    "supera_benchmark": true
  },
  "artefactos": {
    "modelo_pkl": "models/xgboost_alquiler_v2.pkl",
    "params_json": "models/xgboost_alquiler_v2_params.json",
    "metricas_json": "models/xgboost_alquiler_v2_metricas.json",
    "model_config_json": "config/model_alquiler_config.json",
    "modelo_recargado_en_memoria": true
  }
}
```

---

### 3.4. Listar Todos los Jobs de la Sesión

- **Método**: `GET`
- **Ruta**: `/api/v1/admin/entrenamiento/jobs`
- **Headers**:
  - `X-Admin-Token: <token>`

---

### 3.5. Benchmark Comparativo (Producción vs Candidatos)

Permite al frontend renderizar la comparativa métrica lado a lado para la toma de decisión del administrador:

- **Método**: `GET`
- **Ruta**: `/api/v1/admin/modelos/benchmark/{tipo_operacion}` *(venta | alquiler)*
- **Headers**:
  - `X-Admin-Token: <token>`

- **Respuesta (200 OK)**:
```json
{
  "tipo_operacion": "venta",
  "benchmark_referencial_mape": 15.0,
  "modelo_produccion": {
    "nombre_modelo": "xgboost_venta",
    "version": "1",
    "stage": "Production",
    "metricas": {
      "mape_pct": 15.04,
      "r2": 0.7469,
      "mae": 52165.93,
      "rmse": 78170.0
    }
  },
  "candidatos": [
    {
      "candidato": {
        "nombre_modelo": "xgboost_venta",
        "version": "2",
        "stage": "Candidate",
        "metricas": {
          "mape_pct": 13.85,
          "r2": 0.8124,
          "mae": 47520.1,
          "rmse": 71200.5
        }
      },
      "deltas": {
        "delta_mape_pct": -1.19,
        "delta_r2": 0.0655,
        "delta_mae": -4645.83,
        "delta_rmse": -6969.5,
        "mejora_mape": true,
        "mejora_r2": true
      },
      "recomendacion": "RECOMENDADO_PARA_PRODUCCION",
      "motivo": "El candidato mejora el desempeño frente a producción (ΔMAPE: -1.19%, ΔR²: +0.0655) y supera el benchmark exigido (15.0%)."
    }
  ],
  "total_candidatos": 1
}
```

---

### 3.6. Historial de Versiones en Model Registry

- **Método**: `GET`
- **Ruta**: `/api/v1/admin/modelos/historial/{tipo_operacion}`
- **Headers**:
  - `X-Admin-Token: <token>`

---

### 3.7. Promover Modelo Candidato a Producción (Gobernanza Humana)

Acción humana de aprobación: el administrador confirma el pase a producción del modelo candidato.
1. MLflow actualiza la versión a `Production` y archiva la anterior.
2. El microservicio descarga el modelo y lo carga directamente en **memoria RAM**.
3. Reconstruye el `TreeExplainer` de SHAP en RAM con cero downtime.

- **Método**: `POST`
- **Ruta**: `/api/v1/admin/modelos/promover`
- **Headers**:
  - `Content-Type: application/json`
  - `X-Admin-Token: <token>`
- **Body**:
```json
{
  "tipo_operacion": "venta",
  "version": 2,
  "motivo": "Aprobación tras verificar mejora de MAPE de 15.04% a 13.85% en el benchmark"
}
```

- **Respuesta (200 OK)**:
```json
{
  "exito": true,
  "mensaje": "Modelo xgboost_venta v2 promovido exitosamente a Producción y cargado en memoria RAM.",
  "tipo_operacion": "venta",
  "version_promovida": "2",
  "version_anterior_archivada": "1",
  "modelo_cargado_en_ram": true,
  "promovido_en": "2026-10-01T05:00:00Z"
}
```

---

## 4. Detalles Metodológicos Implementados

### 4.1. Extracción de Datos
- Las consultas SQL se ejecutan a través de [db_provider.py](file:///d:/NuevaCarpetaLool/python_modelo_tesis/ml-service/app/core/db_provider.py).
- Para **Venta**:
  - `dataset_inmuebles_venta` unida con `distrito_anio_contexto`.
- Para **Alquiler**:
  - `dataset_inmuebles_alquiler` unida con `distrito_anio_contexto`.

### 4.2. Ponderación Temporal en Alquiler (Estrategia E1)
Siguiendo los hallazgos de investigación y la guía de tesis, el mercado de alquiler muestra mayor sensibilidad a la dinámica post-pandemia. Por ello, el pipeline de reentrenamiento de alquiler aplica pesos muestrales exponenciales:
$$\text{sample\_weight} = 0.85^{(2023 - \text{año})}$$
Esto pondera con mayor relevancia los años recientes sin descartar el historial 2016-2022.

### 4.3. Target Encoding y Manejo de Logaritmo
- Se calcula la media del target transformado $\log(1 + y)$ por distrito sobre el conjunto de entrenamiento (2016-2023).
- Las categorías no vistas en inferencia adoptan la media global (`media_global_target`).

### 4.4. Hot-Reload sin Downtime
Al finalizar el paso 8, `cargar_modelo(state, tipo=operacion, model_path_override=pkl_path)` actualiza de forma atómica la referencia en `model_state`. La siguiente petición a `/api/v1/prediccion/venta` o `/api/v1/prediccion/alquiler` usará inmediatamente el nuevo modelo entrenado y su nuevo `TreeExplainer` de SHAP, sin necesidad de reiniciar el servicio FastAPI.

---

## 5. Ejemplos de Consumo

### 5.1. Con cURL (Bash / CMD)

```bash
# 1. Iniciar reentrenamiento de alquiler
curl -X POST "http://localhost:8000/api/v1/admin/entrenamiento/alquiler" \
     -H "Content-Type: application/json" \
     -d '{}'

# Respuesta: {"job_id": "MI_JOB_ID", "estado": "pending", ...}

# 2. Consultar el estado del proceso
curl -X GET "http://localhost:8000/api/v1/admin/entrenamiento/estado/MI_JOB_ID"
```

### 5.2. Con Python (`requests`)

```python
import time
import requests

BASE_URL = "http://localhost:8000/api/v1"
HEADERS = {"Content-Type": "application/json"}

# 1. Disparar el reentrenamiento
response = requests.post(f"{BASE_URL}/admin/entrenamiento/venta", json={}, headers=HEADERS)
job_info = response.json()
job_id = job_info["job_id"]
print(f"Reentrenamiento en cola. Job ID: {job_id}")

# 2. Polling del estado hasta finalizar
while True:
    estado_resp = requests.get(f"{BASE_URL}/admin/entrenamiento/estado/{job_id}", headers=HEADERS).json()
    print(f"Estado: {estado_resp['estado']} | Progreso: {estado_resp['progreso']}")
    
    if estado_resp["estado"] in ("completed", "failed"):
        if estado_resp["estado"] == "completed":
            print("\n¡Entrenamiento completado!")
            print(f"MAPE obtenido: {estado_resp['metricas']['mape_pct']}%")
            print(f"R² obtenido:   {estado_resp['metricas']['r2']}")
        else:
            print(f"\nError en el job: {estado_resp['error']}")
        break
        
    time.sleep(3)
```

---

## 6. Variables de Entorno Relevantes

En el archivo `.env`:

```env
# Conexión a la base de datos PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=inmobiliaria_ml_db
DB_USER=postgres
DB_PASSWORD=tu_password

# Seguridad administrativa (opcional, si se omite no exige token en dev)
ADMIN_TOKEN=tu_token_secreto_aqui

# Rutas de artefactos
MODEL_PATH=models/xgboost_venta_v2.pkl
MODEL_ALQUILER_PATH=models/xgboost_alquiler_v1.pkl
METADATA_PATH=data/features_metadata.json
METADATA_ALQUILER_PATH=data/features_metadata_alquiler.json
CONFIG_PATH=config/model_config.json
CONFIG_ALQUILER_PATH=config/model_alquiler_config.json
```

---

## 7. Optimización Metodológica: Target Unitario ($/m²) y Calibración IAAO

A partir de la versión validada en octubre de 2026, el pipeline incorpora soporte nativo para dos estándares internacionales de tasación masiva (*CAMA / IAAO*):

1. **Target Unitario (`usar_target_m2=True`):**
   * El target de entrenamiento es $y = \ln(\text{Precio} / \text{Superficie})$, lo cual elimina la heterocedasticidad entre propiedades de distinto tamaño y estabiliza las hojas del árbol XGBoost.
   * En inferencia, el modelo predice el canon unitario y se escala automáticamente por la superficie física: $\widehat{P} = \exp(\widehat{y}) \times \text{Superficie}$.
2. **Calibración Post-Hoc de Ratio IAAO (Estratificada por Distrito):**
   * Corrige el sesgo residual de la Desigualdad de Jensen calculando la mediana de ratios para cada uno de los 22 distritos (`factores_calibracion_distrito`): $\widehat{P}_{\text{calibrado}} = \widehat{P} / \text{Factor}_d$.
   * **Venta:** `MAPE = 14.16%` ($R^2 = 0.7472$, Ratio calibrado $= 1.000$).
   * **Alquiler:** `MAPE = 13.73%` ($R^2 = 0.6865$, Ratio calibrado $= 1.000$).
   * Ambas métricas baten ampliamente el benchmark académico de Lima (Oporto et al., 2024: $17.89\%$) y cumplen con el estándar de calidad de la IAAO ($\le 15\%$).

