# Plan de Implementación: MLOps con MLflow Open Source y Azure Blob Storage

## 1. Visión y Objetivos

Este plan define la hoja de ruta técnica para incorporar **MLflow Open Source** y **Azure Blob Storage** en el microservicio (`ml-service`). 

### Objetivos Clave:
1. **Gobernanza y Validación Humana (*Human-in-the-Loop*):** Ningún modelo reentrenado pasa a producción automáticamente. Queda registrado como candidato (`Staging` / `Candidate`) hasta que el usuario/administrador valide el benchmark comparativo frente al modelo actualmente en producción.
2. **Almacenamiento Desacoplado y Contenedores *Stateless*:** Eliminar la dependencia de archivos binarios `.pkl` en el disco local del repositorio o contenedor Docker. Los modelos se almacenan en Azure Blob Storage y se cargan **exclusivamente en memoria RAM** mediante streaming/descarga en el ciclo de vida de la aplicación.
3. **Costo-Eficiencia y Separación de Bases de Datos:**
   - **`inmobiliaria_ml_db`:** Permanece 100% exclusiva y aislada para los datos históricos de inmuebles y contexto distrital. MLflow **nunca** escribe en ella.
   - **`db_operacional_valuo`:** Es la base de datos operacional del sistema; en ella se utiliza el schema dedicado `mlops` (o `monitoreo`) como backend store de MLflow.
   - **Azure Blob Storage:** Almacena los artefactos pesados (`.pkl`) de forma sumamente económica (~$0.018 USD/GB/mes).
4. **Cero Downtime (*Hot-Reload* Seguro):** Cuando el administrador aprueba una versión candidata, el microservicio descarga el artefacto a RAM y actualiza la referencia en caliente sin interrumpir el servicio de inferencia.

---

## 2. Arquitectura de Gobernanza

```mermaid
flowchart TD
    subgraph UI["Panel Web de Administración"]
        BTN_TRAIN[1. Solicitar Reentrenamiento]
        VIEW_BENCH[4. Visualizar Benchmark Comparativo\n(Producción vs Candidato)]
        BTN_PROMOTE[5. Aprobar y Promover a Producción]
    end

    subgraph Service["ml-service (FastAPI - Servicio de ML)"]
        ROUTER_TRAIN[/api/v1/admin/entrenamiento/*]
        ROUTER_MODELS[/api/v1/admin/modelos/*]
        TRAIN_SVC[EntrenamientoService]
        LOADER[ModelLoader\n(RAM Singleton)]
    end

    subgraph MLOps["Servidor Externo: MLflow Tracking & Registry"]
        TRACKING[MLflow Server - Puerto 5000]
        REGISTRY[Model Registry\n- xgboost_venta\n- xgboost_alquiler]
    end

    subgraph Storage["Bases de Datos y Storage"]
        DB_TRAIN[(PostgreSQL: inmobiliaria_ml_db\nSOLO Datos de Entrenamiento)]
        DB_OPS[(PostgreSQL: db_operacional_valuo\nschema: mlops / monitoreo)]
        AZURE_BLOB[(Azure Blob Storage\nContenedor: ml-artifacts)]
    end

    BTN_TRAIN --> ROUTER_TRAIN
    ROUTER_TRAIN --> TRAIN_SVC
    TRAIN_SVC -->|1. Lee datos de inmuebles| DB_TRAIN
    TRAIN_SVC -->|2. Envía Params y Métricas| TRACKING
    TRACKING -->|Guarda Runs en schema mlops| DB_OPS
    TRAIN_SVC -->|Sube modelo serializado| AZURE_BLOB
    TRAIN_SVC -->|Registra versión candidata| REGISTRY

    VIEW_BENCH --> ROUTER_MODELS
    ROUTER_MODELS -->|Consulta comparativa y diff| REGISTRY

    BTN_PROMOTE --> ROUTER_MODELS
    ROUTER_MODELS -->|Promueve versión a 'Production'| REGISTRY
    ROUTER_MODELS -->|Dispara Hot-Reload a RAM| LOADER
    LOADER -->|Descarga modelo a RAM| AZURE_BLOB
```

---

## 3. Fases de Implementación

```
┌────────────────────────────────────────────────────────────────────────┐
│ FASE 1: Dependencias & Configuración Base de MLflow                     │
│ - Instalar mlflow y azure-storage-blob en ml-service                   │
│ - Variables de entorno (.env) para tracking URI y Storage              │
│ - Cliente singleton de MLflow                                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ FASE 2: Integración de MLflow Tracking en el Reentrenamiento           │
│ - Modificar app/services/entrenamiento_service.py                      │
│ - mlflow.start_run() con registro de parámetros e hiperparámetros       │
│ - Log de métricas oficiales (MAPE, R2, MAE, RMSE)                      │
│ - Registro del modelo en MLflow Model Registry (Versión N, Candidata)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ FASE 3: Módulo de Benchmark y Comparativa                              │
│ - Servicio app/services/model_registry_service.py                      │
│ - Endpoint GET /api/v1/admin/modelos/benchmark/{tipo}                  │
│ - Comparación automática: Versión Actual vs Versiones Candidatas       │
│ - Cálculo de deltas: ΔMAPE, ΔR2 y recomendación automática             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ FASE 4: Gobernanza y Promoción con Hot-Reload a Memoria RAM            │
│ - Endpoint POST /api/v1/admin/modelos/promover                         │
│ - Actualización de Stage o Alias en MLflow ('Production' / 'champion') │
│ - Descarga del artefacto desde Storage directamente a memoria RAM      │
│ - Reconstrucción del TreeExplainer SHAP en RAM sin escribir en disco   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ FASE 5: Startup Stateless desde MLflow (Lifespan)                      │
│ - Modificar app/core/model_loader.py                                   │
│ - Carga al iniciar el servidor desde MLflow Production URI             │
│ - Fallback de resiliencia local en caso de desconexión                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ FASE 6: Pruebas Integrales y Documentación                             │
│ - Prueba end-to-end: Reentrenar -> Ver Benchmark -> Promover -> Inferir│
│ - Documentación técnica y guía de variables para despliegue            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detalle de Cada Fase

### Fase 1: Dependencias y Configuración Base
- **Objetivo:** Preparar el entorno de `ml-service` para interactuar con MLflow y Azure Storage.
- **Acciones:**
  1. Actualizar `ml-service/requirements.txt` agregando:
     - `mlflow>=2.12.0`
     - `azure-storage-blob>=12.19.0`
     - `azure-identity>=1.15.0`
  2. Agregar configuración en `.env` y `.env.example`:
     ```env
     # ─── MLflow Tracking Server ──────────────────────────────────────────
     MLFLOW_TRACKING_URI=http://localhost:5000
     # O conexión directa con backend en PostgreSQL:
     # MLFLOW_TRACKING_URI=postgresql://postgres:pass@localhost:5432/inmobiliaria_ml_db?options=-csearch_path%3Dmlops
     
     # ─── Artifact Store (Azure Blob Storage) ─────────────────────────────
     # En dev: azure / local folder / minio
     MLFLOW_ARTIFACT_STORE=local  # 'azure' o 'local'
     AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
     AZURE_STORAGE_CONTAINER=ml-artifacts
     ```
  3. Crear `app/core/mlflow_client.py` con una clase wrapper que gestione la conexión y reconexión a MLflow de forma resiliente.

---

### Fase 2: Registro de Experimentos y Model Registry en Reentrenamiento
- **Objetivo:** Loguear cada reentrenamiento en MLflow sin tocar la versión de producción activa.
- **Acciones:**
  1. Modificar `app/services/entrenamiento_service.py`:
     - Abrir contexto de ejecución:
       ```python
       with mlflow.start_run(run_name=f"{nombre_modelo}_{timestamp}") as run:
           mlflow.log_params(params)
           mlflow.log_metrics(metricas_dict)
           mlflow.set_tag("tipo_operacion", tipo_operacion)
           mlflow.set_tag("estado_gobernanza", "candidate")
       ```
     - Subir modelo y metadata al Model Registry de MLflow:
       - Nombre registrado: `xgboost_venta` o `xgboost_alquiler`.
       - La nueva versión creada queda con stage `None` o tag `candidate`.
  2. **Regla de negocio:** **NO** ejecutar `cargar_modelo()` inmediatamente en el paso 8. La versión en memoria RAM sigue siendo la de producción actual hasta que el usuario decida promoverla.

---

### Fase 3: Módulo de Benchmark y Comparativa
- **Objetivo:** Proveer a la interfaz administrativa de toda la información para comparar modelos.
- **Acciones:**
  1. Crear `app/services/model_registry_service.py` con métodos para:
     - Obtener la versión activa en `Production` (con sus métricas: MAPE, $R^2$, MAE, RMSE, fecha).
     - Obtener las versiones candidatas registradas más recientes.
     - Calcular la matriz de diferencias (deltas):
       - $\Delta \text{MAPE} = \text{MAPE}_{\text{candidato}} - \text{MAPE}_{\text{producción}}$ (negativo es mejor).
       - $\Delta R^2 = R^2_{\text{candidato}} - R^2_{\text{producción}}$ (positivo es mejor).
     - Determinar recomendación automática: `"RECOMENDADO_PARA_PRODUCCION"` si mejora el MAPE y supera los benchmarks de la tesis.
  2. Implementar endpoint en `app/routers/modelos.py`:
     - `GET /api/v1/admin/modelos/benchmark/{tipo_operacion}`
       - Devuelve comparativa completa estructurada en JSON lista para ser renderizada en cards o tablas comparativas del frontend.

---

### Fase 4: Gobernanza y Promoción con Hot-Reload a RAM
- **Objetivo:** Promover formalmente una versión candidata y actualizar la memoria RAM sin interrumpir peticiones.
- **Acciones:**
  1. Implementar endpoint en `app/routers/modelos.py`:
     - `POST /api/v1/admin/modelos/promover`
     - Body:
       ```json
       {
         "tipo_operacion": "venta",
         "version": 3,
         "motivo": "Mejora de MAPE de 15.04% a 13.85% tras reentrenamiento con datos 2025"
       }
       ```
  2. En el servicio:
     - Mediante `MlflowClient`, archivar la versión previa en `Production` a `Archived`.
     - Promover la versión solicitada a `Production` (o alias `champion`).
     - Registrar tags de auditoría: `promoted_by`, `promoted_at`, `promotion_reason`.
  3. **Carga directa en RAM (Zero-Downtime):**
     - Descargar el binario del modelo desde el artifact store directamente como un buffer de bytes o archivo temporal en RAM.
     - Deserializar con `joblib.load(buffer)`.
     - Construir el `shap.TreeExplainer(nuevo_modelo)` en RAM.
     - Actualizar atómicamente la referencia en el singleton `model_state`.
     - Eliminar cualquier archivo temporal; la RAM pasa a atender la siguiente inferencia con el nuevo modelo.

---

### Fase 5: Startup Stateless desde MLflow (`lifespan`)
- **Objetivo:** Que el microservicio al arrancar (`uvicorn`) obtenga el modelo vigente directamente desde MLflow / Azure Blob Storage.
- **Acciones:**
  1. Modificar `app/core/model_loader.py`:
     - Consultar a MLflow cuál es la versión activa en `Production` para `xgboost_venta` y `xgboost_alquiler`.
     - Descargar los artefactos a RAM y cargarlos en `model_state`.
  2. Mecanismo de **Resiliencia (Fallback)**:
     - Si MLflow o la conexión a Azure está temporalmente inaccesible al arrancar en local, permitir usar el archivo local de contingencia (`models/xgboost_*.pkl`) para que el servicio nunca se caiga en desarrollo.

---

### Fase 6: Pruebas Integrales y Validación
- **Objetivo:** Validar todo el ciclo de vida sin errores.
- **Escenarios de prueba:**
  1. **Prueba 1:** Reentrenar modelo de venta $\rightarrow$ verificar que el modelo en producción NO cambia en memoria.
  2. **Prueba 2:** Consultar endpoint de benchmark $\rightarrow$ verificar que devuelve ambas versiones con métricas y diferencias calculadas.
  3. **Prueba 3:** Promover versión candidata $\rightarrow$ verificar que MLflow actualiza el stage a `Production`, la RAM se actualiza y la siguiente predicción usa el nuevo modelo.
  4. **Prueba 4:** Verificar que no quedan archivos `.pkl` residuales dispersos.

---

## 5. Matriz de Endpoints

| Método | Endpoint | Rol | Descripción |
| :--- | :--- | :---: | :--- |
| `POST` | `/api/v1/admin/entrenamiento/{tipo}` | Admin | Inicia reentrenamiento y registra run + versión candidata en MLflow. |
| `GET` | `/api/v1/admin/entrenamiento/estado/{job_id}` | Admin | Consulta avance del entrenamiento y métricas generadas. |
| `GET` | `/api/v1/admin/modelos/benchmark/{tipo}` | Admin | Obtiene el benchmark comparativo: modelo en producción vs candidatos. |
| `GET` | `/api/v1/admin/modelos/historial/{tipo}` | Admin | Lista todas las versiones registradas en el Model Registry de MLflow. |
| `POST` | `/api/v1/admin/modelos/promover` | Admin | Promueve una versión a producción y ejecuta Hot-Reload en RAM. |
| `POST` | `/api/v1/prediccion/{tipo}` | Público | Realiza inferencias con el modelo activo en RAM (cero downtime). |

---

## 6. Siguientes Pasos Inmediatos

1. Aprobar este plan de implementación.
2. Iniciar con la **Fase 1**: agregar dependencias de MLflow y configurar la conexión de tracking y artifacts.
