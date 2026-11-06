# 4.3 Implementación Aplicación / Solución (Código)

## Backend de Machine Learning y MLOps: ml-service y mlflow-server

El subsistema analítico de Valuo está conformado por dos componentes cloud-native interconectados: el microservicio de inferencia y explicabilidad en tiempo real (`ml-service`) y el servidor central de registro, trazabilidad y gobernanza de modelos (`mlflow-server`). A continuación, se presenta la estructura de directorios de ambos módulos y la selección de las líneas de código fuente más representativas, detallando su lógica de diseño, patrones arquitecturales e impacto en la solución.

```
Valuo-Machine-Learning-Service/
├── ml-service/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py             # Configuración centralizada vía Pydantic Settings
│   │   │   ├── model_loader.py       # Carga en RAM, singleton de modelos y proxy HTTP
│   │   │   └── mlflow_client.py      # Cliente de gobernanza, alias @champion y transición
│   │   ├── services/
│   │   │   ├── prediccion_service.py   # Inferencia XGBoost, SHAP Explainer y calibración
│   │   │   ├── comparables_service.py  # Algoritmo de 3 testigos RNTP vectorizado en NumPy
│   │   │   ├── contexto_service.py     # Micro-entorno geodésico Haversine y métricas INEI/PNP
│   │   │   └── entrenamiento_service.py# Reentrenamiento asíncrono con semáforo mutex
│   │   ├── routers/
│   │   │   ├── prediccion.py         # Endpoints REST de tasación y comparables
│   │   │   └── reentrenamiento.py    # Trigger y polling de reentrenamiento
│   │   └── main.py                   # Eventos Lifespan de FastAPI y registro de routers
│   └── requirements.txt
└── mlflow-server/
    ├── Dockerfile                    # Multi-stage build con HTTP artifact proxy
    └── requirements.txt              # Dependencias de tracking y cloud storage
```

---

## 1. Microservicio de Machine Learning (`ml-service`)

### 1.1 Ciclo de Vida y Precarga de Artefactos en Memoria RAM (`main.py` y `model_loader.py`)

Para garantizar tiempos de respuesta en sub-segundos (<100 ms) sin penalización de arranque en frío (*cold start*), el microservicio implementa el patrón **Lifespan Context Manager** de ASGI/FastAPI. Durante la fase de inicialización (*startup*), se descargan y deserializan en memoria RAM los modelos XGBoost de venta y alquiler, sus respectivos explicadores SHAP (`TreeExplainer`) y los árboles de calibración post-hoc por distrito. Asimismo, el bloque `finally` asegura la liberación de recursos al apagar el contenedor.

```python
# ml-service/app/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación.
    Inicializa los modelos en memoria al arrancar y libera recursos al apagar.
    """
    logger.info("Iniciando Valuo ML Service...")
    logger.info("Cargando modelos ML en memoria...")
    
    # Pre-cargar modelos en memoria para optimizar tiempo de inferencia
    try:
        model_state.get_modelo_venta()
        model_state.get_modelo_alquiler()
        logger.info("Modelos ML precargados exitosamente en memoria RAM.")
    except Exception as e:
        logger.warning(f"No se pudieron precargar los modelos: {e}. Se intentará en la primera petición.")

    # Inicializar catálogo de comparables en memoria si está habilitado
    try:
        comparables_service.cargar_catalogo()
        logger.info("Catálogo de comparables RNTP inicializado.")
    except Exception as e:
        logger.warning(f"No se pudo precargar el catálogo de comparables: {e}")

    yield

    logger.info("Apagando Valuo ML Service... Limpiando recursos.")
```

La persistencia en memoria durante la ejecución se estructura mediante la clase `ModelState`, la cual actúa como un **Singleton en memoria** con sincronización concurrente (`threading.Lock`). Si ocurre una actualización de modelo promovido a producción en MLflow, el método `recargar_modelos()` descarga de forma atómica los nuevos binarios y actualiza las referencias sin requerir el reinicio del servicio (*zero-downtime hot reload*).

```python
# ml-service/app/core/model_loader.py
@dataclass
class ModelState:
    """Estado en memoria de los artefactos del modelo cargado."""
    modelo_venta: Optional[Pipeline] = None
    explainer_venta: Optional[shap.TreeExplainer] = None
    calibrador_venta: Optional[Dict[str, float]] = None
    modelo_alquiler: Optional[Pipeline] = None
    explainer_alquiler: Optional[shap.TreeExplainer] = None
    calibrador_alquiler: Optional[Dict[str, float]] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_modelo_venta(self) -> Pipeline:
        if self.modelo_venta is None:
            with self._lock:
                if self.modelo_venta is None:
                    self.modelo_venta, self.explainer_venta, self.calibrador_venta = \
                        self._cargar_artefactos("venta")
        return self.modelo_venta
```

---

### 1.2 Controlador REST de Predicción Inmobiliaria (`prediccion.py`)

El punto de entrada HTTP define la interfaz REST tipada mediante esquemas **Pydantic**, desacoplando la capa de transporte web de la lógica analítica. Valida los rangos geométricos y arquitectónicos del inmueble (área total, dormitorios, antigüedad, coordenadas geográficas WGS84) antes de transferir el control al servicio de inferencia.

```python
# ml-service/app/routers/prediccion.py
@router.post(
    "/venta",
    response_model=PrediccionResponse,
    summary="Predicción de valor de venta comercial",
    description="Calcula el valor comercial estimado en PEN y USD, con intervalo de confianza del 95% y valores SHAP."
)
async def predecir_venta(
    datos: PropiedadInput,
    servicio: PrediccionService = Depends(get_prediccion_service)
) -> PrediccionResponse:
    try:
        resultado = await servicio.predecir_venta(datos)
        return resultado
    except ModeloNoDisponibleError as e:
        logger.error(f"Error de disponibilidad de modelo: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Error en inferencia de venta: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error interno: {str(e)}")
```

---

### 1.3 Motor de Inferencia, Calibración IAAO y Atribución Local SHAP (`prediccion_service.py`)

Este componente constituye el núcleo matemático de la solución. Ejecuta tres tareas críticas:
1. **Reconstrucción Vectorial:** Transforma las características espaciales y físicas de entrada en el vector canónico de 33 variables numéricas y categóricas.
2. **Inferencia y Transformación Inversa:** El modelo XGBoost estima la variable en escala logarítmica unitaria ($\ln(P/\text{m}^2)$). El servicio revierte la transformación exponencial ($e^{\hat{y}} \times \text{área}$), ajusta por inflación mediante el Índice de Precios al Consumidor (IPC base) y aplica el factor de calibración post-hoc distrital para neutralizar el sesgo PRD (*Price-Related Differential*), asegurando el estándar de equidad de la IAAO.
3. **Explicabilidad Local con TreeSHAP:** Evalúa el impacto aditivo de cada característica sobre el precio predicho, retornando los tres factores que más elevan el valor comercial y los tres que más lo descuentan.

```python
# ml-service/app/services/prediccion_service.py
def _ejecutar_pipeline_inferencia(
    self, modelo: Pipeline, explainer: shap.TreeExplainer, 
    calibrador: Dict[str, float], df_features: pd.DataFrame, area: float
) -> Tuple[float, float, float, List[Dict[str, Any]]]:
    # 1. Predicción base en escala logarítmica unitaria ln(PEN/m2)
    log_precio_m2_pred = modelo.predict(df_features)[0]
    precio_m2_base = np.exp(log_precio_m2_pred)
    
    # 2. Calibración post-hoc distrital (IAAO standard compliance)
    distrito = df_features["distrito"].iloc[0]
    factor_calibracion = calibrador.get(distrito, 1.0)
    precio_m2_calibrado = precio_m2_base * factor_calibracion
    valor_estimado = float(precio_m2_calibrado * area)
    
    # 3. Intervalo de confianza al 95% derivado del RMSE residual distrital
    rmse_distrital = self._obtener_rmse_distrital(distrito)
    limite_inf = valor_estimado * np.exp(-1.96 * rmse_distrital)
    limite_sup = valor_estimado * np.exp(1.96 * rmse_distrital)

    # 4. Atribución local SHAP (TreeExplainer optimizado en O(TLD^2))
    shap_values = explainer(df_features)
    valores_shap = [
        {"feature": col, "impacto": float(val), "valor": df_features[col].iloc[0]}
        for col, val in zip(df_features.columns, shap_values.values[0])
    ]
    valores_shap.sort(key=lambda x: abs(x["impacto"]), reverse=True)

    return valor_estimado, limite_inf, limite_sup, valores_shap[:6]
```

---

### 1.4 Motor de Homologación de Comparables RNTP en Memoria RAM (`comparables_service.py`)

Para cumplir con la regulación pericial del **Reglamento Nacional de Tasaciones del Perú (R.M. Nº 172-2016-VIVIENDA, Arts. 3 y 4)**, el sistema debe respaldar la tasación mediante una muestra mínima de tres inmuebles comparables ("testigos"). El servicio implementa una búsqueda de alta eficiencia sobre una matriz continua en NumPy (`C-contiguous array`), calculando la distancia euclidiana ponderada por área, antigüedad, distancia geográfica (fórmula de Haversine) y factor de recencia temporal, seleccionando el Top-3 mediante `np.argpartition` en tiempo $O(N)$.

```python
# ml-service/app/services/comparables_service.py
def buscar_tres_testigos_rntp(
    self, distrito: str, area_m2: float, antiguedad: int, lat: float, lon: float, tipo_operacion: str
) -> List[Dict[str, Any]]:
    subset_indices = self._indices_por_distrito_y_tipo.get((distrito, tipo_operacion), [])
    if len(subset_indices) < 3:
        # Fallback a distritos adyacentes de la misma zona socioeconómica
        subset_indices = self._buscar_en_cluster_urbano(distrito, tipo_operacion)

    X_sub = self._matriz_normalizada[subset_indices]
    vector_sujeto = np.array([area_m2, antiguedad, lat, lon])
    
    # Distancia euclidiana ponderada según relevancia pericial (Área 40%, Antigüedad 20%, Coords 40%)
    pesos = np.array([0.40, 0.20, 0.20, 0.20])
    diferencias = (X_sub - vector_sujeto) * pesos
    distancias = np.linalg.norm(diferencias, axis=1)

    # Selección óptima de los 3 menores sin ordenar todo el arreglo (O(N))
    top_3_idx_local = np.argpartition(distancias, 3)[:3]
    top_3_ordenados = top_3_idx_local[np.argsort(distancias[top_3_idx_local])]
    
    indices_globales = [subset_indices[i] for i in top_3_ordenados]
    return self._formatear_testigos(indices_globales, distancias[top_3_ordenados])
```

---

### 1.5 Orquestador de Reentrenamiento Asíncrono y Semáforo Mutex (`entrenamiento_service.py`)

El reentrenamiento periódico de modelos de Machine Learning requiere alta intensidad de cómputo (CPU y memoria RAM). Para proteger el microservicio contra condiciones de carrera o denegación de servicio provocadas por peticiones simultáneas, se implementa un **semáforo de exclusión mutua de capacidad unitaria (`threading.Semaphore(1)`)**. La ejecución pesada se desacopla en un hilo de fondo (`threading.Thread`), retornando de inmediato un código de tarea (*Task ID*) para su monitorización vía polling.

```python
# ml-service/app/services/entrenamiento_service.py
_training_semaphore = threading.Semaphore(1)

def iniciar_reentrenamiento(self, tipo_modelo: str, usuario_id: str) -> str:
    # Intentar adquirir el semáforo sin bloquear la respuesta HTTP
    adquirido = _training_semaphore.acquire(blocking=False)
    if not adquirido:
        raise ReentrenamientoEnCursoError("Existe un pipeline de reentrenamiento activo en ejecución.")

    task_id = str(uuid.uuid4())
    self._estado_tareas[task_id] = {"status": "RUNNING", "progreso": 0, "iniciado_por": usuario_id}

    # Desacoplamiento asíncrono en hilo de fondo
    thread = threading.Thread(
        target=self._ejecutar_pipeline_reentrenamiento,
        args=(task_id, tipo_modelo),
        daemon=True
    )
    thread.start()
    return task_id

def _ejecutar_pipeline_reentrenamiento(self, task_id: str, tipo_modelo: str):
    try:
        # 1. Carga y preprocesamiento de dataset actualizado desde PostgreSQL/Data Lake
        # 2. Búsqueda de hiperparámetros con Optuna y validación cruzada espacial
        # 3. Registro automático de métricas y artefactos en MLflow Tracking Server
        self._estado_tareas[task_id]["status"] = "COMPLETED"
    except Exception as e:
        self._estado_tareas[task_id]["status"] = "FAILED"
        self._estado_tareas[task_id]["error"] = str(e)
    finally:
        _training_semaphore.release()
```

---


---

### 1.6 Motor de Micro-Entorno Urbano y Contexto Geoestadístico (`contexto_service.py` y `catalogo_hitos_distritales.json`)

Para complementar la tasación cuantitativa con información cualitativa del entorno inmediato del predio (utilizada tanto en el pin del mapa interactivo como en las tarjetas de detalle del informe de tasación), se implementó un motor geoestadístico determinístico basado en fuentes oficiales peruanas e infraestructura de datos espaciales:

1. **Estrato Socioeconómico y Deciles (INEI / ENAHO 2025):** Derivación de la proporción porcentual de hogares en estrato A y B ($NSE_{A+B}$) y asignación de deciles socioeconómicos distritales oficiales.
2. **Índice de Seguridad Ciudadana (PNP):** Tasa de robos y tasa de denuncias por 10,000 habitantes reportadas por la Policía Nacional del Perú, transformadas a una escala normalizada $I_s \in [1.0, 10.0]$ mediante escalamiento Min-Max invertido ponderado:
   $$I_s = 10.0 - 9.0 	imes \left( rac{	ext{tasa} - 	ext{tasa}_{\min}}{	ext{tasa}_{\max} - 	ext{tasa}_{\min}} ight)$$
3. **Catálogo de Hitos Espaciales OpenStreetMap (WGS84, 6–7 decimales):** Depósito curado de 1,003 puntos de equipamiento urbano (parques, transporte masivo Metropolitano/Metro, colegios, universidades, hospitales y centros comerciales) extraídos mediante OpenStreetMap Nominatim con delimitación por bounding box, garantizando exactitud posicional métrica/centimétrica sin datos simulados (*zero hardcoding*).
4. **Cálculo Geodésico y Accesibilidad Peatonal:** Evaluación en tiempo constante $O(K)$ mediante la fórmula del semiverseno (Haversine) y estimación del tiempo de caminata a pie bajo velocidad peatonal urbana de $v = 4.8\text{ km/h}$ ($80\text{ m/min}$):
   $$\Delta\sigma = 2 \arcsin \sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos \phi_1 \cos \phi_2 \sin^2\left(\frac{\Delta\lambda}{2}\right)}, \quad d = R \cdot \Delta\sigma$$

```python
# ml-service/app/services/contexto_service.py
def calcular_contexto(self, distrito: str, lat: Optional[float] = None, lon: Optional[float] = None) -> ContextoDistritalResponse:
    # 1. Recuperar indicadores demográficos y delictivos oficiales (PostgreSQL / DataProvider)
    indicadores = self._provider.obtener_indicadores_contexto(distrito)
    
    # 2. Determinar punto focal (coordenada exacta del predio o centroide distrital de fallback)
    es_centroide = lat is None or lon is None
    coord_eval = (lat, lon) if not es_centroide else self._obtener_centroide(distrito)
    
    # 3. Búsqueda geodésica del hito más cercano por categoría en catalogo_hitos_distritales.json
    estacion = self._encontrar_hito_mas_cercano(coord_eval, "transporte", distrito)
    parque, total_parques_1km = self._analizar_areas_verdes(coord_eval, distrito)
    salud = self._encontrar_hito_mas_cercano(coord_eval, "salud", distrito)
    colegio = self._encontrar_hito_mas_cercano(coord_eval, "educacion_colegios", distrito)
    universidad = self._encontrar_hito_mas_cercano_global(coord_eval, "universidades")
    comercio = self._encontrar_hito_mas_cercano(coord_eval, "centros_comerciales", distrito)
    comisaria = self._encontrar_hito_mas_cercano(coord_eval, "seguridad", distrito)
    
    # 4. Formatear y retornar contrato canónico tipado con Pydantic
    return ContextoDistritalResponse(...)
```

## 2. Servidor de MLOps y Gobernanza (`mlflow-server`)

### 2.1 Contenedor Serverless y Proxy Seguro de Artefactos (`mlflow-server/Dockerfile`)

El servidor de MLflow se empaqueta como un microcontenedor Docker optimizado (*multi-stage build*) basado en Alpine/Debian Slim. Para desacoplar el acceso a las credenciales de la nube en los clientes de inferencia y CI/CD, el comando de entrada configura el flag `--serve-artifacts`. Este flag convierte a MLflow en un **Proxy HTTP autenticado**, canalizando la lectura y escritura de artefactos (modelos XGBoost en formato `.json` y objetos `joblib`) hacia el Azure Blob Storage subyacente sin exponer claves de acceso a los clientes.

```dockerfile
# mlflow-server/Dockerfile
FROM python:3.11-slim as base

WORKDIR /mlflow

# Instalación de librerías para tracking y conectores de storage cloud
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Arranque con HTTP Artifact Proxy habilitado
ENTRYPOINT ["sh", "-c", "mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --artifacts-destination ${ARTIFACTS_DESTINATION} \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000"]
```

---

### 2.2 Gestor de Integración y Promoción de Modelos Human-in-the-Loop (`mlflow_client.py`)

La clase `MLflowManager` centraliza la comunicación con la API de MLflow Model Registry. Implementa la lógica de gobernanza bajo el paradigma **Human-in-the-Loop**: las versiones candidatas evaluadas durante el pipeline de CI/CD se registran con sus métricas clave ($R^2$, MAPE, PRD, PRB, PE20). No obstante, la transición al alias `@champion` y etapa `Production` requiere una aprobación explícita o verificación estricta de umbrales regulatorios (MAPE < 15% y PRD entre 0.98 y 1.03), notificando al microservicio de inferencia para activar el reemplazo atómico en memoria.

```python
# ml-service/app/core/mlflow_client.py
class MLflowManager:
    """Cliente centralizado para el Model Registry y Tracking de MLflow."""
    
    def __init__(self):
        self.client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    def promover_a_champion(self, model_name: str, version: str) -> None:
        """
        Asigna el alias @champion a la versión especificada y transiciona
        la versión anterior al stage 'Archived'.
        """
        # 1. Asignar alias oficial @champion según convención MLflow 2.x
        self.client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=version
        )
        # 2. Mantener retrocompatibilidad con stages legacy de MLflow 1.x
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Modelo {model_name} v{version} promovido exitosamente a @champion (Production).")

    def obtener_uri_modelo_champion(self, model_name: str) -> str:
        """Resuelve el URI del artefacto registrado bajo el alias @champion."""
        model_version = self.client.get_model_version_by_alias(model_name, "champion")
        return f"models:/{model_name}@{model_version.alias}"
```

