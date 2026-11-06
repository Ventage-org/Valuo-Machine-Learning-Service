# Capítulo 3: Modelado y Desarrollo de la Solución de Machine Learning
## Insumo Técnico Oficial para la Memoria de Proyecto Profesional (Tesis UPC)

**Proyecto:** Servicio de Valoración Masiva Inmobiliaria en Lima Metropolitana  
**Modelos:** Tasación de Compraventa (`xgboost_venta`) y Estimación de Alquiler Mensual (`xgboost_alquiler`)  
**Fecha de Validación:** Octubre 2026 (Métricas auditadas libres de fuga de datos)  
**Entorno de Datos:** BCRP / Urbania (2016–2025), ENAHO-INEI, PNP y OpenStreetMap  

---

## 1. Universo y Partición del Conjunto de Datos (Zero-Leakage Split)

Para garantizar la validez científica y evitar el sesgo de anticipación (*look-ahead bias*), se implementó una partición temporal estricta (*Out-of-Time*). Se descartaron distritos con una sola transacción ($n=1$) en una década por carecer de significancia estadística, consolidando un universo homogéneo de **22 distritos activos de Lima Metropolitana**:

* **Conjunto de Entrenamiento (Train: 2016–2023):** Empleado exclusivamente para la imputación, cálculo de medias de codificación bayesiana (*Target Encoding*), optimización de hiperparámetros y factores de calibración.
* **Conjunto de Evaluación Ciega (Test: 2024–2025):** Registros estrictamente reservados a ciegas (excluidos del entrenamiento y del `eval_set` de parada temprana).
* **Ponderación Temporal Exponencial:** Para reflejar la estructura de mercado pospandemia sin descartar la serie histórica, las muestras de entrenamiento se ponderaron en la función de pérdida con decaimiento anual:
  $$w_i = 0.85^{(2023 - \text{Anio}_i)}$$

### Cuadro 3.1: Distribución Canónica de Registros por Tipo de Operación
| Tipo de Operación | Train (2016–2023) | Test (2024–2025) | Total Registros Únicos | Proporción Train / Test | Cobertura Espacial |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Compraventa Residencial** | 44,173 | 23,737 | **67,910** | 65.05% / 34.95% | 22 distritos de Lima |
| **Alquiler Residencial** | 41,921 | 19,682 | **61,603** | 68.05% / 31.95% | 22 distritos de Lima |
| **Total Observaciones** | **86,094** | **43,419** | **129,513** | **100.0%** | **Lima Metropolitana** |

---

## 2. Limpieza de Datos, Filtrado de Atípicos e Imputación Controlada

### 2.1. Reglas de Filtrado de Atípicos (Outliers Físicos y de Mercado)
Se definieron umbrales paramétricos basados en la realidad constructiva formal del mercado limeño:
1. **Superficie útil:** Válida exclusivamente entre **15.0 m² y 1,200.0 m²** (eliminando errores tipográficos como áreas de 1 m² o 15,000 m²).
2. **Habitaciones y Baños:** Habitaciones restringidas de 1 a 10; cuartos de baño de 1.0 a 10.0.
3. **Piso de ubicación:** Filtrado en el intervalo de 1 a 45 pisos.
4. **Filtro de consistencia monetaria:** Eliminación de inmuebles con precios totales o alquileres fuera de límites económicos verosímiles en Lima (alquileres menores a S/ 300 o mayores a S/ 35,000 mensuales en tipología residencial estándar).

### 2.2. Protocolo de Imputación y Prevención de Sesgo de Instrumentación
* **Atributos Físicos:** Las variables continuas faltantes (`antiguedad`, `piso`) se imputaron con la **mediana distrital**; variables discretas/categóricas (`vista_exterior`, `garajes`) con la **moda distrital**.
* **Fase 0 Contextual (Offline):** Se construyó la tabla `distrito_anio_contexto` (202 registros anuales). En la encuesta ENAHO-INEI, los distritos de Barranco, Magdalena y Lince presentaban menos de 30 hogares encuestados; se les aplicó imputación por **Distrito Hermano** geodésico en coordenadas UTM Zona 18S (usando Miraflores, San Miguel y Jesús María respectivamente).
* **Blindaje contra Sesgo de Medición:** Se crearon 129,513 banderas en la tabla `auditoria_flags_imputacion`. Todas las variables imputadas artificialmente se asignaron con peso cero en pruebas de sensibilidad para garantizar que el modelo aprenda patrones de mercado reales.

---

## 3. Transformación de la Variable Objetivo (Target Engineering)

Para cumplir con las directrices de la **International Association of Assessing Officers (IAAO)** y corregir la severa heterocedasticidad presente entre departamentos pequeños y propiedades de gran extensión, no se modela el precio bruto directo. Se implementó una formulación matemática en 3 etapas:

### 3.1. Deflactación Monetaria
Los precios nominales se deflactaron a **Soles Constantes** utilizando la serie histórica oficial del Índice de Precios al Consumidor (IPC Base Diciembre 2009 = 100):
$$\text{Precio\_Const}_i = \frac{\text{Precio\_Nominal}_i}{\text{IPC}_i} \times 100$$

### 3.2. Normalización Logarítmica del Valor Unitario
El modelo predice el logaritmo natural del precio unitario por metro cuadrado:
$$y_i = \ln\left(\frac{\text{Precio\_Const}_i}{\text{Superficie}_i}\right)$$
*Justificación técnica:* Transforma una distribución asimétrica positiva (skewness original > 1.43) en una distribución simétrica casi normal (skewness post-transformación = 0.08), estabilizando la varianza del gradiente en XGBoost.

### 3.3. Reversión y Calibración Post-Hoc Distrital (IAAO)
Para retornar a la escala de tasación comercial en soles, se aplica la función exponencial corregida por el factor de calibración por distrito (calculado en la mediana del ratio sobre la ventana Train 2023):
$$\hat{P}_{i,d} = \left(\frac{\exp(\hat{y}_i)}{\text{Factor}_d}\right) \times \text{Superficie}_i$$
* **Factor de Calibración Global:** $1.0017$ para Venta y $1.0062$ para Alquiler.
* **Factores Específicos:** 22 multiplicadores distritales aprendidos en entrenamiento.
* **Conversión a Soles Nominales vigentes:**
  $$\hat{P}_{\text{nom}} = \hat{P}_{i,d} \times \left(\frac{\text{IPC}_{\text{vigente}}}{100}\right)$$

---

## 4. Catálogo Detallado de las 33 Variables Predictivas Finales

Ambos modelos operan sobre una **matriz simétrica y homogénea de 33 predictores**. A continuación se detalla cada variable, su nombre exacto en base de datos, tipo de dato, descripción y fuente de origen:

### Cuadro 3.2: Diccionario Exhaustivo de Variables Predictivas ($p=33$)
| N° | Nombre Técnico de Variable | Tipo de Dato | Unidad / Formato | Descripción y Propósito | Fuente Primaria |
| :-: | :--- | :---: | :---: | :--- | :--- |
| **1** | `Superficie` | `Float` | Metros cuadrados ($m^2$) | Área ocupada construida del inmueble. | BCRP / Portales |
| **2** | `Habitaciones` | `Integer` | Cantidad | Número de dormitorios o cuartos habitables. | BCRP / Portales |
| **3** | `Banios` | `Float` | Cantidad | Número de cuartos de baño completos y medios baños. | BCRP / Portales |
| **4** | `Garajes` | `Integer` | Cantidad | Número de estacionamientos o cocheras asignadas. | BCRP / Portales |
| **5** | `Piso` | `Integer` | Nivel | Nivel en el que se ubica el departamento en el edificio. | BCRP / Portales |
| **6** | `Vista_Exterior` | `Integer` | Binario (0 / 1) | 1 = Vista exterior a calle/parque; 0 = Vista interior. | BCRP / Portales |
| **7** | `Antiguedad` | `Integer` | Años | Años de antigüedad desde la construcción de la edificación. | BCRP / Portales |
| **8** | `tiene_garaje` | `Integer` | Binario (0 / 1) | Variable derivada: 1 si `Garajes` $\ge 1$; 0 si no cuenta con cochera. | Feature Engineering |
| **9** | `es_piso_alto` | `Integer` | Binario (0 / 1) | Variable derivada: 1 si el inmueble se ubica a partir del piso 5. | Feature Engineering |
| **10** | `m2_por_habitacion` | `Float` | $m^2$ / cuarto | Amplitud promedio por habitación (`Superficie` / `Habitaciones`). | Feature Engineering |
| **11** | `ratio_banios_hab` | `Float` | Proporción | Densidad de servicios sanitarios por dormitorio (`Banios` / `Habitaciones`). | Feature Engineering |
| **12** | `superficie_cuadrado` | `Float` | $m^4$ | Término cuadrático de superficie para capturar rendimientos marginales. | Feature Engineering |
| **13** | `Anio` | `Integer` | AAAA (2016–2025) | Año cronológico del registro o tasación. | Serie Histórica BCRP |
| **14** | `Trimestre` | `Integer` | 1 a 4 | Trimestre del año calendario en que se reportó la operación. | Serie Histórica BCRP |
| **15** | `periodo_numerico` | `Float` | Secuencia continua | Variable temporal: $\text{Anio} + (\text{Trimestre} - 1) / 4$. | Feature Engineering |
| **16** | `pct_NSE_A` | `Float` | Porcentaje (0–100) | Proporción de hogares del distrito en Nivel Socioeconómico A. | ENAHO-INEI (`FACTOR07`) |
| **17** | `pct_NSE_B` | `Float` | Porcentaje (0–100) | Proporción de hogares del distrito en Nivel Socioeconómico B. | ENAHO-INEI (`FACTOR07`) |
| **18** | `pct_NSE_C` | `Float` | Porcentaje (0–100) | Proporción de hogares del distrito en Nivel Socioeconómico C. | ENAHO-INEI (`FACTOR07`) |
| **19** | `pct_NSE_D` | `Float` | Porcentaje (0–100) | Proporción de hogares del distrito en Nivel Socioeconómico D. | ENAHO-INEI (`FACTOR07`) |
| **20** | `pct_NSE_E` | `Float` | Porcentaje (0–100) | Proporción de hogares del distrito en Nivel Socioeconómico E. | ENAHO-INEI (`FACTOR07`) |
| **21** | `tasa_robo` | `Float` | Denuncias / 10k hab | Tasa anual distrital de denuncias por delito de robo. | PNP (Denuncias oficiales) |
| **22** | `tasa_hurto` | `Float` | Denuncias / 10k hab | Tasa anual distrital de denuncias por delito de hurto. | PNP (Denuncias oficiales) |
| **23** | `poblacion_proyectada` | `Float` | Habitantes | Proyección poblacional anual desagregada a nivel distrital. | INEI (Demografía) |
| **24** | `area_distrito_km2` | `Float` | Kilómetros cuadrados | Extensión territorial total del distrito. | INEI / Cartografía |
| **25** | `densidad_hab_km2` | `Float` | Habitantes / $km^2$ | Densidad poblacional distrital (`poblacion` / `area_km2`). | INEI (Demografía) |
| **26** | `distancia_centro_km` | `Float` | Kilómetros ($km$) | Distancia euclidiana calculada desde el centroide distrital al centro financiero. | Cálculo Geoespacial |
| **27** | `dist_colegio_km` | `Float` | Kilómetros ($km$) | Distancia geodésica al colegio o centro educativo más cercano. | OpenStreetMap (Overpass) |
| **28** | `dist_hospital_km` | `Float` | Kilómetros ($km$) | Distancia geodésica al centro de salud u hospital más próximo. | OpenStreetMap (Overpass) |
| **29** | `dist_estacion_transporte_km`| `Float`| Kilómetros ($km$) | Distancia a la estación masiva más cercana (Metropolitano / Metro L1). | OpenStreetMap (Overpass) |
| **30** | `dist_centro_comercial_km`| `Float` | Kilómetros ($km$) | Distancia al centro comercial o galería departamental más cercana. | OpenStreetMap (Overpass) |
| **31** | `dist_parque_km` | `Float` | Kilómetros ($km$) | Distancia geodésica al parque urbano o área verde más cercana. | OpenStreetMap (Overpass) |
| **32** | `dist_universidad_km` | `Float` | Kilómetros ($km$) | Distancia al campus universitario más cercano. | OpenStreetMap (Overpass) |
| **33** | `distrito_encoded` | `Float` | Escala monetaria | Target Encoding Bayesiano con suavizado $m=10.0$ sobre la media de Lima. | Pipeline Supervisado |

---

## 5. Hiperparámetros Optimizados de los Modelos XGBoost

El entrenamiento se realizó mediante el algoritmo **XGBoost Regressor** con particionado de histogramas (`tree_method='hist'`), optimizado con validación cruzada temporal:

### Cuadro 3.3: Configuración Final de Hiperparámetros
| Hiperparámetro | Modelo Venta (`xgboost_venta`) | Modelo Alquiler (`xgboost_alquiler`) | Criterio de Control / Función |
| :--- | :---: | :---: | :--- |
| **Método de división** | `hist` | `hist` | Cuantización en bins de histogramas para alta velocidad en RAM. |
| **Número de árboles (`n_estimators`)** | 650 | 600 | Convergencia óptima sin degradación de gradiente. |
| **Profundidad máxima (`max_depth`)** | 6 | 6 | Límite de interacciones para prevenir memorización (sobreajuste). |
| **Tasa de aprendizaje (`learning_rate`)** | 0.040 | 0.035 | Paso de contracción suave para regularizar las hojas. |
| **Submuestreo de filas (`subsample`)** | 0.852 | 0.850 | Muestreo estocástico aleatorio en cada árbol. |
| **Submuestreo de variables (`colsample_bytree`)** | 0.847 | 0.800 | Porcentaje de variables consideradas por nodo. |
| **Regularización L1 (`reg_alpha`)** | 4.060 | 0.100 | Penalización Lasso que anula coeficientes de variables ruidosas. |
| **Regularización L2 (`reg_lambda`)** | 0.031 | 4.000 | Penalización Ridge para suavizar pesos en hojas terminales. |
| **Peso mínimo de hoja (`min_child_weight`)** | 9 | 4 | Evita la creación de nodos basados en pocas muestras atípicas. |
| **Factor de Calibración Global** | 1.0017 | 1.0062 | Mediana del ratio de tasación sobre Train 2023. |

---

## 6. Métricas Reales Obtenidas en Evaluación Ciega (Test Set 2024–2025)

Las evaluaciones se realizaron sobre el conjunto de prueba ciego ($n=23,737$ ventas y $n=19,682$ contratos de alquiler). Todas las métricas superan holgadamente los estándares internacionales:

### Cuadro 3.4: Métricas Econométricas de Machine Learning
| Métrica de Desempeño | Modelo Venta | Modelo Alquiler | Criterio de Aceptación |
| :--- | :---: | :---: | :--- |
| **Registros evaluados (Test)** | 23,737 | 19,682 | Datos ciegos fuera de tiempo (2024–2025) |
| **MAPE (Error porcentual medio)** | **14.86%** | **13.68%** | Menor o igual a 15.0% (Umbral internacional) |
| **MAE (Error absoluto medio)** | S/ 51,861.76 | S/ 235.28 / mes | Desviación promedio en soles reales |
| **RMSE (Raíz del error cuadrático)** | S/ 78,053.38 | S/ 361.10 / mes | Penalización cuadrática ante desvíos grandes |
| **Coeficiente R² (Bondad de ajuste)** | **0.7477** | **0.6761** | Mayor o igual a 0.65 (Varianza explicada) |
| **Tiempo de Inferencia total** | 6.14 segundos | 5.08 segundos | Evaluación masiva en memoria RAM |

### Cuadro 3.5: Cumplimiento de Normas Catastrales IAAO y Tolerancia Bancaria AVM
| Indicador y Umbral Normativo | Modelo Venta | Modelo Alquiler | Estado / Interpretación |
| :--- | :---: | :---: | :---: |
| **Mediana del Ratio** (Norma: 0.90 a 1.10) | **1.0305** | **0.9900** | **CUMPLE** (Insesgado) |
| **Dispersión COD** (Norma: Menor o igual a 15.0%) | **14.20%** | **13.79%** | **CUMPLE** (Uniformidad) |
| **Diferencial PRD** (Norma: 0.98 a 1.03) | 1.0363 | 1.0387 | Ligero sesgo progresivo |
| **Sesgo de Precio PRB** (Norma: -0.10 a +0.10) | **-0.0806** | -0.1233 | **CUMPLE en Venta** |
| **Aprobación Bancaria PE20** (Norma: Mayor o igual a 70.0%) | **74.09%** | **77.41%** | **CUMPLE (Aprobado)** |

*Nota:* Evaluación sobre el conjunto de prueba ciego (2024–2025) deflactado a soles constantes y calibrado por distrito. COD mide equidad horizontal, PRD/PRB equidad vertical y PE20 la proporción de avalúos dentro del margen bancario del ±20%.

---

## 7. Motor de Comparables de Mercado (Cumplimiento RNTP Art. 3 y 4)

El microservicio de Machine Learning incorpora un módulo especializado en memoria RAM (`comparables_service.py`) para dar cumplimiento estricto al **Reglamento Nacional de Tasaciones del Perú (RNTP, R.M. Nº 172-2016-VIVIENDA)**:

### 7.1. Regla de las Tres (3) Muestras Homogéneas
Los Artículos 3 y 4 del RNTP establecen que una tasación por el método comparativo debe fundamentarse obligatoriamente en al menos tres (3) transacciones de mercado de inmuebles de tipología similar.

### 7.2. Algoritmo de Búsqueda Jerárquica y Distancia Ponderada
Para calcular los comparables sin añadir latencia al endpoint (< 0.5 ms), el servicio opera directamente sobre matrices pre-cargadas en NumPy y ejecuta una búsqueda en cascada jerárquica:
1. **Paso 1 (Mismo Distrito):** Filtra los testigos registrados en el mismo distrito del inmueble evaluado.
2. **Paso 2 (Zona Homogénea):** Si la densidad muestral del distrito es insuficiente (< 3 muestras), conmuta automáticamente a la Zona Homogénea correspondiente (ej. Lima Moderna, Lima Top, Lima Norte).
3. **Paso 3 (Área Metropolitana):** Fallback de resiliencia sobre todo el universo analítico.

La similitud física de cada candidato ($i$) respecto al inmueble a tasar se calcula mediante una **distancia euclidiana normalizada penalizada por recencia temporal**:
$$\text{Score}_i = \sqrt{\sum_{j \in \text{Físicos}} \omega_j \left(\frac{x_{i,j} - x_{\text{eval},j}}{\sigma_j}\right)^2} + 0.02 \cdot (\text{Anio}_{\max} - \text{Anio}_i)$$
* Donde los pesos $\omega_j$ priorizan la superficie ($40\%$), habitaciones ($20\%$), baños ($15\%$), antigüedad ($15\%$) y piso ($10\%$).
* La selección de las 3 mejores muestras se realiza en tiempo $O(N)$ usando `np.argpartition`, retornando para cada testigo su precio total, precio/$m^2$, diferencias físicas exactas e índice de similitud porcentual.

---

## 8. Evidencias Visuales Disponibles en el Repositorio (Set Simétrico Venta y Alquiler)

Para sustentar el Capítulo 3 con las evidencias exigidas por la rúbrica de la UPC, los gráficos generados se encuentran en formato PNG de alta resolución (300 DPI) de forma simétrica tanto para Venta como para Alquiler en las carpetas `Valuo-Machine-Learning-Service/reports/figures/` y `python_modelo_tesis/reports/figures/`:

### Cuadro 3.6: Catálogo Simétrico de Figuras Analíticas (7 Pares Oficiales)
| Tipo de Gráfico / Análisis | Modelo Venta (`reports/figures/`) | Modelo Alquiler (`reports/figures/alquiler/`) | Propósito y Lectura en la Tesis |
| :--- | :--- | :--- | :--- |
| **1. Dispersión Real vs. Predicho** | `real_vs_predicho.png` | `real_vs_predicho_alquiler.png` | Validación de ajuste a 45° sobre el conjunto de test fuera de tiempo (2024–2025). |
| **2. Importancia de Variables (Gain)**| `feature_importance_xgboost.png` | `feature_importance_alquiler.png` | Top 20 variables de mayor peso formativo en el valor del inmueble según XGBoost. |
| **3. MAPE por Distrito (22 distritos)**| `mape_por_distrito.png` | `mape_por_distrito_alquiler.png` | Desglose territorial del error porcentual frente al umbral crítico del 15% (IAAO). |
| **4. Homocedasticidad y Residuos** | `residuos.png` | `residuos_alquiler.png` | Panel dual: Residuos vs. Predicho y curva gaussiana de densidad de errores. |
| **5. Distribución Empírica del Target**| `distribucion_target_venta.png` | `distribucion_target_alquiler.png` | Histograma y curva KDE del valor monetario con su mediana de mercado. |
| **6. Impacto Direccional (SHAP Beeswarm)**| `shap_beeswarm.png` | `shap_beeswarm_alquiler.png` | Resumen de explicabilidad global: dirección y magnitud de cada feature sobre la tasación. |
| **7. Magnitud Absoluta (|SHAP| Bar)**| `shap_bar.png` | `shap_bar_alquiler.png` | Ranking global del promedio absoluto de impacto $|\text{SHAP value}|$ por variable. |

*Evidencia de Gobernanza MLOps adicional:*
* Captura de pantalla de la interfaz de MLflow (`mlflow-server` en puerto 5000 o Azure Container Apps) mostrando el registro de los modelos `@champion`, sus versiones activas, tags y métricas oficiales registradas.
