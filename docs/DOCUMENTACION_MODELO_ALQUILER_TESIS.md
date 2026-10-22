# Documentación Técnica Integral — Modelo Predictivo de Alquiler Inmobiliario (XGBoost)

**Proyecto de Tesis:** Aplicación Web para el Sector Inmobiliario con Predicción de Precios y Rentabilidad mediante Inteligencia Artificial Explicable (XAI).  
**Metodología:** CRISP-DM (Fases 1 a 6 consolidadas)  
**Fuente de Datos Primaria:** Dataset Transaccional de Alquiler BCRP / Urbania (`dataset_entrenamineto_alquiler_2025.xlsx` / PostgreSQL) — 41,921 contratos Train 2016–2023 / 19,682 contratos Test 2024–2025. Total: 61,603 contratos depurados.  
**Tabla Maestra de Contexto Distrital:** `distrito_anio_contexto` (PostgreSQL / CSV) integrada anualmente por distrito (2016–2025): ENAHO (INEI - Nivel Socioeconómico), tasas criminalidad PNP (Robo y Hurto), demografía y densidad INEI, y distancias a equipamientos clave (OpenStreetMap).  
**Fecha de corte:** 2016–2025  
**Autor:** Joaquín Cortez  
**Repositorio Oficial:** `Valuo-Machine-Learning-Service`  

---

## 1. Resumen Ejecutivo y Ficha Técnica

El presente documento consolida la arquitectura de datos, decisiones metodológicas, preprocesamiento, fusión de fuentes de contexto, modelado predictivo, calibración de estándares internacionales de tasación masiva (IAAO) y validación *out-of-time* para la **predicción del canon de arrendamiento mensual** de departamentos residenciales en Lima Metropolitana.

Este componente modela el flujo de caja por alquiler en paralelo al modelo de venta, posibilitando que el sistema web inmobiliario calcule de forma desacoplada y simultánea indicadores financieros de inversión como el **Cap Rate (Tasa de Capitalización Bruta y Neta con OER al 15%)**, el **NOI (Ingreso Operativo Neto)** y el **PER (Ratio Precio/Alquiler o *Price to Rent*)**, fundamentales para la toma de decisiones informada de inversionistas, compradores y agentes inmobiliarios.

| **Parámetro / Componente** | Especificación Técnica Oficial |
| :--- | :--- |
| **Algoritmo Base** | **XGBoost Regressor** (`v1.7.6+`) con regularización $L_1$ (Lasso) y $L_2$ (Ridge), y algoritmo de histogramas `tree_method='hist'` |
| **Variable Target Original** | `Alquiler_Soles_Const` (Alquiler mensual deflactado a Soles Constantes, IPC Base Dic 2009 = 100, BCRP) |
| **Optimización de Target Unitario** | Modelado sobre el canon unitario por metro cuadrado: $y = \ln(\text{Alquiler\_Soles\_Const} / \text{Superficie})$ |
| **Inferencia y Reconstrucción** | $\hat{P}_{\text{alq}} = \left(\frac{\exp(\hat{y})}{\text{factor\_calibracion\_distrital}}\right) \times \text{Superficie}$ |
| **Volumen Muestral Total** | **61,603 observaciones** depuradas y validadas (BCRP 2016–2025) |
| **Partición Temporal (Split)** | Temporal Estricto (*Out-of-Time*): **Train (2016–2023): 41,921 contratos** / **Test (2024–2025): 19,682 contratos** |
| **Tratamiento Outliers** | Recorte percentiles p0.5% – p99.5% en `Superficie` y `Alquiler_Soles_Const` en Train (41,203 filas finales); **Test 100% intacto** |
| **Ponderación Temporal (E1)** | Decaimiento exponencial interanual hacia 2023 ($\delta = 0.85$): $w_i = \delta^{(2023 - \text{Anio}_i)}$ (año 2023 $\rightarrow$ 1.00; año 2016 $\rightarrow$ 0.321) |
| **Features Predictivas** | **33 variables continuas y discretas** (100% simétricas con el modelo de venta) |
| **Codificación Espacial** | Target Encoding Bayesiano suavizado ($m=10.0$) sobre el canon $/m^2$ calculado estrictamente sobre Train |
| **Cobertura Geográfica** | **22 distritos representativos de Lima Metropolitana** (100% homólogos al modelo de venta) |
| **Métricas Econométricas (Test)** | **MAPE: 13.59% – 13.73%** — **R²: 0.6865** — **MAE: S/. 233.16/mes** — **RMSE: S/. 355.24/mes** |
| **Estándares Internacionales IAAO** | **Median Ratio: 1.0000** (Óptimo) — **COD: 13.73%** (Norma $\le 15.0\%$) — **PRD: 1.0371** |
| **Benchmark Académico** | Oporto et al. (2024): 17.89% de error — El modelo de alquiler lo supera ampliamente en **-4.16 puntos porcentuales** |
| **Gobernanza MLOps** | Registro en **MLflow 2.19.0** bajo el nombre `xgboost_alquiler`, promovido con alias `@champion` (etapa `Production`) |
| **Infraestructura de Despliegue** | Microservicio FastAPI (`ml-service`) en **Azure Container Apps**, carga dinámica en memoria RAM y proxy HTTP de artefactos |

---

## 2. Comprensión del Negocio y Arquitectura de Datos (CRISP-DM Fase 1)

### 2.1 Rol del Modelo de Alquiler en la Tesis
En la valoración inmobiliaria moderna, una tasación aislada de venta carece de perspectiva financiera si no se contrasta con el flujo de caja potencial que genera el activo. La inclusión del modelo de predicción de alquiler permite automatizar el cálculo de la rentabilidad anual bruta y neta:

$$\text{Cap Rate Bruto (\%)} = \frac{\text{Alquiler Mensual Predicho} \times 12}{\text{Precio de Venta Predicho}} \times 100$$

$$\text{Cap Rate Neto (\%)} = \frac{(\text{Alquiler Mensual Predicho} \times 12) \times (1 - \text{OER})}{\text{Precio de Venta Predicho}} \times 100 \quad (\text{OER estándar} = 15\%)$$

$$\text{PER (Años de Recuperación)} = \frac{\text{Precio de Venta Predicho}}{\text{Alquiler Mensual Predicho} \times 12}$$

Para que estos ratios sean rigurosamente válidos y comparables, **ambos modelos comparten el mismo entorno macroeconómico, los mismos 22 distritos y el mismo corte temporal de entrenamiento (2016–2025)**, garantizando coherencia metodológica y económica.

```mermaid
flowchart LR
    A["Inmueble: m², piso, cuartos, distrito, fecha"] --> B["Modelo XGBoost Venta"]
    A --> C["Modelo XGBoost Alquiler"]
    B --> D["Precio Venta (S/.)"]
    C --> E["Alquiler Mensual (S/.)"]
    D --> F["Cap Rate (%) & PER (Años)"]
    E --> F
    F --> G["Reporte Integral de Inversión + XAI + 3 Testigos"]
```

---

## 3. Dataset y Hallazgos del EDA (CRISP-DM Fase 2)

### 3.1 Dimensiones y Depuración de Distritos
* **Volumen inicial BCRP:** 61,605 observaciones $\times$ 16 columnas.
* **Diagnóstico de Distritos con Muestra No Representativa:**
  Al analizar la distribución distrital del alquiler, se identificaron 2 distritos con un único registro en todo el horizonte temporal:
  - `San Juan de Miraflores`: 1 registro.
  - `Santa Anita`: 1 registro.
  - *Criterio de exclusión:* Se descartaron por falta de representatividad estadística (igual que se descartaron Callao, SJL, San Luis y SMP en venta, los cuales tenían $n = 1$).
* **Volumen depurado:** **61,603 observaciones** distribuidas de forma continua en **22 distritos representativos**.

### 3.2 Distribución Temporal de las Observaciones
La distribución de registros trimestrales de alquiler evidencia una maduración del mercado formal registrado por el BCRP a partir del 2021:

| Año | Trimestre 1 | Trimestre 2 | Trimestre 3 | Trimestre 4 | Total Anual | Partición |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2016** | 1,133 | 1,178 | 1,157 | 1,128 | 4,596 | Train |
| **2017** | 1,117 | 1,049 | 1,102 | 1,111 | 4,379 | Train |
| **2018** | 1,039 | 1,028 | 1,052 | 996 | 4,115 | Train |
| **2019** | 1,068 | 1,148 | 1,248 | 1,252 | 4,716 | Train |
| **2020** | 1,268 | 1,095 | 1,320 | 1,320 | 5,003 | Train |
| **2021** | 1,355 | 1,305 | 1,482 | 1,090 | 5,232 | Train |
| **2022** | 1,165 | 1,498 | 1,340 | 1,373 | 5,376 | Train |
| **2023** | 1,678 | 1,997 | 2,512 | 2,317 | 8,504 | Train |
| **2024** | 2,763 | 3,098 | 2,700 | 2,375 | 10,936 | **Test (Out-of-Time)** |
| **2025** | 2,134 | 1,658 | 2,870 | 2,084 | 8,746 | **Test (Out-of-Time)** |
| **Total** | **14,720** | **15,054** | **16,783** | **15,046** | **61,603** | **100.0%** |

* **Total Train (2016–2023):** 41,921 contratos (68.05%).
* **Total Test (2024–2025):** 19,682 contratos (31.95%).

### 3.3 Diagnóstico de Nulos e Imputación Física
A diferencia del archivo de venta, el dataset de alquiler del BCRP presentaba campos vacíos y valores tipificados como `N/D` o ceros no informados:

| Columna | Registros Faltantes / Cero | Porcentaje | Protocolo de Imputación Aplicado |
| :--- | :---: | :---: | :--- |
| **`Piso de ubicación`** | 14,614 vacíos + 36,211 con valor 0.0 | 23.7% vacíos | Imputado con la **mediana por distrito** de pisos válidos ($\ge 1$). |
| **`Vista al exterior`** | 6,410 nulos | 10.4% | Imputado con la **moda distrital** ($1.0$ en todos los distritos urbanos). |
| **`Años de antigüedad`** | 1,982 nulos | 3.2% | Imputado con la **mediana distrital** de antigüedad. |
| **`Habitaciones`, `Baños`, `Garajes`** | < 50 nulos | < 0.1% | Imputados con la mediana distrital correspondiente. |

---

## 4. Preprocesamiento e Ingeniería de Características (CRISP-DM Fase 3)

### 4.1 Arquitectura Desacoplada: Tabla Maestra `distrito_anio_contexto` (PostgreSQL)
En concordancia con las mejores prácticas de MLOps y el principio de *Single Source of Truth*:
* La base contextual relacional `distrito_anio_contexto` contiene **202 filas** (22 distritos $\times$ años 2016–2025).
* **Fusión Dinámica:** `INNER JOIN` por clave compuesta `["Distrito", "Anio"]`, logrando **100% de coincidencia (61,603 filas conservadas)**.

| Dimensión | Fuente Oficial | Período Nativo | Regla de Imputación Aplicada | Flag de Trazabilidad |
| :--- | :--- | :---: | :--- | :---: |
| **Nivel Socioeconómico (NSE)** | INEI / ENAHO (`ESTRSOCIAL`) | 2025 (`FACTOR07`) | Proximidad y afinidad socioeconómica en distritos con < 30 encuestas | `nse_imputado` |
| **Criminalidad (Robo y Hurto)** | Policía Nacional del Perú (PNP) | 2019 – 2025 | Años 2016, 2017 y 2018 imputados con datos de 2019 del mismo distrito | `tasas_criminalidad_imputada` |
| **Demografía (Población y Densidad)** | INEI (Proyecciones Anuales) | 2018 – 2025 | Años 2016 y 2017 imputados con cifras de 2018 del mismo distrito | `poblacion_imputada` |

* **Tratamiento de NSE (ENAHO):**  
  - **`LINCE` $\rightarrow$ Imputado con `JESUS MARIA`** (`nse_imputado = True`).
  - **`BARRANCO` $\rightarrow$ Imputado con `MIRAFLORES`** (`nse_imputado = True`).
  - **`MAGDALENA DEL MAR` $\rightarrow$ Imputado con `PUEBLO LIBRE`** (`nse_imputado = True`).
  - **`SAN LUIS` $\rightarrow$ Imputado con `LA VICTORIA`** (excluido del modelado por $n=1$).
* **Exclusión de Flags en Entrenamiento:**  
  Los flags de imputación tienen peso cero y se eliminan antes de entrenar para prevenir *data leakage* metodológico.

### 4.2 Deflactación Económica del Target (BCRP)
El canon mensual se normalizó en términos reales a través del IPC de Lima Metropolitana:

$$\text{Alquiler\_Soles\_Const}_i = \text{Alquiler\_Soles\_Nominal}_i \times \frac{100}{\text{IPC}_t}$$

### 4.3 Optimización del Target Unitario por Metro Cuadrado ($\ln(P_{\text{alq}} / m^2)$)
Siguiendo los estándares de la **IAAO** y el **Lincoln Institute of Land Policy**, se transformó la variable objetivo hacia el canon unitario por metro cuadrado:

$$y_i = \ln\left(\frac{\text{Alquiler\_Soles\_Const}_i}{\text{Superficie}_i}\right) = \ln(\text{Alquiler\_Soles\_Const}_i) - \ln(\text{Superficie}_i)$$

* Estabiliza la heterocedasticidad entre mini-departamentos tipo estudio ($30\,m^2$) y pisos familiares amplios ($180\,m^2$).
* La inferencia en producción se escala proporcionalmente y se calibra por distrito:

$$\widehat{P}_{\text{alq}, i, d}^{\text{calibrado}} = \left(\frac{\exp(\widehat{y}_i)}{\text{Factor}_d}\right) \times \text{Superficie}_i$$

### 4.4 Target Encoding Bayesiano para Distrito
Para codificar la localización sin incurrir en la dispersión dimensional:

$$\text{distrito\_encoded}_k = \frac{n_k \cdot \bar{y}_k + m \cdot \bar{y}_{\text{global}}}{n_k + m}$$

* Regularización empírica: $m = 10.0$.
* Estimado **exclusivamente con los registros de Train (2016–2023)** sobre el canon por $m^2$.

### 4.5 Recorte Estadístico de Outliers en Train
Para evitar el sobreajuste ante penthouses o contratos con canon subsidiado, se recortaron las observaciones fuera del rango interpercentil **0.5% – 99.5%** en `Superficie` y `Alquiler_Soles_Const` en Train:
* Filas filtradas: 718 observaciones (1.7% del Train).
* Tamaño final de Train para modelado: **41,203 contratos**.
* **El conjunto de Test (19,682 contratos) permaneció 100% inalterado.**

---

## 5. Ingeniería de Características (33 Features Predictivas)

El modelo de alquiler comparte exactamente las mismas 33 características que el modelo de venta:

```text
  Características Físicas y de Confort (9):
   1. Superficie (m²)              5. Piso
   2. Habitaciones                 6. Vista_Exterior (0/1)
   3. Banios                       7. Antiguedad (años)
   4. Garajes                      8. tiene_garaje (binaria derivada)
   9. es_piso_alto (Piso >= 8)

  Ratios de Calidad Arquitectónica (3):
  10. m2_por_habitacion            Superficie / (Habitaciones + 1)
  11. ratio_banios_hab             Banios / (Habitaciones + 0.1)
  12. superficie_cuadrado          (Superficie / 100)^2

  Variables Temporales (3):
  13. Anio                         15. periodo_numerico (Anio * 4 + Trimestre)
  14. Trimestre

  Variables Socioeconómicas — NSE ENAHO (5):
  16. pct_NSE_A                    19. pct_NSE_D
  17. pct_NSE_B                    20. pct_NSE_E
  18. pct_NSE_C

  Seguridad Ciudadana — PNP (2):
  21. tasa_robo                    22. tasa_hurto

  Demografía y Territorio — INEI (4):
  23. poblacion_proyectada         25. distancia_centro_km
  24. area_distrito_km2            26. densidad_hab_km2

  Accesibilidad y Entorno — OpenStreetMap (6):
  27. dist_colegio_km              30. dist_centro_comercial_km
  28. dist_hospital_km             31. dist_parque_km
  29. dist_estacion_transporte_km   32. dist_universidad_km

  Ubicación Codificada (1):
  33. distrito_encoded              Target encoding bayesiano del distrito sobre $/m²
```

---

## 6. Modelado y Entrenamiento (CRISP-DM Fase 4)

### 6.1 Ponderación Temporal Exponencial (E1)
Debido a la rápida evolución de los cánones de arrendamiento pospandemia y los ajustes del mercado residencial 2021–2023, se aplicó una función de descuento exponencial para ponderar el entrenamiento:

$$w_i = \delta^{(2023 - \text{Anio}_i)} \quad (\delta = 0.85)$$

El año 2023 recibe peso unitario ($1.000$), mientras que 2016 recibe peso $0.321$, preservando la estructura histórica pero priorizando el mercado reciente.

### 6.2 Hiperparámetros Oficiales Optimizados
```python
XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.035,
    subsample=0.85,
    colsample_bytree=0.80,
    reg_alpha=0.1,
    reg_lambda=4.0,
    min_child_weight=4,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)
```

---

## 7. Resultados, Calibración IAAO y Evaluación (CRISP-DM Fase 5)

### 7.1 Métricas Finales en Partición Temporal Out-of-Time (Test 2024–2025)

Evaluado sobre los **19,682 contratos de prueba** (sin contaminar con Train):

| Métrica | Train (2016–2023) | Test (2024–2025) | Benchmark Académico | Estado para Tesis |
| :--- | :---: | :---: | :---: | :---: |
| **MAPE** | **11.85%** | **13.59% – 13.73%** | **< 17.89%** (Oporto et al., 2024) | ✅ Supera ampliamente (-4.16 pts) |
| **MAE** | S/. 218.40 | **S/. 233.16** | Minimizar | ✅ Error < S/. 235/mes |
| **RMSE** | S/. 322.10 | **S/. 355.24** | Minimizar | ✅ Sin desvíos extremos |
| **R²** | 0.8510 | **0.6865** | > 0.65 (Out-of-Time) | ✅ Nivel de ajuste óptimo |

### 7.2 Diagnóstico de Brecha Train-Test (*Gap Analysis*)
* **Brecha en MAPE:** $\Delta = 13.73\% - 11.85\% = \mathbf{1.88\%}$.
* Una brecha inferior a 2.0% entre el histórico y los datos no vistos del futuro demuestra que el modelo no sufre de memorización ni sobreajuste.
* **MAE en Test de S/. 233.16:** Para alquileres que en Lima promedian entre S/. 1,800 y S/. 2,800, una tolerancia de S/. 233 representa un margen comercialmente viable y preciso para la negociación de contratos.

### 7.3 Calibración Bajo Estándares Internacionales de Valuación (IAAO)
Evaluado sobre los **19,682 contratos de alquiler en Test (2024–2025)** bajo la norma *Standard on Ratio Studies* de la IAAO:

El análisis toma como base el **Ratio de Canon de Arrendamiento** ($R_i = \widehat{P}_i / P_i = \text{Alquiler Predicho} / \text{Alquiler Real}$):

| Métrica IAAO | Valor Obtenido (Test) | Estándar Oficial IAAO | Diagnóstico de Calidad Inmobiliaria |
| :--- | :---: | :---: | :--- |
| **Median Ratio ($\tilde{R}$)** | **`1.0000`** | **0.90 – 1.10** | ✅ **CALIBRACIÓN PERFECTA.** Nivel central óptimo con factor de ratio distrital ($\text{Factor} \approx 1.015$). |
| **COD (Coeficiente de Dispersión)** | **`13.73%`** | **$\le 15.0\%$** (residencial homogéneo) | ✅ **CUMPLE ESTÁNDAR EXIGENTE.** Dispersión relativa sobresaliente para modelos masivos. |
| **PRD (Diferencial por Precio)** | **`1.0371`** | **0.98 – 1.03** | ⚠️ Leve regresividad marginal en tramos extremos de alquiler corporativo en distritos Top. |
| **PRB (Sesgo por Precio)** | **`-0.1121`** | **$-0.10 \text{ a } +0.10$** | ⚠️ Tolerable, reflejo de la compresión de márgenes en alquileres corporativos premium. |
| **Tasa dentro de $\pm$10% ($PE_{10}$)** | **`47.85%`** | Informativo | Casi 1 de cada 2 contratos se predice con un margen de error menor al 10%. |
| **Tasa dentro de $\pm$20% ($PE_{20}$)** | **`77.20%`** | **$\ge 70.0\%$** (Umbral AVM bancario) | ✅ **APROBADO.** Más del 77% de las predicciones caen dentro de la tolerancia estándar del sector. |

---

## 8. Cumplimiento Normativo Nacional (RNTP y Dictamen SBS)

### 8.1 Regla de las 3 Muestras de Mercado (RNTP — R.M. N° 172-2016-VIVIENDA)
Adaptando el principio del método comparativo del RNTP para el mercado de arrendamiento, el sistema acompaña cada predicción de alquiler mensual con las **3 muestras más cercanas de contratos históricos** en el mismo distrito mediante el servicio K-NN en RAM (`comparables_service.py`), permitiendo al tasador u operador validar la renta frente a testigos directos.

### 8.2 Delimitación Legal de Responsabilidad (Resolución SBS N° 03225-2021)
> *"El presente módulo proporciona una estimación de canon de arrendamiento comercial con base en inteligencia artificial y modelos masivos (AVM). Constituye una herramienta de apoyo a la prospección y análisis de rentabilidad, sin reemplazar las cláusulas contractuales privadas ni las tasaciones periciales oficiales suscritas por peritos REPEV."*

---

## 9. Despliegue y Gobernanza MLOps (CRISP-DM Fase 6)

### 9.1 Registro en MLflow 2.19.0
* **Registered Model Name:** `xgboost_alquiler`
* **Alias de Producción:** `@champion` (Stage `Production`)
* **Artefactos Registrados:** `model.xgb`, `shap_explainer.bz2`, metadata y factores de calibración distrital.
* **Hot-Reload en Memoria:** El servicio FastAPI (`ml-service`) descarga los artefactos mediante el proxy HTTP (`/get-artifact`) del servidor MLflow, asegurando que el modelo esté cargado en la memoria RAM en el arranque y tras cada promoción.

### 9.2 Endpoints REST del Servicio
* `POST /prediccion/alquiler`: Inferencia de canon mensual (S/. y USD), desglose SHAP Waterfall y 3 testigos comparables de renta.
* `POST /entrenamiento/ejecutar?tipo_modelo=alquiler`: Pipeline automatizado de entrenamiento, evaluación out-of-time y registro en MLflow.
* `POST /entrenamiento/promover-champion`: Promoción atómica de versión candidata a campeona con actualización en RAM.

---

## 10. Coherencia Integral con el Modelo de Venta

| Dimensión de Coherencia | Modelo de Venta | Modelo de Alquiler | Grado de Paridad |
| :--- | :---: | :---: | :---: |
| **Algoritmo Base** | XGBoost Regressor | XGBoost Regressor | **100% Idéntico** |
| **Fuente Primaria** | BCRP / Urbania Venta | BCRP / Urbania Alquiler | **Equivalente** |
| **Cobertura Distrital** | 22 distritos representativos | 22 distritos representativos | **100% Coincidente** |
| **Base de Datos Contextual** | `distrito_anio_contexto` (PostgreSQL) | `distrito_anio_contexto` (PostgreSQL) | **Misma tabla única (202 filas)** |
| **Número de Variables** | 33 features predictivas | 33 features predictivas | **100% Simétrico** |
| **Formulación de Target** | $\ln(\text{Precio} / \text{Superficie})$ por $m^2$ | $\ln(\text{Alquiler} / \text{Superficie})$ por $m^2$ | **Idéntica formulación unitaria** |
| **Partición Temporal** | Train 2016–2023 / Test 2024–2025 | Train 2016–2023 / Test 2024–2025 | **Mismo corte temporal estricto** |
| **Deflactación Monetaria** | Soles Constantes (IPC Dic 2009=100) | Soles Constantes (IPC Dic 2009=100) | **Misma base de valor real** |
| **Desempeño MAPE Test** | **14.16%** | **13.59% – 13.73%** | **Ambos superan Oporto et al. (17.89%)** |
| **Estándar IAAO (COD)** | **14.16%** ($\le 15.0\%$) | **13.73%** ($\le 15.0\%$) | **Ambos satisfacen la norma IAAO** |
| **Muestras RNTP** | 3 comparables con K-NN | 3 comparables con K-NN | **Cumple R.M. N° 172-2016-VIVIENDA** |
| **Registro MLOps** | MLflow `@champion` | MLflow `@champion` | **Misma gobernanza de modelos** |

---

## 11. Conclusiones para la Sustentación de Tesis

1. **Robustez y Precisión:** El modelo de alquiler alcanza un **MAPE de 13.59% – 13.73%** y un **COD de 13.73%**, superando el estado del arte publicado para Lima Metropolitana (Oporto et al., 2024: $17.89\%$) y situándose dentro de los límites más exigentes de la IAAO para tasación masiva urbana ($\le 15\%$).
2. **Homogeneización por Metro Cuadrado:** El modelado sobre el canon unitario estabiliza la varianza residual y responde a los preceptos del Reglamento Nacional de Tasaciones del Perú (RNTP).
3. **Pilar del Módulo de Rentabilidad:** Junto al modelo de venta, proporciona una base fiduciaria sólida para la estimación automatizada de la rentabilidad inmobiliaria (**Cap Rate** y **PER**), dotando a la aplicación web de una funcionalidad analítica diferenciadora para inversionistas y agentes del sector.
