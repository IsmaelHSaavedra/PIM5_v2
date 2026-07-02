# PIM5 - Modelo predictivo de pago a tiempo

## Descripción del proyecto

Este proyecto corresponde al Proyecto Integrador del Módulo 5. El objetivo es desarrollar un flujo de trabajo de Machine Learning aplicado a información histórica de créditos, con el fin de construir un modelo capaz de predecir si un usuario realizará su pago a tiempo.

El proyecto sigue una estructura orientada a MLOps, incluyendo carga de datos, análisis exploratorio, ingeniería de características, entrenamiento de modelos supervisados, evaluación de métricas, monitoreo de data drift y una aplicación visual en Streamlit para consultar resultados.

## Caso de negocio

La empresa financiera requiere anticipar el comportamiento de pago de nuevos usuarios a partir de datos históricos de crédito. Para esto, se construye un modelo supervisado que estima la probabilidad de pago a tiempo.

Este tipo de solución puede apoyar procesos internos como:

- Evaluación de riesgo crediticio.
- Priorización de seguimiento a clientes.
- Identificación temprana de posibles incumplimientos.
- Monitoreo del comportamiento de los datos a través del tiempo.

## Estructura del proyecto

```txt
mlops_pipeline/
│
├── src/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_deploy.py
│   ├── model_monitoring.py
│   └── streamlit_app.py
│
├── Base_de_datos.csv
├── Base_de_datos_preparada.csv
├── modelo_pago_atiempo.pkl
├── resultados_modelos.csv
├── drift_report.csv
├── drift_report.json
├── requirements.txt
├── .gitignore
└── readme.md
```

## Flujo de trabajo

### 1. Carga de datos

El notebook `Cargar_datos.ipynb` se utiliza para cargar la base original, revisar su estructura general y preparar el punto de partida del proyecto.

### 2. Comprensión y análisis exploratorio

El notebook `comprension_eda.ipynb` contiene el análisis exploratorio de datos, incluyendo:

- Exploración inicial.
- Revisión de tipos de variables.
- Análisis univariable.
- Análisis bivariable.
- Análisis multivariable.
- Revisión de la variable objetivo `Pago_atiempo`.

### 3. Ingeniería de características

El script `src/ft_engineering.py` realiza el proceso de preparación de variables para el modelamiento. Entre las tareas principales se incluyen:

- Limpieza de columnas.
- Tratamiento de valores faltantes.
- Transformación de fechas.
- Creación de variables derivadas.
- Preparación final del archivo `Base_de_datos_preparada.csv`.

### 4. Entrenamiento y evaluación de modelos

El script `src/model_training_evaluation.py` entrena y compara diferentes modelos supervisados:

- Logistic Regression.
- Decision Tree.
- Random Forest.

Las métricas utilizadas para evaluar el desempeño fueron:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Validación cruzada.

El mejor modelo se guarda como:

```txt
modelo_pago_atiempo.pkl
```

Los resultados comparativos se guardan en:

```txt
resultados_modelos.csv
```

### 5. Monitoreo de data drift

El script `src/model_monitoring.py` compara una muestra histórica de referencia contra una muestra reciente para detectar cambios en la distribución de las variables.

El monitoreo utiliza métricas como:

- PSI.
- Kolmogorov-Smirnov.

Los reportes generados son:

```txt
drift_report.csv
drift_report.json
```

En la ejecución realizada se detectó drift en las siguientes variables:

```txt
mes_prestamo
trimestre_prestamo
promedio_ingresos_datacredito
```

Este resultado indica que algunas variables presentan cambios relevantes entre los datos históricos y recientes, por lo que deben monitorearse antes de usar el modelo en un entorno productivo.

### 6. Aplicación en Streamlit

El archivo `src/streamlit_app.py` implementa una aplicación visual para consultar el modelo y revisar sus resultados.

La aplicación incluye tres secciones principales:

- Predicción individual.
- Monitoreo de drift.
- Resultados del modelo.

La app permite ingresar datos de un caso individual, obtener una clasificación de pago a tiempo y visualizar los resultados del monitoreo de drift y del entrenamiento.

## Resultados principales

Los modelos obtuvieron métricas muy altas. En particular, Decision Tree y Random Forest alcanzaron valores de 1.0000 en varias métricas de prueba.

Esto puede indicar un desempeño elevado del modelo, aunque también se considera importante revisar posible data leakage, ya que algunas variables podrían contener información demasiado cercana al resultado final del pago. Por esta razón, el monitoreo y la revisión de variables se documentan como parte del flujo MLOps.

## Instalación

Crear y activar un entorno virtual:

```bash
python -m venv venv
```

Activar el entorno en Windows:

```bash
venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución de scripts

Ejecutar ingeniería de características:

```bash
python src/ft_engineering.py
```

Ejecutar entrenamiento y evaluación:

```bash
python src/model_training_evaluation.py
```

Ejecutar monitoreo de data drift:

```bash
python src/model_monitoring.py
```

Ejecutar la aplicación en Streamlit:

```bash
streamlit run src/streamlit_app.py
```

La aplicación se abre localmente en:

```txt
http://localhost:8501
```
## API con FastAPI

El archivo `src/model_deploy.py` disponibiliza el modelo entrenado mediante una API construida con FastAPI.

La API incluye los siguientes endpoints:

```txt
GET /
GET /health
POST /predict
```

Para ejecutar la API localmente:

```bash
uvicorn src.model_deploy:app --reload --port 5000
```

La documentación interactiva se puede consultar en:

```txt
http://127.0.0.1:5000/docs
```

## Docker

El proyecto incluye un `Dockerfile` para crear una imagen con la API y sus dependencias.

Construir la imagen:

```bash
docker build -t pim5-api .
```

Ejecutar el contenedor:

```bash
docker run -p 5000:5000 pim5-api
```

Luego abrir:

```txt
http://127.0.0.1:5000/docs
```

El mapeo `5000:5000` conecta el puerto local de la computadora con el puerto interno del contenedor.

## Versionamiento

El proyecto utiliza ramas separadas para representar distintos entornos de trabajo:

- `developer`: desarrollo inicial y cambios activos.
- `certification`: validación de versiones.
- `master`: versión estable del proyecto.

Versiones trabajadas:

- `V1.0.0`: estructura inicial del proyecto.
- `V1.0.1`: carga de datos y comprensión EDA.
- `V1.1.0`: ingeniería de características, modelamiento y evaluación.
- `V1.1.1`: monitoreo de data drift, aplicación Streamlit y documentación.

## Hallazgos

Durante el desarrollo del proyecto se identificaron los siguientes puntos:

- La variable objetivo `Pago_atiempo` permite plantear el problema como clasificación supervisada.
- Los modelos supervisados alcanzaron métricas muy altas.
- La presencia de métricas perfectas requiere revisar posible data leakage.
- El monitoreo detectó drift en variables temporales y financieras.
- La app en Streamlit permite visualizar de forma práctica el modelo, sus resultados y el monitoreo.
- La estructura del proyecto facilita la reproducibilidad y el seguimiento del flujo MLOps.

## Conclusión

El proyecto integra las etapas principales de un flujo MLOps aplicado a un modelo de clasificación crediticia. Se desarrolló la carga y comprensión de datos, ingeniería de características, entrenamiento de modelos, evaluación comparativa, monitoreo de data drift y una aplicación visual en Streamlit.

Aunque los resultados del modelo son altos, se documenta la necesidad de revisar variables sensibles a data leakage y mantener monitoreo continuo para asegurar que el modelo siga siendo útil ante cambios en los datos.