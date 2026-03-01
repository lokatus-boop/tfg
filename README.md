# Padel Analytics 🎾📊

**Padel Analytics** es una herramienta avanzada de análisis de vídeo para pádel que utiliza Inteligencia Artificial (Computer Vision) para extraer métricas de rendimiento, trayectorias y estadísticas de juego a partir de grabaciones de vídeo estándar.

![Padel Analisis TFG](https://github.com/LuismiSanz/padel_analisis/raw/main/docs/demo.gif)


## 🚀 Características Principales

*   **Detección de Jugadores y Pelota**: Rastreo automático mediante modelos YOLO y TrackNet.
*   **Mapa de Calor 2D**: Proyección de la posición de los jugadores en una pista virtual.
*   **Velocidad de Bola**: Estimación de la velocidad de los golpes.
*   **Clasificación de Golpes**: Detección automática de Voleas y Drives.
*   **Reportes Exportables**: Descarga de datos en CSV y vídeos procesados.
*   **Interfaz Web**: Dashboard interactivo y fácil de usar con Streamlit.

---

## 🛠️ Instalación y Uso


### 0. Descargar Pesos (Modelos)

**IMPORTANTE**: Antes de ejecutar la aplicación, necesitas descargar los modelos de Inteligencia Artificial (pesos).

1.  Descarga los archivos desde este enlace: [Google Drive - Model Weights](https://drive.google.com/drive/folders/1joO7w1Am7B418SIqGBq90YipQl81FMzh)
2.  Descomprime o coloca los archivos dentro de la carpeta `weights/` en la raíz del proyecto.
    *   La estructura debe quedar así: `padel_analisis/weights/players_detection/yolov8m.pt`, etc.

### Opción A: Docker (Recomendado)

Esta opción es la **más sencilla y robusta**. Funciona en:
*   **Windows** (con Docker Desktop + WSL2).
*   **Mac** (Intel y Apple Silicon M1/M2/M3).
*   **Linux**.

No necesitas instalar Python ni CUDA manualmente.

1.  **Instala Docker Desktop**: [Descargar aquí](https://www.docker.com/products/docker-desktop/).
2.  **Clona el repositorio**:
    ```bash
    git clone https://github.com/LuismiSanz/padel_analisis.git
    cd padel_analisis
    ```
3.  **Ejecuta la aplicación**:
    ```bash
    docker-compose up --build
    ```
4.  Abre tu navegador y ve a: `http://localhost:8501`

### Opción B: Instalación Local (Python)

Si prefieres ejecutarlo nativamente (requiere Python 3.10+ y preferiblemente GPU NVIDIA):

1.  **Clona el repositorio**:
    ```bash
    git clone https://github.com/LuismiSanz/padel_analisis.git
    cd padel_analisis
    ```
2.  **Crea un entorno virtual** (opcional pero recomendado):
    ```bash
    conda create -n padel_analytics python=3.10
    conda activate padel_analytics
    ```
3.  **Instala las dependencias**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ejecuta la aplicación**:
    ```bash
    streamlit run app.py
    ```

---

## 📖 Guía de Uso (Streamlit)

Una vez abierta la aplicación en el navegador, sigue estos pasos:

### 1. Cargar Datos
Tienes dos opciones al inicio:
*   **Subir Video**: Carga un archivo `.mp4` o `.avi` de un partido. El sistema procesará el vídeo (puede tardar unos minutos dependiendo de tu GPU).
*   **Cargar Reporte CSV**: Si ya procesaste un vídeo anteriormente, puedes subir el archivo `padel_analytics_report.csv` para ver las estadísticas al instante sin esperar.

### 2. Configuración de Detección
*   **Reutilizar detección de pista**: Si vas a procesar varios vídeos grabados desde la **misma posición exacta** (trípode), marca esta casilla. Esto ahorrará tiempo reutilizando la calibración de la pista del vídeo anterior.

### 3. Analítica y Visualización
Una vez procesado, verás:
*   **Vídeo Procesado**: El vídeo original con las cajas de detección y el esqueleto de los jugadores superpuestos.
*   **Mapa 2D**: Una representación cenital de la pista con la posición de los jugadores y la pelota.
*   **Gráficas de Velocidad**: Evolución de la velocidad de cada jugador a lo largo del tiempo.
*   **Estadísticas**: Distancia recorrida, velocidad media y máxima.

### 4. Clasificación de Golpes
El sistema detecta automáticamente los impactos y los clasifica (Volea vs Fondo) basándose en la posición del jugador y la aceleración de la bola.

### 5. Descargas
En la barra lateral o al final del reporte, encontrarás botones para descargar:
*   `padel_analytics_report.csv`: Datos crudos para Excel/Python.
*   `video_procesado.mp4`: El vídeo con las visualizaciones.

---

## Estructura del Proyecto

*   `app.py`: Punto de entrada de la aplicación web (Streamlit).
*   `trackers/`: Módulos de detección (Jugadores, Pelota, Keypoints).
*   `analytics/`: Lógica de negocio (Proyección 2D, Detección de Golpes).
*   `weights/`: Modelos pre-entrenados (YOLO, TrackNet).
*   `docker-compose.yml`: Configuración para despliegue en contenedores.

## Contribución

No se admiten contribuciones al ser un proyecto TF.

## Licencia

Este proyecto está bajo la licencia **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. 
Esto implica que puedes usar, modificar y compartir el código libremente para fines **académicos, educativos o privados, siempre sin ánimo de lucro**. Cualquier uso comercial o de monetización está estrictamente prohibido sin el consentimiento expreso del autor.

## Menciones

Este proyecto nace del proyecto anterior de João Miguel Freitas da Silva. Se han usado sus pesos que estaba disponibles y la estructura para mejorar y modificarlo para la realización de este TFG.
