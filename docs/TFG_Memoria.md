# TRABAJO DE FIN DE GRADO: Padel Analytics

## CAPÍTULO 1 - INTRODUCCIÓN

### 1.1 MOTIVACIÓN
El pádel ha experimentado un crecimiento exponencial en la última década, convirtiéndose en uno de los deportes más practicados a nivel amateur y profesional. Sin embargo, a diferencia de deportes como el fútbol o el baloncesto, el acceso a herramientas de analítica avanzada está restringido al ámbito profesional debido al alto coste de los sistemas de tracking dedicados.

La motivación de este proyecto nace de la necesidad de democratizar el análisis táctico y técnico en el pádel. Utilizando técnicas modernas de Visión Artificial (Computer Vision) y Aprendizaje Profundo (Deep Learning), es posible extraer métricas valiosas (posicionamiento, velocidad de bola, tipos de golpeo) a partir de grabaciones de vídeo estándar, sin necesidad de sensores costosos o instalaciones complejas.

### 1.2 OBJETIVOS
El objetivo principal es desarrollar una aplicación software capaz de procesar vídeos de partidos de pádel para generar informes analíticos automáticos.

Los objetivos específicos son:
1.  **Detección y Seguimiento**: Implementar algoritmos para detectar y rastrear jugadores y la pelota en cada fotograma.
2.  **Proyección 2D**: Transformar las coordenadas de la imagen (píxeles) a coordenadas reales de la pista (metros) mediante homografía.
3.  **Analítica de Datos**: Calcular métricas cinemáticas (velocidad, distancia recorrida) y eventos de juego (golpes, rebotes).
4.  **Visualización**: Crear un cuadro de mando (Dashboard) interactivo para visualizar mapas de calor, trayectorias y estadísticas.
5.  **Despliegue**: Empaquetar la solución para su fácil distribución y uso (Docker).

### 1.3 MÉTODO DE TRABAJO
Se ha seguido una metodología iterativa e incremental, típica en el desarrollo de software moderno y proyectos de ciencia de datos:
1.  **Investigación**: Selección de modelos pre-entrenados (YOLO, TrackNet).
2.  **Prototipado**: Implementación de scripts básicos de detección.
3.  **Integración**: Unificación de módulos en una arquitectura coherente.
4.  **Refinamiento**: Mejora de la robustez (ej. manejo de oclusiones, corrección de perspectiva).
5.  **Validación**: Pruebas con vídeos reales y ajuste de parámetros.

### 1.4 MEDIOS HARDWARE Y SOFTWARE
**Hardware:**
*   Ordenador con GPU compatible con CUDA (NVIDIA) para acelerar la inferencia de modelos de Deep Learning.
*   Cámara de vídeo estándar (resolución mínima 720p/1080p).

**Software:**
*   **Lenguaje**: Python 3.12.
*   **Frameworks de IA**: PyTorch, Ultralytics YOLOv8.
*   **Visión Artificial**: OpenCV, Supervision.
*   **Interfaz de Usuario**: Streamlit.
*   **Contenerización**: Docker y Docker Compose.
*   **Control de Versiones**: Git y GitHub.

### 1.5 ESTRUCTURA DE LA MEMORIA
Este documento se estructura siguiendo el ciclo de vida del proyecto, desde la concepción teórica (Capítulos 1-3) hasta la implementación técnica (Capítulo 4) y las conclusiones finales (Capítulo 5).

---

## CAPÍTULO 2 - INDICACIONES

### 2.1 ESTILOS
(Este apartado se reserva para definir el formato del documento final: tipografía Times New Roman/Arial, tamaño 12, interlineado 1.5, márgenes estándar).

### 2.2 FIGURAS
Las figuras incluidas en este proyecto (diagramas de arquitectura, capturas de la detección, gráficos de velocidad) se numeran secuencialmente por capítulo (ej. Figura 4.1: Arquitectura del Sistema).

### 2.3 REFERENCIAS
Se utiliza el formato IEEE para las citas bibliográficas.

### 2.4 USO DE ACRÓNIMOS
*   **CV**: Computer Vision (Visión Artificial).
*   **CNN**: Convolutional Neural Network (Red Neuronal Convolucional).
*   **FPS**: Frames Per Second (Fotogramas por Segundo).
*   **YOLO**: You Only Look Once (Algoritmo de detección de objetos).
*   **GUI**: Graphical User Interface (Interfaz Gráfica de Usuario).

---

## CAPÍTULO 3 - ESTUDIOS PREVIOS

### 3.1 ESTADO DEL ARTE
La aplicación de la visión artificial al deporte no es nueva, pero ha evolucionado drásticamente:

1.  **Detección de Objetos (Jugadores)**: Históricamente se usaban técnicas de sustracción de fondo. Actualmente, el estado del arte lo dominan las redes neuronales convolucionales de una sola etapa como **YOLO (v5, v8)**, que ofrecen un balance óptimo entre velocidad y precisión.
2.  **Seguimiento de Objetos Pequeños (Pelota)**: La pelota de pádel es un objeto pequeño que se mueve a alta velocidad y sufre "blur" (desenfoque de movimiento). Modelos como **TrackNet** (basado en mapas de calor) han demostrado ser superiores a los detectores de cajas delimitadoras tradicionales para este fin.
3.  **Estimación de Pose (Keypoints)**: Librerías como OpenPose o los modelos de pose de YOLO permiten detectar articulaciones, lo cual es útil para analizar la biomecánica, aunque en este proyecto nos centramos en la posición global (centroide).
4.  **Homografía**: La transformación proyectiva es la técnica estándar para mapear una vista de cámara en perspectiva a un plano 2D cenital, requiriendo la identificación de al menos 4 puntos coplanares correspondientes.

---

## CAPÍTULO 4 - DESARROLLO DEL CASO DE ESTUDIO

### 4.1 ESPECIFICACIÓN DE REQUISITOS
**Requisitos Funcionales:**
*   **RF-01**: El sistema debe permitir la carga de archivos de vídeo en formatos estándar (MP4, AVI).
*   **RF-02**: El sistema debe detectar y diferenciar a los 4 jugadores en pista.
*   **RF-03**: El sistema debe detectar la posición de la pelota en cada fotograma.
*   **RF-04**: El sistema debe permitir la re-utilización de la calibración de pista para vídeos con la misma cámara.
*   **RF-05**: El sistema debe generar un informe descargable (CSV) con las trayectorias y velocidades.
*   **RF-06**: El sistema debe visualizar gráficas de velocidad y mapas de calor de posición.

**Requisitos No Funcionales:**
*   **RNF-01**: El sistema debe ser capaz de ejecutarse en entornos contenerizados (Docker).
*   **RNF-02**: La interfaz debe ser intuitiva para usuarios no técnicos (entrenadores/jugadores).

### 4.2 ARQUITECTURA DEL SISTEMA
La solución se ha diseñado modularmente:
1.  **Módulo `trackers`**: Encapsula la lógica de inferencia.
    *   `PlayerTracker`: Usa YOLOv8 para detectar personas.
    *   `BallTracker`: Usa un modelo especializado para la pelota.
    *   `KeypointsTracker`: Detecta las esquinas de la pista para la calibración.
2.  **Módulo `analytics`**:
    *   `ProjectedCourt`: Maneja la matriz de homografía para transformar píxeles a metros.
    *   `ShotDetector`: Analiza picos de aceleración y proximidad para clasificar golpes (Volea, Drive).
3.  **Interfaz `app.py`**: Construida con Streamlit, orquesta la ejecución, muestra el vídeo procesado y renderiza los gráficos con Plotly.

### 4.3 IMPLEMENTACIÓN DE ALGORITMOS CLAVE
**Calibración de Pista:**
Se detectan 14 puntos clave de la pista. Si la detección automática falla o es ruidosa, el sistema permite usar una calibración guardada previamente (`config.py` y checkbox en UI), asegurando robustez entre partidos grabados igual.

**Detección de Golpes:**
Se implementó un algoritmo heurístico en `ShotDetector` que:
1.  Suaviza la señal de aceleración de la pelota.
2.  Identifica picos locales de aceleración (cambios bruscos de dirección/velocidad).
3.  Correlaciona el pico con la proximidad de un jugador (< 2m).
4.  Clasifica el golpe según la posición en la pista (Cerca de la red = Volea, Fondo = Drive).

---

## CAPÍTULO 5 - CONCLUSIONES Y TRABAJOS FUTUROS

### 5.1 ANÁLISIS DE LA CONSECUCIÓN DE OBJETIVOS
Se ha logrado desarrollar una herramienta funcional que cumple con los objetivos planteados:
*   Procesa vídeos de forma autónoma.
*   Genera visualizaciones tácticas útiles (mapa 2D).
*   Exporta datos para análisis externo.
*   La inclusión de Docker facilita enormemente su despliegue en diferentes máquinas.

### 5.2 ESTIMACIÓN DE ESFUERZOS
El desarrollo ha requerido aproximadamente 300 horas de trabajo, distribuidas en:
*   Investigación y aprendizaje de librerías: 30%
*   Desarrollo de los trackers (Core): 40%
*   Desarrollo de la interfaz y visualización: 20%
*   Pruebas y corrección de errores (Debugging): 10%

### 5.3 FUTUROS TRABAJOS
*   **Análisis en Tiempo Real**: Optimizar los modelos (ej. usando TensorRT) para permitir análisis en directo durante el partido.
*   **Identificación de Jugadores**: Integrar reconocimiento facial o de dorsales para mantener la identidad del jugador tras cruces u oclusiones largas.
*   **Clasificación Avanzada de Golpes**: Entrenar un modelo LSTM o Transformer sobre las secuencias de movimiento para distinguir bandejas, víboras y remates con mayor precisión.
*   **App Móvil**: Migrar la interfaz a una aplicación nativa para grabar y analizar desde el mismo dispositivo.

---

## CAPÍTULO 6 - BIBLIOGRAFÍA Y LUGARES DE INTERNET

### 6.1 BIBLIOGRAFÍA
[1] J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," CVPR 2016.
[2] A. Vaswani et al., "Attention Is All You Need," NIPS 2017.

### 6.2 LUGARES DE INTERNET
*   **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
*   **Streamlit Documentation**: https://docs.streamlit.io
*   **OpenCV**: https://opencv.org

---

## CAPÍTULO 7 - APÉNDICES

### 7.1 APÉNDICE I: TABLA DE SIGLAS
(Ver apartado 2.4)

### 7.2 APÉNDICE II: MANUAL DE USUARIO
1.  Instalar Docker Desktop.
2.  Clonar el repositorio.
3.  Ejecutar `docker-compose up --build`.
4.  Acceder a `localhost:8501`.
5.  Subir vídeo y esperar resultados.
