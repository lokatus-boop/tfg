""" Streamlit dashboard to interact with the data collected """

import json
import numpy as np
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import supervision as sv
from utils.video import VideoReader, save_video

from trackers import (
    Keypoint, 
    Keypoints, 
    PlayerTracker, 
    PlayerKeypointsTracker,
    BallTracker, 
    KeypointsTracker,
    TrackingRunner
)
from analytics import DataAnalytics
from analytics.shot_detector import ShotDetector
from visualizations.padel_court import padel_court_2d
from estimate_velocity import BallVelocityEstimator, ImpactType
from config import *

# --- NUEVO IMPORT PARA PDF ---
from report_generator import create_full_report
# -----------------------------

COLLECT_DATA = True

@st.fragment
def velocity_estimator(video_info: sv.VideoInfo):
        
    frame_index = st.slider(
        "Fotogramas", 
        0, 
        video_info.total_frames, 
        1, 
    )

    if st.session_state["video"] is not None:
        image = np.array(st.session_state["video"][frame_index])
        st.image(image)

    with st.form("choose-frames"):
        frame_index_t0 = st.number_input(
            "Primer fotograma: ", 
            min_value=0,
            max_value=video_info.total_frames,
        )
        frame_index_t1 = st.number_input(
            "Segundo fotograma: ", 
            min_value=1,
            max_value=video_info.total_frames,
        )
        impact_type_ch = st.radio(
            "Tipo de impacto: ",
            options=["Suelo", "Jugador"],
        )
        get_Vz = st.radio(
            "Considerar diferencia en altitud de la bola: ",
            options=[False, True]
        )

        estimate = st.form_submit_button("Calcular velocidad")

    if estimate:

        assert frame_index_t0 < frame_index_t1

        if st.session_state["players_tracker"] is None:
            st.error("Faltan datos.")
        else:
            estimator = BallVelocityEstimator(
                source_video_fps=video_info.fps,
                players_detections=st.session_state["players_tracker"].results.predictions,
                ball_detections=st.session_state["ball_tracker"].results.predictions,
                keypoints_detections=st.session_state["keypoints_tracker"].results.predictions,
            )

            if impact_type_ch == "Suelo":
                impact_type = ImpactType.FLOOR
            elif impact_type_ch == "Jugador":
                impact_type = ImpactType.RACKET

            ball_velocity_data, ball_velocity = estimator.estimate_velocity(
                frame_index_t0, frame_index_t1, impact_type, get_Vz=get_Vz,
            )
            st.write(ball_velocity)
            st.write("Velocidad: ", ball_velocity.norm)
            
            if st.session_state["video"] is not None:
                st.image(ball_velocity_data.draw_velocity(st.session_state["video"]))
            
            padel_court = padel_court_2d()
            padel_court.add_trace(
                go.Scatter(
                    x=[
                        ball_velocity_data.position_t0_proj[0],
                        ball_velocity_data.position_t1_proj[0],
                    ],
                    y=[
                        ball_velocity_data.position_t0_proj[1]*-1,
                        ball_velocity_data.position_t1_proj[1]*-1,
                    ],
                    marker= dict(
                        size=10,
                        symbol= "arrow-bar-up", 
                        angleref="previous",
                    ),
                )                    
            )
            st.plotly_chart(padel_court)


# --- INITIALIZATION ---
if "video" not in st.session_state:
    st.session_state["video"] = None

if "df" not in st.session_state:
    st.session_state["df"] = None

if "fixed_keypoints_detection" not in st.session_state:
    st.session_state["fixed_keypoints_detection"] = None

if "players_keypoints_tracker" not in st.session_state:
    st.session_state["players_keypoints_tracker"] = None

if "players_tracker" not in st.session_state:
    st.session_state["players_tracker"] = None

if "ball_tracker" not in st.session_state:
    st.session_state["ball_tracker"] = None

if "keypoints_tracker" not in st.session_state:
    st.session_state["keypoints_tracker"] = None

if "runner" not in st.session_state:
    st.session_state["runner"] = None

# --- UI START ---
st.title("Analítica de Pádel")

uploaded_csv = st.file_uploader("Cargar reporte CSV existente", type=["csv"])

if uploaded_csv is not None:
    st.session_state["df"] = pd.read_csv(uploaded_csv)
    st.success("Reporte cargado correctamente.")

with st.form("run-video"):
    upload_video_path = st.text_input(
        "Subir video: ",
        INPUT_VIDEO_PATH,
    )
    upload_video = st.form_submit_button("Subir")
    
    reuse_keypoints = st.checkbox(
        "Reutilizar detección de pista anterior (misma cámara)",
        value=False,
        help="Marca esto solo si la cámara no se ha movido nada desde el último video procesado."
    )

# Check for weights
required_weights = [
    PLAYERS_TRACKER_MODEL,
    PLAYERS_KEYPOINTS_TRACKER_MODEL,
    BALL_TRACKER_MODEL,
    BALL_TRACKER_INPAINT_MODEL,
    KEYPOINTS_TRACKER_MODEL,
]
missing_weights = [w for w in required_weights if not os.path.exists(w)]

if missing_weights:
    st.error("¡Faltan archivos de pesos! Por favor descárgalos y colócalos en el directorio `weights`.")
    st.write("Archivos faltantes:")
    for w in missing_weights:
        st.code(w)
    st.warning("La aplicación no puede ejecutar la inferencia sin estos pesos.")
    upload_video = False 

# --- MAIN LOGIC ---
if (upload_video or st.session_state["video"] is not None) and uploaded_csv is None:

    if upload_video:
        st.session_state["df"] = None
        os.system(f"ffmpeg -y -i {upload_video_path} -vcodec libx264 tmp.mp4")
    
    if st.session_state["df"] is None:

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(message, progress):
            status_text.text(message)
            progress_bar.progress(progress)

            
        video_info = sv.VideoInfo.from_video_path(video_path="tmp.mp4")  
        fps, w, h, total_frames = (
            video_info.fps, 
            video_info.width,
            video_info.height,
            video_info.total_frames,
        ) 
        
        if FIXED_COURT_KEYPOINTS_LOAD_PATH is not None:
            if os.path.exists(FIXED_COURT_KEYPOINTS_LOAD_PATH):
                with open(FIXED_COURT_KEYPOINTS_LOAD_PATH, "r") as f:
                    SELECTED_KEYPOINTS = json.load(f)
            else:
                st.warning(f"Archivo de puntos clave no encontrado en {FIXED_COURT_KEYPOINTS_LOAD_PATH}. Usando puntos por defecto.")
                SELECTED_KEYPOINTS = [] 

        if not SELECTED_KEYPOINTS:
                SELECTED_KEYPOINTS = [
                    [0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]
                ]

        # Only use fixed keypoints if we have enough points for homography
        if len(SELECTED_KEYPOINTS) in (12, 18, 22):
            st.session_state["fixed_keypoints_detection"] = Keypoints(
                [
                    Keypoint(
                        id=i,
                        xy=tuple(float(x) for x in v)
                    )
                    for i, v in enumerate(SELECTED_KEYPOINTS)
                ]
            )
        else:
            st.session_state["fixed_keypoints_detection"] = None

        keypoints_array = np.array(SELECTED_KEYPOINTS)
        
        polygon_zone = sv.PolygonZone(
            polygon=np.concatenate(
                (
                    np.expand_dims(keypoints_array[0], axis=0), 
                    np.expand_dims(keypoints_array[1], axis=0), 
                    np.expand_dims(keypoints_array[-1], axis=0), 
                    np.expand_dims(keypoints_array[-2], axis=0),
                ),
                axis=0
            ),
        )

        st.session_state["players_tracker"] = PlayerTracker(
            PLAYERS_TRACKER_MODEL,
            polygon_zone,
            batch_size=PLAYERS_TRACKER_BATCH_SIZE,
            annotator=PLAYERS_TRACKER_ANNOTATOR,
            show_confidence=True,
            load_path=None, 
            save_path=PLAYERS_TRACKER_SAVE_PATH,
        )

        st.session_state["player_keypoints_tracker"] = PlayerKeypointsTracker(
            PLAYERS_KEYPOINTS_TRACKER_MODEL,
            train_image_size=PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE,
            batch_size=PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE,
            load_path=None, 
            save_path=PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH,
        )

        st.session_state["ball_tracker"] = BallTracker(
            BALL_TRACKER_MODEL,
            BALL_TRACKER_INPAINT_MODEL,
            batch_size=BALL_TRACKER_BATCH_SIZE,
            median_max_sample_num=BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
            median=None,
            load_path=None, 
            save_path=BALL_TRACKER_SAVE_PATH,
        )

        st.session_state["keypoints_tracker"] = KeypointsTracker(
            model_path=KEYPOINTS_TRACKER_MODEL,
            batch_size=KEYPOINTS_TRACKER_BATCH_SIZE,
            model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
            fixed_keypoints_detection=st.session_state["fixed_keypoints_detection"],
            load_path=KEYPOINTS_TRACKER_LOAD_PATH if reuse_keypoints else None,
            save_path=KEYPOINTS_TRACKER_SAVE_PATH,
        )

        runner = TrackingRunner(
            trackers=[
                st.session_state["players_tracker"], 
                st.session_state["player_keypoints_tracker"], 
                st.session_state["ball_tracker"],
                st.session_state["keypoints_tracker"],     
            ],
            video_path="tmp.mp4",
            inference_path=OUTPUT_VIDEO_PATH,
            start=0,
            end=MAX_FRAMES,
            collect_data=COLLECT_DATA,
        )

        runner.run(status_callback=update_progress)

        st.session_state["runner"] = runner

        st.session_state["df"]  = runner.data_analytics.into_dataframe(
            runner.video_info.fps,
        )

        st.success("Hecho.")
    
    st.session_state["video"] = VideoReader("tmp.mp4")
    st.subheader("Video Subido")
    st.video("tmp.mp4")
    
    estimate_velocity = st.checkbox("Calcular Velocidad de la Bola")
    if estimate_velocity and st.session_state["runner"] is not None:
        st.write("Selecciona un fotograma para calcular la velocidad de la bola:")
        velocity_estimator(st.session_state["runner"].video_info)
    
if st.session_state["df"] is not None:
    st.header("Datos Recolectados")
    
    # --- DOWNLOAD BUTTONS ---
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        csv = st.session_state["df"].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Descargar Reporte (CSV)",
            data=csv,
            file_name='padel_analytics_report.csv',
            mime='text/csv',
        )
    
    with col_dl2:
        if os.path.exists(OUTPUT_VIDEO_PATH):
            with open(OUTPUT_VIDEO_PATH, "rb") as file:
                st.download_button(
                    label="🎥 Descargar Video Procesado",
                    data=file,
                    file_name="video_procesado.mp4",
                    mime="video/mp4"
                )

    # --- PDF GENERATOR SECTION ---
    st.markdown("---")
    st.subheader("📊 Informe Profesional")
    
    col_pdf1, col_pdf2 = st.columns([1, 2])
    
    with col_pdf1:
        # Botón para generar el PDF
        if st.button("Generar PDF de Rendimiento"):
            with st.spinner("Generando gráficos y maquetando PDF..."):
                try:
                    # 1. Guardamos el dataframe actual a un CSV temporal
                    temp_csv = "temp_data_for_report.csv"
                    st.session_state["df"].to_csv(temp_csv, index=False)
                    
                    # 2. Llamamos a la función generadora del otro archivo
                    output_pdf = "Informe_Partido.pdf"
                    create_full_report(temp_csv, output_pdf)
                    
                    st.session_state['pdf_ready'] = True
                    st.success("¡Informe generado con éxito!")
                    
                except Exception as e:
                    st.error(f"Error generando el reporte: {e}")

    with col_pdf2:
        # Botón para descargar el PDF (aparece solo si ya se generó)
        if os.path.exists("Informe_Partido.pdf") and st.session_state.get('pdf_ready'):
            with open("Informe_Partido.pdf", "rb") as pdf_file:
                st.download_button(
                    label="📥 Descargar PDF Final",
                    data=pdf_file,
                    file_name="Padel_Analyst_Report.pdf",
                    mime="application/pdf",
                    key="pdf_download_btn"
                )
    st.markdown("---")
    # -----------------------------

    st.write("Primeras 5 filas")
    st.dataframe(st.session_state["df"].head())
    st.markdown(f"- Número de filas: {len(st.session_state['df'])}")

    # --- PLOTS & ANALYTICS ---
    velocity_type_choice = st.radio(
        "Tipo", 
        ["Horizontal", "Vertical", "Absoluta"],
    )
    velocity_type_mapper = {
        "Horizontal": "x",
        "Vertical": "y",
        "Absoluta": "norm",
    }
    velocity_type = velocity_type_mapper[velocity_type_choice]
    fig = go.Figure()
    
    for player_id in (1, 2, 3, 4):
        # Check if column exists to avoid errors with partial data
        col_name = f"player{player_id}_V{velocity_type}4"
        if col_name in st.session_state["df"].columns:
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["df"]["time"], 
                    y=np.abs(st.session_state["df"][col_name].to_numpy()),
                    mode='lines',
                    name=f'Jugador {player_id}',
                ),
            )
    
    fig.update_layout(
        title="Velocidad de los jugadores en función del tiempo",
        xaxis_title="Tiempo (s)",
        yaxis_title="Velocidad (m/s)"
    )

    players_data = {
        "player_id": [],
        "total_distance_m": [],
        "mean_velocity_km/h": [],
        "maximum_velocity_km/h": [],
    }
    for player_id in (1, 2, 3, 4):
        dist_col = f"player{player_id}_distance"
        vel_col = f"player{player_id}_V{velocity_type}4"
        
        if dist_col in st.session_state["df"].columns and vel_col in st.session_state["df"].columns:
            players_data["player_id"].append(player_id)
            players_data["total_distance_m"].append(
                st.session_state["df"][dist_col].sum()
            )
            players_data["mean_velocity_km/h"].append(
                st.session_state["df"][vel_col].abs().mean() * 3.6,
            )
            players_data["maximum_velocity_km/h"].append(
                st.session_state["df"][vel_col].abs().max() * 3.6,
            )

    st.dataframe(pd.DataFrame(players_data).set_index("player_id"))

    st.subheader("Velocidad de los jugadores en función del tiempo")
    st.plotly_chart(fig)

    st.markdown("---")
    st.subheader("Seguimiento y Análisis de Jugador Individual")
    
    col1, col2 = st.columns((1, 1))
    
    with col1:
        player_choice = st.radio("Seleccionar Jugador a Rastrear: ", options=[1, 2, 3, 4])
    
    with col2:
        col_name = f"player{player_choice}_V{velocity_type}4"
        if col_name in st.session_state["df"].columns:
            min_value = st.session_state["df"][col_name].abs().min()
            max_value = st.session_state["df"][col_name].abs().max()
            
            if pd.isna(min_value) or pd.isna(max_value):
                min_value, max_value = 0.0, 1.0

            velocity_interval = st.slider(
                "Intervalo de Velocidad",
                float(min_value), 
                float(max_value),
                (float(min_value), float(max_value)),
            )
        else:
            velocity_interval = (0.0, 1.0)
            st.warning("Datos de velocidad no disponibles.")

    if col_name in st.session_state["df"].columns:
        st.session_state["df"]["QUERY_VELOCITY"] = st.session_state["df"][col_name].abs()
        min_choice = velocity_interval[0]
        max_choice = velocity_interval[1]
        
        df_scatter = st.session_state["df"].query(
            "@min_choice <= QUERY_VELOCITY <= @max_choice"
        )
            
        padel_court_viz = padel_court_2d()
        padel_court_viz.add_trace(
            go.Scatter(
                x=df_scatter[f"player{player_choice}_x"],
                y=df_scatter[f"player{player_choice}_y"] * -1,
                mode="markers",
                name=f"Jugador {player_choice}",
                text=df_scatter[col_name].abs() * 3.6,
                marker=dict(
                    color=df_scatter[col_name].abs() * 3.6,
                    size=12,
                    showscale=True,
                    colorscale="jet",
                    cmin=min_value * 3.6,
                    cmax=max_value * 3.6,
                    colorbar=dict(title="Velocidad (km/h)")
                )
            )
        )
        st.plotly_chart(padel_court_viz)

        # Time slider visualization
        padel_court_time = padel_court_2d()
        time_span = st.slider(
            "Intervalo de Tiempo (trayectoria acumulada)",
            0.0, 
            float(st.session_state["df"]["time"].max()),
        )
        df_time = st.session_state["df"].query("time <= @time_span")
        
        padel_court_time.add_trace(
            go.Scatter(
                x=df_time[f"player{player_choice}_x"],
                y=df_time[f"player{player_choice}_y"] * -1,
                mode="markers",
                name=f"Jugador {player_choice}",
                marker=dict(
                    color=df_time[col_name].abs() * 3.6,
                    size=8,
                    showscale=False,
                    colorscale="jet",
                    cmin=min_value * 3.6,
                    cmax=max_value * 3.6,
                )
            )
        )
        st.plotly_chart(padel_court_time)

    # --- SHOT CLASSIFICATION ---
    st.subheader("Clasificación de Golpes")
    
    shot_detector = ShotDetector()
    
    if st.session_state["runner"]:
        current_fps = st.session_state["runner"].video_info.fps
    elif st.session_state["df"] is not None and len(st.session_state["df"]) > 1:
        time_diff = st.session_state["df"]["time"].iloc[1] - st.session_state["df"]["time"].iloc[0]
        current_fps = 1.0 / time_diff if time_diff > 0 else 30.0
    else:
        current_fps = 30.0
        
    shots_df = shot_detector.detect_shots(st.session_state["df"], current_fps)
    
    if not shots_df.empty:
        st.write(f"Total de golpes detectados: {len(shots_df)}")
        st.dataframe(shots_df)
        
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.write("**Golpes por Jugador:**")
            shots_per_player = shots_df.groupby("player_id")["shot_type"].value_counts().unstack().fillna(0)
            st.dataframe(shots_per_player)
        
        with col_stats2:
            st.write("**Velocidad Media (km/h) por Jugador:**")
            avg_speed = shots_df.groupby("player_id")["ball_speed"].mean()
            st.dataframe(avg_speed)

        # Timeline Plot
        fig_timeline = go.Figure()
        for player_id in shots_df["player_id"].unique():
            player_shots = shots_df[shots_df["player_id"] == player_id]
            fig_timeline.add_trace(go.Scatter(
                x=player_shots["frame"] / current_fps,
                y=player_shots["ball_speed"],
                mode='markers',
                name=f'Jugador {player_id}',
                text=player_shots["shot_type"],
                marker=dict(size=10)
            ))
        fig_timeline.update_layout(
            title="Línea de Tiempo de Golpes (Velocidad vs Tiempo)", 
            xaxis_title="Tiempo (s)", 
            yaxis_title="Velocidad Bola (km/h)"
        )
        st.plotly_chart(fig_timeline)

    else:
        st.info("No se han detectado golpes claros en este segmento.")
