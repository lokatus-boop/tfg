""" Streamlit dashboard to interact with the data collected """

import json
import numpy as np
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import supervision as sv
# import pims # Removed dependency
from utils.video import VideoReader

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
from visualizations.padel_court import padel_court_2d
from estimate_velocity import BallVelocityEstimator, ImpactType
from utils.video import save_video
from config import *

COLLECT_DATA = True


@st.fragment
def velocity_estimator(video_info: sv.VideoInfo):
        
    frame_index = st.slider(
        "Fotogramas", 
        0, 
        video_info.total_frames, 
        1, 
    )

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
    # Stop execution or disable button (but button is already rendered)
    # We will just prevent the processing block from running if weights are missing
    upload_video = False 

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
                # Default keypoints (approximate for a standard court view if available, or empty list to trigger detection)
                # For now, we'll try to let the automatic detection handle it or provide a dummy list if needed.
                # Based on main.py, it seems it might fall back to manual selection or automatic detection.
                # Let's initialize it as empty list or handle it downstream.
                SELECTED_KEYPOINTS = [] 

        if not SELECTED_KEYPOINTS:
                # Default to full screen if no keypoints found
                # Order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
                # Indices used: 0 (TL), 1 (TR), -1 (BR), -2 (BL) -> TL -> TR -> BR -> BL
                SELECTED_KEYPOINTS = [
                    [0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]
                ]

        # Only use fixed keypoints if we have enough points for homography (12, 18, or 22)
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
            # Fallback to automatic detection if we don't have a valid fixed set
            st.session_state["fixed_keypoints_detection"] = None

        keypoints_array = np.array(SELECTED_KEYPOINTS)
        # Polygon to filter person detections inside padel court
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
            # frame_resolution_wh=video_info.resolution_wh, # Removed in newer supervision versions
        )

        # Instantiate trackers
        st.session_state["players_tracker"] = PlayerTracker(
            PLAYERS_TRACKER_MODEL,
            polygon_zone,
            batch_size=PLAYERS_TRACKER_BATCH_SIZE,
            annotator=PLAYERS_TRACKER_ANNOTATOR,
            show_confidence=True,
            load_path=None, # PLAYERS_TRACKER_LOAD_PATH, # Disable cache loading to force re-inference
            save_path=PLAYERS_TRACKER_SAVE_PATH,
        )

        st.session_state["player_keypoints_tracker"] = PlayerKeypointsTracker(
            PLAYERS_KEYPOINTS_TRACKER_MODEL,
            train_image_size=PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE,
            batch_size=PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE,
            load_path=None, # PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH,
            save_path=PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH,
        )

        st.session_state["ball_tracker"] = BallTracker(
            BALL_TRACKER_MODEL,
            BALL_TRACKER_INPAINT_MODEL,
            batch_size=BALL_TRACKER_BATCH_SIZE,
            median_max_sample_num=BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
            median=None,
            load_path=None, # BALL_TRACKER_LOAD_PATH,
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
    if estimate_velocity:
        st.write("Selecciona un fotograma para calcular la velocidad de la bola:")
        velocity_estimator(st.session_state["runner"].video_info)
    
if st.session_state["df"] is not None:
    st.header("Datos Recolectados")
    
    # Download buttons
    csv = st.session_state["df"].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar Reporte (CSV)",
        data=csv,
        file_name='padel_analytics_report.csv',
        mime='text/csv',
    )
    
    if os.path.exists(OUTPUT_VIDEO_PATH):
        with open(OUTPUT_VIDEO_PATH, "rb") as file:
            btn = st.download_button(
                label="Descargar Video Procesado",
                data=file,
                file_name="video_procesado.mp4",
                mime="video/mp4"
            )

    st.write("Primeras 5 filas")
    st.dataframe(st.session_state["df"].head())
    st.markdown(f"- Número de filas: {len(st.session_state['df'])}")
    # st.write("- Columns: ")
    # st.write(st.session_state["df"].columns)

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
    padel_court = padel_court_2d()
    for player_id in (1, 2, 3, 4):
        fig.add_trace(
            go.Scatter(
                x=st.session_state["df"]["time"], 
                y=np.abs(
                    st.session_state["df"][
                        f"player{player_id}_V{velocity_type}4"
                    ].to_numpy()
                ),
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
        players_data["player_id"].append(player_id)
        players_data["total_distance_m"].append(
            st.session_state["df"][
                f"player{player_id}_distance"
            ].sum()
        )
        players_data["mean_velocity_km/h"].append(
            st.session_state["df"][
                f"player{player_id}_V{velocity_type}4"
            ].abs().mean() * 3.6,
        )
        players_data["maximum_velocity_km/h"].append(
            st.session_state["df"][
                f"player{player_id}_V{velocity_type}4"
            ].abs().max() * 3.6,
        )

    st.dataframe(pd.DataFrame(players_data).set_index("player_id"))

    st.subheader("Velocidad de los jugadores en función del tiempo")

    st.plotly_chart(fig)

    st.subheader("Analizar posición, velocidad y aceleración de los jugadores")
    
    col1, col2 = st.columns((1, 1))

    st.subheader("Seguimiento y Análisis de Jugador Individual")
    
    with col1:
        player_choice = st.radio("Seleccionar Jugador a Rastrear: ", options=[1, 2, 3, 4])
    
    with col2:
        # Handle potential NaN values if no velocity data is available
        min_value = st.session_state["df"][
            f"player{player_choice}_V{velocity_type}4"
        ].abs().min()
        max_value = st.session_state["df"][
            f"player{player_choice}_V{velocity_type}4"
        ].abs().max()

        if pd.isna(min_value) or pd.isna(max_value):
            st.warning(f"No hay datos de velocidad disponibles para el Jugador {player_choice}.")
            min_value = 0.0
            max_value = 1.0 # Default range to avoid slider error
            velocity_interval = (0.0, 1.0)
        else:
            velocity_interval = st.slider(
                "Intervalo de Velocidad",
                float(min_value), 
                float(max_value),
                (float(min_value), float(max_value)),
            )

    st.session_state["df"]["QUERY_VELOCITY"] = st.session_state["df"][
        f"player{player_choice}_V{velocity_type}4"
    ].abs()
    min_choice = velocity_interval[0]
    max_choice = velocity_interval[1]
    df_scatter = st.session_state["df"].query(
        "@min_choice <= QUERY_VELOCITY <= @max_choice"
    )
        
    padel_court.add_trace(
        go.Scatter(
            x=df_scatter[f"player{player_choice}_x"],
            y=df_scatter[f"player{player_choice}_y"] * -1,
            mode="markers",
            name=f"Jugador {player_choice}",
            text=df_scatter[
                f"player{player_choice}_V{velocity_type}4"
            ].abs() * 3.6,
            marker=dict(
                color=df_scatter[
                        f"player{player_choice}_V{velocity_type}4"
                ].abs() * 3.6,
                size=12,
                showscale=True,
                colorscale="jet",
                cmin=min_value * 3.6,
                cmax=max_value * 3.6,
            )
        )
    )

    st.plotly_chart(padel_court)

    padel_court = padel_court_2d()
    time_span = st.slider(
        "Intervalo de Tiempo",
        0.0, 
        st.session_state["df"]["time"].max(),
    )
    df_time = st.session_state["df"].query(
        "time <= @time_span"
    )
    padel_court.add_trace(
        go.Scatter(
            x=df_time[f"player{player_choice}_x"],
            y=df_time[f"player{player_choice}_y"] * -1,
            mode="markers",
            name=f"Jugador {player_choice}",
            text=df_time[
                f"player{player_choice}_V{velocity_type}4"
            ].abs() * 3.6,
            marker=dict(
                color=df_time[
                        f"player{player_choice}_V{velocity_type}4"
                ].abs() * 3.6,
                size=12,
                showscale=True,
                colorscale="jet",
                cmin=min_value * 3.6,
                cmax=max_value * 3.6,
            )
        )
    )
    st.plotly_chart(padel_court)

    st.subheader("Clasificación de Golpes")
    
    from analytics.shot_detector import ShotDetector
    shot_detector = ShotDetector()
    
    # Run detection
    if st.session_state["runner"]:
        fps = st.session_state["runner"].video_info.fps
    elif st.session_state["df"] is not None and len(st.session_state["df"]) > 1:
        # Infer FPS from time column
        time_diff = st.session_state["df"]["time"].iloc[1] - st.session_state["df"]["time"].iloc[0]
        fps = 1.0 / time_diff if time_diff > 0 else 30.0
    else:
        fps = 30.0
        
    shots_df = shot_detector.detect_shots(st.session_state["df"], fps)
    
    if not shots_df.empty:
        st.write(f"Total de golpes detectados: {len(shots_df)}")
        
        # Display shots dataframe
        st.dataframe(shots_df)
        
        # Stats per player
        st.write("Golpes por Jugador:")
        shots_per_player = shots_df.groupby("player_id")["shot_type"].value_counts().unstack().fillna(0)
        st.dataframe(shots_per_player)
        
        # Average speed per player
        st.write("Velocidad Media de la Bola (km/h) por Jugador:")
        avg_speed = shots_df.groupby("player_id")["ball_speed"].mean()
        st.dataframe(avg_speed)

        # Timeline
        fig_timeline = go.Figure()
        for player_id in shots_df["player_id"].unique():
            player_shots = shots_df[shots_df["player_id"] == player_id]
            fig_timeline.add_trace(go.Scatter(
                x=player_shots["frame"] / fps,
                y=player_shots["ball_speed"],
                mode='markers',
                name=f'Jugador {player_id}',
                text=player_shots["shot_type"]
            ))
        fig_timeline.update_layout(title="Línea de Tiempo de Golpes (Velocidad vs Tiempo)", xaxis_title="Tiempo (s)", yaxis_title="Velocidad (km/h)")
        st.plotly_chart(fig_timeline)

    else:
        st.warning("No se detectaron golpes. Verifica si el rastreo de la bola funciona correctamente.")

    def plotly_fig2array(fig):
        """
        Convert a plotly figure to numpy array
        """
        import io
        from PIL import Image
        print("HERE3")
        fig_bytes = fig.to_image(format="png")
        print("HERE4")
        buf = io.BytesIO(fig_bytes)
        img = Image.open(buf)
        return np.asarray(img)

    def court_frames(player_choice, velocity_type):

        padel_court = padel_court_2d()

        for t in st.session_state["df"]["time"]:

            print("HERE1")

            x_values = st.session_state["df"].query(
                "time <= @t"
            )[f"player{player_choice}_x"]

            y_values = st.session_state["df"].query(
                "time <= @t"
            )[f"player{player_choice}_y"] * -1

            v_values = st.session_state["df"].query(
                "time <= @t"
            )[f"player{player_choice}_V{velocity_type}4"].abs() * 3.6

            padel_court.add_trace(
                go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode="markers",
                            name=f"Jugador {player_choice}",
                            text=v_values,
                            marker=dict(
                                color=v_values,
                                size=12,
                                showscale=True,
                                colorscale="jet",
                                cmin=min_value * 3.6,
                                cmax=max_value * 3.6,
                            )
                        )
            )

            print("HERE2")

            yield plotly_fig2array(padel_court)

    # for frame in court_frames(player_choice, velocity_type):
    #     print(type(frame))
    #    break    

    # save_video(
    #     court_frames(player_choice, velocity_type), 
    #   "positions.mp4", 
    #     fps=st.session_state["runner"].video_info.fps,
    #    w=st.session_state["runner"].video_info.width,
    #    h=st.session_state["runner"].video_info.height,
    #)

        

        
        
        

        
        
        

 
        

