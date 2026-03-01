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


# --- IMPORT PARA PDF ---

from report_generator import create_full_report


COLLECT_DATA = True


@st.fragment

def velocity_estimator(video_info: sv.VideoInfo):

    frame_index = st.slider("Fotogramas", 0, video_info.total_frames, 1)

    if st.session_state["video"] is not None:

        image = np.array(st.session_state["video"][frame_index])

        st.image(image)


    with st.form("choose-frames"):

        frame_index_t0 = st.number_input("Primer fotograma: ", min_value=0, max_value=video_info.total_frames)

        frame_index_t1 = st.number_input("Segundo fotograma: ", min_value=1, max_value=video_info.total_frames)

        impact_type_ch = st.radio("Tipo de impacto: ", options=["Suelo", "Jugador"])

        get_Vz = st.radio("Considerar diferencia en altitud de la bola: ", options=[False, True])

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

            impact_type = ImpactType.FLOOR if impact_type_ch == "Suelo" else ImpactType.RACKET

            ball_velocity_data, ball_velocity = estimator.estimate_velocity(

                frame_index_t0, frame_index_t1, impact_type, get_Vz=get_Vz,

            )

            st.write(ball_velocity)

            st.write("Velocidad: ", ball_velocity.norm)

            

            if st.session_state["video"] is not None:

                st.image(ball_velocity_data.draw_velocity(st.session_state["video"]))

            

            padel_court = padel_court_2d()

            padel_court.add_trace(go.Scatter(

                x=[ball_velocity_data.position_t0_proj[0], ball_velocity_data.position_t1_proj[0]],

                y=[ball_velocity_data.position_t0_proj[1]*-1, ball_velocity_data.position_t1_proj[1]*-1],

                marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"),

            ))                    

            st.plotly_chart(padel_court)


# --- INICIALIZACIÓN DE ESTADO ---

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

st.title("Analítica de Pádel TFG")


uploaded_csv = st.file_uploader("Cargar reporte CSV existente", type=["csv"])

if uploaded_csv is not None:

    st.session_state["df"] = pd.read_csv(uploaded_csv)

    st.success("Reporte cargado correctamente.")


with st.form("run-video"):

    upload_video_path = st.text_input("Subir video: ", INPUT_VIDEO_PATH)

    upload_video = st.form_submit_button("Subir y Procesar")

    reuse_keypoints = st.checkbox(

        "Reutilizar detección de pista anterior", 

        value=False,

        help="Útil si la cámara no se mueve, evita tener que detectar la pista de nuevo."

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

    st.error("¡Faltan archivos de pesos!")

    st.code(missing_weights)

    upload_video = False 


# --- MAIN LOGIC ---

if (upload_video or st.session_state["video"] is not None) and uploaded_csv is None:


    if upload_video:

        st.session_state["df"] = None

        os.system(f"ffmpeg -y -i {upload_video_path} -r 30 -vsync cfr -vcodec libx264 -acodec copy tmp.mp4")

    

    if st.session_state["df"] is None:


        progress_bar = st.progress(0)

        status_text = st.empty()


        def update_progress(message, progress):

            status_text.text(message)

            progress_bar.progress(progress)


        video_info = sv.VideoInfo.from_video_path(video_path="tmp.mp4")  

        w, h = video_info.width, video_info.height

        

        # Carga de Keypoints fijos

        SELECTED_KEYPOINTS = []

        if FIXED_COURT_KEYPOINTS_LOAD_PATH is not None:

            if os.path.exists(FIXED_COURT_KEYPOINTS_LOAD_PATH):

                with open(FIXED_COURT_KEYPOINTS_LOAD_PATH, "r") as f:

                    SELECTED_KEYPOINTS = json.load(f)


        if len(SELECTED_KEYPOINTS) in (12, 18, 22):

            st.session_state["fixed_keypoints_detection"] = Keypoints(

                [Keypoint(id=i, xy=tuple(float(x) for x in v)) for i, v in enumerate(SELECTED_KEYPOINTS)]

            )

        else:

            st.session_state["fixed_keypoints_detection"] = None


        # Configuración de Zona Poligonal

        if SELECTED_KEYPOINTS:

             keypoints_array = np.array(SELECTED_KEYPOINTS)

             polygon = np.concatenate((

                np.expand_dims(keypoints_array[0], axis=0), 

                np.expand_dims(keypoints_array[1], axis=0), 

                np.expand_dims(keypoints_array[-1], axis=0), 

                np.expand_dims(keypoints_array[-2], axis=0),

             ), axis=0)

             polygon_zone = sv.PolygonZone(polygon=polygon)

        else:

             polygon_zone = sv.PolygonZone(polygon=np.array([[0,0], [w,0], [w,h], [0,h]]))


        # --- INICIALIZAR TRACKERS ---

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

        print("\n" + "="*50)
        print("✅ ANÁLISIS DEL VÍDEO COMPLETADO CON ÉXITO")
        print("="*50 + "\n")

        st.session_state["runner"] = runner


        st.session_state["df"]  = runner.data_analytics.into_dataframe(

            runner.video_info.fps,

        )


        st.success("Hecho.")

    

    st.session_state["video"] = VideoReader("tmp.mp4")

    st.subheader("Video Subido")

    st.video("tmp.mp4")

    

    if st.checkbox("Herramienta de Velocidad"):

        velocity_estimator(st.session_state["runner"].video_info)

    

if st.session_state["df"] is not None:

    st.header("Datos Recolectados")

    

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:

        csv = st.session_state["df"].to_csv(index=False).encode('utf-8')

        st.download_button("📄 Descargar Reporte (CSV)", data=csv, file_name='padel_analytics_report.csv', mime='text/csv')

    

    with col_dl2:

        if os.path.exists(OUTPUT_VIDEO_PATH):

            with open(OUTPUT_VIDEO_PATH, "rb") as file:

                st.download_button("🎥 Descargar Video Procesado", data=file, file_name="video_procesado.mp4", mime="video/mp4")


    # --- PDF GENERATOR SECTION ---

    st.markdown("---")

    st.subheader("📊 Informe Profesional")

    

    col_pdf1, col_pdf2 = st.columns([1, 2])

    

    with col_pdf1:

        if st.button("Generar PDF de Rendimiento"):

            with st.spinner("Generando gráficos y maquetando PDF..."):

                try:

                    # 1. Preparar CSV principal

                    temp_csv = "temp_data_for_report.csv"

                    st.session_state["df"].to_csv(temp_csv, index=False)


                    # 2. Detectar Golpes y preparar datos extra

                    shot_detector_pdf = ShotDetector()

                    

                    # Obtener FPS de forma segura

                    if st.session_state["runner"]:

                        fps_pdf = st.session_state["runner"].video_info.fps

                    else:

                        fps_pdf = 30.0

                        

                    shots_df_pdf = shot_detector_pdf.detect_shots(st.session_state["df"], fps_pdf)

                    

                    temp_shots_csv = "temp_shots_data.csv"

                    shots_df_pdf.to_csv(temp_shots_csv, index=False)


                    # 3. Generar y guardar Gráfico de Golpes como Imagen (Plotly -> PNG)

                    fig_shots_pdf = go.Figure()

                    for pid in shots_df_pdf["player_id"].unique():

                        pshots = shots_df_pdf[shots_df_pdf["player_id"] == pid]

                        fig_shots_pdf.add_trace(go.Scatter(

                            x=pshots["frame"]/fps_pdf, y=pshots["ball_speed"],

                            mode='markers', name=f'J{pid}', 

                            marker=dict(size=10)

                        ))

                    fig_shots_pdf.update_layout(

                        title="Cronología de Golpes", 

                        xaxis_title="Tiempo (s)", 

                        yaxis_title="Velocidad (km/h)",

                        width=1000, height=500 

                    )

                    

                    temp_shots_img = "temp_shots_chart.png"

                    # Esto requiere 'pip install kaleido'

                    fig_shots_pdf.write_image(temp_shots_img) 


                    # 4. Llamar al generador de reporte actualizado

                    create_full_report(

                        csv_path=temp_csv, 

                        output_pdf="Informe_Partido.pdf",

                        shots_csv_path=temp_shots_csv,

                        shots_chart_path=temp_shots_img

                    )

                    

                    st.session_state['pdf_ready'] = True

                    st.success("¡Informe generado con éxito!")

                except Exception as e:

                    st.error(f"Error generando el reporte: {e}")


    with col_pdf2:

        if os.path.exists("Informe_Partido.pdf") and st.session_state.get('pdf_ready'):

            with open("Informe_Partido.pdf", "rb") as pdf_file:

                st.download_button("📥 Descargar PDF Final", data=pdf_file, file_name="Padel_Analyst_Report.pdf", mime="application/pdf")

    st.markdown("---")


    # --- ESTADÍSTICAS Y GRÁFICOS (VISUALIZACIÓN EN WEB) ---

    st.write("Primeras 5 filas")

    st.dataframe(st.session_state["df"].head())

    

    # 1. Gráficos de Velocidad

    velocity_type_choice = st.radio("Tipo de Velocidad", ["Horizontal", "Vertical", "Absoluta"])

    velocity_type_mapper = {"Horizontal": "x", "Vertical": "y", "Absoluta": "norm"}

    velocity_type = velocity_type_mapper[velocity_type_choice]

    

    fig = go.Figure()

    for player_id in (1, 2, 3, 4):

        col_name = f"player{player_id}_V{velocity_type}4"

        if col_name in st.session_state["df"].columns:

            fig.add_trace(go.Scatter(

                x=st.session_state["df"]["time"], 

                y=np.abs(st.session_state["df"][col_name].to_numpy()),

                mode='lines', name=f'Jugador {player_id}'

            ))

    st.plotly_chart(fig)


    # 2. Clasificación de Golpes (Web)

    st.subheader("Clasificación de Golpes")

    shot_detector = ShotDetector()

    fps_val = 30.0

    shots_df = shot_detector.detect_shots(st.session_state["df"], fps_val)

    

    if not shots_df.empty:

        st.write(f"Total golpes: {len(shots_df)}")

        st.dataframe(shots_df)

        

        # Timeline

        fig_shots = go.Figure()

        for pid in shots_df["player_id"].unique():

            pshots = shots_df[shots_df["player_id"] == pid]

            fig_shots.add_trace(go.Scatter(

                x=pshots["frame"]/fps_val, y=pshots["ball_speed"],

                mode='markers+text', name=f'J{pid}', text=pshots["shot_type"],

                textposition="top center", marker=dict(size=10)

            ))

        fig_shots.update_layout(title="Cronología de Golpes", xaxis_title="Tiempo (s)", yaxis_title="Velocidad (km/h)")

        st.plotly_chart(fig_shots) 
