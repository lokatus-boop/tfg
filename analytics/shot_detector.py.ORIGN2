from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class Shot:
    frame: int
    player_id: int
    shot_type: str
    ball_speed: float

class ShotDetector:
    def __init__(self):
        pass

    def detect_shots(self, df: pd.DataFrame, fps: float) -> pd.DataFrame:
        """
        Detecta golpes basándose en picos de aceleración y clasifica el tipo
        usando posición en pista y velocidad de salida.
        """
        shots = []
        
        # Verificar datos mínimos
        if "ball_Vnorm1" not in df.columns:
            return pd.DataFrame()

        # --- CONSTANTES Y UMBRALES ---
        ACCEL_THRESHOLD = 40.0      # m/s^2 (Bajado un poco para detectar toques suaves)
        PROXIMITY_THRESHOLD = 2.5   # metros (Margen ampliado para errores de detección)
        
        # Umbrales de Zona (Asumiendo 0=Red, 10=Cristal de fondo)
        NET_ZONE_LIMIT = 5.0        # Metros desde la red. Menos de 5m es "Zona de Red"
        
        # Umbrales de Velocidad (km/h) para clasificación
        SPEED_SMASH = 85.0          # Más de 85 km/h es seguramente un remate
        SPEED_LOB = 35.0            # Menos de 35 km/h desde el fondo suele ser globo
        
        # 1. Suavizado de datos para reducir ruido
        # Usamos aceleración si existe, o calculamos derivada de velocidad
        if "ball_Anorm1" in df.columns:
            accel_series = df["ball_Anorm1"]
        else:
            # Fallback simple si no hay columna de aceleración
            accel_series = df["ball_Vnorm1"].diff().abs().fillna(0)

        df["ball_accel_smooth"] = accel_series.rolling(window=3, center=True).mean().fillna(0)
        
        # 2. Detección de Picos (Impactos)
        potential_impacts = []
        for i in range(2, len(df) - 2):
            curr = df.iloc[i]["ball_accel_smooth"]
            prev = df.iloc[i-1]["ball_accel_smooth"]
            nxt  = df.iloc[i+1]["ball_accel_smooth"]
            
            # Es un pico local y supera el umbral
            if curr > ACCEL_THRESHOLD and curr > prev and curr > nxt:
                potential_impacts.append(i)
        
        # 3. Filtrado y Clasificación
        last_shot_frame = -100
        
        for idx in potential_impacts:
            row = df.iloc[idx]
            frame = int(row["frame"])
            
            # Debounce (evitar dobles detecciones del mismo golpe)
            if frame - last_shot_frame < fps * 0.6: 
                continue
                
            ball_x = row["ball_x"]
            ball_y = row["ball_y"]
            
            if pd.isna(ball_x) or pd.isna(ball_y):
                continue

            # Buscar jugador más cercano
            closest_player_id = None
            min_dist = float("inf")
            
            for player_id in (1, 2, 3, 4):
                p_x_col = f"player{player_id}_x"
                p_y_col = f"player{player_id}_y"
                
                if p_x_col not in row or p_y_col not in row:
                    continue
                    
                p_x = row[p_x_col]
                p_y = row[p_y_col]
                
                if pd.isna(p_x) or pd.isna(p_y):
                    continue
                    
                dist = np.sqrt((ball_x - p_x)**2 + (ball_y - p_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_player_id = player_id
            
            # Si hay un jugador cerca, clasificamos el golpe
            if closest_player_id and min_dist < PROXIMITY_THRESHOLD:
                
                player_y = row[f"player{closest_player_id}_y"]
                ball_speed_kmh = row["ball_Vnorm1"] * 3.6
                
                # --- LÓGICA DE CLASIFICACIÓN AVANZADA ---
                
                dist_to_net = abs(player_y) # Asumiendo que red está en Y=0
                
                # CASO A: JUGADOR EN LA RED (o cerca)
                if dist_to_net < NET_ZONE_LIMIT:
                    if ball_speed_kmh > SPEED_SMASH:
                        shot_type = "Remate (Smash)"
                    elif ball_speed_kmh > 45: 
                        shot_type = "Bandeja / Víbora"
                    else:
                        shot_type = "Volea"
                
                # CASO B: JUGADOR EN EL FONDO
                else:
                    if ball_speed_kmh < SPEED_LOB:
                        shot_type = "Globo"
                    elif ball_speed_kmh > 75:
                        shot_type = "Bajada de Pared"
                    else:
                        # Diferenciar Derecha/Revés requiere saber si es diestro 
                        # y posición relativa bola-cuerpo. 
                        # Heurística simple: Lado derecho/izquierdo de la pista?
                        # Mejor dejarlo genérico si no tenemos keypoints
                        shot_type = "Fondo (Drive/Revés)"

                shots.append(Shot(
                    frame=frame,
                    player_id=closest_player_id,
                    shot_type=shot_type,
                    ball_speed=ball_speed_kmh
                ))
                last_shot_frame = frame

        return pd.DataFrame([vars(s) for s in shots])
