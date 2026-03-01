from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

@dataclass
class Shot:
    frame: int
    player_id: int
    ball_speed: float
    shot_type: str

class ShotDetector:
    def __init__(self):
        # Configuración de Pista (ajusta si tu homografía es distinta)
        self.COURT_WIDTH = 10.0
        self.COURT_HEIGHT = 20.0
        
        # Umbrales de Velocidad (km/h)
        self.SMASH_TH = 100.0
        self.BAJADA_TH = 60.0    # Bajadas son rápidas
        self.LOB_TH = 45.0       # Máximo para ser globo
        
        # Zonas (Metros desde la red, asumiendo red en 0)
        self.NET_ZONE = 4.0
        self.BASE_ZONE = 7.0 # A partir de 7m es fondo
        
    def detect_shots(self, df: pd.DataFrame, fps: float) -> pd.DataFrame:
        shots = []
        
        # Necesitamos la velocidad suavizada de la bola (Vnorm4 es ideal)
        if 'ball_Vnorm4' not in df.columns:
            return pd.DataFrame()

        # 1. DETECCIÓN DE EVENTOS: ¿Cuándo gana energía la bola?
        # Una pala añade velocidad. El suelo y las paredes (casi siempre) la restan.
        # Calculamos la aceleración (cambio de velocidad)
        ball_v = df['ball_Vnorm4'].to_numpy() # m/s
        ball_acc = np.diff(ball_v, prepend=ball_v[0])
        
        # Buscamos picos donde la aceleración sea positiva (bola acelera)
        # height=2.0 m/s² es un umbral para ignorar ruido pequeño
        # distance=10 frames evita detectar el mismo golpe dos veces seguidas
        peaks, _ = find_peaks(ball_acc, height=1.5, distance=int(fps*0.4))
        
        for frame_idx in peaks:
            # --- FILTRADO DE FALSOS POSITIVOS (PAREDES/SUELO) ---
            
            # Posición de la bola en el impacto
            bx = df.loc[frame_idx, 'ball_x']
            by = df.loc[frame_idx, 'ball_y']
            
            # 1. Filtro Paredes: Si el impacto es en los límites de la pista, es rebote
            # Márgenes de 0.5 metros
            if (abs(bx) > (self.COURT_WIDTH/2 - 0.5)) or (abs(by) > (self.COURT_HEIGHT/2 - 0.5)):
                continue 

            # 2. Asignación de Jugador: ¿Quién está más cerca?
            closest_pid = None
            min_dist = 2.5 # Radio máximo de alcance (metros)
            
            for pid in [1, 2, 3, 4]:
                px = df.loc[frame_idx, f'player{pid}_x']
                py = df.loc[frame_idx, f'player{pid}_y']
                dist = np.sqrt((bx - px)**2 + (by - py)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_pid = pid
            
            if closest_pid is None:
                continue # Nadie cerca, probablemente rebote en suelo o error de tracking
                
            # --- CLASIFICACIÓN DEL GOLPE ---
            
            # Velocidad en km/h
            speed_kmh = ball_v[frame_idx] * 3.6
            
            # Datos del jugador
            py_player = df.loc[frame_idx, f'player{closest_pid}_y']
            dist_net = abs(py_player) # Asumiendo red en Y=0
            
            # Heurística Derecha vs Revés (Asumiendo diestros)
            # Vector Jugador -> Bola
            px_player = df.loc[frame_idx, f'player{closest_pid}_x']
            dx = bx - px_player
            dy = by - py_player # Dirección Y relativa
            
            # Determinamos lado de la pista (Arriba/Abajo)
            # Jugadores 1 y 2 suelen estar abajo (Y positivo), 3 y 4 arriba (Y negativo)
            # Ajusta esto según tu configuración de tracking inicial
            
            # Lógica simple de orientación:
            # Si el jugador está mirando a la red...
            # Derecha: Bola a su derecha. Revés: Bola a su izquierda.
            is_forehand = False
            
            if py_player > 0: # Lado inferior, mirando hacia arriba (Y decreciente)
                if dx > 0: is_forehand = True # Bola a la derecha
            else: # Lado superior, mirando hacia abajo (Y creciente)
                if dx < 0: is_forehand = True # Bola a su derecha (izquierda de la pantalla)
                
            side_stroke = "Derecha" if is_forehand else "Revés"

            # Lógica de Tipo de Golpe
            shot_type = "Desconocido"
            
            # 1. REMATE
            if speed_kmh > self.SMASH_TH:
                shot_type = "Remate"
            
            # 2. ZONA FONDO (> 7m)
            elif dist_net > self.BASE_ZONE:
                if speed_kmh > self.BAJADA_TH:
                    shot_type = "Bajada de Pared"
                elif speed_kmh < self.LOB_TH:
                    shot_type = "Globo"
                else:
                    shot_type = side_stroke # Derecha o Revés de fondo
            
            # 3. ZONA RED (< 4m)
            elif dist_net < self.NET_ZONE:
                shot_type = "Volea"
            
            # 4. ZONA MEDIA (4m - 7m)
            else:
                # Aquí suele ser bandeja o volea de transición
                # Usamos la velocidad vertical del jugador para ver si recula
                vy_col = f'player{closest_pid}_Vy4'
                p_vy = df.loc[frame_idx, vy_col] if vy_col in df.columns else 0
                
                moving_back = False
                if py_player > 0 and p_vy > 0.5: moving_back = True
                if py_player < 0 and p_vy < -0.5: moving_back = True
                
                if moving_back and speed_kmh > 40:
                    shot_type = "Bandeja / Víbora"
                elif speed_kmh < self.LOB_TH:
                     shot_type = "Globo" # Globo defensivo corto
                else:
                    shot_type = side_stroke # Golpe de aproximación
            
            shots.append({
                'frame': frame_idx,
                'player_id': closest_pid,
                'shot_type': shot_type,
                'ball_speed': speed_kmh
            })
            
        return pd.DataFrame(shots)
