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
        Detects shots based on ball acceleration peaks and player proximity.
        """
        shots = []
        
        # Ensure we have ball data
        if "ball_Vnorm1" not in df.columns:
            return pd.DataFrame()

        # 1. Detect impacts based on high acceleration/velocity change
        # We look for peaks in ball acceleration or sudden changes in velocity direction
        # For simplicity, let's look for local maxima in acceleration that exceed a threshold
        # AND are close to a player.
        
        # Thresholds (heuristic)
        ACCEL_THRESHOLD = 50.0 # m/s^2 - arbitrary, needs tuning
        PROXIMITY_THRESHOLD = 2.0 # meters
        
        # Smoothing to reduce noise
        df["ball_Anorm1_smooth"] = df["ball_Anorm1"].rolling(window=3, center=True).mean()
        
        # Find peaks
        # A peak is where accel[t] > accel[t-1] and accel[t] > accel[t+1]
        potential_impacts = []
        for i in range(2, len(df) - 2):
            accel_prev = df.iloc[i-1]["ball_Anorm1_smooth"]
            accel_curr = df.iloc[i]["ball_Anorm1_smooth"]
            accel_next = df.iloc[i+1]["ball_Anorm1_smooth"]
            
            if accel_curr > ACCEL_THRESHOLD and accel_curr > accel_prev and accel_curr > accel_next:
                potential_impacts.append(i)
        
        # Filter impacts by player proximity
        last_shot_frame = -100
        
        for idx in potential_impacts:
            row = df.iloc[idx]
            frame = int(row["frame"])
            
            # Debounce: avoid multiple detections for the same shot
            if frame - last_shot_frame < fps * 0.5: # 0.5 seconds debounce
                continue
                
            ball_x = row["ball_x"]
            ball_y = row["ball_y"]
            
            if pd.isna(ball_x) or pd.isna(ball_y):
                continue

            closest_player_id = None
            min_dist = float("inf")
            
            for player_id in (1, 2, 3, 4):
                p_x = row[f"player{player_id}_x"]
                p_y = row[f"player{player_id}_y"]
                
                if pd.isna(p_x) or pd.isna(p_y):
                    continue
                    
                dist = np.sqrt((ball_x - p_x)**2 + (ball_y - p_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_player_id = player_id
            
            if closest_player_id and min_dist < PROXIMITY_THRESHOLD:
                # Classify shot
                # Simple logic: Net vs Baseline
                # Assuming court is roughly 10m half-length (20m total)
                # Net is at 0 (or center). Baseline is at +/- 10.
                # Need to check coordinate system in projected_court.py
                # Usually origin is center or net.
                
                # Let's assume standard coordinates where y is length.
                # If y is close to 0 -> Net -> Volley
                # If y is far -> Baseline -> Drive
                
                player_y = row[f"player{closest_player_id}_y"]
                
                # Heuristic: Volley if within 3 meters of net?
                # We need to know where the net is. 
                # Based on projected_court.py:
                # origin is calculated from k6 (net post).
                # So (0,0) is likely the net center or post.
                
                if abs(player_y) < 3.5:
                    shot_type = "Volley"
                else:
                    shot_type = "Drive"
                
                shots.append(Shot(
                    frame=frame,
                    player_id=closest_player_id,
                    shot_type=shot_type,
                    ball_speed=row["ball_Vnorm1"] * 3.6 # km/h
                ))
                last_shot_frame = frame

        return pd.DataFrame([vars(s) for s in shots])
