import numpy as np

class PadelBallKalmanFilter:
    def __init__(self, fps: int, process_noise: float = 0.5, measurement_noise: float = 10.0):
        """
        Inicializa el filtro de Kalman para una pelota de pádel (2D).
        
        Estado (x): [pos_x, pos_y, vel_x, vel_y]
        
        Args:
            fps: Frames por segundo del video.
            process_noise: Confianza en el modelo físico (menor = más rígido).
            measurement_noise: Ruido de detección TrackNet (mayor = suaviza más).
        """
        self.dt = 1 / fps
        
        # --- Matrices del Modelo ---
        
        # Matriz de Transición (F): Física newtoniana con fricción
        self.friction = 0.995  
        self.F = np.array([
            [1, 0, self.dt, 0],       
            [0, 1, 0, self.dt],       
            [0, 0, self.friction, 0], 
            [0, 0, 0, self.friction]  
        ])

        # Matriz de Control (B) para la gravedad
        # IMPORTANTE: Ajustar 9.8 * X según píxeles/metro de tu video
        g_pixels = 9.8 * 20  
        self.B = np.array([
            [0],
            [0.5 * self.dt**2],
            [0],
            [self.dt]
        ])
        self.u = np.array([[g_pixels]]) 

        # Matriz de Medición (H)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # --- Covarianzas ---
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 1000

        # Estado Inicial
        self.x = np.zeros((4, 1))
        
        self.initialized = False
        self.missed_frames = 0
        self.max_missed_frames_to_reset = int(fps * 1.0) # Reset tras 1 segundo sin bola

    def predict(self) -> tuple[float, float]:
        """
        Paso de Predicción: ¿Dónde debería estar la bola según la física?
        """
        # --- CORRECCIÓN AÑADIDA: Lógica de Reset ---
        if self.missed_frames > self.max_missed_frames_to_reset:
            self.initialized = False
        # -------------------------------------------

        if not self.initialized:
            return 0.0, 0.0

        # x = Fx + Bu
        self.x = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        
        # P = FPF' + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        self.missed_frames += 1
        return float(self.x[0]), float(self.x[1])

    def update(self, measurement_xy: tuple[float, float]):
        """
        Paso de Corrección: Corregir predicción con el dato real.
        """
        z = np.array([[measurement_xy[0]], [measurement_xy[1]]]) 

        if not self.initialized:
            self.x = np.array([
                [z[0,0]], 
                [z[1,0]], 
                [0], 
                [0]
            ])
            self.initialized = True
            self.missed_frames = 0
            return

        # Kalman Gain (K)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        try:
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        except np.linalg.LinAlgError:
            return # Evitar crash si la matriz es singular

        # Update State
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        # Update Covariance
        I = np.eye(4)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        self.missed_frames = 0

    def get_state(self):
        """Devuelve (x, y) suavizado"""
        return float(self.x[0]), float(self.x[1])
