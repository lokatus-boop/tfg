def __init__(
        self, 
        tracking_model_path: str,
        # ... resto de argumentos ...
    ):
        super().__init__(load_path=load_path, save_path=save_path)

        # ... código existente ...

        # --- AÑADE ESTO AL FINAL DEL __INIT__ ---
        # Asumimos 30 FPS por defecto, idealmente pásalo como argumento si varía
        self.kalman = PadelBallKalmanFilter(fps=30)
