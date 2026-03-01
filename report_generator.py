import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
import numpy as np
from datetime import datetime

# --- CONFIGURACIÓN DE ESTILO ---
COLOR_PRIMARY = (23, 43, 77)      # Azul Oscuro
COLOR_SECONDARY = (0, 190, 255)   # Azul Cian
FONT_FAMILY = 'Helvetica'

class ProfessionalPadelReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=False) 

    def header(self):
        self.set_fill_color(*COLOR_PRIMARY)
        self.rect(0, 0, 210, 20, 'F')
        self.set_font(FONT_FAMILY, 'B', 12)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 5)
        self.cell(0, 10, 'PADEL ANALISIS TFG - INFORME TÉCNICO', ln=True, align='L')

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT_FAMILY, 'I', 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'Creado por PADEL ANALISIS TFG Luis Miguel Sanz Fernandez', align='C')

    def draw_cover_page(self, video_name="Análisis de Partido"):
        self.add_page()
        self.set_fill_color(*COLOR_PRIMARY)
        self.rect(0, 0, 210, 297, 'F')
        
        self.set_y(110)
        self.set_font(FONT_FAMILY, 'B', 30)
        self.set_text_color(255, 255, 255)
        self.multi_cell(0, 15, f"INFORME DE RENDIMIENTO\n{video_name}", align='C')
        
        self.set_y(150)
        self.set_font(FONT_FAMILY, '', 14)
        date_str = datetime.now().strftime("%d / %m / %Y")
        self.cell(0, 10, f"Fecha: {date_str}", align='C', ln=True)
        
        self.set_y(260)
        self.set_font(FONT_FAMILY, 'B', 10)
        self.cell(0, 10, "TFG - Luis Miguel Sanz Fernandez", align='C')

    def add_section_title(self, title: str):
        self.ln(10)
        self.set_font(FONT_FAMILY, 'B', 16)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 10, title, ln=True)
        # Línea separadora elegante
        self.set_draw_color(*COLOR_SECONDARY)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

# --- RESUMEN EJECUTIVO (TEXTO AUTO-GENERADO) ---
def build_executive_summary(stats: list, shots_df: pd.DataFrame = None) -> str:
    # stats: [{'id': 1, 'distancia': X, 'v_max': Y, 'v_avg': Z}, ...]
    if not stats: return "No hay suficientes datos físicos para generar un resumen."
    
    # Encontrar MVP físico (el que más corre)
    most_active = max(stats, key=lambda x: x['distancia'])
    fastest = max(stats, key=lambda x: x['v_max'])
    
    summary = (
        f"El Jugador {most_active['id']} ha sido el jugador más activo del fragmento analizado, "
        f"cubriendo una distancia total de {most_active['distancia']:.1f} metros. "
    )
    
    if fastest['v_max'] > 10:
        summary += (
            f"El pico de velocidad punta en carrera se ha registrado en el Jugador {fastest['id']}, "
            f"alcanzando los {fastest['v_max']:.1f} km/h en máxima aceleración. "
        )
        
    if shots_df is not None and not shots_df.empty:
        total_shots = len(shots_df)
        most_shots_pid = shots_df['player_id'].mode()[0] if not shots_df.empty else "N/A"
        summary += (
            f"A nivel técnico, se han detectado un total de {total_shots} impactos reales de bola, "
            f"siendo el Jugador {int(most_shots_pid)} el que más intervenciones con la pala ha tenido."
        )
        
    return summary

# --- GRÁFICOS MATPLOTLIB ---

def draw_court_lines(ax):
    """
    Dibuja la pista centrada.
    """
    line_color = 'black' 
    line_width = 1.5
    z = 10 
    
    # Contorno (-5 a 5 en X, -10 a 10 en Y)
    ax.plot([-5, 5], [-10, -10], color=line_color, lw=line_width, zorder=z)
    ax.plot([-5, 5], [10, 10], color=line_color, lw=line_width, zorder=z)
    ax.plot([-5, -5], [-10, 10], color=line_color, lw=line_width, zorder=z)
    ax.plot([5, 5], [-10, 10], color=line_color, lw=line_width, zorder=z)
    
    # Red (en Y=0)
    ax.plot([-5, 5], [0, 0], color='black', lw=3, linestyle='-', zorder=z) 
    
    # Líneas de saque (+/- 7m)
    ax.plot([-5, 5], [-7, -7], color=line_color, lw=line_width, zorder=z)
    ax.plot([-5, 5], [7, 7], color=line_color, lw=line_width, zorder=z)
    
    # Línea central
    ax.plot([0, 0], [-10, -7], color=line_color, lw=line_width, zorder=z)
    ax.plot([0, 0], [7, 10], color=line_color, lw=line_width, zorder=z)

def generate_player_heatmap(df, player_id, filename):
    fig, ax = plt.subplots(figsize=(4, 8))
    
    # Fondo Claro para contraste estilo pizarra
    ax.set_facecolor('#F8F9FA') 
    
    # Límites con margen
    ax.set_xlim(-6, 6)
    ax.set_ylim(-11, 11)
    
    ax.invert_yaxis() 
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    col_x = f"player{player_id}_x"
    col_y = f"player{player_id}_y"
    
    if col_x in df.columns:
        # Filtrar datos dentro de pista y válidos
        data = df[(df[col_x] >= -6) & (df[col_x] <= 6) & (df[col_y] >= -11) & (df[col_y] <= 11)]
        
        if len(data) > 10:
            sns.kdeplot(
                x=data[col_x], 
                y=data[col_y], 
                fill=True, 
                cmap="YlOrRd", # Mapa de calor Rojo/Naranja
                alpha=0.75,     
                thresh=0.1, 
                levels=10,
                ax=ax,
                zorder=1
            )
            # Puntos dispersos para referencia
            ax.scatter(data[col_x], data[col_y], color='black', s=2, alpha=0.05, zorder=2)
    
    draw_court_lines(ax)
    
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def generate_comparison_chart(stats_data, filename):
    players = [f"J{d['id']}" for d in stats_data]
    values = [d['distancia'] for d in stats_data]
    
    plt.figure(figsize=(8, 3))
    colors = ['#2980b9', '#34495e', '#16a085', '#d35400'] # Paleta más corporativa
    
    # Quitar bordes innecesarios (Spines)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    bars = plt.barh(players, values, color=colors, alpha=0.9)
    plt.title("Distancia Recorrida (Metros)", fontweight='bold', color='#2c3e50')
    plt.xlabel("Metros", color='#7f8c8d')
    plt.grid(axis='x', linestyle=':', alpha=0.5, color='#95a5a6')
    ax.tick_params(axis='y', left=False) # Quitar rayitas del eje Y
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                 f'{int(width)}m', va='center', fontsize=9, color='#34495e', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def generate_timeline(df, filename):
    plt.figure(figsize=(10, 3))
    colors = ['#2980b9', '#34495e', '#16a085', '#d35400']
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i in range(1, 5):
        col = f'player{i}_Vnorm4'
        if col in df.columns:
            # Suavizado para gráfica limpia
            y = df[col].abs().rolling(window=45, min_periods=1).mean() * 3.6
            plt.plot(df['time'], y, label=f'J{i}', color=colors[i-1], lw=1.8, alpha=0.85)
            
    plt.legend(loc='upper right', frameon=False, fontsize='small')
    plt.title("Evolución de Intensidad Físico-Táctica (km/h)", fontweight='bold', color='#2c3e50')
    plt.ylabel("km/h", color='#7f8c8d')
    plt.xlabel("Tiempo (s)", color='#7f8c8d')
    plt.grid(True, linestyle=':', alpha=0.5, color='#95a5a6')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# --- GENERADOR DEL REPORTE ---

def create_full_report(csv_path, output_pdf="Informe_Partido.pdf", shots_csv_path=None, shots_chart_path=None):
    try:
        df = pd.read_csv(csv_path)
    except:
        return

    # Preparar Datos Básicos
    stats = []
    for i in range(1, 5):
        d_col = f'player{i}_distance'
        v_col = f'player{i}_Vnorm4'
        
        dist = df[d_col].sum() if d_col in df.columns else 0
        v_max = (df[v_col].abs().max() * 3.6) if v_col in df.columns else 0
        v_avg = (df[v_col].abs().mean() * 3.6) if v_col in df.columns else 0
        
        stats.append({'id': i, 'distancia': dist, 'v_max': v_max, 'v_avg': v_avg})

    # Generar Imágenes Matplotlib
    generate_comparison_chart(stats, "chart_bars.png")
    generate_timeline(df, "chart_time.png")
    
    heat_imgs = []
    for i in range(1, 5):
        fname = f"heat_p{i}.png"
        generate_player_heatmap(df, i, fname)
        heat_imgs.append(fname)

    # --- INICIO PDF ---
    pdf = ProfessionalPadelReport()
    
    # Intentar cargar CSV de golpes para el resumen
    shots_df_loaded = None
    if shots_csv_path and os.path.exists(shots_csv_path):
        try:
            shots_df_loaded = pd.read_csv(shots_csv_path)
        except:
            pass

    # P1: PORTADA
    pdf.draw_cover_page()
    
    # P2: DATOS FÍSICOS Y RESUMEN
    pdf.add_page()
    pdf.add_section_title("1. Resumen Ejecutivo")
    
    # Bloque de Texto Ejecutivo
    summary_text = build_executive_summary(stats, shots_df_loaded)
    pdf.set_font(FONT_FAMILY, '', 11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 7, summary_text)
    
    pdf.add_section_title("2. Análisis Físico y Métricas")
    
    # Tabla FÍSICA Estilo ZEBRA
    pdf.set_font(FONT_FAMILY, 'B', 10)
    pdf.set_text_color(255, 255, 255) # Texto blanco para cabecera
    pdf.set_fill_color(*COLOR_PRIMARY) # Fondo cabecera corporativo
    
    cols = ["Jugador", "Distancia (m)", "Vel. Máx (km/h)", "Vel. Media (km/h)"]
    widths = [30, 40, 45, 45]
    x_start = (210 - sum(widths)) / 2
    
    pdf.set_x(x_start)
    for w, c in zip(widths, cols):
        pdf.cell(w, 8, c, border=0, fill=True, align='C')
    pdf.ln()
    
    pdf.set_font(FONT_FAMILY, '', 10)
    pdf.set_text_color(40, 40, 40)
    
    for row_idx, s in enumerate(stats):
        pdf.set_x(x_start)
        # Modo Zebra
        if row_idx % 2 == 0:
            pdf.set_fill_color(245, 245, 245) # Fila Gris Clara
        else:
             pdf.set_fill_color(255, 255, 255) # Fila Blanca

        pdf.cell(widths[0], 8, f"Jugador {s['id']}", border='B', align='C', fill=True)
        pdf.cell(widths[1], 8, f"{s['distancia']:.1f}", border='B', align='C', fill=True)
        pdf.cell(widths[2], 8, f"{s['v_max']:.1f}", border='B', align='C', fill=True)
        pdf.cell(widths[3], 8, f"{s['v_avg']:.1f}", border='B', align='C', fill=True)
        pdf.ln()
    
    pdf.ln(8)
    pdf.image("chart_bars.png", x=20, w=170)
    pdf.ln(5)
    pdf.image("chart_time.png", x=15, w=180)
    
    # P3: HEATMAPS (GRID FIJO)
    pdf.add_page()
    pdf.add_section_title("3. Ocupación de Pista (Mapas de Calor)")
    
    # GRID 2x2
    y_row1 = 50
    y_row2 = 160
    
    # J1 (Abajo Izquierda)
    pdf.set_xy(30, y_row1)
    pdf.set_font(FONT_FAMILY, 'B', 12)
    pdf.cell(60, 10, "Jugador 1", align='C')
    pdf.image(heat_imgs[0], x=35, y=y_row1+10, w=50)
    
    # J2 (Abajo Derecha)
    pdf.set_xy(120, y_row1)
    pdf.cell(60, 10, "Jugador 2", align='C')
    pdf.image(heat_imgs[1], x=125, y=y_row1+10, w=50)
    
    # J3 (Arriba Izquierda)
    pdf.set_xy(30, y_row2)
    pdf.cell(60, 10, "Jugador 3", align='C')
    pdf.image(heat_imgs[2], x=35, y=y_row2+10, w=50)
    
    # J4 (Arriba Derecha)
    pdf.set_xy(120, y_row2)
    pdf.cell(60, 10, "Jugador 4", align='C')
    pdf.image(heat_imgs[3], x=125, y=y_row2+10, w=50)
    
    # P4: ANÁLISIS DE GOLPES (NUEVA PÁGINA)
    if shots_csv_path and os.path.exists(shots_csv_path):
        pdf.add_page()
        pdf.add_section_title("4. Análisis de Golpes Detectados")

        # 1. Gráfico de Golpes
        if shots_chart_path and os.path.exists(shots_chart_path):
             pdf.image(shots_chart_path, x=10, w=190)
             pdf.ln(10)

        # 2. Tabla de Golpes ZEBRA
        pdf.set_font(FONT_FAMILY, 'B', 12)
        pdf.set_text_color(*COLOR_PRIMARY)
        pdf.cell(0, 10, "Detalle Cronológico de Impactos", ln=True)
        pdf.ln(3)
        
        try:
            # Reutilizar df cargado arriba
            shots_df = shots_df_loaded if shots_df_loaded is not None else pd.read_csv(shots_csv_path)
            
            # Encabezado Tabla Corporativo
            pdf.set_font(FONT_FAMILY, 'B', 10)
            pdf.set_text_color(255, 255, 255)
            pdf.set_fill_color(*COLOR_PRIMARY)
            
            table_cols = ["Jugador", "Frame", "Velocidad (km/h)", "Tipo"]
            table_widths = [30, 40, 40, 50]
            
            x_table = (210 - sum(table_widths)) / 2
            pdf.set_x(x_table)
            
            for w, c in zip(table_widths, table_cols):
                pdf.cell(w, 8, c, border=0, fill=True, align='C')
            pdf.ln()
            
            # Filas
            pdf.set_font(FONT_FAMILY, '', 9)
            pdf.set_text_color(40, 40, 40)
            
            for idx, row in shots_df.iterrows():
                if idx > 35: break # Evitar desbordamiento de página simple
                
                pdf.set_x(x_table)
                
                if idx % 2 == 0:
                    pdf.set_fill_color(248, 248, 248) 
                else:
                    pdf.set_fill_color(255, 255, 255)

                pid = f"J{int(row.get('player_id', 0))}"
                frame = str(int(row.get('frame', 0)))
                speed = f"{float(row.get('ball_speed', 0)):.1f}"
                stype = str(row.get('shot_type', 'N/A'))
                
                pdf.cell(table_widths[0], 7, pid, border='B', align='C', fill=True)
                pdf.cell(table_widths[1], 7, frame, border='B', align='C', fill=True)
                pdf.cell(table_widths[2], 7, speed, border='B', align='C', fill=True)
                pdf.cell(table_widths[3], 7, stype, border='B', align='C', fill=True)
                pdf.ln()

            if len(shots_df) > 35:
                pdf.set_font(FONT_FAMILY, 'I', 8)
                pdf.cell(0, 10, f"... y {len(shots_df)-35} golpes más no mostrados.", align='C')

        except Exception as e:
            pdf.cell(0, 10, f"Error cargando datos de golpes: {e}", ln=True)

    pdf.output(output_pdf)
    
    # Limpieza de archivos temporales
    cleanup_files = ["chart_bars.png", "chart_time.png"] + heat_imgs
    if shots_csv_path: cleanup_files.append(shots_csv_path)
    if shots_chart_path: cleanup_files.append(shots_chart_path)

    for f in cleanup_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

if __name__ == "__main__":
    create_full_report("padel_analytics_report.csv")
