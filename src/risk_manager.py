"""Gestor de Riesgo Nivel 2"""

import statistics

def obtener_configuracion_perfil(nombre_perfil):
    """
    Diccionario que define la agresividad del inversor.
    
    - 'base_exposure': % máximo de capital a usar en condiciones ideales.
    - 'sensibilidad': Qué tan rápido corta el riesgo al perder dinero.
      (Ej: Sensibilidad 4 significa que con un 25% de Drawdown, la exposición cae a 0).
    """
    perfiles = {
        "bajo_riesgo": {
            "descripcion": "Preservación de capital. Freno muy rápido.",
            "base_exposure": 0.40,  # Max 40% invertido
            "sensibilidad": 5.0     # Kill Switch al 20% de Drawdown (1/0.20)
        },
        "medio_riesgo": {
            "descripcion": "Balanceado. Estándar cripto.",
            "base_exposure": 0.60,  # Max 60% invertido
            "sensibilidad": 3.0     # Kill Switch al 33% de Drawdown (1/0.33)
        },
        "alto_riesgo": {
            "descripcion": "Crecimiento agresivo. Tolera grandes caídas.",
            "base_exposure": 0.85,  # Max 85% invertido
            "sensibilidad": 2.0     # Kill Switch al 50% de Drawdown (1/0.50)
        }
    }
    
    return perfiles.get(nombre_perfil.lower())

def calcular_factor_salud(drawdown_actual, sensibilidad):
    """
    Fórmula: 1 - (Sensibilidad * Drawdown)
    Si el resultado es negativo, devuelve 0 (Protección contra deuda).
    """
    # El drawdown debe entrar como decimal positivo (ej: 0.10 para 10%)
    factor = 1.0 - (sensibilidad * drawdown_actual)
    return max(0.0, factor) # Nunca devolver menos de 0

def calcular_factor_calidad(scores_activos, top_n=3):
    """
    Promedio de los Conviction Scores del Top N de activos.
    """
    if not scores_activos:
        return 0.0
    
    # Ordenamos de mayor a menor por si acaso no vienen ordenados
    scores_ordenados = sorted(scores_activos, reverse=True)
    
    # Tomamos solo el Top N
    top_scores = scores_ordenados[:top_n]
    
    # Calculamos promedio
    promedio = statistics.mean(top_scores)
    return promedio

def motor_de_riesgo(perfil, drawdown_actual, lista_scores_llm, top_n=3):
    """
    Función Principal (Nivel 2)
    """
    # 1. Cargar Configuración
    config = obtener_configuracion_perfil(perfil)
    if not config:
        return "Error: Perfil no encontrado."
    
    base = config['base_exposure']
    sens = config['sensibilidad']
    
    # 2. Calcular Factores
    f_salud = calcular_factor_salud(drawdown_actual, sens)
    f_calidad = calcular_factor_calidad(lista_scores_llm, top_n)
    
    # 3. Fórmula Maestra
    exposicion_final = base * f_salud * f_calidad
    
    return {
        "perfil": perfil,
        "drawdown_input": f"{drawdown_actual*100}%",
        "detalles": {
            "base_exposure": f"{base:.2f}",
            "factor_salud": f"{f_salud:.2f}",
            "factor_calidad_llm": f"{f_calidad:.2f}"
        },
        "RESULTADO_FINAL_CAPITAL": f"{exposicion_final:.4f}", # En decimal
        "RESULTADO_FINAL_PCT": f"{exposicion_final*100:.2f}%" # En porcentaje
    }

if __name__ == "__main__":
    # --- EJEMPLO DE USO ---

    # Datos de entrada simulados (Tu realidad actual)
    mi_drawdown = 0.12  # Estás perdiendo un 12% desde tu máximo
    mis_scores_llm = [0.90, 0.85, 0.80, 0.60, 0.40, 0.20] # El LLM ve el mercado bien (Top 3 muy fuerte)

    # Probamos con el perfil MEDIO (El recomendado)
    #resultado = motor_de_riesgo("bajo_riesgo", mi_drawdown, mis_scores_llm)
    #resultado = motor_de_riesgo("medio_riesgo", mi_drawdown, mis_scores_llm)
    resultado = motor_de_riesgo("alto_riesgo", mi_drawdown, mis_scores_llm)

    print("--- REPORTE DE GESTIÓN DE RIESGO ---")
    print(f"Perfil: {resultado['perfil']}")
    print(f"Estado de cuenta (Drawdown): {resultado['drawdown_input']}")
    print("-" * 30)
    print(f"1. Base Configurada:      {resultado['detalles']['base_exposure']}")
    print(f"2. Factor Salud (PnL):    {resultado['detalles']['factor_salud']} (Te penalizó por el drawdown)")
    print(f"3. Factor Calidad (IA):   {resultado['detalles']['factor_calidad_llm']} (Confianza alta del LLM)")
    print("-" * 30)
    print(f"CAPITAL A USAR: {resultado['RESULTADO_FINAL_PCT']}")