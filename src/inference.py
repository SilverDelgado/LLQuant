"""generar el ranking dado el modelo de ML entrenado antes"""

import pandas as pd
import xgboost as xgb
import joblib
import os
import json

MODEL_PATH = "data/models/alpha_xgboost.json"
DATA_PATH = "data/processed/training_data_4h.parquet"

FEATURES = [
    'alpha_past_neutral',  # (-0.05)
    'vpt_neutral',         # (-0.03)
    'zscore_neutral',      # (+0.02)
    'rsi_neutral',         # (+0.018)
    'atr_neutral',         # (+0.016)
    'ret_vol_ratio_neutral' # (+0.018)
]

def load_inference_artifacts():
    """Carga el modelo entrenado"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No hay modelo entrenado en {MODEL_PATH}")
    
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    print(f"[INFERENCIA] Modelo cargado desde {MODEL_PATH}")
    return model

def get_latest_market_state():
    """
    Obtiene el último estado del mercado para cada activo.
    Toma la última fila disponible de cada activo en el dataset procesado.
    """
    df = pd.read_parquet(DATA_PATH)
    
    # Obtener la última fecha para cada ticker
    latest = df.groupby('ticker').last()
    
    print(f"[INFERENCIA] Analizando {len(latest)} activos en el último estado del mercado")
    return latest

def generate_signals(market_df: pd.DataFrame | None = None):
    model = load_inference_artifacts()
    # Permite usar un dataset en memoria si se provee; si no, carga desde disco
    if market_df is None:
        market_df = get_latest_market_state()
    else:
        # Si viene con múltiples filas por ticker, tomar la última observación
        if 'ticker' in market_df.columns:
            market_df = market_df.groupby('ticker').last()
        else:
            raise ValueError("El dataset de mercado debe incluir columna 'ticker'.")
    
    # Obtener features neutralizadas (las mismas que usamos en entrenamiento)
    feature_cols = [c for c in market_df.columns if c.endswith('_neutral')]
    missing = [f for f in FEATURES if f not in market_df.columns]
    if missing:
        raise ValueError(
            "Faltan features requeridas para inferencia: " + ", ".join(missing) +
            "\nSugerencia: asegúrate de que el pipeline de inferencia calcule 'alpha_past_neutral' como en training."
        )
    print(f"[INFERENCIA] Usando {len(FEATURES)} features: {FEATURES}")
    X_live = market_df[FEATURES]
    
    # Predecir alpha con XGBRegressor
    preds = model.predict(X_live)
    
    market_df['PREDICTED_ALPHA'] = preds
    
    # Generar ranking: ordenar de mayor a menor alpha predicho
    ranking = market_df.sort_values(by='PREDICTED_ALPHA', ascending=False)
    
    # Salida para LLM
    llm_context = "### DATOS DEL MODELO CUANTITATIVO ###\n"
    llm_context += f"Total de activos analizados: {len(ranking)}\n\n"
    llm_context += "RANKING DE PREFERENCIA (De mejor a peor Alpha proyectado):\n"
    
    top_picks = []
    
    for i, (ticker, row) in enumerate(ranking.iterrows()):
        rank = i + 1
        alpha = row['PREDICTED_ALPHA']
        line = f"#{rank} {ticker} (Alpha: {alpha:.6f})"
        llm_context += line + "\n"
        
        top_picks.append({
            "ticker": ticker,
            "rank": rank,
            "alpha": float(alpha)
        })
    
    print("\n" + "="*60)
    print("[INFERENCIA] RANKING GENERADO")
    print("="*60)
    print(llm_context)
    
    return llm_context, top_picks

if __name__ == "__main__":
    generate_signals()