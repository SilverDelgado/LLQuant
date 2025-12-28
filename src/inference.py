"""generar el ranking dado el modelo de ML entrenado antes"""

import pandas as pd
import xgboost as xgb
import os
import sys
from typing import Optional

# Asegurar que el directorio raíz esté en el path para imports locales
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data import get_df_data

MODEL_PATH = "data/models/alpha_xgboost.json"
DATA_PATH = "data/processed/training_data_4h.parquet"


def _prepare_latest_market_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Obtiene la última fila por ticker de un DataFrame arbitrario."""
    if "ticker" in df.columns:
        df_sorted = df.sort_index()
        return df_sorted.groupby("ticker").tail(1)
    return df

def load_inference_artifacts():
    """Carga el modelo entrenado"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No hay modelo entrenado en {MODEL_PATH}")
    
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    print(f"[INFERENCIA] Modelo cargado desde {MODEL_PATH}")
    return model

def get_latest_market_state(rebuild_on_fail: bool = True, symbols: Optional[list] = None, timeframe: str = "4h"):
    """
    Obtiene el último estado del mercado para cada activo.
    Si el parquet está corrupto u ocupa un formato inválido, intenta reconstruirlo on-the-fly
    usando get_df_data en modo inferencia.
    
    Para inferencia, usamos parámetros reducidos:
    - window=50 (en lugar de 500) para fracdiff
    - horizon=8 (predicción a 8 velas)
    - vpt_price_d_window=6 
    - vol_window=12
    """
    try:
        df = pd.read_parquet(DATA_PATH)
    except Exception as err:
        if not rebuild_on_fail:
            raise
        print(f"[INFERENCIA][WARN] No se pudo leer {DATA_PATH}: {err}. Reconstruyendo dataset en memoria...")
        df = get_df_data(
            symbols=symbols, 
            timeframe=timeframe, 
            horizon=8,
            d=0.4,
            window=50,
            vpt_price_d_window=6,
            vol_window=12,
            limit=600,
            verbose=True, 
            inference_mode=True
        )

    latest = df.groupby('ticker').last()
    print(f"[INFERENCIA] Analizando {len(latest)} activos en el último estado del mercado")
    return latest

def generate_signals(market_df: Optional[pd.DataFrame] = None):
    model = load_inference_artifacts()
    market_df = _prepare_latest_market_snapshot(market_df) if market_df is not None else get_latest_market_state()
    
    # Obtener features neutralizadas (las mismas que usamos en entrenamiento)
    feature_cols = [c for c in market_df.columns if c.endswith('_neutral')]
    if not feature_cols:
        raise ValueError("No hay columnas *_neutral disponibles para inferencia")

    print(f"[INFERENCIA] Usando {len(feature_cols)} features: {feature_cols}")
    
    X_live = market_df[feature_cols]
    
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