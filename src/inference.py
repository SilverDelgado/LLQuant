"""generar el ranking dado el modelo de ML entrenado antes"""

import pandas as pd
import xgboost as xgb
import joblib
import os
import json

MODEL_PATH = "data/models/xgb_alpha_model.json"
META_PATH = "data/models/model_metadata.joblib"
DATA_PATH = "data/processed/training_data.parquet"

def load_inference_artifacts():
    """Carga el modelo entrenado y los metadatos necesarios."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No hay modelo entrenado.")
    
    metadata = joblib.load(META_PATH)
    features = metadata["features"]
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    print(f"[INFERENCIA] Modelo cargado. IC Score del entrenamiento: {metadata['ic']:.4f}")
    return model, features

def get_latest_market_state():
    """
    Simula la obtención de datos en tiempo real.
    Toma la ÚLTIMA fila disponible de cada activo en tu dataset procesado.
    """
    df = pd.read_parquet(DATA_PATH)
    # Asumimos que la última fecha del dataset es "HOY"
    last_date = df.index.max()
    current_market = df.loc[last_date].copy()
    print(f"[INFERENCIA] Analizando mercado fecha: {last_date}")
    return current_market

def generate_signals():
    model, feature_names = load_inference_artifacts()
    market_df = get_latest_market_state()
    
    # Predecir; XGBoost nativo requiere DMatrix. Importante que feature_names asegura el orden correcto.
    X_live = xgb.DMatrix(market_df[feature_names])
    preds = model.predict(X_live)
    
    market_df['PREDICTED_ALPHA'] = preds
    
    # GENERAR RANKING; Ordenamos de Mayor Alpha a Menor Alpha
    ranking = market_df.sort_values(by='PREDICTED_ALPHA', ascending=False)
    
    # === SALIDA PARA LLM, el Ranking===
    llm_context = "### DATOS DEL MODELO CUANTITATIVO ###\n"
    llm_context += f"Fecha del corte: {ranking.index[0]}\n\n"
    llm_context += "RANKING DE PREFERENCIA (De mejor a peor Alpha proyectado):\n"
    
    top_picks = []
    
    for i, (ticker, row) in enumerate(ranking.iterrows()):
        rank = i + 1
        alpha = row['PREDICTED_ALPHA']
        line = f"#{rank} {row['ticker']} (Alpha: {alpha:.10f})"
        llm_context += line + "\n"
        
        top_picks.append({
            "ticker": row['ticker'],
            "rank": rank,
            "alpha": float(alpha)
        })
    
    print("[INFERENCIA] INPUT GENERADO PARA EL AGENTE")
    print(llm_context)
    
    return llm_context, top_picks

if __name__ == "__main__":
    generate_signals()