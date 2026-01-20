import pandas as pd
import xgboost as xgb
import joblib
import os
import json
MODEL_PATH = "data/models/alpha_xgboost.json"
DATA_PATH = "data/processed/training_data_4h.parquet"
FEATURES = [
    'alpha_past_neutral',
    'vpt_neutral',
    'zscore_neutral',
    'rsi_neutral',
    'atr_neutral',
    'ret_vol_ratio_neutral'
]
def load_inference_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No hay modelo entrenado en {MODEL_PATH}")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    print(f"[INFERENCIA] Modelo cargado desde {MODEL_PATH}")
    return model
def get_latest_market_state():
    df = pd.read_parquet(DATA_PATH)
    latest = df.groupby('ticker').last()
    print(f"[INFERENCIA] Analizando {len(latest)} activos en el último estado del mercado")
    return latest
def generate_signals(market_df: pd.DataFrame | None = None):
    model = load_inference_artifacts()
    if market_df is None:
        market_df = get_latest_market_state()
    else:
        if 'ticker' in market_df.columns:
            market_df = market_df.groupby('ticker').last()
        else:
            raise ValueError("El dataset de mercado debe incluir columna 'ticker'.")
    feature_cols = [c for c in market_df.columns if c.endswith('_neutral')]
    missing = [f for f in FEATURES if f not in market_df.columns]
    if missing:
        raise ValueError(
            "Faltan features requeridas para inferencia: " + ", ".join(missing) +
            "\nSugerencia: asegúrate de que el pipeline de inferencia calcule 'alpha_past_neutral' como en training."
        )
    print(f"[INFERENCIA] Usando {len(FEATURES)} features: {FEATURES}")
    X_live = market_df[FEATURES]
    preds = model.predict(X_live)
    market_df['PREDICTED_ALPHA'] = preds
    ranking = market_df.sort_values(by='PREDICTED_ALPHA', ascending=False)
    llm_context = "
    llm_context += f"Total de activos analizados: {len(ranking)}\n\n"
    llm_context += "RANKING DE PREFERENCIA (De mejor a peor Alpha proyectado):\n"
    top_picks = []
    for i, (ticker, row) in enumerate(ranking.iterrows()):
        rank = i + 1
        alpha = row['PREDICTED_ALPHA']
        line = f"
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