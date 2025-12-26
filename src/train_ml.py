"""archivo para entrenar el modelo XGBOOST de ML con cross sectional regression
    debe recibir una matriz donde:
        Las filas son cada activo en un momento determinado
        Los features son los indicadores tecnicos calculados T
        El objetivo es el Alpha Relativo que ese activo tuvo en T+1"""
import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

MODEL_DIR = "data/models"
DATA_PATH = "data/processed/training_data.parquet"
TARGET = "TARGET_ALPHA"

FEATURES = [
    'ml_feature_price', 
    'rsi', 
    'bb_pos', 
    'norm_atr', 
    'rel_vol'
]

# ==================== FUNCIONES DE CARGA ====================

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado en: {path}")
    
    df = pd.read_parquet(path)
    df = df.sort_index()
    df = df.dropna()
    
    print(f"[DATA] Datos cargados: {df.shape}")
    print(f"Rango: {df.index.min()} -> {df.index.max()}")
    return df

def data_split(df, val_size=0.15, test_size=0.15):
    """
    Divide los datos en bloques temporales estrictos.
    [ TRAIN (70%) | VALIDATION (15%) | TEST (15%) ]
    """
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size)) # El val se saca del trozo restante
    
    train = df.iloc[:val_idx]
    val = df.iloc[val_idx:test_idx]
    test = df.iloc[test_idx:]
    
    print(f"[DATA] División Temporal")
    print(f"[DATA] Train: {len(train)} filas (Hasta {train.index.max()})")
    print(f"[DATA] Val:   {len(val)} filas")
    print(f"[DATA] Test:  {len(test)} filas")
    
    return train, val, test

# ==================== CORE ML ====================

def train_xgboost(train, val):
    print("[TRAIN] Entrenando XGBoost")
    
    X_train = train[FEATURES]
    y_train = train[TARGET]
    
    X_val = val[FEATURES]
    y_val = val[TARGET]
    
    # model = xgb.XGBRegressor(
    #     objective='reg:squarederror',
    #     n_estimators=5000,      
    #     learning_rate=0.005,    
    #     max_depth=4,            # Árboles x profundos (
    #     subsample=0.7,          # x% de datos por árbol
    #     colsample_bytree=0.8,   # x% de features por árbol
    #     early_stopping_rounds=50, # PARAR si no mejora en x rondas 
    #     n_jobs=-1,
    #     random_state=13
    # )

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=5000,
        learning_rate=0.05,     # ANTES 0.005. Ahora aprende 10x más rápido.
        max_depth=8,            # ANTES 4. Ahora busca relaciones mucho más profundas.
        min_child_weight=1,     # Permite que el árbol haga grupos más pequeños (más diferenciación).
        gamma=0.0,              # Sin barrera mínima para dividir nodos.
        subsample=0.8,          
        colsample_bytree=1.0,   # Que mire TODOS los indicadores, no solo el 80%.
        reg_alpha=0.0,          # Sin regularización L1 (permite features ruidosos).
        reg_lambda=1.0,         # Regularización L2 estándar.
        early_stopping_rounds=200, # Mucha más paciencia.
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    return model

def evaluate_performance(model, test_df):
    """
    Calcula el IC (Information Coefficient).
    Si esto es positivo, el modelo sabe ordenar activos.
    """
    print("[OOS] Evaluación en TEST")
    
    X_test = test_df[FEATURES]
    y_real = test_df[TARGET]
    
    preds = model.predict(X_test)
    results = test_df.copy()
    results['prediction'] = preds
    
    # Correlación de Spearman por cada hora (Cross-Sectional)
    # Pregunta: "¿Acertamos el orden de las monedas en esta hora?"
    daily_ic = results.groupby(level=0).apply(
        lambda x: x['prediction'].corr(x[TARGET], method='spearman')
    )
    
    mean_ic = daily_ic.mean()
    std_ic = daily_ic.std()
    ir = mean_ic / (std_ic + 1e-9) # Information Ratio
    
    print(f"[OOS] RESULTADOS FINANCIEROS:")
    print(f"IC Promedio: {mean_ic:.4f}")# (Objetivo: > 0.01)
    print(f"IC Ratio: {ir:.4f}")#(Objetivo: > 0.05)
    
    xgb.plot_importance(model, max_num_features=10)
    plt.title("Features más importantes") #SABER ASI CUALES IMPORTAN MAS Y CUALES QUITAR
    plt.show()
    
    return mean_ic

def save_model(model, ic_score):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "xgb_alpha_model.json")
    model.get_booster().save_model(path)
    meta_path = os.path.join(MODEL_DIR, "model_metadata.joblib")
    joblib.dump({"features": FEATURES, "ic": ic_score}, meta_path)
    
    print(f"[OK] Modelo guardado en: {path}")

# ==================== ORQUESTADOR ====================

def run_training():
    df = load_data(DATA_PATH)
    train, val, test = data_split(df)
    model = train_xgboost(train, val)
    ic_score = evaluate_performance(model, test)
    """IC mira la correlación entre lo que predijo el modelo y lo que realmente pasó, 
    analizado de forma Cross-Sectional en cada momento del tiempo.
    Sirve para validar el ranking que devolveremos, esa capacidad de ordenar"""
    if ic_score > 0:
        save_model(model, ic_score)
    else:
        print("[FATAL] IC is negative.")

if __name__ == "__main__":
    run_training()