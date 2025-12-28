"""archivo para entrenar el modelo XGBOOST de ML con cross sectional regression
    debe recibir una matriz donde:
        Las filas son cada activo en un momento determinado
        Los features son los indicadores tecnicos calculados T
        El objetivo es el Alpha Relativo que ese activo tuvo en T+1
        
        Entrenamiento XGBoost con RankIC

        """

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

FEATURES = [
    'alpha_past_neutral',  # (-0.05)
    'vpt_neutral',         # (-0.03)
    'zscore_neutral',      # (+0.02)
    'rsi_neutral',         # (+0.018)
    'atr_neutral',         # (+0.016)
    'ret_vol_ratio_neutral' # (+0.018)
]


def train_alpha_model():
    print("="*60)
    print("ENTRENAMIENTO XGBOOST")
    print("="*60)
    
    # Cargar datos
    df = pd.read_parquet("data/processed/training_data_4h.parquet")
    print(f"[INFO] Datos cargados: {df.shape}")
    
    # Features y target
    feature_cols = [c for c in df.columns if c.endswith('_neutral')]
    print(f"[INFO] Features: {len(feature_cols)} -> {feature_cols}")
    print(f"[INFO] Selected features: {FEATURES}")
    X = df[FEATURES]
    y = df['TARGET_ALPHA']
    
    # TimeSeriesSplit (NO shuffle)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Métricas
    rankics = []
    
    # Modelo
    model = xgb.XGBRegressor(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        reg_lambda=1.0,  # L2 regularización
        reg_alpha=0.0,   # L1 regularización,
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=13
    )
    
    print("\n[TRAINING] Cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold+1}/5")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Entrenar
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predicciones
        pred_val = model.predict(X_val)
        
        # RankIC cross-sectional
        val_data = df.iloc[val_idx].copy()
        val_data['pred'] = pred_val
        
        daily_rankic = val_data.groupby('timestamp').apply(
            lambda x: spearmanr(x['pred'], x['TARGET_ALPHA'])[0] if len(x) >= 5 else np.nan
        )
        
        mean_rankic = daily_rankic.mean()
        rankics.append(mean_rankic)
        print(f"  RankIC medio: {mean_rankic:.4f}")
    
    
    print("\n" + "="*60)
    print(f"[RESULTS] RANKIC PROMEDIO: {np.mean(rankics):.4f} ± {np.std(rankics):.4f}")
    print(f"[RESULTS] CONSISTENCIA: {np.mean([r > 0 for r in rankics]):.1%}")
    print("="*60)
    
    # Guardar modelo
    model.get_booster().save_model("data/models/alpha_xgboost.json")
    print("[OK] Modelo guardado en: data/models/alpha_xgboost.json")
    
    return model, rankics

if __name__ == "__main__":
    train_alpha_model()