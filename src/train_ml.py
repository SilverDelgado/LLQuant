import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

FEATURES = [
    'alpha_past_neutral',
    'vpt_neutral',
    'zscore_neutral',
    'rsi_neutral',
    'atr_neutral',
    'ret_vol_ratio_neutral'
]


def train_alpha_model():
    print("="*60)
    print("ENTRENAMIENTO XGBOOST")
    print("="*60)
    
    df = pd.read_parquet("data/processed/training_data_4h.parquet")
    print(f"[INFO] Datos cargados: {df.shape}")
    
    feature_cols = [c for c in df.columns if c.endswith('_neutral')]
    print(f"[INFO] Features: {len(feature_cols)} -> {feature_cols}")
    print(f"[INFO] Selected features: {FEATURES}")
    X = df[FEATURES]
    y = df['TARGET_ALPHA']
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    rankics = []
    
    model = xgb.XGBRegressor(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        reg_lambda=1.0,
        reg_alpha=0.0,
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=13
    )
    
    print("\n[TRAINING] Cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold+1}/5")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        pred_val = model.predict(X_val)
        
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
    
    model.get_booster().save_model("data/models/alpha_xgboost.json")
    print("[OK] Modelo guardado en: data/models/alpha_xgboost.json")
    
    return model, rankics

if __name__ == "__main__":
    train_alpha_model()