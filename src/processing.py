"""archivo para limpiar los datos descargados, generar indicadores y el cálculo de target (alpha relativo)"""
import pandas as pd
import pandas_ta as ta # Librería profesional de indicadores
import numpy as np
import os
import glob

import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import glob

# ==================== Math Utils FracDiff ====================

def get_weights_ffd(d, thres=1e-5):
    """Genera los pesos para la diferenciación fraccional."""
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])

def frac_diff_ffd(series, d=0.4, thres=1e-5):
    """Aplica diferenciación fraccional manteniendo memoria."""
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    if len(series) < width:
        return pd.Series(np.nan, index=series.index)
    return series.rolling(window=len(w)).apply(lambda x: np.dot(x, w), raw=True)

# ==================== FUNCIONES DEL PIPELINE ====================

def clean_and_align_data(input_dir):
    """
    FASE 1: Ingesta y Limpieza
    Carga los CSVs, alinea las fechas y rellena huecos.
    """
    print("--- 1. Cargando y Alineando Datos ---")
    files = glob.glob(os.path.join(input_dir, "*_raw.csv"))
    all_data = []

    for file in files:
        ticker = os.path.basename(file).split('_raw')[0].replace('_', '-')
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        df['ticker'] = ticker
        # Eliminar duplicados en el índice por si acaso
        df = df[~df.index.duplicated(keep='first')]
        all_data.append(df)

    if not all_data:
        raise ValueError("No se encontraron archivos CSV en data/raw")

    full_df = pd.concat(all_data)
    
    # Pivotar para crear una matriz temporal perfecta (todos los activos en las mismas filas)
    df_pivot = full_df.pivot_table(index=full_df.index, columns='ticker', values=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Rellenar huecos con el dato anterior (Forward Fill)
    df_pivot = df_pivot.ffill()
    
    # Volver al formato largo (Filas: Fecha-Activo)
    df_stacked = df_pivot.stack(future_stack=True).reset_index()
    df_stacked.rename(columns={'level_1': 'ticker'}, inplace=True)
    df_stacked = df_stacked.set_index('Date').sort_index()
    
    return df_stacked

def add_technical_indicators(df, use_frac_diff=True):
    """
    FASE 2: Ingeniería de Features
    Calcula FracDiff y osciladores técnicos para cada activo.
    """
    print(f"--- 2. Generando Indicadores (FracDiff={use_frac_diff}) ---")
    processed_dfs = []

    # Agrupamos por ticker para no mezclar datos de BTC con ETH
    for ticker, group in df.groupby('ticker'):
        group = group.copy()
        
        # A. INPUT PARA EL MODELO (Lo que ve el XGBoost)
        if use_frac_diff:
            # Precio con memoria pero estacionario
            group['ml_feature_price'] = frac_diff_ffd(group['Close'], d=0.4)
        else:
            # Fallback clásico
            group['ml_feature_price'] = np.log(group['Close'] / group['Close'].shift(1))
        
        # B. INDICADORES TÉCNICOS (Librería pandas-ta)
        # RSI (Momentum)
        group['rsi'] = group.ta.rsi(length=14)
        
        # Bandas de Bollinger (%B - Posición relativa)
        bb = group.ta.bbands(length=20, std=2)
        group['bb_pos'] = bb['BBP_20_2.0']
        
        # ATR Normalizado (Volatilidad relativa al precio)
        group['atr'] = group.ta.atr(length=14)
        group['norm_atr'] = group['atr'] / group['Close']
        
        # Z-Score de Volumen (Anomalías de volumen)
        vol_mean = group['Volume'].rolling(24).mean()
        vol_std = group['Volume'].rolling(24).std()
        group['rel_vol'] = (group['Volume'] - vol_mean) / (vol_std + 1e-9)

        processed_dfs.append(group)

    return pd.concat(processed_dfs).sort_index()

def calculate_target_alpha(df, horizon=1):
    """
    FASE 3: Definición del Objetivo (Target)
    Calcula el Alpha Relativo (Retorno Activo - Retorno Promedio Mercado).
    """
    print(f"--- 3. Calculando Target (Alpha a {horizon}h) ---")
    
    # 1. Calcular Retorno Futuro Real (Sin FracDiff, dinero real)
    # shift(-horizon) mira hacia el futuro
    df['future_ret'] = df.groupby('ticker')['Close'].pct_change(horizon).shift(-horizon)
    
    # 2. Calcular el Promedio del Mercado en ese futuro (Cross-Sectional Mean)
    # Agrupamos por índice (fecha) para sacar la media de todos los activos en ese momento
    market_mean = df.groupby(level=0)['future_ret'].transform('mean')
    
    # 3. Calcular Alpha (Diferencial)
    df['TARGET_ALPHA'] = df['future_ret'] - market_mean
    
    return df

# ==================== ORQUESTADOR ====================

def process_pipeline(horizon=1, use_frac_diff=True):
    # Definición de rutas
    input_dir = "data/raw"
    output_dir = "data/processed"
    
    # 1. Limpieza
    df_clean = clean_and_align_data(input_dir)
    
    # 2. Indicadores
    df_features = add_technical_indicators(df_clean, use_frac_diff=use_frac_diff)
    
    # 3. Target
    df_final = calculate_target_alpha(df_features, horizon=horizon)
    
    # 4. Guardado Final
    df_final.dropna(inplace=True)
    
    # Seleccionamos solo lo que necesita el modelo
    features = ['ml_feature_price', 'rsi', 'bb_pos', 'norm_atr', 'rel_vol']
    cols_to_save = ['ticker'] + features + ['TARGET_ALPHA']
    
    final_dataset = df_final[cols_to_save]
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "training_data.parquet")
    
    final_dataset.to_parquet(save_path)
    
    print(f"✅ Pipeline completado.")
    print(f"Dimensiones: {final_dataset.shape}")
    print(f"Guardado en: {save_path}")
    print(final_dataset.head())

if __name__ == "__main__":
    process_pipeline(horizon=1, use_frac_diff=True)