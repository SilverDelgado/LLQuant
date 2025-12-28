"""Pipeline Fase 1: FracDiff -> Features Neutras -> Alpha Target"""
import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import norm
import warnings
from utils import (
    fracdiff_fixed_window,
    rsi_price_d,
    bollinger_features,
    atr_price_d,
    volume_imbalance,
    vpt_price_d,
    stoch_price_d,
)
warnings.filterwarnings('ignore')


# Indicadores y transformaciones importados desde utils

# LIMPIEZA Y ALINEACIÓN ====================

def clean_and_align_data(input_dir, timeframe):
    """Carga, pivotea y asegura alineación temporal perfecta"""
    print(f"[DATA] Cargando y alineando archivos de tf: {timeframe}... ")
    all_files = glob.glob(os.path.join(input_dir, "cmt_*.parquet")) 
    
    suffix = f"_{timeframe}"
    files = [f for f in all_files if suffix in f]

    if not files:
        raise ValueError(f"No se encontraron archivos en {input_dir} en tf {timeframe}")
    
    all_data = []
    for file in files:
        filename = os.path.basename(file)
        ticker = filename.replace('cmt_', '').replace(suffix, '').replace('.parquet', '')
        df = pd.read_parquet(file)
        df = df[~df.index.duplicated(keep='first')] #elimina duplicados
        df['ticker'] = ticker
        all_data.append(df)
    
    # Concatenar
    full_df = pd.concat(all_data)
    df_pivot = full_df.pivot_table( #creamos tabla ancha (btc | eth | xxx) (pivotar)
        index=full_df.index, 
        columns='ticker', 
        values=['open', 'high', 'low', 'close', 'volume']
    )
    
    # Forward fill con max 3 huecos
    df_pivot = df_pivot.ffill(limit=3)
    
    # volvemos desde tabla ancha a la larga (lo util para procesar por ML)
    df_stacked = df_pivot.stack(future_stack=True).reset_index()
    df_stacked.columns = ['timestamp', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_stacked = df_stacked.set_index('timestamp').sort_index()
    
    num_tickers = len(files)
    # validacion timestamp con x tickers cada
    counts = df_stacked.groupby('timestamp').size()
    if not (counts == num_tickers).all():
        print(f"[FATAL]: {len(counts[counts != num_tickers])} timestamps incompletos")
        df_stacked = df_stacked.loc[counts[counts == num_tickers].index]
    
    return df_stacked

# FEATURE ENGINEERING ====================

def add_technical_indicators(df, d=0.4, window=500, vpt_price_d_window=168):
    """
    Calcula TODOS los indicadores SOBRE price_d, luego neutraliza (comparando monedas entre si)
    Neutralización: No queremos saber el valor absoluto de cierto indicador para un precio, sino saber que este valor es mejor que los demás (cross sectional)
    """
    print(f"[FEATURES] Calculando indicadores sobre FracDiff (d={d})...")
    
    processed = []
    
    for ticker, group in df.groupby('ticker'):
        group = group.copy()
        
        # FracDiff (truncamos primeros 500 datos)
        price_d = fracdiff_fixed_window(group['Close'], d=d, window=window)
        group = group.loc[price_d.index]  # alieamos
        group['price_d'] = price_d.values
        # Indicadores técnicos (sobre price_d)
        group['rsi'] = rsi_price_d(group['price_d'], period=14)
        group['zscore'], group['bb_width'] = bollinger_features(group['price_d'], period=20)
        group['atr'] = atr_price_d(group['price_d'], period=14)
        group['stoch'] = stoch_price_d(group['price_d'], k=14, d=3)
        group['vpt'] = vpt_price_d(group['Volume'], group['price_d'], window=vpt_price_d_window)
        group['vol_imbalance'] = volume_imbalance(group['Volume'], group['price_d'], period=20)
        group['ret_vol_ratio'] = group['price_d'].diff() / (group['atr'] + 1e-9) # Ratio retorno/volatilidad
        
        processed.append(group)
    
    df_feat = pd.concat(processed).sort_index()
    
    # NEUTRALIZACIÓN CROSS-SECTIONAL ====================
    print("[FEATURES] Neutralizando features...")
    
    feature_cols = ['rsi', 'zscore', 'bb_width', 'atr', 'stoch', 'vpt', 'vol_imbalance', 'ret_vol_ratio']
    
    for col in feature_cols:
        # calcular El Promedio del Mercado
        market_mean = df_feat.groupby('timestamp')[col].transform('mean')
        
        # calcular la posición relativa (Centrado) ***donde está***
        # Restamos el promedio (media) a nuestra moneda. (nos dice la distancia)
        # Si Resultado > 0: El indicador es MAYOR que el del resto (x.Ej: Tiene más volatilidad, o más RSI).
        # Si Resultado < 0: El indicador es MENOR que el del resto (x.Ej: Está más tranquilo, o más sobrevendido).
        centered_value = df_feat[col] - market_mean
        # calculamos la desviacion std para calcular zscore
        market_std = df_feat.groupby('timestamp')[col].transform('std')
        
        # calculamos el z-score para estandarizar  (para escalas) ***respecto del resto***
        # +2.0 significa: "Extremadamente alto comparado con el resto".
        # 0.0 significa: "Exactamente igual al promedio del mercado".
        z_score_neutral = centered_value / (market_std + 1e-9)
        
        # save
        df_feat[f'{col}_neutral'] = z_score_neutral

    return df_feat

# LABEL ENGINEERING (ALPHA) ====================

def calculate_target_alpha(df, horizon=8, vol_window=168, timeframe='1h'):
    """
    Alpha relativo con vol scaling
    ->¿Cuánto va a subir esta moneda en el futuro COMPARADO con el mercado y AJUSTADO por su riesgo?
    """
    print(f"[TARGET] Calculando Alpha (horizonte={horizon} velas, Vol Window={vol_window})...")
    
    # queremos entrenar modelo para que sepa lo que va a predecir el martes si hoy es lunes 
    # (cogemos precio martes y lo movemos a lunes, asi ve indicadores de lunes pero respuesta de martes)
    # calculamos cuanto cambiara el precio en las próximas 'horizon' velas.
    future_return = df.groupby('ticker')['Close'].pct_change(periods=horizon).shift(-horizon)
    
    # neutralización de nuevo (alpha raw)
    # si btc sube 5% pero mercado sube 10%, alpha es -5%. 
    #tenemos neutralizacion en el target y en los indicadores :)
    # market_mean = future_return.groupby(df.index).transform('mean')
    valid_mask = future_return.notna()
    future_return_clean = future_return.where(valid_mask)
    market_mean = future_return_clean.groupby(df.index).transform(
        lambda x: x.fillna(0).sum() / valid_mask.groupby(df.index).sum()
    )
    alpha_raw = future_return - market_mean
    
    # Volatility scaling 
    # ojo: calculamos volatilidad usando SOLO datos pasados (shift 1)
    # importante, ajusta para cada activo: 
    # -moneda estable: raro que se mueva por ejemplo 1% (Alpha 1% / Vol 0.1%) = Puntuación 10.  (mas valor) 
    # -moneda inestable: normal que se mueva 5% (Alpha 1% / Vol 5%) = Puntuación 0.2.
    # dividimos por volatilidad para hacer un escalado en cuanto a la importancia
    returns = df.groupby('ticker')['Close'].pct_change()

    #shift(horizon) para que vol no "vea" el período del target Y rolling con min_periods para evitar arrastre
    vol_1_period = returns.groupby(df['ticker']).shift(horizon).rolling(
        window=vol_window,
        min_periods=vol_window//2,
        center=False  # Asegurar que no mira al futuro
    ).std()
    vol_horizon = vol_1_period
    # vol_horizon = vol_1_period * np.sqrt(horizon)
    alpha_risk_adjusted = alpha_raw / (vol_horizon + 1e-9)

    """
    Imagina que entrenas el modelo en 2022 (mercado bajista, volatilidad alta, Alpha std=0.3). 
    Luego predices en 2023 (ranging, volatilidad baja, Alpha std=0.05). 
    El modelo se vuelve loco porque el target cambió de escala.
    """

    # current_std = alpha_risk_adjusted.std() #normalizamos el Alpha a desviación estándar 0.1, sin importar como de volátil sea el período.
    ref_std = alpha_risk_adjusted.iloc[:1000].std()
    target_std = 0.1
    # alpha_scaled = (alpha_risk_adjusted / current_std) * target_std
    alpha_scaled = (alpha_risk_adjusted / ref_std) * target_std

    clip_val = alpha_scaled.abs().quantile(0.99)
    
    # Clip extremos (outliers)
    df['TARGET_ALPHA'] = np.clip(alpha_scaled, -clip_val, clip_val) # afecta exactamente 1% de outliers

    lag1_corr = df['TARGET_ALPHA'].autocorr(lag=1)
    print(f"[DEBUG] Alpha Autocorr (lag=1): {lag1_corr:.4f}")
    if abs(lag1_corr) > 0.05:
        print("[WARNING] Autocorr sigue alto, considera aumentar vol_window o horizon")
    
    return df

def add_alpha_lag_features(df, horizon=24):
    """
    Usa Retorno Pasado Realizado como feature
    El lag debe ser el Alpha de [t-horizon, t], que está 100% disponible en t
    Añade lag del Alpha para modelar la autocorrelación NEGATIVA (reversión a la media).

    El Alpha calculado muestra una autocorrelación de -0.13 (lag=1). Esto NO es un bug, 
    es una PROPIEDAD DEL MERCADO: los activos que se desvían fuertemente del promedio 
    en un período tienden a revertir hacia la media en el siguiente período.
    
    Si SOL superó al mercado en +5% en las últimas 4 horas, las siguientes 4 horas es 
    estadísticamente más probable que SOL *underperform* para compensar esa desviación.
    Esto es "reversión a la media" o "mean reversion".
    
    El lag feature le permite al modelo aprender.
    
    IMPORTANTE:
    El lag DEBE ser neutralizado cross-sectional (alpha_lag1_neutral), 
    de lo contrario el modelo aprendería "comprar perdedores" en vez de 
    "comprar activos que revierten desde extremos relativos".
    """
 
    print(f"[FEATURES] Añadiendo Alpha Pasado (Reversión a la Media)...")
    
    # 1. Retorno PASADO (24h hacia atrás) [SAFE]
    past_return = df.groupby('ticker')['Close'].pct_change(periods=horizon)
    
    # 2. Neutralización cross-sectional del PASADO (Centrar en 0)
    market_mean_past = past_return.groupby(df.index).transform('mean')
    alpha_past_raw = past_return - market_mean_past
    
    # 3. Vol scaling del PASADO (Ajustar por riesgo individual)
    vol_past = df.groupby('ticker')['Close'].pct_change().rolling(horizon).std() * np.sqrt(horizon)
    alpha_past_risk_adj = alpha_past_raw / (vol_past + 1e-9)
    
    # --- CAMBIO AQUÍ ---
    # 4. Z-Score Cross-Sectional Final
    # En lugar de usar una referencia estática (iloc[:1000]), forzamos a que
    # EN CADA HORA, la distribución de los features sea Mean=0, Std=1.
    # Esto hace que el modelo sea robusto a cambios de régimen de volatilidad.
    
    cs_mean = alpha_past_risk_adj.groupby(df.index).transform('mean')
    cs_std = alpha_past_risk_adj.groupby(df.index).transform('std')
    
    df['alpha_past_neutral'] = (alpha_past_risk_adj - cs_mean) / (cs_std + 1e-9)
    
    # Clipping suave (opcional, para limpieza)
    df['alpha_past_neutral'] = df['alpha_past_neutral'].clip(-4, 4)
    
    return df

# PIPELINE PRINCIPAL ====================

def process_pipeline(
    horizon=8,
    d=0.4,
    window=500,
    vpt_price_d_window=168,
    vol_window=168,
    timeframe='1h',
    df_base: pd.DataFrame | None = None,
    save_file: bool = True,
    inference_mode: bool = False,
):
    """Ejecuta pipeline completo.

    Parámetros:
    - df_base: DataFrame ya cargado con columnas ['timestamp' (index), 'ticker', 'Open','High','Low','Close','Volume'].
               Si se proporciona, se usará directamente y NO se leerán archivos.
    - save_file: Si True, guarda el dataset final en parquet; si False, no guarda.
    - inference_mode: Si True, NO calcula TARGET_ALPHA (modo inferencia, no hay datos futuros).
    """
    input_dir = "data/raw"
    output_dir = "data/processed"
    print(f"[PROCESSING DATA] " + "="*50)

    # Cargar/usar datos base
    if df_base is None:
        print(f"[INFO] input_dir:{input_dir} -> output_dir:{output_dir} -- [OK]")
        df_clean = clean_and_align_data(input_dir, timeframe=timeframe)
    else:
        # Asegurar formato esperado
        df_clean = df_base.copy()
        if df_clean.index.name != 'timestamp':
            df_clean = df_clean.set_index('timestamp')
        df_clean = df_clean.sort_index()
        print(f"[INFO] Datos base recibidos en memoria: {len(df_clean)} filas")
    
    df_features = add_technical_indicators(df_clean, d=d, window=window, vpt_price_d_window=vpt_price_d_window)
    print(f"[INFO] Features calculados: {len(df_features)} filas")
    
    if inference_mode:
        # Modo inferencia: No calcular target alpha, solo features
        df_final = add_alpha_lag_features(df_features, horizon=horizon)
        feature_cols = [c for c in df_final.columns if c.endswith('_neutral')]
        cols_to_save = ['ticker', 'Close'] + feature_cols
        final_dataset = df_final[cols_to_save]
        
        # Eliminar solo filas con NaN en features (no en target que no existe)
        rows_before = len(final_dataset)
        final_dataset = final_dataset.dropna()
        rows_after = len(final_dataset)
        print(f"[INFO] Filas eliminadas por NaNs en features: {rows_before - rows_after}")
    else:
        # Modo entrenamiento: Calcular target alpha
        df_final = calculate_target_alpha(df_features, horizon=horizon, vol_window=vol_window, timeframe=timeframe)
        df_final = add_alpha_lag_features(df_final, horizon=horizon)
        
        feature_cols = [c for c in df_final.columns if c.endswith('_neutral')]
        cols_to_save = ['ticker', 'Close'] + feature_cols + ['TARGET_ALPHA']
        final_dataset = df_final[cols_to_save]
        
        rows_before = len(final_dataset)
        final_dataset = final_dataset.dropna()
        rows_after = len(final_dataset)
        print(f"[INFO] Filas eliminadas por NaNs (Warm-up + Horizon): {rows_before - rows_after}")
    
    # Guardar opcionalmente
    save_path = os.path.join(output_dir, f"training_data_{timeframe}.parquet")
    if save_file:
        os.makedirs(output_dir, exist_ok=True)
        final_dataset.to_parquet(save_path)
    
    print("\n" + "="*50)
    print(f"[OK] PIPELINE COMPLETADO")
    print(f"Dimensiones: {final_dataset.shape}")
    if save_file:
        print(f"Guardado en: {save_path}")
    else:
        print(f"Guardado: [omitido] (save_file=False)")
    print(f"Features finales: {len(feature_cols)}")
    if not inference_mode and 'TARGET_ALPHA' in final_dataset.columns:
        print(f"Rank de Alpha: {final_dataset['TARGET_ALPHA'].min():.4f} a {final_dataset['TARGET_ALPHA'].max():.4f}")
    elif inference_mode:
        print(f"Modo: INFERENCIA (sin target alpha)")
    print("="*50)
    
    return final_dataset

if __name__ == "__main__":
    """
    horizon: distancia al futuro que el modelo intenta adivinar
    """
    # df_final = process_pipeline(horizon=8, d=0.4, window=500, vpt_price_d_window = 168, vol_window=168, timeframe='1h') #8h
    # df_final = process_pipeline(horizon=24, d=0.4, window=50, vpt_price_d_window = 24, vol_window=48, timeframe='1h') #24h
    # df_final = process_pipeline(horizon=8, d=0.4, window=50, vpt_price_d_window = 24, vol_window=48, timeframe='1h') #8h
    # df_final = process_pipeline(horizon=48, d=0.4, window=50, vpt_price_d_window=24, vol_window=96, timeframe='1h') #48h
    df_final = process_pipeline(horizon=8,d=0.4, window=50,vpt_price_d_window=6,vol_window=12,timeframe='4h') # alpha_past_neutral (Spearman: -0.0498)

    