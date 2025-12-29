"""Pipeline Fase 1: FracDiff -> Features Neutras -> Alpha Target"""
import importlib.util
import pandas as pd
import numpy as np
import os
import glob
import sys
from pathlib import Path
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# FRACDIFF  ====================

def fracdiff_fixed_window(series, d=0.4, window=500):
    """
    Fixed-Window Fractional Differenciación con ventana fija para calcular este valor sin look ahead bias.
    para calcular el valor de HOY, la fórmula matemática necesita multiplicar y sumar los precios de las últimas x velas(window)

    """
    if len(series) < window:
        return pd.Series(np.nan, index=series.index)
    
    # Pesos FFD
    weights = [1.0] #peso del dato actual (hoy) == siempre 1
    for k in range(1, window):# calculamos cuánto nos importa el dato de ayer,antesdeayer,hace3dias... (pesos van decayendo)
        #simplemente creamos lista de coeficientes(weights) de len (window) con la info de window y del d
        weights.append(-weights[-1] * (d - k + 1) / k) # weight = -(weight anterior) *(d - k + 1) / k
    weights = np.array(weights[::-1]) #Reordenamos weights: [pequeño, mediano, ...., Grande] para la funcion convolve
    values = series.fillna(method='ffill').fillna(0).values#rellenamos huecos
    diff_values = np.convolve(values, weights, mode='valid') #tomamos la ventana de 500 pesos y la deslizamos sobre las velas, en cada paso
    #mode valid para que solo calcule una vez tiene lo suficiente (numero 499).
    """
    eliminamos primeros datos window que son inestables, como valid
    elimino primero los 499 inestbales, nos queda un array mas corto
    (con 1000 velas y window 500, diff_values tiene 501)
    """
    new_index = series.index[window-1:]
    return pd.Series(diff_values, index=new_index) #obtenemos serie transformada y alineada con los valores estacionarios abstractos
    #estos valores abstractos representan la fuerza del precio conservando la memoria y siendo estacionarios matemáticamente

# INDICADORES MANUALES sobre price_diferenciado fraccionalmente ====================

def rsi_price_d(price_d, period=14):
    """RSI calculado SOBRE la serie diferenciada"""
    delta = price_d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_features(price_d, period=20, std=2):
    """Retorna Z-Score y Band Width sobre price_d"""
    mid = price_d.rolling(period).mean()
    std_dev = price_d.rolling(period).std()
    zscore = (price_d - mid) / std_dev
    width = (std_dev * std * 2) / mid.abs()
    return zscore, width

def atr_price_d(price_d, period=14):
    """ATR simplificado para price_d (usa diferencia ventana 2)"""
    high_low = price_d.rolling(2).max() - price_d.rolling(2).min()
    return high_low.rolling(period).mean()

def volume_imbalance(volume, price_d, period=20):
    """Ratio volumen en velas positivas vs negativas"""
    positive_vol = volume.where(price_d.diff() > 0, 0).rolling(period).sum()
    negative_vol = volume.where(price_d.diff() < 0, 0).rolling(period).sum()
    return (positive_vol - negative_vol) / (positive_vol + negative_vol + 1e-9)

def vpt_price_d(volume, price_d, window=168):
    """Volume Price Trend (Rolling) para mantener estacionariedad"""
    vpt_raw = (volume * price_d.diff() / price_d)
    return vpt_raw.rolling(window).sum()

def stoch_price_d(price_d, k=14, d=3):
    """Stochastic Oscillator sobre price_d"""
    low_min = price_d.rolling(k).min()
    high_max = price_d.rolling(k).max()
    k_line = 100 * (price_d - low_min) / (high_max - low_min + 1e-9)
    return k_line.rolling(d).mean()

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
        vol_relativo = group['Volume'] / (group['Volume'].rolling(24).mean() + 1e-9)
        group['pv_divergence'] = group['price_d'] / (vol_relativo + 1e-9) #Indica movimiento sin fuerza.
        atr_long = group['atr'].rolling(window=100).mean()
        group['volatility_regime'] = group['atr'] / (atr_long + 1e-9)# Ratio entre volatilidad actual y volatilidad de largo plazo
        group['interaction_rsi_vol'] = group['rsi'] * group['volatility_regime']#Un RSI alto = peligroso, pero un RSI alto + ALTA volatilidad = ALERTA.
        
        processed.append(group)
    
    df_feat = pd.concat(processed).sort_index()
    
    # NEUTRALIZACIÓN CROSS-SECTIONAL ====================
    print("[FEATURES] Neutralizando features...")
    
    feature_cols = ['rsi', 'zscore', 'bb_width', 'atr', 'stoch', 'vpt', 'vol_imbalance', 'ret_vol_ratio', 'pv_divergence', 'volatility_regime', 'interaction_rsi_vol']
    
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

# LOOK AHEAD BIAS CHECK ==========================================

def validate_feature_look_ahead(df, feature_name, params, samples=10):
    print(f"[TEST] Validando temporalidad de '{feature_name}'...")

    if 'alpha_past' in feature_name: #test de mutacion
        # Extraemos el horizonte de los params o usamos default
        h = params.get('horizon', 8) 
        passed = run_mutation_check_for_alpha(df, feature_name, horizon=h)
        if passed:
            print(f"[OK] '{feature_name}' pasa la validación (Mutation Test)")
            return True
        else:
            return False

    sample_rows = df.reset_index().iloc[:samples]

    for _, row in sample_rows.iterrows(): # test de recalculo
        timestamp = row['timestamp']
        ticker = row['ticker']
        feature_real = row[feature_name]

        feature_manual = recalcula_con_funciones_originales(df, timestamp, feature_name, ticker, params)

        if np.isnan(feature_manual):
            continue

        # tol
        if abs(feature_real - feature_manual) > 1e-5:
            print(f"[FATAL] FALLA en {timestamp} {ticker}: Real={feature_real:.6f} != Manual={feature_manual:.6f}")
            return False

    print(f"[OK] '{feature_name}' pasa la validación (Recalc Test)")
    return True

def run_mutation_check_for_alpha(df, feature_name, horizon=8):
    """
    Sub-test específico para Alpha Past.
    Modifica el futuro y verifica que el pasado no cambie.
    """
    print(f"---> [SPECIAL TEST] Ejecutando Mutación para {feature_name}...")
    
    # 1. Setup: Copia ligera para no romper nada
    # Cogemos un subconjunto para ir rápido
    tickers = df['ticker'].unique()[:3] 
    df_test = df[df['ticker'].isin(tickers)].copy().sort_index()
    
    # Encontramos un punto T donde tengamos futuro
    unique_times = df_test.index.unique()
    if len(unique_times) < horizon + 5:
        return True # No hay suficientes datos para probar
        
    t_target = unique_times[-10] # Un momento cerca del final
    t_future = unique_times[-9]  # El momento siguiente
    
    # 2. Valor Original
    # (Asumimos que alpha_past ya está calculado en df_test o lo recalculamos)
    # Para estar seguros, lo recalculamos limpio:
    df_clean = df_test.drop(columns=[feature_name], errors='ignore')
    df_orig = add_alpha_lag_features(df_clean.copy(), horizon=horizon)
    
    val_orig = df_orig.loc[t_target, feature_name].iloc[0] # Valor de la primera moneda en T
    
    # 3. Valor Mutado (Destruyendo el futuro)
    df_mutated = df_clean.copy()
    # Multiplicamos el precio de MAÑANA por 1 millón
    df_mutated.loc[t_future, 'Close'] = df_mutated.loc[t_future, 'Close'] * 1_000_000
    
    # Recalculamos feature
    df_mut_calc = add_alpha_lag_features(df_mutated, horizon=horizon)
    val_mut = df_mut_calc.loc[t_target, feature_name].iloc[0]
    
    # 4. Comparación
    print(f"Val original: {val_orig}, Val mutado: {val_mut}")
    diff = abs(val_orig - val_mut)
    if diff < 1e-9:
        return True
    else:
        print(f"[FATAL] MUTATION DETECTED en {feature_name}!")
        print(f"   Original en {t_target}: {val_orig}")
        print(f"   Mutado (con precio {t_future} x1M): {val_mut}")
        return False

def recalcula_con_funciones_originales(df, timestamp, feature_name, ticker, params):
    base_name = feature_name.replace('_neutral', '')
    
    # mascara PARA ELIMINAR FUTURO 
    mask = (df['ticker'] == ticker) & (df.index <= timestamp) #cortamos datos en T
    data_until_t = df.loc[mask].copy()

    if len(data_until_t) < 2:
        return np.nan

    price_d = data_until_t.get('price_d')
    vol = data_until_t.get('Volume')
    base_value = np.nan

    # RECALCULOS PARA COMPARAR
    if 'rsi' in base_name and 'interaction' not in base_name:
        if len(price_d) < params['rsi_period']:
            return np.nan
        base_value = rsi_price_d(price_d, period=params['rsi_period']).iloc[-1]

    elif 'zscore' in base_name or 'bb_width' in base_name:
        if len(price_d) < params['bb_period']:
            return np.nan
        zscore, bb_width = bollinger_features(price_d, period=params['bb_period'], std=params['bb_std'])
        base_value = zscore.iloc[-1] if 'zscore' in base_name else bb_width.iloc[-1]

    elif 'atr' in base_name and 'ret_vol_ratio' not in base_name and 'regime' not in base_name:
        if len(price_d) < params['atr_period']:
            return np.nan
        base_value = atr_price_d(price_d, period=params['atr_period']).iloc[-1]

    elif 'stoch' in base_name:
        if len(price_d) < params['stoch_k']:
            return np.nan
        base_value = stoch_price_d(price_d, k=params['stoch_k'], d=params['stoch_d']).iloc[-1]

    elif 'vpt' in base_name:
        if len(price_d) < params['vpt_price_d_window']:
            return np.nan
        base_value = vpt_price_d(vol, price_d, window=params['vpt_price_d_window']).iloc[-1]

    elif 'vol_imbalance' in base_name:
        if len(price_d) < params['vol_imb_period']:
            return np.nan
        base_value = volume_imbalance(vol, price_d, period=params['vol_imb_period']).iloc[-1]

    elif 'ret_vol_ratio' in base_name:
        atr_series = atr_price_d(price_d, period=params['atr_period'])
        ret = price_d.diff()
        base_value = (ret / (atr_series + 1e-9)).iloc[-1]

    elif 'interaction_rsi_vol' in base_name:
        # recalcular los componentes primero
        rsi = rsi_price_d(price_d, period=params['rsi_period'])
        atr_series = atr_price_d(price_d, period=params['atr_period'])
        atr_long = atr_series.rolling(window=100).mean()
        vol_regime = atr_series / (atr_long + 1e-9)
        base_value = (rsi * vol_regime).iloc[-1]

    elif 'pv_divergence' in base_name:
        # Replicamos la fórmula: price_d / (volumen / media_volumen_24h)
        vol_rel = vol / (vol.rolling(24).mean() + 1e-9)
        base_value = (price_d / (vol_rel + 1e-9)).iloc[-1]
    
    elif 'volatility_regime' in base_name:
        # Replicamos: ATR actual / ATR promedio 100 periodos
        atr_series = atr_price_d(price_d, period=params['atr_period'])
        atr_long = atr_series.rolling(window=100).mean()
        base_value = (atr_series / (atr_long + 1e-9)).iloc[-1]

    else:
        return np.nan

    # Neutralización Cross-Sectional para comparar
    if feature_name.endswith('_neutral'):
        market_vals = df.loc[timestamp, base_name]
        if isinstance(market_vals, pd.Series) or isinstance(market_vals, np.ndarray):
            market_mean = np.mean(market_vals)
            market_std = np.std(market_vals)
        else:
            market_mean = market_vals
            market_std = 0.0
        return (base_value - market_mean) / (market_std + 1e-9)

    return base_value


# PIPELINE PRINCIPAL ====================

def process_pipeline(
    horizon=8,
    d=0.4,
    window=500,
    vpt_price_d_window=168,
    vol_window=168,
    timeframe='1h',
    look_ahead_test=False,
    df_base=None,
    save_file=True,
    inference_mode=False,
):
    """Ejecuta pipeline completo.

    Soporta dos modos de entrada:
    - Sin `df_base`: carga y limpia datos desde `data/raw` (modo entrenamiento).
    - Con `df_base`: usa el DataFrame entregado en memoria (modo inferencia/rápido).

    Si `inference_mode=True`, NO calcula `TARGET_ALPHA` ni features dependientes de ella.
    """
    input_dir = "data/raw"
    output_dir = "data/processed"
    print(f"[PROCESSING DATA] " + "="*50)
    if df_base is None:
        print(f"[INFO] input_dir:{input_dir} -> output_dir:{output_dir} -- [OK]")
        df_clean = clean_and_align_data(input_dir, timeframe=timeframe)
    else:
        print(f"[INFO] usando df_base en memoria -> output_dir:{output_dir}")
        # Asumimos que `df_base` ya viene con índices limpios y alineados (por símbolo)
        df_clean = df_base

    print(f"[INFO] Datos limpios: {len(df_clean)} filas")

    df_features = add_technical_indicators(
        df_clean,
        d=d,
        window=window,
        vpt_price_d_window=vpt_price_d_window,
    )
    print(f"[INFO] Features calculados: {len(df_features)} filas")

    if not inference_mode:
        df_final = calculate_target_alpha(df_features, horizon=horizon, vol_window=vol_window, timeframe=timeframe)
        df_final = add_alpha_lag_features(df_final, horizon=horizon)
    else:
        # En modo inferencia calculamos TARGET para poder derivar alpha_past y su versión neutralizada,
        # pero luego excluimos TARGET_ALPHA de la salida para evitar fuga de información.
        df_tmp = calculate_target_alpha(df_features, horizon=horizon, vol_window=vol_window, timeframe=timeframe)
        df_final = add_alpha_lag_features(df_tmp, horizon=horizon)

    # Selección final (solo features neutrales + target si existe)
    feature_cols = [c for c in df_final.columns if c.endswith('_neutral')]
    cols_to_keep = ['ticker', 'Close'] + feature_cols
    if not inference_mode and 'TARGET_ALPHA' in df_final.columns:
        cols_to_keep.append('TARGET_ALPHA')

    final_dataset = df_final[cols_to_keep]

    rows_before = len(final_dataset)
    final_dataset = final_dataset.dropna()
    rows_after = len(final_dataset)

    print(f"[INFO] Filas eliminadas por NaNs (Warm-up + Horizon): {rows_before - rows_after}")

    # Guardado opcional
    if save_file:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"full_data_{timeframe}.parquet")
        final_dataset.to_parquet(save_path)
    else:
        save_path = None

    print("\n" + "="*50)
    print(f"[OK] PIPELINE COMPLETADO")
    print(f"Dimensiones: {final_dataset.shape}")
    if save_path:
        print(f"Guardado en: {save_path}")
    print(f"Features finales: {len(feature_cols)}")
    if not inference_mode and 'TARGET_ALPHA' in final_dataset.columns:
        print(f"Rank de Alpha: {final_dataset['TARGET_ALPHA'].min():.4f} a {final_dataset['TARGET_ALPHA'].max():.4f}")
    print("="*50)

    if look_ahead_test and not inference_mode:
        print("\n" + "="*50)
        print("[TEST] VALIDANDO TEMPORALIDAD DE FEATURES")
        print("="*50)

        feature_cols = [c for c in df_final.columns if c.endswith('_neutral')]

        params = {
            'horizon': horizon,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'vpt_price_d_window': vpt_price_d_window,
            'vol_imb_period': 20,
        }

        for feature in feature_cols:
            is_valid = validate_feature_look_ahead(df_final, feature, params)
            if not is_valid:
                raise ValueError(f"Feature {feature} tiene look-ahead bias")

        print("[TEST] [OK] Todas las features pasan validación")

    return final_dataset


def split_train_test(full_data_path, train_ratio=0.8, timeframe="4h"):
    """Divide parquet en train test"""
    if not os.path.exists(full_data_path):
        print(f"[ERROR] No se encuentra {full_data_path}")
        return
    
    df = pd.read_parquet(full_data_path)
    print(f"[SPLIT] Dividiendo datos cronológicamente...")
    print(f"[SPLIT] Total de filas: {len(df)}")
    
    timestamps = df.index.unique().sort_values()
    split_idx = int(len(timestamps) * train_ratio)
    
    cutoff_time = timestamps[split_idx]
    
    df_train = df[df.index < cutoff_time]
    df_test = df[df.index >= cutoff_time]
    
    output_dir = "data/processed"
    
    train_path = os.path.join(output_dir, f"training_data_{timeframe}.parquet")
    test_path = os.path.join(output_dir, f"test_data_{timeframe}.parquet")
    
    df_train.to_parquet(train_path)
    df_test.to_parquet(test_path)
    
    print(f"\n[SPLIT] Train: {len(df_train)} filas ({len(df_train.index.unique())} timestamps)")
    print(f"[SPLIT] Train tickers: {df_train['ticker'].nunique()}")
    print(f"[SPLIT] Guardado en: {train_path}\n")
    
    print(f"[SPLIT] Test:  {len(df_test)} filas ({len(df_test.index.unique())} timestamps)")
    print(f"[SPLIT] Test tickers: {df_test['ticker'].nunique()}")
    print(f"[SPLIT] Guardado en: {test_path}\n")
    
    print(f"[SPLIT] Cutoff: {cutoff_time}")
    print("="*50)

if __name__ == "__main__":
    """
    horizon: distancia al futuro que el modelo intenta adivinar
    """
    # df_final = process_pipeline(horizon=8, d=0.4, window=500, vpt_price_d_window = 168, vol_window=168, timeframe='1h') #8h
    # df_final = process_pipeline(horizon=24, d=0.4, window=50, vpt_price_d_window = 24, vol_window=48, timeframe='1h') #24h
    # df_final = process_pipeline(horizon=8, d=0.4, window=50, vpt_price_d_window = 24, vol_window=48, timeframe='1h') #8h
    # df_final = process_pipeline(horizon=48, d=0.4, window=50, vpt_price_d_window=24, vol_window=96, timeframe='1h') #48h
    df_final = process_pipeline(horizon=8,d=0.4, window=50,vpt_price_d_window=6,vol_window=12,timeframe='4h', look_ahead_test=False) # alpha_past_neutral (Spearman: -0.0498)
        
    split_train_test(full_data_path="data/processed/full_data_4h.parquet",train_ratio=0.77,timeframe="4h")

