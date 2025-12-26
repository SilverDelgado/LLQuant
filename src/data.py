"""
WEEX Data Module - Obtención y procesamiento de datos del mercado

Este módulo proporciona funciones para obtener y procesar datos estructurados 
y no estructurados de la API de WEEX, incluyendo:

- Datos OHLCV (Open, High, Low, Close, Volume) de múltiples temporalidades
- Indicadores técnicos: RSI, MACD, Medias Móviles
- Métricas fundamentales del mercado
- Funding rates
- Datos de operaciones y liquidez

Funciones principales:
    - get_structured_data(): Obtiene OHLCV e indicadores técnicos
    - get_unstructured_data(): Obtiene funding rates y datos adicionales

Uso:
    from data import get_structured_data, get_unstructured_data
    
    structured = get_structured_data("cmt_btcusdt")
    unstructured = get_unstructured_data("cmt_btcusdt")
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from utils import _env
from utils.market import (
    get_candles,
    get_ticker_price,
    get_funding_rate,
    get_trades,
    get_contract_info,
)
from noticias import get_market_news


# ======================== MAPEO DE SÍMBOLOS A NOMBRES ========================

SYMBOL_TO_NAME = {
    "cmt_btcusdt": "Bitcoin",
    "cmt_ethusdt": "Ethereum",
    "cmt_solusdt": "Solana",
    "cmt_dogeusdt": "Dogecoin",
    "cmt_xrpusdt": "Ripple",
    "cmt_adausdt": "Cardano",
    "cmt_bnbusdt": "Binance Coin",
    "cmt_ltcusdt": "Litecoin"
}


# ======================== FUNCIONES DE ANÁLISIS SEMÁNTICO ========================

def calculate_slope(values: List[float], lookback: int = 3) -> str:
    """
    Calcula la pendiente de un indicador basándose en los últimos N valores.
    
    Args:
        values: Lista de valores del indicador
        lookback: Número de valores anteriores a comparar (por defecto 3)
    
    Returns:
        "Rising", "Falling" o "Flat" basado en la tendencia
    
    Ejemplo:
        >>> prices = [100, 102, 101, 103, 105]
        >>> slope = calculate_slope(prices)
        >>> print(slope)  # "Rising"
    """
    if len(values) < lookback + 1:
        return "Insufficient Data"
    
    recent_values = values[-lookback:]
    first = recent_values[0]
    last = recent_values[-1]
    
    # Calcular cambio porcentual
    change_pct = ((last - first) / first) * 100 if first != 0 else 0
    
    if change_pct > 1:  # Umbral de 1% para evitar ruido
        return "Rising"
    elif change_pct < -1:
        return "Falling"
    else:
        return "Flat"


def identify_trend(price: float, sma_50: Optional[float], sma_200: Optional[float]) -> str:
    """
    Identifica la tendencia general comparando el precio con las medias móviles.
    
    Args:
        price: Precio actual
        sma_50: Media móvil de 50 períodos
        sma_200: Media móvil de 200 períodos
    
    Returns:
        String descriptivo de la tendencia
    
    Ejemplo:
        >>> trend = identify_trend(65000, 62000, 60000)
        >>> print(trend)  # "Strong Uptrend"
    """
    # Si no hay SMA(50), no podemos determinar la tendencia
    if sma_50 is None:
        return "Insufficient Data"
    
    # Comparar precio con SMA(50) como mínimo
    above_sma50 = price > sma_50
    
    # Si tenemos SMA(200), hacer análisis más profundo
    if sma_200 is not None:
        above_sma200 = price > sma_200
        sma50_above_sma200 = sma_50 > sma_200
        
        if above_sma50 and above_sma200 and sma50_above_sma200:
            return "Strong Uptrend (Price > SMA50 > SMA200)"
        elif above_sma50 and above_sma200:
            return "Uptrend (Price > Both SMAs)"
        elif not above_sma50 and not above_sma200 and not sma50_above_sma200:
            return "Strong Downtrend (Price < SMA50 < SMA200)"
        elif not above_sma50 and not above_sma200:
            return "Downtrend (Price < Both SMAs)"
        elif above_sma50 and sma50_above_sma200:
            return "Bullish Structure (SMA50 > SMA200)"
        else:
            return "Mixed/Consolidation Phase"
    else:
        # Fallback: solo usar SMA(50) si SMA(200) no está disponible
        if above_sma50:
            return "Bullish (Price > SMA50)"
        else:
            return "Bearish (Price < SMA50)"


def get_support_resistance(candles: List[list], lookback: int = 50) -> Dict[str, float]:
    """
    Identifica niveles de soporte y resistencia basados en highs/lows recientes.
    
    Args:
        candles: Lista de velas [timestamp, open, high, low, close, volume]
        lookback: Número de velas anteriores a analizar
    
    Returns:
        Diccionario con "support" y "resistance"
    
    Ejemplo:
        >>> levels = get_support_resistance(candles)
        >>> print(f"Support: ${levels['support']:,.2f}")
    """
    if not candles or len(candles) < lookback:
        return {"support": None, "resistance": None}
    
    recent_candles = candles[-lookback:]
    
    highs = [float(c[2]) for c in recent_candles]
    lows = [float(c[3]) for c in recent_candles]
    
    resistance = max(highs)
    support = min(lows)
    
    return {
        "support": support,
        "resistance": resistance,
        "range": resistance - support
    }


# ======================== INDICADORES TÉCNICOS ========================

def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """
    Calcula la Media Móvil Simple (SMA).
    
    Args:
        prices: Lista de precios
        period: Período de la media móvil
    
    Returns:
        Valor de la SMA o None si no hay suficientes datos
    """
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    """
    Calcula la Media Móvil Exponencial (EMA).
    
    Args:
        prices: Lista de precios
        period: Período de la EMA
    
    Returns:
        Valor de la EMA o None si no hay suficientes datos
    """
    if len(prices) < period:
        return None
    
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period  # Primera EMA es una SMA
    
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Calcula el Índice de Fuerza Relativa (RSI).
    
    Args:
        prices: Lista de precios de cierre
        period: Período del RSI (por defecto 14)
    
    Returns:
        Valor del RSI (0-100) o None si no hay suficientes datos
    """
    if len(prices) < period + 1:
        return None
    
    # Calcular cambios de precio
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    # Separar ganancias y pérdidas
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices: List[float], 
                   fast_period: int = 12, 
                   slow_period: int = 26, 
                   signal_period: int = 9) -> Optional[Dict[str, float]]:
    """
    Calcula el MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Lista de precios de cierre
        fast_period: Período de la EMA rápida (por defecto 12)
        slow_period: Período de la EMA lenta (por defecto 26)
        signal_period: Período de la línea de señal (por defecto 9)
    
    Returns:
        Diccionario con MACD, señal e histograma o None si no hay suficientes datos
    """
    if len(prices) < slow_period:
        return None
    
    # Calcular EMAs
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    
    if ema_fast is None or ema_slow is None:
        return None
    
    # MACD = EMA rápida - EMA lenta
    macd_line = ema_fast - ema_slow
    
    # Para simplificar, usamos SMA como señal (idealmente sería EMA del MACD)
    # En una implementación completa, necesitaríamos calcular MACD histórico
    signal_line = macd_line  # Simplificación
    histogram = macd_line - signal_line
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }


def calculate_bollinger_bands(prices: List[float], 
                               period: int = 20, 
                               std_dev: int = 2) -> Optional[Dict[str, float]]:
    """
    Calcula las Bandas de Bollinger.
    
    Args:
        prices: Lista de precios de cierre
        period: Período de la media móvil (por defecto 20)
        std_dev: Número de desviaciones estándar (por defecto 2)
    
    Returns:
        Diccionario con banda superior, media e inferior o None
    """
    if len(prices) < period:
        return None
    
    sma = calculate_sma(prices, period)
    if sma is None:
        return None
    
    # Calcular desviación estándar
    variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
    std = variance ** 0.5
    
    return {
        "upper": sma + (std * std_dev),
        "middle": sma,
        "lower": sma - (std * std_dev)
    }


def calculate_bollinger_bands(prices: List[float], 
                               period: int = 20, 
                               std_dev: int = 2) -> Optional[Dict[str, float]]:
    """
    Calcula las Bandas de Bollinger.
    
    Args:
        prices: Lista de precios de cierre
        period: Período de la media móvil (por defecto 20)
        std_dev: Número de desviaciones estándar (por defecto 2)
    
    Returns:
        Diccionario con banda superior, media e inferior o None
    """
    if len(prices) < period:
        return None
    
    sma = calculate_sma(prices, period)
    if sma is None:
        return None
    
    # Calcular desviación estándar
    variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
    std = variance ** 0.5
    
    return {
        "upper": sma + (std * std_dev),
        "middle": sma,
        "lower": sma - (std * std_dev)
    }


def analyze_bollinger_position(price: float, bollinger: Dict[str, float]) -> Dict[str, Any]:
    """
    Analiza la posición del precio respecto a las Bandas de Bollinger.
    
    Args:
        price: Precio actual
        bollinger: Diccionario con upper, middle, lower
    
    Returns:
        Diccionario con análisis semántico
    """
    if not bollinger or price is None:
        return {"status": "No Data", "distance_pct": None}
    
    upper = bollinger.get("upper", 0)
    lower = bollinger.get("lower", 0)
    middle = bollinger.get("middle", 0)
    band_width = upper - lower
    
    if band_width == 0:
        return {"status": "Invalid", "distance_pct": None}
    
    # Distancia del precio a cada banda (en porcentaje)
    dist_to_upper_pct = ((upper - price) / band_width) * 100
    dist_to_lower_pct = ((price - lower) / band_width) * 100
    
    # Determinar posición
    if dist_to_upper_pct < 5:
        status = "Testing Upper Band - Possible Breakout"
    elif dist_to_lower_pct < 5:
        status = "Testing Lower Band - Possible Reversal"
    elif price > middle:
        status = "Upper Half - Bullish Bias"
    else:
        status = "Lower Half - Bearish Bias"
    
    # Ancho de banda (volatilidad implícita)
    band_width_pct = (band_width / middle) * 100 if middle > 0 else 0
    if band_width_pct < 3:
        volatility_implication = "Tight Bands - Low Volatility, Breakout Expected"
    elif band_width_pct > 10:
        volatility_implication = "Wide Bands - High Volatility"
    else:
        volatility_implication = "Normal Band Width"
    
    return {
        "status": status,
        "distance_to_upper_pct": round(dist_to_upper_pct, 2),
        "distance_to_lower_pct": round(dist_to_lower_pct, 2),
        "band_width_pct": round(band_width_pct, 2),
        "volatility_implication": volatility_implication
    }


def analyze_rsi_context(rsi: float, rsi_history: List[float] = None) -> Dict[str, str]:
    """
    Analiza el contexto del RSI: nivel, momentum y divergencia.
    
    Args:
        rsi: Valor actual del RSI
        rsi_history: Últimos valores del RSI para detectar momentum
    
    Returns:
        Diccionario con interpretaciones de texto
    """
    if rsi is None:
        return {"level": "No Data", "momentum": "N/A", "overall": "N/A"}
    
    # Analizar nivel
    if rsi > 70:
        level = "Overbought"
        level_context = "Potential correction or consolidation"
    elif rsi < 30:
        level = "Oversold"
        level_context = "Potential bounce or reversal"
    elif rsi > 60:
        level = "Strong"
        level_context = "Upside momentum present"
    elif rsi < 40:
        level = "Weak"
        level_context = "Downside momentum present"
    else:
        level = "Neutral"
        level_context = "No extreme reading"
    
    # Analizar momentum si hay histórico
    momentum = "N/A"
    if rsi_history and len(rsi_history) >= 2:
        recent_slope = calculate_slope(rsi_history, lookback=2)
        if recent_slope == "Rising":
            momentum = "Increasing - Strength Building"
        elif recent_slope == "Falling":
            momentum = "Decreasing - Strength Waning"
        else:
            momentum = "Stable"
    
    overall = f"{level} ({level_context}) - Momentum: {momentum}"
    
    return {
        "level": level,
        "context": level_context,
        "momentum": momentum,
        "overall": overall
    }


def calculate_proximity(price: float, support: float, resistance: float) -> Dict[str, str]:
    """
    Calcula la distancia porcentual del precio al soporte y resistencia.
    Ayuda al LLM a entender qué tan cerca está de los niveles clave.
    
    Args:
        price: Precio actual
        support: Nivel de soporte
        resistance: Nivel de resistencia
    
    Returns:
        Diccionario con distancia % formateada
    """
    if support is None or resistance is None or price is None:
        return {"to_support": "N/A", "to_resistance": "N/A", "zone": "N/A"}
    
    # Distancia en % desde el soporte y la resistencia
    range_width = resistance - support
    if range_width == 0:
        return {"to_support": "N/A", "to_resistance": "N/A", "zone": "Invalid"}
    
    dist_to_support_pct = ((price - support) / range_width) * 100
    dist_to_resistance_pct = ((resistance - price) / range_width) * 100
    
    # Determinar zona
    if dist_to_support_pct < 20:
        zone = "Near Support - Potential Bounce Zone"
    elif dist_to_resistance_pct < 20:
        zone = "Near Resistance - Potential Rejection Zone"
    elif dist_to_support_pct < 40:
        zone = "Mid-Lower Zone"
    elif dist_to_resistance_pct < 40:
        zone = "Mid-Upper Zone"
    else:
        zone = "Center of Range - Consolidation"
    
    return {
        "to_support": f"{dist_to_support_pct:.2f}%",
        "to_resistance": f"{dist_to_resistance_pct:.2f}%",
        "zone": zone
    }


def detect_candle_pattern(candles: List[list], lookback: int = 3) -> str:
    """
    Detecta patrones de velas (Doji, Hammer, Engulfing, etc.)
    Sin dependencias externas.
    
    Args:
        candles: Lista de velas [timestamp, open, high, low, close, volume]
        lookback: Número de velas previas a considerar
    
    Returns:
        String descriptivo del patrón detectado
    """
    if not candles or len(candles) < 2:
        return "Insufficient Data"
    
    # Obtener velas actuales y anterior
    current = candles[-1]
    prev = candles[-2]
    
    # Extraer datos
    curr_open = float(current[1])
    curr_high = float(current[2])
    curr_low = float(current[3])
    curr_close = float(current[4])
    
    prev_open = float(prev[1])
    prev_high = float(prev[2])
    prev_low = float(prev[3])
    prev_close = float(prev[4])
    
    # Calcular tamaños
    curr_body = abs(curr_close - curr_open)
    curr_range = curr_high - curr_low
    curr_upper_shadow = curr_high - max(curr_open, curr_close)
    curr_lower_shadow = min(curr_open, curr_close) - curr_low
    
    prev_body = abs(prev_close - prev_open)
    prev_range = prev_high - prev_low
    
    # Evitar división por cero
    if curr_range == 0:
        return "No Movement"
    
    body_ratio = curr_body / curr_range
    
    # 1. DOJI - Open ≈ Close, sombras balanceadas
    if body_ratio < 0.1 and curr_range > 0:
        if abs(curr_upper_shadow - curr_lower_shadow) < curr_range * 0.3:
            return "Doji (Indecision/Reversal Signal)"
        else:
            return "Long-Legged Doji (High Volatility)"
    
    # 2. HAMMER - Small body en top, long lower shadow (bullish reversal en downtrend)
    if body_ratio < 0.3 and curr_lower_shadow > 2 * curr_body and curr_upper_shadow < curr_body:
        if curr_close > curr_open:
            return "Hammer (Bullish Reversal Signal)"
        else:
            return "Inverted Hammer (Potential Reversal)"
    
    # 3. SHOOTING STAR - Small body en bottom, long upper shadow (bearish en uptrend)
    if body_ratio < 0.3 and curr_upper_shadow > 2 * curr_body and curr_lower_shadow < curr_body:
        if curr_close < curr_open:
            return "Shooting Star (Bearish Reversal Signal)"
        else:
            return "Inverse Shooting Star"
    
    # 4. MARUBOZU - Sin sombras (strong movement)
    if curr_upper_shadow < curr_range * 0.05 and curr_lower_shadow < curr_range * 0.05:
        if curr_close > curr_open:
            return "Bullish Marubozu (Strong Uptrend)"
        else:
            return "Bearish Marubozu (Strong Downtrend)"
    
    # 5. BULLISH ENGULFING - Current cierra arriba del prev open, abre debajo del prev close
    if (curr_close > prev_open and curr_open < prev_close and 
        curr_body > prev_body and curr_close > curr_open):
        return "Bullish Engulfing (Strong Reversal Signal)"
    
    # 6. BEARISH ENGULFING - Current cierra abajo del prev open, abre arriba del prev close
    if (curr_close < prev_open and curr_open > prev_close and 
        curr_body > prev_body and curr_close < curr_open):
        return "Bearish Engulfing (Strong Reversal Signal)"
    
    # 7. HARAMI - Small current body inside prev range (indecision)
    if (curr_open > prev_low and curr_open < prev_high and 
        curr_close > prev_low and curr_close < prev_high and
        curr_body < prev_body * 0.5):
        if curr_close > curr_open:
            return "Bullish Harami (Potential Reversal)"
        else:
            return "Bearish Harami (Potential Reversal)"
    
    # 8. SPINNING TOP - Small body, long shadows on both sides
    if body_ratio < 0.3 and curr_upper_shadow > curr_body and curr_lower_shadow > curr_body:
        return "Spinning Top (Indecision/Consolidation)"
    
    # 9. Determinar trend fuerte vs débil por tamaño del body
    if body_ratio > 0.7:
        if curr_close > curr_open:
            return "Strong Bullish Candle (Strong Uptrend Continuation)"
        else:
            return "Strong Bearish Candle (Strong Downtrend Continuation)"
    elif body_ratio < 0.2:
        return "Weak Candle (Consolidation/Indecision)"
    else:
        if curr_close > curr_open:
            return "Moderate Bullish Candle"
        else:
            return "Moderate Bearish Candle"


def analyze_timeframe_semantic(candles: List[list], timeframe: str) -> Dict[str, str]:
    """
    Analiza semánticamente una temporalidad específica.
    
    Args:
        candles: Lista de velas
        timeframe: Nombre de la temporalidad (ej: "1h", "1d")
    
    Returns:
        Diccionario con análisis semántico
    """
    if not candles or len(candles) < 3:
        return {"summary": "Insufficient Data"}
    
    closes = [float(c[4]) for c in candles]
    price_slope = calculate_slope(closes, lookback=3)
    metrics = calculate_metrics(candles)
    
    # Determinar contexto
    volatility = metrics.get("volatility_pct", 0)
    price_change = metrics.get("price_change_pct", 0)
    
    if volatility > 5:
        vol_context = "High volatility"
    elif volatility < 1:
        vol_context = "Low volatility"
    else:
        vol_context = "Normal volatility"
    
    direction = "↑ UP" if price_change > 0 else "↓ DOWN"
    
    summary = f"{timeframe}: {direction} {abs(price_change):.2f}% ({vol_context}, {price_slope})"
    
    return {
        "timeframe": timeframe,
        "summary": summary,
        "price_slope": price_slope,
        "volatility_context": vol_context,
        "price_change_pct": price_change
    }


# ======================== MÉTRICAS FUNDAMENTALES ========================

def calculate_metrics(candles: List[list]) -> Dict[str, float]:
    """
    Calcula métricas fundamentales de los datos de velas.
    
    Args:
        candles: Lista de velas [timestamp, open, high, low, close, volume]
    
    Returns:
        Diccionario con métricas calculadas
    """
    if not candles or len(candles) == 0:
        return {}
    
    closes = [float(c[4]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    volumes = [float(c[5]) for c in candles]
    
    # Precio actual y estadísticas básicas
    current_price = closes[-1]
    avg_price = sum(closes) / len(closes)
    max_price = max(highs)
    min_price = min(lows)
    
    # Volatilidad (desviación estándar)
    variance = sum((p - avg_price) ** 2 for p in closes) / len(closes)
    volatility = (variance ** 0.5) / avg_price * 100  # Porcentaje
    
    # Cambio de precio
    price_change = ((closes[-1] - closes[0]) / closes[0]) * 100
    
    # Volumen
    total_volume = sum(volumes)
    avg_volume = total_volume / len(volumes)
    
    # Rango de precio
    price_range = max_price - min_price
    price_range_pct = (price_range / avg_price) * 100
    
    return {
        "current_price": current_price,
        "avg_price": avg_price,
        "max_price": max_price,
        "min_price": min_price,
        "volatility_pct": volatility,
        "price_change_pct": price_change,
        "total_volume": total_volume,
        "avg_volume": avg_volume,
        "price_range": price_range,
        "price_range_pct": price_range_pct,
        "num_candles": len(candles)
    }


# ======================== FUNCIÓN PRINCIPAL: DATOS ESTRUCTURADOS ========================

def get_structured_data(
    symbol: str = "cmt_btcusdt",
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    passphrase: Optional[str] = None,
    locale: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Obtiene datos OHLCV de múltiples temporalidades y calcula indicadores técnicos.
    
    Esta función recopila:
    - Datos OHLCV (Open, High, Low, Close, Volume) de varias temporalidades
    - Indicadores técnicos: RSI, MACD, Medias Móviles (SMA, EMA)
    - Bandas de Bollinger
    - Métricas fundamentales: volatilidad, cambio de precio, volumen
    
    Args:
        symbol: Símbolo del activo (ej: "cmt_btcusdt")
        api_key: Clave de API (opcional, se obtiene de env si no se proporciona)
        secret_key: Clave secreta (opcional)
        passphrase: Contraseña (opcional)
        locale: Idioma (opcional, por defecto "en-US")
        verbose: Mostrar información detallada en consola
    
    Returns:
        Diccionario con estructura:
        {
            "symbol": str,
            "timestamp": int,
            "current_price": float,
            "timeframes": {
                "1m": {...},  # Datos de 1 minuto
                "5m": {...},  # Datos de 5 minutos
                "15m": {...}, # Datos de 15 minutos
                "1h": {...},  # Datos de 1 hora
                "4h": {...},  # Datos de 4 horas
                "1d": {...}   # Datos de 1 día
            },
            "indicators": {
                "rsi_14": float,
                "macd": {...},
                "sma_20": float,
                "sma_50": float,
                "sma_200": float,
                "ema_12": float,
                "ema_26": float,
                "bollinger": {...}
            },
            "metrics": {
                "volatility_pct": float,
                "price_change_pct": float,
                "volume": float,
                ...
            }
        }
    
    Ejemplo:
        >>> data = get_structured_data("cmt_btcusdt", verbose=True)
        >>> print(f"Precio: ${data['current_price']:,.2f}")
        >>> print(f"RSI: {data['indicators']['rsi_14']:.2f}")
        >>> print(f"Volatilidad: {data['metrics']['volatility_pct']:.2f}%")
    """
    # Obtener credenciales de variables de entorno si no se proporcionan
    if api_key is None:
        api_key = _env("API_Key", "")
    if secret_key is None:
        secret_key = _env("secret_key", "")
    if passphrase is None:
        passphrase = _env("passphrase", "")
    if locale is None:
        locale = _env("WEEX_LOCALE", "en-US") or "en-US"
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"OBTENCIÓN DE DATOS ESTRUCTURADOS: {symbol}")
        print(f"{'='*70}\n")
    
    # Estructura de resultado
    result = {
        "symbol": symbol,
        "timestamp": int(datetime.now().timestamp() * 1000),
        "current_price": None,
        "timeframes": {},
        "indicators": {},
        "metrics": {},
        "llm_context": {},  # Nuevo: contexto semántico para LLM
        "errors": []
    }
    
    try:
        # 1. Obtener precio actual
        if verbose:
            print("[1/3] Obteniendo precio actual...")
        
        current_price = get_ticker_price(
            api_key, secret_key, passphrase,
            symbol=symbol,
            locale=locale,
            verbose=verbose
        )
        
        result["current_price"] = current_price
        
        # 2. Obtener datos OHLCV de múltiples temporalidades
        if verbose:
            print("\n[2/3] Obteniendo datos OHLCV de múltiples temporalidades...")
        
        timeframes = {
            "1m": 300,   # 100 velas de 1 minuto
            "5m": 300,   # 100 velas de 5 minutos
            "15m": 300,  # 100 velas de 15 minutos
            "1h": 600,   # 300 velas de 1 hora (para calcular SMA(200))
            "4h": 600,   # 300 velas de 4 horas
            "1d": 600    # 300 velas de 1 día
        }
        
        # Recopilar datos de cada temporalidad
        for timeframe, limit in timeframes.items():
            if verbose:
                print(f"\n  - Temporalidad {timeframe}...")
            
            candles = get_candles(
                api_key, secret_key, passphrase,
                symbol=symbol,
                granularity=timeframe,
                limit=limit,
                locale=locale,
                verbose=verbose
            )
            
            if candles and len(candles) > 0:
                # Guardar datos crudos
                result["timeframes"][timeframe] = {
                    "candles": candles,
                    "count": len(candles),
                    "latest": {
                        "timestamp": candles[-1][0],
                        "open": float(candles[-1][1]),
                        "high": float(candles[-1][2]),
                        "low": float(candles[-1][3]),
                        "close": float(candles[-1][4]),
                        "volume": float(candles[-1][5])
                    }
                }
                
                # Calcular métricas específicas de esta temporalidad
                metrics = calculate_metrics(candles)
                result["timeframes"][timeframe]["metrics"] = metrics
                
                # NUEVO: Análisis semántico para cada temporalidad
                semantic_analysis = analyze_timeframe_semantic(candles, timeframe)
                result["timeframes"][timeframe]["llm_summary"] = semantic_analysis["summary"]
                result["timeframes"][timeframe]["semantic"] = semantic_analysis
                if verbose:
                    print(f"    → {semantic_analysis['summary']}")
            else:
                result["errors"].append(f"No se pudieron obtener datos para {timeframe}")
        
        # 3. Calcular indicadores técnicos (usando datos de 1h como referencia)
        if verbose:
            print("\n[3/3] Calculando indicadores técnicos y análisis semántico...")
        
        # Usar datos de 1 hora para indicadores principales
        if "1h" in result["timeframes"] and result["timeframes"]["1h"]["candles"]:
            candles_1h = result["timeframes"]["1h"]["candles"]
            closes = [float(c[4]) for c in candles_1h]
            
            # RSI
            rsi = calculate_rsi(closes, period=14)
            if rsi is not None:
                result["indicators"]["rsi_14"] = rsi
                if verbose:
                    print(f"  ✓ RSI(14): {rsi:.2f}")
            
            # MACD
            macd = calculate_macd(closes)
            if macd is not None:
                result["indicators"]["macd"] = macd
                if verbose:
                    print(f"  ✓ MACD: {macd['macd']:.2f}")
            
            # Medias Móviles Simples
            sma_20 = calculate_sma(closes, 20)
            sma_50 = calculate_sma(closes, 50)
            
            if sma_20 is not None:
                result["indicators"]["sma_20"] = sma_20
                if verbose:
                    print(f"  ✓ SMA(20): ${sma_20:,.2f}")
            
            if sma_50 is not None:
                result["indicators"]["sma_50"] = sma_50
                if verbose:
                    print(f"  ✓ SMA(50): ${sma_50:,.2f}")
            
            # Si hay suficientes datos, calcular SMA(200)
            sma_200 = None
            if len(closes) >= 200:
                sma_200 = calculate_sma(closes, 200)
            if sma_200 is not None:
                result["indicators"]["sma_200"] = sma_200
                if verbose:
                    print(f"  ✓ SMA(200): ${sma_200:,.2f}")
            else:
                if verbose and len(closes) < 200:
                    print(f"  ℹ SMA(200): {len(closes)} velas disponibles (se necesitan 200)")
            
            # Medias Móviles Exponenciales
            ema_12 = calculate_ema(closes, 12)
            ema_26 = calculate_ema(closes, 26)
            
            if ema_12 is not None:
                result["indicators"]["ema_12"] = ema_12
                if verbose:
                    print(f"  ✓ EMA(12): ${ema_12:,.2f}")
            
            if ema_26 is not None:
                result["indicators"]["ema_26"] = ema_26
                if verbose:
                    print(f"  ✓ EMA(26): ${ema_26:,.2f}")
            
            # Bandas de Bollinger
            bollinger = calculate_bollinger_bands(closes, period=20, std_dev=2)
            if bollinger is not None:
                result["indicators"]["bollinger"] = bollinger
                if verbose:
                    print(f"  ✓ Bollinger Bands:")
                    print(f"    - Superior: ${bollinger['upper']:,.2f}")
                    print(f"    - Media: ${bollinger['middle']:,.2f}")
                    print(f"    - Inferior: ${bollinger['lower']:,.2f}")
            
            # Métricas generales (usando datos de 1h)
            result["metrics"] = calculate_metrics(candles_1h)
            
            if verbose:
                print(f"\n  Métricas generales:")
                print(f"    - Volatilidad: {result['metrics']['volatility_pct']:.2f}%")
                print(f"    - Cambio de precio: {result['metrics']['price_change_pct']:+.2f}%")
                print(f"    - Volumen promedio: {result['metrics']['avg_volume']:,.2f}")
            
            # ============= ANÁLISIS SEMÁNTICO PARA LLM =============
            if verbose:
                print(f"\n  [ANÁLISIS SEMÁNTICO PARA LLM]")
            
            # 1. TENDENCIA GENERAL (Trend Structure)
            trend = identify_trend(result["current_price"], sma_50, sma_200)
            result["llm_context"]["trend_structure"] = trend
            if verbose:
                print(f"    ✓ Estructura de Tendencia: {trend}")
            
            # 2. ANÁLISIS RSI CON CONTEXTO
            rsi_history = closes[-5:] if len(closes) >= 5 else closes
            rsi_context = analyze_rsi_context(rsi, rsi_history)
            result["llm_context"]["rsi_analysis"] = rsi_context
            if verbose:
                print(f"    ✓ RSI Context: {rsi_context['overall']}")
            
            # 3. POSICIÓN EN BANDAS DE BOLLINGER
            bollinger_analysis = analyze_bollinger_position(result["current_price"], bollinger)
            result["llm_context"]["bollinger_status"] = bollinger_analysis["status"]
            result["llm_context"]["volatility_alert"] = bollinger_analysis["volatility_implication"]
            if verbose:
                print(f"    ✓ Bollinger Status: {bollinger_analysis['status']}")
                print(f"    ✓ Volatility Alert: {bollinger_analysis['volatility_implication']}")
            
            # 4. DETECCIÓN DE GOLDEN/DEATH CROSS
            if sma_50 is not None and sma_200 is not None:
                if sma_50 > sma_200:
                    cross_status = "Golden Cross Active (SMA50 > SMA200) - Bullish Signal"
                else:
                    cross_status = "Death Cross Active (SMA50 < SMA200) - Bearish Signal"
                result["llm_context"]["moving_average_cross"] = cross_status
                if verbose:
                    print(f"    ✓ MA Cross: {cross_status}")
            
            # 5. PENDIENTE DE PRECIO (Price Slope)
            price_slope = calculate_slope(closes, lookback=5)
            result["llm_context"]["price_momentum"] = price_slope
            if verbose:
                print(f"    ✓ Price Momentum (últimas 5 velas): {price_slope}")
            
            # 6. NIVELES DE SOPORTE Y RESISTENCIA
            support_resistance = get_support_resistance(candles_1h, lookback=50)
            key_levels = f"Support at ${support_resistance['support']:,.2f}, Resistance at ${support_resistance['resistance']:,.2f}"
            result["llm_context"]["key_levels"] = key_levels
            result["llm_context"]["support_resistance"] = {
                "support": support_resistance["support"],
                "resistance": support_resistance["resistance"],
                "range": support_resistance["range"]
            }
            if verbose:
                print(f"    ✓ Key Levels: {key_levels}")
            
            # 7. PROXIMIDAD A NIVELES (Nueva función - ayuda al LLM con cálculos)
            proximity = calculate_proximity(
                result["current_price"],
                support_resistance["support"],
                support_resistance["resistance"]
            )
            result["llm_context"]["proximity"] = proximity
            if verbose:
                print(f"    ✓ Proximity to Support: {proximity['to_support']}")
                print(f"    ✓ Proximity to Resistance: {proximity['to_resistance']}")
                print(f"    ✓ Zone: {proximity['zone']}")
            
            # 8. PATRÓN DE VELAS (Nueva función - detecta reversiones)
            candle_pattern = detect_candle_pattern(candles_1h, lookback=3)
            result["llm_context"]["last_candle_pattern"] = candle_pattern
            if verbose:
                print(f"    ✓ Last Candle Pattern: {candle_pattern}")
            
            # 9. RESUMEN GENERAL PARA LLM
            momentum_status = rsi_context["overall"]
            result["llm_context"]["momentum_status"] = momentum_status
            
            # Crear un resumen de una sola línea útil para LLM
            summary = f"{trend} | {momentum_status} | {bollinger_analysis['status']} | {candle_pattern}"
            result["llm_context"]["summary"] = summary
            if verbose:
                print(f"    ✓ Resumen LLM: {summary}")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"DATOS ESTRUCTURADOS OBTENIDOS EXITOSAMENTE")
            print(f"{'='*70}\n")
        
        return result
    
    except Exception as e:
        error_msg = f"Error al obtener datos estructurados: {e}"
        result["errors"].append(error_msg)
        if verbose:
            print(f"\n[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
        return result


# ======================== FUNCIÓN PRINCIPAL: DATOS NO ESTRUCTURADOS ========================

def get_unstructured_data(
    symbol: str = "cmt_btcusdt",
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    passphrase: Optional[str] = None,
    locale: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Obtiene datos no estructurados del mercado: funding rates, trades, etc.
    
    Esta función recopila:
    - Funding rate actual y predicho
    - Últimas operaciones del mercado
    - Información del contrato
    - Distribución de compras/ventas
    - Análisis de liquidez
    
    Args:
        symbol: Símbolo del activo (ej: "cmt_btcusdt")
        api_key: Clave de API (opcional, se obtiene de env si no se proporciona)
        secret_key: Clave secreta (opcional)
        passphrase: Contraseña (opcional)
        locale: Idioma (opcional, por defecto "en-US")
        verbose: Mostrar información detallada en consola
    
    Returns:
        Diccionario con estructura:
        {
            "symbol": str,
            "timestamp": int,
            "funding_rate": {
                "current_rate": float,
                "predicted_rate": float,
                "next_funding_time": int,
                ...
            },
            "trades": {
                "recent_trades": [...],
                "buy_sell_ratio": float,
                "total_volume": float,
                "avg_trade_size": float,
                ...
            },
            "contract_info": {
                "contract_val": str,
                "min_order_size": float,
                ...
            },
            "liquidity": {
                "buy_pressure": float,
                "sell_pressure": float,
                ...
            }
        }
    
    Ejemplo:
        >>> data = get_unstructured_data("cmt_btcusdt", verbose=True)
        >>> print(f"Funding Rate: {data['funding_rate']['current_rate']:.6f}%")
        >>> print(f"Buy/Sell Ratio: {data['trades']['buy_sell_ratio']:.2f}")
    """
    # Obtener credenciales de variables de entorno si no se proporcionan
    if api_key is None:
        api_key = _env("API_Key", "")
    if secret_key is None:
        secret_key = _env("secret_key", "")
    if passphrase is None:
        passphrase = _env("passphrase", "")
    if locale is None:
        locale = _env("WEEX_LOCALE", "en-US") or "en-US"
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"OBTENCIÓN DE DATOS NO ESTRUCTURADOS: {symbol}")
        print(f"{'='*70}\n")
    
    # Estructura de resultado
    result = {
        "symbol": symbol,
        "timestamp": int(datetime.now().timestamp() * 1000),
        "funding_rate": {},
        "trades": {},
        "contract_info": {},
        "liquidity": {},
        "news": [],
        "errors": []
    }
    
    try:
        # 1. Obtener Funding Rate
        if verbose:
            print("[1/3] Obteniendo funding rate...")
        
        funding_data = get_funding_rate(
            api_key, secret_key, passphrase,
            symbol=symbol,
            locale=locale,
            verbose=verbose
        )
        
        if funding_data:
            result["funding_rate"] = {
                "current_rate": float(funding_data.get("fundingRate", 0)),
                "predicted_rate": float(funding_data.get("fundingRatePredicted", 0)),
                "next_funding_time": funding_data.get("nextFundingRateTime"),
                "raw_data": funding_data
            }
            
            if verbose:
                print(f"  ✓ Funding Rate actual: {result['funding_rate']['current_rate']:.6f}%")
                print(f"  ✓ Funding Rate predicho: {result['funding_rate']['predicted_rate']:.6f}%")
        else:
            result["errors"].append("No se pudo obtener funding rate")
        
        # 2. Obtener últimas operaciones (trades)
        if verbose:
            print("\n[2/3] Obteniendo últimas operaciones del mercado...")
        
        trades = get_trades(
            api_key, secret_key, passphrase,
            symbol=symbol,
            limit=100,
            locale=locale,
            verbose=verbose
        )
        
        if trades and len(trades) > 0:
            # Análisis de operaciones
            buy_trades = [t for t in trades if t.get('side') == 'buy']
            sell_trades = [t for t in trades if t.get('side') == 'sell']
            
            total_volume = sum(float(t.get('qty', t.get('size', 0))) for t in trades)
            buy_volume = sum(float(t.get('qty', t.get('size', 0))) for t in buy_trades)
            sell_volume = sum(float(t.get('qty', t.get('size', 0))) for t in sell_trades)
            
            avg_trade_size = total_volume / len(trades) if trades else 0
            buy_sell_ratio = len(buy_trades) / len(sell_trades) if sell_trades else float('inf')
            
            result["trades"] = {
                "recent_trades": trades[:20],  # Guardar solo las 20 más recientes
                "total_trades": len(trades),
                "buy_count": len(buy_trades),
                "sell_count": len(sell_trades),
                "buy_sell_ratio": buy_sell_ratio,
                "total_volume": total_volume,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "avg_trade_size": avg_trade_size,
                "buy_percentage": (len(buy_trades) / len(trades)) * 100,
                "sell_percentage": (len(sell_trades) / len(trades)) * 100
            }
            
            if verbose:
                print(f"  ✓ Total de operaciones: {len(trades)}")
                print(f"  ✓ Compras: {len(buy_trades)} ({result['trades']['buy_percentage']:.1f}%)")
                print(f"  ✓ Ventas: {len(sell_trades)} ({result['trades']['sell_percentage']:.1f}%)")
                print(f"  ✓ Ratio Compra/Venta: {buy_sell_ratio:.2f}")
                print(f"  ✓ Volumen total: {total_volume:,.2f}")
        else:
            result["errors"].append("No se pudieron obtener trades")
        
        # 3. Obtener información del contrato
        if verbose:
            print("\n[3/3] Obteniendo información del contrato...")
        
        contract = get_contract_info(
            api_key, secret_key, passphrase,
            symbol=symbol,
            locale=locale,
            verbose=verbose
        )
        
        if contract:
            result["contract_info"] = {
                "contract_val": contract.get("contract_val"),
                "min_order_size": float(contract.get("minOrderSize", 0)),
                "size_increment": contract.get("size_increment"),
                "base_currency": contract.get("base_currency"),
                "quote_currency": contract.get("quote_currency"),
                "raw_data": contract
            }
            
            if verbose:
                print(f"  ✓ Valor del contrato: {result['contract_info']['contract_val']}")
                print(f"  ✓ Tamaño mínimo de orden: {result['contract_info']['min_order_size']}")
        else:
            result["errors"].append("No se pudo obtener información del contrato")
        
        # 4. Análisis de liquidez (basado en trades y volumen)
        if "trades" in result and result["trades"]:
            buy_vol = result["trades"].get("buy_volume", 0)
            sell_vol = result["trades"].get("sell_volume", 0)
            total_vol = result["trades"].get("total_volume", 1)
            
            result["liquidity"] = {
                "buy_pressure": (buy_vol / total_vol) * 100 if total_vol > 0 else 0,
                "sell_pressure": (sell_vol / total_vol) * 100 if total_vol > 0 else 0,
                "net_pressure": ((buy_vol - sell_vol) / total_vol) * 100 if total_vol > 0 else 0,
                "market_sentiment": "BULLISH" if buy_vol > sell_vol else "BEARISH"
            }
            
            if verbose:
                print(f"\n  Análisis de liquidez:")
                print(f"    - Presión de compra: {result['liquidity']['buy_pressure']:.2f}%")
                print(f"    - Presión de venta: {result['liquidity']['sell_pressure']:.2f}%")
                print(f"    - Sentimiento: {result['liquidity']['market_sentiment']}")
        
        # 5. Obtener noticias del mercado
        if verbose:
            print(f"\n[4/4] Obteniendo noticias del mercado...")
        
        try:
            # Usar el mapeo de símbolos para obtener el nombre correcto
            news_symbol = SYMBOL_TO_NAME.get(symbol, "Bitcoin")
            
            news = get_market_news(symbol=news_symbol, max_results=10)
            result["news"] = news
            
            if verbose:
                print(f"  ✓ Noticias de {news_symbol} obtenidas: {len(news)}")
                for idx, n in enumerate(news[:3], 1):
                    if 'error' not in n:
                        print(f"    {idx}. [{n.get('published_at', 'N/A')}] {n.get('title', 'Sin título')[:60]}...")
        except Exception as e:
            error_msg = f"Error al obtener noticias: {e}"
            result["errors"].append(error_msg)
            if verbose:
                print(f"  ✗ {error_msg}")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"DATOS NO ESTRUCTURADOS OBTENIDOS EXITOSAMENTE")
            print(f"{'='*70}\n")
        
        return result
    
    except Exception as e:
        error_msg = f"Error al obtener datos no estructurados: {e}"
        result["errors"].append(error_msg)
        if verbose:
            print(f"\n[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
        return result


# ======================== FUNCIÓN DE UTILIDAD ========================

def get_all_data(
    symbol: str = "cmt_btcusdt",
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    passphrase: Optional[str] = None,
    locale: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Obtiene todos los datos (estructurados y no estructurados) en una sola llamada.
    
    Args:
        symbol: Símbolo del activo
        api_key: Clave de API (opcional)
        secret_key: Clave secreta (opcional)
        passphrase: Contraseña (opcional)
        locale: Idioma (opcional)
        verbose: Mostrar información detallada
    
    Returns:
        Diccionario con "structured" y "unstructured" como claves principales
    
    Ejemplo:
        >>> all_data = get_all_data("cmt_btcusdt", verbose=True)
        >>> print(f"RSI: {all_data['structured']['indicators']['rsi_14']:.2f}")
        >>> print(f"Funding: {all_data['unstructured']['funding_rate']['current_rate']:.6f}%")
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"OBTENCIÓN COMPLETA DE DATOS: {symbol}")
        print(f"{'='*70}\n")
    
    structured = get_structured_data(
        symbol, api_key, secret_key, passphrase, locale, verbose
    )
    
    unstructured = get_unstructured_data(
        symbol, api_key, secret_key, passphrase, locale, verbose
    )
    
    return {
        "symbol": symbol,
        "timestamp": int(datetime.now().timestamp() * 1000),
        "structured": structured,
        "unstructured": unstructured
    }

def prepare_llm_payload(raw_data):
    """
    Toma el JSON gigante de data.py y devuelve solo lo que el LLM necesita ver.
    Optimizado para proporcionar contexto semántico y cálculos listos.
    """
    # 1. Extraemos el bloque 'llm_context' que ya creaste (es perfecto)
    context = raw_data.get("llm_context", {})
    
    # 2. Extraemos solo los resúmenes de texto de las temporalidades clave
    timeframes = raw_data.get("timeframes", {})
    tf_summaries = {}
    for tf in ["15m", "1h", "4h", "1d"]:
        if tf in timeframes:
            tf_summaries[tf] = timeframes[tf].get("llm_summary", "No data")

    # 3. Construimos el paquete final limpio y enriquecido
    payload = {
        "asset": raw_data.get("symbol"),
        "current_price": raw_data.get("current_price"),
        "market_structure": {
            "trend": context.get("trend_structure"),
            "support_resistance": context.get("key_levels"),
            "volatility_alert": context.get("volatility_alert")
        },
        "proximity_to_levels": context.get("proximity", {}),  # NUEVO
        "last_candle_pattern": context.get("last_candle_pattern"),  # NUEVO
        "indicators_analysis": {
            "rsi_status": context.get("rsi_analysis", {}).get("overall"),
            "bollinger_position": context.get("bollinger_status"),
            "momentum": context.get("momentum_status")
        },
        "multi_timeframe_context": tf_summaries,
        "summary_for_llm": context.get("summary")  # Resumen de una línea
    }
    
    return payload


def prepare_full_payload(structured, unstructured):
    """
    Combina datos técnicos y fundamentales en un solo prompt coherente.
    Limpia inconsistencias como 'Infinity' o presiones de 0%.
    
    Args:
        structured: Resultado de get_structured_data()
        unstructured: Resultado de get_unstructured_data()
    
    Returns:
        Diccionario completo optimizado para el LLM con análisis técnico y fundamental
    
    Ejemplo:
        >>> structured = get_structured_data("cmt_btcusdt")
        >>> unstructured = get_unstructured_data("cmt_btcusdt")
        >>> payload = prepare_full_payload(structured, unstructured)
        >>> print(json.dumps(payload, indent=2))
    """
    # 1. Base técnica (ya la tienes perfecta)
    technical_payload = prepare_llm_payload(structured)
    
    # 2. Procesamiento de datos no estructurados (limpieza)
    trades = unstructured.get("trades", {})
    liquidity = unstructured.get("liquidity", {})
    funding = unstructured.get("funding_rate", {})
    
    # Corregir casos de datos vacíos/inconsistentes
    buy_pressure = float(liquidity.get("buy_pressure", 0))
    sell_pressure = float(liquidity.get("sell_pressure", 0))
    
    # Si no hay presión real, el sentimiento es NEUTRAL, no Bearish
    market_sentiment = liquidity.get("market_sentiment", "NEUTRAL")
    if buy_pressure == 0 and sell_pressure == 0:
        market_sentiment = "NEUTRAL (Low Volume/No Recent Trades)"
    
    # Manejar buy_sell_ratio con infinity
    buy_sell_ratio = trades.get("buy_sell_ratio")
    if buy_sell_ratio == float('inf'):
        ratio_text = "High Buying Dominance (No Sell Orders)"
    elif buy_sell_ratio is None or buy_sell_ratio == 0:
        ratio_text = "No Data"
    else:
        ratio_text = f"{buy_sell_ratio:.2f}"
    
    # Formatear funding rate para que el LLM lo entienda
    funding_rate = funding.get("current_rate", 0)
    funding_predicted = funding.get("predicted_rate", 0)
    
    funding_context = "Neutral"
    if funding_rate > 0.01:
        funding_context = "High Positive (Longs paying Shorts - Bullish Overcrowding)"
    elif funding_rate < -0.01:
        funding_context = "Negative (Shorts paying Longs - Bearish Overcrowding)"
    elif abs(funding_rate) < 0.001:
        funding_context = "Near Zero (Balanced)"
    
    # 3. Construir el bloque fundamental
    fundamental_payload = {
        "funding_rate": {
            "current": f"{funding_rate:.6f}%",
            "predicted": f"{funding_predicted:.6f}%",
            "context": funding_context
        },
        "market_sentiment_onchain": market_sentiment,
        "order_flow": {
            "buy_pressure": f"{buy_pressure:.2f}%",
            "sell_pressure": f"{sell_pressure:.2f}%",
            "buy_sell_ratio": ratio_text,
            "total_trades": trades.get("total_trades", 0),
            "buy_percentage": f"{trades.get('buy_percentage', 0):.1f}%",
            "sell_percentage": f"{trades.get('sell_percentage', 0):.1f}%"
        },
        "contract_info": {
            "min_order_size": unstructured.get("contract_info", {}).get("min_order_size", "N/A")
        }
    }
    
    # 4. Procesar noticias del mercado con cálculo de tiempo desde publicación
    news = unstructured.get("news", [])
    news_payload = []
    now = datetime.now()
    
    for n in news:
        if 'error' not in n:
            published_at = n.get('published_at', 'Fecha desconocida')
            time_since = "N/A"
            
            # Calcular tiempo desde publicación
            try:
                # Parsear la fecha en formato: 'Tue, 16 Feb 2021 11:50:43 GMT'
                pub_date = datetime.strptime(published_at, '%a, %d %b %Y %H:%M:%S %Z') if published_at != 'Fecha desconocida' else None
                
                if pub_date:
                    # Calcular diferencia
                    delta = now - pub_date
                    
                    # Convertir a formato legible
                    if delta.total_seconds() < 60:
                        time_since = f"{int(delta.total_seconds())} segundos atrás"
                    elif delta.total_seconds() < 3600:
                        time_since = f"{int(delta.total_seconds() / 60)} minutos atrás"
                    elif delta.total_seconds() < 86400:
                        time_since = f"{int(delta.total_seconds() / 3600)} horas atrás"
                    else:
                        time_since = f"{int(delta.total_seconds() / 86400)} días atrás"
            except:
                time_since = "N/A"
            
            news_payload.append({
                "title": n.get('title', 'Sin título'),
                "published_at": published_at,
                "time_since_publication": time_since
            })
    
    # 5. Fusión final - Combinar análisis técnico, fundamental y noticias
    full_payload = {
        "timestamp": {
            "current_time": datetime.now().isoformat(),
            "unix_timestamp": int(datetime.now().timestamp())
        },
        **technical_payload,  # Expande todo lo técnico
        "fundamental_analysis": fundamental_payload,
        "market_news": news_payload,
        "trading_context": {
            "recommendation": "Use this data to make informed trading decisions. "
                            "Consider both technical signals, on-chain sentiment, and recent news.",
            "news_freshness_note": f"News timestamps are relative to system time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        }
    }
    
    return full_payload

# ======================== EJEMPLO DE USO ========================

if __name__ == "__main__":
    """
    Ejemplo de uso del módulo de datos.
    """
    print("\n" + "="*70)
    print("WEEX DATA MODULE - DEMO")
    print("="*70 + "\n")
    
    # Símbolo a consultar
    symbol = "cmt_btcusdt"
    
    # Opción 1: Obtener solo datos estructurados
    print("\n>>> OPCIÓN 1: Datos Estructurados <<<")
    structured = get_structured_data(symbol, verbose=True)
    
    # Opción 2: Obtener solo datos no estructurados
    print("\n>>> OPCIÓN 2: Datos No Estructurados <<<")
    unstructured = get_unstructured_data(symbol, verbose=True)
    
    # Opción 3: Obtener todo
    print("\n>>> OPCIÓN 3: Todos los Datos <<<")
    all_data = get_all_data(symbol, verbose=True)
    
    print("\n" + "="*70)
    print("DEMO COMPLETADO")
    print("="*70 + "\n")
