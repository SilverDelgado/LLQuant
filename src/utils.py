from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

# ======================== TRANSFORMACIONES Y INDICADORES (PIPELINE) ========================

def fracdiff_fixed_window(series: pd.Series, d: float = 0.4, window: int = 500) -> pd.Series:
    """
    Fixed-Window Fractional Differenciación con ventana fija (sin look-ahead).
    """
    if len(series) < window:
        return pd.Series(np.nan, index=series.index)
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights[::-1])
    values = series.fillna(method='ffill').fillna(0).values
    diff_values = np.convolve(values, weights, mode='valid')
    new_index = series.index[window - 1:]
    return pd.Series(diff_values, index=new_index)


def rsi_price_d(price_d: pd.Series, period: int = 14) -> pd.Series:
    delta = price_d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def bollinger_features(price_d: pd.Series, period: int = 20, std: int = 2):
    mid = price_d.rolling(period).mean()
    std_dev = price_d.rolling(period).std()
    zscore = (price_d - mid) / (std_dev + 1e-9)
    width = (std_dev * std * 2) / (mid.abs() + 1e-9)
    return zscore, width


def atr_price_d(price_d: pd.Series, period: int = 14) -> pd.Series:
    high_low = price_d.rolling(2).max() - price_d.rolling(2).min()
    return high_low.rolling(period).mean()


def volume_imbalance(volume: pd.Series, price_d: pd.Series, period: int = 20) -> pd.Series:
    positive_vol = volume.where(price_d.diff() > 0, 0).rolling(period).sum()
    negative_vol = volume.where(price_d.diff() < 0, 0).rolling(period).sum()
    return (positive_vol - negative_vol) / (positive_vol + negative_vol + 1e-9)


def vpt_price_d(volume: pd.Series, price_d: pd.Series, window: int = 168) -> pd.Series:
    vpt_raw = (volume * price_d.diff() / (price_d + 1e-9))
    return vpt_raw.rolling(window).sum()


def stoch_price_d(price_d: pd.Series, k: int = 14, d: int = 3) -> pd.Series:
    low_min = price_d.rolling(k).min()
    high_max = price_d.rolling(k).max()
    k_line = 100 * (price_d - low_min) / (high_max - low_min + 1e-9)
    return k_line.rolling(d).mean()


# ======================== INDICADORES TÉCNICOS (DATA) ========================

def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    return ema


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Optional[Dict[str, float]]:
    if len(prices) < slow_period:
        return None
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    if ema_fast is None or ema_slow is None:
        return None
    macd_line = ema_fast - ema_slow
    signal_line = macd_line  # Simplificación
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: int = 2) -> Optional[Dict[str, float]]:
    if len(prices) < period:
        return None
    sma = calculate_sma(prices, period)
    if sma is None:
        return None
    variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
    std = variance ** 0.5
    return {"upper": sma + (std * std_dev), "middle": sma, "lower": sma - (std * std_dev)}


def analyze_bollinger_position(price: float, bollinger: Dict[str, float]) -> Dict[str, Any]:
    if not bollinger or price is None:
        return {"status": "No Data", "distance_pct": None}
    upper = bollinger.get("upper", 0)
    lower = bollinger.get("lower", 0)
    middle = bollinger.get("middle", 0)
    band_width = upper - lower
    if band_width == 0:
        return {"status": "Invalid", "distance_pct": None}
    dist_to_upper_pct = ((upper - price) / band_width) * 100
    dist_to_lower_pct = ((price - lower) / band_width) * 100
    if dist_to_upper_pct < 5:
        status = "Testing Upper Band - Possible Breakout"
    elif dist_to_lower_pct < 5:
        status = "Testing Lower Band - Possible Reversal"
    elif price > middle:
        status = "Upper Half - Bullish Bias"
    else:
        status = "Lower Half - Bearish Bias"
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
        "volatility_implication": volatility_implication,
    }


def calculate_slope(values: List[float], lookback: int = 3) -> str:
    if len(values) < lookback + 1:
        return "Insufficient Data"
    recent_values = values[-lookback:]
    first = recent_values[0]
    last = recent_values[-1]
    change_pct = ((last - first) / first) * 100 if first != 0 else 0
    if change_pct > 1:
        return "Rising"
    elif change_pct < -1:
        return "Falling"
    else:
        return "Flat"


def analyze_rsi_context(rsi: float, rsi_history: List[float] = None) -> Dict[str, str]:
    if rsi is None:
        return {"level": "No Data", "momentum": "N/A", "overall": "N/A"}
    if rsi > 70:
        level = "Overbought"; level_context = "Potential correction or consolidation"
    elif rsi < 30:
        level = "Oversold"; level_context = "Potential bounce or reversal"
    elif rsi > 60:
        level = "Strong"; level_context = "Upside momentum present"
    elif rsi < 40:
        level = "Weak"; level_context = "Downside momentum present"
    else:
        level = "Neutral"; level_context = "No extreme reading"
    momentum = "N/A"
    if rsi_history and len(rsi_history) >= 2:
        recent_slope = calculate_slope(rsi_history, lookback=2)
        momentum = (
            "Increasing - Strength Building" if recent_slope == "Rising"
            else "Decreasing - Strength Waning" if recent_slope == "Falling"
            else "Stable"
        )
    overall = f"{level} ({level_context}) - Momentum: {momentum}"
    return {"level": level, "context": level_context, "momentum": momentum, "overall": overall}


def identify_trend(price: float, sma_50: Optional[float], sma_200: Optional[float]) -> str:
    if sma_50 is None:
        return "Insufficient Data"
    above_sma50 = price > sma_50
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
        return "Bullish (Price > SMA50)" if above_sma50 else "Bearish (Price < SMA50)"


def get_support_resistance(candles: List[list], lookback: int = 50) -> Dict[str, float]:
    if not candles or len(candles) < lookback:
        return {"support": None, "resistance": None}
    recent_candles = candles[-lookback:]
    highs = [float(c[2]) for c in recent_candles]
    lows = [float(c[3]) for c in recent_candles]
    resistance = max(highs); support = min(lows)
    return {"support": support, "resistance": resistance, "range": resistance - support}


def calculate_proximity(price: float, support: float, resistance: float) -> Dict[str, str]:
    if support is None or resistance is None or price is None:
        return {"to_support": "N/A", "to_resistance": "N/A", "zone": "N/A"}
    range_width = resistance - support
    if range_width == 0:
        return {"to_support": "N/A", "to_resistance": "N/A", "zone": "Invalid"}
    dist_to_support_pct = ((price - support) / range_width) * 100
    dist_to_resistance_pct = ((resistance - price) / range_width) * 100
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
    return {"to_support": f"{dist_to_support_pct:.2f}%", "to_resistance": f"{dist_to_resistance_pct:.2f}%", "zone": zone}


def detect_candle_pattern(candles: List[list], lookback: int = 3) -> str:
    if not candles or len(candles) < 2:
        return "Insufficient Data"
    current = candles[-1]; prev = candles[-2]
    curr_open = float(current[1]); curr_high = float(current[2]); curr_low = float(current[3]); curr_close = float(current[4])
    prev_open = float(prev[1]); prev_high = float(prev[2]); prev_low = float(prev[3]); prev_close = float(prev[4])
    curr_body = abs(curr_close - curr_open); curr_range = curr_high - curr_low
    curr_upper_shadow = curr_high - max(curr_open, curr_close)
    curr_lower_shadow = min(curr_open, curr_close) - curr_low
    prev_body = abs(prev_close - prev_open); prev_range = prev_high - prev_low
    if curr_range == 0:
        return "No Movement"
    body_ratio = curr_body / curr_range
    if body_ratio < 0.1 and curr_range > 0:
        if abs(curr_upper_shadow - curr_lower_shadow) < curr_range * 0.3:
            return "Doji (Indecision/Reversal Signal)"
        else:
            return "Long-Legged Doji (High Volatility)"
    if body_ratio < 0.3 and curr_lower_shadow > 2 * curr_body and curr_upper_shadow < curr_body:
        return "Hammer (Bullish Reversal Signal)" if curr_close > curr_open else "Inverted Hammer (Potential Reversal)"
    if body_ratio < 0.3 and curr_upper_shadow > 2 * curr_body and curr_lower_shadow < curr_body:
        return "Shooting Star (Bearish Reversal Signal)" if curr_close < curr_open else "Inverse Shooting Star"
    if curr_upper_shadow < curr_range * 0.05 and curr_lower_shadow < curr_range * 0.05:
        return "Bullish Marubozu (Strong Uptrend)" if curr_close > curr_open else "Bearish Marubozu (Strong Downtrend)"
    if (curr_close > prev_open and curr_open < prev_close and curr_body > prev_body and curr_close > curr_open):
        return "Bullish Engulfing (Strong Reversal Signal)"
    if (curr_close < prev_open and curr_open > prev_close and curr_body > prev_body and curr_close < curr_open):
        return "Bearish Engulfing (Strong Reversal Signal)"
    if (curr_open > prev_low and curr_open < prev_high and curr_close > prev_low and curr_close < prev_high and curr_body < prev_body * 0.5):
        return "Bullish Harami (Potential Reversal)" if curr_close > curr_open else "Bearish Harami (Potential Reversal)"
    if body_ratio < 0.3 and curr_upper_shadow > curr_body and curr_lower_shadow > curr_body:
        return "Spinning Top (Indecision/Consolidation)"
    if body_ratio > 0.7:
        return "Strong Bullish Candle (Strong Uptrend Continuation)" if curr_close > curr_open else "Strong Bearish Candle (Strong Downtrend Continuation)"
    elif body_ratio < 0.2:
        return "Weak Candle (Consolidation/Indecision)"
    else:
        return "Moderate Bullish Candle" if curr_close > curr_open else "Moderate Bearish Candle"


def analyze_timeframe_semantic(candles: List[list], timeframe: str) -> Dict[str, str]:
    if not candles or len(candles) < 3:
        return {"summary": "Insufficient Data"}
    closes = [float(c[4]) for c in candles]
    price_slope = calculate_slope(closes, lookback=3)
    metrics = calculate_metrics(candles)
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
    return {"timeframe": timeframe, "summary": summary, "price_slope": price_slope, "volatility_context": vol_context, "price_change_pct": price_change}


def calculate_metrics(candles: List[list]) -> Dict[str, float]:
    if not candles or len(candles) == 0:
        return {}
    closes = [float(c[4]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    volumes = [float(c[5]) for c in candles]
    current_price = closes[-1]
    avg_price = sum(closes) / len(closes)
    max_price = max(highs)
    min_price = min(lows)
    variance = sum((p - avg_price) ** 2 for p in closes) / len(closes)
    volatility = (variance ** 0.5) / avg_price * 100
    price_change = ((closes[-1] - closes[0]) / closes[0]) * 100
    total_volume = sum(volumes)
    avg_volume = total_volume / len(volumes)
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
        "num_candles": len(candles),
    }
