"""
WEEX Data Module - Obtención y procesamiento de datos del mercado
    - get_structured_data(): Obtiene OHLCV e indicadores técnicos
    - get_unstructured_data(): Obtiene funding rates y datos adicionales
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from processing import process_pipeline
from api import _env
from api.market import (
    get_candles,
    get_ticker_price,
    get_funding_rate,
    get_trades,
    get_contract_info,
)
from noticias import get_market_news
from utils import (
    calculate_slope,
    identify_trend,
    get_support_resistance,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    analyze_bollinger_position,
    analyze_rsi_context,
    calculate_proximity,
    detect_candle_pattern,
    analyze_timeframe_semantic,
    calculate_metrics,
)


# ======================== MAPEO DE SÍMBOLOS A NOMBRES

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

# Cache simple en memoria para minimizar llamadas a la API
_CANDLES_CACHE: Dict[Tuple[str, str, int], List[list]] = {}

# ======================== FUNCIÓN PRINCIPAL: DATOS ESTRUCTURADOS

def get_structured_data(
    symbol: str = "cmt_btcusdt",
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    passphrase: Optional[str] = None,
    locale: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
        >>> data = get_structured_data("cmt_btcusdt", verbose=True)
        >>> print(f"Precio: ${data['current_price']:,.2f}")
        >>> print(f"RSI: {data['indicators']['rsi_14']:.2f}")
        >>> print(f"Volatilidad: {data['metrics']['volatility_pct']:.2f}%")
    """
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
    
    result = {
        "symbol": symbol,
        "timestamp": int(datetime.now().timestamp() * 1000),
        "current_price": None,
        "timeframes": {},
        "indicators": {},
        "metrics": {},
        "llm_context": {},
        "errors": []
    }
    
    try:
        if verbose:
            print("[1/3] Obteniendo precio actual...")
        
        current_price = get_ticker_price(
            api_key, secret_key, passphrase,
            symbol=symbol,
            locale=locale,
            verbose=verbose
        )
        
        result["current_price"] = current_price
        
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
                
                metrics = calculate_metrics(candles)
                result["timeframes"][timeframe]["metrics"] = metrics
                semantic_analysis = analyze_timeframe_semantic(candles, timeframe)
                result["timeframes"][timeframe]["llm_summary"] = semantic_analysis["summary"]
                result["timeframes"][timeframe]["semantic"] = semantic_analysis
                if verbose:
                    print(f"    → {semantic_analysis['summary']}")
            else:
                result["errors"].append(f"No se pudieron obtener datos para {timeframe}")
        
        #indicadores técnicos (usando datos de 1h como referencia)
        if verbose:
            print("\n[3/3] Calculando indicadores técnicos y análisis semántico...")
        
        # datos de 1 hora para indicadores principales
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

# ======================== NUEVO: DATAFRAME PARA PIPELINE ========================

def _candles_to_df(symbol: str, candles: List[list]) -> pd.DataFrame:
    """Convierte lista de velas a DataFrame con formato del pipeline.
    Espera velas como [timestamp, open, high, low, close, volume].
    """
    if not candles:
        return pd.DataFrame(columns=["timestamp", "ticker", "Open", "High", "Low", "Close", "Volume"]).set_index("timestamp")
    expected_cols = ["timestamp", "Open", "High", "Low", "Close", "Volume"]

    # Algunas respuestas de la API incluyen una columna extra (p.ej. turnover). Nos quedamos con las 6 primeras.
    if len(candles[0]) > len(expected_cols):
        candles = [row[: len(expected_cols)] for row in candles]
    elif len(candles[0]) < len(expected_cols):
        raise ValueError(f"Formato de vela inesperado: se esperaban {len(expected_cols)} columnas y llegaron {len(candles[0])}")

    df = pd.DataFrame(candles, columns=expected_cols)
    # Asegurar tipos numéricos
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Index temporal
    # Las APIs suelen dar timestamp en ms
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    except Exception:
        # Si no es ms, intentar segundos
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["ticker"] = symbol
    df = df.set_index("timestamp").sort_index()
    return df


def _get_candles_cached(
    api_key: str,
    secret_key: str,
    passphrase: str,
    symbol: str,
    timeframe: str,
    limit: int,
    locale: str,
    verbose: bool,
) -> List[list]:
    """Obtiene velas con cache para evitar llamadas repetidas a la API."""
    cache_key = (symbol, timeframe, limit)
    if cache_key in _CANDLES_CACHE:
        if verbose:
            print(f"[CACHE] Reutilizando velas {symbol} {timeframe} ({limit})")
        return _CANDLES_CACHE[cache_key]

    candles = get_candles(
        api_key, secret_key, passphrase,
        symbol=symbol,
        granularity=timeframe,
        limit=limit,
        locale=locale,
        verbose=verbose,
    )
    _CANDLES_CACHE[cache_key] = candles or []
    return _CANDLES_CACHE[cache_key]


def get_df_data(
    symbols: Optional[List[str]] = None,
    timeframe: str = "1h",
    horizon: int = 8,
    d: float = 0.4,
    window: int = 500,
    vpt_price_d_window: int = 168,
    vol_window: int = 168,
    limit: int = 600,
    verbose: bool = False,
    inference_mode: bool = False,
):
    """
    Construye el dataset del pipeline en memoria usando datos OHLCV de la API.
    Optimiza llamadas: una por símbolo y temporalidad, con cache en proceso.

    Args:
        symbols: Lista de símbolos (ej: ["cmt_btcusdt", "cmt_ethusdt"]). Si None, usa claves de SYMBOL_TO_NAME.
        timeframe: Temporalidad para las velas (ej: "1h", "4h").
        horizon, d, window, vpt_price_d_window, vol_window: Parámetros del pipeline.
        limit: Número de velas por símbolo a solicitar.
        verbose: Logs informativos.
        inference_mode: Si True, NO calcula TARGET_ALPHA (modo inferencia).

    Returns:
        DataFrame final del pipeline (features neutrales + TARGET_ALPHA si no es inference_mode).
    """
    if symbols is None or len(symbols) == 0:
        symbols = list(SYMBOL_TO_NAME.keys())

    # Credenciales
    api_key = _env("API_Key", "")
    secret_key = _env("secret_key", "")
    passphrase = _env("passphrase", "")
    locale = _env("WEEX_LOCALE", "en-US") or "en-US"

    if verbose:
        print(f"\n{'='*70}")
        print(f"GET DF DATA - timeframe={timeframe}, symbols={len(symbols)}")
        print(f"{'='*70}\n")

    # 1) Obtener y construir DF base por símbolo
    dfs = []
    for sym in symbols:
        if verbose:
            print(f"[FETCH] {sym} ({timeframe}) limit={limit}")
        candles = _get_candles_cached(api_key, secret_key, passphrase, sym, timeframe, limit, locale, verbose)
        df_sym = _candles_to_df(sym, candles)
        # Eliminar duplicados de timestamp
        df_sym = df_sym[~df_sym.index.duplicated(keep='first')]
        dfs.append(df_sym)

    if not dfs:
        raise ValueError("No se pudieron construir datos base (sin velas)")

    df_base = pd.concat(dfs).sort_index()

    # 2) Ejecutar pipeline en memoria (sin guardar)
    final_dataset = process_pipeline(
        horizon=horizon,
        d=d,
        window=window,
        vpt_price_d_window=vpt_price_d_window,
        vol_window=vol_window,
        timeframe=timeframe,
        df_base=df_base,
        save_file=False,
        inference_mode=inference_mode,
    )

    if verbose:
        print(f"[DONE] Dataset listo: {final_dataset.shape}")

    return final_dataset

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

    # Opción 4: Dataset del pipeline en memoria
    print("\n>>> OPCIÓN 4: Dataset del Pipeline (in-memory) <<<")

    df = get_df_data(symbols=["cmt_btcusdt", "cmt_ethusdt"], timeframe="1h", horizon=8, d=0.4, window=500, vpt_price_d_window=168, vol_window=168, limit=600, verbose=True)
    print("\nPreview del dataset (head 5):")
    print(df.head(5))
    
    print("\n" + "="*70)
    print("DEMO COMPLETADO")
    print("="*70 + "\n")
