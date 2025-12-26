"""
WEEX Historical Data Test - Prueba de datos históricos del mercado

Este script obtiene y analiza datos históricos de la API de WEEX:

1. Datos de velas (candlestick) en diferentes intervalos de tiempo
2. Últimas operaciones ejecutadas en el mercado
3. Tasa de financiamiento actual
4. Análisis básico de los datos históricos

Requisitos:
    - Variables de entorno: API_Key, secret_key, passphrase (opcional para endpoints públicos)
    - Paquetes: requests, python-dotenv (opcional)
    - Conexión a: https://api-contract.weex.com

Uso:
    python historical_data_test.py

Documentación API:
    https://www.weex.com/api-doc/ai/marketAPI
    
Nota:
    Los endpoints de mercado son públicos y no requieren autenticación,
    pero igualmente se recomienda incluir las credenciales.
"""

import sys
import os
# Agregar el directorio padre al path para importar módulos locales
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime
from api import _env
from api.market import (
    get_candles,
    get_trades,
    get_funding_rate,
    get_ticker_price,
    get_server_time
)


def format_timestamp(ts_ms):
    """Convierte timestamp en milisegundos a formato legible."""
    try:
        # Si el timestamp es mayor a 10^10, probablemente ya está en milisegundos
        if ts_ms > 1e10:
            return datetime.fromtimestamp(ts_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Si no, asumimos que está en segundos
            return datetime.fromtimestamp(ts_ms).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(ts_ms)


def analyze_candles(candles):
    """Analiza datos de velas y muestra estadísticas básicas."""
    if not candles or len(candles) == 0:
        print("[WARNING] No hay datos de velas para analizar")
        return
    
    print("\n" + "="*70)
    print("ANÁLISIS DE DATOS HISTÓRICOS")
    print("="*70)
    
    # Formato esperado: [timestamp, open, high, low, close, volume]
    print(f"\nTotal de velas: {len(candles)}")
    
    # Primera y última vela
    first = candles[0]
    last = candles[-1]
    
    print(f"\nPrimera vela:")
    print(f"  Tiempo: {format_timestamp(first[0])}")
    print(f"  Apertura: ${float(first[1]):,.2f}")
    print(f"  Máximo: ${float(first[2]):,.2f}")
    print(f"  Mínimo: ${float(first[3]):,.2f}")
    print(f"  Cierre: ${float(first[4]):,.2f}")
    print(f"  Volumen: {float(first[5]):,.4f}")
    
    print(f"\nÚltima vela:")
    print(f"  Tiempo: {format_timestamp(last[0])}")
    print(f"  Apertura: ${float(last[1]):,.2f}")
    print(f"  Máximo: ${float(last[2]):,.2f}")
    print(f"  Mínimo: ${float(last[3]):,.2f}")
    print(f"  Cierre: ${float(last[4]):,.2f}")
    print(f"  Volumen: {float(last[5]):,.4f}")
    
    # Calcular estadísticas
    closes = [float(c[4]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    volumes = [float(c[5]) for c in candles]
    
    avg_close = sum(closes) / len(closes)
    max_high = max(highs)
    min_low = min(lows)
    total_volume = sum(volumes)
    
    # Calcular cambio porcentual
    price_change = ((closes[-1] - closes[0]) / closes[0]) * 100
    
    print(f"\nEstadísticas del período:")
    print(f"  Precio promedio: ${avg_close:,.2f}")
    print(f"  Máximo del período: ${max_high:,.2f}")
    print(f"  Mínimo del período: ${min_low:,.2f}")
    print(f"  Volumen total: {total_volume:,.4f}")
    print(f"  Cambio de precio: {price_change:+.2f}%")
    
    # Volatilidad simple (desviación estándar)
    variance = sum((x - avg_close) ** 2 for x in closes) / len(closes)
    std_dev = variance ** 0.5
    volatility_pct = (std_dev / avg_close) * 100
    
    print(f"  Volatilidad: {volatility_pct:.2f}%")
    
    print("="*70 + "\n")


def analyze_trades(trades):
    """Analiza las últimas operaciones del mercado."""
    if not trades or len(trades) == 0:
        print("[WARNING] No hay operaciones para analizar")
        return
    
    print("\n" + "="*70)
    print("ANÁLISIS DE OPERACIONES RECIENTES")
    print("="*70)
    
    print(f"\nTotal de operaciones: {len(trades)}")
    
    # Mostrar las primeras 5 operaciones
    print("\nÚltimas 5 operaciones:")
    for i, trade in enumerate(trades[:5], 1):
        timestamp = format_timestamp(trade.get('ts', trade.get('timestamp', 0)))
        price = float(trade.get('price', 0))
        qty = float(trade.get('qty', trade.get('size', 0)))
        side = trade.get('side', 'unknown')
        
        print(f"  {i}. [{timestamp}] {side.upper():4} - Precio: ${price:,.2f}, Cantidad: {qty:.4f}")
    
    # Calcular estadísticas
    buy_trades = [t for t in trades if t.get('side') == 'buy']
    sell_trades = [t for t in trades if t.get('side') == 'sell']
    
    print(f"\nDistribución:")
    print(f"  Compras: {len(buy_trades)} ({len(buy_trades)/len(trades)*100:.1f}%)")
    print(f"  Ventas: {len(sell_trades)} ({len(sell_trades)/len(trades)*100:.1f}%)")
    
    # Volumen total
    total_volume = sum(float(t.get('qty', t.get('size', 0))) for t in trades)
    print(f"  Volumen total: {total_volume:,.4f}")
    
    print("="*70 + "\n")


def test_historical_data():
    """
    Función principal que ejecuta todas las pruebas de datos históricos.
    """
    print("\n" + "="*70)
    print("WEEX HISTORICAL DATA TEST")
    print("="*70 + "\n")
    
    # Obtener credenciales (opcionales para endpoints públicos)
    api_key = _env("API_Key", "")
    secret_key = _env("secret_key", "")
    passphrase = _env("passphrase", "")
    locale = _env("WEEX_LOCALE", "en-US") or "en-US"
    
    # Configuración
    symbol = "cmt_btcusdt"
    
    # 1. Verificar tiempo del servidor
    print("1. Verificando conexión con el servidor...")
    server_time = get_server_time(verbose=True)
    print()
    
    if not server_time or server_time.get("http_status") == 521:
        print("[ERROR] No se pudo conectar al servidor. Verifica tu IP en la whitelist.")
        return
    
    # 2. Obtener precio actual
    print("2. Consultando precio actual...")
    current_price = get_ticker_price(
        api_key, secret_key, passphrase, 
        symbol=symbol, 
        locale=locale, 
        verbose=True
    )
    print()
    
    # 3. Obtener datos históricos en diferentes intervalos
    print("3. Obteniendo datos históricos de velas...")
    
    # Velas de 1 hora (últimas 100)
    print("\n--- Datos de 1 hora ---")
    candles_1h = get_candles(
        api_key, secret_key, passphrase,
        symbol=symbol,
        granularity="1h",
        limit=100,
        locale=locale,
        verbose=True
    )
    
    if candles_1h:
        analyze_candles(candles_1h)
        time.sleep(1)  # Pequeña pausa entre solicitudes
    
    # Velas de 15 minutos (últimas 50)
    print("\n--- Datos de 15 minutos ---")
    candles_15m = get_candles(
        api_key, secret_key, passphrase,
        symbol=symbol,
        granularity="15m",
        limit=50,
        locale=locale,
        verbose=True
    )
    
    if candles_15m:
        analyze_candles(candles_15m)
        time.sleep(1)
    
    # Velas diarias (últimas 30)
    print("\n--- Datos diarios ---")
    candles_1d = get_candles(
        api_key, secret_key, passphrase,
        symbol=symbol,
        granularity="1d",
        limit=30,
        locale=locale,
        verbose=True
    )
    
    if candles_1d:
        analyze_candles(candles_1d)
        time.sleep(1)
    
    # 4. Obtener últimas operaciones
    print("4. Obteniendo últimas operaciones del mercado...")
    trades = get_trades(
        api_key, secret_key, passphrase,
        symbol=symbol,
        limit=100,
        locale=locale,
        verbose=True
    )
    
    if trades:
        analyze_trades(trades)
        time.sleep(1)
    
    # 5. Consultando tasa de financiamiento
    print("5. Consultando tasa de financiamiento...")
    funding_rate = get_funding_rate(
        api_key, secret_key, passphrase,
        symbol=symbol,
        locale=locale,
        verbose=True
    )
    print()
    
    # Resumen final
    print("="*70)
    print("RESUMEN DE LA PRUEBA")
    print("="*70)
    print(f"✓ Símbolo: {symbol}")
    print(f"✓ Precio actual: ${current_price:,.2f}" if current_price else "✗ No se pudo obtener el precio")
    print(f"✓ Velas 1H obtenidas: {len(candles_1h) if candles_1h else 0}")
    print(f"✓ Velas 15m obtenidas: {len(candles_15m) if candles_15m else 0}")
    print(f"✓ Velas 1D obtenidas: {len(candles_1d) if candles_1d else 0}")
    print(f"✓ Operaciones obtenidas: {len(trades) if trades else 0}")
    print(f"✓ Tasa de financiamiento: {'Obtenida' if funding_rate else 'No disponible'}")
    print("="*70 + "\n")
    
    print("[INFO] Prueba de datos históricos completada exitosamente!")


def main():
    """Punto de entrada principal."""
    try:
        test_historical_data()
    except KeyboardInterrupt:
        print("\n\n[INFO] Prueba interrumpida por el usuario.")
    except Exception as e:
        print(f"\n[ERROR] Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
