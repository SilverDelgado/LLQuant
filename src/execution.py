"""
Sistema de ejecución de órdenes y gestión de portfolio.
Maneja el rebalanceo automático basado en pesos objetivo.
"""
import sys
import os
import time
import random
from typing import Dict, Any, List

# Agregar el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import _env, ALLOWED_SYMBOLS
from api.account import get_account_assets
from api.trade import get_positions, place_order, close_position

# --- CONFIGURACIÓN ---
MIN_TRADE_AMOUNT = 6.0   # Mínimo en USDT para operar
REBALANCE_THRESHOLD = 0.02 # 2% de tolerancia (evita operar por cambios pequeños)
LEVERAGE_RATIO = 0.50    # Usar el 50% del Equity total


def get_credentials():
    """Obtiene las credenciales de la API desde variables de entorno."""
    api_key = _env("API_Key", "")
    secret_key = _env("secret_key", "")
    passphrase = _env("passphrase", "")
    locale = _env("WEEX_LOCALE", "en-US") or "en-US"
    return api_key, secret_key, passphrase, locale


def get_market_data(api_key, secret_key, passphrase, locale) -> Dict[str, Dict]:
    """
    Obtiene el estado actual de la cartera con SIGNOS.
    Long = Positivo, Short = Negativo.
    """
    market_state = {}
    
    for symbol in ALLOWED_SYMBOLS:
        market_state[symbol] = {'value': 0.0}
        pos_list = get_positions(api_key, secret_key, passphrase, symbol, locale, verbose=False)
        
        if pos_list and len(pos_list) > 0:
            pos = pos_list[0]
            # Valor absoluto de la posición
            abs_value = float(pos.get('open_value', 0)) 
            
            # DETERMINAR EL SIGNO
            # 'holdSide' o 'side' suele indicar 'long' o 'short' en la API
            side = pos.get('holdSide', pos.get('side', 'long')).lower()
            
            if side == 'short':
                market_state[symbol]['value'] = -abs_value
            else:
                market_state[symbol]['value'] = abs_value
            
    return market_state


def get_real_equity(api_key, secret_key, passphrase, locale) -> float:
    """
    Obtiene el Equity real de la cuenta directamente de la API.
    Equity = Balance Disponible + Margen Bloqueado + PnL No Realizado.
    """
    assets = get_account_assets(api_key, secret_key, passphrase, locale, verbose=False)
    
    if isinstance(assets, list):
        for coin in assets:
            if coin.get('coinName') == 'USDT':
                # ¡AQUÍ ESTÁ LA CLAVE! Usamos el campo 'equity' directamente.
                if 'equity' in coin:
                    return float(coin['equity'])
                else:
                    # Fallback por si acaso la API cambia
                    return float(coin.get('available', 0))
    
    # Si devuelve un dict o falla
    if isinstance(assets, dict):
         return float(assets.get('equity', assets.get('available', 0)))
         
    return 0.0


def generate_random_weights(symbols: List[str], mode: str = 'both') -> Dict[str, float]:
    """
    Genera pesos optimizados para control de riesgo (Suma Absoluta = 1).
    Esto asegura que la exposición total sea siempre Equity * LEVERAGE_RATIO.
    
    Args:
        symbols: Lista de símbolos a ponderar
        mode: 'longonly', 'shortonly', o 'both'
    """
    if mode == 'longonly':
        raw_weights = [random.random() for _ in range(len(symbols))]
    elif mode == 'shortonly':
        raw_weights = [-random.random() for _ in range(len(symbols))]
    elif mode == 'both':
        # Genera valores entre -1 y 1
        raw_weights = [random.uniform(-1, 1) for _ in range(len(symbols))]
    else:
        raise ValueError(f"Modo '{mode}' no válido.")

    # NORMALIZACIÓN CRÍTICA:
    # Dividimos cada peso por la suma de los valores absolutos.
    # Así, no importa si son todos negativos o mixtos, el total nocional es constante.
    total_abs = sum(abs(w) for w in raw_weights)
    
    if total_abs == 0: 
        return {s: 0.0 for s in symbols}  # Evitar división por cero
    
    weights = {sym: w / total_abs for sym, w in zip(symbols, raw_weights)}
    
    # Debug para que veas el sesgo de la cartera
    net_exposure = sum(weights.values())
    print(f" [INFO] Sesgo de Cartera (Net Exposure): {net_exposure:.2%}")  # -100% a 100%
    
    return weights


def rebalance_portfolio(api_key, secret_key, passphrase, locale, target_weights, mode: str = 'longonly', leverage: int = 1):
    """
    Rebalancea el portfolio según los pesos objetivo.
    
    Args:
        api_key, secret_key, passphrase, locale: Credenciales de API
        target_weights: Diccionario {symbol: peso} donde la suma de valores absolutos = 1
        mode: Modo de operación ('longonly', 'shortonly', 'both')
        leverage: Leverage fijo para todas las operaciones (por defecto 1)
    """
    # Usar el mismo leverage para todos los símbolos
    leverage_str = str(leverage)
    
    print("-" * 60)
    total_equity = get_real_equity(api_key, secret_key, passphrase, locale)
    target_exposure_total = total_equity * LEVERAGE_RATIO
    
    print(f"Equity Real: {total_equity:.2f} USDT | Objetivo Exposición: {target_exposure_total:.2f} USDT")

    current_positions = get_market_data(api_key, secret_key, passphrase, locale)
    
    for symbol in ALLOWED_SYMBOLS:
        target_value_usdt = target_exposure_total * target_weights.get(symbol, 0.0)
        current_val = current_positions[symbol]['value']  # Ya viene con signo (- para shorts)
        
        delta_usdt = target_value_usdt - current_val
        
        # Filtro de trade mínimo
        if abs(delta_usdt) < MIN_TRADE_AMOUNT:
            continue
            
        # Filtro de umbral (solo si ya hay algo abierto)
        if abs(current_val) > 0:
            pct_diff = abs(delta_usdt) / abs(target_value_usdt) if target_value_usdt != 0 else 1
            if pct_diff < REBALANCE_THRESHOLD:
                print(f"  [SKIP] {symbol}: Diferencia del {pct_diff*100:.1f}% insuficiente.")
                continue

        # LÓGICA DE EJECUCIÓN SIMPLE:
        # Si delta > 0: Necesito comprar (Long)
        # Si delta < 0: Necesito vender (Short)
        
        if delta_usdt > 0:
            action = "LONG (Aumentar Long / Reducir Short)"
            print(f" > {symbol}: Actual={current_val:.1f} -> Target={target_value_usdt:.1f} | Delta={delta_usdt:.2f} ({action}) | Leverage: {leverage_str}x")
            message = f"Rebalancing portfolio: Adjusting {symbol} position from {current_val:.2f} to {target_value_usdt:.2f} USDT. Delta: +{delta_usdt:.2f} USDT. Target weight: {target_weights.get(symbol, 0.0)*100:.1f}%. Leverage: {leverage_str}x."
            place_order(api_key, secret_key, passphrase, leverage_str, symbol, "long", abs(delta_usdt), "0", message, locale, verbose=False)
        else:
            action = "SHORT (Aumentar Short / Reducir Long)"
            print(f" > {symbol}: Actual={current_val:.1f} -> Target={target_value_usdt:.1f} | Delta={delta_usdt:.2f} ({action}) | Leverage: {leverage_str}x")
            message = f"Rebalancing portfolio: Adjusting {symbol} position from {current_val:.2f} to {target_value_usdt:.2f} USDT. Delta: {delta_usdt:.2f} USDT. Target weight: {target_weights.get(symbol, 0.0)*100:.1f}%. Leverage: {leverage_str}x."
            place_order(api_key, secret_key, passphrase, leverage_str, symbol, "short", abs(delta_usdt), "0", message, locale, verbose=False)


def close_all_positions(api_key, secret_key, passphrase, locale):
    """Cierra todas las posiciones abiertas."""
    print("\nCerrando todas las posiciones...")
    for symbol in ALLOWED_SYMBOLS:
        close_position(api_key, secret_key, passphrase, symbol, "long", locale, verbose=False)
        close_position(api_key, secret_key, passphrase, symbol, "short", locale, verbose=False)
    print("Hecho.")


def run_portfolio_manager(mode: str = 'longonly', sleep_interval: int = 10):
    """
    Ejecuta el gestor de portfolio en modo continuo.
    
    Modos disponibles:
    - longonly: Solo posiciones largas (pesos positivos)
    - shortonly: Solo posiciones cortas (pesos negativos)  
    - both: Posiciones mixtas (pesos positivos y negativos)
    
    Args:
        mode: Modo de operación
        sleep_interval: Segundos entre rebalanceos
    """
    api_key, secret_key, passphrase, locale = get_credentials()
    if not api_key:
        print("Faltan credenciales.")
        sys.exit(1)

    print(f"Iniciando Portfolio Manager v2 (Equity Based) - Modo: {mode.upper()}")
    print(f"Activos: {len(ALLOWED_SYMBOLS)}")
    
    try:
        while True:
            weights = generate_random_weights(ALLOWED_SYMBOLS, mode)
            rebalance_portfolio(api_key, secret_key, passphrase, locale, weights, mode)
            
            print(f"\nEsperando {sleep_interval} segundos...")
            time.sleep(sleep_interval)
            
    except KeyboardInterrupt:
        close_all_positions(api_key, secret_key, passphrase, locale)