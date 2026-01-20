import sys
import os
import time
import random
from typing import Dict, Any, List
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import _env, ALLOWED_SYMBOLS
from api.account import get_account_assets
from api.trade import get_positions, place_order, close_position
MIN_TRADE_AMOUNT = 6.0   
REBALANCE_THRESHOLD = 0.02 
LEVERAGE_RATIO = 0.50    
def get_credentials():
    api_key = _env("API_Key", "")
    secret_key = _env("secret_key", "")
    passphrase = _env("passphrase", "")
    locale = _env("WEEX_LOCALE", "en-US") or "en-US"
    return api_key, secret_key, passphrase, locale
def get_market_data(api_key, secret_key, passphrase, locale) -> Dict[str, Dict]:
    market_state = {}
    for symbol in ALLOWED_SYMBOLS:
        market_state[symbol] = {'value': 0.0}
        pos_list = get_positions(api_key, secret_key, passphrase, symbol, locale, verbose=False)
        if pos_list and len(pos_list) > 0:
            pos = pos_list[0]
            abs_value = float(pos.get('open_value', 0)) 
            side = pos.get('holdSide', pos.get('side', 'long')).lower()
            if side == 'short':
                market_state[symbol]['value'] = -abs_value
            else:
                market_state[symbol]['value'] = abs_value
    return market_state
def get_real_equity(api_key, secret_key, passphrase, locale) -> float:
    assets = get_account_assets(api_key, secret_key, passphrase, locale, verbose=False)
    if isinstance(assets, list):
        for coin in assets:
            if coin.get('coinName') == 'USDT':
                if 'equity' in coin:
                    return float(coin['equity'])
                else:
                    return float(coin.get('available', 0))
    if isinstance(assets, dict):
         return float(assets.get('equity', assets.get('available', 0)))
    return 0.0
def generate_random_weights(symbols: List[str], mode: str = 'both') -> Dict[str, float]:
    if mode == 'longonly':
        raw_weights = [random.random() for _ in range(len(symbols))]
    elif mode == 'shortonly':
        raw_weights = [-random.random() for _ in range(len(symbols))]
    elif mode == 'both':
        raw_weights = [random.uniform(-1, 1) for _ in range(len(symbols))]
    else:
        raise ValueError(f"Modo '{mode}' no válido.")
    total_abs = sum(abs(w) for w in raw_weights)
    if total_abs == 0: 
        return {s: 0.0 for s in symbols}  
    weights = {sym: w / total_abs for sym, w in zip(symbols, raw_weights)}
    net_exposure = sum(weights.values())
    print(f" [INFO] Sesgo de Cartera (Net Exposure): {net_exposure:.2%}")  
    return weights
def rebalance_portfolio(api_key, secret_key, passphrase, locale, target_weights, mode: str = 'longonly', leverage: int = 1, aggregate_threshold: float = None):
    leverage_str = str(leverage)
    print("-" * 60)
    total_equity = get_real_equity(api_key, secret_key, passphrase, locale)
    target_exposure_total = total_equity * LEVERAGE_RATIO
    threshold = AGGREGATE_TURNOVER_THRESHOLD if aggregate_threshold is None else aggregate_threshold
    print(f"Equity Real: {total_equity:.2f} USDT | Objetivo Exposición: {target_exposure_total:.2f} USDT")
    current_positions = get_market_data(api_key, secret_key, passphrase, locale)
    deltas = {}
    for symbol in ALLOWED_SYMBOLS:
        target_value_usdt = target_exposure_total * target_weights.get(symbol, 0.0)
        current_val = current_positions[symbol]['value']
        delta_usdt = target_value_usdt - current_val
        if abs(delta_usdt) >= MIN_TRADE_AMOUNT:
            deltas[symbol] = delta_usdt
    total_turnover = sum(abs(v) for v in deltas.values())
    turnover_pct = (total_turnover / total_equity) if total_equity > 0 else 1.0
    if turnover_pct < threshold:
        print(f"[SKIP] Turnover total {turnover_pct*100:.2f}% < {threshold*100:.1f}% -> no se rebalancea")
        return False
    for symbol in ALLOWED_SYMBOLS:
        target_value_usdt = target_exposure_total * target_weights.get(symbol, 0.0)
        current_val = current_positions[symbol]['value']
        delta_usdt = target_value_usdt - current_val
        if abs(delta_usdt) < MIN_TRADE_AMOUNT:
            continue
        if abs(current_val) > 0:
            pct_diff = abs(delta_usdt) / abs(target_value_usdt) if target_value_usdt != 0 else 1
            if pct_diff < REBALANCE_THRESHOLD:
                print(f"  [SKIP] {symbol}: Diferencia del {pct_diff*100:.1f}% insuficiente.")
                continue
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
    return True
def close_all_positions(api_key, secret_key, passphrase, locale):
    print("\nCerrando todas las posiciones...")
    for symbol in ALLOWED_SYMBOLS:
        close_position(api_key, secret_key, passphrase, symbol, "long", locale, verbose=False)
        close_position(api_key, secret_key, passphrase, symbol, "short", locale, verbose=False)
    print("Hecho.")
def run_portfolio_manager(mode: str = 'longonly', sleep_interval: int = 10):
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