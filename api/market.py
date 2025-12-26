"""
WEEX Market API - Funciones para obtener información del mercado
"""

import json
from typing import Optional, Dict, Any
from . import send_get, ALLOWED_SYMBOLS


def _validate_symbol(symbol: str, verbose: bool = True) -> bool:
    """
    Valida que el símbolo esté en la lista de permitidos.
    
    Args:
        symbol: Símbolo a validar
        verbose: Mostrar mensaje de error
    
    Returns:
        True si es válido, False si no
    """
    if symbol not in ALLOWED_SYMBOLS:
        if verbose:
            print(f"[ERROR] Símbolo '{symbol}' no está permitido. Símbolos permitidos: {ALLOWED_SYMBOLS}")
        return False
    return True


def get_ticker_price(
    api_key: str,
    secret_key: str,
    passphrase: str,
    symbol: str = "cmt_btcusdt",
    locale: str = "en-US",
    verbose: bool = True,
) -> Optional[float]:
    """
    Obtiene el precio actual de un símbolo.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        symbol: Par de trading (ej: cmt_btcusdt)
        locale: Idioma
        verbose: Mostrar información en consola
    
    Returns:
        Precio actual como float, o None si hay error
    
    Ejemplo:
        >>> price = get_ticker_price(api_key, secret_key, passphrase, "cmt_btcusdt")
        >>> print(f"BTC: ${price:,.2f}")
    """
    if not _validate_symbol(symbol, verbose):
        return None
        
    if verbose:
        print(f"[INFO] Consultando precio de {symbol}...")
    
    try:
        resp = send_get(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            request_path="/capi/v2/market/ticker",
            params={"symbol": symbol},
            auth=False,  # Endpoint público
            locale=locale,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            
            if isinstance(data, dict) and "last" in data:
                price = float(data["last"])
                if verbose:
                    print(f"[INFO] Precio actual de {symbol}: ${price:,.2f}")
                return price
            
            elif isinstance(data, list) and len(data) > 0:
                price = float(data[0].get("last", 0))
                if verbose:
                    print(f"[INFO] Precio actual de {symbol}: ${price:,.2f}")
                return price
        
        if verbose:
            print(f"[WARNING] Status {resp.status_code}: {resp.text[:200]}")
        return None
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al consultar precio: {e}")
        return None


def get_contract_info(
    api_key: str,
    secret_key: str,
    passphrase: str,
    symbol: str = "cmt_btcusdt",
    locale: str = "en-US",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Obtiene información del contrato incluyendo tamaño mínimo y precisión.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        symbol: Par de trading
        locale: Idioma
        verbose: Mostrar información en consola
    
    Returns:
        Diccionario con información del contrato, o None si hay error
    
    Información retornada:
        - contract_val: Valor del contrato
        - minOrderSize: Tamaño mínimo de orden
        - size_increment: Número de decimales permitidos
    
    Ejemplo:
        >>> info = get_contract_info(api_key, secret_key, passphrase, "cmt_btcusdt")
        >>> print(f"Tamaño mínimo: {info['minOrderSize']}")
    """
    if not _validate_symbol(symbol, verbose):
        return None
        
    if verbose:
        print(f"[INFO] Consultando información del contrato {symbol}...")
    
    try:
        resp = send_get(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            request_path="/capi/v2/market/contracts",
            params={"symbol": symbol},
            auth=False,  # Endpoint público
            locale=locale,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            
            if isinstance(data, list) and len(data) > 0:
                contract = data[0]
                if verbose:
                    print(f"[INFO] Valor del contrato: {contract.get('contract_val', 'N/A')}")
                    print(f"[INFO] Tamaño mínimo: {contract.get('minOrderSize', 'N/A')}")
                    print(f"[INFO] Incremento de tamaño: {contract.get('size_increment', 'N/A')}")
                return contract
            
            elif isinstance(data, dict):
                if verbose:
                    print(f"[INFO] Valor del contrato: {data.get('contract_val', 'N/A')}")
                    print(f"[INFO] Tamaño mínimo: {data.get('minOrderSize', 'N/A')}")
                return data
        
        if verbose:
            print(f"[WARNING] Status {resp.status_code}: {resp.text[:200]}")
        return None
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al consultar contrato: {e}")
        return None


def get_server_time(verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Obtiene la hora del servidor (endpoint público sin autenticación).
    
    Args:
        verbose: Mostrar información en consola
    
    Returns:
        Diccionario con información del servidor
    
    Ejemplo:
        >>> time_info = get_server_time()
        >>> print(time_info)
    """
    from . import BASE_URL
    import requests
    
    path = "/capi/v2/market/time"
    try:
        resp = requests.get(BASE_URL + path, timeout=10)
        if resp.ok:
            data = resp.json()
            if verbose:
                print(f"[INFO] Hora del servidor: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return data
        else:
            return {"http_status": resp.status_code, "text": resp.text}
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al obtener hora del servidor: {e}")
        return {"error": str(e)}


def get_candles(
    api_key: str,
    secret_key: str,
    passphrase: str,
    symbol: str = "cmt_btcusdt",
    granularity: str = "15m",
    limit: int = 100,
    locale: str = "en-US",
    verbose: bool = True,
) -> Optional[list]:
    """
    Obtiene datos históricos de velas (candlestick) para un símbolo.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        symbol: Par de trading (ej: cmt_btcusdt)
        granularity: Intervalo de tiempo (1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w)
        limit: Número de velas a obtener (max 300)
        locale: Idioma
        verbose: Mostrar información en consola
    
    Returns:
        Lista de velas con formato [timestamp, open, high, low, close, volume]
        o None si hay error
    
    Ejemplo:
        >>> candles = get_candles(api_key, secret_key, passphrase, "cmt_btcusdt", "1h", 100)
        >>> for candle in candles[-5:]:
        >>>     print(f"Time: {candle[0]}, Close: {candle[4]}")
    """
    if not _validate_symbol(symbol, verbose):
        return None
        
    if verbose:
        print(f"[INFO] Consultando datos históricos de {symbol} (intervalo: {granularity}, límite: {limit})...")
    
    try:
        params = {
            "symbol": symbol,
            "granularity": granularity,
            "limit": str(limit)
        }
        
        resp = send_get(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            request_path="/capi/v2/market/candles",
            params=params,
            auth=False,  # Endpoint público
            locale=locale,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            
            if isinstance(data, list) and len(data) > 0:
                if verbose:
                    print(f"[INFO] Se obtuvieron {len(data)} velas históricas")
                    print(f"[INFO] Primera vela: {data[0]}")
                    print(f"[INFO] Última vela: {data[-1]}")
                return data
            else:
                if verbose:
                    print(f"[WARNING] No se encontraron datos históricos")
                return []
        
        if verbose:
            print(f"[WARNING] Status {resp.status_code}: {resp.text[:200]}")
        return None
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al consultar datos históricos: {e}")
        return None


def get_trades(
    api_key: str,
    secret_key: str,
    passphrase: str,
    symbol: str = "cmt_btcusdt",
    limit: int = 100,
    locale: str = "en-US",
    verbose: bool = True,
) -> Optional[list]:
    """
    Obtiene las últimas operaciones (trades) ejecutadas en el mercado.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        symbol: Par de trading (ej: cmt_btcusdt)
        limit: Número de trades a obtener
        locale: Idioma
        verbose: Mostrar información en consola
    
    Returns:
        Lista de trades con información de precio, cantidad y tiempo
        o None si hay error
    
    Ejemplo:
        >>> trades = get_trades(api_key, secret_key, passphrase, "cmt_btcusdt", 50)
        >>> for trade in trades[:5]:
        >>>     print(f"Precio: {trade['price']}, Cantidad: {trade['qty']}")
    """
    if not _validate_symbol(symbol, verbose):
        return None
        
    if verbose:
        print(f"[INFO] Consultando últimas operaciones de {symbol} (límite: {limit})...")
    
    try:
        params = {
            "symbol": symbol,
            "limit": str(limit)
        }
        
        resp = send_get(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            request_path="/capi/v2/market/trades",
            params=params,
            auth=False,  # Endpoint público
            locale=locale,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            
            if isinstance(data, list) and len(data) > 0:
                if verbose:
                    print(f"[INFO] Se obtuvieron {len(data)} operaciones")
                    print(f"[INFO] Última operación: {data[0]}")
                return data
            else:
                if verbose:
                    print(f"[WARNING] No se encontraron operaciones")
                return []
        
        if verbose:
            print(f"[WARNING] Status {resp.status_code}: {resp.text[:200]}")
        return None
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al consultar operaciones: {e}")
        return None


def get_funding_rate(
    api_key: str,
    secret_key: str,
    passphrase: str,
    symbol: str = "cmt_btcusdt",
    locale: str = "en-US",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Obtiene la tasa de financiamiento actual para un contrato perpetuo.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        symbol: Par de trading (ej: cmt_btcusdt)
        locale: Idioma
        verbose: Mostrar información en consola
    
    Returns:
        Diccionario con información de la tasa de financiamiento
        o None si hay error
    
    Ejemplo:
        >>> funding = get_funding_rate(api_key, secret_key, passphrase, "cmt_btcusdt")
        >>> print(f"Tasa actual: {funding['fundingRate']}")
    """
    if not _validate_symbol(symbol, verbose):
        return None
        
    if verbose:
        print(f"[INFO] Consultando tasa de financiamiento de {symbol}...")
    
    try:
        params = {"symbol": symbol}
        
        resp = send_get(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            request_path="/capi/v2/market/currentFundRate",
            params=params,
            auth=False,  # Endpoint público
            locale=locale,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            
            if isinstance(data, list) and len(data) > 0:
                funding_data = data[0]  # Tomar el primer elemento del array
                if verbose:
                    print(f"[INFO] Tasa de financiamiento: {json.dumps(funding_data, indent=2, ensure_ascii=False)}")
                return funding_data
            elif isinstance(data, dict):
                if verbose:
                    print(f"[INFO] Tasa de financiamiento: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return data
        
        if verbose:
            print(f"[WARNING] Status {resp.status_code}: {resp.text[:200]}")
        return None
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al consultar tasa de financiamiento: {e}")
        return None


__all__ = [
    '_validate_symbol',
    'get_ticker_price',
    'get_contract_info',
    'get_server_time',
    'get_candles',
    'get_trades',
    'get_funding_rate',
]
