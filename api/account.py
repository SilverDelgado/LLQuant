"""
WEEX Account API - Funciones para gestionar la cuenta y configuración
"""

import json
from typing import Optional, Dict, Any
from . import send_get, send_post, ALLOWED_SYMBOLS


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


def get_account_assets(
    api_key: str,
    secret_key: str,
    passphrase: str,
    locale: str = "en-US",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Obtiene información de los activos y balance de la cuenta.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        locale: Idioma
        verbose: Mostrar información en consola
    
    Returns:
        Diccionario con información de activos, o None si hay error
    
    Información retornada:
        - total_equity: Patrimonio total
        - unrealized_pnl: PnL no realizado
        - realised_pnl: PnL realizado
        - balance: Balance disponible
        - coins: Lista de monedas con sus balances
    
    Ejemplo:
        >>> assets = get_account_assets(api_key, secret_key, passphrase)
        >>> print(f"Balance: ${assets.get('balance', 0)}")
    """
    if verbose:
        print("\n" + "="*60)
        print("CONSULTANDO ESTADO DE LA CUENTA")
        print("="*60)
    
    try:
        resp = send_get(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            request_path="/capi/v2/account/assets",
            params=None,
            auth=True,
            locale=locale,
        )
        
        if verbose:
            print(f"HTTP Status: {resp.status_code}")
        
        data = resp.json()
        
        if verbose:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        
        return data
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al consultar activos: {e}")
        return None


def set_leverage(
    api_key: str,
    secret_key: str,
    passphrase: str,
    symbol: str = "cmt_btcusdt",
    leverage: str = "1",
    locale: str = "en-US",
    verbose: bool = True,
) -> bool:
    """
    Configura el leverage (apalancamiento) para un símbolo específico.
    
    Nota: El leverage máximo permitido en la competición es 20x.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        symbol: Par de trading
        leverage: Valor de leverage (ej: "1", "5", "10", "20")
        locale: Idioma
        verbose: Mostrar información en consola
    
    Returns:
        True si se configuró exitosamente, False en caso contrario
    
    Ejemplo:
        >>> if set_leverage(api_key, secret_key, passphrase, "cmt_btcusdt", "10"):
        ...     print("Leverage configurado a 10x")
    """
    if not _validate_symbol(symbol, verbose):
        return False
        
    if verbose:
        print(f"\n[INFO] Configurando leverage {leverage}x para {symbol}...")
    
    leverage_data = {
        "symbol": symbol,
        "marginMode": 1,  # 1: Cross Mode, 3: Isolated Mode
        "longLeverage": leverage,
        "shortLeverage": leverage,
    }
    
    try:
        resp = send_post(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            request_path="/capi/v2/account/leverage",
            body_obj=leverage_data,
            auth=True,
            locale=locale,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            
            if isinstance(data, dict) and data.get("code") == "200":
                if verbose:
                    print(f"[INFO] ✓ Leverage configurado exitosamente a {leverage}x")
                return True
            else:
                if verbose:
                    print(f"[WARNING] Respuesta al configurar leverage: {data}")
                return False
        
        if verbose:
            print(f"[ERROR] Error HTTP {resp.status_code}")
        return False
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al configurar leverage: {e}")
        return False


def get_public_ip(verbose: bool = False) -> Optional[str]:
    """
    Obtiene la dirección IP pública actual (usando ipify).
    Útil para debugging de whitelist.
    
    Args:
        verbose: Mostrar información en consola
    
    Returns:
        String con la dirección IP, o None si hay error
    
    Ejemplo:
        >>> ip = get_public_ip()
        >>> print(f"Tu IP pública: {ip}")
    """
    import requests
    
    if verbose:
        print("[INFO] Detectando IP pública...")
    
    try:
        r = requests.get("https://api.ipify.org?format=json", timeout=10)
        if r.ok:
            j = r.json()
            ip = j.get("ip")
            if verbose:
                print(f"[INFO] IP pública detectada: {ip}")
            return ip
        
        if verbose:
            print(f"[WARNING] Error al detectar IP: {r.status_code}")
        return None
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al detectar IP: {e}")
        return None


__all__ = [
    '_validate_symbol',
    'get_account_assets',
    'set_leverage',
    'get_public_ip',
]
