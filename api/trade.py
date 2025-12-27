"""
WEEX Trade API - Funciones para ejecutar y gestionar operaciones (órdenes y posiciones)
"""

import json
import uuid
from typing import Optional, Dict, Any
from . import send_get, send_post, ALLOWED_SYMBOLS
from .market import get_ticker_price, get_contract_info


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


def upload_ai_log(
    api_key: str,
    secret_key: str,
    passphrase: str,
    order_id: Optional[int] = None,
    stage: str = "Strategy Generation",
    model: str = "Trading Algorithm",
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    explanation: str = "",
    locale: str = "en-US",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Sube un log de AI al endpoint de WEEX.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        order_id: ID de la orden (opcional)
        stage: Etapa del trading donde participó la AI
        model: Nombre o versión del modelo de AI
        input_data: Datos de entrada al modelo
        output_data: Datos de salida del modelo
        explanation: Explicación del razonamiento de la AI (máximo 1000 caracteres)
        locale: Idioma
        verbose: Mostrar información en consola
    
    Returns:
        Diccionario con respuesta de la API, o None si hay error
    """
    if input_data is None:
        input_data = {}
    if output_data is None:
        output_data = {}
    
    # Truncar explicación a 1000 caracteres
    if len(explanation) > 1000:
        explanation = explanation[:997] + "..."
    
    log_data = {
        "orderId": order_id,
        "stage": stage,
        "model": model,
        "input": input_data,
        "output": output_data,
        "explanation": explanation,
    }
    
    if verbose:
        print("\n[INFO] Subiendo log de AI...")
        print(json.dumps(log_data, indent=2, ensure_ascii=False))
        print()
    
    try:
        resp = send_post(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            request_path="/capi/v2/order/uploadAiLog",
            body_obj=log_data,
            auth=True,
            locale=locale,
        )
        
        if verbose:
            print(f"HTTP Status: {resp.status_code}")
        
        data = resp.json()
        
        if verbose:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        
        if resp.status_code == 200:
            if isinstance(data, dict) and data.get("code") == "00000":
                if verbose:
                    print(f"\n✓ Log de AI subido exitosamente")
                return data
        
        if verbose:
            print(f"\n✗ Error al subir log de AI")
        return data
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al subir log de AI: {e}")
        return None


def place_order(
    api_key: str,
    secret_key: str,
    passphrase: str,
    symbol: str = "cmt_btcusdt",
    position_side: str = "long",
    notional_value: float = 10.0,
    price: str = "0",
    message: str = "",
    locale: str = "en-US",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Coloca una orden en el mercado y registra el log de AI.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        symbol: Par de trading (ej: cmt_btcusdt)
        position_side: 'long' para compra o 'short' para venta
        notional_value: Valor nocional de la orden en USDT (ej: 10.0)
        price: Precio (0 para market orders, ignorado en órdenes de mercado)
        message: Mensaje explicativo de la decisión de trading (para AI log)
        locale: Idioma
        verbose: Mostrar información en consola
    
    Returns:
        Diccionario con respuesta de la API, o None si hay error
    
    Nota:
        En WEEX Futures:
        - size = cantidad en BTC (cantidad del asset base)
        - Valor nocional = size * precio
        - Ejemplo: 0.0001 BTC * $87,000 = $8.7 USDT de valor nocional
    
    Ejemplo:
        >>> order = place_order(
        ...     api_key, secret_key, passphrase,
        ...     symbol="cmt_btcusdt",
        ...     position_side="long",
        ...     notional_value=10.0
        ... )
        >>> if order:
        ...     print(f"Orden ID: {order.get('order_id')}")
    """
    if not _validate_symbol(symbol, verbose):
        return None
        
    if verbose:
        print("\n[INFO] Consultando información de mercado para colocar orden...")
    
    # Obtener precio actual
    current_price = get_ticker_price(
        api_key, secret_key, passphrase, symbol, locale, verbose=False
    )
    if not current_price:
        if verbose:
            print("[ERROR] No se pudo obtener el precio actual")
        return None
    
    # Obtener información del contrato
    contract_info = get_contract_info(
        api_key, secret_key, passphrase, symbol, locale, verbose=False
    )
    
    # Valores por defecto
    min_order_size = 0.0001  # Tamaño mínimo por defecto (en BTC)
    size_increment = 4  # Número de decimales
    
    if contract_info:
        min_order_size = float(contract_info.get('minOrderSize', 0.0001))
        size_increment = int(contract_info.get('size_increment', 4))
    
    # Calcular size basado en valor nocional deseado
    # size (en BTC) = notional_value (en USDT) / price (USDT por BTC)
    size = notional_value / current_price
    
    # Asegurar que cumple con el tamaño mínimo
    if size < min_order_size:
        size = min_order_size
        if verbose:
            print(f"[INFO] Ajustado al tamaño mínimo: {min_order_size} BTC")
    
    # Redondear según precisión del contrato
    size_str = f"{size:.{size_increment}f}"
    actual_notional = size * current_price
    
    if verbose:
        print(f"\n[INFO] Cálculo de orden:")
        print(f"  - Precio actual: ${current_price:,.2f}")
        print(f"  - Valor nocional deseado: ${notional_value:.2f} USDT")
        print(f"  - Tamaño calculado: {size_str} {symbol.split('_')[1].lower()}")
        print(f"  - Valor nocional real: ${actual_notional:.2f} USDT")
    
    # Generar client_oid único
    client_oid = str(uuid.uuid4()).replace("-", "")[:32]
    
    # type: 1=Open long, 2=Open short, 3=Close long, 4=Close short
    order_type_map = {"long": "1", "short": "2"}
    
    if verbose:
        print("\n" + "="*60)
        print(f"COLOCANDO ORDEN: OPEN {position_side.upper()} {size_str} en {symbol}")
        print("="*60)
    
    order_data = {
        "symbol": symbol,
        "client_oid": client_oid,
        "size": size_str,
        "type": order_type_map.get(position_side, "1"),  # 1: Open long
        "order_type": "0",  # 0: Normal order
        "match_price": "1",  # 1: Market price
        "price": price,
    }
    
    try:
        resp = send_post(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            request_path="/capi/v2/order/placeOrder",
            body_obj=order_data,
            auth=True,
            locale=locale,
        )
        
        if verbose:
            print(f"HTTP Status: {resp.status_code}")
        
        data = resp.json()
        
        if verbose:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        
        if resp.status_code == 200:
            # Validar que la orden fue exitosa
            order_id = None
            if isinstance(data, dict):
                if data.get("code") == "00000":
                    if verbose:
                        print(f"\n✓ Orden colocada exitosamente")
                    order_id = data.get("data", {}).get("orderId") if isinstance(data.get("data"), dict) else None
                elif "order_id" in data:
                    if verbose:
                        print(f"\n✓ Orden colocada exitosamente")
                        print(f"Order ID: {data.get('order_id')}")
                    order_id = data.get("order_id")
                
                # Subir log de AI si hay mensaje
                if message:
                    input_data = {
                        "symbol": symbol,
                        "position_side": position_side,
                        "notional_value": notional_value,
                        "price": current_price,
                    }
                    output_data = {
                        "signal": position_side.upper(),
                        "size": size_str,
                        "actual_notional": actual_notional,
                    }
                    upload_ai_log(
                        api_key=api_key,
                        secret_key=secret_key,
                        passphrase=passphrase,
                        order_id=order_id,
                        stage="Order Execution",
                        model="Trading System",
                        input_data=input_data,
                        output_data=output_data,
                        explanation=message,
                        locale=locale,
                        verbose=verbose,
                    )
                
                return data
            else:
                if verbose:
                    print(f"\n✗ Error al colocar orden: respuesta inesperada")
        else:
            if verbose:
                print(f"\n✗ Error HTTP al colocar orden")
        
        return data
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al colocar orden: {e}")
        return None


def get_positions(
    api_key: str,
    secret_key: str,
    passphrase: str,
    symbol: str = "cmt_btcusdt",
    locale: str = "en-US",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Obtiene las posiciones abiertas para un símbolo.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        symbol: Par de trading
        locale: Idioma
        verbose: Mostrar información en consola
    
    Returns:
        Diccionario con posiciones abiertas, o None si hay error
    
    Nota:
        Este endpoint requiere que la IP esté en la whitelist.
        Si retorna 521, las operaciones siguen funcionando correctamente.
    
    Ejemplo:
        >>> positions = get_positions(api_key, secret_key, passphrase, "cmt_btcusdt")
        >>> if positions:
        ...     print(f"Posiciones: {positions}")
    """
    if not _validate_symbol(symbol, verbose):
        return None
        
    if verbose:
        print("\n" + "="*60)
        print(f"CONSULTANDO POSICIONES ABIERTAS: {symbol}")
        print("="*60)
    
    try:
        resp = send_get(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            request_path="/capi/v2/account/position/singlePosition",
            params={"symbol": symbol},
            auth=True,
            locale=locale,
        )
        
        if verbose:
            print(f"HTTP Status: {resp.status_code}")
        
        if resp.status_code == 521:
            if verbose:
                print("[WARNING] Error 521: IP no está en la whitelist para este endpoint")
                print("[INFO] Las operaciones de trading funcionan correctamente")
            return None
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                if verbose:
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                return data
            except Exception as e:
                if verbose:
                    print(f"[ERROR] Error al parsear respuesta: {e}")
                return None
        else:
            if verbose:
                print(f"[WARNING] Respuesta no exitosa: {resp.text[:200]}")
            return None
            
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al consultar posiciones: {e}")
        return None


def close_position(
    api_key: str,
    secret_key: str,
    passphrase: str,
    symbol: str = "cmt_btcusdt",
    hold_side: str = "long",
    locale: str = "en-US",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Cierra una posición abierta.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        symbol: Par de trading
        hold_side: 'long' o 'short' - el lado de la posición que se quiere cerrar
        locale: Idioma
        verbose: Mostrar información en consola
    
    Returns:
        Diccionario con respuesta de la API, o None si hay error
    
    Ejemplo:
        >>> result = close_position(
        ...     api_key, secret_key, passphrase,
        ...     symbol="cmt_btcusdt",
        ...     hold_side="long"
        ... )
        >>> if result:
        ...     print("Posición cerrada")
    """
    if not _validate_symbol(symbol, verbose):
        return None
        
    if verbose:
        print("\n" + "="*60)
        print(f"CERRANDO POSICIÓN {hold_side.upper()} en {symbol}")
        print("="*60)
    
    close_data = {
        "symbol": symbol,
    }
    
    try:
        resp = send_post(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            request_path="/capi/v2/order/closePositions",
            body_obj=close_data,
            auth=True,
            locale=locale,
        )
        
        if verbose:
            print(f"HTTP Status: {resp.status_code}")
        
        data = resp.json()
        
        if verbose:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        
        if resp.status_code == 200:
            if isinstance(data, list):
                if len(data) == 0:
                    if verbose:
                        print(f"\n⚠ No hay posiciones abiertas para cerrar")
                else:
                    if verbose:
                        print(f"\n✓ Posición cerrada exitosamente")
            elif isinstance(data, dict):
                if data.get("code") == "00000":
                    if verbose:
                        print(f"\n✓ Posición cerrada exitosamente")
                else:
                    if verbose:
                        print(f"\n✗ Error al cerrar posición: {data.get('msg', 'Unknown error')}")
        else:
            if verbose:
                print(f"\n✗ Error HTTP al cerrar posición")
        
        return data
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Error al cerrar posición: {e}")
        return None


__all__ = [
    '_validate_symbol',
    'upload_ai_log',
    'place_order',
    'get_positions',
    'close_position',
]
