"""
WEEX API Client - Funciones base para autenticación y comunicación
"""

import os
import time
import hmac
import hashlib
import base64
import json
import requests
import sys
from typing import Optional, Dict, Any
from urllib.parse import urlencode

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = "https://api-contract.weex.com"


# Activos permitidos para la competición AI Wars
ALLOWED_SYMBOLS = [
    "cmt_btcusdt",
    "cmt_ethusdt", 
    "cmt_solusdt",
    "cmt_dogeusdt",
    "cmt_xrpusdt",
    "cmt_adausdt",
    "cmt_bnbusdt",
    "cmt_ltcusdt"
]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Obtiene variable de entorno."""
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def generate_signature(
    secret_key: str,
    timestamp_ms: int,
    method: str,
    request_path: str,
    query_string: Optional[str] = None,
    body: Optional[str] = None,
) -> str:
    """
    Genera firma HMAC-SHA256 según especificación WEEX.
    
    Args:
        secret_key: Clave secreta de API
        timestamp_ms: Timestamp en milisegundos
        method: Método HTTP (GET, POST)
        request_path: Ruta de la solicitud
        query_string: Query string (opcional)
        body: Body de la solicitud (opcional)
    
    Returns:
        Firma en formato Base64
    """
    method_up = method.upper()
    ts_str = str(timestamp_ms)
    msg = ts_str + method_up + request_path
    
    if query_string:
        msg += "?" + query_string
    if body:
        msg += body

    digest = hmac.new(
        secret_key.encode("utf-8"),
        msg.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def build_headers(
    api_key: str,
    passphrase: str,
    signature: str,
    timestamp_ms: int,
    locale: str = "en-US",
) -> Dict[str, str]:
    """
    Construye headers para solicitud autenticada.
    
    Args:
        api_key: Clave de API
        passphrase: Contraseña de API
        signature: Firma HMAC
        timestamp_ms: Timestamp en milisegundos
        locale: Idioma (default: en-US)
    
    Returns:
        Diccionario de headers
    """
    return {
        "ACCESS-KEY": api_key,
        "ACCESS-PASSPHRASE": passphrase,
        "ACCESS-TIMESTAMP": str(timestamp_ms),
        "ACCESS-SIGN": signature,
        "Content-Type": "application/json",
        "locale": locale,
    }


def send_get(
    api_key: str,
    secret_key: str,
    passphrase: str,
    request_path: str,
    params: Optional[Dict[str, Any]] = None,
    auth: bool = True,
    locale: str = "en-US",
):
    """
    Envía solicitud GET a la API WEEX.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        request_path: Ruta del endpoint
        params: Parámetros de query
        auth: Si requiere autenticación
        locale: Idioma
    
    Returns:
        Response object de requests
    """
    params = params or {}
    query_string = urlencode(params) if params else None
    timestamp_ms = int(time.time() * 1000)

    headers = {}
    if auth:
        signature = generate_signature(
            secret_key=secret_key,
            timestamp_ms=timestamp_ms,
            method="GET",
            request_path=request_path,
            query_string=query_string,
            body=None,
        )
        headers = build_headers(
            api_key=api_key,
            passphrase=passphrase,
            signature=signature,
            timestamp_ms=timestamp_ms,
            locale=locale,
        )

    url = BASE_URL + request_path
    return requests.get(url, params=params, headers=headers, timeout=15)


def send_post(
    api_key: str,
    secret_key: str,
    passphrase: str,
    request_path: str,
    body_obj: Optional[Dict[str, Any]] = None,
    auth: bool = True,
    locale: str = "en-US",
):
    """
    Envía solicitud POST a la API WEEX.
    
    Args:
        api_key: Clave de API
        secret_key: Clave secreta
        passphrase: Contraseña de API
        request_path: Ruta del endpoint
        body_obj: Objeto para el body
        auth: Si requiere autenticación
        locale: Idioma
    
    Returns:
        Response object de requests
    """
    body_obj = body_obj or {}
    body_str = json.dumps(body_obj, separators=(",", ":"))
    timestamp_ms = int(time.time() * 1000)

    headers = {}
    if auth:
        signature = generate_signature(
            secret_key=secret_key,
            timestamp_ms=timestamp_ms,
            method="POST",
            request_path=request_path,
            query_string=None,
            body=body_str,
        )
        headers = build_headers(
            api_key=api_key,
            passphrase=passphrase,
            signature=signature,
            timestamp_ms=timestamp_ms,
            locale=locale,
        )

    url = BASE_URL + request_path
    return requests.post(url, data=body_str, headers=headers, timeout=15)


__all__ = [
    'BASE_URL',
    'ALLOWED_SYMBOLS',
    '_env',
    'generate_signature',
    'build_headers',
    'send_get',
    'send_post',
]
