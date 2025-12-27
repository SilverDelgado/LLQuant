"""
WEEX API Test Task - Automated Trading Script

Este script ejecuta un flujo completo de trading automatizado en la plataforma WEEX:

1. Consulta el balance inicial de la cuenta
2. Ejecuta una orden de mercado de 10 USDT en el par cmt_btcusdt
3. Verifica el estado de la cuenta después de la orden
4. Consulta las posiciones abiertas
5. Cierra todas las posiciones del símbolo
6. Verifica el estado final de la cuenta

Requisitos:
    - Variables de entorno: API_Key, secret_key, passphrase
    - Paquetes: requests, python-dotenv (opcional)
    - Conexión a: https://api-contract.weex.com

Uso:
    python api_test_task.py

Documentación API:
    https://www.weex.com/api-doc/ai/introduction/ParticipantGuide

Nota:
    Solo se permiten los siguientes pares de trading en la competición:
    cmt_btcusdt, cmt_ethusdt, cmt_solusdt, cmt_dogeusdt, cmt_xrpusdt,
    cmt_adausdt, cmt_bnbusdt, cmt_ltcusdt
"""

import sys
import os
# Agregar el directorio padre al path para importar módulos locales
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from api import _env
from api.account import get_account_assets, set_leverage
from api.trade import place_order, get_positions, close_position


def main():
    """
    Flujo principal:
    1. Ejecutar una orden de 10 USDT en cmt_btcusdt
    2. Consultar el estado de la cuenta
    3. Cerrar la posición
    """
    # Obtener credenciales
    api_key = _env("API_Key", "")
    secret_key = _env("secret_key", "")
    passphrase = _env("passphrase", "")
    locale = _env("WEEX_LOCALE", "en-US") or "en-US"

    if not api_key or not secret_key or not passphrase:
        print("[ERROR] Faltan credenciales de API.")
        print("Configure las variables de entorno:")
        print("  - API_Key")
        print("  - secret_key")
        print("  - passphrase")
        print("\nEjemplo (PowerShell):")
        print("  $env:API_Key = 'tu_api_key'")
        print("  $env:secret_key = 'tu_secret_key'")
        print("  $env:passphrase = 'tu_passphrase'")
        return

    print("\n" + "="*60)
    print("INICIANDO TAREA: Orden de 10 USDT en cmt_btcusdt")
    print("="*60)

    # Paso 1: Consultar estado inicial de la cuenta
    print("\n[PASO 1] Estado inicial de la cuenta:")
    get_account_assets(api_key, secret_key, passphrase, locale)

    # Paso 1.5: Configurar leverage (máximo 20x en la competición)
    print("\n[PASO 1.5] Configurando leverage:")
    set_leverage(api_key, secret_key, passphrase, "cmt_btcusdt", "10", locale)

    # Paso 2: Colocar orden de compra con valor nocional de 10 USDT en cmt_btcusdt
    print("\n[PASO 2] Ejecutando orden:")
    order_result = place_order(
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
        symbol="cmt_btcusdt",
        position_side="long",  # Abrir posición long
        notional_value=10.0,  # 10 USDT de valor nocional
        price="0",  # Market price
        message="Test order execution: Opening 10 USDT long position on BTC for API testing purposes.",
        locale=locale
    )

    # Esperar un momento para que se procese la orden
    time.sleep(2)

    # Paso 3: Consultar estado de la cuenta después de la orden
    print("\n[PASO 3] Estado de la cuenta después de ejecutar la orden:")
    get_account_assets(api_key, secret_key, passphrase, locale)

    # Paso 4: Consultar posiciones abiertas (paso informativo, no crítico)
    print("\n[PASO 4] Consultando posiciones abiertas:")
    print("[INFO] Este paso es opcional - si falla por whitelist no afecta la operación")
    positions = get_positions(api_key, secret_key, passphrase, "cmt_btcusdt", locale)

    # Esperar un momento antes de cerrar
    time.sleep(2)

    # Paso 5: Cerrar la posición
    print("\n[PASO 5] Cerrando la posición:")
    close_result = close_position(
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
        symbol="cmt_btcusdt",
        hold_side="long",  # Cerrar posición long
        locale=locale
    )

    # Esperar un momento para que se procese el cierre
    time.sleep(2)

    # Paso 6: Consultar estado final de la cuenta
    print("\n[PASO 6] Estado final de la cuenta:")
    get_account_assets(api_key, secret_key, passphrase, locale)

    print("\n" + "="*60)
    print("TAREA COMPLETADA")
    print("="*60)


if __name__ == "__main__":
    main()
