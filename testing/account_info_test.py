"""
WEEX Account Info Test - Prueba simplificada de conexión a la API

Verifica:
1. Conectividad con el servidor (endpoint público)
2. Configuración de credenciales
3. Información de la cuenta
"""

import sys
import os
# Agregar el directorio padre al path para importar módulos locales
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import _env
from api.market import get_server_time
from api.account import get_account_assets, get_public_ip

def print_account_assets():
	"""Obtiene y muestra los activos de la cuenta."""
	api_key = _env("API_Key", "")
	secret_key = _env("secret_key", "")
	passphrase = _env("passphrase", "")
	locale = _env("WEEX_LOCALE", "en-US") or "en-US"

	if not api_key or not secret_key or not passphrase:
		print("[INFO] Falta configuración de credenciales.")
		print("Configure las variables de entorno:")
		print("  - API_Key")
		print("  - secret_key")
		print("  - passphrase")
		print("\nEjemplo (PowerShell):")
		print("  $env:API_Key = 'tu_api_key'")
		print("  $env:secret_key = 'tu_secret_key'")
		print("  $env:passphrase = 'tu_passphrase'\n")
		return

	get_account_assets(api_key, secret_key, passphrase, locale)


def main():
	"""Flujo principal de prueba."""
	# 1) Comprobación rápida de conectividad (público)
	print("== Tiempo del servidor (público) ==")
	st = get_server_time(verbose=True)
	
	if isinstance(st, dict) and st.get("http_status") == 521:
		ip = get_public_ip(verbose=True)
		print("\n[Diagnóstico] Código 521 indica IP no autorizada/servidor indisponible.")
		if ip:
			print("Verifique que esta IP esté en la whitelist de WEEX.")
	print()

	# 2) Intento de obtener información de activos de cuenta (privado)
	print("== Activos de la cuenta (privado) ==")
	print_account_assets()


if __name__ == "__main__":
	main()

