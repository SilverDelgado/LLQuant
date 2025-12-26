"""
Descarga de datos históricos externos para todos los activos permitidos.

Este script descarga datos históricos de velas (candlestick) para todos los símbolos
en ALLOWED_SYMBOLS, con granularidades de 1h y 4h, y los guarda en formato Parquet
en el directorio data/raw/.
"""

import os
import sys
import time
import pandas as pd

# Agregar el directorio padre al path para importar módulos locales
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import ALLOWED_SYMBOLS, _env
from api.market import get_candles


def download_historical_data():
    """
    Descarga datos históricos para todos los símbolos permitidos y granularidades especificadas.
    """
    # Obtener credenciales
    api_key = _env("API_Key", "")
    secret_key = _env("secret_key", "")
    passphrase = _env("passphrase", "")
    locale = _env("WEEX_LOCALE", "en-US") or "en-US"

    # Granularidades a descargar
    granularities = ["1h", "4h"]

    # Crear directorio si no existe
    raw_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("Iniciando descarga de datos históricos...")
    print(f"Símbolos: {ALLOWED_SYMBOLS}")
    print(f"Granularidades: {granularities}")
    print(f"Directorio de destino: {raw_dir}")
    print()

    total_downloads = 0

    for symbol in ALLOWED_SYMBOLS:
        for granularity in granularities:
            print(f"Descargando {symbol} - {granularity}...")

            try:
                # Obtener datos históricos (últimas 1000 velas)
                candles = get_candles(
                    api_key=api_key,
                    secret_key=secret_key,
                    passphrase=passphrase,
                    symbol=symbol,
                    granularity=granularity,
                    limit=1000,
                    locale=locale,
                    verbose=False  # Menos verboso para batch
                )

                if candles and len(candles) > 0:
                    # Inspeccionar la estructura de los datos
                    first_candle = candles[0]
                    num_cols = len(first_candle)
                    
                    # Definir columnas basadas en el número de elementos
                    if num_cols == 6:
                        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    elif num_cols == 7:
                        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                    else:
                        columns = [f'col_{i}' for i in range(num_cols)]
                    
                    # Convertir a DataFrame
                    df = pd.DataFrame(candles, columns=columns)

                    # Convertir tipos numéricos (excepto timestamp)
                    numeric_cols = [col for col in columns if col != 'timestamp']
                    df[numeric_cols] = df[numeric_cols].astype(float)
                    df['timestamp'] = df['timestamp'].astype(int)

                    # Guardar en Parquet
                    filename = f"{symbol}_{granularity}.parquet"
                    filepath = os.path.join(raw_dir, filename)
                    df.to_parquet(filepath, index=False)

                    print(f"  ✓ Guardado: {filename} ({len(df)} registros)")
                    total_downloads += 1
                else:
                    print(f"  ✗ No se obtuvieron datos para {symbol} - {granularity}")

            except Exception as e:
                print(f"  ✗ Error descargando {symbol} - {granularity}: {e}")

            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)

    print()
    print(f"Descarga completada. Total de archivos descargados: {total_downloads}")


if __name__ == "__main__":
    download_historical_data()