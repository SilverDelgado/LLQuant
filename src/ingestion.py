"""archivo para descargar datos"""
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def download_data(tickers, interval='1h', days_back=60):
    """
    Descarga datos OHLCV de Yahoo Finance y los guarda en CSV.
    """
    # 1. Crear el directorio si no existe
    save_dir = os.path.join("data", "raw")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"--- Iniciando Ingesta de Datos ---")
    print(f"Guardando en: {save_dir}")
    
    # 2. Iterar sobre cada activo
    for ticker in tickers:
        print(f"Descargando: {ticker}...")
        
        try:
            # Instanciamos el objeto Ticker
            asset = yf.Ticker(ticker)
            
            # Descargamos el historial
            # period: rango de tiempo hacia atrás desde hoy
            # interval: tamaño de la vela (1h, 1d, etc.)
            df = asset.history(period=f"{days_back}d", interval=interval)
            
            if df.empty:
                print(f"No se encontraron datos para {ticker}")
                continue
            
            # Limpieza básica inicial: eliminar timezone si existe para evitar problemas luego
            df.index = df.index.tz_localize(None)
            
            # Guardamos solo las columnas OHLCV necesarias
            # Yahoo devuelve: Open, High, Low, Close, Volume, Dividends, Stock Splits
            cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[cols_to_keep]
            
            # 3. Guardar en CSV
            # Reemplazamos el '-' en el nombre del archivo (ej. BTC-USD -> BTC_USD)
            filename = f"{ticker.replace('-', '_')}_raw.csv"
            file_path = os.path.join(save_dir, filename)
            
            df.to_csv(file_path)
            print(f"Guardado: {filename} ({len(df)} filas)")
            
        except Exception as e:
            print(f"Error descargando {ticker}: {e}")

if __name__ == "__main__":
    # LISTA DE ACTIVOS (Ejemplo Mix Cripto)
    # Yahoo Finance usa el sufijo -USD para criptos
    tickers_list = [
        "BTC-USD",  # Bitcoin
        "ETH-USD",  # Ethereum
        "SOL-USD",  # Solana
        "BNB-USD",  # Binance Coin
        "XRP-USD",  # Ripple
        "ADA-USD",  # Cardano
        "DOGE-USD", # Dogecoin
        "AVAX-USD"  # Avalanche
    ]
    
    download_data(tickers_list, interval='1h', days_back=700)