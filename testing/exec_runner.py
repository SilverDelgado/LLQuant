"""
Script de testing para el sistema de ejecución de órdenes.
Ejecuta el portfolio manager en modo de prueba.
"""
import sys
import os

# Agregar el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution import run_portfolio_manager


if __name__ == "__main__":
    # Ejecutar en modo 'both' (posiciones largas y cortas)
    # Otros modos: 'longonly', 'shortonly'
    run_portfolio_manager(mode="both", sleep_interval=10)