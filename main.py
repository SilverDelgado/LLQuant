"""
LLQuant - Gestor de Portfolio Inteligente con IA

Pipeline de Ejecución:
    1. Adquisición: Datos estructurados y no estructurados del mercado
    2. ML Base: Modelo inicial de portfolio
    3. LLM Enhancement: Ajustes basados en análisis semántico e IA. Así sabemos si hay que rebalancear o no y como lo rebalanceamos.
    4. Risk Control: Aplicación de controles de riesgo
    5. Optimization: Aplicación de Black-Litterman
    6. Execution: Colocación de órdenes en mercado
    7. Monitoring: Seguimiento y rebalanceo automático

Arquitectura limpia y escalable para trading cuantitativo.
"""

import sys
import os
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data import get_structured_data, get_unstructured_data, prepare_full_payload
from src.risk_manager import motor_de_riesgo
from src.llm import get_llm_analysis
from api import ALLOWED_SYMBOLS
# from src.black_litterman import apply_black_litterman
# from src.execution import execute_rebalance
# from src.train_ml import get_initial_portfolio


# ======================= CONFIGURACIÓN =======================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

CONFIG = {
    "symbols": ALLOWED_SYMBOLS,  # Usar todos los símbolos permitidos
    "risk_profile": "medio_riesgo",
    "rebalance_threshold": 0.02,  # 2%
    "check_interval": 3600,  # 1 hora
    "max_loops": None,  # None = infinito
}


# ======================= PIPELINE PRINCIPAL =======================

def acquire_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fase 1: Adquisición de datos del mercado
    
    Obtiene datos estructurados (OHLCV, indicadores técnicos) y no 
    estructurados (noticias, funding rates) del mercado.
    """
    try:
        logger.info(f"📊 Adquiriendo datos para {symbol}...")
        
        structured = get_structured_data(symbol, verbose=False)
        unstructured = get_unstructured_data(symbol, verbose=False)
        
        # Fusión inteligente de datos
        payload = prepare_full_payload(structured, unstructured)
        
        logger.info(f"✓ Datos adquiridos para {symbol}")
        return payload
        
    except Exception as e:
        logger.error(f"✗ Error adquiriendo datos para {symbol}: {e}")
        return None


def get_initial_portfolio(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fase 2: Obtener portfolio inicial del modelo ML
    
    TODO: Integrar con train_ml.py para obtener predicciones del modelo
    """
    logger.info("🤖 Generando portfolio base (ML)...")
    
    # Placeholder - será reemplazado con modelo ML real
    initial_portfolio = {
        "conviction_scores": [0.75, 0.68, 0.82],
        "base_weights": [0.40, 0.35, 0.25],
        "confidence": 0.73
    }
    
    logger.info(f"✓ Portfolio ML generado")
    return initial_portfolio


def enhance_with_llm(market_data: Dict[str, Any], base_portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fase 3: Mejora con LLM (Gemini)
    
    El LLM analiza el contexto macro, noticias y datos técnicos para ajustar
    el portfolio y determinar si se debe rebalancear.
    """
    logger.info("🧠 Analizando con LLM (Gemini)...")
    
    try:
        llm_analysis = get_llm_analysis(market_data, base_portfolio)
        logger.info(f"✓ Análisis LLM completado - Rebalancear: {llm_analysis['should_rebalance']}")
        return llm_analysis
    except Exception as e:
        logger.error(f"Error en análisis LLM: {e}")
        # Fallback conservador
        return {
            "should_rebalance": False,
            "recommendation": "Error en LLM, usando pesos actuales",
            "conviction_scores": base_portfolio.get("conviction_scores", [0.5] * len(market_data)),
            "confidence": 0.3,
            "rationale": "Fallback a valores por defecto"
        }


def apply_risk_controls(
    market_data: Dict[str, Any],
    llm_analysis: Dict[str, Any],
    drawdown: float = 0.0
) -> Dict[str, Any]:
    """
    Fase 4: Aplicar controles de riesgo
    
    Usa risk_manager.py para calcular el nivel de exposición permitido
    basado en drawdown y perfil de riesgo.
    """
    logger.info("⚠️  Aplicando controles de riesgo...")
    
    conviction_scores = llm_analysis.get("conviction_scores", [])
    
    risk_result = motor_de_riesgo(
        perfil=CONFIG["risk_profile"],
        drawdown_actual=drawdown,
        lista_scores_llm=conviction_scores,
        top_n=3
    )
    
    logger.info(f"✓ Exposición final: {risk_result['RESULTADO_FINAL_PCT']}")
    return risk_result


def optimize_with_black_litterman(
    risk_result: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fase 5: Optimización con Black-Litterman
    
    TODO: Integrar con black_litterman.py para optimizar pesos del portfolio
    """
    logger.info("📈 Optimizando pesos (Black-Litterman)...")
    
    # Placeholder - será reemplazado con optimización real
    optimized_weights = {
        "cmt_btcusdt": 0.42,
        "cmt_ethusdt": 0.33,
        "cmt_solusdt": 0.25
    }
    
    logger.info(f"✓ Pesos optimizados")
    return {
        "weights": optimized_weights,
        "risk_adjusted_return": 0.18,
        "sharpe_ratio": 1.42
    }


def execute_rebalance(
    optimized_portfolio: Dict[str, Any],
    should_rebalance: bool
) -> bool:
    """
    Fase 6: Ejecución de rebalanceo
    
    Envía órdenes al mercado basadas en los pesos optimizados.
    
    TODO: Integrar con execution.py para colocar órdenes
    """
    if not should_rebalance:
        logger.info("⏭️  Rebalanceo no necesario, esperando...")
        return False
    
    logger.info("🚀 Ejecutando rebalanceo...")
    
    # Placeholder - será reemplazado con ejecución real
    for symbol, weight in optimized_portfolio["weights"].items():
        logger.info(f"   → {symbol}: {weight*100:.1f}%")
    
    logger.info("✓ Rebalanceo completado")
    return True


def main():
    """
    Orquestador principal: ejecuta el pipeline completo de forma iterativa.
    """
    logger.info("=" * 60)
    logger.info("LLQuant - Sistema de Trading Cuantitativo Inteligente")
    logger.info("=" * 60)
    
    loop_count = 0
    
    while True:
        loop_count += 1
        logger.info(f"\n[CICLO {loop_count}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 1️⃣ Adquisición de datos
            market_data = {}
            for symbol in CONFIG["symbols"]:
                data = acquire_market_data(symbol)
                if data:
                    market_data[symbol] = data
            
            if not market_data:
                logger.warning("No se pudieron adquirir datos, reintentando...")
                time.sleep(CONFIG["check_interval"])
                continue
            
            # 2️⃣ Portfolio base (ML)
            base_portfolio = get_initial_portfolio(market_data)
            
            # 3️⃣ Análisis LLM
            llm_analysis = enhance_with_llm(market_data, base_portfolio)
            
            # 4️⃣ Controles de riesgo
            risk_result = apply_risk_controls(market_data, llm_analysis)
            
            # 5️⃣ Optimización Black-Litterman
            optimized_portfolio = optimize_with_black_litterman(risk_result, market_data)
            
            # 6️⃣ Ejecución
            execute_rebalance(optimized_portfolio, llm_analysis["should_rebalance"])
            
            logger.info("✓ Ciclo completado exitosamente")
            
            # Verificar límite de loops
            if CONFIG["max_loops"] and loop_count >= CONFIG["max_loops"]:
                logger.info("Límite de ciclos alcanzado, finalizando...")
                break
            
            # Esperar hasta próximo chequeo
            logger.info(f"⏰ Siguiente chequeo en {CONFIG['check_interval']}s")
            time.sleep(CONFIG["check_interval"])
            
        except KeyboardInterrupt:
            logger.info("\n⛔ Ejecución interrumpida por usuario")
            break
        except Exception as e:
            logger.error(f"Error en ciclo {loop_count}: {e}", exc_info=True)
            time.sleep(CONFIG["check_interval"])


if __name__ == "__main__":
    main()
