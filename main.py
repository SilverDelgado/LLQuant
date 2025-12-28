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

from src.data import get_structured_data, get_unstructured_data, prepare_full_payload, get_df_data
from src.risk_manager import motor_de_riesgo
from src.llm import get_llm_analysis
from src.inference import generate_signals
from src.execution import rebalance_portfolio, get_credentials, close_all_positions
from api import ALLOWED_SYMBOLS

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
    "execution_mode": "both",  # longonly | shortonly | both
}


# ======================= PIPELINE PRINCIPAL =======================

def acquire_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fase 1: Adquisición de datos del mercado
    
    Obtiene datos estructurados (OHLCV, indicadores técnicos) y no 
    estructurados (noticias, funding rates) del mercado.
    
    NOTA: Las señales cuantitativas se generan UNA VEZ para todos los símbolos,
    no símbolo por símbolo (ver acquire_all_market_data).
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


def acquire_all_market_data(symbols: list) -> Dict[str, Any]:
    """
    Adquiere datos de todos los símbolos y genera señales cuantitativas.
    
    Las señales ML se generan UNA VEZ para todos los activos simultáneamente
    (no símbolo por símbolo) ya que requieren neutralización cross-sectional.
    """
    market_data = {}
    
    # 1. Obtener datos estructurados/no estructurados por símbolo
    for symbol in symbols:
        data = acquire_market_data(symbol)
        if data:
            market_data[symbol] = data
    
    if not market_data:
        return {}
    
    # 2. Generar señales cuantitativas para TODOS los símbolos a la vez
    try:
        logger.info("🤖 Generando señales cuantitativas (ML)...")
        ml_dataset = get_df_data(
            symbols=list(market_data.keys()), 
            timeframe="4h",
            horizon=8,
            d=0.4,
            window=50,
            vpt_price_d_window=6,
            vol_window=12,
            limit=600,
            verbose=False,
            inference_mode=True
        )
        llm_context, top_picks = generate_signals(market_df=ml_dataset)
        
        # Distribuir señales a cada símbolo
        for symbol in market_data:
            market_data[symbol]["quant_signals"] = {
                "llm_context": llm_context,
                "top_picks": top_picks,
            }
        logger.info(f"✓ Señales cuantitativas generadas para {len(top_picks)} activos")
    except Exception as e:
        logger.error(f"Error generando señales ML: {e}")
        # Continuar sin señales cuantitativas
    
    return market_data


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


def _extract_top_picks(market_data: Dict[str, Any]) -> list:
    """Recupera el ranking cuantitativo ya generado por generate_signals."""

    for data in market_data.values():
        quant = data.get("quant_signals") if isinstance(data, dict) else None
        top_picks = quant.get("top_picks") if quant else None
        if top_picks:
            return top_picks
    return []


def _quant_fallback_llm_output(
    market_data: Dict[str, Any],
    base_portfolio: Dict[str, Any]
) -> Dict[str, Any]:
    """Fallback determinista basado en ranking cuantitativo, no aleatorio."""

    assets = list(market_data.keys()) if market_data else []
    num_assets = len(assets) or len(base_portfolio.get("conviction_scores", [])) or 3

    top_picks = _extract_top_picks(market_data)
    rank_map = {str(p.get("ticker")): p.get("rank") for p in top_picks if "ticker" in p}

    def score_from_rank(rank: Optional[int]) -> float:
        if not rank:
            return 0.52
        # Escala decreciente desde 0.82
        return max(0.52, 0.82 - 0.05 * (rank - 1))

    conviction_scores = []
    for sym in assets:
        conviction_scores.append(score_from_rank(rank_map.get(sym)))

    if not conviction_scores:
        conviction_scores = [0.52] * num_assets

    top_line = ", ".join(str(p.get("ticker", "?")) for p in top_picks[:3]) if top_picks else "sin ranking"

    return {
        "should_rebalance": True,
        "recommendation": "Gemini no respondió; se usa ranking cuantitativo para rebalancear ligero.",
        "conviction_scores": conviction_scores,
        "confidence": 0.58,
        "rationale": f"Fallback cuantitativo: prioriza top alpha ({top_line})."
    }


def enhance_with_llm(market_data: Dict[str, Any], base_portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fase 3: Mejora con LLM (Gemini)
    
    El LLM analiza el contexto macro, noticias y datos técnicos para ajustar
    el portfolio y determinar si se debe rebalancear.
    """
    logger.info("🧠 Analizando con LLM (Gemini)...")
    
    fallback_llm = _quant_fallback_llm_output(market_data, base_portfolio)

    try:
        llm_analysis = get_llm_analysis(market_data, base_portfolio)
        expected_assets = len(market_data) if market_data else len(base_portfolio.get("conviction_scores", []))
        if (
            "error" in llm_analysis.get("recommendation", "").lower()
            or llm_analysis.get("confidence", 0.0) <= 0.0
            or len(llm_analysis.get("conviction_scores", [])) != expected_assets
        ):
            logger.warning("Gemini falló o devolvió datos incompletos; usando fallback cuantitativo.")
            llm_analysis = fallback_llm
        logger.info(f"✓ Análisis LLM completado - Rebalancear: {llm_analysis['should_rebalance']}")
        return llm_analysis
    except Exception as e:
        logger.error(f"Error en análisis LLM: {e}")
        return fallback_llm


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normaliza pesos para que la suma de valores absolutos sea 1."""

    total_abs = sum(abs(v) for v in weights.values())
    if total_abs == 0:
        return {k: 0.0 for k in weights}
    return {k: v / total_abs for k, v in weights.items()}


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
    
    """
    if not should_rebalance:
        logger.info("⏭️  Rebalanceo no necesario, esperando...")
        return False

    weights = optimized_portfolio.get("weights", {})
    if not weights:
        logger.warning("No hay pesos optimizados disponibles; se omite ejecución.")
        return False

    target_weights = _normalize_weights(weights)
    mode = CONFIG.get("execution_mode", "both")

    api_key, secret_key, passphrase, locale = get_credentials()
    if not api_key:
        logger.warning("Credenciales de API no configuradas; ejecución omitida.")
        return False

    logger.info("🚀 Ejecutando rebalanceo real vía execution.py...")
    rebalance_portfolio(api_key, secret_key, passphrase, locale, target_weights, mode)
    logger.info("✓ Rebalanceo enviado al mercado")
    return True


def _close_positions_on_exit():
    """Cierra todas las posiciones al terminar el proceso, si hay credenciales."""

    api_key, secret_key, passphrase, locale = get_credentials()
    if not api_key:
        logger.warning("Sin credenciales de API; no se cierran posiciones en salida.")
        return

    try:
        logger.info("🔻 Cerrando posiciones abiertas antes de salir...")
        close_all_positions(api_key, secret_key, passphrase, locale)
        logger.info("✓ Posiciones cerradas")
    except Exception as e:
        logger.error(f"Error al cerrar posiciones en salida: {e}")


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
            # 1️⃣ Adquisición de datos (incluyendo señales cuantitativas)
            market_data = acquire_all_market_data(CONFIG["symbols"])
            
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

    _close_positions_on_exit()


if __name__ == "__main__":
    main()
