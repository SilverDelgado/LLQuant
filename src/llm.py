"""
Módulo LLM - Análisis Inteligente con Gemini

Proporciona análisis fundamental y de contexto macro usando el modelo Gemini de Google.
El LLM evalúa el mercado y determina si se debe rebalancear el portfolio, además de
proporcionar conviction scores ajustados para cada activo.

Funciones principales:
    - get_llm_analysis(): Análisis completo del mercado y recomendación de rebalanceo
    - parse_llm_response(): Parser robusto de la respuesta de Gemini
    
Arquitectura:
    - Prompt engineering optimizado para trading cuantitativo
    - Manejo de errores y fallback a valores por defecto
    - Caching opcional para reducir llamadas a API
"""

import logging
import json
import re
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import os


# ======================= CONFIGURACIÓN =======================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configurar API de Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("⚠️  GEMINI_API_KEY no encontrada en .env")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Modelo a usar
MODEL_NAME = "gemini-2.0-flash"  # Modelo rápido y económico para trading


# ======================= SISTEMA DE PROMPTS =======================

def _build_market_prompt(market_data: Dict[str, Any], base_portfolio: Dict[str, Any]) -> str:
    """
    Construye un prompt optimizado para análisis de trading con Gemini.
    
    El prompt está diseñado para:
    - Extraer información clave del mercado
    - Evaluar riesgo y oportunidad
    - Generar conviction scores ajustados
    - Recomendar rebalanceo
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Construir contexto de activos
    assets_context = ""
    if market_data:
        for symbol, data in market_data.items():
            price = data.get("price", "N/A")
            rsi = data.get("rsi", "N/A")
            trend = data.get("trend", "N/A")
            funding = data.get("funding_rate", "N/A")
            news = data.get("recent_news", ["Sin noticias recientes"])
            
            assets_context += f"""
    {symbol.upper()}:
        - Precio actual: ${price}
        - RSI: {rsi}
        - Tendencia: {trend}
        - Funding Rate: {funding}
        - Últimas noticias: {news[0] if news else 'N/A'}
"""
    
    # Contexto del portfolio base
    portfolio_context = ""
    if base_portfolio:
        scores = base_portfolio.get("conviction_scores", [])
        weights = base_portfolio.get("base_weights", [])
        confidence = base_portfolio.get("confidence", 0.0)
        
        portfolio_context = f"""
Portfolio Base (Modelo ML):
    - Conviction Scores Iniciales: {scores}
    - Pesos Iniciales: {weights}
    - Confianza del Modelo: {confidence:.2%}
"""
    
    prompt = f"""
=== ANÁLISIS DE MERCADO Y RECOMENDACIÓN DE PORTFOLIO ===
Timestamp: {timestamp}
Rol: Eres un analista cuantitativo experto en criptomonedas y trading algorítmico.

CONTEXTO ACTUAL DEL MERCADO:
{assets_context}

{portfolio_context}

INSTRUCCIONES:
1. Analiza el contexto técnico y fundamental de cada activo.
2. Evalúa si el mercado recomienda un REBALANCEO del portfolio (cambios significativos en pesos).
3. Genera Conviction Scores ajustados (0.0 a 1.0) para cada activo basados en:
   - Tendencia técnica (RSI, precio)
   - Noticias y contexto macro
   - Risk/Reward ratio
4. Proporciona una recomendación clara y concisa.

FORMATO DE RESPUESTA (JSON):
{{
    "should_rebalance": boolean (true/false),
    "recommendation": "Breve análisis: qué está pasando en el mercado y por qué (o no) rebalancear. Máx 2 oraciones.",
    "conviction_scores": [score1, score2, score3, ...],  // Un score por activo, en el mismo orden que aparecen arriba
    "confidence": número entre 0.0 y 1.0,
    "rationale": "Explicación técnica de las conviction scores (max 3 oraciones)"
}}

RESTRICCIONES CRÍTICAS:
- Los conviction scores deben estar entre 0.0 y 1.0
- Debe haber un score por cada activo mencionado
- Confidence debe ser un número único entre 0.0 y 1.0
- Si no hay suficiente información, usa valores por defecto conservadores
- NO incluyas comentarios adicionales fuera del JSON
"""
    
    return prompt


# ======================= PARSING Y VALIDACIÓN =======================

def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Extrae y parsea JSON de la respuesta de Gemini.
    
    Maneja casos donde el JSON está embebido en texto adicional.
    """
    try:
        # Intenta parsear directamente
        return json.loads(text)
    except json.JSONDecodeError:
        # Si falla, busca JSON entre llaves
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("No se pudo extraer JSON válido de la respuesta")
                return None
    return None


def _validate_and_fix_response(response: Dict[str, Any], num_assets: int) -> Dict[str, Any]:
    """
    Valida y corrige la respuesta del LLM.
    
    Asegura que:
    - Los campos requeridos existan
    - Los tipos de datos sean correctos
    - Los conviction scores tengan la cantidad correcta de elementos
    """
    
    # Campos requeridos con valores por defecto
    validated = {
        "should_rebalance": response.get("should_rebalance", False),
        "recommendation": str(response.get("recommendation", "Sin análisis disponible"))[:200],
        "conviction_scores": response.get("conviction_scores", [0.5] * num_assets),
        "confidence": float(response.get("confidence", 0.5)),
        "rationale": str(response.get("rationale", ""))[:300]
    }
    
    # Validar tipos
    if not isinstance(validated["should_rebalance"], bool):
        validated["should_rebalance"] = str(validated["should_rebalance"]).lower() == "true"
    
    if not isinstance(validated["confidence"], (int, float)):
        validated["confidence"] = 0.5
    else:
        validated["confidence"] = max(0.0, min(1.0, float(validated["confidence"])))
    
    # Validar conviction scores
    if not isinstance(validated["conviction_scores"], list):
        validated["conviction_scores"] = [0.5] * num_assets
    else:
        # Asegurar cantidad correcta de scores
        scores = validated["conviction_scores"]
        try:
            scores = [max(0.0, min(1.0, float(s))) for s in scores]
        except (ValueError, TypeError):
            scores = [0.5] * num_assets
        
        # Si hay diferencia en cantidad, ajustar
        if len(scores) != num_assets:
            if len(scores) < num_assets:
                scores.extend([0.5] * (num_assets - len(scores)))
            else:
                scores = scores[:num_assets]
        
        validated["conviction_scores"] = scores
    
    return validated


# ======================= FUNCIÓN PRINCIPAL =======================

def get_llm_analysis(
    market_data: Dict[str, Any],
    base_portfolio: Dict[str, Any],
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> Dict[str, Any]:
    """
    Obtiene análisis LLM del mercado para recomendación de rebalanceo.
    
    Args:
        market_data: Diccionario con datos de mercado para cada activo
                     Estructura: {symbol: {price, rsi, trend, funding_rate, recent_news}}
        base_portfolio: Diccionario con portfolio base del modelo ML
                       Estructura: {conviction_scores, base_weights, confidence}
        temperature: Control de creatividad del modelo (0.0-1.0, default 0.7)
        max_tokens: Tokens máximos en la respuesta
    
    Returns:
        Diccionario con:
            - should_rebalance: bool
            - recommendation: str
            - conviction_scores: list
            - confidence: float
            - rationale: str
    
    Raises:
        Exception: Si hay error en la API de Gemini
    """
    
    if not GEMINI_API_KEY:
        logger.warning("API Key de Gemini no configurada, usando fallback conservador")
        num_assets = len(market_data) if market_data else 3
        return {
            "should_rebalance": False,
            "recommendation": "API no disponible, usando pesos por defecto",
            "conviction_scores": [0.5] * num_assets,
            "confidence": 0.3,
            "rationale": "Fallback a valores conservadores"
        }
    
    try:
        # Construir prompt
        prompt = _build_market_prompt(market_data, base_portfolio)
        
        logger.info("📡 Llamando a Gemini...")
        
        # Llamar a Gemini
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                },
            ]
        )
        
        # Extraer texto de la respuesta
        response_text = response.text
        logger.debug(f"Respuesta Gemini:\n{response_text}")
        
        # Parsear JSON
        parsed = _parse_json_response(response_text)
        
        if not parsed:
            logger.warning("No se pudo parsear respuesta de Gemini, usando fallback")
            num_assets = len(market_data) if market_data else 3
            return {
                "should_rebalance": False,
                "recommendation": "Error en parsing de respuesta",
                "conviction_scores": [0.5] * num_assets,
                "confidence": 0.3,
                "rationale": "Respuesta no válida"
            }
        
        # Validar y corregir
        num_assets = len(market_data) if market_data else len(base_portfolio.get("conviction_scores", [3]))
        validated = _validate_and_fix_response(parsed, num_assets)
        
        logger.info(f"✓ Análisis LLM completado - Rebalancear: {validated['should_rebalance']}")
        logger.info(f"  Confianza: {validated['confidence']:.1%}")
        logger.info(f"  Conviction Scores: {[f'{s:.2f}' for s in validated['conviction_scores']]}")
        
        return validated
        
    except Exception as e:
        logger.error(f"Error en get_llm_analysis: {e}")
        num_assets = len(market_data) if market_data else 3
        return {
            "should_rebalance": False,
            "recommendation": f"Error: {str(e)[:50]}",
            "conviction_scores": [0.5] * num_assets,
            "confidence": 0.0,
            "rationale": "Error en llamada a Gemini"
        }


def batch_llm_analysis(
    multiple_markets: list[Tuple[str, Dict[str, Any], Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Analiza múltiples portfolios en una sola sesión.
    
    Útil para backtesting o análisis de múltiples estrategias.
    
    Args:
        multiple_markets: Lista de tuplas (portfolio_name, market_data, base_portfolio)
    
    Returns:
        Diccionario {portfolio_name: analysis_result}
    """
    results = {}
    
    logger.info(f"Procesando {len(multiple_markets)} portfolios...")
    
    for name, market_data, base_portfolio in multiple_markets:
        logger.info(f"Analizando: {name}")
        results[name] = get_llm_analysis(market_data, base_portfolio)
    
    return results


# ======================= DEBUG Y TESTING =======================

if __name__ == "__main__":
    # Datos de ejemplo para testing
    example_market = {
        "cmt_btcusdt": {
            "price": 67500.50,
            "rsi": 65.2,
            "trend": "Rising",
            "funding_rate": 0.00012,
            "recent_news": ["Bitcoin se acerca a máximos históricos"]
        },
        "cmt_ethusdt": {
            "price": 3250.75,
            "rsi": 55.0,
            "trend": "Flat",
            "funding_rate": 0.00008,
            "recent_news": ["ETH consolidando después de rally"]
        },
        "cmt_solusdt": {
            "price": 210.30,
            "rsi": 72.1,
            "trend": "Rising",
            "funding_rate": 0.00015,
            "recent_news": ["SOL alcanza máximos de temporada"]
        }
    }
    
    example_portfolio = {
        "conviction_scores": [0.75, 0.68, 0.82],
        "base_weights": [0.40, 0.35, 0.25],
        "confidence": 0.73
    }
    
    # Probar función principal
    result = get_llm_analysis(example_market, example_portfolio)
    print("\n" + "=" * 60)
    print("RESULTADO DEL ANÁLISIS LLM:")
    print("=" * 60)
    print(json.dumps(result, indent=2))
