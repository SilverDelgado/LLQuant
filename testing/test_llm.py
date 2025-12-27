"""
Script de Testing - Verifica que llm.py funciona correctamente

Uso:
    python test_llm.py
"""

import sys
import os
import json

# Agregar src al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm import get_llm_analysis


def test_llm_basic():
    """Test básico de la función LLM"""
    
    print("\n" + "=" * 70)
    print("TEST 1: Análisis básico de mercado")
    print("=" * 70)
    
    market_data = {
        "cmt_btcusdt": {
            "price": 67500.50,
            "rsi": 65.2,
            "trend": "Rising",
            "funding_rate": 0.00012,
            "recent_news": ["Bitcoin alcanza máximos históricos"]
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
    
    portfolio = {
        "conviction_scores": [0.75, 0.68, 0.82],
        "base_weights": [0.40, 0.35, 0.25],
        "confidence": 0.73
    }
    
    result = get_llm_analysis(market_data, portfolio)
    
    print("\n📊 Resultado del análisis:")
    print(json.dumps(result, indent=2))
    
    # Validaciones
    assert "should_rebalance" in result, "Falta campo should_rebalance"
    assert isinstance(result["should_rebalance"], bool), "should_rebalance debe ser bool"
    assert "recommendation" in result, "Falta campo recommendation"
    assert "conviction_scores" in result, "Falta campo conviction_scores"
    assert len(result["conviction_scores"]) == 3, "Debe haber 3 conviction scores"
    assert "confidence" in result, "Falta campo confidence"
    assert 0.0 <= result["confidence"] <= 1.0, "Confidence debe estar entre 0 y 1"
    
    print("\n✅ Todas las validaciones pasaron")
    return True


def test_llm_empty_market():
    """Test con datos vacíos (fallback)"""
    
    print("\n" + "=" * 70)
    print("TEST 2: Análisis con mercado vacío (fallback)")
    print("=" * 70)
    
    result = get_llm_analysis({}, {})
    
    print("\n📊 Resultado con datos vacíos:")
    print(json.dumps(result, indent=2))
    
    assert result["confidence"] <= 0.5, "Confianza debe ser baja en fallback"
    assert not result["should_rebalance"], "No debe rebalancear en fallback"
    
    print("\n✅ Fallback funcionando correctamente")
    return True


def test_llm_response_validation():
    """Test de validación de respuestas"""
    
    print("\n" + "=" * 70)
    print("TEST 3: Validación de respuestas")
    print("=" * 70)
    
    market_data = {
        "cmt_btcusdt": {"price": 67500, "rsi": 65, "trend": "Rising", "funding_rate": 0.0001, "recent_news": ["BTC sube"]},
        "cmt_ethusdt": {"price": 3250, "rsi": 55, "trend": "Flat", "funding_rate": 0.00008, "recent_news": ["ETH estable"]},
    }
    
    portfolio = {"conviction_scores": [0.75, 0.68], "base_weights": [0.5, 0.5], "confidence": 0.7}
    
    result = get_llm_analysis(market_data, portfolio)
    
    # Verificar que los scores están en rango válido
    for score in result["conviction_scores"]:
        assert 0.0 <= score <= 1.0, f"Score {score} fuera de rango [0, 1]"
    
    # Verificar que hay el número correcto de scores
    assert len(result["conviction_scores"]) == 2, f"Se esperaban 2 scores, se obtuvieron {len(result['conviction_scores'])}"
    
    print(f"✓ Conviction scores válidos: {result['conviction_scores']}")
    print(f"✓ Confidence válido: {result['confidence']:.2%}")
    print(f"✓ Should rebalance: {result['should_rebalance']}")
    
    print("\n✅ Validación de respuestas completada")
    return True


if __name__ == "__main__":
    print("\n" + "🧪 INICIANDO TESTS DE LLM" + "\n")
    
    tests = [
        ("Test básico", test_llm_basic),
        ("Test mercado vacío", test_llm_empty_market),
        ("Test validación", test_llm_response_validation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"\n❌ {test_name} falló: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {test_name} error: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTADOS: {passed} pasaron, {failed} fallaron")
    print("=" * 70)
    
    sys.exit(0 if failed == 0 else 1)
