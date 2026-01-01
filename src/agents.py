"""coger la salida de inference de ML, enviara a LLM de gemini, y auditará la reorganizacion con noticias de una api"""
import os
import json
import yfinance as yf
import google.genai as genai
from dotenv import load_dotenv
from inference import generate_signals

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


def get_latest_news(ticker, limit=5):
    """Descarga noticias recientes de Yahoo Finance para dar contexto al LLM."""
    print(f"[AGENTE] Buscando noticias para {ticker}...")
    # Yahoo usa tickers como 'BTC-USD'
    asset = yf.Ticker(ticker)
    news_list = asset.news
    
    formatted_news = ""
    count = 0
    for item in news_list:
        if count >= limit: break
        title = item.get('title', 'Sin título')
        publisher = item.get('publisher', 'Desconocido')
        # limpiamos fechas o enlaces
        formatted_news += f"- [{publisher}]: {title}\n"
        count += 1
        
    if not formatted_news:
        return "No se encontraron noticias recientes relevantes."
        
    return formatted_news


def portfolio_manager_agent(top_candidates):
    """Recibe el TOP x del modelo matemático, Busca noticias para cada uno y re-ordena la lista basándose en Riesgo Fundamental.
    """
    print("[AGENTE] Analizando el Top x para re-ranking...")
    
    # contexto con noticias para CADA candidato
    candidates_context = ""
    for item in top_candidates:
        ticker = item['ticker']
        alpha = item['alpha']
        rank = item['rank']
        
        print(f"[AGENTE] Leyendo noticias de {ticker}...")
        news = get_latest_news(ticker)
        
        candidates_context += f"""
        --- ACTIVO: {ticker} (Rank Matemático: #{rank}) ---
        Alpha Predicho: {alpha:.2e}
        Noticias Recientes:
        {news}
        ----------------------------------------------------
        """

    # reranking prompt
    
    prompt = f"""
    Eres un Gestor de Portafolio Senior.
    
    TAREA: Re-ordenar el ranking basado en riesgo fundamental.
    
    CANDIDATOS:
    {candidates_context}
    
    SALIDA (JSON ESTRICTO):
    {{
        "final_ranking": [
            {{ "rank": 1, "ticker": "AAA", "original_rank": 2, "reasoning": "..." }}
        ]
    }}
    """

    response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )
    return json.loads(response.text)


def run_system_pipeline():
    print("[ML]Ejecutando Modelo XGBoost...")
    quant_text, top_picks_list = generate_signals()
    
    top_3_candidates = top_picks_list[:3]
    
    print("[RERANK] Ejecutando Agente de Re-Ranking...")
    decision_json = portfolio_manager_agent(top_3_candidates)
    
    print("\n" + "="*60)
    print("RESULTADO FINAL DEL COMITÉ DE INVERSIÓN")
    print("="*60)
    
    final_list = decision_json["final_ranking"]
    if decision_json is None:
        print("[FATAL] Fallo del agente")
        return

    print(f"{'RANK':<5} {'TICKER':<10} {'CAMBIO':<10} {'RAZÓN'}")
    print("-" * 60)
    
    for item in final_list:
        change = "Igual"
        if item['rank'] < item['original_rank']: change = "[UP]"
        if item['rank'] > item['original_rank']: change = "[DOWN]"
        
        print(f"#{item['rank']:<4} {item['ticker']:<10} {change:<10} {item['reasoning'][:50]}...")
        
    winner = final_list[0]
    print(f"[OK] Winner: {winner['ticker']}")
    print(f"Reasoning: {winner['reasoning']}")

if __name__ == "__main__":
    run_system_pipeline()
    # for model in client.models.list():
    #     print(model)
