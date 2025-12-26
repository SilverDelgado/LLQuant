from gnews import GNews
from datetime import datetime

def get_market_news(symbol="Bitcoin", max_results=5):
    try:
        # Mantenemos 12h para asegurar frescura
        google_news = GNews(language='es', period='12h', max_results=max_results)
        news = google_news.get_news(f'{symbol} crypto market')
        
        formatted_news = []
        for item in news:
            # Limpiamos el título
            title = item['title'].split(' - ')[0]
            # Extraemos la fecha (formato original: 'Tue, 16 Feb 2021 11:50:43 GMT')
            date = item.get('published date', 'Fecha desconocida')
            
            formatted_news.append({
                "title": title,
                "published_at": date
            })
        
        # Ordenamos por fecha (de más reciente a más antigua)
        formatted_news.sort(key=lambda x: datetime.strptime(x['published_at'], '%a, %d %b %Y %H:%M:%S %Z') 
                           if x['published_at'] != 'Fecha desconocida' else datetime.min, 
                           reverse=True)
        
        return formatted_news
    except Exception as e:
        return [{"error": f"Error obteniendo noticias: {str(e)}"}]
    
if __name__ == "__main__":
    noticias = get_market_news("bitcoin", max_results=10)
    for idx, n in enumerate(noticias, 1):
        # Imprimimos ambos datos para que el LLM vea la relevancia temporal
        print(f"{idx}. [{n['published_at']}] {n['title']}")