LLQuant es un sistema hibrido de gestion cuantitativa que combina ML para generar alphas relativos, LLM para auditoria contextual y un motor de riesgo para ejecutar rebalanceos reales. El entrypoint operativo es main.py; el dashboard en tiempo real vive en gui.py.

## Caracteristicas clave
- Pipeline end-to-end: adquisicion de datos, senales ML cross-sectional, validacion LLM, controles de riesgo, optimizacion Black-Litterman y ejecucion en mercado.
- Enfoque de alpha relativo: prioriza rendimiento vs. promedio del mercado para mantener estacionariedad y robustez en bull y bear markets.
- Controles de riesgo integrados: perfil configurable, limite de turnover agregado y leverage fijo configurable.
- Observabilidad en consola: dashboard Rich para balance, precios y posiciones; logging estructurado en el loop principal.
- Extensible: modulos separados para data, ingestion, ejecucion, riesgo y agentes en src/.

## Arquitectura y pipeline
La arquitectura sigue un flujo cíclico y desacoplado por capas, con contratos claros entre etapas para que cada módulo pueda evolucionar de forma independiente.

1) Ingesta y features (src/ingestion.py, src/data.py): recolecta OHLCV, métricas técnicas, noticias y datos macro; construye payloads estructurados/no estructurados y los normaliza (incluye diferenciación fraccional y features normalizados).
2) ML cuantitativo (src/inference.py, modelos en data/models/): generate_signals ejecuta inferencia cross-sectional (alphas relativos) y devuelve ranking + contexto para el resto del pipeline. Se prioriza estacionariedad y robustez en bull/bear markets.
3) Auditoría LLM (src/llm.py): get_llm_analysis audita el ranking, ajusta conviction, detecta riesgos fundamentales y decide si rebalancear; fallback determinista si el LLM falla.
4) Controles de riesgo (src/risk_manager.py): motor_de_riesgo calibra exposición según perfil/drawdown y límites configurables (top_n, perfil, etc.).
5) Optimización (src/black_litterman.py): combina views cuant/LLM con volatilidades y proxies de liquidez (volumen) para obtener pesos; soporta modos long_only/short_only/both.
6) Ejecución (src/execution.py): rebalance_portfolio aplica pesos normalizados, respeto de aggregate_turnover_threshold y leverage fijo; get_credentials abstrae el acceso a claves.
7) Monitorización (gui.py): dashboard Rich en tiempo real con cuenta, precios y posiciones; configurable vía GUI_SYMBOLS/GUI_REFRESH_SECONDS.

### 1. Fase de ingesta
- Se ejecuta de forma cíclica (loop principal en [main.py](main.py)), con una cadencia configurable (check_interval).
- Recolecta datos crudos:
	- Precios: velas OHLCV del universo de símbolos configurado.
	- Noticias: titulares recientes por activo (cuando están disponibles) vía los helpers de [src/ingestion.py](src/ingestion.py) y [src/noticias.py](src/noticias.py).
	- Datos macro: proxies de riesgo como índices globales, volatilidad y sentimiento (cuando estén configurados).
- En [src/data.py](src/data.py) y [src/processing.py](src/processing.py) se construyen los features:
	- Diferenciación fraccional aproximada sobre precios para mantener memoria de largo plazo sin romper estacionariedad.
	- Indicadores técnicos normalizados (RSI, volatilidad relativa, bandas de Bollinger, etc.).
	- Limpieza y normalización del texto de noticias para consumo posterior del LLM.

### 2. Fase cuantitativa (ML)
- Objetivo: construir una base matemática estable detectando ineficiencias estadísticas de forma cross-sectional.
- Input X:
	- Vector de features numéricos por activo/instante (serie temporal transformada, indicadores normalizados, métricas de volatilidad y volumen).
- Target Y:
	- Retorno excedente a corto plazo (alpha relativo frente al promedio del universo), no el precio absoluto.
- En [src/inference.py](src/inference.py) se carga el modelo (por defecto XGBoost en [data/models/alpha_xgboost.json](data/models/alpha_xgboost.json)) y se ejecuta generate_signals:
	- Devuelve un ranking ordenado de activos (1, 2, 3, …) según su alpha esperado.
	- Devuelve también un vector de views numéricas (retornos relativos esperados) que se usará más tarde en Black–Litterman.
- Este diseño hace que el modelo sea robusto en bull y bear markets: el sistema se centra en diferenciales respecto a la media (alpha) y no en niveles absolutos de precio.

### 3. Fase de agentes (LLM / MoE)
- Objetivo: añadir contexto cualitativo y de riesgo que no está en los precios.
- Módulos principales: [src/agents.py](src/agents.py) y [src/llm.py](src/llm.py).
- Nivel 1 – Auditor de señales por activo:
	- Recibe el ranking cuantitativo ("SOL es la oportunidad #1", etc.) junto con las últimas noticias del activo.
	- El agente LLM actúa como analista de riesgos: valida si hay eventos fundamentales que invaliden la señal numérica.
	- Produce un conviction score continuo (0.0–1.0) que modula la fuerza de cada view cuantitativa.
- Nivel 2 – Agente de riesgo macro:
	- Consume datos macro/sentimiento y resume el régimen de mercado (risk-on vs. risk-off).
	- Devuelve un parámetro de aversión al riesgo δ que se conecta con el motor de riesgo y/o Black–Litterman para hacer la cartera más defensiva u ofensiva.

### 4. Fase de optimización y ejecución
- Optimización (modelo de riesgo y Black–Litterman):
	- En [src/risk_manager.py](src/risk_manager.py) se calibran límites de exposición, top_n de activos, perfil de riesgo y drawdown.
	- En [src/black_litterman.py](src/black_litterman.py) se combinan:
		- Views cuantitativas (alphas relativos).
		- Conviction de los agentes LLM.
		- Parámetros de aversión al riesgo y volatilidades/volúmenes.
	- El resultado son pesos objetivo matemáticamente consistentes para cada activo (por ejemplo {"BTC": 0.22, "ETH": 0.18, …}).
- Ejecución:
	- [src/execution.py](src/execution.py) toma los pesos objetivo y el estado actual de la cartera, respeta el aggregate_turnover_threshold y aplica un leverage fijo.
	- Ejecuta órdenes reales a través de la API de Weex definida en [api/](api/), utilizando credenciales abstraídas por get_credentials.
	- El loop vuelve al inicio y el sistema se mantiene en rebalanceo continuo.

Componentes principales:
- Pipeline: [main.py](main.py)
- Dashboard: [gui.py](gui.py)
- Logica de negocio: modulos en [src/](src/) (data, ingestion, processing, risk_manager, execution, etc.).

## Requisitos
- Python 3.12+
- Dependencias declaradas en pyproject.toml (numpy, pandas, xgboost, scikit-learn, rich, google-genai, etc.).

## Configuracion
Credenciales Weex (necesarias para ejecucion y dashboard):
- WEEX_API_KEY
- WEEX_SECRET_KEY
- WEEX_PASSPHRASE
- WEEX_LOCALE (opcional, por defecto en-US)

Ajustes rapidos via variables de entorno:
- GUI_SYMBOLS: lista separada por comas de simbolos a monitorear (por defecto ALLOWED_SYMBOLS de api).
- GUI_REFRESH_SECONDS: segundos entre refrescos del dashboard (default 5).
- API_Key/secret_key/passphrase: alias alternativos soportados en gui.py.

Parametros principales en main.py (CONFIG):
- symbols: universo de trading (por defecto ALLOWED_SYMBOLS de api).
- risk_profile: perfil de riesgo para motor_de_riesgo (ej. medio_riesgo).
- aggregate_turnover_threshold: rebalancear solo si el turnover agregado supera el umbral (ej. 0.05 = 5%).
- check_interval: cadencia del loop en segundos.
- execution_mode: both | long_only | short_only.
- default_leverage: multiplicador fijo aplicado a las posiciones.