import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Importamos la lógica DE TUS MISMOS ARCHIVOS para que sea idéntico
from src.risk_manager import motor_de_riesgo
from src.black_litterman import BlackLittermanModel
# No importamos execution ni data porque los vamos a simular

# Configuración del Backtest
BACKTEST_CONFIG = {
    "initial_capital": 10000,
    "fee_rate": 0.0006,
    "data_path": "data/processed/test_data_4h.parquet",
    "model_path": "data/models/alpha_xgboost.json",
    "risk_profile": "medio_riesgo",
    "timeframe_hours": 4,
    "aggregate_turnover_threshold": 0.05,  # no rebalancear si el turnover total <5%
    "features_elite": [
        'alpha_past_neutral', 'vpt_neutral', 'zscore_neutral', 
        'rsi_neutral', 'atr_neutral', 'ret_vol_ratio_neutral'
    ]
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtester")

class VirtualExchange:
    """Simula Binance: Mantiene saldo y posiciones."""
    def __init__(self, initial_capital, fee_rate):
        self.cash = initial_capital
        self.positions = {}
        self.fee_rate = fee_rate
        self.equity_curve = []
        self.timestamps = []
        self.initial_capital = initial_capital
        self.trades = []
        self.rebalances = 0

    def update_mark_to_market(self, current_prices, timestamp):
        """Calcula el valor total actual de la cuenta."""
        portfolio_value = 0
        for ticker, qty in self.positions.items():
            price = current_prices.get(ticker, 0)
            portfolio_value += qty * price
        
        total_equity = self.cash + portfolio_value
        self.equity_curve.append(total_equity)
        self.timestamps.append(timestamp)
        return total_equity

    def execute_rebalance(self, target_weights, current_prices, leverage=1, aggregate_threshold=0.05):
        """Rebalancea cartera hacia los pesos objetivo con filtro de turnover total."""
        total_equity = self.equity_curve[-1]
        
        target_values = {
            ticker: total_equity * weight * leverage 
            for ticker, weight in target_weights.items()
        }

        # Calcular deltas primero para medir turnover total
        deltas = {}
        all_tickers = set(list(self.positions.keys()) + list(target_values.keys()))
        for ticker in all_tickers:
            price = current_prices.get(ticker)
            if not price:
                continue
            current_qty = self.positions.get(ticker, 0)
            current_val = current_qty * price
            target_val = target_values.get(ticker, 0)
            delta_val = target_val - current_val
            deltas[ticker] = delta_val

        total_turnover = sum(abs(v) for v in deltas.values())
        turnover_pct = (total_turnover / total_equity) if total_equity > 0 else 1.0

        if turnover_pct < aggregate_threshold:
            return False
        
        rebalance_trades = 0
        
        for ticker, qty in list(self.positions.items()):
            price = current_prices.get(ticker)
            if not price:
                continue
            
            current_val = qty * price
            target_val = target_values.get(ticker, 0)
            
            if target_val < current_val:
                sell_val = current_val - target_val
                sell_qty = sell_val / price
                
                self.positions[ticker] -= sell_qty
                self.cash += sell_val * (1 - self.fee_rate)
                
                self.trades.append({
                    'ticker': ticker,
                    'type': 'SELL',
                    'quantity': sell_qty,
                    'price': price,
                    'value': sell_val
                })
                rebalance_trades += 1
                
                if self.positions[ticker] < 1e-6:
                    del self.positions[ticker]

        for ticker, target_val in target_values.items():
            price = current_prices.get(ticker)
            if not price:
                continue
            
            current_qty = self.positions.get(ticker, 0)
            current_val = current_qty * price
            
            if target_val > current_val:
                buy_val = target_val - current_val
                cost = buy_val * (1 + self.fee_rate)
                
                if self.cash >= cost:
                    buy_qty = buy_val / price
                    self.positions[ticker] = current_qty + buy_qty
                    self.cash -= cost
                    
                    self.trades.append({
                        'ticker': ticker,
                        'type': 'BUY',
                        'quantity': buy_qty,
                        'price': price,
                        'value': buy_val
                    })
                    rebalance_trades += 1
        
        if rebalance_trades > 0:
            self.rebalances += 1
        return True

def mock_llm_analysis(base_portfolio):
    """Simula al LLM diciendo 'Todo OK' para acelerar el backtest."""
    return {
        "should_rebalance": True,
        "recommendation": "Backtest Simulation - Strong Buy",
        "conviction_scores": base_portfolio["conviction_scores"],
        "confidence": 0.9,
        "rationale": "Simulated LLM Approval"
    }

def run_backtest_pipeline():
    logger.info("[BACKTEST] INICIANDO...")
    
    if not os.path.exists(BACKTEST_CONFIG["data_path"]):
        raise FileNotFoundError("No se encuentra el parquet de datos.")
    
    full_df = pd.read_parquet(BACKTEST_CONFIG["data_path"])
    
    missing = [f for f in BACKTEST_CONFIG["features_elite"] if f not in full_df.columns]
    if missing:
        raise ValueError(f"Faltan features en el parquet: {missing}")

    model = xgb.XGBRegressor()
    booster = xgb.Booster()
    booster.load_model(BACKTEST_CONFIG["model_path"])
    model._Booster = booster
    
    exchange = VirtualExchange(BACKTEST_CONFIG["initial_capital"], BACKTEST_CONFIG["fee_rate"])
    
    timestamps = full_df.index.unique().sort_values()
    warmup_period = 20 
    
    logger.info(f"[BACKTEST] Periodo: {timestamps[0]} a {timestamps[-1]}")
    logger.info(f"[BACKTEST] Velas totales: {len(timestamps)}")
    
    for i, t in enumerate(timestamps):

        if i < warmup_period:
            continue
            
        current_market_slice = full_df.loc[t] 
        
        if isinstance(current_market_slice, pd.Series):
             continue
             
        current_prices = current_market_slice.set_index('ticker')['Close'].to_dict()
        equity = exchange.update_mark_to_market(current_prices, t)
        
        dtest = xgb.DMatrix(current_market_slice[BACKTEST_CONFIG["features_elite"]])
        pred_alphas = booster.predict(dtest)
        
        tickers = current_market_slice['ticker'].values
        
        min_a, max_a = pred_alphas.min(), pred_alphas.max()
        if max_a == min_a: scores = np.full(len(pred_alphas), 0.5)
        else: scores = (pred_alphas - min_a) / (max_a - min_a)
        
        base_portfolio = {
            "conviction_scores": scores.tolist(),
            "tickers": tickers.tolist(),
            "alphas": pred_alphas.tolist()
        }
        
        llm_analysis = mock_llm_analysis(base_portfolio)
            
        peak = max(exchange.equity_curve)
        current_dd = (equity - peak) / peak
        
        risk_result = motor_de_riesgo(
            perfil=BACKTEST_CONFIG["risk_profile"],
            drawdown_actual=abs(current_dd),
            lista_scores_llm=llm_analysis["conviction_scores"],
            top_n=3
        )
        
        past_data = full_df.loc[timestamps[i-20]:t]
        pivot_close = past_data.pivot_table(index=past_data.index, columns='ticker', values='Close')
        
        valid_tickers = [tick for tick in tickers if tick in pivot_close.columns]
        returns_df = pivot_close[valid_tickers].pct_change().dropna()
        
        if returns_df.empty:
            continue
            
        bl = BlackLittermanModel()
        mcaps = [1.0] * len(valid_tickers) 
    
        bl.fit(returns_df, mcaps)
        
        alpha_dict = dict(zip(base_portfolio["tickers"], base_portfolio["alphas"]))
        
        views = {}
        for ticker in valid_tickers:
            alpha = alpha_dict.get(ticker, 0)
            views[ticker] = (alpha * 0.1, 0.6)
            
        raw_weights = bl.predict(views, mode="both")
        
        exposure_str = risk_result.get("RESULTADO_FINAL_PCT", "100%")
        exposure = float(exposure_str.strip('%')) / 100.0
        final_weights = {k: v * exposure for k, v in raw_weights.items()}

        exchange.execute_rebalance(
            final_weights,
            current_prices,
            leverage=1,
            aggregate_threshold=BACKTEST_CONFIG.get("aggregate_turnover_threshold", 0.0)
        )
        
        if i % 100 == 0:
            logger.info(f"Step {i}/{len(timestamps)} | Equity: ${equity:.2f} | DD: {current_dd:.2%}")

    equity_curve = np.array(exchange.equity_curve)
    total_ret = (equity_curve[-1] - BACKTEST_CONFIG["initial_capital"]) / BACKTEST_CONFIG["initial_capital"]
    
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 6)
    
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = dd.min()
    
    periodo_inicio = exchange.timestamps[0]
    periodo_fin = exchange.timestamps[-1]
    
    if isinstance(periodo_fin, (pd.Timestamp, datetime)):
        duracion_dias = (periodo_fin - periodo_inicio).days
        periodo_str = f"{periodo_inicio} a {periodo_fin}"
    else:
        num_velas = int(periodo_fin) - int(periodo_inicio) + 1
        timeframe_hours = BACKTEST_CONFIG["timeframe_hours"]
        duracion_dias = (num_velas * timeframe_hours) / 24
        periodo_str = f"Candle {int(periodo_inicio)} a {int(periodo_fin)} (~{duracion_dias:.1f} días)"

    total_trades = len(exchange.trades)
    buy_trades = sum(1 for t in exchange.trades if t['type'] == 'BUY')
    sell_trades = sum(1 for t in exchange.trades if t['type'] == 'SELL')
    total_fees = sum(t['value'] * BACKTEST_CONFIG["fee_rate"] for t in exchange.trades)
    avg_trade_value = np.mean([t['value'] for t in exchange.trades]) if exchange.trades else 0

    print("\n" + "="*50)
    print("[BACKTEST] RESULTADOS BACKTEST PIPELINE INTEGRAL")
    print("="*50)
    print(f"Periodo: {periodo_str}")
    print(f"Capital Inicial: ${BACKTEST_CONFIG['initial_capital']:,.2f}")
    print(f"Capital Final: ${equity_curve[-1]:,.2f}")
    print(f"Retorno Total: {total_ret*100:.2f}%")
    print(f"Sharpe Ratio:  {sharpe:.2f}")
    print(f"Max Drawdown:  {max_dd*100:.2f}%")
    print("="*50)
    print("[OPERACIONES]")
    print(f"Total Rebalances: {exchange.rebalances}")
    print(f"Total Trades: {total_trades}")
    print(f"  - Buy Trades:  {buy_trades}")
    print(f"  - Sell Trades: {sell_trades}")
    print(f"Total Fees Pagadas: ${total_fees:,.2f}")
    print(f"Trade Value Promedio: ${avg_trade_value:,.2f}")
    print("="*50)

    plt.figure(figsize=(14, 7))
    plt.plot(exchange.timestamps, equity_curve, linewidth=2)
    plt.title(f"Pipeline Backtest (ML+Risk+BL) - Sharpe: {sharpe:.2f}\n{periodo_str}", fontsize=12)
    plt.xlabel("Candle", fontsize=10)
    plt.ylabel("Equity ($)", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_backtest_pipeline()