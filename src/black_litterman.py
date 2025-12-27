"""Black-Litterman Model Simplificado para Criptomonedas"""

import numpy as np
import pandas as pd

class BlackLittermanModel:
    def __init__(self, risk_aversion=3.0, tau=0.05):
        self.delta = risk_aversion  # Aversión al riesgo del mercado
        self.tau = tau              # Incertidumbre sobre el prior
        self.assets = None
        self.cov_matrix = None
        self.market_prior = None
        
    def fit(self, returns_df, market_caps):
        """
        'Entrena' el modelo calculando la matriz de covarianza 
        y los retornos implícitos del mercado (Equilibrio).
        """
        self.assets = returns_df.columns
        # 1. Calcular Matriz de Covarianza
        self.cov_matrix = returns_df.cov().values
        
        # 2. Calcular Pesos de Mercado (Normalizados)
        mcap_weights = np.array(market_caps) / sum(market_caps)
        
        # 3. Optimización Inversa: Retornos Implícitos (Pi)
        # Pi = delta * Cov * weights_mercado
        self.market_prior = self.delta * (self.cov_matrix @ mcap_weights)
        print("Modelo 'entrenado' con equilibrio de mercado.")

    def predict(self, views_dict, mode='both'):
        """
        Infiere los pesos óptimos combinando el mercado con el LLM.
        views_dict: {'Asset': (expected_return, confidence_0_to_1)}
        mode: 'long_only', 'short_only', 'both'
        """
        n = len(self.assets)
        Q = [] # Vector de retornos de las opiniones
        P = [] # Matriz que identifica a qué activo pertenece cada opinión
        omega_diag = [] # Incertidumbre de cada opinión

        for asset, (ret, confidence) in views_dict.items():
            if asset in self.assets:
                idx = list(self.assets).index(asset)
                # Crear vector fila para P (matriz de pick)
                row = np.zeros(n)
                row[idx] = 1
                P.append(row)
                Q.append(ret)
                
                # Traducir confianza (0-1) a varianza (Omega)
                # Si confianza es 1, varianza es baja. Si es 0.01, es alta.
                conf_adj = max(0.001, confidence)
                variance = (1 / conf_adj - 1) * self.tau # Simplificación de calibración
                omega_diag.append(max(1e-6, variance))

        P = np.array(P)
        Q = np.array(Q)
        Omega = np.diag(omega_diag)

        # --- FÓRMULA MAESTRA DE BLACK-LITTERMAN ---
        # Combinamos retornos de mercado (Pi) con opiniones (Q)
        term1 = np.linalg.inv(self.tau * self.cov_matrix)
        term2 = P.T @ np.linalg.inv(Omega) @ P
        
        combined_return = np.linalg.inv(term1 + term2) @ \
                         (term1 @ self.market_prior + P.T @ np.linalg.inv(Omega) @ Q)

        # --- OPTIMIZACIÓN DE PESOS ---
        # w = (1/delta) * Cov^-1 * Combined_Returns
        raw_weights = (1 / self.delta) * np.linalg.inv(self.cov_matrix) @ combined_return
        
        return self._apply_constraints(raw_weights, mode)

    def _apply_constraints(self, weights, mode):
        """Ajusta los pesos según el modo deseado"""
        if mode == 'long_only':
            weights = np.maximum(weights, 0)
        elif mode == 'short_only':
            weights = np.minimum(weights, 0)
        
        # Normalizar para que la suma absoluta no exceda 1 (o según tu política)
        if np.sum(np.abs(weights)) > 0:
            weights = weights / np.sum(np.abs(weights))
            
        return pd.Series(weights, index=self.assets)

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Datos ficticios: BTC, ETH, SOL
    data = pd.DataFrame(np.random.randn(100, 3) / 100, columns=['BTC', 'ETH', 'SOL'])
    mcaps = [600, 200, 50] # Market caps relativos
    
    bl = BlackLittermanModel()
    bl.fit(data, mcaps)
    
    # Tres escenarios de opiniones del LLM (retorno esperado, confianza 0-1)
    escenarios = [
        (
            "Escenario 1: base (alcista BTC, cauteloso SOL)",
            {
                'BTC': (0.05, 0.8),
                'SOL': (-0.02, 0.4)
            }
        ),
        (
            "Escenario 2: optimista (ETH y SOL al alza)",
            {
                'ETH': (0.07, 0.7),
                'SOL': (0.03, 0.6)
            }
        ),
        (
            "Escenario 3: defensivo (BTC a la baja)",
            {
                'BTC': (-0.04, 0.9)
            }
        ),
        (
            "Escenario 4: rotación a ETH", {
                'ETH': (0.06, 0.85),
                'BTC': (0.01, 0.5)
            }
        ),
        (
            "Escenario 5: impulso SOL", {
                'SOL': (0.08, 0.75),
                'ETH': (0.02, 0.4)
            }
        ),
        (
            "Escenario 6: toma de ganancias BTC", {
                'BTC': (-0.03, 0.7),
                'ETH': (0.01, 0.3)
            }
        ),
        (
            "Escenario 7: rally amplio", {
                'BTC': (0.04, 0.6),
                'ETH': (0.05, 0.65),
                'SOL': (0.04, 0.55)
            }
        ),
        (
            "Escenario 8: risk-off cripto", {
                'BTC': (-0.05, 0.8),
                'ETH': (-0.06, 0.8),
                'SOL': (-0.1, 0.9)
            }
        ),
        (
            "Escenario 9: volatilidad SOL", {
                'SOL': (0.12, 0.5)
            }
        ),
        (
            "Escenario 10: ETH ganador relativo", {
                'ETH': (0.04, 0.9),
                'BTC': (-0.01, 0.4)
            }
        ),
        (
            "Escenario 11: BTC líder claro", {
                'BTC': (0.08, 0.9),
                'ETH': (0.02, 0.5),
                'SOL': (0.0, 0.3)
            }
        ),
        (
            "Escenario 12: rotación a mid-cap", {
                'SOL': (0.06, 0.7),
                'ETH': (0.0, 0.4),
                'BTC': (-0.01, 0.4)
            }
        ),
        (
            "Escenario 13: ETH en rango", {
                'ETH': (0.0, 0.6)
            }
        ),
        (
            "Escenario 14: sorpresa inflacionaria (cripto sube)", {
                'BTC': (0.06, 0.7),
                'ETH': (0.07, 0.7)
            }
        ),
        (
            "Escenario 15: endurecimiento regulatorio", {
                'BTC': (-0.06, 0.85),
                'ETH': (-0.04, 0.8),
                'SOL': (-0.08, 0.85)
            }
        ),
        (
            "Escenario 16: desacople BTC vs altcoins", {
                'BTC': (0.03, 0.7),
                'ETH': (-0.02, 0.6),
                'SOL': (-0.05, 0.6)
            }
        ),
        (
            "Escenario 17: momentum ETH/SOL", {
                'ETH': (0.05, 0.75),
                'SOL': (0.07, 0.8)
            }
        ),
        (
            "Escenario 18: refugio parcial en BTC", {
                'BTC': (0.02, 0.8),
                'ETH': (-0.01, 0.5),
                'SOL': (-0.03, 0.5)
            }
        ),
        (
            "Escenario 19: lateralidad general", {
                'BTC': (0.0, 0.5),
                'ETH': (0.0, 0.5),
                'SOL': (0.0, 0.5)
            }
        ),
        (
            "Escenario 20: shock positivo tecnología", {
                'BTC': (0.05, 0.6),
                'ETH': (0.09, 0.8),
                'SOL': (0.1, 0.7)
            }
        ),
    ]

    for nombre, opiniones_llm in escenarios:
        pesos_finales = bl.predict(opiniones_llm, mode='both')
        print(f"\n{nombre} - pesos finales (mode=both):")
        print(pesos_finales)