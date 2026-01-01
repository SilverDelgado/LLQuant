import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from scipy.stats import probplot
import glob
import os

os.makedirs('data/processed/img', exist_ok=True)
files = glob.glob("data/processed/training_data_*.parquet")

df = pd.read_parquet(files[0])

def quick_validation(df):
    """
    Hacemos 3 checks:
        Alpha Autocorrelation (Memoria): El valor de mi Target de HOY se parece sospechosamente al de AYER? Debe ser 0: 
        Si tu Target tiene memoria (ej: 0.9), significa que no estás prediciendo un cambio de precio, estás prediciendo una tendencia que ya existe.
        < 0.05: Perfecto. El movimiento es independiente.
        > 0.10: Peligro. Tu target está sucio.

        Variation de Features:
        Mide si los indicadores cambian entre las monedas en un mismo momento.
        Por qué: Si todas las monedas tienen el mismo valor (variación 0), la neutralización falló. 
        Necesitas dispersión para poder elegir una ganadora y una perdedora.


        Test ADF (Estacionariedad)
        Prueba estadística dura.
        La media y la volatilidad son constantes en el tiempo.
        p-value < 0.05 es el objetivo
    """
    print("=== VALIDACIÓN RÁPIDA ===")
    
    # 1. Alpha no debe tener autocorrelación (si no, hay leak)
    alpha_ac = df['TARGET_ALPHA'].autocorr(lag=1)
    print(f"Alpha Autocorr(lag=1): {alpha_ac:.4f} (debe ser < 0.05)")
    if abs(alpha_ac) > 0.05:
        print("[WARNING]: Alpha tiene memoria temporal, posible lookahead bias")
    
    # 2. Features deben variar entre tickers (si no, neutralización falló)
    feature_cols = [c for c in df.columns if c.endswith('_neutral')]
    variation = df.groupby('timestamp')[feature_cols].std().mean()
    print(f"\nVariation media de features: {variation.mean():.4f} (debe ser > 0.1)")
    
    # 3. Alpha debe ser estacionario (ADF test)
    adf_stat, p_value, _, _, _, _ = adfuller(df['TARGET_ALPHA'].dropna())
    print(f"\nADF p-value: {p_value:.6f} (debe ser < 0.05)")
    if p_value < 0.05:
        print("[OK] Alpha es estacionario")
    else:
        print("[FATAL]Alpha no es estacionario, revisa FracDiff")

    return alpha_ac, variation.mean()

# RANK IC ------ (Correlación con Target)
"""
Spearman > 0.03: Feature tiene señal débil pero usable
Spearman > 0.05: Feature fuerte
Spearman < 0.01: Basura, eliminar en Fase 2
Signo: Debe ser lógico (ej. rsi_neutral positivo = overbought = Alpha negativo)
"""

def analyze_feature_signal(df):
    """
    calculamos el Information Coefficient (IC)
        Pearson: Busca líneas rectas perfectas.
        Spearman (importante): Busca Ranking. 
            "¿Si mi indicador dice que BTC es el nº1, BTC termina siendo el nº1 (o de los mejores)?". 
            Esto es lo que usa el modelo de ML.

            0.00 - 0.01: Ruido. El indicador no sirve. Borrar.
            0.02 - 0.05: Oro puro. Parece poco, pero en trading es una ventaja enorme.
            > 0.10: Sospechoso. "Demasiado bueno para ser verdad". Revisa si hay errores.
            Negativo: Si es -0.05 es BUENO. Significa que funciona al revés (indicador alto = precio baja). El modelo aprenderá a invertirlo.
    """
    feature_cols = [c for c in df.columns if c.endswith('_neutral')]
    
    print("\n=== CORRELACIÓN FEATURES vs ALPHA ===")
    correlations = {}
    
    for col in feature_cols:
        # Correlación normal (screening rápido)
        corr_pearson = df[col].corr(df['TARGET_ALPHA'])
        corr_spearman = spearmanr(df[col], df['TARGET_ALPHA'])[0]
        
        correlations[col] = {
            'pearson': corr_pearson,
            'spearman': corr_spearman
        }
        
        print(f"{col:20s} | Pearson: {corr_pearson:7.4f} | Spearman: {corr_spearman:7.4f}")
    
    # Identificar mejores features
    best_spearman = max(correlations.items(), key=lambda x: abs(x[1]['spearman']))
    print(f"[FEATURES] Mejor feature: {best_spearman[0]} (Spearman: {best_spearman[1]['spearman']:.4f})")
    
    return correlations

"""
Una correlación alta en todo el dataset puede ser fraude de régimen. 
Std(RankIC) < 0.05: Feature es estable y usará poco
Std(RankIC) > 0.08: Feature es esquizofrénica (señal aparece/desaparece)
"""

def temporal_stability(df, window=1000):
    """
    Calcula RankIC rolling para cada feature
    Si el RankIC es estable, la feature es robusta
    Imagina un indicador que en 2020 tuvo una correlación de 0.20 (increíble) y en 2021, 2022 y 2023 tuvo 0.00. La media te saldrá 0.05 (parece bueno).

    < 0.05: El indicador es robusto y fiable año tras año.
    > 0.10: El indicador es "esquizofrénico"; A veces genio, a veces idiota.

    Mean/Media: Debe ser casi 0.00.
    Std/Desviación: Debe estar controlada (cerca de 1).
    Bias/Sesgo: Debe ser bajo. Si es muy alto, significa que tienes muchas más subidas que bajadas (o viceversa), lo cual no es realista a largo plazo.
    Kurtosis/Colas: En cripto suelen ser altas (muchos eventos extremos). El clip ayuda a reducir esto.
    """
    print("\n=== ESTABILIDAD TEMPORAL (Rolling RankIC) ===")
    feature_cols = [c for c in df.columns if c.endswith('_neutral')]
    
    # Tomar un feature como ejemplo (el mejor según análisis previo)
    best_feature = 'zscore_neutral'  # Cambia según tu mejor feature
    
    rolling_rankic = []
    for i in range(window, len(df), window//4):  # Ventanas overlap
        subset = df.iloc[i-window:i]
        if len(subset) < 100:
            continue
        rankic = spearmanr(subset[best_feature], subset['TARGET_ALPHA'])[0]
        rolling_rankic.append(rankic)
    
    rolling_rankic = pd.Series(rolling_rankic)
    stability = rolling_rankic.std()
    
    print(f"Feature: {best_feature}")
    print(f"RankIC medio: {rolling_rankic.mean():.4f}")
    print(f"RankIC std:   {stability:.4f} (debe ser < 0.05 para ser estable)")
    
    if stability < 0.05:
        print("[OK] Feature ESTABLE")
    else:
        print("[WARN] Feature INESTABLE (puede sobreajustar)")
    
    return rolling_rankic

#  Distribución del Alpha

"""
Std: 0.05-0.15 es saludable
Skew: < 0.5 (no muy sesgado)
Kurtosis: > 5 (colas gordas, normal en crypto)
Outliers: < 1% si clips a 3σ
"""


def analyze_alpha_distribution(df):
    print("\n=== DISTRIBUCIÓN DEL ALPHA ===")
    
    plt.figure(figsize=(12, 5))
    
    # Histograma
    plt.subplot(1, 2, 1)
    df['TARGET_ALPHA'].hist(bins=50, alpha=0.7)
    plt.title('Distribución Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Frecuencia')
    
    # QQ-Plot (para ver si es normal)
    plt.subplot(1, 2, 2)
    probplot(df['TARGET_ALPHA'], dist="norm", plot=plt)
    plt.title('QQ-Plot vs Normal')
    
    plt.tight_layout()
    plt.savefig('data/processed/img/alpha_distribution.png')
    plt.close()
    print("[IMG] Gráfico guardado: alpha_distribution.png")
    
    # Estadísticas
    print(f"\nAlpha Stats:")
    print(f"  Mean:  {df['TARGET_ALPHA'].mean():.6f}")
    print(f"  Std:   {df['TARGET_ALPHA'].std():.4f}")
    print(f"  Skew:  {df['TARGET_ALPHA'].skew():.3f}")
    print(f"  Kurt:  {df['TARGET_ALPHA'].kurtosis():.3f} (normal=3.0)")
    
    # % de outliers
    outliers = (df['TARGET_ALPHA'] >= 3.0).sum() + (df['TARGET_ALPHA'] <= -3.0).sum()
    print(f"  Outliers clipped: {outliers} ({outliers/len(df)*100:.2f}%)")

# Alpha por Ticker Importante con pocos activos (8)

"""
Std similar: Todos los tickers deben tener volatilidad de Alpha parecida
Count similar: No debe haber tickers con <50% de datos
"""

def alpha_by_ticker(df):
    """
    verifica que no hayamos creado un monstruo que solo sabe operar x activo, Mean/Std: Deben ser parecidos entre monedas.
    """
    print("\n=== ALPHA POR TICKER ===")
    
    stats = df.groupby('ticker')['TARGET_ALPHA'].agg(['mean', 'std', 'count'])
    stats = stats.sort_values('std', ascending=False)
    
    print(stats)
    
    # Verificar que todos contribuyen
    if stats['count'].std() > 10:  # Si hay mucha diferencia en # de muestras
        print("[WARN]: Algunos tickers tienen muchos menos datos")
    
    # Test de homogeneidad: el Alpha debe ser similar entre tickers
    # (si no, tu modelo aprenderá solo a operar 1-2 activos)
    print(f"\nHomogeneidad (std de medias): {stats['mean'].std():.4f}")
    if stats['mean'].std() > 0.005:
        print("[WARN]: Alpha no es homogéneo entre tickers")


# MATRIZ DE CORRELACIÓN FEATURE-FEATURE

def feature_correlation_heatmap(df):
    """
    Compara Features entre sí, NO con el objetivo, queremos opiniones distintas.

    Si tienes RSI y Stoch y su correlación es 0.95, significa que dicen exactamente lo mismo.

    < 0.70: Indicadores independientes. Bien.
     > 0.85: Redundancia crítica. Debes elegir uno de los dos y borrar el otro antes de entrenar.
    """
    feature_cols = [c for c in df.columns if c.endswith('_neutral')]
    
    # Si tienes 8 features, la matriz es 8x8
    corr_matrix = df[feature_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('data/processed/img/feature_correlation.png')
    plt.close()
    print("[IMG] Gráfico guardado: feature_correlation.png")
    
    # Identificar pares muy correlacionados (>0.8)
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr:
        print("[WARN] PARES ALTAMENTE CORRELACIONADOS (>0.8):")
        for f1, f2, corr in high_corr:
            print(f"  {f1} - {f2}: {corr:.3f}")
    else:
        print("[OK] No hay features redundantes")




if __name__ == "__main__":
    print("=" * 60)
    print("ANALYSIS START")

    validation_results = quick_validation(df)

    corr_results = analyze_feature_signal(df)

    stability_series = temporal_stability(df)

    alpha_distribution = analyze_alpha_distribution(df)

    alpha_por_ticker = alpha_by_ticker(df)

    feature_correlation = feature_correlation_heatmap(df)

    print("=" * 60)
