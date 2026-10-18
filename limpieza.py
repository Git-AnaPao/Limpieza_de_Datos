import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer

# ==========================================
# 1. Generación de dataset con ruido sintético
# ==========================================
data = {
    'edad': [
        25,
        30,
        np.nan,
        45,
        22,
        120,
        35,
        28,
        np.nan,
        40,
    ],  # 120 es valor atípico/erróneo
    'ingreso_mensual': [
        15000,
        22000,
        18000,
        np.nan,
        12000,
        950000,
        25000,
        21000,
        19000,
        np.nan,
    ],  # 950k es outlier
    'scoring_credito': [650, 700, 680, 710, 590, 800, 670, 640, 620, 730],
}

# Guardamos copia del dataset original antes de modificarlo
df_original = pd.DataFrame(data)
df_dirty = df_original.copy()

print("=== DATASET ORIGINAL (CON RUIDO/NULOS) ===")
print(df_dirty)
print("-" * 50)

# ==========================================
# 2. Tratamiento de Valores Atípicos (Outliers e Inconsistencias)
# ==========================================
# Cálculo de rango intercuartílico (IQR) para ingreso_mensual
Q1 = df_dirty['ingreso_mensual'].quantile(0.25)
print(f"Q1: ${Q1:,.2f}")
Q3 = df_dirty['ingreso_mensual'].quantile(0.75)
print(f"Q3: ${Q3:,.2f}")
IQR = Q3 - Q1
print(f"IQR: ${IQR:,.2f}")
limite_superior = Q3 + 1.5 * IQR

print(f"Límite Superior para Outliers en Ingreso: ${limite_superior:,.2f}")

# Reemplazar outliers extremos de ingresos con NaN para tratarlos en la imputación
df_dirty.loc[df_dirty['ingreso_mensual'] > limite_superior, 'ingreso_mensual'] = (
    np.nan
)

# Corrección de regla de negocio: Edad mayor a 100 se convierte a NaN
df_dirty.loc[df_dirty['edad'] > 100, 'edad'] = np.nan

print("\n=== DATASET TRATADO (OUTLIERS E INCONSISTENCIAS REEMPLAZADOS POR NaN) ===")
print(df_dirty)
print("-" * 50)

# ==========================================
# 3. Visualización Gráfica: Antes y Después del Tratamiento de Outliers
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    'Detección y Tratamiento de Outliers: Antes vs Después',
    fontsize=16,
    fontweight='bold',
    y=0.98,
)

color_antes = '#e74c3c'    # Rojo suave
color_despues = '#27ae60'  # Verde suave

# --- Fila 1: Ingreso Mensual ---
# 1.1 Ingreso Mensual: Antes
sns.boxplot(ax=axes[0, 0], y=df_original['ingreso_mensual'], color=color_antes, width=0.35, showfliers=False)
sns.stripplot(ax=axes[0, 0], y=df_original['ingreso_mensual'], color='#2c3e50', alpha=0.8, jitter=0.1, size=7)
axes[0, 0].set_title('Ingreso Mensual - ANTES\n(Outlier extremo: $950,000)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Ingreso Mensual ($)', fontsize=11)
axes[0, 0].annotate(
    'Outlier: $950,000\n(Comprime la caja)',
    xy=(0, 950000),
    xytext=(0.15, 750000),
    arrowprops=dict(facecolor='darkred', edgecolor='darkred', arrowstyle='->', lw=2),
    fontsize=10,
    fontweight='bold',
    color='darkred'
)

# 1.2 Ingreso Mensual: Después
sns.boxplot(ax=axes[0, 1], y=df_dirty['ingreso_mensual'], color=color_despues, width=0.35, showfliers=False)
sns.stripplot(ax=axes[0, 1], y=df_dirty['ingreso_mensual'], color='#2c3e50', alpha=0.8, jitter=0.1, size=7)
axes[0, 1].set_title('Ingreso Mensual - DESPUÉS\n(Tratado con IQR -> NaN)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Ingreso Mensual ($)', fontsize=11)

# --- Fila 2: Edad ---
# 2.1 Edad: Antes
sns.boxplot(ax=axes[1, 0], y=df_original['edad'], color=color_antes, width=0.35, showfliers=False)
sns.stripplot(ax=axes[1, 0], y=df_original['edad'], color='#2c3e50', alpha=0.8, jitter=0.1, size=7)
axes[1, 0].set_title('Edad - ANTES\n(Inconsistencia: 120 años)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Edad (Años)', fontsize=11)
axes[1, 0].annotate(
    'Inconsistencia: 120 años\n(Fuera de rango lógico)',
    xy=(0, 120),
    xytext=(0.15, 102),
    arrowprops=dict(facecolor='darkred', edgecolor='darkred', arrowstyle='->', lw=2),
    fontsize=10,
    fontweight='bold',
    color='darkred'
)

# 2.2 Edad: Después
sns.boxplot(ax=axes[1, 1], y=df_dirty['edad'], color=color_despues, width=0.35, showfliers=False)
sns.stripplot(ax=axes[1, 1], y=df_dirty['edad'], color='#2c3e50', alpha=0.8, jitter=0.1, size=7)
axes[1, 1].set_title('Edad - DESPUÉS\n(Regla edad > 100 -> NaN)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Edad (Años)', fontsize=11)

plt.tight_layout()
plt.savefig('comparativa_outliers.png', dpi=300)
print("[+] Gráfico guardado exitosamente como 'comparativa_outliers.png'")
# Descomenta la siguiente línea si deseas abrir la ventana interactiva al ejecutar:
# plt.show()

# ==========================================
# 4. Experimentación: Variación de k en KNNImputer vs Métodos Tradicionales
# ==========================================
print("\n" + "=" * 50)
print("=== EXPERIMENTACIÓN: KNNImputer (k variable) vs MEDIA vs MEDIANA ===")
print("=" * 50)

# Imputación por Media y Mediana (métodos tradicionales univariados)
imp_media = SimpleImputer(strategy='mean').fit_transform(df_dirty)
df_media = pd.DataFrame(imp_media, columns=df_dirty.columns)

imp_mediana = SimpleImputer(strategy='median').fit_transform(df_dirty)
df_mediana = pd.DataFrame(imp_mediana, columns=df_dirty.columns)

# Imputación con KNN para distintos valores de k (k=1, k=2, k=3, k=5)
valores_k = [1, 2, 3, 5]
dict_knn = {}

for k in valores_k:
    imputer_k = KNNImputer(n_neighbors=k)
    dict_knn[f'KNN (k={k})'] = pd.DataFrame(imputer_k.fit_transform(df_dirty), columns=df_dirty.columns)

# Índices donde había valores faltantes en 'edad' e 'ingreso_mensual'
idx_edad_nulos = df_dirty[df_dirty['edad'].isna()].index
idx_ingreso_nulos = df_dirty[df_dirty['ingreso_mensual'].isna()].index

# Tabla comparativa para la columna EDAD
comp_edad = pd.DataFrame({
    'Scoring': df_dirty.loc[idx_edad_nulos, 'scoring_credito'],
    'Media': df_media.loc[idx_edad_nulos, 'edad'],
    'Mediana': df_mediana.loc[idx_edad_nulos, 'edad'],
    **{nombre: dict_knn[nombre].loc[idx_edad_nulos, 'edad'] for nombre in dict_knn}
})

print("\n--- COMPARATIVA DE VALORES IMPUTADOS: EDAD (Registros con NaN) ---")
print(comp_edad.round(2))

# Tabla comparativa para la columna INGRESO MENSUAL
comp_ingreso = pd.DataFrame({
    'Scoring': df_dirty.loc[idx_ingreso_nulos, 'scoring_credito'],
    'Media': df_media.loc[idx_ingreso_nulos, 'ingreso_mensual'],
    'Mediana': df_mediana.loc[idx_ingreso_nulos, 'ingreso_mensual'],
    **{nombre: dict_knn[nombre].loc[idx_ingreso_nulos, 'ingreso_mensual'] for nombre in dict_knn}
})

print("\n--- COMPARATIVA DE VALORES IMPUTADOS: INGRESO MENSUAL (Registros con NaN) ---")
print(comp_ingreso.round(2))

# Dataset final limpio con el modelo base (k=2)
df_clean = dict_knn['KNN (k=2)']
print("\n=== DATASET FINAL LIMPIO E IMPUTADO (KNN k=2) ===")
print(df_clean)

# ==========================================
# 5. Breve Comparativa y Conclusiones Teóricas
# ==========================================
print("\n" + "=" * 50)
print("=== RESUMEN COMPARATIVO DE MÉTODOS DE IMPUTACIÓN ===")
print("=" * 50)
print("""
1. Métodos Tradicionales (Media / Mediana):
   - Asignan un valor idéntico y estático a todas las filas faltantes (ej. Media ingreso: $18,857.14, Mediana: $19,000.00).
   - Ignoran la información de las otras variables del individuo (por ejemplo, el scoring crediticio).
   - Reducen artificialmente la variabilidad y distorsionan la covarianza entre características.

2. KNNImputer (Multivariado):
   - Imputa valores basándose en individuos con perfiles similares (distancia euclidiana sobre variables disponibles como scoring_credito).
   - Para un individuo con scoring alto (ej. 800), busca donantes cercanos en el espacio multidimensional en lugar de asignar el promedio global.

3. Efecto de modificar k en KNN:
   - k=1: Asigna exactamente el valor del vecino más próximo. Puede capturar particularidades locales, pero es muy sensible a anomalías o ruido puntual.
   - k=2 y k=3: Promedia los 2 o 3 vecinos más parecidos, logrando un balance entre especificidad local y estabilidad numérica.
   - k grande (k=5): Al incluir más vecinos dispersos, las imputaciones se suavizan y tienden a converger hacia la media del conjunto.
""")