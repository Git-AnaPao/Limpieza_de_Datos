"""
Extensión de la Práctica 3: análisis inicial de calidad y experimento con escalado.

Genera:
  - fig_faltantes.png      : mapa de celdas faltantes/anómalas del dataset original.
  - fig_comparativa_k.png  : valores imputados por Media, Mediana y KNN (k = 1, 2, 3, 5).

Imprime además la tabla del experimento con KNNImputer sobre datos escalados (MinMax),
que sirve para discutir el efecto de las escalas distintas en la distancia euclidiana.
"""
import matplotlib

matplotlib.use("Agg")  # no abre ventanas; solo guarda archivos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------
# Datos (idénticos a limpieza.py)
# ------------------------------------------------------------------
data = {
    "edad": [25, 30, np.nan, 45, 22, 120, 35, 28, np.nan, 40],
    "ingreso_mensual": [15000, 22000, 18000, np.nan, 12000, 950000, 25000, 21000, 19000, np.nan],
    "scoring_credito": [650, 700, 680, 710, 590, 800, 670, 640, 620, 730],
}
df_original = pd.DataFrame(data)

# Mismo tratamiento que limpieza.py
Q1 = df_original["ingreso_mensual"].quantile(0.25)
Q3 = df_original["ingreso_mensual"].quantile(0.75)
limite_superior = Q3 + 1.5 * (Q3 - Q1)

df_dirty = df_original.copy()
df_dirty.loc[df_dirty["ingreso_mensual"] > limite_superior, "ingreso_mensual"] = np.nan
df_dirty.loc[df_dirty["edad"] > 100, "edad"] = np.nan

# ------------------------------------------------------------------
# 1. Figura de calidad inicial: faltantes y anómalos
# ------------------------------------------------------------------
codigo = pd.DataFrame(0, index=df_original.index, columns=df_original.columns)
codigo[df_original.isna()] = 1                                   # faltante
codigo.loc[df_original["edad"] > 100, "edad"] = 2                # anómalo
codigo.loc[df_original["ingreso_mensual"] > limite_superior, "ingreso_mensual"] = 2

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
cmap = ListedColormap(["#d5f5e3", "#f9e79f", "#f5b7b1"])
sns.heatmap(codigo, ax=axes[0], cmap=cmap, cbar=False, linewidths=1, linecolor="white",
            annot=df_original.fillna("NaN").astype(object).map(
                lambda v: v if v == "NaN" else f"{v:,.0f}"),
            fmt="", annot_kws={"fontsize": 9})
axes[0].set_title("Celdas del dataset original", fontweight="bold")
axes[0].set_ylabel("Índice de registro")
axes[0].tick_params(axis="x", rotation=15)
axes[0].legend(
    handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#d5f5e3", "#f9e79f", "#f5b7b1"]],
    labels=["Válido", "Faltante (NaN)", "Anómalo"], loc="upper center",
    bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=9)

resumen = pd.DataFrame({
    "Faltantes": df_original.isna().sum(),
    "Anómalos": [(df_original["edad"] > 100).sum(),
                 (df_original["ingreso_mensual"] > limite_superior).sum(), 0],
})
resumen.plot(kind="bar", ax=axes[1], color=["#f1c40f", "#e74c3c"], rot=15)
axes[1].set_title("Conteo por variable", fontweight="bold")
axes[1].set_ylabel("Registros")
axes[1].set_yticks(range(0, 4))
axes[1].legend(frameon=False)
plt.tight_layout()
plt.savefig("fig_faltantes.png", dpi=200)
plt.close()

# ------------------------------------------------------------------
# 2. Comparativa de imputaciones (Media, Mediana, KNN con k variable)
# ------------------------------------------------------------------
metodos = {
    "Media": pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(df_dirty), columns=df_dirty.columns),
    "Mediana": pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df_dirty), columns=df_dirty.columns),
}
for k in [1, 2, 3, 5]:
    metodos[f"KNN k={k}"] = pd.DataFrame(
        KNNImputer(n_neighbors=k).fit_transform(df_dirty), columns=df_dirty.columns)

idx_edad = df_dirty.index[df_dirty["edad"].isna()]
idx_ing = df_dirty.index[df_dirty["ingreso_mensual"].isna()]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, col, idx, titulo in [(axes[0], "edad", idx_edad, "Edad imputada (años)"),
                             (axes[1], "ingreso_mensual", idx_ing, "Ingreso mensual imputado ($)")]:
    tabla = pd.DataFrame({m: d.loc[idx, col] for m, d in metodos.items()})
    tabla.index = [f"Reg. {i} (score {df_dirty.loc[i, 'scoring_credito']})" for i in idx]
    colores = sns.color_palette("viridis", len(tabla.columns))
    marcas = ["s", "D", "o", "o", "o", "o"]
    for j, m in enumerate(tabla.columns):
        ax.scatter(np.arange(len(tabla)) + (j - 2.5) * 0.09, tabla[m], s=70, color=colores[j],
                   marker=marcas[j], label=m, edgecolor="white", zorder=3)
    ax.set_xticks(np.arange(len(tabla)))
    ax.set_xticklabels(tabla.index)
    ax.set_xlim(-0.5, len(tabla) - 0.5)
    ax.set_title(titulo, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    margen = (tabla.values.max() - tabla.values.min()) * 0.25
    ax.set_ylim(tabla.values.min() - margen, tabla.values.max() + margen * 2)
plt.tight_layout()
plt.savefig("fig_comparativa_k.png", dpi=200)
plt.close()

# ------------------------------------------------------------------
# 3. Experimento adicional: KNNImputer con variables escaladas (MinMax)
# ------------------------------------------------------------------
escalador = MinMaxScaler()
X = escalador.fit_transform(df_dirty)  # MinMaxScaler ignora los NaN al ajustar

print("=== KNNImputer con datos escalados (MinMax) ===")
filas = []
for k in [1, 2, 3, 5]:
    Z = escalador.inverse_transform(KNNImputer(n_neighbors=k).fit_transform(X))
    z = pd.DataFrame(Z, columns=df_dirty.columns)
    for i in idx_edad:
        filas.append({"k": k, "registro": i, "variable": "edad", "imputado": round(z.loc[i, "edad"], 2)})
    for i in idx_ing:
        filas.append({"k": k, "registro": i, "variable": "ingreso_mensual",
                      "imputado": round(z.loc[i, "ingreso_mensual"], 2)})
resultado = pd.DataFrame(filas).pivot_table(index=["variable", "registro"], columns="k", values="imputado")
print(resultado)
