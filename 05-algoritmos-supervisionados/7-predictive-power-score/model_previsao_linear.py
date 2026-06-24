import pandas as pd
import matplotlib.pyplot as plt
import ppscore as pps
import seaborn as sns
from sklearn.linear_model import LinearRegression


# 1: Carregar dados
# Mesmo dataset de colesterol mas investigando PPS
print("Lendo dados:")
df_colesterol = pd.read_csv("./dataset/colesterol.csv")
df_colesterol.info()
print(df_colesterol.describe()) # médias e medianas próximas
print(df_colesterol.head(5))

#--------------------------------------------------------------------------
# 2: EDA - Análise exploratória de dados
print("\nAnálise exploratória dos dados:")
df_eda = df_colesterol.copy()

# One Hot Encoding em Grupo Sanguíneo e Fumante, pois não seguem uma ordem.
df_eda.drop(columns=["Id"], inplace=True)
df_eda = pd.get_dummies(df_eda, columns=['Grupo Sanguíneo', 'Fumante', 'Nível de Atividade'])
df_eda.info()

# Calcular PPS
pps_matrix = pps.matrix(df=df_eda)

# Ajustar matriz para plot -> inverter a mariz na coluna x, y, e ppscore
# eixo y = "é previsto por", x = "prevê".
# A linha indica quanto cada x é capaz de prever y.
pps_matrix_pivo = pps_matrix[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore') # type: ignore

plt.figure(figsize=(16,14))
sns.heatmap(
  pps_matrix_pivo,
  annot=True,
  vmax=1,
  vmin=0,
  cmap="Purples",
)
plt.ylabel("Explicado")
plt.xlabel("Explica")
plt.title("Heatmap Predictive Power Score (PPS)")
plt.savefig("./dataviz/previsao/ppscores.png")
plt.close()

# Calcular pps entre 2 variáveis específicas
pps_peso_colesterol = pps.score(
  df_eda, 'Peso', 'Colesterol'
)
print(pps_peso_colesterol)

# Compara matriz de correlação de Pearson
# A matriz indica correlação em muitas variáveis, sobretudo peso e se é fumante.
corr_matrix = df_eda.corr('pearson')
plt.figure(figsize=(16,14))
sns.heatmap(
  corr_matrix,
  annot=True,
  vmax=1,
  vmin=-1,
  cmap="coolwarm",
)
plt.title("Heatmap Correlação de Pearson")
plt.savefig("./dataviz/previsao/corr-pearson.png")
plt.close()

# Um resyultado dessa correlação é que os fumantes tem um shift à direita no histograma
# |-- Fumantes tem maior tendência a ter colesterol alto.
# |-- Os dados estão desbalanceados, mas a curva de fumantes é mais à direita, enquanto
# |-- a de não fumantes tem valores mais concentrados em colesterol mais baixo.
sns.histplot(
  data=df_eda,
  x='Colesterol',
  hue='Fumante_Sim',
  palette="coolwarm"
)
plt.ylabel("Frequência")
plt.savefig("./dataviz/previsao/colesterol-fumante-histplot.png")
plt.close()

# Peso e Colesterol fazem uma relação grosseiramente "diretamente proporcional"
# No scatter é possível ver que quando o colesterol é mais elevado, há concentração de fumantes.
sns.scatterplot(
  data=df_eda,
  y='Colesterol',
  x='Peso',
  hue='Fumante_Sim',
  palette="coolwarm"
)
plt.xlabel("Peso")
plt.ylabel("Colesterol")
plt.savefig("./dataviz/previsao/colesterol-peso-scatter.png")
plt.close()

# -------------------------------------------
# 3: Treinar modelo Linear - Regressão
X = df_eda.drop(columns='Colesterol')
y = df_eda['Colesterol']

model_reg = LinearRegression()
model_reg.fit(X, y)

# Avaliar a importância das features com base nos coeficientes do modelo
# Sangue AB e altura obtiveram coeficientes muito altos no modelo de regressão.
# |-- Altura não faz sentido pelo PPS e pela correlação.
# |-- AB também não faz tanto sentido, já que o sangue mais influente teoricamente era O.
feat_importance_reg = pd.Series(model_reg.coef_, index=X.columns)
feat_importance_reg.plot(kind='barh')
plt.xlabel("Importância")
plt.ylabel("Feature")
plt.title("Importância das Features na Regressão")
plt.savefig("./dataviz/previsao/importancia-das-features-na-rl.png")
plt.close()