import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import ppscore as pps

# 1: Ler csv
# Dados passaram por um Scaler Normal
# Consiste em ver qual fruta está boa e qual não.
# transformar quality ('good', 'bad') -> (1, 0)
# Remover Id
df_frutas = pd.read_csv("./dataset/fruits.csv")
df_frutas.info()
print(df_frutas.head(10))
df_frutas.drop(columns=['A_id'], inplace=True)
df_frutas['Quality'] = df_frutas['Quality'].map({ 'good': 1, 'bad': 0})

print("Dataframe transformado:")
print(df_frutas.head(10))

# 2: Predictive Power Score x Correlação de Pearson

# Calcular PPS
pps_matrix = pps.matrix(df=df_frutas)

# Ajustar matriz para plot -> inverter a mariz na coluna x, y, e ppscore
# eixo y = "é previsto por", x = "prevê".
# A linha indica quanto cada x é capaz de prever y.
# Nenhuma variável tem um poder grande para explicar Quality
pps_matrix_pivo = pps_matrix[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore') # type: ignore
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
plt.savefig("./dataviz/classificacao/ppscores.png")
plt.close()

# Compara matriz de correlação de Pearson
# A matriz indica correlação em muitas variáveis, sobretudo peso e se é fumante.
# Correlação dos dados é muito baixa
corr_matrix = df_frutas.corr('pearson')
sns.heatmap(
  corr_matrix,
  annot=True,
  vmax=1,
  vmin=-1,
  cmap="coolwarm",
)
plt.title("Heatmap Correlação de Pearson")
plt.savefig("./dataviz/classificacao/corr-pearson.png")
plt.close()

# -------------------------------------------
# 3: Treinar modelo Linear - Regressão
X = df_frutas.drop(columns='Quality')
y = df_frutas['Quality']

model_lr = LogisticRegression()
model_lr.fit(X, y)

# Avaliar a importância das features com base nos coeficientes do modelo
# Tamanho, doçura e sculência sobressaíram, embora não expressem bem o problema.
feat_importance_reg = pd.Series(model_lr.coef_[0], index=X.columns)
feat_importance_reg.plot(kind='barh')
plt.xlabel("Importância")
plt.ylabel("Feature")
plt.title("Importância das Features na Regressão")
plt.savefig("./dataviz/classificacao/importancia-das-features-na-rl.png")
plt.close()
