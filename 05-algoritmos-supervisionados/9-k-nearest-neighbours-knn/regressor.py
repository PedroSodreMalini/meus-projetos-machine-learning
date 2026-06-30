import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, learning_curve, LearningCurveDisplay
from sklearn.metrics import f1_score, root_mean_squared_error
import numpy as np

# 1: Ler dados
df_colesterol = pd.read_csv("./dataset/colesterol.csv")
df_colesterol.drop(columns=['Id'], inplace=True)
df_colesterol = pd.get_dummies(df_colesterol, columns=['Grupo Sanguíneo', 'Nível de Atividade', 'Fumante'])


# 2: Treinar modelo
X = df_colesterol.drop(columns=['Colesterol'])
y = df_colesterol['Colesterol']

model_reg = LinearRegression()

# Gerar scores da curva de aprendizado de regressão
trains_size, train_scores, test_scores = learning_curve( # type: ignore
  model_reg,
  X,
  y,
  train_sizes=np.linspace(0.1, 0.9, 9),
  scoring="neg_mean_squared_error",
  cv=3,
)

# Gerar plot com curva de aprendizado
sns.lineplot(
  y=np.mean(train_scores, axis=1),
  x=np.linspace(0.1, 0.9, 9),
  color="blue",
  label="Treino"
)
sns.lineplot(
  y=np.mean(test_scores, axis=1),
  x=np.linspace(0.1, 0.9, 9),
  color="red",
  label="Teste"
)
plt.title("Curva de Aprendizado - Regressão Linear")
plt.xlabel("Train Size")
plt.ylabel("Negative Mean Squared Error (-MSE)")
plt.savefig("./dataviz/curva-de-aprendizado.png")
plt.close()