import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Setar modelo
df_colesterol = pd.read_csv("./dataset/colesterol.csv")
df_colesterol.drop(columns=['Id'], inplace=True)
df_colesterol = pd.get_dummies(df_colesterol, columns=['Nível de Atividade', 'Grupo Sanguíneo', 'Fumante'])
df_colesterol.info()

# Separar dados de treino e teste
X = df_colesterol.drop(columns=['Colesterol'])
y = df_colesterol['Colesterol']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)

# Treinar modelo
model_reg = LinearRegression()
model_reg.fit(X=X_train, y=y_train)

# Rodar o explainer no conjunto de treinamento
explainer_reg = shap.Explainer(model_reg, X_train)

# Calcular Shapley Values no conjunto de testes
shap_values_reg = explainer_reg(X_test)

# Plot: contribuição geral por Shapley Value
shap.plots.bar(shap_values_reg, show=False)
plt.savefig(
  "./dataviz/shapley-values-regression.png",
  dpi=300,
  bbox_inches='tight'
)
plt.close()

# Plot: contribuição para um exemplo específico
shap.plots.waterfall(shap_values_reg[0], max_display=13, show=False)
plt.savefig(
  "./dataviz/shapley-values-regression-especific-case.png",
  dpi=300,
  bbox_inches='tight',
)
plt.close()

# Plot: heatmap  contribuição geral
# shap.plots.heatmap(
#    shap_values_reg,
#     instance_order=np.arange(len(shap_values_reg)),
#     max_display=13,
#     show=False,
#     plot_width=12,
# )
# plt.savefig(
#   "./dataviz/heatmap-shap-values.png",
#   dpi=300,
#   bbox_inches="tight",
# )
# plt.close()