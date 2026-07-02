import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1: Ler dados
df_frutas = pd.read_csv("./dataset/fruits.csv")
df_frutas.drop(columns=['A_id'], inplace=True)
df_frutas['Quality'] = df_frutas['Quality'].map({ 'good': 1, 'bad': 0 })

# 2: Separar dados de treino e teste
X = df_frutas.drop(columns=['Quality'])
y = df_frutas['Quality']

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.3,
  random_state=51
)

# 3: Treinar modelo
model_lr = LogisticRegression()
model_lr.fit(X=X_train, y=y_train)

# 4: plots
explainer_classificator = shap.Explainer(model_lr, X_train)

shap_values_classificator = explainer_classificator(X_test)

# Plot: contribuição geral por Shapley Value
shap.plots.bar(shap_values_classificator, show=False)
plt.savefig(
  "./dataviz/shapley-values-classification.png",
  dpi=300,
  bbox_inches='tight'
)
plt.close()

# Plot: contribuição para um exemplo específico
shap.plots.waterfall(shap_values_classificator[0], max_display=13, show=False)
plt.savefig(
  "./dataviz/shapley-values-classification-especific-case.png",
  dpi=300,
  bbox_inches='tight',
)
plt.close()