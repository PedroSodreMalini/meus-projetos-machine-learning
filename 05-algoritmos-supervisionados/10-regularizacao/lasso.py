import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Setar modelo
df_colesterol = pd.read_csv("./dataset/colesterol.csv")
df_colesterol.info()
df_colesterol.drop(columns=['Id'], inplace=True)
df_colesterol = pd.get_dummies(df_colesterol, ['Nível de Atividade', 'Grupo Sanguíneo', 'Fumante'])

# Separar dados de treino e teste
X = df_colesterol.drop(columns=['Colesterol'])
y = df_colesterol['Colesterol']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)

# Treinar modelo - Lasso
# Mostrar importância de Features para cada alphas
def importancia_features(modelo):
    importance = np.abs(modelo.coef_)
    print("Importância de Features")
    for i, feature in enumerate(modelo.feature_names_in_):
        print(f"{feature}: {importance[i]}")

# Root Mean Squared Error
rmse_values = []
def performance_regressao(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    rmse = root_mean_squared_error(y_pred=y_pred, y_true=y_test)
    rmse_values.append(rmse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

print("Lasso")
alphas = [0.01, 0.1, 1, 10, 100, 1000]
for alpha in alphas:
    print(f"\nAlpha = {alpha:.2f}")
    model_lasso = Lasso(alpha=alpha)
    model_lasso.fit(X_train, y_train)
    importancia_features(model_lasso)
    performance_regressao(model_lasso, X_test, y_test)

sns.lineplot(
    x=alphas,
    y=rmse_values,
    color='purple'
)
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.xlabel("Lasso Alpha")
plt.title("Root Mean Squared Error x Alpha")
plt.grid(visible=True)
plt.savefig("./dataviz/lasso-rmse-x-alpha.png")
plt.close()

# Treinar modelo - Lasso CV
# Testa os alphas e faz cross-validation
print("\n\nLasso CV")
model_lasso_cv = LassoCV(alphas=[0.1, 0.5, 1], cv=5, random_state=51)
model_lasso_cv.fit(X_train, y_train)
importancia_features(model_lasso_cv)
performance_regressao(model_lasso_cv, X_test, y_test)
print(f"Alpha escolhido: {model_lasso_cv.alpha_}")
