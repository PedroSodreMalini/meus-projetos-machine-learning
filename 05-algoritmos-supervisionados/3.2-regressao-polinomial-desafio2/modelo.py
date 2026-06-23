import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 1: Ler dados
df_funcao = pd.read_csv("./dataset/funcao.csv")
print(df_funcao.describe())
df_funcao.info()

# 2: EDA
sns.lineplot(
  data=df_funcao,
  x='x',
  y='y',
  color="blue",
  label="Função",
)
plt.axhline(
  y=0,
  color="black",
  label='Eixo X',
  linestyle='--',
)
plt.axvline(
  x=0,
  color="black",
  label="Eixo Y",
  linestyle='--',
)
plt.title("Função f(x)")
plt.legend()
plt.savefig("./dataviz/function-lineplot.png")
plt.close()

# 3: Treinamento do modelo
# graus = [1,2,3,4,5,6,7,8,9,10]
graus = [8]

X = df_funcao.drop(columns=['y'])
y = df_funcao.drop(columns=['x'])

colunas_numericas = ['x']

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  shuffle=True,
  random_state=51,
  train_size=0.8
)

rmse_train_values=[]
rmse_test_values=[]
r2score_test_values=[]
percentual_rmse_values=[]

for grau in graus:
    transformer_numericas = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', transformer_numericas, colunas_numericas)
    ])

    poly_features = PolynomialFeatures(degree=grau, include_bias=False)

    model_poly = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly_features', poly_features),
        ('regressor', LinearRegression()),
    ])

    model_poly.fit(X=X_train, y=y_train)

    y_train_pred = model_poly.predict(X_train)
    y_pred = model_poly.predict(X_test)

    r2score = r2_score(y_true=y_test, y_pred=y_pred)
    rmse_train = root_mean_squared_error(y_pred=y_train_pred, y_true=y_train)
    rmse_test = root_mean_squared_error(y_pred=y_pred, y_true=y_test)

    rmse_test_values.append(rmse_test)
    rmse_train_values.append(rmse_train)
    r2score_test_values.append(r2score)

if len(graus) > 1:
    sns.lineplot(
        y=rmse_train_values,
        x=graus,
        color='purple',
        label="RMSE valores de treino"
    )
    sns.lineplot(
        y=rmse_test_values,
        x=graus,
        color='green',
        label="RMSE valores de teste"
    )
    plt.ylabel("RMSE")
    plt.xlabel("Grau")
    plt.title("RMSE - Teste x Treino")
    plt.grid(visible=True)
    plt.savefig("./dataviz/rmse-teste-x-treino-lineplot.png.png")
    plt.close()

    sns.lineplot(
        x=graus,
        y=r2score_test_values,
        color='purple',
        label='R2-Score'
    )
    plt.xlabel("Grau")
    plt.ylabel("R2-Score")
    plt.grid(visible=True)
    plt.savefig("./dataviz/r2-score-lineplot.png")
    plt.close()

    print("R2-scores:")
    print(r2score_test_values)

# 4: Métricas
print(f"R2-Score: {r2score_test_values[0]:.6f}")
print(f"RMSE: {rmse_test_values[0]:.6f}")

joblib.dump(model_poly, "modelo_polinomial.pkl")
