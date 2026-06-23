import joblib
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

# 1: Carregar dados do CSV
# Scatter revela dados claramente lineares.
# 0 horas regadas = 0 m², interceptor = 0
df_irrigacao = pd.read_csv("./dataset/dados_de_irrigacao.csv")
print(df_irrigacao.describe())
print(df_irrigacao.info())

# 2: Visualização dos dados
#   - Dados muito próximos de uma linha crescente
plt.figure(figsize=(12,6))
sns.scatterplot(
    df_irrigacao,
    x="Horas de Irrigação",
    y="Área Irrigada por Ângulo",
    color="g"
    )
plt.savefig("./dataviz/scatter-horas-irrigacao-x-area-irrigada.png")
plt.close()

#   - Ver se tem outliers
#   - Não há outliers na área irrigada
plt.figure(figsize=(6, 8))
ax = sns.boxplot(df_irrigacao, y="Área Irrigada por Ângulo")
ax.set_xlabel(xlabel="Box plot Área Irrigada")
ax.set_ylabel(ylabel="Área irrigada")
plt.savefig("./dataviz/boxplot-horas-irrigacao.png")
plt.close()

#   - Correlação entre variáveis
#   - Correlação completamente linear
sns.heatmap(
    annot=True,
    data=df_irrigacao.corr("pearson"),
    cmap="crest",
    vmin=-1,
    vmax=1
)
plt.savefig("./dataviz/heatmap-correlacao-variaveis.png")
plt.close()

# 3: Treinar  modelo
X = df_irrigacao["Horas de Irrigação"].values.reshape(-1, 1)
y = df_irrigacao["Área Irrigada por Ângulo"].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=51,
    test_size=0.3
)

model = LinearRegression()
model.fit(X=X_train, y=y_train)

print("\nEquação da reta:")
print(f"area_irrigada = {model.coef_[0][0]:.2f}*horas_de_irrigacao + {model.intercept_[0]:.2f}")

# 4: Métricas
print("\nMétricas")
y_pred = model.predict(X=X_test)

#   - RMSE
#   |---> 2.25*10⁻¹² (muito pequeno, o modelo é praticamente perfeito)
rmse = root_mean_squared_error(y_pred=y_pred, y_true=y_test)
print(f"Round Mean Squared Error (RMSE): {rmse:.12f}")

#   - Mean Absolute Erro (MAE)
#   |---> Aproximadamente 2 * 10⁻¹², muito pequeno
mae = mean_absolute_error(y_pred=y_pred, y_true=y_test)
print(f"Mean Absolute Error (MAE): {mae:.12f}")

# Scatter dados reais x dados previstos
#   - Pontos exatamente acima um do outro: modelo está ótimo.
x_axis = range(len(y_pred))
plt.figure(figsize=(12,6))
sns.scatterplot(x=x_axis, y=y_test.reshape(-1), label="Dados reais", color="b")
sns.scatterplot(x=x_axis, y=y_pred.reshape(-1), label="Dados previstos", color="r")
plt.savefig("./dataviz/dados-reais-x-previstos.png")
plt.close()

# 5: Análise de resíduos
residuos = y_test - y_pred
residuos_std = zscore(residuos)

# Há dados que fogem da análise e se encontram além de -2 pontos
sns.scatterplot(x=y_pred.reshape(-1), y=residuos_std.reshape(-1))
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Valores previstos")
plt.ylabel("Resíduos padronizados")
plt.savefig("./dataviz/scatter-residuos.png")
plt.close()

# Resíduos dos dados desviam da normalidade. 
pg.qqplot(residuos_std, dist="norm", confidence=0.95)
plt.xlabel('Quantis teóricos')
plt.ylabel('Resíduos na escala padrão')
plt.savefig("./dataviz/qqplot-residuos.png")
plt.close()

# 6: Previsões
# Acerta em até 10 casas decimais
area_prevista = model.predict([[15]])
print(f"Área prevista: {area_prevista[0][0]:.6f}")

# 7: Salvar modelo
joblib.dump(model, "./irrigacao-model.pkl")