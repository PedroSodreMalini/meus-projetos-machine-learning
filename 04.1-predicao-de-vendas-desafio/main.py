import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

dados_vendas = {
    'mes': [
      'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',  'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
    ],
    'vendas': [
       2000, 2200, 2300, 2500, 2600, 2700,
       2800, 2900, 3000, 3100, 3200, 3300
    ]
}

# 1: Leitura de dados
df_vendas = pd.DataFrame.from_dict(dados_vendas)

# 2: Transformação de features -> transformar mês em número
df_vendas["mes"] = range(1, len(df_vendas) + 1)

# 3: Criação de features -> Houve criação sem redução de dimensionalidade.

# 4: Visualização dos dados
# Gráficos mostram uma relação muito próxima da linear
df_vendas.plot.line(x="mes", y="vendas", color="red", figsize=(12, 6))
plt.savefig("./dataviz/line-graph.png")
plt.close()

df_vendas.plot.scatter(x="mes", y="vendas", color='red', figsize=(12,6))
plt.savefig("./dataviz/scatter-graph.png") # visivelmente próximo de uma linha
plt.close()

# 5: Treinar o modelo
X = df_vendas.drop(columns=["vendas"])
y = df_vendas.drop(columns=["mes"])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=51)

model = LinearRegression().fit(X=X_train, y=y_train)

y_pred = model.predict(X_test)
#   - Equação da reta resultante:
#   - vendas = 114.17 * mes + 1963.33
print("Predições de resultado a partir do modelo 1 para X_test", end=" ")
print(y_pred)
print("Coefiecientes para cada variável:", end=" ")
print(model.coef_)
print("Valor de b:", end=" ")
print(model.intercept_)

# 6: Prever vendas de dezembro
previsao_dezembro = model.predict(pd.DataFrame({"mes": [12]}))
reais_vendas_dezembro = df_vendas.loc[df_vendas['mes'] == 12, 'vendas'].values[0]
print(f"Previsão de vendas para Dezembro: R$ {round(previsao_dezembro[0][0], 2)}")
print(f"Reais vendas em Dezembro: R$ {reais_vendas_dezembro}")

# 7: Métricas
print("Métricas")
# Pontuação R2: de -infinito até 1
# Resultado está dando  0.96 para testes. Parece um bom modelo.
print("R2 Score para testes e previsão:", end=" ")
print(r2_score(y_test, y_pred))

# MAE: Quanto mais próximo de 0 melhor.
# A resposta vem na da dimensão da variável target.
# Erro médio está resultando em 36 reais. A depender pode ser bom ou ruim.
print("MAE (Mean Absolute Error):", end=" ")
print(mean_absolute_error(y_true=y_test, y_pred=y_pred))

# Histograma de todas as variáveis
df_vendas.hist(bins=10, figsize=(10,6), grid=True)
plt.suptitle("Histograma de Vendas e Meses")
plt.savefig("./dataviz/histogram-df-vendas.png")
plt.close()

# Scatter graph:
plt.figure(figsize=(10,5))
plt.scatter(X_test, y_test, color="green", label="Dados reais")
plt.plot(X_test, y_pred, color="blue", label="Reta do modelo")
plt.legend()
plt.savefig("./dataviz/model-scatter.png")
plt.close()