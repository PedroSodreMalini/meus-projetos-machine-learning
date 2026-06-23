import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import shapiro, kstest, zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

# ANÁLISE CRÍTICA DO MODELO:
# Conclusão: modelo consegue prever de certa forma bem os resultados.
# Melhorias: talvez o modelo precisasse de mais dados.
# O gráfico tem 2 padrões de crescimentos, logo talvez ter 2 modelos fosse melhor.
# Outra possibilidade seria melhor uma regressão linear múltipla.

# 1: Carregar dados
# Modelo não tem valores nulos
# horas_estudo -> float64
# pontuacao_teste -> int64
df_pontuacoes = pd.read_csv("./datasets/pontuacao_teste.csv")
print(df_pontuacoes.info())

# 2: Análise exploratória
# Mostrar relação das variáveis e identificar outliers
print(df_pontuacoes.describe())

#   - Plot de dispersão
#   |---- Mostra uma relação próxima da linear
plt.figure(figsize=(12,6))
sns.scatterplot(data=df_pontuacoes ,y="pontuacao_teste", x="horas_estudo")
plt.savefig("./dataviz/scatter-dataframe.png")
plt.close()

#   - Plot de boxplot
#   |---- Mostra se há outliers
#   |---- Horas estudo não tem outliers
#   |---- Pontuações não tem outliers
plt.figure(figsize=(6,6))
sns.boxplot(data=df_pontuacoes, y="horas_estudo")
plt.savefig("./dataviz/boxplot-horas-estudo")
plt.close()

plt.figure(figsize=(6,6))
sns.boxplot(data=df_pontuacoes, y="pontuacao_teste")
plt.savefig("./dataviz/boxplot-pontuacao_teste")
plt.close()

# - Correlação de Pearson
#   |---- Ideal para correlações aproximadamente lineares
#   |---- Correlação de 0.99 por Spearman -> Muito alta
plt.figure(figsize=(6,6))
sns.heatmap(df_pontuacoes.corr('pearson'), annot=True, vmin=-1, vmax=1, cmap="crest")
plt.savefig("./dataviz/heatmap-dataframe.png")
plt.close()

# - Histograma
#   |---- Checar distribuição dos dados
#   |---- Revela uma assimetria negativa para horas_estudo, muitos dados concentrados no fim do gráfico
#   |---- Ápice agudo para horas_estudo, indicando muitos dados ao fim
#   |---- Revela uma assimetria negativa para pontuacao_teste, muitos dados concentrados no fim do gráfico
#   |---- Ápice suave para horas_estudo, indicando dados mais bem distribuídos
plt.figure(figsize=(6,6))
sns.histplot(data=df_pontuacoes, x="horas_estudo", kde=True, binwidth=4)
plt.savefig("./dataviz/hist-horas-estudo.png")
plt.close()

plt.figure(figsize=(6,6))
sns.histplot(data=df_pontuacoes, x="pontuacao_teste", kde=True, binwidth=40)
plt.savefig("./dataviz/hist-pontuacao_teste.png")
plt.close()

# 3: Treinar modelo
# - Fazer processo de reshape quando há somente uma variável independente
X = df_pontuacoes['horas_estudo'].values.reshape(-1, 1)
y = df_pontuacoes["pontuacao_teste"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

model = LinearRegression()
model.fit(X=X_train, y=y_train)

print("A equação da reta é:")
print(f"pontuacao = {model.coef_[0][0]:.4f}x + {model.intercept_[0]:.4f}")

# 4: Validar modelo / Obter métricas
y_pred = model.predict(X=X_test)

#   - R-SQuared ou Coeficiente de Determinação
#   - R2 Representa a proporção na variação na variável dependente
# que é explicada pela variável independente.
#   - Vai de 0 - 1 -> quanto mais próximo de 1, melhor.
r2 = r2_score(y_pred=y_pred, y_true=y_test)
print("R-Squared (R2): ", r2)

#   - Mean Absolute Error (MAE)
#   - Representa o erro médio do modelo
#   - Métrica fácil de interpretar
#   - MAE é menos sensível a outliers
#   - O próprio usuário consegue falar se o erro do modelo está alto ou não.
mae = mean_absolute_error(y_pred=y_pred, y_true=y_test)
print("Mean Absolute Error (MAE): ", mae)

#   - Mean Squared Error (MSE)
#   - MSE = Média (y_test - y_pred)²
#   - Média dos erros ao quadrado
#   - Não é uma métrica fácil de interpretar (não é da ordem da variável dependente).
#   - Penaliza outliers, por ser sensível a eles (ex: 100² = 10000).
mse = mean_squared_error(y_pred=y_pred, y_true=y_test)
print("Mean Squared Error (MSE): ", mse)

#   - Root Mean Squared Error
#   - MRSE = sqrt(MSE)
#   - Métrica fácil de interpretar (da ordem da variável dependente)
#   - Penaliza outliers.
mrse = root_mean_squared_error(y_pred=y_pred, y_true=y_test)
print("Root Mean Squared Error (MRSE): ", mrse)

#   - Análise gráfica
#   - Possível ver em scatter quais pontos tão mais pŕoximos do real e quais não.
x_axis = range(len(y_test))
plt.figure(figsize=(10,6))
sns.scatterplot(x=x_axis, y=y_test.reshape(-1), color="blue", label="Valores Reais")
sns.scatterplot(x=x_axis, y=y_pred.reshape(-1), color="red", label="Valores Preditos")
plt.legend()
plt.savefig("./dataviz/scatter-predict-real.png")
plt.close()

#   - Calcular resíduos
#   - Zscore
#     |---> Para cada elemento do conjunto (X - media) / desvio_padrao
residuos = y_test - y_pred
residuos_std = zscore(residuos)

#   - Teste de linearidade do modelo
#   - Se os resíduos estiverem entre -2 e +2, indica linearidade
#   - Verificar homogeineidade das variâncias:
#   |---> Valores estiverem em torno da reta, há homogeneidade.
#   |---> Caso contrário (cone, funil, etc), há heterogeneidade.
#   - Nesse caso: alguns valores fogem da linha demais (passam de + 2).
#   |---> Esse modelo não é tão linear assim, talvez pudesse ter 2 modelos para representá-lo.
#   - Nesse caso: valores não está exatamente na reta, logo há uma certa heterogeneidade.
sns.scatterplot(x=y_pred.reshape(-1), y=residuos_std.reshape(-1))
plt.axhline(y=0)
plt.savefig("./dataviz/scatter-residuos.png")
plt.close()

#   - Checar se o modelo segue uma distribuição normal.
#   QQ (Quantile-Quantile) Plot: avalia se uma amostra segue uma distribuição normal.
#   |---> Pontos estão envoltos da reta, logo não é totalmente normal, mas próximo disto.
#   |---> Análises revelam que os resíduos não são exatamente lineares.
pg.qqplot(residuos_std, dist="norm", confidence=0.95)
plt.xlabel('Quantis teóricos')
plt.ylabel('Resíduos na escala padrão')
plt.savefig("./dataviz/qqplot-residuos.png")
plt.close()

# 4.1 Testes de normalidade
# H0 - Distribuição segue distribuição normal
# H1 - Distribuição não segue distribuição normal
# Se o p-valor > 0.05 aceita H0, caso contrário rejeita

#   - Teste de Shapiro Wilk
#   |---> p-valor ≃ 0.15, logo a distribuição é normal
stat_shapiro, p_valor_shapiro = shapiro(residuos.reshape(-1))
print(f'\nTestes de normalidade:\n1- Shapiro\nEstatística do teste: {stat_shapiro} e P-valor: {p_valor_shapiro}')

#   - Teste de Kolmogorov-Smirnov
#   |---> p_valor muito menor que 0.05, logo não segue distribuição normal para esse teste.
stat_ks, p_valor_ks = kstest(residuos.reshape(-1), "norm")
print(f'2-Kolmogorv-Smirnov\nEstatística do teste: {stat_ks} e P-valor: {p_valor_ks}')

# 5: Predições
# Se eu estudar 30.4 horas, qual a pontuação prevista pelo modelo ?
# No dado real é 460 pontos
# No modelo é 484
# Erro é de 24 pontos nesse caso
print("\nPredições:")
pont_prev = model.predict([[30.4]])
print(f"Pontuação prevista 30.4h de estudo: {pont_prev[0][0]} pontos")

# Caso reverso: quanto eu devo estudar para tirar 600 pontos?
# Caso real: ≃ 40.3h
# Modelo: 37.8h
# Erro está grande, algo em tono de 2.5h
# pontos = a * horas + b <---> horas = (pontos - b)/a
horas_prev = (600 - model.intercept_[0])/model.coef_[0][0]
print(f"Horas previstas para tirar 600 pontos: {horas_prev}h")

# 6: Salvar modelo para usar em algum momento
joblib.dump(model, "./modelo_regressao.pkl")
