import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1: Leitura de dados
print("Leitura de dados:")
df_diabetes = pd.read_csv("./datasets/exame-diabetes.csv")
print(df_diabetes.head(10)) # ver o esquema rapidamente
print(df_diabetes.info()) # notar que não há NaN

# 2: Transformação de features
#   - Id do paciente não é importante
#   - Genero deve ser transformado em uma variável numérica (F = 0, M = 1)
print("\nTransformação de features:")
df_diabetes.drop(columns=['id_paciente'], inplace=True)
print(df_diabetes.columns)
df_diabetes = pd.get_dummies(df_diabetes, columns=["genero"], dtype='int64')
print("Coluna genero:\n", df_diabetes.head(5))

# 3: Criação de features
#   - Redução de dimensionalidade: Peso & Altura -> IMC
#   - IMC = peso(kg) / altura(m)²
df_diabetes["imc"] = df_diabetes.peso / ((df_diabetes.altura / 100)**2) # transforma em metros depois eleva ao quadrado
print("\nColuna IMC:")
print(df_diabetes.imc.head(5))

# 4: Visualização de dados
#   - Apresentar um heatmap com correlação entre as variáveis
#   - Apresentar correlação entre o resultado e o resto
sns.heatmap(df_diabetes.corr(), annot=True, vmin=-1, vmax=1)
plt.savefig("./dataviz/heatmap.png")
plt.close()

#   - Heatmap
sns.heatmap(
    df_diabetes.corr()[['resultado']].sort_values(by="resultado", ascending=False), 
    vmin=-1, 
    vmax=1,
    annot=True,
    cmap="coolwarm"
)
plt.savefig("./dataviz/heatmap-resultado.png")
plt.close()

#   - Plot de Scatter (Dispersão) com distribuição
pd.plotting.scatter_matrix(df_diabetes, alpha=0.2, figsize=(6,6), diagonal="kde")
plt.savefig("./dataviz/scatter-matrix.png")
plt.close()

#   - Histograma de todas as variáveis
df_diabetes.hist(layout=(2,4), figsize=(10,5))
plt.savefig("./dataviz/all-histograms.png")
plt.close()

# 5: Modelo 1
#   - Nesse modelo não será usado IMC.
#   - Nesse modelo irão ser usados altura e peso.
#   - X = variáveis dependentes.
#   - Y = variável a ser prevista.
print("\n\nMODELO 1")
X = df_diabetes.drop(columns=["imc", "resultado"])
y = df_diabetes["resultado"]

#   - Dividir conjunto entre treino e testes
#   - 70% treinamento, 30% teste.
#   - OBS: dados não estão na proporção correta da distribuição.
#   - random_state = 51 -> dados para treinos são sempre os mesmos em cada execução do programa
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=51)

#   - Treinamento do modelo 1 pelo algoritmo de Regressão Linear
model_1 = LinearRegression().fit(X_train, y_train)

#   - Gerar predições do conjunto de testes com base no Modelo 1
y_pred = model_1.predict(X_test)
print("Predições de resultado a partir do modelo 1 para X_test:")
print(y_pred)
print("\nCoeficientes para cada variável:")
print(model_1.coef_)
print("\nValor de b em ax + b:")
print(model_1.intercept_)

#   - Métricas do modelo 1
#   - R2Score -> vai de -infinito até 1.
#     |-------> mede o quanto X explica y. Quanto mais próximo de 1, melhor.
#     |-------> A nota está dando 0.06, o que é baixo para o conjunto de treino.
#     |-------> A nota está dando -0.13, o que é baixo para o conjunto de teste.
#   - MAE (Mean Absolute Error)
#     |-------> Qual a diferença média entre y_test e y_pred
#     |-------> O erro médio é de 12, o que é alto: pode diagnosticar alguém
#     |          como sem diabetes e ela ter diabetes, e alguém como com diabetes
#     |          e ela não ter diabetes.
# EM RESUMO: O MODELO 1 É RUIM!
print("R2 Score X_train e y_train")
print(model_1.score(X_train, y_train))
print("\nR2 Score x_test e y_test <-> R2 Score y_test e y_pred")
print(model_1.score(X_test, y_test))
print(r2_score(y_test, y_pred)) # mesma coisa que model1_score(X_test, y_test)

print("\nMAE (Mean Absolute Error)")
print(mean_absolute_error(y_true=y_test, y_pred=y_pred))

# 6: Modelo 2
#   - Nesse modelo será usado IMC.
#   - Nesse modelo não irão ser usados altura e peso.
#   - X = variáveis dependentes.
#   - Y = variável a ser prevista.
#   - R2 Score = -0.08 -> RUIM
#   - MAE ≃ 12.5 -> RUIM
# EM RESUMO: MODELO 2 TAMBÉM É RUIM!
print("\n\nMODELO 2")
X = pd.DataFrame(df_diabetes["imc"])
y = df_diabetes["resultado"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=51)

model_2 = LinearRegression().fit(X_train, y_train)
y_pred = model_2.predict(X_test)
print("Predições de resultado a partir do modelo 2 para X_test:")
print(y_pred)
print("\nCoeficientes para cada variável:")
print(model_2.coef_)
print("\nValor de b em ax + b:")
print(model_2.intercept_)

print("\nR2 Score X_train e y_train")
print(model_2.score(X_train, y_train))
print("\nR2 Score x_test e y_test <-> R2 Score y_test e y_pred")
print(model_2.score(X_test, y_test))
print(r2_score(y_test, y_pred)) # mesma coisa que model2_score(X_test, y_test)

print("\nMAE (Mean Absolute Error)")
print(mean_absolute_error(y_true=y_test, y_pred=y_pred))

# 7: Mostrar como a reta foi calculada
# Pelo scatter dá para ver que o conjunto está aleatório e não é um
# bom caso para a regressão linear.
# Pela reta no gráfico, vê-se que os pontos estão distantes.
plt.scatter(X_test, y_test, color='g')
plt.plot(X_test, y_pred, color='k')
plt.savefig("./dataviz/model2-scatter.png")
plt.close()