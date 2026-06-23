import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import zscore, shapiro, kstest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error

# 1: Carregar Dados
print("Carregar dados:")
df_salarios = pd.read_csv("./dataset/dataset.csv")
print(df_salarios.info())

df_salarios_eda = df_salarios.copy()
df_salarios_bucketing = df_salarios.copy()

# 2: Análise Exploratória dos Dados
# |---> Não tem valores nulos.
# |---> Salario tem média e mediana muito distintas (5093 | 3237)
#       | Maioria das pessoas ganham pouco e 
#       | os salários grandes explodem a média.
# |---> Boxplot mostra salários altos como outliers.
# |---> Scatter tempo x salario mostra curva, logo correlação de pearson não faz sentido.
#       | Quanto mais tempo ela está na empresa, mais ela ganha.
# |---> Scatter nivel x salario mostra uma distribuição uniforme sem correlação.
# |---> Scatter nivel x tempo mostra uma distribuição uniforme sem correlação.
# |--> histograma de nivel e tempo mostram distribuição uniformes
# |--> histograma de salario mostra assimetria positiva (cauda à direita):
#      | Muitos ganham pouco e poucos ganham muito.
# |---> Correlação de Spearman mostra correlação positiva muito forte entre salario e tempo
# |---> Correlação de Spearman mostra correlação negativa muito fraca entre salario e nivel
# |---> Bucketing transforma tempo de empresa em variável contínua.
#       | indica crescimento em curva entre escala de tempo na empresa e salário.
print("\nAnálise Exploratória de Dados (EDA)")
print(df_salarios_eda.head(10))
print("Ver se há valores nulos:")
print(df_salarios_eda.isna().sum())
print("Medidas estatísticas:")
print(df_salarios_eda.describe())

# 2.1 - Boxplot
sns.boxplot(data=df_salarios_eda, x='tempo_na_empresa')
plt.savefig("./dataviz/tempo-na-empresa-boxplot.png")
plt.close()

sns.boxplot(data=df_salarios_eda, x='nivel_na_empresa')
plt.savefig("./dataviz/nivel-na-empresa-boxplot.png")
plt.close()

sns.boxplot(data=df_salarios_eda, x='salario_em_reais')
plt.savefig("./dataviz/salario-boxplot.png")
plt.close()

# 2.2 - Scatterplots com target e histogramas
sns.scatterplot(data=df_salarios_eda, x="tempo_na_empresa", y="salario_em_reais")
plt.savefig("./dataviz/tempo-na-empresa-x-salario-scatter.png")
plt.close()

sns.scatterplot(data=df_salarios_eda, x="nivel_na_empresa", y="salario_em_reais")
plt.savefig("./dataviz/nivel-na-empresa-x-salario-scatter.png")
plt.close()

sns.scatterplot(data=df_salarios_eda, x="nivel_na_empresa", y="tempo_na_empresa")
plt.savefig("./dataviz/nivel-na-empresa-x-tempo-na-empresa-scatter.png")
plt.close()

sns.pairplot(data=df_salarios_eda)
plt.savefig("./dataviz/pairplot.png")
plt.close()

# 2.3 - Análise de correlação
sns.heatmap(
    data=df_salarios_eda.corr(method='pearson'),
    vmin=-1,
    vmax=1,
    cmap='crest',
    annot=True
)
plt.savefig("./dataviz/correlacoes-pearson-heatmap.png")
plt.close()

sns.heatmap(
    data=df_salarios_eda.corr(method='spearman'),
    vmin=-1,
    vmax=1,
    cmap='crest',
    annot=True
)
plt.savefig("./dataviz/correlacoes-spearman-heatmap.png")
plt.close()

plt.figure(figsize=(7,6))
sns.heatmap(
    data=df_salarios_eda.corr(method='spearman')[['salario_em_reais']].sort_values(by='salario_em_reais', ascending=False),
    vmin=-1,
    vmax=1,
    cmap='crest',
    annot=True
)
plt.savefig("./dataviz/correlacoes-ranking-salario-heatmap.png")
plt.close()

# 2.4- Bucketing tempo de empresa
bins_tempos_casa = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129]
labels_tempo_casa = ['0-9', '10-19', '20-29', '30-39', '40-49','50-59','60-69','70-79','80-89','90-99', '100-109','110-119', '120-129']
df_salarios_bucketing['escala_tempo'] = pd.cut(
    x=df_salarios_bucketing['tempo_na_empresa'],
    bins=bins_tempos_casa,
    labels=labels_tempo_casa,
    include_lowest=True
)

plt.figure(figsize=(14,8))
sns.boxplot(data=df_salarios_bucketing, x='escala_tempo', y='salario_em_reais')
plt.savefig("./dataviz/escala-tempo-x-salario.png")
plt.close()

# 3: Treinamento do modelo - Modelo de Regressão Linear Múltipla
# |---> Primeiro será feito um modelo em regressão linear para comparação futura

# 3.1 - Divisão de dados com KFold
print("----------")
print("Modelo de Regressão Linear Múltipla")
print("----------")
X = df_salarios.drop(columns='salario_em_reais')
y = df_salarios['salario_em_reais']

kf = KFold(n_splits=5, shuffle=True, random_state=51)

# 3.2 - Estrutura do Pipeline
colunas_numericas = ['tempo_na_empresa', 'nivel_na_empresa']

transformer_numericas = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocesser = ColumnTransformer(transformers=[
    ('num', transformer_numericas, colunas_numericas)
])

model_linear = Pipeline(steps=[
    ('preprocessor', preprocesser),
    ('regressor', LinearRegression())
])

# 3.3 - Aplicar K-Folds
# |---> Para cada repetição de K-fold precisa tirar as metricas
#       | Tirar a média no final para servir para o modelo
# Encontrar overfitting: 
#       | modelo de predição tem erros para testes grandes em comparação com dados de treinamento
# Encontrar underfitting:
#       | modelo de predição tem erros para treinamento grandes.
#   - Guardar métricas para cada iteração
rmse_scores_fold_train = []
rmse_scores_fold_test = []
r2_score_fold_test = [] 
residuos = [] # residuos em cada iteracao
y_pred_total = [] # predicoes em cada iteração

# 3.4 - Loop de treinamento do modelo
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model_linear.fit(X_train, y_train)

    # Faça predições nos conjuntos de treinamento e teste
    y_train_pred = model_linear.predict(X_train)
    y_test_pred = model_linear.predict(X_test)

    # Calcule o RMSE, R²-Score e Resíduos
    rmse_train = root_mean_squared_error(y_true=y_train, y_pred=y_train_pred)
    rmse_test = root_mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    r2score_test = r2_score(y_true=y_test, y_pred=y_test_pred)
    residuos_test = np.array(y_test - y_test_pred) # type: ignore

    rmse_scores_fold_train.append(rmse_train)
    rmse_scores_fold_test.append(rmse_test)
    r2_score_fold_test.append(r2score_test)
    residuos.append(residuos_test)
    y_pred_total.append(y_test_pred)

# 4: Análise do modelo - Modelo de Regressão Linear
# 4.1 - Análise de métricas
# Pelas métricas já se vê que o R² é menor que O.9, o que mostra que o modelo não é tão preciso
# A diferença entre RMSE de treinamento e de teste é de 4 reais, 
# | logo não é tão diferente entre teste e treinamento.
# Contudo um erro de 1800 reais é terrível.
# |--> Diferença de ~ 0.2% percentual, muito baixa
rmse_train_final = np.mean(rmse_scores_fold_train)
rmse_test_final = np.mean(rmse_scores_fold_test)
r2_final = np.mean(r2_score_fold_test)
percentual_rmse_final = ((rmse_test_final - rmse_train_final) / rmse_train_final) * 100
residuos = np.array(residuos).reshape(-1)
y_pred_total = np.array(y_pred_total).reshape(-1)

print(f"Root Mean Squared Error - Test: {rmse_test_final}")
print(f"Root Mean Squared Error - Train: {rmse_train_final}")
print(f"R²-Score final: {r2_final}")
print(f"% Diferença Root Mean Squared Error Test and Train: {percentual_rmse_final}")

# 4.2 - Análise Gráfica de Resíduos.
# Linearidade / Homocedasticidade:
# | Resíduos formam uma estrutura, logo não há linearidade nem homocedasticidade
# | Resíduos formam uma espécie de parábola.
# Distrtibuição normal dos resíduos:
# | QQplot mostra que os resíduos não estão sobre a reta
# | Muitos pontos estão fora do limite.
residuos_std = zscore(residuos)

sns.scatterplot(x=y_pred_total, y=residuos_std) # type: ignore
plt.axhline(y=2, color='r')
plt.axhline(y=-2, color='r')
plt.axhline(y=0, color='gray')
plt.savefig("./dataviz/rglinear-residuos-scatter.png")
plt.close()

plt.figure(figsize=(14,8))
pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.xlabel('Quantis teóricos')
plt.ylabel('Resíduos na escala padrão')
plt.savefig("./dataviz/rglinear-qqplot.png")
plt.close()

# 4.3 - Testes estatísticos de normalidade
# H0: OS resíduos seguem uma distribuição normal.
# H1: OS resíduos não seguem uma distribuição normal.
# Shapiro-Wilk | Kolmogorov-Smirnov | Lilliefors
# |---> p-value de todos é < 0.05, logo H0 é rejeitada por falta de evidência.
# QQplot
# |---> Já indicava resíduos longe de uma distribuição normal.
stat_shapiro, p_value_shapiro = shapiro(residuos)
print(f"Shapiro-Wilk p-value: {p_value_shapiro}\nShapiro-Wilk stat: {stat_shapiro}")
stat_ks, p_value_ks = kstest(residuos, 'norm')
print(f"Kolmogorov-Smirnov p-value: {p_value_ks}\nKolmogorov-Smirnov stat: {stat_ks}")
stat_ll, p_value_ll = lilliefors(x=residuos, dist='norm', pvalmethod='table')
print(f"Lilliefors p-value: {p_value_ll}\nLilliefors stat: {stat_ll}")

# 5: Treinamento do modelo - Modelo de Regressão Polinomial
print("\n----------")
print("Modelo de Regressão Polinomial")
print("----------")

# 5.1 - Features polinomiais
feat_poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = feat_poly.fit_transform(X)
print(f"Nome das features de entrada usadas: {feat_poly.feature_names_in_}")
print(f"Todas as features geradas: {feat_poly.get_feature_names_out(feat_poly.feature_names_in_)}")

# 5.2 - Treinamento do modelo polinomial
# |---> Fazer 10 modelos
# |---> Após testar 10, o de grau 4 foi o mais interessante
# graus_polynomial = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
graus_polynomial = [4]

rmse_train_values  = []
rmse_test_values = []
percentual_rmse_values = []
r2score_test_values = []

kf = KFold(n_splits=5, shuffle=True, random_state=51)

for grau in graus_polynomial:
    # Estrutura do Pipeline
    colunas_numericas = ['tempo_na_empresa', 'nivel_na_empresa']

    transformer_numericas = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocesser = ColumnTransformer(transformers=[
        ('num', transformer_numericas, colunas_numericas)
    ])

    # Criar features polinomiais
    poly_features = PolynomialFeatures(degree=grau, include_bias=False)

    model_poly = Pipeline(steps=[
        ('preprocessor', preprocesser),
        ('poly_features', poly_features),
        ('regressor', LinearRegression()),
    ])

    rmse_scores_fold_test = []
    rmse_scores_fold_train = []
    r2score_fold_test = []
    residuos = []
    y_pred_total = []

    # Loop de treinamento do modelo
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_poly.fit(X_train, y_train)

        # Faça predições nos conjuntos de treinamento e teste
        y_train_pred = model_poly.predict(X_train)
        y_test_pred = model_poly.predict(X_test)

        # Calcule o RMSE, R²-Score e Resíduos
        rmse_train = root_mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        rmse_test = root_mean_squared_error(y_true=y_test, y_pred=y_test_pred)
        r2score_test = r2_score(y_true=y_test, y_pred=y_test_pred)
        residuos_test = np.array(y_test - y_test_pred) # type: ignore

        rmse_scores_fold_train.append(rmse_train)
        rmse_scores_fold_test.append(rmse_test)
        r2score_fold_test.append(r2score_test)
        residuos.append(residuos_test)
        y_pred_total.append(y_test_pred)
    
    rmse_train_final = np.mean(rmse_scores_fold_train)
    rmse_test_final = np.mean(rmse_scores_fold_test)
    r2_final = np.mean(r2score_fold_test)
    percentual_rmse_final = ((rmse_test_final - rmse_train_final) / rmse_train_final) * 100
    residuos = np.array(residuos).reshape(-1)

    y_pred_total = np.array(y_pred_total).reshape(-1)

    rmse_train_values.append(rmse_train_final)
    rmse_test_values.append(rmse_test_final)
    r2score_test_values.append(r2_final)
    percentual_rmse_values.append(percentual_rmse_final)

# 5.3 - Análisa gŕafica de overfitting

# Gráfico para comparar RMSE por grau de polinômio
# Linhas em cima da outra indicam pouco overfitting, 
# já que prevê de forma parecida teste e treino
if len(graus_polynomial) == 10:
    plt.figure(figsize=(12,8))
    plt.plot(graus_polynomial, rmse_train_values, label='RMSE (Treino)')
    plt.plot(graus_polynomial, rmse_test_values, label='RMSE (Teste)')
    plt.xlabel("Grau polinômio")
    plt.ylabel('RMSE')
    plt.title('RMSE por grau de polinômio')
    plt.legend()
    plt.grid(True)
    plt.savefig("./dataviz/rmse-por-grau-polinomial.png")
    plt.close()

# Gráfico para comparar percentual de diferença RMSE treino e teste
# Diferença percentual sobe de acordo com o grau
if len(graus_polynomial) == 10:
    plt.figure(figsize=(12,8))
    plt.plot(graus_polynomial, percentual_rmse_values, label='% Dif RMSE Treino e Teste')
    plt.xlabel("Grau polinômio")
    plt.ylabel('% Dif RMSE')
    plt.title('% Dif RMSE por grau de polinômio')
    plt.legend()
    plt.grid(True)
    plt.savefig("./dataviz/dif-percentage-rmse-por-grau-polinomial.png")
    plt.close()

# Conclusão para escolha do grau: 
# 1- Graus baixos dão underfitting, graus altos dão overfiting
# 2- Grau 4 é o mais interessante.

# 6: Análise do modelo - Modelo de Regressão Linear

# 6.1 - Análise de Métricas
# |---> R2 = 0.998 = explica muito bem os dados
# |---> RMSE ≃ 54 reais -> muito menor que o linear
print(f"Root Mean Squared Error - Test: {rmse_test_final}")
print(f"Root Mean Squared Error - Train: {rmse_train_final}")
print(f"R²-Score final: {r2_final}")
print(f"% Diferença Root Mean Squared Error Test and Train: {percentual_rmse_final}")

# 6.2 - Análise de Resíduos
# scatter - Homocedasticidade e Linearidade:
# | Valores espalhados sem formar estrutura (Homocedasticidade)
# | Alguns valores acima e abaixo de +-2
# qqplot - Distribuição Normal dos Resíduos:
# | Valores muito mais ajustados a reta.
# | Há evidência de resíduos normalizados.
# H0: OS resíduos seguem uma distribuição normal.
# H1: OS resíduos não seguem uma distribuição normal.
# Shapiro-Wilk | Lilliefors
# |---> p-value de todos é > 0.05, logo H0 é aceita por haver evidência.
# Kolmogorov-Smirnov
# |---> Alega rejeição da hipótese nula por falta de evidência.
# QQplot
# |---> Já indicava resíduos longe de uma distribuição normal.
residuos_std = zscore(residuos)

sns.scatterplot(x=y_pred_total, y=residuos_std) # type: ignore
plt.axhline(y=0, color='gray')
plt.axhline(y=2, color='red')
plt.axhline(y=-2, color='red')
plt.savefig("./dataviz/rgpoly-residuos-scatter.png")
plt.close()

pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.xlabel('Quantis teóricos')
plt.ylabel('Resíduos na escala padrão')
plt.savefig("./dataviz/rgpoly-qqplot.png")
plt.close()

stat_shapiro, p_value_shapiro = shapiro(residuos)
print(f"Shapiro-Wilk p-value: {p_value_shapiro}\nShapiro-Wilk stat: {stat_shapiro}")
stat_ks, p_value_ks = kstest(residuos, 'norm')
print(f"Kolmogorov-Smirnov p-value: {p_value_ks}\nKolmogorov-Smirnov stat: {stat_ks}")
stat_ll, p_value_ll = lilliefors(x=residuos, dist='norm', pvalmethod='table')
print(f"Lilliefors p-value: {p_value_ll}\nLilliefors stat: {stat_ll}")

# Conclusão: Modelo polinomial é melhor e mais ajustado.

# 7: Predições e salvar modelo
input_features = {
    'tempo_na_empresa': 80,
    'nivel_na_empresa': 10
}

pred_df = pd.DataFrame(input_features, index=[1])
prediction = model_poly.predict(pred_df)
print(f"Predição: R$ {prediction[0]:.2f}")

joblib.dump(model_poly, './modelo_poly_salario.pkl')
