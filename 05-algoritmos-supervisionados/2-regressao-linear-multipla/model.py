import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import zscore, shapiro, kstest, anderson
from statsmodels.stats.diagnostic import lilliefors, het_goldfeldquandt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.utils import estimator_html_repr

# 1: Carregar dados
# - ID não é útil

df_colesterol = pd.read_csv("./dataset/dataset.csv")
print("Tabela inicial")
print(df_colesterol.info())

df_colesterol.drop(columns=["Id"], inplace=True)
df_colesterol.columns = [
    'grupo_sanguineo',
    'fumante',
    'nivel_atividade_fisica',
    'idade',
    'peso',
    'altura',
    'nivel_colesterol'
]
print("\nTabela para o modelo:")
print(df_colesterol.info())
print("10 primeiros registros")
print(df_colesterol.head(10))

# 2: Análise Exploratória dos Dados (EDA)
# - Bucketing: Variáveis númericas para categóricas
#   |---> Alguns gráficos não aceitam variáveis categóricas, logo é bom analisar a parte
# Idade
#   |---> Mediana é 50 e média 50.2, logo usar mediana por ser inteira.
#   |---> Fazer isso pois idade não é float.
# Peso -> Max de 158kg e minimo de 15kg -> pesos muito extremos, amplitude muito grande
# |---> Peso mínimo deve ser erro
# |---> Cortar outliers
# Altura: Valores entre 160cm e 190cm, valores ok
# |---> Média e mediana próximas = boa distribuição
# Cocnlusão numérica: Mediana é boa para as n3 variáveis.
df_colesterol_eda = df_colesterol.copy()
df_colesterol_bucket = df_colesterol.copy()
print(df_colesterol_eda.describe())

# 2.1- Lidando com valores ausentes
# - Possum valores nulos: grupo_sanguineo, fumante, Nível de Atividade, idade, peso, altura
# - Variáveis numéricas: substituir por mediana
# - Variáveis categóricas: substituir por moda
# |---> OBS: moda tem mais de 1, logo escolher a primeira moda
print("\nLidando com valores ausentes:")
print(df_colesterol_eda.isna().sum())

print("*Variáveis categóricas nulas -> moda\n*Variáveis numéricas nulas -> mediana")
moda_grupo_sanguineo = df_colesterol_eda.grupo_sanguineo.mode()
moda_fumante = df_colesterol_eda.fumante.mode()
moda_nivel_atividade_fisica = df_colesterol_eda.nivel_atividade_fisica.mode()
median_idade = df_colesterol_eda.idade.median()
median_peso = df_colesterol_eda.peso.median()
median_altura = df_colesterol_eda.altura.median()

df_colesterol_eda.fillna(value={
    "grupo_sanguineo": moda_grupo_sanguineo[0],
    "fumante": moda_fumante[0],
    "nivel_atividade_fisica": moda_nivel_atividade_fisica[0],
    "idade": median_idade,
    "peso": median_peso,
    "altura": median_altura
}, inplace=True)

print("Sem dados ausentes:")
print(df_colesterol_eda.isna().sum())

# 2.2 - Converter valores numéricas
# - Converter idade e altura para inteiro
# |---> Só é possível fazer isso caso todos os valores estejam preenchidos
# |---> Logo tratar valores nulos antes
df_colesterol_eda.idade = df_colesterol_eda.idade.astype(int)
df_colesterol_eda.altura = df_colesterol_eda.altura.astype(int)

print("Dados convertidos:")
print(df_colesterol_eda.info())
print(df_colesterol_eda.head(10))

# 2.3 - Tratamento de outliers
# Idade: NÃO tem outliers
# Peso
# |---> Há alguns outliers devido a pessoas com sobrepeso.
# |---> Há alguns pesos possivelmente errados por serem muito baixos, ex: pesos abaixo de 40kg.
# |---> Remover pesos muito baixos
# |---> Manter pesos altos
# Altura: NÃO tem outliers
# Nível colesterol
# |---> Outliers para valores altos: fazem sentido
# |---> Não há outliers baixos
print("\nTratamento de outliers:")
sns.boxplot(data=df_colesterol_eda, x="idade", color='r')
plt.savefig("./dataviz/idade-boxplot.png")
plt.close()

sns.boxplot(data=df_colesterol_eda, x="peso", color='r')
plt.savefig("./dataviz/peso-boxplot.png")
plt.close()
print("Número de pessoas abaixo de 40kg: ", df_colesterol_eda[df_colesterol_eda.peso < 40].peso.count())
df_colesterol_eda.drop(
    df_colesterol_eda[df_colesterol_eda.peso < 40].index,
    axis=0,
    inplace=True
)

sns.boxplot(data=df_colesterol_eda, x="altura", color='r')
plt.savefig("./dataviz/altura-boxplot.png")
plt.close()

sns.boxplot(data=df_colesterol_eda, x="nivel_colesterol", color='r')
plt.savefig("./dataviz/colesterol-boxplot.png")
plt.close()

# 2.4 - Análise gráfica
# 2.4.1 - Variáveis categóricas x Target
# Uso de BOXPLOT
# Grupo Sanguineo: Na prática isso influencia mesmo.
# |---> Tipo O e tipo B: menor colesterol em geral
# |---> Tipo A e tipo AB: maior colesterol em geral
# Fumante
# |---> Fumantes tem mais colesterol que não fumantes
# Nível atividade física
# |---> Visivelmente quem se exercita mais tem menos colesterol
# |---> Relação de correlação negativa claro: quanto mais um, menos o outro
print("\nAnálise Exploratória dos Dados:")
plt.figure(figsize=(8,6))
ax = sns.boxplot(data=df_colesterol_eda, x="grupo_sanguineo", y="nivel_colesterol")
ax.set_xlabel("Grupo sanguíneo")
ax.set_ylabel("Colesterol")
plt.savefig("./dataviz/grupo-sanguineo-por-colesterol-boxplot.png")
plt.close()

plt.figure(figsize=(8,6))
ax = sns.boxplot(data=df_colesterol_eda, x="fumante", y="nivel_colesterol")
ax.set_xlabel("Fumante")
ax.set_ylabel("Colesterol")
plt.savefig("./dataviz/fumante-por-colesterol-boxplot.png")
plt.close()

plt.figure(figsize=(8,6))
ax = sns.boxplot(data=df_colesterol_eda, x="nivel_atividade_fisica", y="nivel_colesterol")
ax.set_xlabel("Engajamento em atividades físicas")
ax.set_ylabel("Colesterol")
plt.savefig("./dataviz/atividade-fisica-por-colesterol-boxplot.png")
plt.close()

# 2.4.2 - Variáveis numéricas x Target
# Idade:
# |---> Parece não influenciar devido a distribuição aleatória.
# Peso:
# |---> Quanto mais pesado, visivelmete mais colesterol.
# Altura:
# |---> Parece não influenciar devido a distribuição aleatória.
# Uso de DIAGRAMA DE DISPERSÃO
sns.scatterplot(data=df_colesterol_eda, x='idade', y='nivel_colesterol')
plt.savefig("./dataviz/idade-colesterol-scatter.png")
plt.close()

sns.scatterplot(data=df_colesterol_eda, x='peso', y='nivel_colesterol')
plt.savefig("./dataviz/peso-colesterol-scatter.png")
plt.close()

sns.scatterplot(data=df_colesterol_eda, x='altura', y='nivel_colesterol')
plt.savefig("./dataviz/altura-colesterol-scatter.png")
plt.close()

# 2.4.3 - Histogramas
# |---> Diagonal do pairplot mosra todos os histogramas.
# |---> No pairplot é possivel ver todos os outros scatter também.
# Altura e idade:
# |---> distribuição balanceada sem um valor predominante.
# Colesterol e peso:
# |---> distribuição normal: há faixa de valores que predomina
sns.pairplot(df_colesterol_eda)
plt.savefig("./dataviz/pairplot.png")
plt.close()

# 2.5 - Análise de Correlação
# Converter variáveis categóricas nominais em numéricas usando One-Hot Encoder do Pandas
# |---> fumante tera colunas para nao ou sim atribuindo 0 ou 1 se for daquele tipo
# |---> grupo_sanguineo terá colunas para cada tipo de grupo sanguineo atribuindo 1 se for do tipo
# Converter variável categórica ordinal em numéricas
# |---> nivel_atiividade_fisica (Baixo, Moderado, Alto) -> (0, 1, 2)
# Mapa de calor com correlação de variáveis
# |---> Ser fumante afeta
# |---> Peso afeta principalmente
# |---> Tipos sanguineos A e O afetam principalmente
# |---> Atividade física importa principalmente
# Mapa de calor somete com colesterol (variável dependente/target)
# |---> Dados mais ao centro não afetam muito.
# |---> Mais afetam: peso, atividade fisica, fumar e tipo sanguíneo O.
df_colesterol_eda = pd.get_dummies(
    df_colesterol_eda,
    columns=['grupo_sanguineo', 'fumante'],
    dtype='int64',
)
df_colesterol_eda.nivel_atividade_fisica = pd.factorize(
    values=df_colesterol_eda.nivel_atividade_fisica,
)[0] + 1

print("Dataframe após dummies e conversão de nivel_atividade_fisica:")
print(df_colesterol_eda.info())
print(df_colesterol_eda.head(7))

plt.figure(figsize=(15,6))
sns.heatmap(
    data=df_colesterol_eda.corr("pearson"),
    vmin=-1,
    vmax=1,
    annot=True,
    cmap='crest'
)
plt.savefig("./dataviz/correlacoes-heatmap.png")
plt.close()

sns.heatmap(
    data=df_colesterol_eda.corr("pearson")[['nivel_colesterol']].sort_values(by='nivel_colesterol', ascending=False),
    vmin=-1,
    vmax=1,
    annot=True,
    cmap='rocket',
)
plt.savefig("./dataviz/correlations-colesterol-heatmap.png")
plt.close()

# 2.6 - Análise gráfica com Bucketing
# Transformar idade em intervalos de 10 em 10 anos
# |---> Idade vai de 20 a 79 anos
# |---> Mostra que a idade não afeta tem uma relação clara com a target colesterol.
print("Bucketing")
print(f"Idade máxima: {df_colesterol_bucket.idade.max()}\nIdade mínima: {df_colesterol_bucket.idade.min()}")
bins = [20, 29, 39, 49, 59, 69, 79]
labels_idade = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
df_colesterol_bucket['escala_idade'] = pd.cut(
    x = df_colesterol_bucket['idade'],
    bins=bins,
    labels=labels_idade,
    include_lowest=True,
)
print("Exemplo intervalar:")
print(df_colesterol_bucket[['idade', 'escala_idade']].head(4))
sns.boxplot(data=df_colesterol_bucket, x='escala_idade', y='nivel_colesterol')
plt.savefig("./dataviz/escala-idade-por-nivel-colesterol.png")
plt.close()

# 3: Preparação do Dataset para treinamento
# 3.1 - Split do Dataset para treino e teste
print("\nPreparação do Dataset para treinamento:")
df_colesterol.drop(
    df_colesterol[df_colesterol['peso'] < 40].index,
    axis=0,
    inplace=True
)

X = df_colesterol.drop(columns='nivel_colesterol')
y = df_colesterol['nivel_colesterol']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)
print("Tamanho X-treinamento: ", X_train.shape)
print("Tamanho X-teste: ", X_train.shape)
print("Tamanho y-treinamento: ", y_train.shape)
print("Tamanho y-teste: ", y_train.shape)

# 3.2 - Transformação de variáveis categóricas e de variaveis numéricas para o modelo
# Construção de Pipeline
# Transformar valores ausentes pela moda
# |---> OneHotEncoding nas variáveis categóricas nominais
# |---> OrdinalEncoder nas variáveis categóricas ordinais
# |---> grupo_sanguineo, fumante, nivel_atividade_fisica
# Padronizar variáveis numéricas
# |---> idade, altura, peso
# |---> Transformar valores ausentes pela mediana
colunas_categoricas_nominais = ['grupo_sanguineo', 'fumante']
colunas_categoricas_ordinais = ['nivel_atividade_fisica']
colunas_numericas = ['idade', 'altura', 'peso']

transformer_categoricas_nominais = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

transformer_categoricas_ordinais = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OrdinalEncoder(
        categories=[['Baixo', 'Moderado', 'Alto']],
        handle_unknown='error',
    )),
])

transformer_numericas = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', transformer_numericas, colunas_numericas),
        ('cat', transformer_categoricas_nominais, colunas_categoricas_nominais),
        ('ord', transformer_categoricas_ordinais, colunas_categoricas_ordinais)
    ]
)

# 4: Treinar modelo
# Criar pipeline principal: preprocessamento + treinamento
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X=X_train, y=y_train)

# 5: Validar modelo
# - Gerar predição
# - Analisar r2_score
# - Analisar MAE e RMSE
# MAE ≃ 7.3 -> Ok
# RMSE ≃ 9.1 -> Ok, mas afeta muito em alguns pontos
# R2-Score ≃ 0.96 -> O modelo prevê bem.
# Outliers nos resíduos padronizados
# |---> 11 de 300 estão fora de +-2, então não há muitos outliers nos resíduos padronizados.
# Teste de Linearidade
# |---> Pontos em linearidade-dos-residuos-scatter não estão formando uma estrutura, 
# |---> e estão em volta da reta, o que mostra uma linearidade.
# Testes de Normalidade dos Resíduos:
# |---> p-value <= 0.05 em Kolmogorov-Smirnov e Shapiro-Wilk, 
# |---> logo rejeitam evidência de distribuição normal nos resíduos
# |---> Anderson-Darling mostra stat para 5% > valor_critico, logo também rejeita evidência 
# |---> p-value > 0.05 em Lilliefors, logo aceita H0, faltando evidência contra normalidade
# |---> QQPlot indica normalidade numa totalidade.
# Teste de Homocedasticidade
# |---> p-value = 0.98 em Goldfeld, logo os resíduos tem variância aproximadamente constante
# Conclusão:
# - Poderia ter mais dados para pessoas com sobrepeso (mais de 100kg), 
# |---> pois embora sejam outliers, entram no cálculo do modelo. 
print("\nValidar modelo:")
y_pred = model.predict(X=X_test)

r2score = r2_score(y_true=y_test, y_pred=y_pred)
mae = mean_absolute_error(y_pred=y_pred, y_true=y_test)
rmse = root_mean_squared_error(y_pred=y_pred, y_true=y_test)
print(f"R2-Score: {r2score:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

# z-score dos residuos = (residuo - media) / desvio_padrao 
residuos = y_test - y_pred
residuos_std = zscore(residuos)

# Verificar linearidade dos resíduos: Valores entre +-2.
# Verificar homocedasticidade: Valores em torno da reta.
ax = sns.scatterplot(x=y_pred, y=residuos_std)
ax.set_ylabel("Resíduos Z-Score de Nível Colesterol", loc='center')
ax.set_xlabel("Colesterol Previsto", loc="center")
plt.axhline(y=0, color='black')
plt.axhline(y=2, color='r')
plt.axhline(y=-2, color='r')
plt.savefig("./dataviz/linearidade-dos-residuos-scatter.png")
plt.close()

# Checar se resíduos seguem uma distribuição normal com Quantile-Quantile Plot
# Visivelmente os dados seguem uma distribuição normal, mas os outliers fogem um pouco.
plt.figure(figsize=(14,8))
pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.xlabel('Quantis Teóricos')
plt.ylabel('Resíduos na escala padrão')
plt.savefig("./dataviz/residuos-qqplot.png")
plt.close()

# Teste de normalidade de Shapiro-Wilk
# H0: A distribuição dos dados é normal
# H1: Os dados não são normais
# Se o p-valor > 0.05, não há evidência contra a normalidade
stat_shapiro, p_value_shapiro = shapiro(x=residuos)
print(f"Estatística de Teste de Shapiro-Wilk: {stat_shapiro}")
print(f"P-valor de Shapiro-Wilk: {p_value_shapiro}")

# Teste de normalidade de Kolmogorov-Smirnov
# H0: A distribuição dos dados é normal
# H1: Os dados não são normais
# Se o p-valor > 0.05, não há evidência contra a normalidade
stat_ks, p_value_ks = kstest(residuos, 'norm')
print(f"Estatística de Teste de Kolmogorov-Smirnov: {stat_ks}")
print(f"P-valor de Kolmogorov-Smirnov: {p_value_ks}")

# Teste de normalida de Lilliefors
# H0: A distribuição dos dados é normal
# H1: Os dados não são normais
# Se o p-valor > 0.05, não há evidência contra a normalidade
stat_ll, p_value_ll = lilliefors(residuos, dist='norm', pvalmethod='table')
print(f"Estatística de Teste de Lilliefors: {stat_ll}")
print(f"P-valor de Lilliefors: {p_value_ll}")

# Teste de Anderson-Darling
# H0: Não há evidências dos resíduos seguirem uma distribuição dos dados normal
# H1: Os dados não são normais
# Se stat and para 5% <= valor_critico -> aceita H0
# Caso contrário, rejeita H0
stat_and, critical_and, _ = anderson(residuos, dist='norm')
print(f"Estatística de Anderson para 5%: {stat_and:.3f}")
print(f"Valor crítico de Anderson para 5%: {critical_and[2]:.3f}")

# Teste de Homocedasticidade de Goldfeld-Quandt
# H0: A variancia dos erros é constante (homocedasticidade)
# H1: A variancia dos erros não é constante (heterocedasticidade)
# Se o p-valor > 0.05, há evidencia de homocedasticidade
X_test_transformed = model.named_steps['preprocessor'].transform(X=X_test)
test_goldfeld = het_goldfeldquandt(residuos, X_test_transformed)
stat_goldfeld = test_goldfeld[0]
p_value_goldfeld = test_goldfeld[1]
print(f"Estatística de Goldfeld: {stat_goldfeld}")
print(f"P-value de Goldfeld: {p_value_goldfeld}")

# 6: Predição dos Valores
# Nesse cenário erra por 10 acima. O que faz sentido devido ao alto peso.
print("\nPredição de valores")
predicao_individual = {
    'grupo_sanguineo': 'A',
    'fumante': 'Não',
    'nivel_atividade_fisica': 'Moderado',
    'idade': 68,
    'peso': 105,
    'altura': 184
}
print("Variáveis independentes:")
sample_df = pd.DataFrame(predicao_individual, index=[1])
print(sample_df)
colesterol = model.predict(sample_df)[0]
print(f"Previsão de colesterol: {colesterol}")

# 7: Salvar o modelo
# Salvar o modelo em pkl e o seu pipeline em html
joblib.dump(model, "./modelo_colesterol.pkl")

html = estimator_html_repr(model)
with open("pipeline-modelo.html", "w", encoding="utf-8") as f:
    f.write(html)

# 8: Coeficientes e interceptor
regressor = model.named_steps['regressor']
print("\nCoeficientes e Interceptor:")
print("Coeficientes: ")
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": regressor.coef_
}).sort_values(by="coef", key=abs, ascending=False)
print(coef_df)
print("Interceptor: ", regressor.intercept_)