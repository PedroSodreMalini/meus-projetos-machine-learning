import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import kstest, shapiro, zscore
from statsmodels.stats.diagnostic import lilliefors, het_goldfeldquandt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1: Carregar dados e checar valores faltantes.
# Não há valores nulos ou faltantes.
# Coluna id será inútil.
print("Carregar dados")
df_aluguel = pd.read_csv("./dataset/dataset_aluguel.csv")
print(df_aluguel.info())
print("\nRemovendo coluna id...\n")
df_aluguel.drop(columns=['id'], inplace=True)
df_aluguel['garagem'] = df_aluguel['garagem'].map({ 0: False, 1: True})
df_aluguel.columns = [
    'metros_quadrados',
    'n_quartos',
    'idade',
    'garagem',
    'periferia',
    'suburbio',
    'aluguel'
]
print(df_aluguel.describe())
print(df_aluguel.info())

# 2: Análise Exploratória dos Dados
# Não há outliers nos valores
print("\nAnálise Exploratória.")
df_aluguel_eda = df_aluguel.copy()

sns.pairplot(data=df_aluguel_eda, y_vars=['aluguel'])
plt.savefig("./dataviz/pairplot.png")
plt.close()

sns.boxplot(data=df_aluguel_eda, x='n_quartos', y='aluguel')
plt.xlabel("Número de Quartos")
plt.ylabel("Valor do Aluguel")
plt.savefig('./dataviz/n-quartos-x-aluguel-boxplot.png')
plt.close()

sns.boxplot(data=df_aluguel_eda, x='periferia', y='aluguel')
plt.xlabel("Está na Periferia")
plt.ylabel("Valor aluguel")
plt.savefig('./dataviz/periferia-x-aluguel-boxplot.png')
plt.close()

sns.scatterplot(data=df_aluguel_eda, x='metros_quadrados', y='aluguel')
plt.xlabel('Metros Quadrados')
plt.ylabel("Valor do Aluguel")
plt.savefig("./dataviz/metros-quadrados-x-aluguel-scatter.png")
plt.close()

sns.boxplot(data=df_aluguel_eda, y='aluguel')
plt.savefig("./dataviz/aluguel-hist.png")
plt.close()

sns.heatmap(
    data=df_aluguel_eda.corr("pearson"),
    vmin=-1,
    vmax=1,
    annot=True,
    cmap='crest',
)
plt.savefig("./dataviz/pearson-corr-heatmap.png")
plt.close()

sns.heatmap(
    data=df_aluguel_eda.corr("spearman"),
    vmin=-1,
    vmax=1,
    annot=True,
    cmap='crest',
)
plt.savefig("./dataviz/spearman-corr-heatmap.png")
plt.close()

sns.heatmap(
    data=df_aluguel_eda.corr("pearson")[['aluguel']].sort_values(by='aluguel', ascending=False),
    vmin=-1,
    vmax=1,
    annot=True,
    cmap='crest',
)
plt.title("Coeficiente de Pearson")
plt.savefig("./dataviz/aluguel-pearson-corr-heatmap.png")
plt.close()

sns.heatmap(
    data=df_aluguel_eda.corr("spearman")[['aluguel']].sort_values(by='aluguel', ascending=False),
    vmin=-1,
    vmax=1,
    annot=True,
    cmap='crest',
)
plt.title("Coeficiente de Spearman")
plt.savefig("./dataviz/aluguel-spearman-corr-heatmap.png")
plt.close()

# 3: Treinamento do Modelo
X = df_aluguel.drop(columns=['aluguel'])
y = df_aluguel['aluguel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51 )

colunas_categoricas_nominais = ['garagem', 'periferia', 'suburbio']
colunas_numericas = ['n_quartos', 'metros_quadrados', 'idade']

transformer_categoricas_nominais = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

transformer_numericas = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', transformer_numericas, colunas_numericas),
        ('cat', transformer_categoricas_nominais, colunas_categoricas_nominais)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X=X_train, y=y_train)

# 4: Fazer previsão do modelo
print("\nPrevisão teste do modelo:")
dict_predict = {
    'metros_quadrados': 174.86639612006326,
    'n_quartos': 4,
    'idade': 44.13181715946698,
    'garagem': False,
    'periferia': False,
    'suburbio': True,
}

df_predict = pd.DataFrame(dict_predict, index=[1])

aluguel_previsto = model.predict(X=df_predict)
print(f"Aluguel previsto: R$ {aluguel_previsto[0]:.2f}")

# 5: Análise do modelo
print("\nAnálise do modelo")

y_pred = model.predict(X=X_test)
residuos = y_test - y_pred
residuos_std = zscore(residuos)

# 5.1 - Análise de linearidade e homocedasticidade
mae = mean_absolute_error(y_pred=y_pred, y_true=y_test)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
rmse = root_mean_squared_error(y_pred=y_pred, y_true=y_test)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
r2score = r2_score(y_true=y_test, y_pred=y_pred)
print(f"R²-Score: {r2score:.4f}")

pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.title("Quantile-Quantile Plot")
plt.savefig("./dataviz/qqplot.png")
plt.close()

sns.scatterplot(x=y_pred, y=residuos_std) # type: ignore
plt.axhline(y=2, color='red')
plt.axhline(y=-2, color='red')
plt.title("Resíduos Std Scatter")
plt.savefig("./dataviz/residuos-std-scatter.png")
plt.close()

# 5.2 - Estatísticas de distribuição normal dos resíduos
stat_shapiro, p_value_shapiro = shapiro(residuos)
print(f"Shapiro-Wilk p-value: {p_value_shapiro}")
stat_ks, p_value_ks = kstest(residuos, 'norm')
print(f"Kolmogorv-Smirnov p-value: {p_value_ks}")
stat_ll, p_value_ll = lilliefors(x=residuos, dist='norm', pvalmethod='table')
print(f"Lilliefors p-value: {p_value_ll}")

X_test_transformed = model.named_steps['preprocessor'].transform(X=X_test)
test_goldfeld = het_goldfeldquandt(residuos, X_test_transformed)
p_value_goldfeld = test_goldfeld[1]
print(f"Goldfeld p-value: {p_value_goldfeld}")

# 6: Coeficientes
print("\nCoeficientes e Interceptor:")
regressor = model.named_steps['regressor']
print("Coeficientes:")
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": regressor.coef_
}).sort_values(by='coef', key=abs, ascending=False)
print(coef_df)
print(f"Interceptor: {regressor.intercept_}")

# 7: Salvar modelo
joblib.dump(model, "./modelo_aluguel.pkl")
