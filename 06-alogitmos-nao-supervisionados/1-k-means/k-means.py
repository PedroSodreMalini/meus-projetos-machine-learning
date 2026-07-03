import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bartlett, shapiro
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from pingouin import welch_anova
import optuna
import joblib

# Valores reais impressos com 2 dígitos decimais
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 1: Ler dados
df_customers = pd.read_csv("./dataset/clientes.csv")
df_customers.info()
print(df_customers.head(10))

# 2: Análise Exploratória de Dados
# Distribuição da variável inovação
# Distribuição uniforme aproximadamente
percentual_inovacao = df_customers.groupby('inovacao').count().rename(columns={
  'idade': 'count'
})
percentual_inovacao['count'] = percentual_inovacao['count'] / len(df_customers) * 100
ax = sns.barplot(
  data=percentual_inovacao,
  x='inovacao',
  y='count',
  hue='inovacao',
  palette='rainbow',
  legend=False,
)
plt.title("Distribuição Percentual - Inovação")
plt.ylabel("Porcentagem")
plt.xlabel("Grau de Inovação")
plt.grid(visible=True, alpha=0.5)
plt.savefig("./dataviz/distribuicao-inovacao.png", dpi=600)
plt.close()

# Teste ANOVA (Análise de Variância)
# Verifica se há diferenças significativas na média de faturamento
# |-- mensal para diferentes níveis de inovação
# Suposições / Pressupostos:
# -- Observações devem ser independentes uma das outras.
# -- Variável dependente é contínua.
# -- Segue uma distribuição normal.
# -- Há homogeneidade das variâncias.
# -- Amostras sejam de tamanhos iguais. 

# Checar se as variâncias (faturamento) entre os grupos (inovação)
# |-- são homogêneas
# Aplicar teste de Bartlett
# -- H0: Evidência de que as variâncias são iguais 
# -- H1: Evidência de que as variâncias não são iguais
# -- p_value >= 0.05 = não rejeita H0.
# -- p_value < 0.05 = rejeita H0 e não rejeita H1.
# Resultado: Não rejeita H0.
# Separar os dados de faturamento em grupos com base na coluna 'inovacao'
print("\nTestar homogeneidade das variâncias de faturamento entre grupos de inovação:")
dados_agrupados = [df_customers['faturamento_mensal'][df_customers['inovacao'] == grupo] for grupo in df_customers['inovacao'].unique()]
_, bartlett_p_value = bartlett(*dados_agrupados)
print(f"Bartlett p-value: {bartlett_p_value:.6f}")

# Aplicar teste de Shapiro-Wilk
# Checar se há distribuição normal no faturamento mensal.
# -- H0: Evidência de que a variável segue uma distribuição normal. 
# -- H1: Evidência de que as variável segue uma distribuição normal.
# -- p_value >= 0.05 = não rejeita H0.
# -- p_value < 0.05 = rejeita H0 e não rejeita H1.
# Resultado: Não rejeita H0
print("\nTestar ditribuição normal de faturamento mensal:")
_, shapiro_p_value = shapiro(df_customers['faturamento_mensal'])
print(f"Shapiro-Wilk p-value: {shapiro_p_value:.6f}")

# Como há evidência de homogeneidade das variâncias de fatramento entre grupos de inovação
# |-- e evidência de distribuição normal do faturamento mensal
# |-- é possível fazer o teste ANOVA.
# OBS: Como as amostras não tem tamanhos iguais, fará Welsh Anova,
# |-- que é mais adequado para esses casos de distribuição não uniforme.
# Aplicar a ANOVA de Welch, pois as amostras são de tamanhos diferentes
# -- H0: Há similaridade(ou diferenças insignificativas) entre as médias dos grupos
# -- H1: Não há similaridade entre as médias dos grupos
# -- p_value >= 0.05 = não rejeita H0.
# -- p_value < 0.05 = rejeita H0 e não rejeita H1.
print("\nTeste Welch ANOVA")
aov = welch_anova(
  data=df_customers,
  dv='faturamento_mensal',
  between='inovacao',
)
print(f"P-value do Teste de ANOVA Welch: {aov.loc[0, 'p_unc']}") # type: ignore

# 3: Treinar algoritmo K-Means e Optuna
X = df_customers.copy()

numeric_features = ['faturamento_mensal', 'numero_de_funcionarios', 'idade']
categorical_features = ['atividade_economica', 'localizacao']
ordinal_features = ['inovacao']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()
ordinal_transformer = OrdinalEncoder()

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
    ('ord', ordinal_transformer, ordinal_features),
  ]
)

X_transformed = preprocessor.fit(X).transform(X) # Mesma coisa que fit_transform(X)
print("\nDataframe alterado para servir de input ao modelo:")
print(X_transformed)

def kmeans_objective(trial):
    n_clusters = trial.suggest_int('n_clusters', 3, 10)
    distance_metric = trial.suggest_categorical('distance_metric', ['euclidean', 'cityblock'])
    
    model_km = KMeans(
        n_clusters=n_clusters,
        random_state=51
    )

    model_km.fit(X_transformed)

    distances = pairwise_distances(X_transformed, metric=distance_metric)
    silhouette_avg = silhouette_score(distances, model_km.labels_)

    return silhouette_avg

search_space = {
    'n_clusters': [3, 4, 5, 6, 7, 8, 9, 10],
    'distance_metric': ['euclidean', 'cityblock']
}
sampler = optuna.samplers.GridSampler(search_space=search_space)
estudo_kmeans = optuna.create_study(
  sampler=sampler,
  study_name='K-means',
  directions=['maximize'],
)
estudo_kmeans.optimize(kmeans_objective, n_trials=20) # type: ignore

# 4: Melhor configuração encontrada pelo Optuna
best_params = estudo_kmeans.best_params

model_km = KMeans(
    n_clusters=best_params['n_clusters'],
    random_state=51,
)

model_km.fit(X_transformed)

distances = pairwise_distances(X_transformed, metric=best_params['distance_metric'])
best_silhouette = silhouette_score(distances, model_km.labels_)

print(f'k (Número de clusters): {best_params["n_clusters"]}')
print(f'Distância selecionada: {best_params["distance_metric"]}')
print(f'Silhouette Score: {best_silhouette}')

df_customers['cluster'] = model_km.labels_ # coloca os clusters

# Silhouette Score resulta em 0.44, o que é fraco.
# Separa em 3 clusters
# Isso se deve provavelmente ao mal da dimensionalidade.

# 5: Visualizar resultados
# Obs: ordem do cluster não importa.
# Cruzar idade e faturamento, apresentando clusters
sns.scatterplot(
    data=df_customers,
    x='idade',
    y='faturamento_mensal',
    hue='cluster',
    palette='bright',
)
plt.ylabel("Faturamento Mensal")
plt.xlabel("Idade")
plt.title("Faturamento Mensal x Idade")
plt.grid(visible=True)
plt.savefig("./dataviz/idade-faturamento-mensal-cluster.png")
plt.close()

# cruzar faturamento, inovacao e clusters
sns.scatterplot(
    data=df_customers,
    x='inovacao',
    y='faturamento_mensal',
    hue='cluster',
    palette='bright',
)
plt.ylabel("Faturamento Mensal")
plt.xlabel("Inovação")
plt.title("Faturamento Mensal x Inovação")
plt.grid(visible=True)
plt.savefig("./dataviz/inovacao-faturamento-mensal-cluster.png")
plt.close()

# cruzar numero de funcionarios, faturamento mensal e clusters
sns.scatterplot(
    data=df_customers,
    x='numero_de_funcionarios',
    y='faturamento_mensal',
    hue='cluster',
    palette='bright',
)
plt.ylabel("Faturamento Mensal")
plt.xlabel("Número de Funcionários")
plt.title("Faturamento Mensal x Número de Funcionários")
plt.grid(visible=True)
plt.savefig("./dataviz/numero-de-funcionarios-faturamento-mensal-cluster.png")
plt.close()

# 5: Salvar modelo
joblib.dump(model_km, 'clustering_model_km.pkl')
joblib.dump(preprocessor, 'clustering_pipeline.pkl')