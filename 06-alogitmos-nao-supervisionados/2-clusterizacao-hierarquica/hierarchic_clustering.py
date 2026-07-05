import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import optuna
import joblib

# 1: Ler dados
# Desvio do preço é muito alto (max: 5450.88, min:117.54)
print("Ler dados")
df_laptops = pd.read_csv("./dataset/laptops.csv")
df_laptops.info()
print(df_laptops.head(10))
print(df_laptops.describe())

# 2: Limpeza dos dados
# Anos de garantia
# Como a larga maioria tem 1 ano de garantia, setar No information para 1.
print("\n\nLimpar dados")
print("Quantidade percentual de registros por ano de garantia:")
print(df_laptops['year_of_warranty'].value_counts(normalize=True))
df_laptops['year_of_warranty'] = df_laptops['year_of_warranty'].map({
  '1': 1,
  '2': 2,
  '3': 3,
  'No information': 1,
})

# É touch screen
# Mudar de booleano para int.
df_laptops['is_touch_screen'] = df_laptops['is_touch_screen'].map({
  False: 0,
  True: 1,
})

print("\nDados após tratamento de Touch Screen e Anos de Garantia:")
df_laptops.info()

# 3: Análise Exploratória dos Dados
# 3.1 - Análise Univariada
# Distribuição da variável brand
distribuicao_brand = df_laptops['brand'].value_counts(normalize=True)
plt.figure(figsize=(24,6))
sns.barplot(
  x=distribuicao_brand.index.values,
  y=distribuicao_brand.values,
  hue=distribuicao_brand.index.values,
  palette='bright',
)
plt.title("Brands Percentual Frequency")
plt.ylabel("Percentual Frequency")
plt.xlabel("Brand")
plt.grid(visible=True, alpha=0.3)
plt.savefig(
  "./dataviz/brand-distribution.png",
  bbox_inches='tight',
)
plt.close()

# Distribuição da variável processor brand
distribuicao_processor_brand = df_laptops['processor_brand'].value_counts(normalize=True)
sns.barplot(
  x=distribuicao_processor_brand.index.values,
  y=distribuicao_processor_brand.values,
  hue=distribuicao_processor_brand.index.values,
  palette='Reds',
  legend=True,
)
plt.title("Processor Brands Percentual Frequency")
plt.ylabel("Percentual Frequency")
plt.xlabel("Processor Brand")
plt.grid(visible=True, alpha=0.3)
plt.savefig("./dataviz/processor-brand-distribution.png")
plt.close()

# Distribuição dos preços
sns.histplot(
  x=df_laptops['price'],
  color='red',
  kde=True,
)
plt.title("Price Histplot")
plt.ylabel("Price Frequency")
plt.xlabel("Prices")
plt.grid(visible=True, alpha=0.3)
plt.savefig("./dataviz/price-distribution.png")
plt.close()

# Distribuição rating
sns.histplot(
  x=df_laptops['rating'],
  color='green',
  kde=True,
)
plt.title("Rating Histplot")
plt.ylabel("Rating Frequency")
plt.xlabel("Ratings")
plt.grid(visible=True, alpha=0.3)
plt.savefig("./dataviz/rating-distribution.png")
plt.close()

# 3.2 - Análise Bivariada
# Preço e Marca
plt.figure(figsize=(7,6))
sns.boxplot(
  data=df_laptops,
  x='price',
  y='brand',
  hue='brand',
  orient='h',
  palette='magma',
)
plt.title("Boxplot Price x Brand")
plt.ylabel("Brand")
plt.xlabel("Price")
plt.grid(visible=True, alpha=0.3)
plt.savefig("./dataviz/price-and-brand-boxplot.png", bbox_inches='tight')
plt.close()

# Avaliação e Marca
plt.figure(figsize=(7,6))
sns.boxplot(
  data=df_laptops,
  x='rating',
  y='brand',
  hue='brand',
  orient='h',
  palette='magma',
)
plt.title("Boxplot Rating x Brand")
plt.ylabel("Brand")
plt.xlabel("Rating")
plt.grid(visible=True, alpha=0.3)
plt.savefig("./dataviz/rating-and-brand-boxplot.png", bbox_inches='tight')
plt.close()

# Preço e Rating
sns.scatterplot(
  data=df_laptops,
  x='price',
  y='rating',
  color='purple',
)
plt.grid(visible=True, alpha=.3)
plt.title("Rating x Price Scatterplot")
plt.xlabel("Price")
plt.ylabel("Rating")
plt.savefig('./dataviz/price-and-rating-scatter.png')
plt.close()

# 4: Treinamento do modelo
X = df_laptops.copy()
X.drop(columns=['index', 'model'], inplace=True)

numeric_features = [
  'price',
  'rating',
  'num_cores',
  'num_threads',
  'ram_memory',
  'primary_storage_capacity',
  'display_size',
  'resolution_width',
  'resolution_height',
]
categorical_features = [
  'brand',
  'processor_brand',
  'gpu_brand',
  'gpu_type',
  'os',
]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(transformers=[
  ('num', numeric_transformer, numeric_features),
  ('cat', categorical_transformer, categorical_features),
])

# fica como matriz esparsa devido ao alto número de colunas
# --> deve ser transformado em array com .toarray()
X_transformed = preprocessor.fit_transform(X=X)

# 4.1 - Clusterização Hierárquica Aglomerativa (Bottom-Up)
def hierarchical_aglomeration_objective(trial: optuna.Trial):
    n_clusters = trial.suggest_int('n_cluster', 10, 150)
    
    # Critério de aglomeração
    # Ward: Usa variância da distância de 2 conmjuntos de dados
    # Average: Usa a média das distâncias de 2 conjuntos de dados
    # Complete: Usa a distância máxima entre 2 conjuntos de dados
    # Single: Usa a distância mínima entre 2 conjuntos de dados
    linkag = trial.suggest_categorical(
      'linkage',
      ['ward', 'average', 'complete', 'single'],
    )

    model_hca = AgglomerativeClustering(
      linkage=linkag, # type: ignore
      n_clusters=n_clusters,
    )

    y = model_hca.fit_predict(X_transformed.toarray()) # type: ignore

    s_score = silhouette_score(X_transformed, y)

    return s_score

search_space = {
    'n_cluster': list(range(10, 150+1)),
    'linkage': ['ward', 'average', 'complete', 'single'],
}
sampler = optuna.samplers.GridSampler(search_space=search_space)
opt_study_agg = optuna.create_study(
    sampler=sampler,
    study_name='Agglomerative Hierarchic Clustering Study',
    direction='maximize'
)
opt_study_agg.optimize(
  func=hierarchical_aglomeration_objective, # type: ignore
  n_trials=1000,
)

best_params_agg = opt_study_agg.best_params

print("\nAgglomeration Method")
print(f"Quantidade de clusters: {best_params_agg['n_cluster']}")
print(f"Linkage: {best_params_agg['linkage']}")
print(f'Silhouette Score: {opt_study_agg.best_value}')

# 4.2 - Clusterização Hierárquica Divisiva (Top-Down)
def hierarchical_division_objective(trial: optuna.Trial):
    n_clusters = trial.suggest_int('n_cluster', 10, 150)

    hierarchical_model = BisectingKMeans(n_clusters=n_clusters)

    y = hierarchical_model.fit_predict(X_transformed.toarray()) # type: ignore

    s_score = silhouette_score(X_transformed, y)

    return s_score

search_space = {
    'n_cluster': list(range(10, 150+1)),
}
sampler = optuna.samplers.GridSampler(search_space=search_space)
opt_study_div = optuna.create_study(
    sampler=sampler,
    study_name='Divisive Hierarchic Clustering Study',
    direction='maximize'
)
opt_study_div.optimize(
  func=hierarchical_division_objective, # type: ignore
  n_trials=1000,
)

best_params_div = opt_study_div.best_params

print("\nDivision Method:")
print(f"Quantidade de clusters: {best_params_div['n_cluster']}")
print(f'Silhouette Score: {opt_study_div.best_value}')

# 5: Gerar melhor modelo
# OBS: Nesse caso, método aglomeraivo ganhou,
# |-- logo selecionaremos esse
# Contudo seu silhouette score é baixo(≃ 0.33), logo não é um bom agrupamento. 
best_model = AgglomerativeClustering(
      linkage=best_params_agg['linkage'],
      n_clusters=best_params_agg['n_cluster'],
)

best_model.fit(X_transformed.toarray()) # type: ignore

best_s_score = silhouette_score(
  X_transformed,
  best_model.labels_,
)

df_laptops['cluster'] = best_model.labels_

print("\nDataset com clusters")
df_laptops.info()

# 6: Análise de dados com clusters
# 6.1 - Dendograma
# fig = optuna.visualization.plot_optimization_history(
#   study=opt_study_agg,
# )
# fig.write_image("./dataviz/optimization_history.png")

# Printar dendrograma
modelo_dendrogram = linkage(
  X_transformed.toarray(), # type: ignore
  method=best_params_agg['linkage'],
  optimal_ordering=True,
)
plt.figure(figsize=(10, 6))
dendrogram(
  modelo_dendrogram,
  truncate_mode='lastp', # últimos p níveis do dendrograma
  p=15, # p = número de clusters desejados
  leaf_rotation=90, # ver dendrograma de cima para baixo
  leaf_font_size=10,
)
plt.title("Agglomerative Hierarchic Clustering Dendrogram")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.savefig("./dataviz/dendrogram-15-clusters.png")
plt.close()

# Cortar dendrograma
clusters_de_scipy = cut_tree(modelo_dendrogram, height=32)
print(f'\nNúmero de clusters no corte de altura 32: {len(np.unique(clusters_de_scipy))}')

# 6.2 - Resultados da clusterização
# Price x Cluster x Brand
plt.figure(figsize=(24,6))
sns.scatterplot(
    data=df_laptops,
    x='cluster',
    y='price',
    hue='brand',
    palette='coolwarm',
    legend=False,
)
plt.title("Price x Cluster Scatterplot")
plt.ylabel("Price")
plt.xlabel("Cluster")
plt.savefig("./dataviz/price-cluster-brand-scatter.png", bbox_inches='tight')
plt.close()

# Rating x Cluster x Brand
plt.figure(figsize=(24,6))
sns.scatterplot(
    data=df_laptops,
    x='cluster',
    y='rating',
    hue='brand',
    palette='coolwarm',
    legend=False,
)
plt.title("Rating x Cluster Scatterplot")
plt.ylabel("Rating")
plt.xlabel("Cluster")
plt.savefig("./dataviz/rating-cluster-brand-scatter.png", bbox_inches='tight')
plt.close()

# Distribuição Clusters
percentual_clusters = df_laptops.value_counts(
  "cluster",
  normalize=True, # type: ignore
).sort_index(ascending=True)
plt.figure(figsize=(18,6))
sns.barplot(
  x=percentual_clusters.index.values,
  y=percentual_clusters.values,
  hue=percentual_clusters.index.values,
  palette='magma',
  legend=False,
)
plt.title("Cluster Distribution")
plt.ylabel("Frequency")
plt.savefig("./dataviz/cluster-distribution.png", bbox_inches='tight')
plt.close()

# Distribuição Marcas x Cluster
crosstable_cluster_brand = pd.crosstab(
    df_laptops['brand'],
    df_laptops['cluster']
)
crosstable_cluster_brand.to_csv("./dataset/crosstable-cluster-brand.csv")

# 7: Salvar modelo e dataset
joblib.dump(best_model, './model_hca.pkl')
df_laptops.to_csv('./dataset/laptops-with-cluster.csv')
