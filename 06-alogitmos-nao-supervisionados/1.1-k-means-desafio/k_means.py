import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import optuna
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# 1: Iniciar modelo
iris = load_iris(as_frame=True)
df_iris: pd.DataFrame = iris.frame # type: ignore
df_iris.info()
df_iris = df_iris.rename(columns={
  'sepal length (cm)': 'sepal_length',
  'sepal width (cm)': 'sepal_width',
  'petal length (cm)': 'petal_length',
  'petal width (cm)': 'petal_width',
  'target': 'target',
})
df_iris.info()
print(df_iris.describe())
df_modelo = df_iris.drop(columns=['target'])

# 2: Análise Exploratória dos Dados
# DIstribuição comprimento da pétala
sns.histplot(
  data=df_modelo,
  x='petal_length',
  color='blue',
)
plt.title("Petal Length Distribution")
plt.ylabel("Frequency")
plt.xlabel("Petal Length (cm)")
plt.grid(visible=True, alpha=0.3)
plt.savefig("./dataviz/petal-length-distribution.png")
plt.close()

# Distribuição largura da pétala
sns.histplot(
  data=df_modelo,
  x='petal_width',
  color='blue',
)
plt.title("Petal Width Distribution")
plt.ylabel("Frequency")
plt.xlabel("Petal Width (cm)")
plt.grid(visible=True, alpha=0.3)
plt.savefig("./dataviz/petal-width-distribution.png")
plt.close()

# Distribuição largura da sépala
sns.histplot(
  data=df_modelo,
  x='sepal_width',
  color='blue',
)
plt.title("Sepal Width Distribution")
plt.ylabel("Frequency")
plt.xlabel("Sepal Width (cm)")
plt.grid(visible=True, alpha=0.3)
plt.savefig("./dataviz/sepal-width-distribution.png")
plt.close()

# Distribuição largura da pétala
sns.histplot(
  data=df_modelo,
  x='sepal_length',
  color='blue',
)
plt.title("Sepal Length Distribution")
plt.ylabel("Frequency")
plt.xlabel("Sepal Length (cm)")
plt.grid(visible=True, alpha=0.3)
plt.savefig("./dataviz/sepal-length-distribution.png")
plt.close()

# 3: Treinar modelo
# OBS: EDA revela que as pétalas são candidatas fortes.
X = df_modelo.copy()

colunas_numericas = [
  'sepal_length',
  'sepal_width',
  'petal_length',
  'petal_width',
]

numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, colunas_numericas)
])

X_transformed = preprocessor.fit_transform(X)

def kmean_objective(trial: optuna.Trial):
    n_clusters = trial.suggest_int('n_clusters', 3, 10)
    distance = trial.suggest_categorical('distance', ['euclidean', 'cityblock', 'chebyshev'])

    model_opt = KMeans(
        n_clusters=n_clusters,
        random_state=51,
    )

    model_opt.fit(X_transformed)

    distances = pairwise_distances(X_transformed, metric=distance)
    score = silhouette_score(distances, model_opt.labels_)

    return score

search_space = {
    'n_clusters': [3, 4, 5, 6, 7, 8, 9, 10],
    'distance': ['euclidean', 'cityblock', 'chebyshev'],
}
sampler = optuna.samplers.GridSampler(search_space=search_space)
opt_study = optuna.create_study(
    sampler=sampler,
    direction='maximize',
    study_name='K-Means Study',
)
opt_study.optimize(
    kmean_objective, # type: ignore
    n_trials=50,
)

## Optuna revela que a melhor escala é a de Manhattan,
## logo KMeans não faz sentido por usar Euclidiana.
model_km = KMeans(
    n_clusters=opt_study.best_params['n_clusters'],
    random_state=51,
)

model_km.fit(X_transformed)

distances = pairwise_distances(X=X_transformed, metric=opt_study.best_params['distance'])
s_score = silhouette_score(distances, model_km.labels_) # type: ignore

print(f"\nDistância escolhida: {opt_study.best_params['distance']}")
print(f"Número de clusters escolhido: {opt_study.best_params['n_clusters']}")
print(f"Silhouette score: {s_score}")

df_iris['cluster'] = model_km.labels_

# 5: Análise de Dados
# Comparar classificação real com  clusters
df_iris['target'] = df_iris['target'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica',
})
crosstab = pd.crosstab(df_iris['target'], df_iris['cluster'], normalize='index')
sns.heatmap(
    crosstab,
    annot=True,
    cmap='Reds',
)
plt.ylabel("species")
plt.title("Crosstab: Classification x Cluster\n(normalizing indexes)")
plt.savefig("./dataviz/crosstab-result.png")
plt.close()

# Distribuição da largura das pétalas com hue do cluster
sns.histplot(
  data=df_iris,
  x='petal_width',
  hue='cluster',
  palette='magma',
)
plt.title("Petal Width Distribution")
plt.ylabel("Frequency")
plt.xlabel("Petal Width (cm)")
plt.grid(visible=True, alpha=0.3)
plt.savefig("./dataviz/petal-width-distribution-with-cluster.png")
plt.close()

# Distribuição largura da sépala com hue do cluster
sns.histplot(
  data=df_iris,
  x='sepal_width',
  hue='cluster',
  palette='magma',
)
plt.title("Sepal Width Distribution")
plt.ylabel("Frequency")
plt.xlabel("Sepal Width (cm)")
plt.grid(visible=True, alpha=0.3)
plt.savefig("./dataviz/sepal-width-distribution-with-cluster.png")
plt.close()