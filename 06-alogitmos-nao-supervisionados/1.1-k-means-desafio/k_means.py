import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pingouin
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
