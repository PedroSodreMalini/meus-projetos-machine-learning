import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 1: Ler dados
df_frutas = pd.read_csv("./dataset/fruits.csv")
df_frutas.drop(columns=['A_id'], inplace=True)
df_frutas['Quality'] = df_frutas['Quality'].map({ 'good': 1, 'bad': 0 })

# 2: Treinar modelo
X = df_frutas.drop(columns=['Quality'])
y = df_frutas['Quality']

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.3,
  random_state=51
)

scores_train = []
scores_test = []

for i in range(1, 20, 2):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X=X_train, y=y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    f1_score_train = f1_score(y_pred=y_train_pred, y_true=y_train)
    f1_score_test = f1_score(y_pred=y_test_pred, y_true=y_test)
    scores_train.append(f1_score_train)
    scores_test.append(f1_score_test)
    print(f'{i}: Predição de treino: {f1_score_train} Predição de teste: {f1_score_test}')

df_results = pd.DataFrame({
    'k': range(1, 20, 2),
    'train': scores_train,
    'test': scores_test,
})

sns.scatterplot(
    data=df_results,
    x='k',
    y='train',
    label="Treino",
    color='blue',
)
sns.scatterplot(
    data=df_results,
    x='k',
    y='test',
    label="Teste",
    color='red',
)
plt.ylabel('F1-Score')
plt.xlabel("K value")
plt.title("KNN Performance - Mudando K")
plt.savefig("./dataviz/knn-changing-k.png")
