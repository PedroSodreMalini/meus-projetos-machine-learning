import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
# 1: Ler dados
df_diabetes = pd.read_csv("./dataset/diabetes.csv")
print(df_diabetes.info())
print(df_diabetes.describe())

# 2: EDA
df_diabetes_eda = df_diabetes.copy()
df_diabetes_eda['diabetes_str'] = df_diabetes_eda['diabetes'].map({ 0: "Não", 1: "Sim"})

# Mostra uma distribuiçaõ normal dos dados de glicemia
sns.histplot(
  data=df_diabetes_eda,
  x='glicemia',
  color='purple',
)
plt.ylabel("Frequência")
plt.xlabel("Glicemia")
plt.title("Histograma - Glicemia")
plt.savefig("./dataviz/glicemia-histogram.png")
plt.close()

# Os extremos são diabeticos, já no centro a tendência é ter menos diabéticos
sns.histplot(
  data=df_diabetes_eda,
  x='glicemia',
  palette='coolwarm',
  hue='diabetes_str',
)
plt.ylabel("Frequência")
plt.xlabel("Glicemia")
plt.title("Histograma baseado em Diabetes - Glicemia")
plt.savefig("./dataviz/glicemia-hue-histogram.png")
plt.close()

# tem outliers com 70 e 20 de glicemia
# |-- É realmente possível, mas o caso de 20 é quase absurdo.
# |-- Pesquisando, o excesso de insulina pode causar isso em um diabético
sns.boxplot(
  data=df_diabetes_eda,
  y='glicemia',
  fill=False,
  color='purple'
)
plt.ylabel("Glicemia")
plt.savefig("./dataviz/glicemia-boxplot.png")
plt.close()

# comparando ambos boxplot de diabeticos e nao diabeticos
sns.boxplot(
  data=df_diabetes_eda,
  y='glicemia',
  palette='coolwarm',
  hue='diabetes_str',
)
plt.ylabel("Glicemia")
plt.title("Boxplot com base em diabetes - Glicemia")
plt.savefig("./dataviz/glicemia-hue-boxplot.png")
plt.close()

# Distribuição da pressão arterial é mais aleatória com crescimento à direita.
sns.histplot(
  data=df_diabetes_eda,
  x='pressao_arterial',
  color='purple',
)
plt.title("Histograma - Pressão Arterial")
plt.ylabel("Frequência")
plt.xlabel("Pressão Arterial")
plt.savefig("./dataviz/pressao-arterial-histogram.png")
plt.close()

# Há um claro deslocamento à direita do histograma para pessoas com diabetes
# |-- Pessoas diabéticas tem pressão mais alta no geral.
sns.histplot(
  data=df_diabetes_eda,
  x='pressao_arterial',
  palette='coolwarm',
  hue='diabetes_str'
)
plt.title("Histograma com base em diabetes - Pressão Arterial")
plt.ylabel("Frequência")
plt.xlabel("Pressão Arterial")
plt.savefig("./dataviz/pressao-arterial-hue-diabetes-histogram.png")
plt.close()

# Diabéticos assume uma distribuição uniforme.
porcentagem_diabeticos = df_diabetes_eda['diabetes_str'].value_counts(normalize=True)
sns.barplot(
  x=porcentagem_diabeticos.index,
  y=porcentagem_diabeticos.values,
  color='cyan',
  linewidth=1,
  edgecolor='black',
  width=.2
)
plt.title("Porcentagem de Diabéticos e Não Diabéticos")
plt.ylabel("Porcentagem")
plt.xlabel("É diabético?")
plt.savefig("./dataviz/diabetes-porcentagem-barplot.png")
plt.close()

# 3: Treinamento do modelo
X = df_diabetes.drop(columns=['diabetes'])
y = df_diabetes.drop(columns=['pressao_arterial', 'glicemia'])

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  train_size=0.8,
  shuffle=True,
  random_state=51
)

model = GaussianNB()
model.fit(X=X_train, y=y_train)
y_pred = model.predict(X=X_test)

# 4: Métricas
classification_report_str = classification_report(
  y_pred=y_pred,
  y_true=y_test
)
print(f"Classification Report:\n{classification_report_str}")

confusion_matrix_result = confusion_matrix(
  y_pred=y_pred,
  y_true=y_test
)

disp_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix_result)

disp_confusion_matrix.plot(cmap='Purples')
plt.title("Matriz de Confusão")
plt.savefig("./dataviz/matriz-de-confusao.png")
plt.close()

# Disponibilizar modelo
joblib.dump(model, "modelo_diabetes.pkl")