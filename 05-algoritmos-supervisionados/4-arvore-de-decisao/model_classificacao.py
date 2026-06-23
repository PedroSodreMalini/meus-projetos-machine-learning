import joblib
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
import plotly.express as px
import plotly.figure_factory as ff
import optuna
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1: Carregar os Dados
print("Carregar os dados:")
df_segmento = pd.read_csv("./dataset/dataset_empresas.csv")
print(df_segmento.head())
print(df_segmento.info())
print(df_segmento.describe())
print(df_segmento["atividade_economica"].unique())
print(df_segmento["localizacao"].unique())
print(df_segmento["inovacao"].unique())
print(df_segmento["segmento_de_cliente"].unique())

# 2: Análise exploratória dos dados

# 2.1 - Análise das Variáveis Qualitativas
print("\nAnálise exploratória dos dados (EDA):")

contagem_target = df_segmento.value_counts('segmento_de_cliente')
lista_ordenada_target = ['Starter', 'Bronze', 'Silver', 'Gold']
print(contagem_target)

# Gráfico de barras percentual (densidade) da variável target por categoria
# Grande maioria é Silver ou Bronze
percentual_target = contagem_target / len(df_segmento)
fig = px.bar(
  percentual_target,
  color=percentual_target.index,
  category_orders={
    'segmento_de_cliente': lista_ordenada_target
  }
)
fig.write_image("./dataviz/segmentos-densidade.png")

# Gráfico de barras percentual (densidade) da variável localização por cidade
# OBS: localização não é ordinal, mas nominal, logo não tem ordem natural.
# assimetria positiva leve
percentual_localizacao = df_segmento.value_counts('localizacao') / len(df_segmento)
fig = px.bar(
  percentual_localizacao,
  color=percentual_localizacao.index,
)
fig.write_image("./dataviz/localizacao-densidade.png")

# Gráfico de barras percentual (densidade) da variável atividade economica
# OBS: atividade economica não é ordinal, mas nominal, logo não tem ordem natural.
# assimetria positiva leve
percentual_atividade = df_segmento.value_counts('atividade_economica') / len(df_segmento)
fig = px.bar(
  percentual_atividade,
  color=percentual_atividade.index,
)
fig.write_image("./dataviz/atividade-densidade.png")

# Gráfico de barras percentual (densidade) da variável inovação
# parece uma distribuição uniforme
percentual_inovacao = df_segmento.value_counts('inovacao') / len(df_segmento)
fig = px.bar(
  percentual_inovacao,
  color=percentual_inovacao.index,
)
fig.write_image("./dataviz/inovacao-densidade.png")

# 2.2- Crosstab Variáveis Qualitativas

# Tabela de contingência localização x target
# Não mostra uma relação tão clara
crosstab_localizacao = pd.crosstab(
  df_segmento['localizacao'],
  df_segmento['segmento_de_cliente'],
  margins=True
)[lista_ordenada_target].reset_index()

tabela_localizacao = ff.create_table(crosstab_localizacao)
tabela_localizacao.write_image("./dataviz/contingencia-localizacao-x-target.png")

# Tabela de contingência atividade economica x target
# Não mostra uma relação tão clara
crosstab_atividade = pd.crosstab(
  df_segmento['atividade_economica'],
  df_segmento['segmento_de_cliente'],
  margins=True
)[lista_ordenada_target].reset_index()

tabela_atividade = ff.create_table(crosstab_atividade)
tabela_atividade.write_image("./dataviz/contingencia-atividade-x-target.png")

# Tabela de contingência inovacao x target
# Starter e Bronze = concentradas em inovações baixas
# Silver = boa parte com inovação médio/alto
# Gold = inovacao acima de 5
crosstab_inovacao = pd.crosstab(
  df_segmento['inovacao'],
  df_segmento['segmento_de_cliente'],
  margins=True
)[lista_ordenada_target].reset_index()

tabela_inovacao = ff.create_table(crosstab_inovacao)
tabela_inovacao.write_image("./dataviz/contingencia-inovacao-x-target.png")

# 2.3 - Análise de Variáveis Quantitativas

# Idade mostra algo próximo da distribuição normal
fig = px.histogram(df_segmento, x='idade')
fig.write_image("./dataviz/idade-hist.png")

# Faturamento também mostra uma espécie de distribuição normal
fig = px.histogram(df_segmento, x="faturamento_mensal")
fig.write_image("./dataviz/faturamento-hist.png")

# Boxplot idade e segmento
# Mediana de idade aumenta de acordo com os segmentos
# Contudo é possível encontrar empresas de idades variadas em cada segmento. 
fig = px.box(
    df_segmento,
    x='segmento_de_cliente',
    y='idade',
    color='segmento_de_cliente',
    category_orders={
        'segmento_de_cliente': lista_ordenada_target
    }
)
fig.write_image("./dataviz/idade-x-target-boxplot.png")

# Boxplot faturamento e segmento
# Mediana de faturamento mensal aumenta de acordo com os segmentos
# Contudo é possível encontrar empresas de faturamentos variadas em cada segmento.
# Há outliers nas empresas Bronze:
# | Tem empresas que tem um faturamneto alto, mas ainda são bronze.
# Para empresas starters vê-se que grande maioria fatura menos que o resto
fig = px.box(
    df_segmento,
    x='segmento_de_cliente',
    y='faturamento_mensal',
    color='segmento_de_cliente',
    category_orders={
        'segmento_de_cliente': lista_ordenada_target
    }
)
fig.write_image("./dataviz/faturamento-x-target-boxplot.png")

# 2.4 - Correlação Variáveis Qualitativas

# Chi Squared Test de Pearson (Chi² Test) localização x categoria
# - H0 - As variáveis são independentes
# - H1 - As variáveis não são independentes
# - p-value > 0.05: Não rejeita a hipótese nula
# | Se for menor ou igual a 0.05, há relação !!!!
# Valor esperado = frequencia que seria esperada se não houvesse associação entre as variáveis.
# | É calculado utilizando a distribuição assumida no teste qui-quadrado
# Valor observado = É a frequência real dos dados coletados
# P-value = p valor
# P-valor obtido = 0.8171 -> não rejeita H0
print("Teste chi² localizacao x segmento do cliente:")
valor_esperado, valor_observado, estatisticas = pg.chi2_independence(
    df_segmento,
    'segmento_de_cliente',
    'localizacao'
)
print("Valor esperado do teste Chi²:")
print(valor_esperado)
print("Valor observado no teste Chi²:")
print(valor_observado)
print("Estatísticas do teste Chi²:")
print(estatisticas)

# Chi² Test atividade economica x categoria
# p-value = 0.35 -> não rejeita a hipótese nula
print("Teste chi² atividade economica x segmento do cliente:")
_, _, estatisticas = pg.chi2_independence(
    df_segmento,
    'segmento_de_cliente',
    'atividade_economica'
)
print(estatisticas.round(5))

# Chi² Test atividade inovação x categoria
# p-value = 0 -> rejeita a hipótese nula e aceita a complementar
print("Teste chi² atividade inovação x segmento do cliente:")
_, _, estatisticas = pg.chi2_independence(
    df_segmento,
    'segmento_de_cliente',
    'inovacao'
)
print(estatisticas.round(5))

# 3: Treinamento do modelo
print("\nTreinamento do modelo:")

X = df_segmento.drop(columns=['segmento_de_cliente'])
y = df_segmento['segmento_de_cliente']

categorical_features = ['atividade_economica', 'localizacao']

categorical_tranformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_tranformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# Validação Cruzada - StratifiedKFold, dado que as classes estão desbalanceadas
# Métricas de acurácia para cada um dos splits: [0.47904192, 0.47305389, 0.46987952]
cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=51)
metrics_result = cross_validate(
    model,
    X,
    y,
    cv=cv_folds,
    scoring=['accuracy'],
    return_estimator=True
)
print(metrics_result)

# Média da acurácia considerando os 3 splits
# Está acertando menos que a metade, então não tá muito bom
media_accuracy = metrics_result['test_accuracy'].mean()
print(f"Média de acurácia: {media_accuracy}")

# 3.2 - Métricas do modelo
# Fazendo predições usando cross validation
y_pred = cross_val_predict(model, X, y, cv=cv_folds)
classification_report_str = classification_report(y_true=y, y_pred=y_pred)
print(f"Relatório de classificação:\n{classification_report_str}")

# Matriz de Confusão
# Diagonal são os acertos
model_confusion_matrix = confusion_matrix(
    y_true=y,
    y_pred=y_pred,
    labels=lista_ordenada_target,
)

disp = ConfusionMatrixDisplay(
    model_confusion_matrix,
    display_labels=lista_ordenada_target    
)

disp.plot(cmap='Blues')

plt.savefig("./dataviz/matriz-confusao.png")
plt.close()

# 3.3 - Tuning de Hiperparâmetros
# Optuna: Automatiza busca por melhores hiperparâmetros
# - min_samples_leaf = Mínimo de instâncias requerido para formar uma folha
# - max_depth = Profundidade máxima da árvore
def decisiontree_optuna(trial):
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_depth = trial.suggest_int('max_depth', 2, 8)

    model.set_params(classifier__min_samples_leaf=min_samples_leaf)
    model.set_params(classifier__max_depth=max_depth)

    scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")

    return scores.mean()

# estudo tem o objetivo de maximizar a função acima
# printa o best try na tela
estudo_decision_tree = optuna.create_study(direction='maximize')
estudo_decision_tree.optimize(decisiontree_optuna, n_trials=100)
print(f"Melhor acurácia: {estudo_decision_tree.best_value}")
print(f"Melhores parâmetros: {estudo_decision_tree.best_params}")

# 3.4 - Visualizar Árvore de Decisão

# Preparar o Conjunto de Dados para treinar e conseguir visualizar a árvore
X_train_tree = X.copy()
X_train_tree['localizacao_label'] = X_train_tree.localizacao.astype('category').cat.codes
X_train_tree['atividade_label'] = X_train_tree.atividade_economica.astype("category").cat.codes
X_train_tree.drop(columns=['localizacao', 'atividade_economica'], inplace=True)
X_train_tree.rename(columns={
    'localizacao_label': 'localizacao',
    'atividade_label': 'atividade'
}, inplace=True)
print(X_train_tree.head(5))

# Treinar o modelo com o conjunto de hiperparâmetros ideal
clf_decision_tree = DecisionTreeClassifier(
    min_samples_leaf=estudo_decision_tree.best_params['min_samples_leaf'],
    max_depth=estudo_decision_tree.best_params['max_depth']
)

y_train_tree = y.copy()

clf_decision_tree.fit(X_train_tree, y_train_tree)

# Visualizar árvore de decisão
# Modelo não coinseguiu identificar diferenas claras entre bronze e silver
plt.subplots(nrows=1, ncols=1, figsize=(10,10), dpi=600)
plot_tree(
    decision_tree=clf_decision_tree,
    feature_names=X_train_tree.columns.to_numpy(), # type: ignore
    class_names=lista_ordenada_target,
    filled=True
)
plt.savefig("./dataviz/decision-tree.png")
plt.close()

# 4: Salvar modelo
print("\nSalvar modelo:")
model_tunado = Pipeline(steps=[
    (
        'preprocessor', 
        preprocessor
    ),
    (
        'classifier', 
        DecisionTreeClassifier(
            min_samples_leaf=estudo_decision_tree.best_params['min_samples_leaf'],
            max_depth=estudo_decision_tree.best_params['max_depth'])
    ),
])

model_tunado.fit(X, y)
joblib.dump(model_tunado, "./classification_model.pkl")