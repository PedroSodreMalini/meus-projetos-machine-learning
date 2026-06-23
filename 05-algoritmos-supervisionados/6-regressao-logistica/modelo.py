import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from pingouin import ttest
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, log_loss, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# 1: Carregar dados
# Não há valores faltando.
# Colunas todas menos target e id foram convertidos para um scaler.
df_frutas = pd.read_csv("./dataset/frutas.csv")
print(df_frutas.head(3))
df_frutas.info()

# 2: Análise Exploratória de Dados (EDA)
# 2.1 - Análise Target e Transformação de Variáveis
# Frequência de good e bad -> Quase uniforme
sns.barplot(
  x=df_frutas['Quality'].value_counts(normalize=True).index,
  y=df_frutas['Quality'].value_counts(normalize=True).values,
  color='purple',
  width=0.2
)
plt.xlabel("Classe")
plt.ylabel("Qualidade")
plt.title("Gráfico de Barras - Qualidade")
plt.savefig("./dataviz/quality-barplot.png")
plt.close()

print("Alterando dataset - tirando id e colocando quality para int(0 e 1)")
df_frutas['Quality'] = df_frutas["Quality"].map({ 'bad': 0, 'good': 1}).astype(int)
df_frutas.drop(columns=['A_id'], inplace=True)
df_frutas.info()

# 2.2 - Análise Bivariada
sns.pairplot(
  data=df_frutas,
  diag_kind='hist',
)
plt.title("Pairplot")
plt.savefig("./dataviz/pairplot.png")
plt.close()

# Olhando peso a partir de quality
# |-- Amplitude de frutas boas é maior para o peso
# |-- Mediana de peso é quase similar para ambos os casos de qualidade.
sns.boxplot(
  data=df_frutas,
  y='Weight',
  hue='Quality',
)
plt.title("Weight by Quality Boxplot")
plt.savefig("./dataviz/weight-by-quality-boxplot.png")
plt.close()

# Olhando a partir de Sweetness
# |-- Frutas boas nessa amostra são mais doces
# |-- Ainda assim não é uma variável tão impactante por si só
sns.boxplot(
  data=df_frutas,
  y='Sweetness',
  hue='Quality',
)
plt.title("Sweetness by Quality Boxplot")
plt.savefig("./dataviz/sweetness-by-quality-boxplot.png")
plt.close()

# Olhando a partir do tamanho
# |-- Frutas boas também tendem a ser maiores
# |-- Ainda sim, não é uma variável tão impactante por si só
sns.boxplot(
  data=df_frutas,
  y='Size',
  hue='Quality',
)
plt.title("Size by Quality Boxplot")
plt.savefig("./dataviz/size-by-quality-boxplot.png")
plt.close()

# 2.3 - Teste T-Student
# Teste estatístico para verificar se existe uma diferença
# |-- significativa entre as médias de 2 grupos.
# |-- é importante que ambos os grupos estejam balanceados.
# |-- H0: Não há diferença significativa entre as médias dos grupos.
# |-- H1: Há uma diferença significativa entre as médias dos grupos. (p-val < 0.05)

# Teste T-Student para peso
# |-- p_value ≃ 0.9284 -> não rejeita hipótese nula
grupo_good_weight = df_frutas[(df_frutas['Quality'] == 1)]['Weight']
grupo_bad_weight = df_frutas[(df_frutas['Quality'] == 0)]['Weight']
result = ttest(x=grupo_good_weight, y=grupo_bad_weight, paired=False)
p_value = result['p_val'].iloc[0]
print(f"T-Student (Weight) p-value: {p_value:.6f}")

# Teste T-Student para sweetness
# |-- p_value ≃ 0 -> não rejeita hipótese nula
# |-- Sweetness é uma variável possivelmente boa.
grupo_good_weight = df_frutas[(df_frutas['Quality'] == 1)]['Sweetness']
grupo_bad_weight = df_frutas[(df_frutas['Quality'] == 0)]['Sweetness']
result = ttest(x=grupo_good_weight, y=grupo_bad_weight, paired=False)
p_value = result['p_val'].iloc[0]
print(f"T-Student (Sweetness) p-value: {p_value:.6f}")

# Teste T-Student para size
# |-- p_value ≃ 0 -> rejeita hipótese nula
# |-- Size é uma variável possivelmente boa.
grupo_good_weight = df_frutas[(df_frutas['Quality'] == 1)]['Size']
grupo_bad_weight = df_frutas[(df_frutas['Quality'] == 0)]['Size']
result = ttest(x=grupo_good_weight, y=grupo_bad_weight, paired=False)
p_value = result['p_val'].iloc[0]
print(f"T-Student (Size) p-value: {p_value:.6f}")

# 2.4: Matriz de Correlação
# A correlação das variáveis não é muito forte.
matriz_correlacao = df_frutas.corr('pearson')
sns.heatmap(
  matriz_correlacao,
  cmap="coolwarm",
  annot=True,
  vmin=-1,
  vmax=1
)
plt.title("Matriz de Correlação")
plt.savefig("./dataviz/matriz-correlacao.png")
plt.close()

# 3: Treinamento do Modelo
# 3.1 - Modelo Baseline
# |-- Modelo sem ajustes.
X = df_frutas.drop(columns=['Quality'])
y = df_frutas['Quality']

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  random_state=51,
  shuffle=True,
  train_size=0.7
)

# Para datasets pequenos é indicado usar o solver liblinear,
# Há outro solvers na documentação do scikit-learn.
model_logistic_regression = LogisticRegression(solver='liblinear')
model_logistic_regression.fit(X=X_train, y=y_train)

# Retorna 0 ou 1 para cada dado de teste
y_pred = model_logistic_regression.predict(X=X_test)

# Decision function:
# |-- Retona o valor (resultado) de cada instância, considerando
# |-- os coeficientes obtidos da reta de regressão 
# |-- Retorna a pontuação de cada registro de teste
# |-- é a soma dos coeficientes da reta de regressão logística.
y_decision = model_logistic_regression.decision_function(X=X_test)

# Probabilidades
# |-- Retorna as probabilidades em tuplas de cada registro ser 0 ou 1.
# |-- Suponha que o primeiro par de y_prod é [0.7, 0.3] para 0 e 1 respectivamente
# |-- O modelo classifica como 0 se o limiar for maior que 0.3.
# |-- Se o limiar fosse 0.3 ou menos, marcaria como 1
# |-- Em geral  limiar é 0.5, logo 0.7 é maior que 0.5, por isso classifica como 0.
y_prob = model_logistic_regression.predict_proba(X=X_test)

# 2.2 - Curva ROC e AUC

# Retornar valores da curva ROC:
# |-- TPR(True Positive Rate)
# |-- FPR(False Positive Rate)
# |-- Threshold
fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_decision)

# Área sobre a curva ROC (AUC):
# |-- Precisa de valores da curva ROC
# |-- Área é de ≃ .8415, logo é melhor que chutar com 50% de chance(area > 0.5).
roc_auc = auc(x=fpr, y=tpr)
print(f"Aréa sob a curva ROC: {roc_auc:.6f}")

# Plot curva ROC
sns.lineplot(
  x=fpr,
  y=tpr,
  label="Curva ROC",
  color='blue'
)
plt.fill_between(
  fpr,
  tpr, # type: ignore
  alpha=0.2,
  color='blue',
  label='Area sob curva ROC',
)
sns.lineplot(
  x=fpr,
  y=fpr,
  color="red",
  linestyle='--',
  label="Classificador Aleatório"
)
plt.grid(visible=True)
plt.legend(loc='lower right') # type: ignore
plt.title("Curva ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("./dataviz/curva-roc-baseline.png")
plt.close()

# 2.3 - Importância das Features
# |-- Como obter a importância das features com base nos coeficientes
# |-- obtidos na regressão logística.
# |-- Tamanho, sweetness e juiciness tem mais importância no modelo
# |-- isso significa que têm os maiores coeficientes.
importance = np.abs(model_logistic_regression.coef_)

print("Importância das Features:")
for i, feature in enumerate(model_logistic_regression.feature_names_in_):
    print(f'\t{feature}: {importance[0][i]}')

# 2.4 -F1-Score, BCE e Matriz de Confusão

# F1-Score
# |-- 2 * (precisão * recall)/(precisao + recall)
# |-- Resultados: próximo de 1 = perfeito | próximo de 0 = péssimo
# |-- Resultad deu 0.778, o que é razoável.
f1_score_baseline = f1_score(y_pred=y_pred, y_true=y_test)
print(f"f1-score baseline: {f1_score_baseline:.6f}")
classification_report_str = classification_report(
  y_pred=y_pred,
  y_true=y_test
)
print(f"Classification Report:\n{classification_report_str}")
accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
print(f"Accuracy: {accuracy}")

# Binary Cross Entropy (BCE) - Log Loss
# |-- Quanto mais próximo de 0 melhor, pois indica 0 perda.
log_loss_baseline = log_loss(y_pred=y_pred, y_true=y_test)
print(f"BCE: {log_loss_baseline}")

# Matriz de confusão
# |-- Modelo tem muitos falsos positivos/ falsos negativos.
# |-- Erra cerca de 1/6 da amostra.
confusion_matrix_baseline = confusion_matrix(y_pred=y_pred, y_true=y_test)
disp_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix_baseline)
disp_confusion_matrix.plot()
plt.title("Matriz de Confusão")
plt.savefig("./dataviz/modelo/matriz-de-confusao-baseline.png")
plt.close()

# 3: Otimização de Hiperparâmetros com Optuna

# Hiperparâmetro penalty
# |-- Punição para coeficientes calculados pelo modelo.
# |-- Controla a complexidade do modelo reduzindo o overfitting.
# |-- L1 (Ridge): útil para fins de Feature Selection e para modelos esparsos
#     |-- Atua na soma dos valores absolutos dos coeficientes do modelo.
# |-- L2 (Lasso): útil para evitar overfitting, principalmente quando há
#     |-- multicolinearidade. Atua na soma dos quadrados dos coeficientes do modelo.

# Hiperparâmetro C
# |-- Valores maiores de C, indica uma regularização mais fraca.
# |-- Valores menores de C, indica uma regularização mais forte.

# 3.1 - Criar função de otimização

def logistical_regression_optuna(trial):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2']) # sugere valores na lista
    c_value = trial.suggest_categorical('c', [100, 10, 1, 0.1, 0.01]) # sugere valores na lista

    model_optuna = LogisticRegression(
      solver='liblinear',
      penalty=penalty,
      C=c_value
    )

    model_optuna.fit(X_train, y_train)

    # Otimizar:
    # 1. Maximizar área sob curva HOC.
    # 2. Maximizar f1-score.
    # 3. Minimizar log-loss.

    y_decision_optuna = model_optuna.decision_function(X=X_test)

    fpr_optuna, tpr_optuna, _ = roc_curve(y_true=y_test, y_score=y_decision_optuna)
    roc_auc_optuna = auc(fpr_optuna, tpr_optuna)

    y_pred_optuna = model_optuna.predict(X=X_test)

    f1_score_optuna = f1_score(y_pred=y_pred_optuna, y_true=y_test, average='macro')
    log_loss_optuna = log_loss(y_pred=y_pred_optuna, y_true=y_test)

    return roc_auc_optuna, f1_score_optuna, log_loss_optuna

# 3.2 - Criar estudo de otimização
search_space = { 
  'penalty': ['l1', 'l2'],
  'c': [100, 10, 1, 0.1, 0.01]
}
sampler = optuna.samplers.GridSampler(search_space=search_space)
estudo_logistical_regression = optuna.create_study(
    directions=['maximize', 'maximize', 'minimize']
)
estudo_logistical_regression.optimize(
  logistical_regression_optuna, # type: ignore
  n_trials=20
)

# mostrar melhor resultado
melhor_trial = max(
  estudo_logistical_regression.best_trials,
  key=lambda t: t.values[1]
)

print("Trial com melhor área sobre a curva e f1, e menor log_loss:")
print(f"\tnumber: {melhor_trial.number}")
print(f"\tparams: {melhor_trial.params}")
print(f"\tnumber: {melhor_trial.values}")

# Chart 3D com melhores trials -> nao da pra salvar
# fig = optuna.visualization.plot_pareto_front(estudo_logistical_regression)
# plt.savefig(",./dataviz/modelo/estudo.png")
# plt.close()

# 3.3 - Testar threshold diferentes
#|-- Baseline foi melhor em AUC
#|-- Baseline foi melhor em F1-Score
#|-- BAseline foi melhor em Log Loss
# Conclusão: os hiperparâmetros alterados pelo optuna não melhoraram o modelo.
print(f'AUC - Baseline: {roc_auc}')
print(f'AUC - Optuna: {melhor_trial.values[0]}')
print(f"F1-Score - Baseline: {f1_score_baseline}")
print(f"F1-Score - Optuna: {melhor_trial.values[1]}")
print(f"Log Loss - Baseline: {log_loss_baseline}")
print(f"Log Loss - Optuna: {melhor_trial.values[2]}")

# Alterar threshold
# |-- o modelo assume o threshold em 0.5.
# |-- Porém, pode-se alterar isso para mais alto para garantir que somente
# |-- os mais certos sejam classificados como 1.
# |-- Usar modelo baseline pois está melhor que o com optuna
# Conclusão: aumentar o threshold não foi uma boa estratégia
# |-- pois o f1-score cai. 
lista_thresholds = [0.55, 0.6, 0.65, 0.6, 0.75, 0.8, 0.85, 0.9]
lista_resultados = {
    'cenario': [],
    'resultado': [],
}
lista_resultados['cenario'].append('baseline')
lista_resultados['resultado'].append(f1_score_baseline)
lista_resultados['cenario'].append('optuna')
lista_resultados['resultado'].append(melhor_trial.values[1])

for novo_threshold in lista_thresholds:
    y_pred_threshold = (model_logistic_regression.predict_proba(X=X_test)[:, 1] >= novo_threshold).astype(int)
    f1_score_threshold = f1_score(
      y_true=y_test,
      y_pred=y_pred_threshold,
      average='macro'
    )
    lista_resultados['cenario'].append(str(novo_threshold))
    lista_resultados['resultado'].append(f1_score_threshold)

df_thresholds = pd.DataFrame(lista_resultados)
sns.lineplot(
  data=df_thresholds,
  x='cenario',
  y='resultado',
  color='yellow',
  label="F1-Score"
)
plt.grid(visible=True)
plt.title("F1-Score x Aumento do Threshold")
plt.ylabel("F1-Score")
plt.xlabel("Threshold")
plt.savefig("./dataviz/modelo/aumento-do-threshold-x-f1-score.png")
plt.close()

# 4: Salvar modelo
# Salvar baseline pois ficou melhor.
joblib.dump(model_logistic_regression, "./model_lr.pkl")