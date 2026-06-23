# EDA
import pandas as pd
import statsmodels.api as sm
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
import joblib

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score

# Otimização de Hiperparâmetros
import optuna
from optuna.trial import Trial

# 1: Ler Dados
df_obesidade = pd.read_csv("./dataset/obesidade.csv")

print(df_obesidade.tail(5))
print(df_obesidade.info())

# 2: Análise Exploratória de Dados (EDA)

# 2.1 - Converter tudo que é float para inteiro.
# São essas colunas: idade, consumo-de-vegetais, refeições por dia, consumo de água
# |- nivel de atividade fisica e nivel de uso de tela.
colunas_float = [
  'Idade',
  'Consumo_Vegetais_Com_Frequencia',
  'Refeicoes_Dia',
  'Consumo_Agua',
  'Nivel_Atividade_Fisica',
  'Nivel_Uso_Tela'
]
df_obesidade[colunas_float] = df_obesidade[colunas_float].astype(int)
print("\nConvertendo todas as colunas para int e vendo se alguma coluna tem valor nulo:")
print(df_obesidade.info())

# 2.2 - Análise de variáveis quantitativas
# Número de obesos e não obesos. (Próximos 55% x 45%)
obesidade = df_obesidade['Obeso'].value_counts()
plt.figure(figsize=(8,5))
plt.bar(
  x=obesidade.index,
  height=obesidade.values,  # type: ignore
  label="Quantidade de Registros",
  color='purple',
  width=0.1,
  edgecolor='black',
  linewidth=1,
  alpha=.8
)
plt.xlabel("Obesidade (0=Não, 1=Sim)")
plt.ylabel("Frequência")
plt.title("Distribuição de Obesidade")
plt.legend()
plt.grid(visible=True, linestyle='--', alpha=0.4)
plt.savefig("./dataviz/obesidade-barplot.png")
plt.close()

# Obesos correspondem a aproximadamente 45% dos dados
percentual = df_obesidade['Obeso'].value_counts(normalize=True)
plt.figure(figsize=(8,5))
plt.bar(
  x=percentual.index,
  height=percentual.values,  # type: ignore
  label="Percentual de Registros",
  color='purple',
  width=0.1,
  edgecolor='black',
  linewidth=1,
  alpha=.8
)
plt.xlabel("Obesidade (0=Não, 1=Sim)")
plt.ylabel("Percentual")
plt.title("Distribuição de Obesidade em Percentual")
plt.legend()
plt.grid(visible=True, linestyle='--', alpha=0.4)
plt.savefig("./dataviz/percentual-obesidade-barplot.png")
plt.close()

# Idades muito concentradas na faixa dos 18-28 anos.
idade = df_obesidade['Idade']
plt.hist(x=idade, bins=idade.max()-idade.min(), color="#FFD000")
plt.ylabel("Frequência")
plt.xlabel("Idade")
plt.title("Histograma de Idade")
plt.grid(visible=True, linestyle='--', alpha=0.4)
plt.savefig("./dataviz/idade-histplot.png")
plt.close()

# Boxplot indica outliers a partir de ≃ 37 anos
ax = sns.boxplot(y=idade, color="purple", fill=False)
ax.set_ylabel("Idade")
plt.title("Boxplot de Idade")
plt.grid(visible=True, linestyle='--', alpha=0.4)
plt.savefig("./dataviz/idade-boxplot.png")
plt.close()

# 2.3 - Distribuição de Variáveis Qualitativas
# Distribuição de generos está bem uniforme
genero = df_obesidade["Genero_Masculino"].value_counts(normalize=True)
plt.bar(
  x=genero.index,
  height=genero.values, # type: ignore
  label="Percentual de Gêneros",
  color='purple',
  edgecolor="black",
  linewidth=1,
  width=0.1
)
plt.xlabel("Genero (0=Feminino, 1=Masculino)")
plt.ylabel("Porcentagem")
plt.title("Distribuição de Gênero em Porcentagem")
plt.legend()
plt.grid(visible=True, linestyle='--', alpha=0.4)
plt.savefig("./dataviz/percentual-genero-barplot.png")
plt.close()

# Histórico de obesidade na família
# 81% relatou que tem histórico de obesidade na família
hist_obesidade = df_obesidade["Historico_Familiar_Sobrepeso"].value_counts(normalize=True)
plt.bar(
  x=hist_obesidade.index,
  height=hist_obesidade.values, # type: ignore
  label="Percentual de Histórico de Obesidade na Família",
  color='purple',
  edgecolor="black",
  linewidth=1,
  width=0.1
)
plt.xlabel("Histórico de Obesidade na Família (0=Não, 1=Sim)")
plt.ylabel("Porcentagem")
plt.title("Distribuição de Histórico de Obesidade na Família em Porcentagem")
plt.legend()
plt.grid(visible=True, linestyle='--', alpha=0.4)
plt.savefig("./dataviz/percentual-historico-obesidade-na-familia-barplot.png")
plt.close()

# Nível de atvidade física
# Assimetria a esquerda. A maioria não faz exercício.
# A porcentagem vai caindo entre os níveis 0-3.
nvl_atividade_fisica = df_obesidade["Nivel_Atividade_Fisica"].value_counts(normalize=True)
plt.bar(
  x=nvl_atividade_fisica.index,
  height=nvl_atividade_fisica.values, # type: ignore
  label="Percentual de Nível de Atividade Física",
  color='purple',
  edgecolor="black",
  linewidth=1,
  width=0.1
)
plt.xlabel("Nível")
plt.ylabel("Porcentagem")
plt.title("Distribuição de Nível de Atividade Física em Porcentagem")
plt.legend()
plt.grid(visible=True, linestyle='--', alpha=0.4)
plt.savefig("./dataviz/percentual-nivel-atividade-fisica-barplot.png")
plt.close()

# Nível de uso de Tela
# A maioria diz que não usa muito a tela.
# Contudo é interessante ver a partir de dados de pessoas que usam muito.
nvl_tela = df_obesidade["Nivel_Uso_Tela"].value_counts(normalize=True)
plt.bar(
  x=nvl_tela.index,
  height=nvl_tela.values, # type: ignore
  label="Percentual de Nível de Uso de Tela",
  color='purple',
  edgecolor="black",
  linewidth=1,
  width=0.1
)
plt.xlabel("Nível")
plt.ylabel("Porcentagem")
plt.title("Distribuição de Nível de Uso de Tela em Porcentagem")
plt.legend()
plt.grid(visible=True, linestyle='--', alpha=0.4)
plt.savefig("./dataviz/percentual-nivel-uso-de-tela-barplot.png")
plt.close()

# Formulação de Hipótese
# Faixa etária influencia em obesidade ?
print("\nVariável idade:")
print(df_obesidade["Idade"].describe())

bins = [x * 10 for x in range(1,8)]
bins_ordinal = list(range(0,6))
labels_faixa_etaria = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70']
df_obesidade["Faixa_Etaria_String"] = pd.cut(
  x=df_obesidade['Idade'],
  bins=bins,
  labels=labels_faixa_etaria,
  include_lowest=True
)
df_obesidade["Faixa_Etaria"] = pd.cut(
  x=df_obesidade['Idade'],
  bins=bins,
  labels=bins_ordinal,
  include_lowest=True
)
print("\nTabela com faixa etárias de idade:")
print(df_obesidade[['Faixa_Etaria_String', 'Faixa_Etaria']].tail(5))

# 2.4 - Teste Chi-Squared (Qui-Quadrado)
print("\nTabela de Contingência Idade x Obesidade")
tabela_contingencia_faixa_etaria = sm.stats.Table.from_data(
  df_obesidade[['Obeso', 'Faixa_Etaria_String']]
)
print(tabela_contingencia_faixa_etaria.table_orig)

# Teste Chi-Squared:
# grau de liberdade = 5
# p-value = 0.0 -> abaixo de 0.05, variáveis são dependentes
# |-- p-value < 0.05 = rejeitar a hipótese nula (H0)
# |-- Há evidência de não serem independentes.
# estatistica 170
print("Teste Qui-Quadrado Idade x Obesidade")
print(tabela_contingencia_faixa_etaria.test_nominal_association())

# 2.5 - Análise Exploratória Automatizada
# Fazer isso devido ao grande número de variáveis.
# Usar sweetviz
# usa correlação de pearson. OBS: lembrar que pearson é boa para assciação linear.
# Revela que a idade explica muitas outras variáveis
# Revela que a obesidade não é tão dependente de muitas variáveis.
# Variáveis mais dependentes: Histórico Familiar, Idade, Consumo de Alimentos entre refeições,
# |-- Nivel atividade fisica e consumo de calorias.

# sv_obesidade_report = sv.analyze(df_obesidade, target_feat='Obeso')
# sv_obesidade_report.show_html()

# 3: Treinamento do modelo
# 3.1 - Baseline: uso de todas as variáveis
# Treinar modelo
X = df_obesidade.drop(columns=['Obeso', 'Idade', 'Faixa_Etaria_String'])
y = df_obesidade['Obeso']

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.3,
  random_state=51,
  shuffle=True
)

model_baseline = GaussianNB()
model_baseline.fit(X_train, y_train)

y_pred = model_baseline.predict(X_test)

# Métricas
# Falso negativo = diagnostica como não obeso, mas é obeso.
# Falso postivo = diagnostica como obeso, mas não é.
# Nesse caso o falso negativo parece ser mais danoso -> Recall
# Re-call: 0.7759 (true_positive/(false_negative + true_positive))
# |-- Dos realmente positivos, 77% estão sendo egos, o que é relativamente bom.
classification_report_str = classification_report(
  y_pred=y_pred,
  y_true=y_test,
)

recall_baseline = recall_score(
  y_pred=y_pred,
  y_true=y_test,
  average='macro'
)

print(f"\nRelatório de classificação:\n{classification_report_str}")
print(f"Recall:\n{recall_baseline}")

# Mostrar matriz de confusão
# Resultados
# Obesos = acerta muito
# Não obesos: erra muito -> falso positivo ocorre muito(≃40% dos não obesos).
confusion_matrix_modelo_baseline = confusion_matrix(
  y_pred=y_pred,
  y_true=y_test
)

disp_modelo_baseline = ConfusionMatrixDisplay(
  confusion_matrix=confusion_matrix_modelo_baseline,
  display_labels=['Não obeso', 'Obeso']
)
disp_modelo_baseline.plot()
plt.title("Matriz Confusão - Modelo Baseline")
plt.ylabel("Valor verdadeiro")
plt.xlabel("Valor previsto")
plt.savefig("./dataviz/modelo/matriz-confusao-modelo-baseline.png")
plt.close()

# 3.2 - Seleção de Features
# |-- Fazer modelo somente com melhores features.
# |-- Processo de escolha de features é automático(Select KBest).
# |-- Select KBest seleciona as K melhores features baseada em um teste.
# |-- Aqui vamos usar Chi-Squared (Qui-Quadrado)
# Conclusão obtida:
# |-- Acerta ainda mais obesos.
# |-- Erra mais os não obesos. Erra mais que acerta. (dos não obesos, acerta cerca de 40%)
# |-- Pessoas não obesas estão sendo classificadas demais como obesas.
# |-- Recall é bem alto, cerca de 86%$, ou seja acerta muito os obesos
# |-- Precisão é baixa: Tem muitos falsos positivos, cerca de 59% é a precisão.
# |-- Recall geral é de 69.5 %, o que é não ideal.

# Obter as 5 melhores features.
kbest = SelectKBest(score_func=chi2, k=5)

X_train_kbest = kbest.fit_transform(X=X_train, y=y_train)

# Saber as features selecionadas.
# Escolheu:
# |-- Historico Familiar Sobrepeso, Monitora Calorias Ingeridas,
# |-- Nível ATividade Física, Nível Uso Tela e Faixa_Etaria.
kbest_features = kbest.get_support(indices=True) # devolve os indices das features no dataset
X_train_best_features = X_train.iloc[:, kbest_features] # pegar no dataset os indices
print("\nMelhores 5 features selecionadas por KBest:")
print(X_train_best_features.info())

# treinar modelo com essas 5 variáveis
# Idealmente o fluxo sem sabver as variáveis usadas seria:
# |-- Estou fazendo X_train_kbest de novo só pra ficar bonito, mas não seria necessário
X_train_kbest = kbest.fit_transform(X=X_train, y=y_train)
X_test_kbest = kbest.transform(X_test)

modelo_kbest = GaussianNB()
modelo_kbest.fit(
  X=X_train_kbest,
  y=y_train,
)

y_pred_kbest = modelo_kbest.predict(X=X_test_kbest)

classification_report_str = classification_report(y_pred=y_pred_kbest, y_true=y_test)
recall_kbest = recall_score(y_pred=y_pred_kbest, y_true=y_test, average='macro')
print(f"\nRelatório de Classificação (KBest=5)\n{classification_report_str}")
print(f"Recall (KBest = 5): {recall_kbest}")

confusion_matrix_kbest = confusion_matrix(y_pred=y_pred_kbest, y_true=y_test)
disp_confusion_matrix_kbest = ConfusionMatrixDisplay(confusion_matrix_kbest)
disp_confusion_matrix_kbest.plot()
plt.title("Matriz de Confusão - Modelo KBest")
plt.savefig("./dataviz/modelo/matriz-confusao-modelo-best.png")
plt.close()

# 4: Otimização de hiperparâmetros (Tuning de Hiperparâmetros)
# 4.1 - Criação da Função
# |-- Ajustar hiperparâmetros de SelectKBest
# |-- k = k melhores features conforme chi2
def naivebayes_optuna(trial: Trial):
    """Função de Tunning de Hiperparâmetros"""
    k = trial.suggest_int('k', 1, 18)

    kbest = SelectKBest(score_func=chi2, k = k)

    kbest.fit_transform(X_train, y_train)

    kbest_features = kbest.get_support(indices=True)

    X_train_best_features = X_train.iloc[:, kbest_features]

    model_kbest_optuna = GaussianNB()
    model_kbest_optuna.fit(X_train_best_features, y_train)

    kbest.transform(X_test)
    X_test_best_features = X_test.iloc[:, kbest_features]

    y_pred_kbest = model_kbest_optuna.predict(X_test_best_features)

    recall_optuna = recall_score(y_test, y_pred_kbest, average='macro')

    return k, recall_optuna

# rodar o estudo dos hiperparâmetros
search_space = {
    'k': range(1,19)
}
estudo_naivebayes = optuna.create_study(
  sampler=optuna.samplers.GridSampler(search_space=search_space),
  directions=['minimize', 'maximize'] # minimiza k, maximiza recall
)
estudo_naivebayes.optimize(naivebayes_optuna) # type: ignore

# 4.2 - Análise
# Mostrar melhor resultado
# |-- Resultado foi 8 variáveis e recall de ≃ 78.01%
# |-- Logo altera no kbets para 8 e vai achar o melhor result.
trial_com_melhor_recall = max(estudo_naivebayes.best_trials, key = lambda t: t.values[1])
print("Trial com maior recall e menor k:")
print(f"\tnumber: {trial_com_melhor_recall.number}")
print(f"\tparam (k): {trial_com_melhor_recall.params}")
print(f"\tvalues (k, recall): {trial_com_melhor_recall.values}")

fig = optuna.visualization.plot_pareto_front(estudo_naivebayes)
fig.write_image("./dataviz/optuna-hiperparametros.png", width=2)
plt.close()

# 4.3 - Fazer modelo optimizado com optuna (8 variáveis)
kbest = SelectKBest(score_func=chi2, k=8)
X_train_kbest = kbest.fit_transform(X=X_train, y=y_train)
X_test_kbest = kbest.transform(X_test)

modelo_kbest_optimized = GaussianNB()
modelo_kbest_optimized.fit(
  X=X_train_kbest,
  y=y_train,
)

y_pred_kbest = modelo_kbest_optimized.predict(X=X_test_kbest)

classification_report_str = classification_report(y_pred=y_pred_kbest, y_true=y_test)
recall_kbest = recall_score(y_pred=y_pred_kbest, y_true=y_test, average='macro')
print(f"\nRelatório de Classificação (KBest=8)\n{classification_report_str}")
print(f"Recall (KBest = 8): {recall_kbest}")

confusion_matrix_kbest = confusion_matrix(y_pred=y_pred_kbest, y_true=y_test)
disp_confusion_matrix_kbest = ConfusionMatrixDisplay(confusion_matrix_kbest)
disp_confusion_matrix_kbest.plot()
plt.title("Matriz de Confusão Otimizada(8 variáveis) - Modelo KBest")
plt.savefig("./dataviz/modelo/matriz-confusao-modelo-best-otimizado.png")
plt.close()

# 5: Salvar modelo
joblib.dump(modelo_kbest_optimized, "modelo_obesidade.pkl")