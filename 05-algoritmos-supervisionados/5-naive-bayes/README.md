# Modelo de Classificação de Obesidade usando Naive Bayes
![Python](https://img.shields.io/badge/Python-3.11-orange?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-orange?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-orange?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/seaborn-orange?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Statsmodels](https://img.shields.io/badge/statsmodels-orange?style=for-the-badge&logo=python&logoColor=white)
![Optuna](https://img.shields.io/badge/optuna-orange?style=for-the-badge&logo=python&logoColor=white)
![Sweetviz](https://img.shields.io/badge/sweetviz-orange?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/flask-orange?style=for-the-badge&logo=flask&logoColor=white)
![Pydantic](https://img.shields.io/badge/pydantic-orange?style=for-the-badge&logo=pydantic&logoColor=white)
![PyArrow](https://img.shields.io/badge/pyarrow-orange?style=for-the-badge&logo=apache&logoColor=white)
![Kaleido](https://img.shields.io/badge/kaleido-orange?style=for-the-badge&logo=plotly&logoColor=white)
![IPyWidgets](https://img.shields.io/badge/ipywidgets-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![NBFormat](https://img.shields.io/badge/nbformat-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![Optuna](https://img.shields.io/badge/optuna-orange?style=for-the-badge&logo=optuna&logoColor=white)

## Resumo
Modelo de classificação de pessoas quanto a obesidade. A dificuldade está ao não conter dados do peso e da altura das pessoas. O modelo usa classificação Naive Bayes para tntar se adequar ao problema. é disposto uma API para testar o modelo.

## Bibliotecas utilizadas
1. pandas
2. matplotlib
3. seaborn
4. statsmodels
5. scikit-learn
6. sweetviz
7. optuna
8. flask
9.  pydantic
10. flask-pydantic
11. pyarrow
12. kaleido
13. ipywidgets
14. nbformat
15. matplotlib-stubs

## Como rodar projeto
### Rodar API
```bash
pipenv sync
pipenv shell
python api.py
```
### Rodar construção do modelo
```bash
pipenv sync
pipenv shell
python api.py
```
### Observação
> Requer Python 3.11 instalado.

## Análise Exploratória do Modelo - EDA
Devido à presença de 18 variáveis, apenas algumas variáveis foram usadas para a análise exploratória de dados.
<br />
Com o uso de **Sweetviz**, é possível ver a matriz de correlação de Pearson entre as variáveis e a variável target 'Obeso'. A partir dessa análise, foramm escolhidas as variáveis a seguir.

### Target - Obesidade
![Obesidade](./dataviz/percentual-obesidade-barplot.png)

A obesidade é quase que uma distribuição uniforme. A presença de obesos é quase 50% do gráfico(≃ 45%).

### Idade
![Idade Histograma](./dataviz/idade-histplot.png)

A idade está fortemente concentrada na faixa entre 18-28 anos.

![Idade Boxplot](./dataviz/idade-boxplot.png)

O boxplot revela que idades aproximadamente acima de 36 anos estão sendo consideradas como outliers.

### Genero
![Genero](./dataviz/percentual-genero-barplot.png)

Com a matriz de correlação, pode-se ver que o gênero não influencia muito. Além disso, sua distribuição é praticamente uniforme.

### Nivel de Atividade Fisica
![Nivel atividade fisica](./dataviz/percentual-nivel-atividade-fisica-barplot.png)

A maioria das pessoas praticam pouco ou nenhum exercício, estando entre 0 e 1 nesse sentido. Isso tem uma influência na obesidade.

### Histórico de obesidade na família
![Historico de obesidade na família](./dataviz/percentual-historico-obesidade-na-familia-barplot.png)

Cerca de 81% têm histórico de obesidade na família. 

### Uso de Tela (Dispositivos Eletrônicos)
![Nível de uso de tela](./dataviz/percentual-nivel-uso-de-tela-barplot.png)

A maioria usa pouco a tela, o que parece ser algo bem específico dessa mostra. Possivelmente nesse local de coleta os dispositivos eletrônicos não são muito usados por alguma razão, ou então se passa em algum momento em que esses dispositivos não são acessíveis.

## Treinamento do modelo
### Métrica orientadora principal: Re-call
A estratégia escolhida para avaliar um bom desempenho do modelo foi o ***re-call***. O motivo se explica por esse considerar principalmente os falsos negativos, que são um problema em questão, que é quando são classificados pessoas obesas como pessoas sem obesidade.

### Modelo Baseline
+ Primeiro fez-se o treinamento do modelo com todas as variáveis.
+ A variável Idade foi descartada e transformada em uma variável qualitativa de 0 a 6, separando as pessoas entre intervalos de idade de [10-20), [20-30), ..., [60-70].
+ O trenamento do modelo separa os dados de treinamento e teste em 70% para treinamento e 30% para teste.

### Modelo 5 KBest
+ Similar ao treinamento inicial, mas com o uso de *SelectKBest* com *Teste Qui-Quadrado* são escolhidas as 5 variáveis mais correlacionadas ao problema para treinar o modelo.
+ As variáveis escolhidas automaticamente foram: 
    > Historico Familiar Sobrepeso, Monitora Calorias Ingeridas, Nível Atividade Física, Nível Uso Tela e Faixa Etaria.

### Modelo optimizado com Optuna
+ Com o uso da biblioteca ***Optuna***, foi possível encontrar que o melhor(maior re-call) resultado para o re-call ocorre quando o número de variáveis se torna 8.
+ Assim, seleciona-se estas 8 melhores variáveis para treinar o modelo também com *Select KBest* com apoio do teste *Qui-Quadrado*.

## Conclusões
### Resultado - Baseline
![Matriz COnfusão - Baseline](./dataviz/modelo/matriz-confusao-modelo-baseline.png)

### Resultado - 5 KBest
![Matriz COnfusão - 5 KBest](./dataviz/modelo/matriz-confusao-modelo-best.png)

### Resultado - 8 KBest
![Matriz de Confusão - 8 KBest](./dataviz/modelo/matriz-confusao-modelo-best-otimizado.png)

### Otimizações do Optuna
![Otimizações Optuna](./dataviz/optuna-hiperparametros.png)

Os pontos mais em vermelho marcam as melhores tentativas. O eixo y revela o re-call. O eixo x revela o número de variáveis selecionadas.

Como é preciso, nessa ordem, maximizar o re-call e minimizar as variáveis, o ponto mais desejado ocorre no eixo x = 8, em que o recall é o maior e o número de variáveis é menor (8).

## Resultado
|Métrica|Todas variáveis|5 melhores|8 melhores|
|:-:|:-:|:-:|:-:|
|Re-Call|≃ 0.7759|≃ 0.6950|≃ 0.7801|

Evidentemente o tuning de hiperparâmetros ajudou a encontrar o modelo que o re-call é menor. Portanto, o uso das 8 melhores features é mais ideal para se ajustar ao problema.

## Créditos
Pedro Sodré, 6 de Junho de 2026