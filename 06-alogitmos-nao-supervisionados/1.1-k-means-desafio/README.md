# K-Means - Conjunto de dados Iris
## Resumo
Algoritmo K-Means para agrupar as amostras do conjunto de dados Iris, otimizando os hiperparâmetros do modelo com o auxílio da biblioteca Optuna. O dataset Iris possui 150 amostras de flores, divididas em três espécies (Setosa, Versicolor e Virginica), cada uma descrita por quatro atributos numéricos:
1. Comprimento da sépala (sepal length)
2. Largura da sépala (sepal width)
3. Comprimento da pétala (petal length)
4. Largura da pétala (petal width)

Variável Target:
- 0: setosa
- 1: versicolor
- 2: virginica

Para fim de práticas, finge-se que não se sabe quantas espécies tem no dataset.

## Rodar app local
```bash
pipenv sync
pipenv shell
python app.py
```
É necessário realizar um drop do csv em clientes.csv na aplicação.
## Análise Exploratória dos Dados
### Distribuição de inovação
![Distribuição Inovação](./dataviz/distribuicao-inovacao.png)

A distribuição de inovação não é uniforme, mas é próximo disso.

### Teste Welsh ANOVA
Usa-se as variáveis *faturamento_mensal* (dependente) e *inovacao* para realizar o teste ***Welsh ANOVA***. Isso ocorre em razão dos dados não serem exatamente uniformes para inovação. Se fossem, faria o teste ANOVA simples.

Para realizar o teste, devemos assumir os seguintes fatos:
1. As observações devem ser independentes uma das outras.
2. Variável dependente é contínua.
3. Há homogeneidade das variâncias. (Teste de Bartlett)
4. Variável dependente segue uma distribuição normal(Teste de Shapiro-Wilk)

#### Mostrando validade do Welsh ANOVA com Teste de Shapiro Wilk
|Hipótese|Significado|Resultado|
|:-:|:-:|:-:|
|H0|A variável apresenta evidência de distribuição normal|p-value >= 0.05|
|H1|A variável não apresenta evidência de distribuição normal|p-value < 0.05|

Resultado - há evidência de distribuição normal para faturamento mensal.
|p-valor|resultado|
|:-:|:-:|
|≃ 0.235|Não rejeita H0|

#### Mostrando validade do Welsh ANOVA com Teste de Barlett
|Hipótese|Significado|Resultado|
|:-:|:-:|:-:|
|H0|Evidência de que as variâncias são significativamente iguais|p-value >= 0.05|
|H1|Evidência de que as variâncias não são significativamente iguais|p-value < 0.05|

Resultado - As variâncias são significativamente iguais entre os agrupamentos por inovação..
|p-valor|resultado|
|:-:|:-:|
|≃ 0.283|Não rejeita H0|

## Otimização de Hiperparâmetros
Os hiperparâmetros testaram foram:
1. k = [3, 4, 5, 6, 7, 8, 9, 10]
2. distancia = ['euclidean', 'cityblock(Manhattan)']

Com uso de optuna, obteve-se que os melhores hiperparâmetros eram formados por:
1. k = 3 
2. distancia = euclidiana

## Visualização da clusterização por variáveis
### Faturamento Mensal, Inovação e Cluster
![view1](./dataviz/inovacao-faturamento-mensal-cluster.png)

O gráfico mostra de forma clara que a inovação foi capaz de definir o cluster. Mostra também que em cada nível de inovação há diferentes níveis de faturamento mensal, mas isso não afeta o cluster.

Cluster 0 = inovação <=2
Cluster 2 = 2 < inovação <= 5
Cluster 1 = inovação > 5

### Idade da empresa cliente, Faturamento Mensal e Cluster
![view2](./dataviz/idade-faturamento-mensal-cluster.png)

Claramente vê-se que o cluster não foi influenciado por essa relação, visto que em todas idades é possível ver indivíduos de vários clusters e de diferentes faturamentos.

### Número de Funcionários, Faturamento Mensal e Cluster
![view3](./dataviz/numero-de-funcionarios-faturamento-mensal-cluster.png)

Claramente vê-se que o cluster não foi influenciado por essa relação, visto que em todas idades é possível ver indivíduos de vários clusters e de diferentes faturamentos.

## Créditos
Pedro Malini, 03 de Julho de 2026
