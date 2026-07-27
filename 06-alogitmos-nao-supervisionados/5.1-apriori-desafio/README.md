# Algoritmo Apriori - Desafio: E-commerce de Videogames

## Resumo
1. Aplicação do algoritmo Apriori em um dataset contendo registros que significam a venda de um jogo de videogame. Cada registro contém o id do cliente, o id do jogo e o nome do jogo.
2. Nesse cenário, as regras de associação são para formar relação entre a compra dos jogos.

## Dataset
- `Client ID` = ID do cliente.
- `Game ID` = ID do jogo.
- `Game Name` = Nome do jogo.

## Análise Exploratória

### Jogos que mais venderam
![most-appearances-most-sold](dataviz/jogos-mais-vendidos.png)

Considerando que há 591 clientes, a diferença entre o jogo mais vendido e o menos vendido não é tão grande(47 unidades). 47 equivale a aproximadamente 8% do total de registros.

### Percentual de Jogos que mais venderam em relação ao número total de clientes.
![top-20-std-top-20-mean](dataviz/jogos-mais-vendidos-por-total-de-clientes.png)

Considerando que o cliente só pode comprar um jogo uma vez, e que não há duplicatas no dataset, vê-se que os jogos estão bem próximos, no geral, de estarem em metade do dataset observado.

### Corelação de Pearson - Pivot Table
Realiza-se a pivot table por cliente, em que os registros são os clientes e as colunas são os jogos. Marca-se 'False' quando o cliente não comprou o jogo, e marca-se 'True' quando o cliente comprou o jogo.

![Corr_Pearson](dataviz/pearson-correlation.png)

A correlação obtida mostra que há correlação positiva entre quase todos os jogos no geral.

## Algoritmo Apriori
<!-- 1. Obtém os itemsets frequentes utilizando suporte mínimo de 2%.
2. Esse valor baixo é selecionado em razão de o departamento mais frequente aparecer em aproximadamente 16% das transações apenas. Logo, a combinação deste com outros pode obter um valor no máximo igual a 16%.
3. Com isso, obtém-se somente 12 regras de associação no formato A => B, em que o tamanho máximo encontrado nos itemsets foi de 2.
4. Considerando uma confiança mínima de 40%, obtêm-se as regras de associação desejadas:
   - PERSONAL_CARE => HEALTH_AIDS (Confiança = 42.67%)
   - GENERAL_GROCERIES => BEVERAGES (Confiança = 43.43%)
   - WINE => SPIRITS (Confiança = 40.31%) -->