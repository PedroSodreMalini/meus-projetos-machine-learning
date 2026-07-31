# Local Outlier Factor (LOF)

## Resumo
O *LOF* nesse exemplo é usado para encontrar anomalias, que no caso são o Churn. A métrica utilizada repete a métrica do problema original, que era o re-call, visando ter o menor número de falso negativos, isto é, minimizar a presença de um churn real ser classificado como um não churn.
- Obs: a análise exploratória já havia sido feita anteriormente, logo não é repetida

## Sobre o Algoritmo Local Outlier Factor
O *LOF* assume que pontos outliers estão em zonas isoladas, logo estão em zonas de baixa densidade em relação aos vizinhos. **O algoritmo usa a distância de Minkowski com p=2**, ou seja, a **distância euclidiana** para definir a distância entre os pontos, e assim definir os k-vizinho mais próximos.

O *LOF* recebe a manipulação do parâmetro `contamination`, que é a proporção de outliers no dataset. No caso, há cerca de 26% de Churn(Churn == 1) no dataset, logo 0.26 é o parametro passado. Esse parâmetro é usado de threshold para determinar a pontuação das amostras.

## Execução
1. Realiza a leitura do dataset.
2. Encontra a proporção de outliers para inliers usando a coluna churn. Essa proporção é passada para o parâmetro ***contamination*** do *LOF*.
3. Transforma os dados:
   1. Passa variáveis numéricas contínuas para Standard Scaler.
   2. Passa variáveis categóricas nominais para OneHotEncoder.
   3. Passa variáveis categóricas nominais do tipo 'Yes'/'No' para um transformador binário.
   4. Inclui a variável *'Mais65Anos'* por já estar correta.
4. Define o *LOF* com parâmetro *contamination* e *n_neighbours* já feitos(respecivamente 0.26 e 20).
5. Adequa e obtêm o resultado do modelo com os dados transformados.
6. Com isso, obtem-se que pontos normais assumem o valor de 1, já os pontos anômalos assumem o valor de -1.
7. Também é mostrado a abordagem de *negative_outlier_factor*, em que pontos normais tendem a ficar próximos de -1, e pontos anômalos, próximo a 1.
## Resultado
|Re-call Score|
|:-:|
|≃ 0.7515|

- O re-call score é razoável.

## Data de Conclusão
31 de Julho de 2026