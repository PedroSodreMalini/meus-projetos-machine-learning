import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, f1_score

# 1: Carregar dataset de colesterol
df_colesterol = pd.read_csv("./dataset/colesterol.csv")
df_colesterol.drop(columns=['Id'], inplace=True)
df_colesterol = pd.get_dummies(df_colesterol, ['Grupo Sanguíneo', 'Fumante', 'Nível de Atividade'])
df_colesterol.info()

# 2: Treinar modelo de regressão linear múltipla
X = df_colesterol.drop(columns=['Colesterol'])
y = df_colesterol['Colesterol']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)

# 2.1: com Recursive Feature Elimination (RFE).
# |-- Usa um modelo estimador e define uma quantidade de features.
# |-- Faz iterações com todas as features, eliminando as menos importantes
# |-- até atingir a quantidade desejada.
rfe_method = RFE(estimator=LinearRegression(), n_features_to_select=6) # Quero 6 features com base em Linear Regression
rfe_method.fit(X=X_train, y=y_train)

# Retorna quais colunas escolheu
# Encontrou peso e altura
print(X_train.columns[(rfe_method.get_support())])

# Ranking de features:
def mostrar_ranking(metodo_fs, X_train):
    ranking = metodo_fs.ranking_ # traz a posição em index das features

    nomes_features = X_train.columns.to_list()

    df_ranking = pd.DataFrame({ 
        'Feature': nomes_features,
        'Ranking': ranking,
    })

    df_ranking = df_ranking.sort_values(by='Ranking')

    print("\nFeatures ranking")
    print(df_ranking)

mostrar_ranking(rfe_method, X_train)


# Função para avaliar performance do modelo
# |-- RMSE ≃ 8.91
# |-- Fez sentido escolher um modelo com menos features
def performance_regressao(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    rmse = root_mean_squared_error(y_pred=y_pred, y_true=y_test)
    print("\nModel evaluation:")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    return rmse

performance_regressao(rfe_method, X_test, y_test)

# 2.2 Treinar modelo sem RFE
# |-- RMSE é levemente maior(≃ 9.01), e é mais propício ao overfitting
model_reg = LinearRegression()
model_reg.fit(X=X_train, y=y_train)

performance_regressao(model_reg, X_test, y_test)

# 2.3 Treinar modelo com RFECV
# |-- Usa os folds para testar e treinar.
# |-- RMSE ≃ 9.02
# |-- Usar folds mudou as features escolhidas e piorou levemente o modelo
# |-- O hiperparametro é o mínimo de features, mas pode escolher mais
# |-- escolheu 11 features, então foi ruim.
rfe_method_cv = RFECV(
    estimator=LinearRegression(),
    min_features_to_select=6,
    cv=5 # divide os dados em 5 folds
)

rfe_method_cv.fit(X=X_train, y=y_train)

performance_regressao(rfe_method_cv, X_test, y_test)

# Ver quais features foram escolhidas
print("Features chosen (CV): ", end='')
print(X_train.columns[(rfe_method_cv.get_support())])
print("Number of features (CV): ", end='')
print(rfe_method_cv.n_features_)

# 2.4 Treinar modelo com Select From Model
# |-- tem o parâmetro Threshold caso necessário
# |-- Escolhe os mesmos parâmetros que o primeiro,
# |-- valor igual para RMSE
sfm_method = SelectFromModel(
    estimator=model_reg,
    max_features=6,
    # threshold=0.5,
)
sfm_method.fit(X=X_train, y=y_train)

# Ver features escolhidas
print(f"\nFeatures chosen: {X_train.columns[(sfm_method.get_support())]}")

# Treinar modelo com as features selecionadas
X_train_ajustado = sfm_method.transform(X_train)
X_test_ajustado = sfm_method.transform(X_test)
model_reg.fit(X_train_ajustado, y_train)
performance_regressao(model_reg, X_test=X_test_ajustado, y_test=y_test)
