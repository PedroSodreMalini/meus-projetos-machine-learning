import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# 1: Carregar dados
df_frutas = pd.read_csv("./dataset/fruits.csv")
df_frutas.drop(columns=['A_id'], inplace=True)
df_frutas["Quality"] = df_frutas["Quality"].map({ 'good': 1, 'bad': 0 })
df_frutas.info()

# 2: Treinar modelo de regressão logística
# |-- Performance pura obtem maior recall, mas isso é sucetível ao overfitting
# |-- Contudo cortar features mostrou-se interessante, já que batem f1-score proximos
X = df_frutas.drop(columns=['Quality'])
y = df_frutas['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)

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

def performance_classificacao(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    f1 = f1_score(y_pred=y_pred, y_true=y_test)
    print("\nModel evaluation:")
    print(f"F1-Score: {f1}")

    return f1

# 2.1: Com RFE
rfe_method = RFE(estimator=LogisticRegression(), n_features_to_select=4)
rfe_method.fit(X=X_train, y=y_train)
mostrar_ranking(rfe_method, X_train)
print("Features chosen (RFE): ", end='')
print(X_train.columns[(rfe_method.get_support())])
performance_classificacao(rfe_method, X_test, y_test)

# 2.2: Modelo puro
model_lg = LogisticRegression()
model_lg.fit(X=X_train, y=y_train)
performance_classificacao(model_lg, X_test, y_test)

# 2.3: RFECV -> com f1_score
rfe_method_cv = RFECV(
    estimator=LogisticRegression(),
    cv=5,
    min_features_to_select=4,
    scoring='f1_weighted',
)
rfe_method_cv.fit(X=X_train, y=y_train)
performance_classificacao(rfe_method_cv, X_test, y_test)
print("Features chosen (RFECV): ", end='')
print(X_train.columns[(rfe_method_cv.get_support())])
print("Number of features (RFECV): ", end='')
print(rfe_method_cv.n_features_)

# 2.4: SelectFromModel
sfm_method = SelectFromModel(
    estimator=model_lg,
    max_features=4,
    # threshold=0.5
)
sfm_method.fit(X_train, y_train)

X_test_transformed = sfm_method.transform(X_test)
X_train_transformed = sfm_method.transform(X_train)
model_lg.fit(X_train_transformed, y_train)
performance_classificacao(model_lg, X_test_transformed, y_test)
