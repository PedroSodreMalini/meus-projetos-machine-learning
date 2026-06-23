import pandas as pd

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")

# Passando total charges para float
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')

# Unificando tudo
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

# remover uma coluna:
print(df_churn.drop(columns=["TotalCharges"]).info(), "\n")

# remover colunas com valores not a number
print(df_churn.dropna(axis=1).head(), "\n")

# remover coluna em que todos os valores são ausentes -> não muda nada nesse caso
df_churn.dropna(how="all")

# remover linhas com valores ausentes -> axis padrão é zero
print(df_churn.dropna().__len__())

# Remover linhas com todos os valores ausentes
df_churn.dropna(how="all")

# Preencher todos os valores ausentes por 0 obs: não é inplace
print(df_churn.fillna(0))

# preencher valores padrão conforme a coluna
print(df_churn.fillna(value={"TotalCharges": 0, "gender": "Não Declarado"}))

# preencher valores com média
media_total_charges = df_churn["TotalCharges"].mean()
print(df_churn.fillna(value={"TotalCharges": media_total_charges}))

