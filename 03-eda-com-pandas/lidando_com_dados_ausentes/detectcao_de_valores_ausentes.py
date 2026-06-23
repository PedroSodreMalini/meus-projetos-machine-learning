import pandas as pd

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")

# Passando total charges para float
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')

# Unificando tudo
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

# Achar not a number (total charge tem 11 valores ausentes por transformação para string vazia)
print(df_churn.isna().sum())

# Achar not a number em uma coluna:
print(df_churn.TotalCharges.isna().sum())

# Achar quantas linhas tem pelo menos 1 coluna com valor ausente:
print(df_churn[df_churn.isna().any(axis=1)])

# Achar quantas colunas tem pelo menos 1 valor ausente
print(df_churn.isna().any(axis=0)).sum()