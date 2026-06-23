import pandas as pd

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")

# Unificar dataframes de tamanhos IGUAIS
# Tamanho dos dataframes podem ser difrentes, mas este não é o caso:
print(len(df_services), len(df_contracts), len(df_customers)) # todos tem 7043 registros nesse caso

# Unificar customers com services
df_temp = df_customers.merge(df_services, on=["customerID"])
print(df_temp.head())

# unificar com contracts (imaginando que tivessem nomes de colunas diferentes)
df_churn_temp = df_temp.merge(df_contracts, left_on="customerID", right_on="customerID")

# Unificar tudo de uma vez:
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])
print(df_churn.info())