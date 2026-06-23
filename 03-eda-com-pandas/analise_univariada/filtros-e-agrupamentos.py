import pandas as pd
import matplotlib.pyplot as plt

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")

# Passando total charges para float
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')

# Unificando tudo
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

# Quantos clientes possuem 1 mês de contrato
print(len(df_churn[(df_churn.tenure == 1)]), " clientes tem apenas 1 mês de contrato.")

# Percentual de clientes com apenas 1 mês de contrato
print("Eles são: ", 100 * len(df_churn[(df_churn.tenure == 1)]) / len(df_churn), "%")

# Quantos clientes entre 1 e 6 meses de contrato
print(len(df_churn[(df_churn.tenure >= 1) & (df_churn.tenure <= 6)]), "clientes entre 1 e 6 meses de contrato.")

# Apresentar a quantidade de clientes por tempo de contrato e ordenar descendente
# Mesma coisa que value_counts
# agrupa por tenure e mostra o número de vezes que tenure, ordenando por quem apareceu mais
print(df_churn.groupby(["tenure"])["tenure"].count().sort_values(ascending=False))
print(df_churn.tenure.value_counts())
df_churn.tenure.value_counts().sort_values().plot.barh(figsize=(20,20))
plt.savefig("./analise_univariada/barh-tenure-values.png")
plt.close()