import pandas as pd
import matplotlib.pyplot as plt

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

# Detecção de outliers(Valores atípicos)
# Total Charges tem Not a Number
# Remover linhas com valores nulos
df_churn.dropna(axis=0, inplace=True)

# Box Plot Geral
df_churn.TotalCharges.plot.box()
plt.savefig("./lidando-com-outliers/hist-total-charges.png") # Mostra que não há outliers
plt.close()

# Box plot agrupado por Contract
df_churn.plot.box(column="TotalCharges", by="Contract")
plt.savefig("./lidando-com-outliers/total-charges-by-contract-type.png") # Nos contratos mensais há muitos outliers
plt.close()

# Criar dataframe de clientes mensais
df_churn_clientes_mensais = df_churn[df_churn.Contract == "Month-to-month"]

# Histograma - CHecar visualmente se os dados seguem uma distribuição normal.
df_churn_clientes_mensais.TotalCharges.plot.hist()
plt.savefig("./lidando-com-outliers/hist-clientes-mensais.png") # assimetria positiva
plt.close()
