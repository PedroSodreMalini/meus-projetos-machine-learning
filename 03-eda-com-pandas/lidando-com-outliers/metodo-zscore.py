import pandas as pd
import numpy as np
from scipy.stats import zscore

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

df_churn.dropna(axis=0, inplace=True)
df_churn_clientes_mensais = df_churn[df_churn.Contract == "Month-to-month"]

# Método de Z-Score (Distribuição NORMAL)
# Indica quantos desvios padrões um ponto específico de dados está distante da média.
# zscore = (x - media) / desvio padrão
# zscore já é um método do scipy!!!!
# um bom limiar para outliers são os que tem o zscore maior que 3 para os valores em módulo.
# menor que -3 é pq ta abaixo do limite inferior, maior que 3 ta acima do limite superior
z = np.abs(zscore(df_churn_clientes_mensais.TotalCharges))
df_com_outliers = df_churn_clientes_mensais[z > 3]
print(df_com_outliers) # 65 outliers nesse caso.

