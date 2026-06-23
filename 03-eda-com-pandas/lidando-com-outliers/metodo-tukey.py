import pandas as pd

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

df_churn.dropna(axis=0, inplace=True)
df_churn_clientes_mensais = df_churn[df_churn.Contract == "Month-to-month"]

# Método de Tukey - IQR (Distribuição NÃO NORMAL)
# IQR - Range Interquartil
# IQR = 3° quartil - 1° quartil
q1_TotalCharges_month = df_churn_clientes_mensais.TotalCharges.quantile(0.25)
q3_TotalCharges_month = df_churn_clientes_mensais.TotalCharges.quantile(0.75)
iqr_TotalCharges_month = q3_TotalCharges_month - q1_TotalCharges_month
print(iqr_TotalCharges_month)

# Limite inferior e superior
#   Números abaixo do limite inferior devem ser retirados
#   Números acima do limite superior devem ser retirados
limite_inferior = q1_TotalCharges_month - (iqr_TotalCharges_month * 1.5)
limite_superior = q3_TotalCharges_month + (iqr_TotalCharges_month * 1.5)

df_com_outliers = df_churn_clientes_mensais[(df_churn_clientes_mensais.TotalCharges < limite_inferior) | (df_churn_clientes_mensais.TotalCharges > limite_superior)]
print(df_com_outliers) # tem 200 outliers