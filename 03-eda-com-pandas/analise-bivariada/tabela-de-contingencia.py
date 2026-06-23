import pandas as pd

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

# Hipótese: Clientes com contrato mensal são mais propensos ao abandonamento
print(pd.crosstab(
    df_churn.Churn, 
    df_churn.Contract, 
    normalize="index", 
    margins=True, 
    margins_name="Total")
    )
# Conclusão: a proporção de clientes com churn e sem churn
# é muito diferente. A maioria dos churns vem de contratos mensais.
# Os clientes mensais representam 55%, mas se considerarmos somente os que abandonaram o serviço
# os clientes mensais se tornam 88%.