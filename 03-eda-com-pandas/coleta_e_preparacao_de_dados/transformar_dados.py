import pandas as pd

df_contracts = pd.read_csv("./datasets/churn_contracts.csv")

# Transformação para float (total charges string -> float)
# '1243.45' -> 1243.45
# se alguma string for empty, coloca None
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')
print(df_contracts.info())

# remover uma coluna
print("\nSem remover customerID:")
print(df_contracts.info())
df_contracts.drop(["customerID"], axis=1, inplace=True)
print("\nApós remover customerID:")
print(df_contracts.info())