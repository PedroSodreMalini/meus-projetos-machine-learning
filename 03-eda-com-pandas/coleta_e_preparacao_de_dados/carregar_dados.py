import pandas as pd

# Criar dataframe coom base em um csv
df_customers = pd.read_csv("./datasets/churn_customers.csv")

# ver os 5 primeiros
print("5 primeiros")
print(df_customers.head(5))

# ver os 5 últimos
print("\n5 ultimos")
print(df_customers.tail(5))

# Resumo do df
print("\nResumo do Dataframe")
print(df_customers.info())