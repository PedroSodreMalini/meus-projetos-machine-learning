import pandas as pd

df_customers = pd.read_csv("./datasets/churn_customers.csv")

print(df_customers.head())

# Renomear coluna
# Opção 1 - renomear uma coluna out place
df_renomeado = df_customers.rename(columns={"SeniorCitizen": "Above65yrs"})
print(df_renomeado.head(5))
# Opção 2 - renomear uma coluna in place
df_customers.rename(columns={"SeniorCitizen": "Above65yrs"}, inplace=True)
print(df_customers.head(5))
# Opção 3 - renomear tudo
df_customers.columns = ["IDCliente", "Genero", "Mais65Anos", "TemParceiro", "TemDependentes"]
print(df_customers.head(5))