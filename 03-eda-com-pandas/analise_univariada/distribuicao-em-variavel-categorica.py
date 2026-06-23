import pandas as pd
import matplotlib.pyplot as plt

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")

# Passando total charges para float
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')

# Unificando tudo
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

# COLUNA CHURN
# identificar todos valores em uma coluna
print("Valores possíveis: ", df_churn["Churn"].unique())

# identificar número de valores possíveis
print("Valores possíveis: ", df_churn["Churn"].unique().size)

# identificar quantos valores diferentes são possíveis
print("Número de linhas por valor possível: ", df_churn["Churn"].value_counts())

# identificar proporção de linhas por valor possível
print("Porcentagem de linhas por vbalor possível: ", df_churn["Churn"].value_counts(normalize=True))

# Plot distribuição Churn (quantidade)
ax = df_churn["Churn"].value_counts().plot.bar()
ax.bar_label(ax.containers[0]) # coloca número em cima da barra
plt.show()
plt.close()

# Plot distribuição porcentagem
ax = df_churn["Churn"].value_counts(normalize=True).plot.bar()
ax.bar_label(ax.containers[0]) # coloca número em cima da barra
plt.show()
plt.close()

# Quais são os tipos de contrato
print(df_churn.Contract.unique())

# Plot -> tipos de contrato x número de registros
ax = df_churn["Contract"].value_counts().plot.bar()
ax.bar_label(ax.containers[0]) # coloca número em cima da barra
plt.show()
plt.close()

# Plot -> tipos de contrato x porcentagem de registros
ax = df_churn["Contract"].value_counts(normalize=True).plot.bar()
ax.bar_label(ax.containers[0]) # coloca número em cima da barra
plt.show()
plt.close()
