import pandas as pd
import matplotlib.pyplot as plt

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")

# Passando total charges para float
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')

# Unificando tudo
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

# Coluna tenure
# resultado mostra um gráfico longe da distribuição normal
# distribuição em sela: muitos clientes com contratos curtos ou longos
df_churn.tenure.plot.hist()
plt.savefig("./analise_univariada/hist_tenure.png")
plt.close()

# Coluna Monthly Charges
# resultado também não em uma distribuição normal
# Muitos clientes pagando valores baixos e poucos intermediários
df_churn.MonthlyCharges.plot.hist()
plt.savefig("./analise_univariada/hist_monthly_charges.png")
plt.close()

# Media tempo de contrato em meses
tenure_mean = df_churn.tenure.mean()
tenure_median = df_churn.tenure.median()
tenure_mode = df_churn.tenure.mode()

# medidas de dispersão - desvio padrão de meses
tenure_std = df_churn.tenure.std()
tenure_variance_coef = (tenure_std / tenure_mean) * 100