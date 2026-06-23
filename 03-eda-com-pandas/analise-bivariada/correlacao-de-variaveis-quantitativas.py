import pandas as pd
import matplotlib.pyplot as plt

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

# Correlação entre 2 variáveis numéricas
# Tenure com TotalCharges
# A intuição é que quanto mais tempo de contrato maior o valor pago.

# Correlação entre 2 variáveis numéricas - Pearson
#   - Positivo: quanto mais cresce um, mais cresce outro
#   - Negativo: quanto mais cresce um, mais decresce o outro
#   - Nulo: sem relação.
#   OBS: Sensível a outliers.
print("Pearson: ", df_churn.tenure.corr(df_churn.TotalCharges))

# Correlação entre 2 variáveis numéricas - Spearman
#   - Positivo: Quando uma variável tende a aumentar, a outra aumenta
#   - Negativo: Quando uma variável tende a aumentar, a outra diminui
#   - Nulo: sem relação
#   OBS: Menos sensível a outliers.
print("Spearman: ", df_churn.tenure.corr(df_churn.TotalCharges, method="spearman"))

# OBS: Correlação não significa causalidade. Duas variáveis podem estar correlacionadas
# sem que uma cause a outra.

# Apresentar Plot Scatter entre Tenure e TotalCharges
# Gráfico de dispersão.
df_churn.plot.scatter(x="tenure", y="TotalCharges")
plt.savefig("./analise-bivariada/scatter_tenure_totalcharges.png")
plt.close()

# Desafio 1: Vlaidar a primeira hipótese de que a faixa etária do cliente tem uma forte associação
#   com o churn
# Desafio 2: Validar com Teste de Hipótese se Contrato Mensal está mais propenso ao Churn.