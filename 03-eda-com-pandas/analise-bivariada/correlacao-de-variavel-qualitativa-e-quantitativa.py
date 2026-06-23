import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

# Analisar a correlação entre uma variável qualitativa e quantitativa.
# Usar o mesmo teste de Chi-Square, mas transformar a variável quantitativa em qualitativa.
# Hipótese: Cliente com menos de 6 meses de contrato é mais propenso ao Churn.

# Criar uma variável qualitativa em quantitativa:
#   - Yes: cliente com menos de 6 meses.
#   - No: cliente com menos de 6 meses.
df_churn["LessThan6Months"] = np.where(df_churn.tenure < 6, "Yes", "No")
print(df_churn.LessThan6Months.head())

# Obter Chi-Square e P-value
df_crosstab_churn_time = pd.crosstab(df_churn.Churn, df_churn.LessThan6Months)
chi_scores_churn_time = chi2_contingency(df_crosstab_churn_time)
scores_churn_time = pd.Series(chi_scores_churn_time[0])
pvalues_churn_time = pd.Series(chi_scores_churn_time[1])
pd.set_option("display.float_format", lambda x: '%.15f' % x)
df_chi_scores_churn_time = pd.DataFrame({
    "Qui2": scores_churn_time,
    "P-value": pvalues_churn_time
})
print(df_chi_scores_churn_time)

# Como o p-value é menor que 0.05, rejeitamos H0, ou seja as variáveis não são independentes.
# Como o Chi-Square é menor que o Chi-Square de churn e tipo de contrato, vale concluir que
# há mais relação do Churn com o tipo de contrato, que estar amenos de 6 meses de contrato.