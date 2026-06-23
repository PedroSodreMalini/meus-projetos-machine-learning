import pandas as pd
from scipy.stats import chi2_contingency

df_customers = pd.read_csv("./datasets/churn_customers.csv")
df_contracts = pd.read_csv("./datasets/churn_contracts.csv")
df_services = pd.read_csv("./datasets/churn_services.csv")
df_contracts["TotalCharges"] = pd.to_numeric(df_contracts["TotalCharges"], errors='coerce')
df_churn = df_customers.merge(df_services, on=["customerID"]).merge(df_contracts, on=["customerID"])

# Avaliar a correlação entre duas variáveis categóricas

# Executar o teste Chi-Square(Qui-Quadrado) de Pearson
# Em um teste de hipótese, duas são formulados:
# -  H0 (Hipótese Nula): As duas variáveis são independentes
# -  H1 (Hipótese Complementar): as duas variáveis não são independentes

# O teste serve para confirmar ou recusar a hipótese nula (H0).
# Quando a probabilidade de observarmos H0 é inferior a 0.05 (p-value)
# recusamos a hipótese nula e seguimos com a complementar.

# Gerar um dataframe da crosstab (sem totais)
df_crosstab_churn_contract = pd.crosstab(df_churn.Churn, df_churn.Contract)

# Tupla de valores incluindo p-value e scores(chi-square)
chi_scores_churn_contract = chi2_contingency(df_crosstab_churn_contract)
scores_churn_contract = pd.Series(chi_scores_churn_contract[0])
p_values_churn_contract = pd.Series(chi_scores_churn_contract[1])
pd.set_option("display.float_format", lambda x: '%.15f' % x)

df_chi_scores_churn_contract = pd.DataFrame({
    "Qui2": scores_churn_contract, 
    "p-value": p_values_churn_contract
    })

print(df_chi_scores_churn_contract)

# O P-value é menor que 0.05, desta forma rejeitamos a hipótese nula, ou seja,
# as variáveis não são independentes. Pelo chi-square alto, podemos afirmar que há
# uma forte correlação.