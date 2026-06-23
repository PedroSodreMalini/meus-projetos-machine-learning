import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Dicionário de faturamento
dict_faturamento = {
    'data_ref': [
        '2023-01-01', 
        '2023-02-01', 
        '2023-03-01', 
        '2023-04-01', 
        '2023-05-01',
        '2023-06-01', 
        '2023-07-01', 
        '2023-08-01', 
        '2023-09-01', 
        '2023-10-01',
        '2023-11-01', 
        '2023-12-01',
        ],
    'valor': [
        400000, 
        890000, 
        760000, 
        430000, 
        920000,
        340000, 
        800000, 
        500000, 
        200000, 
        900000,
        570000, 
        995000,
        ]
}

# média de vendas 
df_faturamento = pd.DataFrame.from_dict(dict_faturamento)
print(f"Média do faturamento mensal: {df_faturamento['valor'].mean():.2f}")

# Média por mês
df_faturamento["mes"] = pd.to_datetime(df_faturamento["data_ref"]).dt.month_name()
plot = df_faturamento.plot.bar(x="mes", y="valor", color="green", figsize=(12,6))
plot.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _ : f"{int(x/1000)}k"))
plt.show()
plt.close()

plot = df_faturamento.plot.line(x="mes", y="valor", color="red", figsize=(12,6))
plot.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _ : f"{int(x/1000)}k"))
plt.show()
plt.close()

