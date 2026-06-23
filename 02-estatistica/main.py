import pandas as pd
import matplotlib.pyplot as plt

# versão pandas
print("\nVersão pandas:", pd.__version__)

# series em pandas
dict_medidas = {
    "idade": [15, 18, 25, 25, 40, 55, 58, 60, 80],
    "altura": [160, 162, 165, 168, 172, 174, 174, 174, 176]
}

df_medidas = pd.DataFrame.from_dict(dict_medidas)

print("\nDataframe idade x altura:\n", df_medidas)

# Medidas de posição
print("Média de idade: ", df_medidas["idade"].mean())
print("Mediana de idade: ", df_medidas["idade"].median())
print("Moda de idade: ", df_medidas["idade"].mode())

# Medidas de dispersão
print("Variância de idade: ", df_medidas["idade"].var())
print("Desvio padrão de idade: ", df_medidas["idade"].std())
print("Coeficiente de idade: ", df_medidas["idade"].std() / df_medidas["idade"].mean())

# Medidas de forma
print("Assimetria da idade: ", df_medidas["idade"].skew())
print("Curtose da idade: ", df_medidas["idade"].kurtosis())

# Resumo
print("Resumo:\n", df_medidas["idade"].describe())

# Correlação
dict_salarios ={
    "tempo": [6,10,12,18,24,30,36,50],
    "salario": [1800,2000,2400,3000,3600,4300,5100,6000],
}

df_salarios = pd.DataFrame.from_dict(dict_salarios)
print("\nDataframe salario x tempo de serviço\n", df_salarios)

print("Coeficiente de Pearson:\n", df_salarios.corr(method="pearson"))
print("Coeficiente de Spearman:\n", df_salarios.corr(method="spearman"))

# PLOTS
df_medidas["idade"].hist() # fazer histograama

dict_vendas = {
    "categoria": ["masculino", "feminino", "infantil", "casa"],
    "valor": [400000, 600000, 250000, 580000],
    "quantidade": [3000, 5000, 1500, 2500],
}

df_vendas = pd.DataFrame.from_dict(dict_vendas)

# barras verticais
df_vendas.plot.bar(x="categoria", y="valor")
df_vendas.sort_values("valor", ascending=False).plot.bar(x="categoria", y="valor")

# barras horizontais
df_vendas.plot.barh(x="categoria", y="quantidade")
df_vendas.sort_values("quantidade", ascending=False).plot.barh(x="categoria", y="quantidade")

# Gráfico de Dispersão
df_medidas.plot.scatter(x="idade", y="altura")

# BoxPlot
df_medidas.altura.plot.box()

df_medidas.idade.plot.box()

plt.close("all")

# Gráfico de Linhas
dict_faturamento = {
    "data_ref": ["2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
    "valor": [400000, 800000, 500000, 800000, 900000]
}
df_faturamento = pd.DataFrame.from_dict(dict_faturamento)
df_faturamento["data_ref"] = pd.to_datetime(df_faturamento["data_ref"])
print("\nDataframe faturamento:\n", df_faturamento)
df_faturamento.plot.line(x="data_ref", y="valor", color="green")
plt.show()
plt.close()

plt.close()