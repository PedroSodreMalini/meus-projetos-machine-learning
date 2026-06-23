import pandas as pd
import matplotlib.pyplot as plt

df_netflix = pd.read_csv("./netflix_daily_top_10.csv")

# Tipos de dados
print("Tipos de dados:")
print(df_netflix.info())

# Período da análise
print("\nPeríodo da análise:")
print("Início: ", df_netflix["As of"].min())
print("Fim: ", df_netflix["As of"].max())

# Dimensões
print("\nTamanho da base de dados:")
print("Registros: ", df_netflix.shape[0])
print("Variáveis: ", df_netflix.shape[1])

# Dados nulos:
print("\nDados nulos:")
print(df_netflix.isna().any(axis=0)) # Netflix Exclusive tem NaN
print("\nNetflix Exclusive tem dados nulos!!!")
print("Sem preencher:\n", df_netflix["Netflix Exclusive"].head(3))
df_netflix.fillna({ "Netflix Exclusive": "No"}, inplace=True)
print("Após preencher:\n", df_netflix["Netflix Exclusive"].head(3))

# Outliers
# Há muitos outliers em 'Days in top 10' e em 'Viewership Score'
print("\nOutliers:")
df_netflix.plot.box(figsize=(10,6), subplots=True)
plt.savefig("./dataviz/hist-dataframe.png")
plt.close()
print(
    "Os que ficaram com pelo menos 100 dias no top 10:\n",
    df_netflix[df_netflix["Days In Top 10"] >= 100 ]
)
print(
    "Vezes que o filme aparece no dataset\n", 
    df_netflix.Title.value_counts()
)

print(
    "\nTipos de filme nos dados e suas presenças no dataset:\n",
    df_netflix.Type.value_counts()
)
df_netflix.Type.value_counts().plot.bar()
plt.savefig("./dataviz/type-of-films-bar-plot.png")
plt.close()

df_netflix["Viewership Score"].hist()
plt.savefig("./dataviz/viewership-score-hist.png")
plt.close()