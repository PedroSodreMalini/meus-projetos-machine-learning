import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dict_regressao = { 
    'tempo_casa': [1,3, 6, 9, 10, 14, 18],
    'salario': [1500, 3000, 4500, 6000, 7000, 8500, 10000],
}

df_regressao_simples = pd.DataFrame.from_dict(dict_regressao)

sns.regplot(data=df_regressao_simples, x="tempo_casa", y="salario")
plt.savefig("./estudo/regressao-linear.png")
plt.close()